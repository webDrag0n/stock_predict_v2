"""截面自注意力预测器（GPU）—— 把"股票互相比较"做成模型的归纳偏置。

动机
----
GBM / 线性 / lambdarank 都已试过（见 optimize_accuracy.py），OOS IC 卡在 ~0.018。
它们都【独立】给每只股票打分，再隐式比较。但 long_topk 真正消费的是【当日截面内
的相对排序】。注意力天然适合：让同一天的所有股票【互相注意】，模型直接学"在今天
这批股票里，谁相对更强"，而非孤立预测每只的绝对收益。这是尚未尝试的架构杠杆。

结构
----
  per-date 截面（一天的全部股票当作一个【集合】，非序列 → 无位置编码）：
    factor z-score (99维, 按日截面，复用 walkforward._cross_sectional_norm，防泄露)
      → Linear 投影到 d_model
      → N × TransformerEncoder(self-attention over stocks in the day)
      → Linear → 每股一个标量分
  损失 = 每日【负 IC】(预测分 与 目标的 Pearson 相关，标准化后)，直接优化评测指标，
        而非 MSE。等价于"训练去最大化当日排序相关"。

防泄露
------
- 训练只用 train 段；test 段只前向、不回传。
- 截面 z-score 每天只用当日自身横截面（_cross_sectional_norm 已保证）。
- 目标 = C.TARGET（与 GBM 同口径）→ 精度可直接对照。

签名与 walkforward._train_predict 的各 leg 一致：attn_predict(train, test, feat_cols)
返回与 test 行对齐的预测分（排序空间）。CPU 无 GPU 时自动降级（慢，仅供冒烟）。
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import config as C
from walkforward import _cross_sectional_norm


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(n_feat: int, d_model: int, n_heads: int, n_layers: int,
                 dropout: float):
    import torch
    import torch.nn as nn

    class CrossSectionalAttn(nn.Module):
        """一天的股票集合 → 每股一个排序分。批维=1天，序列维=当日股票数。"""
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(
                nn.Linear(n_feat, d_model), nn.GELU(), nn.LayerNorm(d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        def forward(self, x, key_padding_mask=None):
            # x: (B=1, S_stocks, n_feat)
            h = self.inp(x)
            h = self.enc(h, src_key_padding_mask=key_padding_mask)
            return self.head(h).squeeze(-1)        # (1, S)

    return CrossSectionalAttn()


def _ic_loss(pred, target, eps=1e-6):
    """负的当日 IC：1 − corr(pred, target)。pred/target: (S,) 一天的截面。

    用标准化后的 Pearson 相关近似 RankIC（可微）；最大化相关 = 最小化 1−corr。
    """
    import torch
    p = pred - pred.mean()
    t = target - target.mean()
    denom = torch.sqrt((p * p).sum() * (t * t).sum()) + eps
    corr = (p * t).sum() / denom
    return 1.0 - corr


def _by_date(dates, uniq):
    """date -> 该日行在数组中的索引（按日切批用）。"""
    idx = {}
    order = np.argsort(dates, kind="stable")
    ds = dates[order]
    for d in uniq:
        lo, hi = np.searchsorted(ds, d, "left"), np.searchsorted(ds, d, "right")
        idx[d] = order[lo:hi]
    return idx


def fit_attn(train: pd.DataFrame, feat_cols: list[str],
             params: dict | None = None):
    """训练截面注意力，返回 (model, cfg)。供 IncrementalEnsemble 在全量重建时调用，
    之后 predict_attn 复用同一 model（训练贵、预测廉，必须分离）。"""
    import torch
    p = dict(params or {})
    d_model = int(p.get("d_model", os.environ.get("ATTN_DMODEL", "128")))
    n_heads = int(p.get("n_heads", os.environ.get("ATTN_HEADS", "4")))
    n_layers = int(p.get("n_layers", os.environ.get("ATTN_LAYERS", "2")))
    dropout = float(p.get("dropout", os.environ.get("ATTN_DROPOUT", "0.1")))
    epochs = int(p.get("epochs", os.environ.get("ATTN_EPOCHS", "40")))
    lr = float(p.get("lr", os.environ.get("ATTN_LR", "1e-3")))
    wd = float(p.get("weight_decay", os.environ.get("ATTN_WD", "1e-4")))
    patience = int(p.get("patience", os.environ.get("ATTN_PATIENCE", "8")))
    seed = int(p.get("seed", C.RANDOM_STATE))
    dev = _device()
    torch.manual_seed(seed); np.random.seed(seed)

    Xtr_all, _ = _cross_sectional_norm(train, feat_cols)
    ytr_all = train[C.TARGET].to_numpy(np.float64)
    tr_dates = train["date"].to_numpy()
    uniq_tr = np.sort(pd.unique(tr_dates))
    tr_idx = _by_date(tr_dates, uniq_tr)

    n_val = max(int(len(uniq_tr) * 0.15), 1)
    emb = int(getattr(C, "EMBARGO_DAYS", 20))
    if len(uniq_tr) > n_val + emb + 5:
        fit_dates = uniq_tr[: -(n_val + emb)]; val_dates = uniq_tr[-n_val:]
    else:
        fit_dates = uniq_tr; val_dates = uniq_tr[-n_val:]

    model = _build_model(len(feat_cols), d_model, n_heads, n_layers, dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def _day(d):
        ii = tr_idx[d]
        if len(ii) < 10:
            return None
        X = Xtr_all[ii]; y = ytr_all[ii]
        ok = np.isfinite(y)
        if ok.sum() < 10:
            return None
        xt = torch.as_tensor(X[ok], dtype=torch.float32, device=dev).unsqueeze(0)
        yt = torch.as_tensor(y[ok], dtype=torch.float32, device=dev)
        return xt, yt

    best_val, best_state, bad = -1e9, None, 0
    for _ep in range(epochs):
        model.train()
        for d in np.random.permutation(fit_dates):
            t = _day(d)
            if t is None:
                continue
            xt, yt = t
            opt.zero_grad()
            loss = _ic_loss(model(xt).squeeze(0), yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            ics = [1.0 - float(_ic_loss(model(t[0]).squeeze(0), t[1]))
                   for d in val_dates if (t := _day(d)) is not None]
            val_ic = float(np.mean(ics)) if ics else -1e9
        if val_ic > best_val + 1e-5:
            best_val, bad = val_ic, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"dev": dev}


def predict_attn(model_cfg, test: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """用已训练 model 对 test 预测（每天一个截面前向）。返回与 test 行对齐的分。"""
    import torch
    model, cfg = model_cfg
    dev = cfg["dev"]
    Xte_all, _ = _cross_sectional_norm(test, feat_cols)
    te_dates = test["date"].to_numpy()
    out = np.zeros(len(test), dtype=np.float64)
    te_idx = _by_date(te_dates, np.sort(pd.unique(te_dates)))
    model.eval()
    with torch.no_grad():
        for d, ii in te_idx.items():
            xt = torch.as_tensor(Xte_all[ii], dtype=torch.float32,
                                 device=dev).unsqueeze(0)
            out[ii] = model(xt).squeeze(0).cpu().numpy()
    return out


def attn_predict(train: pd.DataFrame, test: pd.DataFrame,
                 feat_cols: list[str], params: dict | None = None) -> np.ndarray:
    """walkforward 便捷封装：训练 + 预测一步到位（与各 leg 签名一致）。"""
    mc = fit_attn(train, feat_cols, params)
    return predict_attn(mc, test, feat_cols)

