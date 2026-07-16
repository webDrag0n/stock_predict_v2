"""时序+截面注意力预测器（GPU）—— 在截面互注意力之上加"每股历史序列"编码。

动机
----
纯截面 attn（model_attn）只看【当日】因子截面，丢掉了每股自身的【时间演化】
（动量加速/波动率爬升/趋势拐点等只有看序列才识别的形态）。本变体：

  每股过去 L 日因子序列 → GRU 时序编码 → 每股一个"历史向量"
    → 当日把全部股票的历史向量做截面互注意力（复用 model_attn 的结构）
    → 每股一个排序分

防泄露
------
- 序列只向【过去】看（trailing L 日，含当日），绝不含未来。
- 截面 z-score 每天只用当日自身横截面（_cross_sectional_norm，逐日独立）。
- 序列按 (symbol, date) 排序后用 sliding-window 构造，左侧不足 L 的左 padding（mask）。
- test 序列可回看 train 末尾的历史（在 concat 面板上构造 gather），但 test 标签
  绝不进训练，且序列只回看 → 无泄露。

接口：attn_seq_predict(train, test, feat_cols, params) 与各 leg 签名一致。
seq_len=1 时退化为纯截面（等同 model_attn）。
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import config as C
from walkforward import _cross_sectional_norm
from model_attn import _ic_loss, _by_date, _device


def _build_seq_gather(panel: pd.DataFrame, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """构造 (N, L) gather 矩阵：每行 = 该 (symbol,date) 行的【过去 L 日同股】行号。

    左侧不足 L 的用 -1 占位（前向时 gather 成 0 = 截面均值，中性）。
    返回 (seq_idx[N,L] int64, valid[N,L] bool)。仅向过去看，防泄露。
    """
    n = len(panel)
    syms = panel["symbol"].astype("category").cat.codes.to_numpy()
    dates = panel["date"].to_numpy()
    # 主序 symbol、次序 date → 同股连续且按时间升序
    order = np.lexsort((dates, syms))
    sym_s = syms[order]
    seq_idx = np.full((n, seq_len), -1, dtype=np.int64)
    # 找每个 symbol 组的边界
    bounds = np.flatnonzero(np.diff(sym_s)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])
    from numpy.lib.stride_tricks import sliding_window_view
    for s, e in zip(starts, ends):
        r = order[s:e]                       # 该股的原始行号，时间升序
        padded = np.concatenate([np.full(seq_len - 1, -1, dtype=np.int64), r])
        win = sliding_window_view(padded, seq_len)   # (len(r), L)
        seq_idx[r] = win
    valid = seq_idx >= 0
    return seq_idx, valid


def _build_seq_model(n_feat, d_model, n_heads, n_layers, dropout, seq_len):
    import torch
    import torch.nn as nn

    class SeqCrossAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(
                nn.Linear(n_feat, d_model), nn.GELU(), nn.LayerNorm(d_model))
            # 时序编码：GRU over L 天（每股独立）。比时序 attn 轻、对短序列稳。
            self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
            # 截面互注意力（同 model_attn）
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        def forward(self, seq):
            # seq: (S_stocks, L, n_feat) —— 一天 S 只股，每只 L 日序列
            h = self.inp(seq)                      # (S, L, d_model)
            _, hn = self.gru(h)                    # hn: (1, S, d_model) 末态
            emb = hn.squeeze(0).unsqueeze(0)       # (1, S, d_model) 截面集合
            z = self.enc(emb)                      # 截面互注意力
            return self.head(z).squeeze(-1).squeeze(0)   # (S,)

    return SeqCrossAttn()


def attn_seq_predict(train: pd.DataFrame, test: pd.DataFrame,
                     feat_cols: list[str], params: dict | None = None) -> np.ndarray:
    """时序+截面 attn：train 上训练、test 上 OOS 预测。返回与 test 行对齐的分。"""
    import torch
    p = dict(params or {})
    seq_len = int(p.get("seq_len", os.environ.get("ATTN_SEQ_LEN", "20")))
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

    n_tr = len(train)
    full = pd.concat([train, test], ignore_index=True)
    # 截面 z-score（逐日独立，防泄露）在 concat 面板上算
    Xall, _ = _cross_sectional_norm(full, feat_cols)
    Xall = np.nan_to_num(Xall, nan=0.0).astype(np.float32)
    # 序列 gather（仅向过去；test 行可回看 train 历史）
    seq_idx, _ = _build_seq_gather(full, seq_len)
    # gather 用：在 Xall 末尾追加一行 0 作为 padding 目标（-1 → 最后一行）
    Xpad = np.vstack([Xall, np.zeros((1, len(feat_cols)), dtype=np.float32)])
    seq_idx_pad = np.where(seq_idx < 0, len(Xall), seq_idx)

    yall = full[C.TARGET].to_numpy(np.float64)
    dates_all = full["date"].to_numpy()
    is_tr = np.zeros(len(full), dtype=bool); is_tr[:n_tr] = True

    uniq_tr = np.sort(pd.unique(dates_all[is_tr]))
    idx_all = _by_date(dates_all, np.sort(pd.unique(dates_all)))

    n_val = max(int(len(uniq_tr) * 0.15), 1)
    emb_d = int(getattr(C, "EMBARGO_DAYS", 20))
    if len(uniq_tr) > n_val + emb_d + 5:
        fit_dates = uniq_tr[: -(n_val + emb_d)]; val_dates = uniq_tr[-n_val:]
    else:
        fit_dates = uniq_tr; val_dates = uniq_tr[-n_val:]

    Xpad_t = torch.as_tensor(Xpad, device=dev)

    def _day_seq(d, train_only):
        ii = idx_all[d]
        if train_only:
            ii = ii[is_tr[ii]]
        if len(ii) < 10:
            return None, None
        y = yall[ii]
        if train_only:
            ok = np.isfinite(y)
            if ok.sum() < 10:
                return None, None
            ii = ii[ok]; y = y[ok]
        si = seq_idx_pad[ii]                     # (S, L)
        seq = Xpad_t[torch.as_tensor(si, device=dev)]   # (S, L, n_feat)
        yt = (torch.as_tensor(y, dtype=torch.float32, device=dev)
              if train_only else None)
        return seq, yt

    model = _build_seq_model(len(feat_cols), d_model, n_heads, n_layers,
                             dropout, seq_len).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val, best_state, bad = -1e9, None, 0
    for _ep in range(epochs):
        model.train()
        for d in np.random.permutation(fit_dates):
            seq, yt = _day_seq(d, True)
            if seq is None:
                continue
            opt.zero_grad()
            loss = _ic_loss(model(seq), yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            ics = []
            for d in val_dates:
                seq, yt = _day_seq(d, True)
                if seq is None:
                    continue
                ics.append(1.0 - float(_ic_loss(model(seq), yt)))
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

    # 预测 test（每天一个截面前向）
    model.eval()
    out = np.zeros(len(test), dtype=np.float64)
    te_dates_u = np.sort(pd.unique(dates_all[~is_tr]))
    with torch.no_grad():
        for d in te_dates_u:
            ii = idx_all[d]
            ii = ii[~is_tr[ii]]
            if len(ii) == 0:
                continue
            si = seq_idx_pad[ii]
            seq = Xpad_t[torch.as_tensor(si, device=dev)]
            pr = model(seq).cpu().numpy()
            out[ii - n_tr] = pr            # test 行在 full 中偏移 n_tr
    return out
