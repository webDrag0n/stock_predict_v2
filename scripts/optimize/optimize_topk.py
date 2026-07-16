"""阶段一(改目标): 把"top-20% 超额收益"做到最高 —— long-only 真正能吃到的精度。

分层诊断证明全断面 IC 高 ≠ 可交易(最高IC的预测器 edge 在底层)。所以精度的
正确目标是 long_topk 实际收割的部分: **top-20% 组合的未来超额收益**。

主指标(全部严格防泄露 OOS, 逐窗口锁定预测器预测后, 把每日 top-20% 选出来):
  top20_excess : 顶部20%名字的平均未来收益 − 全体均值(越高越好, 这才是 long_topk 吃到的)
  top20_pos    : top20_excess 为正的窗口数
辅助: top20_excess 的 ICIR(跨窗口稳定性)、单调性。

候选预测器(都偏"头部"):
  l2_gbm        : 63因子 GBM(基准, 已知头部有 edge)
  lambdarank    : LGBMRanker NDCG —— 天然加权顶部排序, 与 top-K 选择最对齐
  ret_weighted  : l2 GBM 但样本权重 ∝ 未来收益的截面 rank(放大头部样本)
  topk_clf      : 二分类(是否落在当日 top-20%), 预测正类概率当分数
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import (_cross_sectional_norm, _gbm_l2_predict,
                         _train_predict_rank)
from factor_eval import _windows


def _ret_weighted_gbm(train, test, feats, n_seeds=3) -> np.ndarray:
    """l2 GBM, 但样本权重 ∝ 训练段未来收益的按日截面 rank(0..1), 放大头部。"""
    Xtr, _ = _cross_sectional_norm(train, feats)
    ytr = train[C.TARGET].to_numpy(dtype=np.float64)
    Xte, _ = _cross_sectional_norm(test, feats)
    # 头部加权: 未来收益越高(当日截面 rank 越大)权重越大
    rnk = (train.groupby("date", group_keys=False)[C.TARGET]
                .rank(pct=True)).to_numpy()
    w = 0.2 + rnk            # 0.2..1.2, 头部约6倍底部
    preds = np.zeros(len(test))
    for s in range(n_seeds):
        m = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
            colsample_bytree=0.6, min_child_samples=40, reg_alpha=0.1,
            reg_lambda=0.5, objective="l2", random_state=C.RANDOM_STATE + s,
            n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr, sample_weight=w)
        preds += np.asarray(m.predict(Xte)).ravel()
    return preds / n_seeds


def _topk_classifier(train, test, feats, top_frac=0.2, n_seeds=3) -> np.ndarray:
    """二分类: 标签=当日未来收益是否在 top_frac。预测正类概率当排序分。"""
    Xtr, _ = _cross_sectional_norm(train, feats)
    Xte, _ = _cross_sectional_norm(test, feats)
    rnk = (train.groupby("date", group_keys=False)[C.TARGET].rank(pct=True))
    ytr = (rnk >= (1.0 - top_frac)).astype(int).to_numpy()
    preds = np.zeros(len(test))
    for s in range(n_seeds):
        m = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
            colsample_bytree=0.6, min_child_samples=40, reg_alpha=0.1,
            reg_lambda=0.5, random_state=C.RANDOM_STATE + s, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr)
        preds += np.asarray(m.predict_proba(Xte)[:, 1]).ravel()
    return preds / n_seeds


def _top_excess(pred, test, top_frac=0.2) -> float:
    """该窗口: 按 pred 选当日 top_frac, 其未来收益均值 − 全体均值。"""
    t = test[["date", C.TARGET]].copy()
    t["pred"] = pred
    def _one(g):
        k = max(1, int(round(len(g) * top_frac)))
        top = g.nlargest(k, "pred")[C.TARGET].mean()
        return top - g[C.TARGET].mean()
    ex = t.groupby("date", group_keys=False).apply(_one, include_groups=False)
    return float(ex.dropna().mean())


def main() -> None:
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    top_frac = float(os.environ.get("TOP_FRAC", "0.2"))
    print(f"加载面板... 主指标=top-{int(top_frac*100)}% 超额收益")
    panel = load_factor_panel(keep_price=True)
    feats = list_factor_columns(panel)
    dates = np.sort(panel["date"].unique())
    wins = list(_windows(dates))
    print(f"全因子 {len(feats)} | {len(wins)} 窗口 | FW={C.FORWARD_WINDOW}\n")

    variants = ["l2_gbm", "lambdarank", "ret_weighted", "topk_clf"]
    per_win = {v: [] for v in variants}
    for win_id, tr, te in wins:
        train = panel[panel["date"].isin(tr)]
        test = panel[panel["date"].isin(te)].copy()

        # single-seed for the relative search (3x faster; we compare ranking of
        # variants, not final production numbers — the winner gets re-run at 3 seeds)
        preds = {
            "l2_gbm": _gbm_l2_predict(train, test, feats, n_seeds=1),
            "lambdarank": _train_predict_rank(train, test, feats, n_seeds=1),
            "ret_weighted": _ret_weighted_gbm(train, test, feats, n_seeds=1),
            "topk_clf": _topk_classifier(train, test, feats, top_frac, n_seeds=1),
        }
        print(f"  [窗口{win_id} 训练完成]", flush=True)
        line = [f"窗口{win_id}:"]
        for v in variants:
            ex = _top_excess(preds[v], test, top_frac)
            per_win[v].append(ex)
            line.append(f"{v}={ex:+.4f}")
        print("  " + " ".join(line))

    rows = []
    for v in variants:
        arr = np.array(per_win[v])
        icir = arr.mean() / arr.std() if arr.std() > 1e-12 else np.nan
        rows.append({"predictor": v, "top_excess_mean": float(arr.mean()),
                     "stability_ICIR": float(icir),
                     "pos_windows": int((arr > 0).sum()), "n": len(arr)})
    tbl = pd.DataFrame(rows).sort_values("top_excess_mean", ascending=False)
    print(f"\n===== top-{int(top_frac*100)}% 超额收益对照(连续OOS, 降序)=====")
    with pd.option_context("display.float_format", lambda v: f"{v:+.5f}"):
        print(tbl.to_string(index=False))
    tbl.to_csv(os.path.join(C.OUTPUT_DIR, "optimize_topk.csv"), index=False)
    best = tbl.iloc[0]
    print(f"\n最高 top-{int(top_frac*100)}% 超额: {best['predictor']} "
          f"(超额={best['top_excess_mean']:+.5f}, "
          f"稳定ICIR={best['stability_ICIR']:+.2f}, "
          f"为正窗口={int(best['pos_windows'])}/{int(best['n'])})")
    print("-> outputs/optimize_topk.csv")


if __name__ == "__main__":
    main()
