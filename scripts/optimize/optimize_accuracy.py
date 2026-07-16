"""阶段一: 把"预测精度(OOS IC/ICIR)"做到最高 —— 一次性搜索最优预测器。

只评估预测精度,不碰交易层(Sharpe/PnL 在 FW=20 下虚高,不可信)。所有 IC 都是
严格防泄露的样本外: 权重/符号只来自每个窗口的训练段,z-score 按日截面。

候选预测器:
  l2_gbm          : 63因子 GBM(生产参照)
  ic_linear       : 训练段 IC 加权的去重因子线性组合(当前最优)
  icir_weighted   : 用训练段 ICIR(IC/std) 当权重 —— 奖励"稳定"的因子而非"幅度大"
  shrink_t        : IC 加权 + 显著性收缩(|t|<2 的因子权重砍半),压噪声因子
  size_neutral    : 因子先对 ln_mv 做按日截面回归取残差,再 IC 加权(去规模暴露)
  fusion_aXX      : l2_gbm 与最优线性按 rank 平均,网格搜 GBM 权重 0/0.25/0.5/0.75/1

输出: 每个预测器的 连续OOS IC / 各窗口IC均值 / ICIR / 为正窗口数,按 IC 降序。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import _cross_sectional_norm, _gbm_l2_predict, _per_date_rank
from factor_eval import _windows, _daily_rank_ic


def _train_ic_stats(train: pd.DataFrame, factor: str) -> tuple[float, float, float]:
    """训练段每日 RankIC 的 (mean, std, t)。防泄露: 只用训练段。"""
    ic = _daily_rank_ic(train, factor, C.TARGET).dropna()
    if len(ic) < 2:
        return 0.0, 0.0, 0.0
    m, s = float(ic.mean()), float(ic.std())
    t = m / (s / np.sqrt(len(ic))) if s > 1e-12 else 0.0
    return m, s, t


def _size_neutralize(df: pd.DataFrame, factors: list[str],
                     size_col: str = "ln_mv") -> pd.DataFrame:
    """因子对 ln_mv 做按日截面回归,用残差替换原值(去规模暴露)。leak-free:
    每个日期只用自己当天的截面拟合。size_col 缺失则原样返回。"""
    if size_col not in df.columns:
        return df
    out = df.copy()
    sz = out[size_col].to_numpy(dtype=np.float64)
    for f in factors:
        if f == size_col or f not in out.columns:
            continue
        v = out[f].to_numpy(dtype=np.float64)

        def _resid(g):
            x = g[size_col].to_numpy(dtype=np.float64)
            y = g[f].to_numpy(dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            r = np.full(len(g), np.nan)
            if mask.sum() >= 10 and np.nanstd(x[mask]) > 1e-9:
                b = np.polyfit(x[mask], y[mask], 1)
                r[mask] = y[mask] - (b[0] * x[mask] + b[1])
            return pd.Series(r, index=g.index)

        out[f] = out.groupby("date", group_keys=False).apply(
            _resid, include_groups=False)
    _ = sz, v
    return out


def _linear_pred(train, test, factors, mode: str) -> np.ndarray:
    """训练段统计 → 测试段线性组合分。mode 决定权重方案(全部 leak-free)。"""
    feats = [f for f in factors if f in train.columns and f in test.columns]
    Zte, _ = _cross_sectional_norm(test, feats)
    score = np.zeros(len(test), dtype=np.float64)
    for j, f in enumerate(feats):
        m, s, t = _train_ic_stats(train, f)
        if mode == "ic":
            w = m
        elif mode == "icir":
            w = m / s if s > 1e-12 else 0.0
        elif mode == "shrink_t":
            w = m * (1.0 if abs(t) >= 2.0 else 0.5)
        else:
            w = m
        score += w * np.nan_to_num(Zte[:, j], nan=0.0)
    return score


def _oos_ic(pred: np.ndarray, test: pd.DataFrame) -> float:
    t2 = test.copy()
    t2["__p"] = pred
    ic = _daily_rank_ic(t2, "__p", C.TARGET).dropna()
    return float(ic.mean()) if len(ic) else np.nan


def _summary(name: str, per_win: list[float]) -> dict:
    arr = np.array([v for v in per_win if v == v], dtype=float)
    if len(arr) == 0:
        return {"predictor": name, "IC_mean": np.nan, "ICIR": np.nan, "pos": 0,
                "n": 0}
    icir = arr.mean() / arr.std() if len(arr) > 1 and arr.std() > 1e-12 else np.nan
    return {"predictor": name, "IC_mean": float(arr.mean()), "ICIR": float(icir),
            "pos": int((arr > 0).sum()), "n": len(arr)}


def main() -> None:
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    print("加载命名因子面板...")
    panel = load_factor_panel(keep_price=True)
    all_factors = list_factor_columns(panel)
    lin_factors = [f for f in getattr(C, "LINEAR_FACTORS", []) if f in all_factors]
    dates = np.sort(panel["date"].unique())
    wins = list(_windows(dates))
    print(f"全因子 {len(all_factors)} | 线性去重因子 {len(lin_factors)} | "
          f"{len(wins)} 个 OOS 窗口 | FW={C.FORWARD_WINDOW}\n")

    # 预先按窗口算一次 size-neutral 面板(只对参与线性的因子,省时间)
    variants = ["l2_gbm", "ic_linear", "icir_weighted", "shrink_t", "size_neutral"]
    per_win = {v: [] for v in variants}
    gbm_rank_by_win = []   # 缓存每窗口的 GBM rank,供 fusion 复用
    lin_rank_by_win = []   # 缓存最优线性(先用 ic_linear)的 rank

    for win_id, tr, te in wins:
        train = panel[panel["date"].isin(tr)]
        test = panel[panel["date"].isin(te)].copy()

        gbm = _gbm_l2_predict(train, test, all_factors)
        per_win["l2_gbm"].append(_oos_ic(gbm, test))

        p_ic = _linear_pred(train, test, lin_factors, "ic")
        per_win["ic_linear"].append(_oos_ic(p_ic, test))
        per_win["icir_weighted"].append(
            _oos_ic(_linear_pred(train, test, lin_factors, "icir"), test))
        per_win["shrink_t"].append(
            _oos_ic(_linear_pred(train, test, lin_factors, "shrink_t"), test))

        # size-neutral: 对 train+test 一起做残差化更稳(仍按日截面,无跨日泄露)
        sn = _size_neutralize(
            pd.concat([train, test], ignore_index=True), lin_factors)
        sn_tr = sn[sn["date"].isin(tr)]
        sn_te = sn[sn["date"].isin(te)].copy()
        per_win["size_neutral"].append(
            _oos_ic(_linear_pred(sn_tr, sn_te, lin_factors, "ic"), sn_te))

        dvec = test["date"].to_numpy()
        gbm_rank_by_win.append((_per_date_rank(gbm, dvec), dvec, test))
        lin_rank_by_win.append(_per_date_rank(p_ic, dvec))
        print(f"  窗口{win_id}: l2={per_win['l2_gbm'][-1]:+.4f} "
              f"ic={per_win['ic_linear'][-1]:+.4f} "
              f"icir={per_win['icir_weighted'][-1]:+.4f} "
              f"shrink={per_win['shrink_t'][-1]:+.4f} "
              f"sizeN={per_win['size_neutral'][-1]:+.4f}")

    # fusion 权重网格(rank 平均): a*GBM_rank + (1-a)*linear_rank
    fusion_rows = {}
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        pw = []
        for (grank, dvec, test), lrank in zip(gbm_rank_by_win, lin_rank_by_win):
            fused = a * grank + (1 - a) * lrank
            pw.append(_oos_ic(fused, test))
        fusion_rows[f"fusion_a{a:.2f}"] = pw

    rows = [_summary(v, per_win[v]) for v in variants]
    rows += [_summary(k, pw) for k, pw in fusion_rows.items()]
    tbl = pd.DataFrame(rows).sort_values("IC_mean", ascending=False)
    print("\n===== 预测器精度对照(连续OOS,按 IC 降序)=====")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(tbl.to_string(index=False))
    tbl.to_csv(os.path.join(C.OUTPUT_DIR, "optimize_accuracy.csv"), index=False)
    best = tbl.iloc[0]
    print(f"\n最高精度预测器: {best['predictor']} "
          f"(IC={best['IC_mean']:+.4f}, ICIR={best['ICIR']:+.3f}, "
          f"为正窗口={int(best['pos'])}/{int(best['n'])})")
    print("-> outputs/optimize_accuracy.csv")


if __name__ == "__main__":
    main()
