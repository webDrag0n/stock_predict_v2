"""确认测试: 简单线性组合 vs GBM(都严格防泄露 OOS)。

诊断3 显示 GBM 0/7 窗口跑赢"最强单因子"。但"最强单因子"是逐窗口事后挑的
(oracle 上界,不可交易)。本测试做一个【完全防泄露】的公平对照:

  组合权重只用每个窗口的【训练段】IC 决定符号/权重,再到【测试段】算 OOS IC。

三种 a-priori 组合(都不偷看测试段):
  - value_only        : 只用 value_composite(单因子基线)
  - eqw_signed        : 训练段 IC 定符号,等权叠加去重后的稳定因子 z-score
  - ic_weighted       : 训练段 IC 直接当权重(IC 加权),叠加去重因子 z-score
对照: 同窗口的 GBM 模型 OOS IC(walkforward._train_predict)。
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
from walkforward import _train_predict, _cross_sectional_norm
from factor_eval import _windows, _daily_rank_ic

# 去重后的稳定候选(每个冗余簇取一个代表 + 跨类低相关 alpha 维度)。
# 选取依据: 诊断2 的簇代表 + 诊断1 里 ICIR 稳定的方向。符号由训练段 IC 决定,
# 这里只挑"哪些因子参与",不预设方向(避免用 OOS 知识)。
DEDUP_FACTORS = [
    "value_composite", "bp_ind_rank", "ep", "sp", "roe", "gross_margin",
    "revenue_yoy", "profit_yoy", "macd_diff", "price_to_ma_60", "momentum_20",
    "volatility_20", "high_52w_prox", "amihud_20", "ret_skew_20", "ln_mv",
    "turnover_rate", "rsi_14",
]


def _train_ic(train: pd.DataFrame, factor: str, target: str) -> float:
    ic = _daily_rank_ic(train, factor, target).dropna()
    return float(ic.mean()) if len(ic) else 0.0


def _composite_pred(train, test, factors, target, mode: str) -> np.ndarray:
    """用训练段 IC 给 test 造组合分(防泄露: 权重只来自 train)。"""
    z_test, _ = _cross_sectional_norm(test, factors)
    score = np.zeros(len(test))
    for j, f in enumerate(factors):
        tic = _train_ic(train, f, target)
        if mode == "eqw_signed":
            w = np.sign(tic)
        elif mode == "ic_weighted":
            w = tic
        else:
            w = 1.0
        score += w * np.nan_to_num(z_test[:, j], nan=0.0)
    return score


def main() -> None:
    panel = load_factor_panel(keep_price=True)
    all_factors = list_factor_columns(panel)
    factors = [f for f in DEDUP_FACTORS if f in all_factors]
    dates = np.sort(panel["date"].unique())
    print(f"去重因子 {len(factors)}/{len(all_factors)} 个参与线性组合: {factors}\n")

    rows = []
    for win_id, tr, te in _windows(dates):
        train = panel[panel["date"].isin(tr)]
        test = panel[panel["date"].isin(te)].copy()

        # GBM(全因子,与生产一致)
        gbm = _train_predict(train, test, all_factors)
        test["__g"] = gbm
        gbm_ic = _daily_rank_ic(test, "__g", C.TARGET).dropna().mean()

        win = {"window": win_id, "GBM": float(gbm_ic)}
        # value-only
        test["__v"] = test["value_composite"].to_numpy()
        win["value_only"] = float(_daily_rank_ic(test, "__v", C.TARGET).dropna().mean())
        # 两种线性组合
        for mode in ("eqw_signed", "ic_weighted"):
            test["__c"] = _composite_pred(train, test, factors, C.TARGET, mode)
            win[mode] = float(_daily_rank_ic(test, "__c", C.TARGET).dropna().mean())
        rows.append(win)
        print(f"  窗口{win_id}: GBM={win['GBM']:+.4f} | value_only={win['value_only']:+.4f} "
              f"| eqw_signed={win['eqw_signed']:+.4f} | ic_weighted={win['ic_weighted']:+.4f}")

    df = pd.DataFrame(rows)
    print("\n===== 连续 OOS IC 均值 / ICIR / 为正窗口 =====")
    for col in ["GBM", "value_only", "eqw_signed", "ic_weighted"]:
        s = df[col]
        icir = s.mean() / s.std() if s.std() > 1e-12 else float("nan")
        print(f"  {col:<12}: IC={s.mean():+.4f} | ICIR={icir:+.3f} | "
              f"为正窗口={int((s>0).sum())}/{len(s)}")
    df.to_csv(os.path.join(C.OUTPUT_DIR, "diag_linear_vs_gbm.csv"), index=False)
    print("\n-> outputs/diag_linear_vs_gbm.csv")


if __name__ == "__main__":
    main()
