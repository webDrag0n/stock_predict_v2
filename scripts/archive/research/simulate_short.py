"""做空模拟: dollar-neutral long-short top/bottom-K(预设仓位法, 研究用)。

回答"能否用预设仓位模拟做空, 把 long-only 吃不到的底层 edge 变成真 alpha"。
对每个预测器跑一次 walkforward(long_short_topk 策略), 周期调仓回测, 再对等权
基准做 CAPM+Newey-West 回归。dollar-neutral → beta 应≈0, 收益≈纯 alpha。

预测器对比(关键): ic_linear 全断面 IC 最高、edge 在底层(做空该捕到);
ret_weighted edge 在头部(为 long-only 优化, 做空未必占优)。两者都测。

注意: A股散户不可单股做空, 这是研究诊断(融券/机构/股指期货才可行);
且未计融券借券成本(实盘需在空头名义上加~8-10%/年)。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import run_walkforward
from measure_alpha import _newey_west_alpha_t


def _alpha_report(res: dict, panel_oos: pd.DataFrame, tag: str) -> dict:
    curve = res["oos_result"]["curve"]
    strat = curve["net"].to_numpy(dtype=np.float64)
    ret_col = "return_1d" if "return_1d" in panel_oos.columns else "pct_chg"
    bench = panel_oos.groupby("date")[ret_col].mean().sort_index().to_numpy()
    if ret_col == "pct_chg":
        bench = bench / 100.0
    n = min(len(strat), len(bench))
    strat, bench = strat[:n], bench[:n]

    nw_lags = int(os.environ.get("NW_LAGS", str(C.REBALANCE_DAYS)))
    alpha_d, beta, t_nw, r2 = _newey_west_alpha_t(strat, bench, nw_lags)
    yrs = n / 252.0
    cum = float(np.prod(1.0 + strat) - 1.0)
    cagr = (1.0 + cum) ** (1.0 / yrs) - 1.0 if cum > -1 else float("nan")
    m = res["oos_result"]["metrics"]
    return {"predictor": tag, "n_days": n, "cum_ret": cum, "cagr": cagr,
            "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
            "beta": beta, "alpha_ann": alpha_d * 252, "alpha_t_nw": t_nw,
            "r2": r2}


def main() -> None:
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    predictors = os.environ.get("PREDICTORS", "ic_linear,ret_weighted").split(",")
    print(f"做空模拟 | 策略=long_short_topk | topk={C.TOPK_RATIO} "
          f"周期调仓={C.REBALANCE_DAYS}日 | TRAIN={C.TRAIN_WINDOW}")
    panel = load_factor_panel(keep_price=True)
    factors = list_factor_columns(panel)

    rows = []
    for pred_name in predictors:
        C.MODEL_OBJECTIVE = pred_name.strip()
        print(f"\n--- 预测器: {C.MODEL_OBJECTIVE} ---")
        res = run_walkforward(panel, factors, strategy="long_short_topk",
                              verbose=False)
        rep = _alpha_report(res, res["oos"], C.MODEL_OBJECTIVE)
        rows.append(rep)
        print(f"  OOS={rep['n_days']}d(~{rep['n_days']/252:.2f}y) "
              f"累计={rep['cum_ret']:+.2%} CAGR={rep['cagr']:+.2%} "
              f"Sharpe={rep['sharpe']:+.2f} MaxDD={rep['maxdd']:+.3f}")
        print(f"  Beta={rep['beta']:+.3f} (dollar-neutral应≈0) "
              f"年化Alpha={rep['alpha_ann']:+.2%} "
              f"NW t={rep['alpha_t_nw']:+.2f} "
              f"{'显著✓' if abs(rep['alpha_t_nw'])>=2 else '不显著'} R²={rep['r2']:.3f}")

    tbl = pd.DataFrame(rows)
    print("\n===== 做空模拟汇总(dollar-neutral long-short)=====")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}",
                           "display.width", 200):
        print(tbl.to_string(index=False))
    tbl.to_csv(os.path.join(C.OUTPUT_DIR, "simulate_short.csv"), index=False)
    print("\n判读: dollar-neutral 下 beta≈0, 故年化Alpha≈策略全部收益。"
          " NW t≥2 且 Alpha>0 → 做空真能把底层 edge 变成可捕获 alpha。"
          " 否则 long-only 的结论不变(无稳健 alpha)。⚠️未计融券借券成本。")
    print("-> outputs/simulate_short.csv")


if __name__ == "__main__":
    main()
