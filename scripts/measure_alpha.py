"""阶段二-c: 在修正后的周期调仓回测上, 显式测 alpha/beta vs 等权基准。

核心问题: 策略是真有 alpha, 还是只是吃 beta(牛市裸多)? 用锁定配置跑一次
walkforward, 取连续 OOS 的策略日净收益, 与"等权持有全universe"的日收益做
CAPM 回归: strat_ret = alpha + beta * bench_ret。alpha>0 且显著 = 真主动收益。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import run_walkforward
from measure_alpha import _newey_west_alpha_t


def main() -> None:
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    print(f"锁定配置: 预测器={C.MODEL_OBJECTIVE} topk={C.TOPK_RATIO} "
          f"hold={C.HOLD_RATIO} 周期调仓={C.REBALANCE_DAYS}日 周期回测={C.PERIODIC_REBALANCE}")
    panel = load_factor_panel(keep_price=True)
    factors = list_factor_columns(panel)
    res = run_walkforward(panel, factors, strategy="long_topk", verbose=False)

    curve = res["oos_result"]["curve"]            # 日级净收益(周期调仓)
    oos = res["oos"]
    strat = curve["net"].to_numpy(dtype=np.float64)

    # 等权基准: 每个日期全 universe 的平均日收益(return_1d, 已是 decimal)
    ret_col = "return_1d" if "return_1d" in oos.columns else "pct_chg"
    bench = (oos.groupby("date")[ret_col].mean().sort_index().to_numpy())
    if ret_col == "pct_chg":
        bench = bench / 100.0
    n = min(len(strat), len(bench))
    strat, bench = strat[:n], bench[:n]

    # CAPM 回归 strat = alpha + beta*bench (OLS 点估计 + Newey-West HAC t)
    beta, alpha_d, r, p, se = stats.linregress(bench, strat)
    nw_lags = int(os.environ.get("NW_LAGS", str(C.REBALANCE_DAYS)))
    alpha_nw, beta_nw, t_alpha_nw, r2_nw = _newey_west_alpha_t(strat, bench, nw_lags)
    ann = 252
    alpha_ann = alpha_d * ann
    # 年化收益对比(简单累乘)
    strat_cum = float(np.prod(1.0 + strat) - 1.0)
    bench_cum = float(np.prod(1.0 + bench) - 1.0)
    yrs = n / 252.0
    strat_cagr = (1.0 + strat_cum) ** (1.0 / yrs) - 1.0
    bench_cagr = (1.0 + bench_cum) ** (1.0 / yrs) - 1.0
    # 朴素 OLS t(有偏, 仅作对比)
    resid = strat - (alpha_d + beta * bench)
    t_alpha_ols = (alpha_d / (resid.std(ddof=2) / np.sqrt(n))
                   if resid.std() > 1e-12 else 0.0)

    print("\n===== Alpha / Beta vs 等权基准(周期调仓, 连续OOS)=====")
    print(f"  OOS 交易日数        : {n}  (~{yrs:.2f} 年)")
    print(f"  策略累计收益        : {strat_cum:+.2%}  (CAGR {strat_cagr:+.2%})")
    print(f"  基准累计收益(等权)  : {bench_cum:+.2%}  (CAGR {bench_cagr:+.2%})")
    print(f"  超额(策略-基准)     : {strat_cum-bench_cum:+.2%}")
    print(f"  Beta (对基准)       : {beta:+.3f}")
    print(f"  日 Alpha            : {alpha_d:+.6f}  (年化 {alpha_ann:+.2%})")
    print(f"  Alpha t (OLS朴素)   : {t_alpha_ols:+.2f}  (有偏: 持仓自相关致高估)")
    print(f"  Alpha t (Newey-West): {t_alpha_nw:+.2f}  (lags={nw_lags}, HAC稳健) "
          f"{'显著' if abs(t_alpha_nw)>=2 else '不显著'} @95%")
    print(f"  回归 R²             : {r**2:.3f}")
    print("\n判读: beta<1 且年化 alpha>0 且 NW t≥2 → 真主动收益。"
          " NW t 是有效判据(朴素 OLS t 因持仓自相关高估)。")

    pd.DataFrame({"strat_ret": strat, "bench_ret": bench}).to_csv(
        os.path.join(C.OUTPUT_DIR, "alpha_beta.csv"), index=False)
    print("-> outputs/alpha_beta.csv")


if __name__ == "__main__":
    main()
