"""回测绩效指标计算（Sharpe/Sortino/年化alpha+NW t/回撤/胜率）。

从 pipeline/steps_backtest.py 的 performance 步骤抽取，供两处复用：
  - pipeline performance 步骤（webapp 回测报告，含图表序列）；
  - src/model_gate.py 的 champion/staging 门控评估（只要标量指标）。
单一实现保证两处口径逐项一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from measure_alpha import _newey_west_alpha_t


def compute_performance(daily: pd.DataFrame, panel: pd.DataFrame,
                        initial_cash: float, lags: int = 20) -> dict:
    """从每日状态 + 面板算绩效指标。

    Args:
        daily        : run_daily_replay 返回的 daily（date, total_value, ...）。
        panel        : 因子面板（用于等权基准与 alpha；需 return_1d 或 pct_chg）。
        initial_cash : 初始资金（res["meta"]["initial_cash"]）。
        lags         : Newey-West 滞后阶数（≈调仓持有期）。

    Returns:
        dict: 标量指标（sharpe/sortino/max_drawdown/alpha_ann/beta/t_nw/
              hit_rate/total_return/n_days，max_drawdown 为百分数，如 -12.3）
              + 中间序列（dts/strat_ret/strat_cum/bench_cum/dd，供画图）。
    """
    dts_raw = daily["date"].to_numpy()
    dts = [str(pd.Timestamp(x).date()) for x in dts_raw]
    tv = daily["total_value"].to_numpy(dtype=np.float64)
    # 策略日收益率（基于总值）
    strat_ret = np.zeros(len(tv))
    strat_ret[1:] = tv[1:] / tv[:-1] - 1.0
    strat_cum = (tv / initial_cash - 1.0) * 100.0

    # 基准：全市场等权日收益，对齐到回测交易日
    ret_col = ("return_1d" if "return_1d" in panel.columns
               else ("pct_chg" if "pct_chg" in panel.columns else None))
    bench_cum = None
    alpha_ann = beta = t_nw = float("nan")
    if ret_col is not None:
        bser = panel.groupby("date")[ret_col].mean()
        if ret_col == "pct_chg":
            bser = bser / 100.0
        bench = bser.reindex(pd.Index(dts_raw)).fillna(0.0).to_numpy()
        bench_cum = (np.cumprod(1.0 + bench) - 1.0) * 100.0
        # CAPM + Newey-West（滞后≈调仓持有期）
        a_d, beta, t_nw, _ = _newey_west_alpha_t(
            strat_ret[1:], bench[1:], max(1, min(lags, len(strat_ret) - 2)))
        alpha_ann = a_d * 252

    # 回撤（按总值，百分数）
    peak = np.maximum.accumulate(tv)
    dd = (tv - peak) / peak * 100.0
    max_dd = float(dd.min())

    # Sharpe / Sortino（日收益年化）
    rr = strat_ret[1:]
    sharpe = float(rr.mean() / rr.std() * np.sqrt(252)) if rr.std() > 1e-12 else 0.0
    downside = rr[rr < 0]
    sortino = (float(rr.mean() / downside.std() * np.sqrt(252))
               if len(downside) > 1 and downside.std() > 1e-12 else 0.0)
    hit = float((rr > 0).mean()) if len(rr) else float("nan")

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "alpha_ann": float(alpha_ann),
        "beta": float(beta),
        "t_nw": float(t_nw),
        "hit_rate": hit,
        "total_return": float(strat_cum[-1]) if len(strat_cum) else 0.0,
        "n_days": int(len(daily)),
        # 中间序列（画图用；标量调用方可忽略）
        "dts": dts,
        "strat_ret": strat_ret,
        "strat_cum": strat_cum,
        "bench_cum": bench_cum,
        "dd": dd,
    }


def scalar_metrics(perf: dict) -> dict:
    """从 compute_performance 结果提取可 JSON 序列化的标量指标子集。"""
    keys = ("sharpe", "sortino", "max_drawdown", "alpha_ann", "beta", "t_nw",
            "hit_rate", "total_return", "n_days")
    return {k: perf[k] for k in keys}
