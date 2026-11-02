"""回测绩效指标计算（Sharpe/Sortino/年化alpha+NW t/回撤/胜率）。

- compute_performance / scalar_metrics：由原 pipeline backtest 模式的
  performance 步骤抽取，供 src/model_gate.py 的 champion/staging 门控评估
  （只要标量指标）复用，单一实现保证口径逐项一致。
- daily_replay_metrics / format_metrics：每日重放（run_daily_replay，唯一回测
  引擎）daily 帧的标量指标与文本汇总，供 walkforward / multiseed 等研究入口
  复用（format_metrics 从已删除的 src/backtest.py 迁入）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from measure_alpha import newey_west_alpha_t


def compute_performance(
    daily: pd.DataFrame, panel: pd.DataFrame, initial_cash: float, lags: int = 20
) -> dict:
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
    ret_col = (
        "return_1d"
        if "return_1d" in panel.columns
        else ("pct_chg" if "pct_chg" in panel.columns else None)
    )
    bench_cum = None
    alpha_ann = beta = t_nw = float("nan")
    if ret_col is not None:
        bser = panel.groupby("date")[ret_col].mean()
        if ret_col == "pct_chg":
            bser = bser / 100.0
        bench = bser.reindex(pd.Index(dts_raw)).fillna(0.0).to_numpy()
        bench_cum = (np.cumprod(1.0 + bench) - 1.0) * 100.0
        # CAPM + Newey-West（滞后≈调仓持有期）
        a_d, beta, t_nw, _ = newey_west_alpha_t(
            strat_ret[1:], bench[1:], max(1, min(lags, len(strat_ret) - 2))
        )
        alpha_ann = a_d * 252

    # 回撤（按总值，百分数）
    peak = np.maximum.accumulate(tv)
    dd = (tv - peak) / peak * 100.0
    max_dd = float(dd.min())

    # Sharpe / Sortino（日收益年化）
    rr = strat_ret[1:]
    sharpe = float(rr.mean() / rr.std() * np.sqrt(252)) if rr.std() > 1e-12 else 0.0
    downside = rr[rr < 0]
    sortino = (
        float(rr.mean() / downside.std() * np.sqrt(252))
        if len(downside) > 1 and downside.std() > 1e-12
        else 0.0
    )
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
    keys = (
        "sharpe",
        "sortino",
        "max_drawdown",
        "alpha_ann",
        "beta",
        "t_nw",
        "hit_rate",
        "total_return",
        "n_days",
    )
    return {k: perf[k] for k in keys}


# ---------------------------------------------------------------------------
def daily_replay_metrics(daily: pd.DataFrame, initial_cash: float) -> dict:
    """从 run_daily_replay 的 daily 帧计算绩效指标（占本金分数口径）。

    供 walkforward 窗口/拼接 OOS 与 multiseed_eval 的 seed 回测复用，键集合与
    format_metrics 兼容（total_pnl/sharpe/max_drawdown/hit_rate/成本拆解等）。
    与 compute_performance 的差异：这里不依赖面板基准（无 alpha/beta 序列），
    max_drawdown 是相对峰值的分数（如 -0.12 而非 -12.3），成本列直接来自
    每日重放的逐笔撮合（commission/stamp/impact，元）。
    """
    tv = daily["total_value"].to_numpy(dtype=np.float64)
    n = len(tv)
    if n == 0 or initial_cash <= 0:
        return {
            "total_pnl": 0.0,
            "gross_pnl": 0.0,
            "total_cost": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": float("nan"),
            "mean_step_pnl": 0.0,
            "avg_gross_exposure": 0.0,
            "n_steps": 0,
            "n_dates": 0,
            "weighted_r2": float("nan"),
            "total_cost_commission": 0.0,
            "total_cost_impact": 0.0,
        }
    # 日收益：首日记首日相对初始现金的盈亏（含首笔成交成本）
    r = np.empty(n, dtype=np.float64)
    r[0] = tv[0] / initial_cash - 1.0
    if n > 1:
        r[1:] = tv[1:] / tv[:-1] - 1.0

    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    downside = r[r < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sharpe = mean / std * np.sqrt(252) if std > 1e-12 else 0.0
    sortino = mean / dstd * np.sqrt(252) if dstd > 1e-12 else 0.0
    ann_return = mean * 252
    ann_vol = std * np.sqrt(252)

    peak = np.maximum.accumulate(tv)
    max_dd = float(((tv - peak) / peak).min())
    total_pnl = float(tv[-1] / initial_cash - 1.0)
    hit_rate = float((r > 0).mean())

    cost_comm = float((daily["commission"] + daily["stamp"]).sum()) / initial_cash
    cost_impact = float(daily["impact"].sum()) / initial_cash
    total_cost = float(daily["cost"].sum()) / initial_cash
    exposure = float((daily["stock_value"] / daily["total_value"].replace(0, np.nan)).mean())
    # Calmar = 年化收益 / |最大回撤|（年化口径，跨回测区间可比）
    calmar = (ann_return / abs(max_dd)) if max_dd < -1e-12 else 0.0

    return {
        "total_pnl": total_pnl,
        "gross_pnl": total_pnl + total_cost,
        "total_cost": total_cost,
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "mean_step_pnl": mean,
        "avg_gross_exposure": exposure,
        "n_steps": int(n),
        "n_dates": int(n),
        "weighted_r2": float("nan"),
        "total_cost_commission": cost_comm,
        "total_cost_impact": cost_impact,
    }


def format_metrics(m: dict) -> str:
    return (
        "Backtest performance\n"
        "--------------------\n"
        f"  Total net PnL        : {m['total_pnl']:+.4f}\n"
        f"  Gross PnL            : {m['gross_pnl']:+.4f}\n"
        f"  Transaction cost     : {m['total_cost']:.4f}\n"
        f"  Ann. return          : {m['ann_return']:+.2%}\n"
        f"  Ann. volatility      : {m['ann_vol']:.2%}\n"
        f"  Sharpe (annualized)  : {m['sharpe']:+.3f}\n"
        f"  Sortino (annualized) : {m['sortino']:+.3f}\n"
        f"  Calmar (ann.ret/DD)  : {m['calmar']:+.3f}\n"
        f"  Max drawdown         : {m['max_drawdown']:+.4f}\n"
        f"  Hit rate             : {m['hit_rate']:.2%}\n"
        f"  Avg gross exposure   : {m['avg_gross_exposure']:.3f}\n"
        f"  Mean step PnL        : {m['mean_step_pnl']:+.6f}\n"
        f"  Steps / dates        : {m['n_steps']} / {m['n_dates']}\n"
        f"  Weighted R2 (pred)   : {m['weighted_r2']:+.5f}\n"
        + (
            f"  Pred OOS IC          : {m['pred_ic']:+.5f}" f"  (ICIR {m['pred_icir']:+.3f})\n"
            if m.get("pred_ic") == m.get("pred_ic") and "pred_ic" in m
            else ""
        )
        + (
            f"  Mean price RMSE (¥)  : {m['mean_price_rmse']:.4f}\n"
            if m.get("mean_price_rmse") == m.get("mean_price_rmse")  # not NaN
            and "mean_price_rmse" in m
            else ""
        )
    )
