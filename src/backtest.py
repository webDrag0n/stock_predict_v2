"""Walk-forward backtest engine and performance metrics.

PnL model
---------
`responder_6` is the realized forward return of holding the instrument over the
next step. A position `pos` taken at a row therefore earns `pos * responder_6`.
Transaction cost is charged on the change in position vs. the previous step for
the same symbol (turnover * COST_PER_TURNOVER).

We aggregate PnL per (date_id, time_id) step to form an equity curve, then
compute return/risk metrics. There is no lookahead: positions at step t use only
predictions available at t, and the model that produced them was trained purely
on earlier dates (the held-out test set).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _ashare_cost(df: pd.DataFrame, pos: np.ndarray,
                 commission: float, stamp_duty: float) -> np.ndarray:
    """A-share transaction cost on per-symbol position changes.

    Buys (position increases) pay commission only; sells (position decreases)
    pay commission + stamp duty. Cost is on the |change| in position notional.
    """
    tmp = df[[C.DATE_COL, C.TIME_COL, C.SYMBOL_COL]].copy()
    tmp["pos"] = pos
    tmp = tmp.sort_values([C.SYMBOL_COL, C.DATE_COL, C.TIME_COL])
    prev = tmp.groupby(C.SYMBOL_COL)["pos"].shift(1).fillna(0.0)
    delta = tmp["pos"] - prev
    buy = delta.clip(lower=0.0)        # increase in exposure
    sell = (-delta).clip(lower=0.0)    # decrease in exposure
    cost = buy * commission + sell * (commission + stamp_duty)
    return cost.reindex(df.index).to_numpy()


def run_backtest(df: pd.DataFrame, pred: np.ndarray, pos: np.ndarray,
                 commission: float = None, stamp_duty: float = None) -> dict:
    """Compute step-level PnL and an equity curve. Returns a result dict."""
    commission = C.COMMISSION_RATE if commission is None else commission
    stamp_duty = C.STAMP_DUTY_RATE if stamp_duty is None else stamp_duty
    y = df[C.TARGET].to_numpy(dtype=np.float64)
    w = df[C.WEIGHT].to_numpy(dtype=np.float64)

    gross_pnl = pos * y
    tc = _ashare_cost(df, pos, commission, stamp_duty)
    net_pnl = gross_pnl - tc

    step = pd.DataFrame({
        C.DATE_COL: df[C.DATE_COL].to_numpy(),
        C.TIME_COL: df[C.TIME_COL].to_numpy(),
        "gross": gross_pnl,
        "tc": tc,
        "net": net_pnl,
        "abs_pos": np.abs(pos),
    })
    # one PnL number per timestamp = sum across symbols held at that step
    curve = (step.groupby([C.DATE_COL, C.TIME_COL])[["gross", "tc", "net", "abs_pos"]]
                 .sum().reset_index())
    curve["equity"] = curve["net"].cumsum()

    metrics = _metrics(curve, weighted_r2_val=_weighted_r2(y, pred, w))
    return {"curve": curve, "metrics": metrics, "net_pnl": net_pnl}


def _weighted_r2(y, yhat, w) -> float:
    num = np.sum(w * (y - yhat) ** 2)
    den = np.sum(w * y ** 2) + 1e-38
    return float(1.0 - num / den)


def _metrics(curve: pd.DataFrame, weighted_r2_val: float) -> dict:
    net = curve["net"].to_numpy()
    n_steps = len(net)
    n_dates = curve[C.DATE_COL].nunique()
    steps_per_day = max(n_steps / max(n_dates, 1), 1.0)
    ann = 252 * steps_per_day  # annualization factor on per-step PnL

    mean, std = net.mean(), net.std(ddof=1) if n_steps > 1 else 0.0
    downside = net[net < 0]
    dstd = downside.std(ddof=1) if len(downside) > 1 else 0.0

    sharpe = (mean / std * np.sqrt(ann)) if std > 1e-12 else 0.0
    sortino = (mean / dstd * np.sqrt(ann)) if dstd > 1e-12 else 0.0

    equity = curve["equity"].to_numpy()
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = float(drawdown.min()) if n_steps else 0.0

    total_pnl = float(equity[-1]) if n_steps else 0.0
    hit_rate = float((net > 0).mean()) if n_steps else 0.0
    total_tc = float(curve["tc"].sum())
    avg_gross_exposure = float(curve["abs_pos"].mean())
    calmar = (total_pnl / abs(max_dd)) if max_dd < -1e-12 else 0.0

    return {
        "total_pnl": total_pnl,
        "gross_pnl": float(curve["gross"].sum()),
        "total_cost": total_tc,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "mean_step_pnl": float(mean),
        "avg_gross_exposure": avg_gross_exposure,
        "n_steps": int(n_steps),
        "n_dates": int(n_dates),
        "weighted_r2": weighted_r2_val,
    }


def format_metrics(m: dict) -> str:
    return (
        "Backtest performance\n"
        "--------------------\n"
        f"  Total net PnL        : {m['total_pnl']:+.4f}\n"
        f"  Gross PnL            : {m['gross_pnl']:+.4f}\n"
        f"  Transaction cost     : {m['total_cost']:.4f}\n"
        f"  Sharpe (annualized)  : {m['sharpe']:+.3f}\n"
        f"  Sortino (annualized) : {m['sortino']:+.3f}\n"
        f"  Calmar               : {m['calmar']:+.3f}\n"
        f"  Max drawdown         : {m['max_drawdown']:+.4f}\n"
        f"  Hit rate             : {m['hit_rate']:.2%}\n"
        f"  Avg gross exposure   : {m['avg_gross_exposure']:.3f}\n"
        f"  Mean step PnL        : {m['mean_step_pnl']:+.6f}\n"
        f"  Steps / dates        : {m['n_steps']} / {m['n_dates']}\n"
        f"  Weighted R2 (pred)   : {m['weighted_r2']:+.5f}\n"
    )


def plot_results(curve: pd.DataFrame, metrics: dict, out_path: str) -> str:
    """Save an equity-curve + drawdown figure. Returns the path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    equity = curve["equity"].to_numpy()
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    x = np.arange(len(equity))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(x, equity, color="#1f77b4", lw=1.3, label="Cumulative net PnL")
    ax1.axhline(0, color="grey", lw=0.7, ls="--")
    ax1.set_ylabel("Cumulative net PnL  (sum of daily long_topk returns minus costs)")
    ax1.set_title(
        f"Out-of-sample Walk-forward Backtest\n"
        f"Sharpe {metrics['sharpe']:.2f}  |  "
        f"Sortino {metrics['sortino']:.2f}  |  "
        f"Total PnL {metrics['total_pnl']:+.3f}  |  "
        f"Max DD {metrics['max_drawdown']:+.3f}  |  "
        f"Hit rate {metrics['hit_rate']:.1%}",
        fontsize=10)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.fill_between(x, dd, 0, color="#d62728", alpha=0.5, label="Drawdown")
    ax2.set_ylabel("Drawdown from peak")
    ax2.set_xlabel("Trading day (chronological, each tick = 1 trading day)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(alpha=0.3)

    # How-to-read annotation
    note = ("How to read:\n"
            "• Top panel: running sum of daily net PnL across the long portfolio.\n"
            "  Rising = strategy making money; a sustained positive slope is the goal.\n"
            "• Bottom panel: loss from the most recent equity peak (always ≤ 0).\n"
            "  Deep troughs = large drawdowns; shallower/shorter = better risk control.\n"
            "• Sharpe: annualised return / volatility. >1 is good; >2 is strong.\n"
            "• Hit rate: fraction of days with positive net PnL.")
    fig.text(0.01, 0.01, note, fontsize=7, va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9", ec="#cccccc", alpha=0.85))

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
