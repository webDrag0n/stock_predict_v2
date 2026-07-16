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
                 commission: float = None, stamp_duty: float = None,
                 pred_in_return_space: bool = True) -> dict:
    """Compute step-level PnL and an equity curve. Returns a result dict.

    When `df` carries a `close` column AND `pred_in_return_space` is True, also
    computes a per-date price-RMSE curve: the model predicts the forward return
    responder_6, so the predicted next-window price is close*(1+pred) and the
    realized one is close*(1+y). The per-row price error is close*(pred-y); RMSE
    on each date is sqrt(mean_stocks(error**2)), in yuan. No look-ahead.

    For ranking objectives (lambdarank) the prediction is an ORDERING score, not
    a return, so pass pred_in_return_space=False to skip price-RMSE/weighted-R2;
    the rank-based IC/ICIR below stay valid either way.
    """
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

    rmse_curve = (_price_rmse_curve(df, pred, y, curve)
                  if pred_in_return_space else None)

    r2 = _weighted_r2(y, pred, w) if pred_in_return_space else float("nan")
    metrics = _metrics(curve, weighted_r2_val=r2)
    metrics["mean_price_rmse"] = (float(rmse_curve["rmse"].mean())
                                  if rmse_curve is not None and len(rmse_curve)
                                  else float("nan"))
    ic_mean, icir = _prediction_ic(df, pred)
    metrics["pred_ic"] = ic_mean
    metrics["pred_icir"] = icir
    return {"curve": curve, "metrics": metrics, "net_pnl": net_pnl,
            "rmse_curve": rmse_curve}


def run_backtest_periodic(df: pd.DataFrame, pred: np.ndarray, pos: np.ndarray,
                          rebalance_days: int = None, commission: float = None,
                          stamp_duty: float = None,
                          pred_in_return_space: bool = True) -> dict:
    """Non-overlapping periodic-rebalance backtest — the leak-free, un-inflated
    counterpart to run_backtest for an N-day forward target.

    run_backtest earns pos * responder_6 on EVERY date. With a FORWARD_WINDOW=20
    target that is the 20-day forward return, so each 20-day return is counted on
    20 overlapping entry days, and _metrics annualizes per-step as if it were a
    1-day return (ann=252) — both inflate Sharpe/PnL massively.

    This engine instead:
      1. adopts the strategy's target weights only on a rebalance grid (every
         `rebalance_days` trading dates), holding them fixed in between;
      2. earns the ACTUAL daily return (return_1d / pct_chg) of held names each
         day — so each day's return is counted exactly once;
      3. charges A-share costs only on the turnover at each rebalance date.
    The equity curve is then daily and self-consistent: _metrics' ann=252 is now
    correct because each step really is a 1-day return. Requires a per-row daily
    return column; falls back to run_backtest if none is present.
    """
    rebalance_days = rebalance_days or C.FORWARD_WINDOW
    commission = C.COMMISSION_RATE if commission is None else commission
    stamp_duty = C.STAMP_DUTY_RATE if stamp_duty is None else stamp_duty

    ret_col = ("return_1d" if "return_1d" in df.columns
               else ("pct_chg" if "pct_chg" in df.columns else None))
    if ret_col is None:                       # no daily return → cannot do this
        return run_backtest(df, pred, pos, commission, stamp_duty,
                            pred_in_return_space)

    work = pd.DataFrame({
        "date": df[C.DATE_COL].to_numpy(),
        "sym": (df[C.SYMBOL_COL].to_numpy() if C.SYMBOL_COL in df.columns
                else np.arange(len(df))),
        "target": np.asarray(pos, dtype=np.float64),
        "ret": df[ret_col].to_numpy(dtype=np.float64),
    })
    # pct_chg is in percent; return_1d is already decimal
    if ret_col == "pct_chg":
        work["ret"] = work["ret"] / 100.0
    # ADV (avg daily dollar volume, 元) for the square-root impact model. tushare
    # `amount` is in 千元 → ×1000 = 元. Absent → impact cost disabled.
    has_adv = "amount" in df.columns
    if has_adv:
        work["adv"] = df["amount"].to_numpy(dtype=np.float64) * 1000.0

    dates = np.sort(work["date"].unique())
    # rebalance on every `rebalance_days`-th date (grid anchored at the start)
    rebal_dates = set(dates[::rebalance_days].tolist())

    # held weight per (date, sym): target on rebalance dates, else carry forward
    wide_t = work.pivot_table(index="date", columns="sym", values="target",
                              aggfunc="sum").reindex(dates).fillna(0.0)
    held = wide_t.copy()
    held.loc[~held.index.isin(rebal_dates)] = np.nan
    held = held.ffill().fillna(0.0)           # carry last rebalance weights

    # the weight earning day t's return is the weight held coming INTO t
    # (set at the previous close) → shift down one trading date
    entering = held.shift(1).fillna(0.0)

    # Align to wide_t's columns: pivot_table(dropna=True, default) silently drops
    # any symbol whose `ret` is all-NaN over the window (e.g. newly listed or
    # suspended the whole test window in whole-market mode). Without this reindex
    # wide_r would have fewer columns than `entering` (built from wide_t) and the
    # element-wise product below would fail to broadcast.
    wide_r = (work.pivot_table(index="date", columns="sym", values="ret",
                               aggfunc="mean")
              .reindex(dates).reindex(columns=wide_t.columns).fillna(0.0))

    # daily gross portfolio return = sum_sym entering_weight * daily_return
    gross_daily = (entering.to_numpy() * wide_r.to_numpy()).sum(axis=1)

    # turnover cost: |held_t - held_{t-1}| per sym, only nonzero on rebalance days
    dw = held.to_numpy() - held.shift(1).fillna(0.0).to_numpy()
    buy = np.clip(dw, 0.0, None).sum(axis=1)
    sell = np.clip(-dw, 0.0, None).sum(axis=1)
    tc_daily = buy * commission + sell * (commission + stamp_duty)

    # Liquidity-aware market impact (square-root model), the realistic small-cap
    # cost the flat commission understates. For a trade of weight |dw| in a stock,
    # the traded notional is |dw|·capital; participation = notional / ADV; impact
    # fraction = IMPACT_K·sqrt(participation), and the yuan cost as a fraction of
    # the book is impact_fraction·|dw|. Summed across stocks per rebalance day.
    impact_daily = np.zeros(len(dates), dtype=np.float64)
    if has_adv and getattr(C, "USE_IMPACT_COST", True):
        cap = getattr(C, "INITIAL_CAPITAL", 1_000_000.0)
        k = getattr(C, "IMPACT_K", 0.1)
        wide_adv = work.pivot_table(index="date", columns="sym", values="adv",
                                    aggfunc="mean").reindex(dates).reindex(
                                        columns=wide_t.columns)
        adv_arr = wide_adv.to_numpy()
        adv_arr = np.where(np.isfinite(adv_arr) & (adv_arr > 0), adv_arr, np.inf)
        trade_notional = np.abs(dw) * cap                  # 元 traded per stock
        participation = trade_notional / adv_arr           # fraction of ADV
        impact_frac = k * np.sqrt(participation)           # cost as frac of trade
        impact_daily = np.nansum(impact_frac * np.abs(dw), axis=1)
    tc_daily = tc_daily + impact_daily

    net_daily = gross_daily - tc_daily
    curve = pd.DataFrame({
        C.DATE_COL: dates,
        C.TIME_COL: 0,
        "gross": gross_daily,
        "tc": tc_daily,
        "tc_commission": (buy * commission + sell * (commission + stamp_duty)),
        "tc_impact": impact_daily,
        "net": net_daily,
        "abs_pos": np.abs(entering.to_numpy()).sum(axis=1),
    })
    curve["equity"] = curve["net"].cumsum()

    r2 = float("nan")
    metrics = _metrics(curve, weighted_r2_val=r2)
    metrics["total_cost_commission"] = float(curve["tc_commission"].sum())
    metrics["total_cost_impact"] = float(curve["tc_impact"].sum())

    # Price-RMSE: compute on rebalance days only (where `pred` is meaningful).
    # Between rebalances we're holding, so pred-vs-realized doesn't apply daily.
    rmse_curve = None
    if pred_in_return_space:
        # pred is a forward return → can compute price error on rebalance dates.
        # For simplicity: report NaN (full daily RMSE requires daily predictions).
        # Future enhancement: compute per-rebalance-window price error at horizon.
        rmse_curve = None    # placeholder; daily RMSE invalid for periodic strat
        metrics["mean_price_rmse"] = float("nan")
    else:
        metrics["mean_price_rmse"] = float("nan")

    ic_mean, icir = _prediction_ic(df, pred)
    metrics["pred_ic"] = ic_mean
    metrics["pred_icir"] = icir
    metrics["rebalance_days"] = rebalance_days
    return {"curve": curve, "metrics": metrics, "net_pnl": net_daily,
            "rmse_curve": rmse_curve}


def _prediction_ic(df: pd.DataFrame, pred: np.ndarray) -> tuple[float, float]:
    """Daily cross-sectional rank-IC of the prediction vs realized target.

    The single most important accuracy number for a ranking strategy: per date,
    Spearman corr(pred, responder_6) across stocks, then mean (IC) and
    mean/std (ICIR) over dates. Rank-based, so it is identical whether `pred` is
    a return forecast or a ranking score, and comparable across FORWARD_WINDOW.
    """
    from scipy import stats
    tmp = pd.DataFrame({
        "date": df[C.DATE_COL].to_numpy(),
        "pred": np.asarray(pred, dtype=np.float64),
        "y": df[C.TARGET].to_numpy(dtype=np.float64),
    })

    def _one(g):
        if len(g) < 5 or g["pred"].nunique() < 2 or g["y"].nunique() < 2:
            return np.nan
        return stats.spearmanr(g["pred"], g["y"]).correlation

    daily = tmp.groupby("date", group_keys=False).apply(_one, include_groups=False)
    daily = daily.dropna()
    if len(daily) == 0:
        return float("nan"), float("nan")
    ic = float(daily.mean())
    icir = float(daily.mean() / daily.std()) if daily.std() > 1e-12 else float("nan")
    return ic, icir


def _price_rmse_curve(df: pd.DataFrame, pred: np.ndarray, y: np.ndarray,
                      curve: pd.DataFrame) -> pd.DataFrame | None:
    """Per-date price-space RMSE of the prediction, aligned to `curve` dates.

    Returns a DataFrame [date_id, rmse, n] sorted to match `curve`'s date
    ordering (so plots share the equity curve's x-axis), or None when no
    `close` column is available.
    """
    if "close" not in df.columns:
        return None
    close = df["close"].to_numpy(dtype=np.float64)
    price_err = close * (np.asarray(pred, dtype=np.float64) - y)  # yuan per row
    tmp = pd.DataFrame({C.DATE_COL: df[C.DATE_COL].to_numpy(),
                        "sq": price_err ** 2})
    tmp = tmp.dropna(subset=["sq"])
    agg = tmp.groupby(C.DATE_COL)["sq"].agg(["mean", "count"])
    agg["rmse"] = np.sqrt(agg["mean"])
    # align to the equity curve's date ordering so x-axes match exactly
    out = (curve[[C.DATE_COL]]
           .merge(agg[["rmse", "count"]].reset_index(), on=C.DATE_COL, how="left")
           .rename(columns={"count": "n"}))
    return out


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
        + (f"  Pred OOS IC          : {m['pred_ic']:+.5f}"
           f"  (ICIR {m['pred_icir']:+.3f})\n"
           if m.get("pred_ic") == m.get("pred_ic") and "pred_ic" in m else "")
        + (f"  Mean price RMSE (¥)  : {m['mean_price_rmse']:.4f}\n"
           if m.get("mean_price_rmse") == m.get("mean_price_rmse")  # not NaN
           and "mean_price_rmse" in m else "")
    )


def plot_results(curve: pd.DataFrame, metrics: dict, out_path: str,
                 rmse_curve: pd.DataFrame = None) -> str:
    """Save an equity-curve + drawdown (+ price-RMSE) figure. Returns the path.

    When `rmse_curve` is provided, a third panel plots the per-trading-day
    price-RMSE (yuan) of the model's predicted prices across the cross-section.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    equity = curve["equity"].to_numpy()
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    x = np.arange(len(equity))

    has_rmse = rmse_curve is not None and len(rmse_curve)
    if has_rmse:
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 10), sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1.4]})
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(x, equity, color="#1f77b4", lw=1.3, label="Cumulative net PnL")
    ax1.axhline(0, color="grey", lw=0.7, ls="--")
    ax1.set_ylabel("Cumulative net PnL  (sum of daily long_topk returns minus costs)")
    rmse_title = (f"  |  Mean price RMSE ¥{metrics['mean_price_rmse']:.3f}"
                  if metrics.get("mean_price_rmse") == metrics.get("mean_price_rmse")
                  and "mean_price_rmse" in metrics else "")
    ax1.set_title(
        f"Out-of-sample Walk-forward Backtest\n"
        f"Sharpe {metrics['sharpe']:.2f}  |  "
        f"Sortino {metrics['sortino']:.2f}  |  "
        f"Total PnL {metrics['total_pnl']:+.3f}  |  "
        f"Max DD {metrics['max_drawdown']:+.3f}  |  "
        f"Hit rate {metrics['hit_rate']:.1%}{rmse_title}",
        fontsize=10)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.fill_between(x, dd, 0, color="#d62728", alpha=0.5, label="Drawdown")
    ax2.set_ylabel("Drawdown from peak")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(alpha=0.3)

    if has_rmse:
        rmse = rmse_curve["rmse"].to_numpy(dtype=float)
        xr = np.arange(len(rmse))
        mean_rmse = np.nanmean(rmse)
        ax3.plot(xr, rmse, color="#8c564b", lw=1.1,
                 label="Daily price RMSE (¥)")
        ax3.axhline(mean_rmse, color="#8c564b", lw=0.8, ls="--",
                    label=f"Mean ¥{mean_rmse:.3f}")
        ax3.set_ylabel("Price RMSE (¥)")
        ax3.set_xlabel("Trading day (chronological, each tick = 1 trading day)")
        ax3.legend(loc="upper left", fontsize=8)
        ax3.grid(alpha=0.3)
    else:
        ax2.set_xlabel("Trading day (chronological, each tick = 1 trading day)")

    # How-to-read annotation
    note = ("How to read:\n"
            "• Top panel: running sum of daily net PnL across the long portfolio.\n"
            "  Rising = strategy making money; a sustained positive slope is the goal.\n"
            "• Middle panel: loss from the most recent equity peak (always ≤ 0).\n"
            "  Deep troughs = large drawdowns; shallower/shorter = better risk control.\n"
            "• Bottom panel: per-day RMSE of predicted vs realized price (¥), across stocks.\n"
            "  Lower = sharper price forecasts; spikes mark days the model misread the most.\n"
            "• Sharpe: annualised return / volatility. >1 is good; >2 is strong.\n"
            "• Hit rate: fraction of days with positive net PnL.")
    fig.text(0.01, 0.01, note, fontsize=7, va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9", ec="#cccccc", alpha=0.85))

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
