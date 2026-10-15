"""Walk-forward backtest engine and performance metrics.

唯一的回测引擎是 `run_backtest_periodic`（非重叠周期调仓）。旧的每日重叠引擎
`run_backtest` 已删除：对 N 日前瞻目标，它每天都在每条 entry 上赚 pos*responder_6
（N 日收益被 N 个重叠 entry 日重复计），且按 ann=252 年化（把 N 步当 1 日步），
Sharpe/PnL 双重虚高。周期引擎在每个调仓日采用策略权重、中间持有，每天赚持有
标的的【真实日收益】，每个收益只计一次。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def run_backtest_periodic(df: pd.DataFrame, pred: np.ndarray, pos: np.ndarray,
                          rebalance_days: int = None, commission: float = None,
                          stamp_duty: float = None,
                          pred_in_return_space: bool = True) -> dict:
    """Non-overlapping periodic-rebalance backtest — the leak-free, un-inflated
    engine for an N-day forward target (the legacy daily-overlap `run_backtest`
    was removed: it counted each N-day return on N overlapping entry days and
    mis-annualized per-step, inflating Sharpe/PnL massively).

    This engine:
      1. adopts the strategy's target weights only on a rebalance grid (every
         `rebalance_days` trading dates), holding them fixed in between;
      2. earns the ACTUAL daily return (return_1d / pct_chg) of held names each
         day — so each day's return is counted exactly once;
      3. charges A-share costs only on the turnover at each rebalance date.
    The equity curve is then daily and self-consistent: _metrics' ann=252 is now
    correct because each step really is a 1-day return. Requires a per-row daily
    return column (return_1d 或 pct_chg)；缺失时显式报错而非回落到旧引擎。
    """
    rebalance_days = rebalance_days or C.FORWARD_WINDOW
    commission = C.COMMISSION_RATE if commission is None else commission
    stamp_duty = C.STAMP_DUTY_RATE if stamp_duty is None else stamp_duty

    ret_col = ("return_1d" if "return_1d" in df.columns
               else ("pct_chg" if "pct_chg" in df.columns else None))
    if ret_col is None:
        # 旧的 run_backtest（每日重叠引擎）已删除；没有逐日收益列就无法
        # 构造口径正确的日频净值，显式失败优于静默用虚高引擎。
        raise ValueError(
            "run_backtest_periodic 需要逐日收益列（return_1d 或 pct_chg）；"
            "旧的每日重叠引擎 run_backtest 已删除（Sharpe/PnL 虚高）。")

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

    # 成本与每日重放引擎共用唯一实现（trading.trade_costs）：佣金(双边, 每笔最低
    # MIN_COMMISSION) + 印花税(卖出) + 平方根冲击。逐 (date, sym) 计额再跨股求和，
    # 保证两套引擎的成本公式不再各自实现、永不漂移。
    from trading import trade_costs
    dw = held.to_numpy() - held.shift(1).fillna(0.0).to_numpy()
    cap = getattr(C, "INITIAL_CAPITAL", 1_000_000.0)
    cost_params = {"commission": commission, "stamp_duty": stamp_duty,
                   "use_impact": getattr(C, "USE_IMPACT_COST", True)}
    buy_w = np.clip(dw, 0.0, None)
    sell_w = np.clip(-dw, 0.0, None)
    if has_adv:
        # ADV (avg daily dollar volume, 元) —— 已在上面从 amount(千元)×1000 得到。
        wide_adv = work.pivot_table(index="date", columns="sym", values="adv",
                                    aggfunc="mean").reindex(dates).reindex(
                                        columns=wide_t.columns)
        adv_arr = wide_adv.to_numpy()
        adv_arr = np.where(np.isfinite(adv_arr) & (adv_arr > 0), adv_arr, np.inf)
    else:
        adv_arr = np.full(dw.shape, np.inf)
    comm_b, _, imp_b = trade_costs(buy_w * cap, "buy", adv_arr, cost_params)
    comm_s, stamp_s, imp_s = trade_costs(sell_w * cap, "sell", adv_arr, cost_params)
    # trade_costs 以元计；净值曲线是权重口径（占本金分数），除以 cap 换算
    tc_flat = (comm_b + comm_s + stamp_s) / cap    # 佣金+印花税（不含冲击）
    impact_arr = (imp_b + imp_s) / cap             # 冲击成本
    tc_daily = (tc_flat + impact_arr).sum(axis=1)
    impact_daily = impact_arr.sum(axis=1)

    net_daily = gross_daily - tc_daily
    curve = pd.DataFrame({
        C.DATE_COL: dates,
        C.TIME_COL: 0,
        "gross": gross_daily,
        "tc": tc_daily,
        "tc_commission": tc_flat.sum(axis=1),
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
    # 年化口径：mean×ann 是算术年化收益（占本金分数），std×√ann 是年化波动。
    ann_return = float(mean * ann)
    ann_vol = float(std * np.sqrt(ann))

    equity = curve["equity"].to_numpy()
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = float(drawdown.min()) if n_steps else 0.0

    total_pnl = float(equity[-1]) if n_steps else 0.0
    hit_rate = float((net > 0).mean()) if n_steps else 0.0
    total_tc = float(curve["tc"].sum())
    avg_gross_exposure = float(curve["abs_pos"].mean())
    # Calmar = 年化收益 / |最大回撤|（年化口径，跨回测区间可比；旧口径
    # total_pnl/|max_dd| 随区间长度缩放，不可比）。
    calmar = (ann_return / abs(max_dd)) if max_dd < -1e-12 else 0.0

    return {
        "total_pnl": total_pnl,
        "gross_pnl": float(curve["gross"].sum()),
        "total_cost": total_tc,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
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
        + (f"  Pred OOS IC          : {m['pred_ic']:+.5f}"
           f"  (ICIR {m['pred_icir']:+.3f})\n"
           if m.get("pred_ic") == m.get("pred_ic") and "pred_ic" in m else "")
        + (f"  Mean price RMSE (¥)  : {m['mean_price_rmse']:.4f}\n"
           if m.get("mean_price_rmse") == m.get("mean_price_rmse")  # not NaN
           and "mean_price_rmse" in m else "")
    )

