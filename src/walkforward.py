"""滚动走动回测（walk-forward）—— 严格防泄露的样本外验证。

针对单次划分回测 Sharpe 虚高（窗口运气 + 因子选择偷看测试集）的问题，本引擎：

  1. 滚动训练：每步用前 TRAIN_WINDOW 天训练，预测其后 TEST_WINDOW 天（样本外）
  2. embargo：训练段与测试段之间剔除 EMBARGO_DAYS 天，杜绝 N 日标签跨界泄露
  3. 因子选择隔离：每个窗口的因子选择只用该窗口的训练段（select_passing_factors）
  4. 拼接所有窗口的样本外预测 → 一条长达 1~2 年的连续 OOS 净值曲线

直接在「命名因子面板」(load_factor_panel) 上操作，模型用命名因子列训练，
避免 feature_NN 映射，保证窗口内因子选择与训练用的是同一套列。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
from factor_eval import select_passing_factors
from strategy import generate_positions
from backtest import run_backtest, format_metrics


def _regime_scale(panel: pd.DataFrame, test_dates, factors: list[str],
                  lookback: int = 10) -> tuple[float, float]:
    """Trailing IC of value factors before the test window (10-day lookback).
    Returns (scale, avg_ic): scale=0.3 if avg_ic<0, else 1.0.
    """
    from scipy import stats
    te_min = test_dates[0]
    pre = [d for d in sorted(panel["date"].unique()) if d < te_min][-lookback:]
    if len(pre) < lookback // 3:
        return 1.0, np.nan
    sub = panel[panel["date"].isin(pre)]
    ics = []
    for f in factors:
        if f not in sub.columns:
            continue
        day_ic = sub.groupby("date", group_keys=False).apply(
            lambda g, fac=f: (
                stats.spearmanr(g[fac].dropna(),
                                g.loc[g[fac].notna(), C.TARGET]).correlation
                if g[fac].notna().sum() >= 20 else np.nan),
            include_groups=False)
        ics.extend(day_ic.dropna().tolist())
    avg_ic = float(np.mean(ics)) if ics else np.nan
    scale = 1.0 if (np.isnan(avg_ic) or avg_ic >= 0.0) else 0.3
    return scale, avg_ic


def _train_predict(train: pd.DataFrame, test: pd.DataFrame,
                   feat_cols: list[str]) -> np.ndarray:
    """在 train 上训 LGB，对 test 样本外预测。命名因子列直接用。"""
    Xtr = train[feat_cols].to_numpy(np.float32, copy=True)
    np.nan_to_num(Xtr, copy=False, nan=0.0)
    ytr = train[C.TARGET].to_numpy()
    wtr = train[C.WEIGHT].to_numpy() if C.WEIGHT in train else None

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
        reg_lambda=0.1, objective="l2",
        random_state=C.RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(Xtr, ytr, sample_weight=wtr)

    Xte = test[feat_cols].to_numpy(np.float32, copy=True)
    np.nan_to_num(Xte, copy=False, nan=0.0)
    return np.asarray(model.predict(Xte), dtype=np.float64).ravel()


def run_walkforward(panel: pd.DataFrame, all_factor_cols: list[str],
                    select_features: bool = False,
                    strategy: str = "long_topk",
                    verbose: bool = True) -> dict:
    """滚动回测。返回拼接后的 OOS 结果与各窗口指标分布。

    panel: 命名因子面板（含 date, symbol, 因子..., responder_6, weight）
    all_factor_cols: 全部候选因子名
    select_features: True 则每窗口在训练段内做因子准入筛选（隔离，无泄露）
    """
    # weight 列保障（load_factor_panel 默认不含 weight，这里补等权）
    if C.WEIGHT not in panel.columns:
        panel = panel.copy()
        panel[C.WEIGHT] = 1.0
    # 整数日期序号，便于按窗口切分
    panel = panel.sort_values("date").reset_index(drop=True)
    dates = np.sort(panel["date"].unique())
    n_dates = len(dates)

    train_w, test_w = C.TRAIN_WINDOW, C.TEST_WINDOW
    step, embargo = C.WF_STEP, C.EMBARGO_DAYS

    oos_frames = []
    oos_pos_frames = []
    window_metrics = []

    start = train_w + embargo
    win_id = 0
    while start + test_w <= n_dates:
        tr_lo_i = start - embargo - train_w
        tr_hi_i = start - embargo                  # exclusive
        te_lo_i = start
        te_hi_i = start + test_w                   # exclusive

        tr_dates = dates[tr_lo_i:tr_hi_i]
        te_dates = dates[te_lo_i:te_hi_i]
        train = panel[panel["date"].isin(tr_dates)]
        test = panel[panel["date"].isin(te_dates)].copy()

        # 窗口内因子选择（仅用训练段日期）
        if select_features:
            feat = select_passing_factors(panel, date_lo=tr_dates[0],
                                          date_hi=tr_dates[-1])
            feat = [f for f in feat if f in all_factor_cols]
            if not feat:                            # 该窗口无因子达标 → 退回全特征
                feat = all_factor_cols
        else:
            feat = all_factor_cols

        # #4 multi-seed ensemble
        pred = _train_predict(train, test, feat)

        test["prediction"] = pred

        # Method A: dynamic position scaling — reduce exposure when value IC negative
        value_factors = [f for f in ["bp", "ep", "value_composite"] if f in all_factor_cols]
        scale, avg_ic = _regime_scale(panel, te_dates, value_factors)

        pos = generate_positions(_with_ids(test), pred, strategy=strategy)
        res = run_backtest(_with_ids(test), pred, pos)
        m = res["metrics"]
        window_metrics.append({
            "window": win_id,
            "train": f"{pd.Timestamp(tr_dates[0]).date()}~{pd.Timestamp(tr_dates[-1]).date()}",
            "test": f"{pd.Timestamp(te_dates[0]).date()}~{pd.Timestamp(te_dates[-1]).date()}",
            "n_feat": len(feat), "regime_ic": round(avg_ic, 4) if not np.isnan(avg_ic) else None,
            "pos_scale": scale,
            "PnL": m["total_pnl"], "Sharpe": m["sharpe"],
            "MaxDD": m["max_drawdown"], "hit": m["hit_rate"],
        })
        oos_frames.append(test)
        oos_pos_frames.append(pos)   # keep risk-adjusted positions
        if verbose:
            regime_str = f"IC={avg_ic:+.3f} scale={scale:.0%}" if not np.isnan(avg_ic) else ""
            print(f"  窗口{win_id}: 训练{tr_dates[0].astype('datetime64[D]')}~"
                  f"{tr_dates[-1].astype('datetime64[D]')} → "
                  f"测试{te_dates[0].astype('datetime64[D]')}~"
                  f"{te_dates[-1].astype('datetime64[D]')} | "
                  f"{len(feat)}因子 | {regime_str} | OOS Sharpe={m['sharpe']:+.2f}")
        start += step
        win_id += 1

    if not oos_frames:
        raise ValueError("数据不足以构成任何滚动窗口，请减小 TRAIN_WINDOW/TEST_WINDOW")

    # 拼接所有窗口的连续 OOS 净值（使用已经过熔断/波动率调整的持仓）
    oos = pd.concat(oos_frames, ignore_index=True)
    oos_pred = oos["prediction"].to_numpy()
    oos_pos = np.concatenate(oos_pos_frames)   # risk-adjusted, not re-generated
    oos_res = run_backtest(_with_ids(oos), oos_pred, oos_pos)

    return {
        "oos_result": oos_res,
        "window_metrics": pd.DataFrame(window_metrics),
        "n_windows": win_id,
        "oos": oos,
    }


def _apply_circuit_breaker(pos: np.ndarray, test: pd.DataFrame,
                            thresh1: float = -0.05, thresh2: float = -0.10,
                            scale1: float = 0.3, scale2: float = 0.0) -> np.ndarray:
    """Intra-window circuit breaker: reduce exposure when cumulative loss deepens.

    Tracks daily portfolio P&L using actual stock returns (pct_chg).
    Scale at day t is based solely on returns from test_start to day t-1 (no look-ahead).
      cumLoss < thresh1 → scale = scale1 (0.3)
      cumLoss < thresh2 → scale = scale2 (0.0, full exit)
    """
    if "pct_chg" not in test.columns:
        return pos
    dates = sorted(test["date"].unique())
    pos_out = pos.copy()
    running = 0.0
    for d in dates:
        mask = (test["date"] == d).to_numpy()
        # scale based on loss SO FAR (before today's return)
        scale = 1.0
        if running < thresh2:
            scale = scale2
        elif running < thresh1:
            scale = scale1
        pos_out[mask] *= scale
        # update with today's actual 1-day return (pct_chg is in %)
        daily_ret = (pos_out[mask] * test.loc[test["date"] == d, "pct_chg"].to_numpy() / 100.0).sum()
        running += daily_ret
    return pos_out


def _apply_vol_targeting(pos: np.ndarray, test: pd.DataFrame,
                         target_annual_vol: float = 0.15,
                         vol_window: int = 20) -> np.ndarray:
    """Volatility targeting: scale positions to hit a fixed annualised volatility.

    Uses realised daily portfolio vol (computed from pct_chg) over the past
    vol_window days to set position scale. Reduces exposure in turbulent periods.
    """
    if "pct_chg" not in test.columns:
        return pos
    target_daily = target_annual_vol / np.sqrt(252)
    dates = sorted(test["date"].unique())
    pos_out = pos.copy()
    daily_rets = []
    for d in dates:
        mask = (test["date"] == d).to_numpy()
        scale = 1.0
        if len(daily_rets) >= 5:
            vol = np.std(daily_rets[-vol_window:]) + 1e-9
            scale = min(1.0, target_daily / vol)
        pos_out[mask] *= scale
        dr = (pos_out[mask] * test.loc[test["date"] == d, "pct_chg"].to_numpy() / 100.0).sum()
        daily_rets.append(dr)
    return pos_out


def _with_ids(df: pd.DataFrame) -> pd.DataFrame:
    """给命名因子面板补 date_id/time_id/symbol_id（策略与回测按这些列分组）。"""
    out = df.copy()
    out[C.DATE_COL] = out["date"].rank(method="dense").astype(int)
    out[C.TIME_COL] = 0
    out[C.SYMBOL_COL] = out["symbol"].astype("category").cat.codes
    return out


def plot_dashboard(result: dict, out_path: str, tag: str = "",
                   initial_capital: float = None) -> str:
    """Combine all backtest views into one multi-panel dashboard figure.

    Panels:
      A. Account value in money terms (¥), compounding daily net returns
      B. Cumulative return (%) — same shape as A but unit-free
      C. Per-window out-of-sample Sharpe (bar chart) — robustness across time
      D. Drawdown in money terms (¥) — peak-to-trough loss
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"

    cap0 = initial_capital if initial_capital is not None else C.INITIAL_CAPITAL
    curve = result["oos_result"]["curve"]
    m = result["oos_result"]["metrics"]
    wm = result["window_metrics"]

    net = curve["net"].to_numpy()
    equity_ret = np.cumprod(1.0 + net)
    account = cap0 * equity_ret
    money_pnl = account - cap0
    cum_ret_pct = (equity_ret - 1.0) * 100.0
    peak = np.maximum.accumulate(account)
    dd_money = account - peak
    x = np.arange(len(account))

    # Benchmark: equal-weight buy-and-hold of the full universe.
    # Use return_1d (daily close-to-close pct change, decimal form) from the OOS panel.
    oos = result.get("oos")
    ret_col = "return_1d" if (oos is not None and "return_1d" in oos.columns) else None
    bench_account = bench_cum_pct = alpha_pct = None
    if ret_col and oos is not None:
        bench_daily = (oos.groupby("date")[ret_col].mean()
                          .sort_index().to_numpy())
        if len(bench_daily) == len(net):
            bench_ret = np.cumprod(1.0 + bench_daily)
            bench_account = cap0 * bench_ret
            bench_cum_pct = (bench_ret - 1.0) * 100.0
            alpha_pct = cum_ret_pct - bench_cum_pct  # excess return vs benchmark

    final_pnl = money_pnl[-1] if len(money_pnl) else 0.0
    total_ret_pct = cum_ret_pct[-1] if len(cum_ret_pct) else 0.0
    max_dd_money = dd_money.min() if len(dd_money) else 0.0
    bench_end = (f"  |  Benchmark {bench_cum_pct[-1]:+.1f}%"
                 f"  Alpha {alpha_pct[-1]:+.1f}%"
                 if bench_cum_pct is not None else "")

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.3, 1.3], hspace=0.42, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])

    title = (f"Walk-forward Out-of-Sample Dashboard{(' — ' + tag) if tag else ''}\n"
             f"Start ¥{cap0:,.0f}  ->  End ¥{account[-1]:,.0f}   "
             f"(Net P/L ¥{final_pnl:+,.0f}, {total_ret_pct:+.1f}%){bench_end}   |   "
             f"Sharpe {m['sharpe']:.2f}  Sortino {m['sortino']:.2f}  "
             f"Calmar {m['calmar']:.2f}  Hit {m['hit_rate']:.1%}")
    fig.suptitle(title, fontsize=11, y=0.99)
    _dashboard_panels(ax_a, ax_b, ax_c, ax_d, x, account, money_pnl,
                      cum_ret_pct, dd_money, cap0, wm, final_pnl, max_dd_money,
                      bench_account=bench_account, alpha_pct=alpha_pct,
                      bench_cum_pct=bench_cum_pct)

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _money_fmt(v: float) -> str:
    """Format a yuan amount compactly (e.g. 1.2M, -340K)."""
    a = abs(v)
    s = "-" if v < 0 else ""
    if a >= 1e6:
        return f"{s}¥{a/1e6:.2f}M"
    if a >= 1e3:
        return f"{s}¥{a/1e3:.0f}K"
    return f"{s}¥{a:.0f}"


def _dashboard_panels(ax_a, ax_b, ax_c, ax_d, x, account, money_pnl,
                      cum_ret_pct, dd_money, cap0, wm, final_pnl, max_dd_money,
                      bench_account=None, alpha_pct=None, bench_cum_pct=None):
    import matplotlib.ticker as mticker

    # --- Panel A: account value (¥) vs benchmark ---
    ax_a.plot(x, account, color="#1a7f37", lw=1.6, label="Strategy account (¥)")
    if bench_account is not None:
        ax_a.plot(x, bench_account, color="#ff7f0e", lw=1.2, ls="--",
                  label="Equal-weight benchmark (¥)")
    ax_a.axhline(cap0, color="grey", lw=0.7, ls=":",
                 label=f"Initial capital ¥{cap0:,.0f}")
    ax_a.fill_between(x, cap0, account, where=(account >= cap0),
                      color="#1a7f37", alpha=0.10)
    ax_a.fill_between(x, cap0, account, where=(account < cap0),
                      color="#d62728", alpha=0.10)
    ax_a.set_title(
        "A. Account Value vs Equal-weight Universe Benchmark  (both starting at initial capital)",
        fontsize=10, loc="left")
    ax_a.set_ylabel("Account value (¥)")
    ax_a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"¥{v/1e6:.2f}M"))
    ax_a.legend(loc="upper left", fontsize=8)
    ax_a.grid(alpha=0.3)
    bench_label = (f"\nBenchmark: {_money_fmt(bench_account[-1])}"
                   if bench_account is not None else "")
    ax_a.annotate(
        f"Strategy end: {_money_fmt(account[-1])}\nNet P/L: {_money_fmt(final_pnl)}{bench_label}",
        xy=(x[-1], account[-1]), fontsize=8, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#eefaf0", ec="#1a7f37"))

    # --- Panel B: alpha curve (strategy return % − benchmark %) ---
    if alpha_pct is not None:
        ax_b.plot(x, alpha_pct, color="#9467bd", lw=1.3, label="Alpha (strategy − benchmark)")
        ax_b.fill_between(x, alpha_pct, 0, where=(alpha_pct >= 0),
                          color="#9467bd", alpha=0.15)
        ax_b.fill_between(x, alpha_pct, 0, where=(alpha_pct < 0),
                          color="#d62728", alpha=0.15)
        ax_b.axhline(0, color="grey", lw=0.8, ls="--")
        ax_b.set_title("B. Cumulative Alpha (Strategy % Return − Benchmark % Return)",
                       fontsize=10, loc="left")
        ax_b.set_ylabel("Alpha (%)")
        ax_b.annotate(f"End alpha: {alpha_pct[-1]:+.1f}%",
                      xy=(x[-1], alpha_pct[-1]), fontsize=8, ha="right",
                      bbox=dict(boxstyle="round,pad=0.2", fc="#f4f0ff", ec="#9467bd"))
    else:
        ax_b.plot(x, cum_ret_pct, color="#1f77b4", lw=1.3)
        ax_b.axhline(0, color="grey", lw=0.7, ls="--")
        ax_b.set_title("B. Cumulative Return (%)", fontsize=10, loc="left")
        ax_b.set_ylabel("Return (%)")
    ax_b.set_xlabel("Trading day")
    ax_b.legend(loc="upper left", fontsize=8)
    ax_b.grid(alpha=0.3)

    # --- Panel C: per-window OOS Sharpe ---
    if wm is not None and len(wm):
        colors = ["#1a7f37" if s >= 0 else "#d62728" for s in wm["Sharpe"]]
        ax_c.bar(wm["window"].astype(str), wm["Sharpe"], color=colors, alpha=0.8)
        ax_c.axhline(0, color="grey", lw=0.7)
        npos = int((wm["Sharpe"] > 0).sum())
        ax_c.set_title(f"C. Per-window OOS Sharpe  ({npos}/{len(wm)} positive)",
                       fontsize=10, loc="left")
        ax_c.set_ylabel("Sharpe")
        ax_c.set_xlabel("Walk-forward window #")
        ax_c.grid(alpha=0.3, axis="y")

    # --- Panel D: drawdown in money (¥) ---
    ax_d.fill_between(x, dd_money, 0, color="#d62728", alpha=0.5)
    ax_d.set_title("D. Drawdown in Money Terms (loss from peak account value)",
                   fontsize=10, loc="left")
    ax_d.set_ylabel("Drawdown (¥)")
    ax_d.set_xlabel("Trading day")
    ax_d.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"¥{v/1e3:.0f}K"))
    ax_d.grid(alpha=0.3)
    ax_d.annotate(f"Max drawdown: {_money_fmt(max_dd_money)}",
                  xy=(x[int(np.argmin(dd_money))], max_dd_money),
                  fontsize=8, ha="center", va="top",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#fdeeee", ec="#d62728"))

    # --- How-to-read note ---
    note = ("How to read this dashboard:\n"
            "A. Strategy account value (green) vs equal-weight buy-and-hold of the same stock pool (orange dashed), both starting at the initial capital.\n"
            "   Strategy above benchmark = active selection is adding value; below = underperforming buy-and-hold.\n"
            "B. Cumulative alpha = strategy cumulative return % minus benchmark cumulative return % day by day.\n"
            "   Above zero (purple) = outperforming; below zero (red) = underperforming the universe average.\n"
            "C. Sharpe of each independent OOS window. More green bars = the edge repeats across different time periods.\n"
            "D. Drawdown in yuan: how far the account sits below its own historical peak. Shallower/shorter = better risk control.")
    ax_d.figure.text(0.01, -0.03, note, fontsize=7.5, va="top",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9",
                               ec="#cccccc", alpha=0.9))
