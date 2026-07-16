"""事件驱动回测: 每日预测+信号衰减+偏离触发调仓(个股异步)。

核心差异 vs 固定周期回测:
1. **每日预测**: 每个交易日都有最新排序/目标仓位(非每N天才预测一次)
2. **信号衰减**: 目标仓位随持仓天数衰减: target = base_weight × decay_factor^days_held
   (linear decay: 1 - days_held/MAX_HOLD; exp decay: 0.9^days_held)
3. **事件触发**: 当 sum(|current - target|) > TURNOVER_THRESHOLD 时调仓,只调偏离大的股
4. **个股异步**: 不同股票在不同日期调仓(非全组合同步调)

参数:
- MAX_HOLD: 信号衰减到0的天数(类似旧的REBALANCE_DAYS但是软约束)
- DECAY_MODE: 'linear' | 'exp'
- TURNOVER_THRESHOLD: 触发全局调仓的组合偏离阈值(0~2, 0.3≈每3天调一次)
- REBAL_THRESHOLD: 个股级最小调整量(避免micro交易, 如0.005=0.5%仓位)

优势: 及时止损/止盈、只调需要调的股、成本可能更低、每日RMSE可用。
劣势: 参数更多、回测更复杂、可能过拟合短期noise。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from backtest import _metrics


def run_backtest_event_driven(
    df: pd.DataFrame,
    pred_daily: np.ndarray,
    pos_daily: np.ndarray,
    max_hold: int = 20,
    decay_mode: str = "linear",
    turnover_threshold: float = 0.3,
    rebal_threshold: float = 0.005,
    pred_in_return_space: bool = True,
) -> dict:
    """事件驱动回测: 每日预测 + 信号衰减 + 偏离触发调仓。

    Args:
        df: 带 [date, symbol, close, return_1d/pct_chg, ...] 的面板(日频)
        pred_daily: 每日预测(len=len(df), 每天都有新预测)
        pos_daily: 每日目标仓位(策略生成, 如long_topk每天的理想持仓)
        max_hold: 信号衰减到0的最大持有天数
        decay_mode: 'linear' (1 - t/MAX) | 'exp' (0.9^t)
        turnover_threshold: 触发调仓的组合级偏离阈值(sum|current-target|)
        rebal_threshold: 个股最小调整量(|Δw| < 阈值则不调)
        pred_in_return_space: pred是收益空间(可算RMSE)还是排序空间

    Returns:
        dict with curve, metrics, rmse_curve
    """
    df = df.copy()
    df["pred"] = pred_daily
    df["target_base"] = pos_daily   # 策略生成的"理想"仓位(未衰减)

    ret_col = "return_1d" if "return_1d" in df.columns else "pct_chg"
    if ret_col == "pct_chg":
        df["ret"] = df["pct_chg"] / 100.0
    else:
        df["ret"] = df["return_1d"]

    dates = np.sort(df["date"].unique())
    syms = df["symbol"].unique()
    n_dates = len(dates)

    # 宽表索引
    date_to_idx = {d: i for i, d in enumerate(dates)}
    sym_to_idx = {s: i for i, s in enumerate(syms)}
    df["date_idx"] = df["date"].map(date_to_idx)
    df["sym_idx"] = df["symbol"].map(sym_to_idx)

    # 构建宽表: [date, sym] -> ret/target_base/pred
    wide_ret = np.zeros((n_dates, len(syms)), dtype=np.float64)
    wide_target_base = np.zeros((n_dates, len(syms)), dtype=np.float64)
    wide_pred = np.zeros((n_dates, len(syms)), dtype=np.float64)
    wide_close = np.zeros((n_dates, len(syms)), dtype=np.float64)
    for row in df.itertuples(index=False):
        di, si = row.date_idx, row.sym_idx
        wide_ret[di, si] = row.ret
        wide_target_base[di, si] = row.target_base
        wide_pred[di, si] = row.pred
        if "close" in df.columns:
            wide_close[di, si] = row.close

    # 状态: held[sym] = 实际持仓, days_held[sym] = 连续持有天数
    held = np.zeros(len(syms), dtype=np.float64)
    days_held = np.zeros(len(syms), dtype=int)

    # 记录
    gross_daily = np.zeros(n_dates, dtype=np.float64)
    tc_daily = np.zeros(n_dates, dtype=np.float64)
    rebal_flags = np.zeros(n_dates, dtype=bool)   # 是否发生调仓

    commission = C.COMMISSION_RATE
    stamp_duty = C.STAMP_DUTY_RATE

    for t in range(n_dates):
        # 1. 信号衰减: target_decayed = target_base × decay_factor
        if decay_mode == "linear":
            decay_factor = np.clip(1.0 - days_held / max_hold, 0.0, 1.0)
        elif decay_mode == "exp":
            decay_factor = 0.9 ** days_held
        else:
            raise ValueError(f"Unknown decay_mode: {decay_mode}")

        target = wide_target_base[t] * decay_factor

        # 2. 判断是否触发调仓(全局偏离 > 阈值)
        drift = np.sum(np.abs(held - target))
        if drift > turnover_threshold:
            # 3. 只调整 |held - target| > rebal_threshold 的股
            delta = target - held
            mask = np.abs(delta) > rebal_threshold
            held[mask] = target[mask]
            days_held[mask] = 0         # 调仓后重置持有天数
            rebal_flags[t] = True

            # 成本: buy/sell turnover
            buy = np.sum(np.clip(delta[mask], 0, None))
            sell = np.sum(np.clip(-delta[mask], 0, None))
            tc_daily[t] = buy * commission + sell * (commission + stamp_duty)
        else:
            tc_daily[t] = 0.0

        # 4. 当天收益 = held权重(进入今天的) × 今天收益
        gross_daily[t] = np.sum(held * wide_ret[t])

        # 5. 更新持有天数(未调仓的+1)
        days_held += 1

    net_daily = gross_daily - tc_daily
    curve = pd.DataFrame({
        C.DATE_COL: dates,
        C.TIME_COL: 0,
        "gross": gross_daily,
        "tc": tc_daily,
        "net": net_daily,
        "abs_pos": np.abs(held).sum(),  # 最后一天的持仓(简化)
        "rebalanced": rebal_flags,
    })
    curve["equity"] = curve["net"].cumsum()

    # 计算 metrics
    r2 = float("nan")
    metrics = _metrics(curve, weighted_r2_val=r2)
    metrics["n_rebalances"] = int(rebal_flags.sum())
    metrics["rebal_freq_days"] = float(n_dates / rebal_flags.sum()
                                      if rebal_flags.sum() > 0 else float("inf"))

    # RMSE: 每日预测 vs 实现价格(pred是return → realized_price = close*(1+ret))
    rmse_curve = None
    if pred_in_return_space and "close" in df.columns:
        # 简化: 每日RMSE = sqrt(mean( (pred_price - realized_price)^2 ))
        # pred_price = close * (1 + pred), realized = close * (1 + ret)
        price_err = np.zeros(n_dates, dtype=np.float64)
        for t in range(n_dates):
            mask = wide_close[t] > 0
            pred_price = wide_close[t][mask] * (1.0 + wide_pred[t][mask])
            real_price = wide_close[t][mask] * (1.0 + wide_ret[t][mask])
            if mask.sum() > 0:
                price_err[t] = np.sqrt(np.mean((pred_price - real_price) ** 2))
        rmse_curve = pd.DataFrame({C.DATE_COL: dates, "rmse": price_err})
        metrics["mean_price_rmse"] = float(np.mean(price_err[price_err > 0]))
    else:
        metrics["mean_price_rmse"] = float("nan")

    # IC(日度横截面rank相关). _prediction_ic expects C.DATE_COL; the event df
    # carries the human-readable `date` column, so compute IC inline here.
    from scipy import stats
    ic_tmp = pd.DataFrame({"date": df["date"].to_numpy(),
                           "pred": np.asarray(pred_daily, dtype=np.float64),
                           "y": df[C.TARGET].to_numpy(dtype=np.float64)})

    def _one(g):
        if len(g) < 5 or g["pred"].nunique() < 2 or g["y"].nunique() < 2:
            return np.nan
        return stats.spearmanr(g["pred"], g["y"]).correlation

    daily_ic = ic_tmp.groupby("date", group_keys=False).apply(
        _one, include_groups=False).dropna()
    ic_mean = float(daily_ic.mean()) if len(daily_ic) else float("nan")
    icir = (float(daily_ic.mean() / daily_ic.std())
            if len(daily_ic) and daily_ic.std() > 1e-12 else float("nan"))
    metrics["pred_ic"] = ic_mean
    metrics["pred_icir"] = icir

    return {"curve": curve, "metrics": metrics, "net_pnl": net_daily,
            "rmse_curve": rmse_curve}
