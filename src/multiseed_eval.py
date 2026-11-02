"""多 seed 回测评估（纯计算，无画图）。

原 scripts/plot_multiseed_dashboard.py 的逐 seed OOS 回测逻辑上移至此，
供前端/报表复用：输入单个 seed 的 OOS 帧与 amount 映射，输出逐日
net/bench/cost 序列与指标 dict（含 Newey-West alpha）。

回测口径：唯一回测引擎是每日重放 run_daily_replay。seed OOS 缓存帧只含
date/symbol/prediction/target/return_1d(+amount)，没有 OHLC——这里把
return_1d 累乘重建每股价格路径（gapless：次日 open = 当日 close），使
T+1/整手/成本/参与率上限在重建行情上生效，持仓股每日赚的真实收益与旧
周期调仓引擎一致，但成交/成本口径与实盘同源（trading.py）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _synth_bars_from_returns(oos: pd.DataFrame) -> pd.DataFrame:
    """由 return_1d 重建每股价格路径，补 run_daily_replay 撮合所需的交易列。

    seed OOS 缓存帧不带 OHLC（见 scripts/research/multiseed_robust.run_one_seed
    的落盘列）。重建规则：close_t = 10 × Π(1+r)，次日 open = 当日 close
    （gapless，持仓日收益恰为 return_1d），high/low ±0.1% 包裹开收，
    pre_close = 前收（price_map 的除权检测因此永不误触发），limit_flag=0。
    """
    df = oos.sort_values(["symbol", "date"]).copy()
    r = df["return_1d"].fillna(0.0).to_numpy(dtype=np.float64)
    df["close"] = 10.0 * pd.Series(1.0 + r, index=df.index).groupby(df["symbol"]).cumprod()
    pre = df.groupby("symbol")["close"].shift(1)
    first = pre.isna()
    pre[first] = df.loc[first, "close"].to_numpy() / (1.0 + r[first.to_numpy()])
    df["pre_close"] = pre
    df["open"] = df["pre_close"]
    df["high"] = df[["open", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "close"]].min(axis=1) * 0.999
    df["pct_chg"] = r * 100.0
    df["limit_flag"] = 0.0
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def seed_backtest(oos: pd.DataFrame, amount_map: pd.DataFrame) -> dict:
    """单seed: 合并amount → 每日重放(score_cache 跳训练, 含逐笔成本) → curve序列+metrics+alpha序列。"""
    # 延迟导入：backtest_daily/measure_alpha 是 src 内平铺模块，
    # 调用方（脚本/前端）需先把 src 加入 sys.path；保持本模块可独立 import。
    from backtest_daily import run_daily_replay
    from model_store import fingerprint_of
    from measure_alpha import newey_west_alpha_t

    oos = oos.merge(amount_map, on=["date", "symbol"], how="left")
    panel = _synth_bars_from_returns(oos)
    scores = {
        np.datetime64(d): dict(zip(g["symbol"].to_numpy(), g["prediction"].to_numpy(float)))
        for d, g in oos.groupby("date", sort=True)
    }
    feat_cols = ["prediction"]
    # warmup_days=0：seed OOS 帧本身已是样本外（上游逐窗口训练+embargo），
    # 重放无需再预留训练预热；config 的 EMBARGO_DAYS 闸门仍保留（头部数日不交易）。
    params = {"warmup_days": 0, "progress": False}
    params["score_cache"] = {"scores": scores, "fingerprint": fingerprint_of(feat_cols, params)}
    res = run_daily_replay(panel, feat_cols, params=params)

    daily = res["daily"]
    cash0 = res["meta"]["initial_cash"]
    tv = daily["total_value"].to_numpy(dtype=np.float64)
    net = np.empty(len(tv), dtype=np.float64)
    net[0] = tv[0] / cash0 - 1.0
    if len(tv) > 1:
        net[1:] = tv[1:] / tv[:-1] - 1.0
    # 基准对齐到实际成交日（daily.date 是撮合发生的次日口径）
    bench = (
        oos.groupby("date")["return_1d"]
        .mean()
        .reindex(pd.Index(daily["date"]))
        .fillna(0.0)
        .to_numpy()
    )
    a_d, beta, t_nw, r2 = newey_west_alpha_t(
        net, bench, max(1, min(C.REBALANCE_DAYS, len(net) - 2))
    )
    strat_cum = np.cumprod(1 + net) - 1.0
    bench_cum = np.cumprod(1 + bench) - 1.0

    # 标量指标（占本金分数口径，与旧周期调仓引擎输出同键）
    n = len(net)
    mean = float(net.mean())
    std = float(net.std(ddof=1)) if n > 1 else 0.0
    sharpe = mean / std * np.sqrt(252) if std > 1e-12 else 0.0
    ann_return = mean * 252
    peak = np.maximum.accumulate(tv)
    max_dd = float(((tv - peak) / peak).min())
    calmar = (ann_return / abs(max_dd)) if max_dd < -1e-12 else 0.0
    tc_comm = (daily["commission"] + daily["stamp"]).to_numpy(dtype=np.float64) / cash0
    tc_impact = daily["impact"].to_numpy(dtype=np.float64) / cash0
    return {
        "net": net,
        "bench": bench,
        "strat_cum_pct": strat_cum * 100,
        "bench_cum_pct": bench_cum * 100,
        "alpha_pct": (strat_cum - bench_cum) * 100,
        "tc_comm": tc_comm,
        "tc_impact": tc_impact,
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "maxdd": max_dd,
        "alpha_ann": a_d * 252,
        "nw_t": t_nw,
        "beta": beta,
        "total_pnl": float(tv[-1] / cash0 - 1.0),
        "cost_comm": float(tc_comm.sum()),
        "cost_impact": float(tc_impact.sum()),
        "hit": float((net > 0).mean()),
    }
