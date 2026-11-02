"""seed_backtest 纯计算测试。

seed_backtest 在小合成 OOS 帧（prediction 与 return_1d 正相关）上指标有限且
符号合理（多头 topk 应赚正 PnL）。

路径由 tests/conftest.py 统一处理。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from multiseed_eval import seed_backtest


def _make_oos(n_dates: int = 80, n_syms: int = 30) -> pd.DataFrame:
    """合成 OOS 帧：信号按 20 日块恒定，prediction 与 return_1d 同块相关。

    回测引擎把持仓 shift 一日（pred_t 赚 ret_{t+1}），块内信号恒定使该相关性
    穿透到持仓期，多头 topk 应为正收益。
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    block = np.arange(n_dates) // 20
    rows = []
    for i in range(n_syms):
        sig_blk = rng.normal(0, 0.01, block.max() + 1)
        sig = sig_blk[block]
        ret = sig + rng.normal(0, 0.01, n_dates)
        pred = sig + rng.normal(0, 0.005, n_dates)
        for j, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "symbol": f"S{i:03d}",
                    "return_1d": ret[j],
                    C.TARGET: sig[j] + rng.normal(0, 0.01),
                    "prediction": pred[j],
                }
            )
    return pd.DataFrame(rows)


def _amount_map(oos: pd.DataFrame) -> pd.DataFrame:
    am = oos[["date", "symbol"]].copy()
    am["amount"] = 50000.0
    return am


def test_seed_backtest_synthetic_metrics():
    """正相关信号 → total_pnl/sharpe 为正；全部指标有限。"""
    oos = _make_oos()
    r = seed_backtest(oos, _amount_map(oos))
    for key in (
        "net",
        "bench",
        "strat_cum_pct",
        "bench_cum_pct",
        "alpha_pct",
        "tc_comm",
        "tc_impact",
    ):
        arr = np.asarray(r[key], dtype=np.float64)
        assert np.isfinite(arr).all(), f"{key} 含非有限值"
    for key in (
        "sharpe",
        "calmar",
        "maxdd",
        "alpha_ann",
        "nw_t",
        "beta",
        "total_pnl",
        "cost_comm",
        "cost_impact",
        "hit",
    ):
        v = float(r[key])
        assert np.isfinite(v), f"{key} 非有限: {v}"
    assert r["total_pnl"] > 0, "正相关信号下多头 topk 应为正 PnL"
    assert r["sharpe"] > 0
    assert r["maxdd"] <= 0
    assert 0.0 <= r["hit"] <= 1.0
    assert r["cost_comm"] >= 0 and r["cost_impact"] >= 0
    # alpha 序列 = 策略累计 − 基准累计
    assert len(r["alpha_pct"]) == len(r["net"])
    np.testing.assert_allclose(
        r["alpha_pct"], r["strat_cum_pct"][: len(r["alpha_pct"])] - r["bench_cum_pct"], rtol=1e-9
    )
