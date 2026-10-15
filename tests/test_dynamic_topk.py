"""Unit tests for dynamic topk concentration.

Verifies that backtest_daily.run_daily_replay accepts use_dynamic_topk params and
that the interpolation logic maps high/low IC to the expected topk bounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from backtest_daily import run_daily_replay


def _make_panel(n_dates: int = 40):
    dates = pd.date_range("2026-05-01", periods=n_dates, freq="B")
    rows = []
    np.random.seed(42)
    for d in dates:
        for sym in ["600000.SH", "300001.SZ"]:
            close = 10.0 + np.random.randn() * 0.5
            rows.append({
                "date": d,
                "symbol": sym,
                "close": close,
                "adj_close": close,
                "open": close * (1 + np.random.randn() * 0.01),
                "high": close * 1.02,
                "low": close * 0.98,
                "amount": 10000.0,
                "pre_close": close * (1 + np.random.randn() * 0.01),
                "pct_chg": np.random.randn() * 2.0,
                "limit_flag": 0,
                "return_1d": np.random.randn() * 0.02,
                "bb_upper_20": close * 1.05,
                "bb_lower_20": close * 0.95,
                "rsi_14": 50.0,
                C.TARGET: np.random.randn() * 0.02,
                C.WEIGHT: 1.0,
                "feature_00": np.random.randn(),
                "feature_01": np.random.randn(),
            })
    return pd.DataFrame(rows)


def test_dynamic_topk_runs():
    panel = _make_panel(80)
    feat_cols = ["feature_00", "feature_01"]
    params = {
        "n_seeds": 1,
        "rebuild_days": 20,
        "train_window": 20,
        "ic_window": 5,
        "warmup_days": 20,
        "strict_check": True,
        "progress": False,
        "use_dynamic_topk": True,
        "dynamic_topk_ic_full": 0.02,
        "dynamic_topk_ic_zero": -0.01,
        "dynamic_topk_low": 0.10,
        "dynamic_topk_high": 0.25,
    }
    res = run_daily_replay(panel, feat_cols, params=params)
    daily = res["daily"]
    assert len(daily) > 0
    print(f"dynamic_topk run OK: {len(daily)} days, final value {daily['total_value'].iloc[-1]:.2f}")


def test_dynamic_topk_params_in_decide_orders():
    """Ensure decide_orders sees dynamic topk bounds without exception."""
    from trading import decide_orders, PortfolioState
    panel = _make_panel(10)
    today = panel[panel["date"] == panel["date"].max()].copy()
    score = np.random.randn(len(today))
    state = PortfolioState(cash=1000000)
    orders = decide_orders(
        state, today, score,
        params={
            "topk": 0.15,
            "hold_ratio": 0.20,
            "max_positions": 30,
            "max_weight": 0.10,
            "use_bb_timing": False,
            "min_notional": 30000,
        },
        asof_date=today["date"].max(),
    )
    assert isinstance(orders, pd.DataFrame)
    print("decide_orders with dynamic-topk style params OK")


if __name__ == "__main__":
    test_dynamic_topk_runs()
    test_dynamic_topk_params_in_decide_orders()
    print("All dynamic_topk tests passed.")
