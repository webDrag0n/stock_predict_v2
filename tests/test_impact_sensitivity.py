from __future__ import annotations

import pandas as pd

from scripts.research.impact_sensitivity import _clip_for_windows, sweep_impact


def test_sweep_impact_returns_sorted_metrics():
    calls = []

    def replay_fn(k):
        calls.append(k)
        daily = pd.DataFrame(
            {
                "date": pd.bdate_range("2026-01-01", periods=3),
                "total_value": [1_000_000.0, 1_001_000.0, 1_002_000.0 - k * 1000.0],
                "stock_value": 0.0,
                "commission": 10.0,
                "stamp": 0.0,
                "impact": 100.0 * k,
                "cost": 10.0 + 100.0 * k,
            }
        )
        return {"daily": daily, "meta": {"initial_cash": 1_000_000.0}}

    result = sweep_impact(replay_fn, [0.2, 0.0, 0.1])
    assert calls == [0.0, 0.1, 0.2]
    assert len(result) == 3
    assert list(result.columns) == ["k", "sharpe", "ann_return", "max_drawdown", "total_cost"]
    assert result["k"].tolist() == [0.0, 0.1, 0.2]
    assert result["total_cost"].is_monotonic_increasing


def test_clip_for_windows_limits_history(monkeypatch):
    import config as C

    monkeypatch.setattr(C, "TRAIN_WINDOW", 10)
    monkeypatch.setattr(C, "TEST_WINDOW", 5)
    monkeypatch.setattr(C, "WF_STEP", 5)
    monkeypatch.setattr(C, "EMBARGO_DAYS", 2)
    panel = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=40), "symbol": "A"})
    clipped = _clip_for_windows(panel, 2)
    assert clipped["date"].nunique() == 22
    assert clipped["date"].max() == sorted(panel["date"].unique())[21]
    assert _clip_for_windows(panel, None) is panel
