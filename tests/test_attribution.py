from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config as C
from attribution import size_exposure, small_cap_benchmark
from walkforward import save_walkforward_result


def _make_panel(n_days=80):
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    symbols = [f"S{i}" for i in range(5)]
    small = rng.normal(0.01, 0.004, n_days)
    others = rng.normal(0.004, 0.003, (n_days, 4))
    rows = []
    for day_i, date in enumerate(dates):
        for sym_i, symbol in enumerate(symbols):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "ln_mv": float(sym_i + 1),
                    "return_1d": small[day_i] if sym_i == 0 else others[day_i, sym_i - 1],
                }
            )
    panel = pd.DataFrame(rows)
    market = panel.groupby("date")["return_1d"].mean()
    small_benchmark = small_cap_benchmark(panel)
    strategy = 0.3 * market + 0.5 * (small_benchmark - market) + 0.0002
    return panel, strategy


def test_small_cap_benchmark_uses_bottom_quantile():
    panel, _ = _make_panel(n_days=5)
    panel["return_1d"] = np.where(panel["symbol"] == "S0", 0.01, 0.0)
    benchmark = small_cap_benchmark(panel, quantile=0.2)
    assert benchmark.to_numpy() == pytest.approx(0.01)


def test_size_exposure_recovers_betas_and_alpha():
    panel, strategy = _make_panel()
    exposure = size_exposure(strategy, panel, window=20)
    assert exposure[["beta_mkt", "beta_size", "alpha_daily"]].iloc[:19].isna().all().all()
    last = exposure.iloc[-1]
    assert last["beta_mkt"] == pytest.approx(0.3, abs=0.05)
    assert last["beta_size"] == pytest.approx(0.5, abs=0.05)
    assert last["alpha_daily"] == pytest.approx(0.0002, abs=1e-8)


def test_size_exposure_is_order_independent():
    panel, strategy = _make_panel()
    expected = size_exposure(strategy, panel, window=20)
    shuffled_panel = panel.sample(frac=1.0, random_state=3)
    shuffled_strategy = strategy.sample(frac=1.0, random_state=4)
    actual = size_exposure(shuffled_strategy, shuffled_panel, window=20)
    pd.testing.assert_frame_equal(expected, actual)


def test_save_walkforward_writes_attribution_without_panel(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "OUTPUT_DIR", str(tmp_path))
    panel, strategy = _make_panel()
    daily = pd.DataFrame(
        {
            "date": strategy.index,
            "decision_date": strategy.index,
            "total_value": 100.0 * (1.0 + strategy).cumprod(),
            "commission": 0.0,
            "stamp": 0.0,
            "impact": 0.0,
            "cost": 0.0,
            "stock_value": 0.0,
        }
    )
    result = {
        "oos_result": {
            "curve": daily[["date"]].copy(),
            "metrics": {"sharpe": 0.5, "ann_return": 0.1, "n_steps": len(daily)},
            "daily": daily,
        },
        "window_metrics": pd.DataFrame(columns=["Sharpe", "PredIC"]),
        "n_windows": 1,
        "oos": panel[["date", "symbol"]],
        "panel_for_attribution": panel,
    }
    assert save_walkforward_result(result) == str(tmp_path / "walkforward")

    meta = json.loads((tmp_path / "walkforward" / "meta.json").read_text(encoding="utf-8"))
    small_ann = float(small_cap_benchmark(panel).mean() * 252.0)
    assert meta["excess_vs_smallcap_ann"] == pytest.approx(0.1 - small_ann)
    exposure = pd.read_parquet(tmp_path / "walkforward" / "size_exposure.parquet")
    assert meta["mean_beta_size"] == pytest.approx(float(exposure["beta_size"].mean()))
    assert exposure["beta_size"].iloc[-1] == pytest.approx(0.5, abs=0.05)
    assert list(exposure.columns) == ["date", "beta_mkt", "beta_size", "alpha_daily"]
    assert len(exposure) == len(daily)
    assert "panel_for_attribution" not in meta
    assert (tmp_path / "walkforward" / "meta.json").stat().st_size < 10_000
