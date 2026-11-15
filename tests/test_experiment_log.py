from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.stats import norm

from experiment_log import count_trials, deflated_sharpe, log_trial


def test_trials_are_deduplicated_by_config_hash(tmp_path):
    path = tmp_path / "trials.jsonl"
    config_a = {"train_window": 500, "topk": 0.15}
    log_trial("walkforward", config_a, {"sharpe": 1.0}, path=path)
    log_trial("walkforward", config_a, {"sharpe": 1.1}, path=path)
    log_trial("walkforward", {"topk": 0.2}, {"sharpe": 0.8}, path=path)
    log_trial("other", config_a, {"sharpe": 0.5}, path=path)

    assert count_trials(path=path) == 2
    assert count_trials("walkforward", path=path) == 2
    assert count_trials("other", path=path) == 1
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4
    assert lines[0]["config_hash"] == lines[1]["config_hash"]
    assert len(lines[0]["config_hash"]) == 12
    assert {"ts", "kind", "config", "metrics", "config_hash"} <= set(lines[0])


def test_deflated_sharpe_reference_values():
    assert deflated_sharpe(0.0, n_obs=500, n_trials=1) == pytest.approx(0.5, abs=1e-12)

    by_trials = [deflated_sharpe(0.1, 500, n) for n in (1, 10, 100, 1000)]
    assert all(a > b for a, b in zip(by_trials, by_trials[1:]))

    by_sharpe = [deflated_sharpe(sr, 500, 10) for sr in (0.05, 0.1, 0.15)]
    assert all(a < b for a, b in zip(by_sharpe, by_sharpe[1:]))

    sr = 0.1
    n_obs = 500
    den = math.sqrt(1 + (3 - 1) / 4 * sr**2)
    expected = norm.cdf(sr * math.sqrt(n_obs - 1) / den)
    assert deflated_sharpe(sr, n_obs, 1) == pytest.approx(expected, abs=1e-12)
    assert np.isnan(deflated_sharpe(2.0, 10, 1, skew=2.0, kurt=3.0))


def _synthetic_oos():
    import pandas as pd

    dates = pd.bdate_range("2026-01-01", periods=6)
    daily = pd.DataFrame(
        {
            "date": dates,
            "decision_date": dates,
            "total_value": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "commission": 0.0,
            "stamp": 0.0,
            "impact": 0.0,
            "cost": 0.0,
            "stock_value": 0.0,
        }
    )
    oos = pd.DataFrame(
        {
            "date": np.repeat(dates, 5),
            "symbol": ["A", "B", "C", "D", "E"] * len(dates),
            "prediction": [5.0, 4.0, 3.0, 2.0, 1.0] * len(dates),
            "responder_6": [0.05, 0.04, 0.03, 0.02, 0.01] * len(dates),
        }
    )
    return daily, oos


def test_holdout_partition_covers_all_oos_days():
    from walkforward import _holdout_summary

    daily, oos = _synthetic_oos()
    before, holdout = _holdout_summary(daily, oos, 100.0, str(daily["date"].iloc[2].date()))
    assert before["n_days"] == 2
    assert holdout["n_days"] == 4
    assert before["n_days"] + holdout["n_days"] == len(daily)
    assert holdout["pred_ic"] == pytest.approx(1.0)


def test_walkforward_meta_contains_trial_and_holdout_metrics(monkeypatch, tmp_path):
    import config as C
    from walkforward import save_walkforward_result

    monkeypatch.setattr(C, "OUTPUT_DIR", str(tmp_path))
    daily, oos = _synthetic_oos()
    log_trial("walkforward", {"synthetic": 1}, {"sharpe": 1.0})
    result = {
        "oos_result": {
            "curve": daily[["date"]].copy(),
            "metrics": {"sharpe": 1.0, "n_steps": len(daily)},
            "daily": daily,
        },
        "window_metrics": __import__("pandas").DataFrame(columns=["Sharpe", "PredIC"]),
        "n_windows": 1,
        "oos": oos,
        "pre_holdout": {"sharpe": 0.5, "pred_ic": 0.1, "n_days": 2},
        "holdout": {"sharpe": 1.5, "pred_ic": 0.2, "n_days": 4},
    }
    outdir = save_walkforward_result(result)
    meta = json.loads((tmp_path / "walkforward" / "meta.json").read_text(encoding="utf-8"))
    assert outdir == str(tmp_path / "walkforward")
    assert meta["n_trials"] == 1
    assert 0.0 <= meta["deflated_sharpe"] <= 1.0
    assert meta["holdout"] == result["holdout"]
    assert meta["pre_holdout"] == result["pre_holdout"]
