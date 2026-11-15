from __future__ import annotations

import importlib

import pytest

import config as C
from live_tracking import compute_report
from pipeline.param_env import env_overrides


def test_multi_horizon_weight_length_validation(monkeypatch):
    monkeypatch.setattr(C, "MODEL_OBJECTIVE", "multi_ret_weighted")
    monkeypatch.setattr(C, "MULTI_HORIZON_WINDOWS", [5, 10, 60])
    monkeypatch.setattr(C, "EMBARGO_DAYS", 60)
    monkeypatch.setattr(C, "MULTI_HORIZON_WEIGHTS", [0.5, 0.5])
    with pytest.raises(AssertionError, match="2.*3|3.*2"):
        C._validate()

    monkeypatch.setattr(C, "MULTI_HORIZON_WEIGHTS", [0.25] * 4)
    C._validate()
    monkeypatch.setattr(C, "MODEL_OBJECTIVE", "fusion")
    monkeypatch.setattr(C, "MULTI_HORIZON_WEIGHTS", [1.0])
    C._validate()


def test_default_multi_horizon_weights_are_equal_length():
    assert len(C.MULTI_HORIZON_WEIGHTS) == len(C.MULTI_HORIZON_WINDOWS) + 1
    assert C.MULTI_HORIZON_WEIGHTS == [0.25] * 4


def test_incremental_seed_has_independent_env(monkeypatch):
    monkeypatch.setenv("N_SEEDS", "3")
    monkeypatch.delenv("ENSEMBLE_N_SEEDS", raising=False)
    try:
        importlib.reload(C)
        assert C.N_SEEDS == 3
        assert C.ENSEMBLE_N_SEEDS == 1
    finally:
        monkeypatch.undo()
        importlib.reload(C)

    env = env_overrides({"n_seeds": 5})
    assert env["N_SEEDS"] == "5"
    assert env["ENSEMBLE_N_SEEDS"] == "5"


def test_status_gate_uses_tracking_trading_day_age(monkeypatch):
    from webapp.pages import pipeline_lab

    monkeypatch.setattr(pipeline_lab.dataio, "champion_manifest", lambda: {"model_id": "m1"})
    monkeypatch.setattr(C, "MAX_MODEL_AGE_DAYS", 60)
    for age, expected in ((10, "ok"), (48, "warn"), (61, "bad")):
        monkeypatch.setattr(
            pipeline_lab.dataio,
            "tracking_report",
            lambda age=age: {"champion": {"age_days": age}},
        )
        level, text = pipeline_lab._status_gate()
        assert level == expected
        assert str(age) in text and "交易日" in text

    monkeypatch.setattr(pipeline_lab.dataio, "tracking_report", lambda: None)
    assert pipeline_lab._status_gate()[0] == "idle"
    monkeypatch.setattr(pipeline_lab.dataio, "champion_manifest", lambda: None)
    assert pipeline_lab._status_gate()[0] == "bad"


def test_tracking_report_passes_manifest_training_metadata():
    manifest = {
        "model_id": "m1",
        "train_end_date": "2026-07-01",
        "trained_at": "2026-08-10T18:00:00",
        "label_maturity_gap_days": 40,
        "promotion": None,
    }
    report = compute_report(manifest, {}, {"has_nav": False}, panel=None)
    assert report["champion"]["trained_at"] == manifest["trained_at"]
    assert report["champion"]["label_maturity_gap_days"] == 40
