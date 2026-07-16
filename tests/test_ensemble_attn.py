"""Integration test for IncrementalEnsemble with attn leg.

Verifies that IncrementalEnsemble can include 'attn' as one of its model legs
and run through update/predict/combine without error on a small panel.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import config as C
from ensemble import IncrementalEnsemble


def _make_panel(n_dates: int = 40):
    dates = pd.date_range("2026-05-01", periods=n_dates, freq="B")
    rows = []
    np.random.seed(42)
    for d in dates:
        for sym in ["600000.SH", "300001.SZ", "000001.SZ"]:
            rows.append({
                "date": d,
                "symbol": sym,
                C.TARGET: np.random.randn() * 0.02,
                "feature_00": np.random.randn(),
                "feature_01": np.random.randn(),
                "feature_02": np.random.randn(),
            })
    return pd.DataFrame(rows)


def test_ensemble_with_attn():
    # Use small attn architecture for speed.
    os.environ["ATTN_DMODEL"] = "16"
    os.environ["ATTN_HEADS"] = "2"
    os.environ["ATTN_LAYERS"] = "1"
    os.environ["ATTN_EPOCHS"] = "2"
    os.environ["ATTN_PATIENCE"] = "1"

    panel = _make_panel(60)
    feat_cols = ["feature_00", "feature_01", "feature_02"]
    ens = IncrementalEnsemble(feat_cols, models=("attn", "ic_linear"),
                              train_window=30, rebuild_days=20, ic_window=5)
    latest = panel["date"].max()
    matured = panel[panel["date"] <= panel["date"].unique()[35]]
    info = ens.update(matured, asof_date=latest, force_full=True)
    print("update info:", info)

    today = panel[panel["date"] == latest].copy()
    preds = ens.predict(today)
    score = ens.combine(preds, today["date"].to_numpy())
    assert len(score) == len(today)
    assert np.isfinite(score).all()
    print("ensemble with attn OK, score range:", score.min(), score.max())


if __name__ == "__main__":
    test_ensemble_with_attn()
    print("IncrementalEnsemble+attn integration test passed.")
