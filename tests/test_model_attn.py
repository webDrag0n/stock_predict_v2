"""Smoke test for attention model leg.

Verifies that model_attn.attn_predict can train and predict on a tiny panel.
This is a GPU test if CUDA is available; otherwise CPU (slow).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import config as C
from model_attn import attn_predict


def _make_panel(n_dates: int = 30):
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


def test_attn_predict():
    panel = _make_panel(40)
    dates = panel["date"].unique()
    train = panel[panel["date"] < dates[25]].copy()
    test = panel[panel["date"] >= dates[25]].copy()
    pred = attn_predict(train, test, ["feature_00", "feature_01", "feature_02"],
                        params={"epochs": 2, "d_model": 16, "n_heads": 2,
                                "n_layers": 1, "patience": 1})
    assert len(pred) == len(test)
    assert np.isfinite(pred).all()
    print("attn_predict OK, output range:", pred.min(), pred.max())


if __name__ == "__main__":
    test_attn_predict()
    print("Attention model smoke test passed.")
