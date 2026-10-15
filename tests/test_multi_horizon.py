"""Unit tests for multi-horizon target fusion.

Verifies that _train_predict_multi_ret_weighted and _train_predict_multi_fusion
run without error on a small panel containing responder_fw_* columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from walkforward import (_train_predict_multi_ret_weighted,
                         _train_predict_multi_fusion)


def _make_panel(n_dates: int = 30):
    dates = pd.date_range("2026-05-01", periods=n_dates, freq="B")
    rows = []
    np.random.seed(42)
    for d in dates:
        for sym in ["600000.SH", "300001.SZ", "000001.SZ"]:
            close = 10.0 + np.random.randn() * 0.5
            rows.append({
                "date": d,
                "symbol": sym,
                "close": close,
                "adj_close": close,
                C.TARGET: np.random.randn() * 0.02,
                "responder_fw_5": np.random.randn() * 0.02,
                "responder_fw_10": np.random.randn() * 0.02,
                "responder_fw_60": np.random.randn() * 0.02,
                C.WEIGHT: 1.0,
                "feature_00": np.random.randn(),
                "feature_01": np.random.randn(),
            })
    return pd.DataFrame(rows)


def test_multi_ret_weighted():
    panel = _make_panel(40)
    dates = panel["date"].unique()
    train = panel[panel["date"] < dates[25]].copy()
    test = panel[panel["date"] >= dates[25]].copy()
    score = _train_predict_multi_ret_weighted(train, test, ["feature_00", "feature_01"], n_seeds=1)
    assert len(score) == len(test)
    assert np.isfinite(score).all()
    print("multi_ret_weighted OK, score range:", score.min(), score.max())


def test_multi_fusion():
    panel = _make_panel(40)
    dates = panel["date"].unique()
    train = panel[panel["date"] < dates[25]].copy()
    test = panel[panel["date"] >= dates[25]].copy()
    score = _train_predict_multi_fusion(train, test, ["feature_00", "feature_01"], n_seeds=1)
    assert len(score) == len(test)
    assert np.isfinite(score).all()
    print("multi_fusion OK, score range:", score.min(), score.max())


if __name__ == "__main__":
    test_multi_ret_weighted()
    test_multi_fusion()
    print("All multi-horizon tests passed.")
