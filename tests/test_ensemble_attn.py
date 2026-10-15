"""Integration test for IncrementalEnsemble with attn leg.

Verifies that IncrementalEnsemble can include 'attn' as one of its model legs
and run through update/predict/combine without error on a small panel.
torch 是可选依赖（uv sync --extra attn），缺失时整个模块 skip。

路径由 tests/conftest.py 统一处理；standalone 运行：
PYTHONPATH=src:. uv run python tests/test_ensemble_attn.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="torch 未安装（可选依赖 attn extra）")

import config as C  # noqa: E402
from ensemble import IncrementalEnsemble  # noqa: E402


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
