"""champion 冻结预测路径测试（live.champion_predict_latest）。

覆盖：冻结预测与训练路径逐位一致；工件回存（weights/ic_hist 保鲜、
train_end_date/promotion 不变）；pred_hist 定权状态跨"运行"连续。

路径由 tests/conftest.py 统一处理；standalone 运行：
PYTHONPATH=src:. uv run python tests/test_champion_predict.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as C
import model_store
from ensemble import IncrementalEnsemble
from live import champion_predict_latest, predict_latest_with_ens

FEATS = ["feature_00", "feature_01"]
PARAMS = {"ensemble_models": "gbm_l2,ic_linear", "n_seeds": 1, "incr_trees": 5,
          "rebuild_days": 10, "train_window": 30, "ic_window": 5}
N_SYMS = 6


def _make_panel(n_dates: int = 70) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    rng = np.random.RandomState(3)
    rows = []
    for d in dates:
        for i in range(N_SYMS):
            f0, f1 = rng.randn(), rng.randn()
            rows.append({
                "date": d, "symbol": f"S{i:03d}",
                "feature_00": f0, "feature_01": f1,
                C.TARGET: 0.01 * f0 + 0.005 * rng.randn(),
                C.WEIGHT: 1.0,
            })
    return pd.DataFrame(rows)


def _trained_ens(panel: pd.DataFrame) -> IncrementalEnsemble:
    ens = IncrementalEnsemble(FEATS, models=("gbm_l2", "ic_linear"),
                              n_seeds=1, incr_trees=5, rebuild_days=10,
                              train_window=30, ic_window=5)
    ens.update(panel.dropna(subset=[C.TARGET]), force_full=True)
    return ens


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODEL_DIR", str(tmp_path / "models"))
    return C.MODEL_DIR


def _make_champion(panel, ens):
    sdir, manifest = model_store.stage_candidate(ens, FEATS, PARAMS)
    model_store.promote(manifest["model_id"],
                        promotion_record={"reason": "t", "decided_at": "d0"})
    return manifest


# ---------------------------------------------------------------------------
def test_frozen_prediction_matches_trained(model_dir):
    """冻结 champion 预测与同一 ens 直接预测逐位一致（无 pred_hist → 等权）。"""
    panel = _make_panel()
    ens = _trained_ens(panel)
    today_ref, info_ref = predict_latest_with_ens(ens, panel, pred_hist=None)
    _make_champion(panel, ens)

    today, info = champion_predict_latest(panel, pred_hist=None)
    assert np.allclose(today["score"].to_numpy(), today_ref["score"].to_numpy())
    assert info["weights"] == info_ref["weights"]
    assert info["action"] == "frozen" and info["champion_id"]


def test_artifact_saved_back_with_fresh_weights(model_dir):
    """回存：weights/ic_hist 随 pred_hist 成熟刷新，train_end_date 与
    promotion 记录不变。"""
    panel = _make_panel()
    ens = _trained_ens(panel)
    _make_champion(panel, ens)
    train_end_before = model_store.load_champion_manifest()["train_end_date"]

    # 构造 pred_hist：两个已成熟预测日（d + FW ≤ latest），含各腿 pred_ 列
    dates = sorted(panel["date"].unique())
    rng = np.random.RandomState(9)
    rows = []
    for d in dates[:2]:
        for s in panel[panel["date"] == d]["symbol"]:
            rows.append({"date": str(pd.Timestamp(d).date()), "symbol": s,
                         "score": rng.randn(),
                         "pred_gbm_l2": rng.randn(),
                         "pred_ic_linear": rng.randn()})
    pred_hist = pd.DataFrame(rows)

    today, info = champion_predict_latest(panel, pred_hist=pred_hist)

    m = model_store.load_champion_manifest()
    assert m["train_end_date"] == train_end_before, "冻结预测不得改 train_end_date"
    assert m["promotion"]["reason"] == "t", "promotion 记录必须保留"

    # 定权状态已持久化：重新加载工件，ic_hist 非空（成熟日已打分入队）
    loaded = model_store.load_champion()
    assert loaded is not None
    ens2, _ = loaded
    n_scored = min(len(ens2._ic_hist[mm]) for mm in ens2.models)
    assert n_scored >= 1, "pred_hist 成熟日打分未持久化到工件"

    # 跨"运行"连续：再次冻结预测不会重复打分（_scored_dates 已恢复）
    today2, info2 = champion_predict_latest(panel, pred_hist=pred_hist)
    loaded2 = model_store.load_champion()
    ens3, _ = loaded2
    assert {k: list(v) for k, v in ens3._ic_hist.items()} == \
        {k: list(v) for k, v in ens2._ic_hist.items()}, "重复打分导致 IC 队列漂移"


def test_champion_predict_requires_champion(model_dir):
    """无 champion → 明确报错（调用方应走 bootstrap 训练路径）。"""
    panel = _make_panel()
    with pytest.raises(ValueError, match="champion"):
        champion_predict_latest(panel, pred_hist=None)


if __name__ == "__main__":
    import os
    import tempfile
    from unittest import mock

    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(C, "MODEL_DIR", os.path.join(td, "models")):
            test_frozen_prediction_matches_trained(C.MODEL_DIR)
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(C, "MODEL_DIR", os.path.join(td, "models")):
            test_artifact_saved_back_with_fresh_weights(C.MODEL_DIR)
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(C, "MODEL_DIR", os.path.join(td, "models")):
            test_champion_predict_requires_champion(C.MODEL_DIR)
    print("champion_predict tests passed")
