"""model_store + ensemble 序列化测试。

覆盖：state_dict 往返预测一致、staging 落盘、promote/archive 轮换、
champion 刷新保留晋升记录、损坏备份、指纹对 ensemble_models 形态稳定。

路径由 tests/conftest.py 统一处理；standalone 运行：
PYTHONPATH=src:. uv run python tests/test_model_store.py
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pytest

import config as C
import model_store
from ensemble import IncrementalEnsemble

FEATS = ["feature_00", "feature_01"]


def _make_frame(n_dates: int = 30, n_syms: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2026-05-01", periods=n_dates, freq="B")
    rng = np.random.RandomState(7)
    rows = []
    for d in dates:
        for i in range(n_syms):
            rows.append({
                "date": d, "symbol": f"S{i:03d}",
                "feature_00": rng.randn(), "feature_01": rng.randn(),
                C.TARGET: rng.randn() * 0.02, C.WEIGHT: 1.0,
            })
    return pd.DataFrame(rows)


def _trained_ens(frame: pd.DataFrame) -> IncrementalEnsemble:
    ens = IncrementalEnsemble(FEATS, models=("gbm_l2", "ic_linear"),
                              n_seeds=1, incr_trees=5, rebuild_days=10,
                              train_window=20, ic_window=5)
    ens.update(frame.dropna(subset=[C.TARGET]), force_full=True)
    return ens


PARAMS = {"ensemble_models": "gbm_l2,ic_linear", "n_seeds": 1, "incr_trees": 5,
          "rebuild_days": 10, "train_window": 20, "ic_window": 5}


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODEL_DIR", str(tmp_path / "models"))
    return C.MODEL_DIR


# ---------------------------------------------------------------------------
def test_state_dict_roundtrip_predictions_identical():
    """state_dict → pickle → from_state_dict 后，各腿预测与权重逐位一致。"""
    frame = _make_frame()
    ens = _trained_ens(frame)
    test = frame[frame["date"] == frame["date"].max()]

    preds = ens.predict(test)
    score = ens.combine(preds, test["date"].to_numpy())

    state = pickle.loads(pickle.dumps(ens.state_dict()))
    ens2 = IncrementalEnsemble.from_state_dict(state, FEATS)
    preds2 = ens2.predict(test)
    score2 = ens2.combine(preds2, test["date"].to_numpy())

    assert set(preds2) == set(preds)
    for m in preds:
        assert np.allclose(preds[m], preds2[m]), f"{m} 腿预测不一致"
    assert np.allclose(score, score2)
    assert ens2.weights == ens.weights
    assert ens2._last_trained_date == ens._last_trained_date


def test_fingerprint_stable_across_models_form():
    """ensemble_models 传 str / list / 缺省（走 config）→ 同一配方同一指纹。"""
    fp_str = model_store.fingerprint_of(FEATS, dict(PARAMS))
    fp_list = model_store.fingerprint_of(
        FEATS, {**PARAMS, "ensemble_models": ["gbm_l2", "ic_linear"]})
    fp_default = model_store.fingerprint_of(
        FEATS, {k: v for k, v in PARAMS.items() if k != "ensemble_models"})
    assert fp_str == fp_list
    # 缺省回退 C.ENSEMBLE_MODELS；仅当 config 默认恰为同配方时相等
    if C.ENSEMBLE_MODELS.replace(" ", "") == "gbm_l2,ic_linear":
        assert fp_str == fp_default
    # 改一个超参 → 指纹必须变
    fp_other = model_store.fingerprint_of(FEATS, {**PARAMS, "incr_trees": 6})
    assert fp_other != fp_str


def test_stage_promote_load_champion(model_dir):
    """候选落盘 staging → promote → champion 可读且预测一致；二次 promote 归档旧冠军。"""
    frame = _make_frame()
    ens = _trained_ens(frame)
    test = frame[frame["date"] == frame["date"].max()]
    preds = ens.predict(test)

    sdir, manifest = model_store.stage_candidate(ens, FEATS, PARAMS)
    assert os.path.exists(os.path.join(sdir, "manifest.json"))
    assert os.path.exists(os.path.join(sdir, "ensemble_state.pkl"))
    assert manifest["fingerprint"] == model_store.fingerprint_of(FEATS, PARAMS)

    model_store.promote(manifest["model_id"],
                        promotion_record={"reason": "t", "decided_at": "now"})
    loaded = model_store.load_champion()
    assert loaded is not None
    champ_ens, champ_manifest = loaded
    assert champ_manifest["promotion"]["reason"] == "t"
    preds2 = champ_ens.predict(test)
    for m in preds:
        assert np.allclose(preds[m], preds2[m])

    # 二次 promote：旧 champion 进 archive
    sdir2, manifest2 = model_store.stage_candidate(ens, FEATS, PARAMS)
    model_store.promote(manifest2["model_id"])
    archived = os.listdir(model_store.archive_root())
    assert len(archived) == 1
    assert model_store.load_champion()[1]["model_id"] == manifest2["model_id"]


def test_refresh_champion_keeps_promotion_record(model_dir):
    """同配方滚动重训：refresh 原位刷新工件，promotion 记录不丢。"""
    frame = _make_frame()
    ens = _trained_ens(frame)
    sdir, manifest = model_store.stage_candidate(ens, FEATS, PARAMS)
    model_store.promote(manifest["model_id"],
                        promotion_record={"reason": "t", "decided_at": "d0"})

    ens2 = _trained_ens(_make_frame(n_dates=31))
    model_store.refresh_champion(ens2, FEATS, PARAMS)
    m = model_store.load_champion_manifest()
    assert m["promotion"]["reason"] == "t"
    assert m["fingerprint"] == model_store.fingerprint_of(FEATS, PARAMS)
    assert m["refreshed_at"]


def test_load_champion_missing_and_corrupt(model_dir):
    """无 champion → None；manifest 损坏 → 备份为 .corrupt-* 并返回 None。"""
    assert model_store.load_champion() is None
    assert model_store.load_champion_manifest() is None

    frame = _make_frame()
    ens = _trained_ens(frame)
    sdir, manifest = model_store.stage_candidate(ens, FEATS, PARAMS)
    model_store.promote(manifest["model_id"])

    mpath = os.path.join(model_store.champion_dir(), "manifest.json")
    with open(mpath, "w") as f:
        f.write("{not json")
    assert model_store.load_champion() is None
    corrupts = [d for d in os.listdir(model_dir) if ".corrupt-" in d]
    assert corrupts, "损坏的 champion 未备份"


def test_archive_prune_keeps_latest(model_dir, monkeypatch):
    """archive 只保留最近 ARCHIVE_KEEP 个。"""
    monkeypatch.setattr(C, "ARCHIVE_KEEP", 1)
    frame = _make_frame()
    ens = _trained_ens(frame)
    last_id = None
    for _ in range(3):
        _, manifest = model_store.stage_candidate(ens, FEATS, PARAMS)
        model_store.promote(manifest["model_id"])
        last_id = manifest["model_id"]
    archived = os.listdir(model_store.archive_root())
    assert len(archived) <= 1
    assert model_store.load_champion()[1]["model_id"] == last_id


if __name__ == "__main__":
    import tempfile
    from unittest import mock

    test_state_dict_roundtrip_predictions_identical()
    test_fingerprint_stable_across_models_form()
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(C, "MODEL_DIR", os.path.join(td, "models")):
            test_stage_promote_load_champion(C.MODEL_DIR)
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(C, "MODEL_DIR", os.path.join(td, "models")):
            test_refresh_champion_keeps_promotion_record(C.MODEL_DIR)
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(C, "MODEL_DIR", os.path.join(td, "models")):
            test_load_champion_missing_and_corrupt(C.MODEL_DIR)
    with tempfile.TemporaryDirectory() as td:
        mp = pytest.MonkeyPatch()
        with mock.patch.object(C, "MODEL_DIR", os.path.join(td, "models")):
            test_archive_prune_keeps_latest(C.MODEL_DIR, mp)
        mp.undo()
    print("model_store tests passed")
