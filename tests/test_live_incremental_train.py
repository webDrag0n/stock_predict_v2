"""train_candidate_ensemble 增量续训分支 + _fit_full early stopping 测试。

覆盖：
(a) 无 champion → 全量重建（force_full=True）；
(b) champion 存在且配方指纹一致 → 加载工件增量续训（force_full=False，
    只在新成熟日续树，训练截止日前移）；
(c) 配方变化 → 回退全量重建；
(d) _fit_full early stopping：树数 ≤ FULL_TREES 上限（噪声数据提前停），
    预测/续训/state_dict 往返均按 booster 实际树数工作（无固定树数假设）。

路径由 tests/conftest.py 统一处理；standalone 运行：
PYTHONPATH=src:. uv run python tests/test_live_incremental_train.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as C
import model_store
from ensemble import IncrementalEnsemble
from live import train_candidate_ensemble

FEATS = ["feature_00", "feature_01"]
PARAMS = {
    "ensemble_models": "gbm_l2,ic_linear",
    "n_seeds": 1,
    "incr_trees": 5,
    "rebuild_days": 10,
    "train_window": 30,
    "ic_window": 5,
}
N_SYMS = 20


def _make_panel(n_dates: int = 80, pure_noise: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    rng = np.random.RandomState(3)
    rows = []
    for d in dates:
        for i in range(N_SYMS):
            f0, f1 = rng.randn(), rng.randn()
            tgt = rng.randn() * 0.02 if pure_noise else 0.01 * f0 + 0.005 * rng.randn()
            rows.append(
                {
                    "date": d,
                    "symbol": f"S{i:03d}",
                    "feature_00": f0,
                    "feature_01": f1,
                    C.TARGET: tgt,
                    C.WEIGHT: 1.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODEL_DIR", str(tmp_path / "models"))
    return C.MODEL_DIR


@pytest.fixture
def update_spy(monkeypatch):
    """记录 train_candidate_ensemble 内部 IncrementalEnsemble.update 的 force_full。"""
    calls = []
    orig = IncrementalEnsemble.update

    def _spy(self, train_frame, asof_date=None, force_full=False):
        calls.append(force_full)
        return orig(self, train_frame, asof_date=asof_date, force_full=force_full)

    monkeypatch.setattr(IncrementalEnsemble, "update", _spy)
    return calls


def _promote_as_champion(ens):
    _, manifest = model_store.stage_candidate(ens, FEATS, PARAMS)
    model_store.promote(manifest["model_id"], promotion_record={"reason": "t", "decided_at": "d0"})
    return manifest


# ---------------------------------------------------------------------------
def test_no_champion_full_rebuild(model_dir, update_spy):
    """(a) 无 champion → 全量重建。"""
    panel = _make_panel()
    ens, recipe = train_candidate_ensemble(panel, FEATS, params=PARAMS)
    assert update_spy == [True]
    assert recipe["train_mode"] == "full"
    assert ens._last_trained_date is not None
    assert len(ens._boosters["gbm_l2"]) == 1


def test_same_recipe_incremental_continue(model_dir, update_spy):
    """(b) champion 配方指纹一致 → 续训：只在新成熟日续树，训练截止日前移。"""
    panel = _make_panel()
    ens1, _ = train_candidate_ensemble(panel, FEATS, params=PARAMS)
    _promote_as_champion(ens1)
    train_end_1 = ens1._last_trained_date
    trees_1 = ens1._boosters["gbm_l2"][0].num_trees()

    # 面板向后延 5 个交易日 → 5 个新成熟标签日（同日用同一 RNG 序列重放生成）
    panel2 = _make_panel(n_dates=85)
    ens2, recipe2 = train_candidate_ensemble(panel2, FEATS, params=PARAMS)
    assert update_spy == [True, False]
    assert recipe2["train_mode"] == "incremental"
    assert ens2._last_trained_date > train_end_1
    # 续训树数 = INCR_TREES × 新成熟天数（5）为上界；玩具数据行数少，
    # LightGBM 可能提前报 "no further splits" 而少于上界，断言区间即可
    trees_2 = ens2._boosters["gbm_l2"][0].num_trees()
    assert trees_1 < trees_2 <= trees_1 + PARAMS["incr_trees"] * 5
    # 续训后可正常预测
    latest = panel2[panel2["date"] == panel2["date"].max()]
    preds = ens2.predict(latest)
    assert np.isfinite(preds["gbm_l2"]).all()


def test_recipe_change_falls_back_to_full(model_dir, update_spy):
    """(c) 配方变化（incr_trees 5→6）→ 回退全量重建。"""
    panel = _make_panel()
    ens1, _ = train_candidate_ensemble(panel, FEATS, params=PARAMS)
    _promote_as_champion(ens1)

    _, recipe2 = train_candidate_ensemble(panel, FEATS, params={**PARAMS, "incr_trees": 6})
    assert update_spy == [True, True]
    assert recipe2["train_mode"] == "full"


def test_feat_cols_change_falls_back_to_full(model_dir, update_spy):
    """(c') 特征集变化（配方指纹含 feat_cols）→ 回退全量重建。"""
    panel = _make_panel()
    ens1, _ = train_candidate_ensemble(panel, FEATS, params=PARAMS)
    _promote_as_champion(ens1)

    _, recipe2 = train_candidate_ensemble(panel, FEATS[:1], params=PARAMS)
    assert update_spy == [True, True]
    assert recipe2["train_mode"] == "full"


def test_fit_full_early_stopping(model_dir, monkeypatch):
    """(d) 纯噪声标签上 early stopping 提前停（树数 < FULL_TREES 上限）；
    state_dict 往返预测逐位一致（ES 裁剪到 best_iteration，序列化不丢语义）；
    续训接在实际树数之上正常增长（无固定树数假设）。"""
    monkeypatch.setattr(C, "FULL_TREES", 200)
    noise = _make_panel(pure_noise=True)
    ens = IncrementalEnsemble(
        FEATS,
        models=("gbm_l2",),
        n_seeds=1,
        incr_trees=5,
        rebuild_days=10,
        train_window=60,
        ic_window=5,
    )
    ens.update(noise.dropna(subset=[C.TARGET]), force_full=True)
    bst = ens._boosters["gbm_l2"][0]
    assert 0 < bst.num_trees() < 200, "纯噪声数据上 early stopping 未提前停"

    today = noise[noise["date"] == noise["date"].max()]
    p_before = ens.predict(today)["gbm_l2"]
    ens2 = IncrementalEnsemble.from_state_dict(ens.state_dict(), FEATS)
    assert np.allclose(p_before, ens2.predict(today)["gbm_l2"])

    # 续训：含信号面板延 5 个新成熟日，树数增长（上界 INCR_TREES × 5）
    panel2 = _make_panel(n_dates=85)
    ens2.update(panel2.dropna(subset=[C.TARGET]), force_full=False)
    n2 = ens2._boosters["gbm_l2"][0].num_trees()
    assert bst.num_trees() < n2 <= bst.num_trees() + 5 * 5


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
