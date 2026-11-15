"""model_gate 门控测试。

覆盖：should_promote 各分支；run_gate 的 bootstrap → refreshed → 配方变更
触发双回放评估（合成小面板，eval_days 缩小）全链路。

路径由 tests/conftest.py 统一处理；standalone 运行：
PYTHONPATH=src:. uv run python tests/test_model_gate.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import config as C
import model_store
from ensemble import IncrementalEnsemble
from model_gate import paired_diff_t, run_gate, should_promote

FEATS = ["feature_00", "feature_01"]
PARAMS = {
    "ensemble_models": "gbm_l2,ic_linear",
    "n_seeds": 1,
    "incr_trees": 5,
    "rebuild_days": 20,
    "train_window": 20,
    "ic_window": 5,
}


def _make_panel(n_dates: int = 80) -> pd.DataFrame:
    """与 test_backtest_synthetic 同构的小面板（含 OHLC/撮合列）。"""
    dates = pd.date_range("2026-05-01", periods=n_dates, freq="B")
    rng = np.random.RandomState(42)
    rows = []
    for d in dates:
        for sym in ["600000.SH", "300001.SZ"]:
            close = 10.0 + rng.randn() * 0.5
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "close": close,
                    "adj_close": close,
                    "open": close * (1 + rng.randn() * 0.01),
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "amount": 10000.0,
                    "pre_close": close * (1 + rng.randn() * 0.01),
                    "pct_chg": rng.randn() * 2.0,
                    "limit_flag": 0,
                    "return_1d": rng.randn() * 0.02,
                    "bb_upper_20": close * 1.05,
                    "bb_lower_20": close * 0.95,
                    "rsi_14": 50.0,
                    C.TARGET: rng.randn() * 0.02,
                    C.WEIGHT: 1.0,
                    "feature_00": rng.randn(),
                    "feature_01": rng.randn(),
                }
            )
    return pd.DataFrame(rows)


def _trained_ens(panel: pd.DataFrame) -> IncrementalEnsemble:
    ens = IncrementalEnsemble(
        FEATS,
        models=("gbm_l2", "ic_linear"),
        n_seeds=1,
        incr_trees=5,
        rebuild_days=20,
        train_window=20,
        ic_window=5,
    )
    ens.update(panel.dropna(subset=[C.TARGET]), force_full=True)
    return ens


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODEL_DIR", str(tmp_path / "models"))
    return C.MODEL_DIR


# ---------------------------------------------------------------------------
def test_should_promote_branches():
    cand = {"sharpe": 1.5, "max_drawdown": -8.0}
    champ = {"sharpe": 1.0, "max_drawdown": -10.0}

    ok, reason = should_promote(cand, champ, sharpe_margin=0.0, dd_tolerance=0.02)
    assert ok and "Sharpe" in reason

    # Sharpe 不足 → 拒
    ok, reason = should_promote(
        {**cand, "sharpe": 0.9}, champ, sharpe_margin=0.0, dd_tolerance=0.02
    )
    assert not ok and "Sharpe" in reason

    # Sharpe 高但未过 margin → 拒
    ok, _ = should_promote({**cand, "sharpe": 1.05}, champ, sharpe_margin=0.1, dd_tolerance=0.02)
    assert not ok

    # Sharpe 高但回撤劣于 champion − 容忍（-15 vs -10，容忍 2pp）→ 拒
    ok, reason = should_promote(
        {**cand, "max_drawdown": -15.0}, champ, sharpe_margin=0.0, dd_tolerance=0.02
    )
    assert not ok and "回撤" in reason

    # 回撤在容忍内（-11.5 vs -10，差 1.5pp ≤ 2pp）→ 过
    ok, _ = should_promote(
        {**cand, "max_drawdown": -11.5}, champ, sharpe_margin=0.0, dd_tolerance=0.02
    )
    assert ok


def test_run_gate_bootstrap_then_refresh(model_dir):
    """无 champion → bootstrap 直接晋升；同配方再来 → refreshed 不评估。"""
    panel = _make_panel()
    ens = _trained_ens(panel)

    g1 = run_gate(panel, FEATS, ens, PARAMS, eval_days=10, progress=False)
    assert g1["action"] == "bootstrap"
    assert g1["use_candidate_for_live"] is True
    assert model_store.load_champion_manifest()["model_id"] == g1["champion_id"]

    g2 = run_gate(panel, FEATS, _trained_ens(panel), PARAMS, eval_days=10, progress=False)
    assert g2["action"] == "refreshed"
    assert g2["use_candidate_for_live"] is True
    assert g2["evaluation"] is None
    # refresh 不产生新 staging 留档
    assert (
        not os.path.isdir(model_store.staging_root())
        or len(os.listdir(model_store.staging_root())) == 1
    )  # 仅 bootstrap 那次


def test_run_gate_config_change_triggers_evaluation(model_dir):
    """配方变更（incr_trees 5→6）→ 双回放评估 + 留档 + evaluation.json；
    结论必须是 promoted/rejected 之一且与指标判定一致。"""
    panel = _make_panel()
    g1 = run_gate(panel, FEATS, _trained_ens(panel), PARAMS, eval_days=10, progress=False)
    assert g1["action"] == "bootstrap"
    old_champion = g1["champion_id"]

    cand_params = {**PARAMS, "incr_trees": 6}
    g2 = run_gate(panel, FEATS, _trained_ens(panel), cand_params, eval_days=10, progress=False)
    assert g2["action"] in ("promoted", "rejected")
    ev = g2["evaluation"]
    assert ev is not None
    for side in ("candidate", "champion"):
        for k in ("sharpe", "max_drawdown", "alpha_ann", "n_days"):
            assert k in ev[side], f"evaluation.{side} 缺 {k}"
            assert np.isfinite(ev[side][k]) or k == "alpha_ann"

    # 判定与 should_promote 一致
    ok_expected, _ = should_promote(
        ev["candidate"],
        ev["champion"],
        cand_ret=ev["candidate_ret"],
        champ_ret=ev["champion_ret"],
    )
    assert (g2["action"] == "promoted") == ok_expected

    # 候选留档 + evaluation.json；收益序列只在内存中传递
    assert len(ev["candidate_ret"]) == len(ev["champion_ret"])
    sdir = model_store.staging_dir_for(g2["candidate_id"])
    assert os.path.isdir(sdir)
    with open(os.path.join(sdir, "evaluation.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["decision"] == g2["action"]
    assert "candidate_ret" not in saved and "champion_ret" not in saved

    # champion 归属与 use_candidate_for_live 一致
    champ_now = model_store.load_champion_manifest()["model_id"]
    if g2["action"] == "promoted":
        assert champ_now == g2["candidate_id"] != old_champion
        assert g2["use_candidate_for_live"] is True
    else:
        assert champ_now == old_champion
        assert g2["use_candidate_for_live"] is False
        assert g2["champion_recipe"] is not None


def test_paired_diff_t_and_promotion_threshold(monkeypatch):
    n = 120
    same_a = np.random.RandomState(0).randn(n) * 0.0005
    same_b = same_a.copy()
    assert paired_diff_t(same_a, same_b, lags=20) == 0.0
    assert paired_diff_t([0.001] * 10, [0.0] * 10, lags=20) == 0.0

    champion = np.random.RandomState(1).randn(n) * 0.0005
    candidate = np.random.RandomState(2).randn(n) * 0.0005 + 0.001
    assert paired_diff_t(candidate, champion, lags=20) > 5.0

    cand_metrics = {"sharpe": 1.5, "max_drawdown": -8.0}
    champ_metrics = {"sharpe": 1.0, "max_drawdown": -10.0}
    monkeypatch.setattr(C, "PROMOTE_MIN_T", 1.0)
    ok, reason = should_promote(cand_metrics, champ_metrics, cand_ret=same_a, champ_ret=same_b)
    assert not ok and "t=" in reason

    ok, reason = should_promote(cand_metrics, champ_metrics, cand_ret=candidate, champ_ret=champion)
    assert ok and "t=" in reason

    monkeypatch.setattr(C, "PROMOTE_MIN_T", 0.0)
    ok, _ = should_promote(cand_metrics, champ_metrics, cand_ret=same_a, champ_ret=same_b)
    assert ok


def test_bootstrap_records_backtest_expectation(monkeypatch, tmp_path):
    import model_gate as gate

    def _fake_replay(*args, **kwargs):
        return {"sharpe": 1.23, "max_drawdown": -4.0, "strat_ret": [0.001] * 10}

    monkeypatch.setattr(gate, "_replay_metrics", _fake_replay)
    panel = _make_panel()
    ens = _trained_ens(panel)
    for mode in ("run_gate", "stage_and_evaluate"):
        monkeypatch.setattr(C, "MODEL_DIR", str(tmp_path / mode))
        if mode == "run_gate":
            gate.run_gate(panel, FEATS, ens, PARAMS, eval_days=10, progress=False)
        else:
            gate.stage_and_evaluate(
                ens,
                panel,
                FEATS,
                PARAMS,
                eval_days=10,
                auto_promote=True,
                progress=False,
            )
        promotion = model_store.load_champion_manifest()["promotion"]
        assert promotion["candidate_metrics"]["sharpe"] == 1.23
        assert "strat_ret" not in promotion["candidate_metrics"]


if __name__ == "__main__":
    import tempfile
    from unittest import mock

    test_should_promote_branches()
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(C, "MODEL_DIR", os.path.join(td, "models")):
            test_run_gate_bootstrap_then_refresh(C.MODEL_DIR)
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(C, "MODEL_DIR", os.path.join(td, "models")):
            test_run_gate_config_change_triggers_evaluation(C.MODEL_DIR)
    print("model_gate tests passed")
