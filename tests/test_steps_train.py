"""train 会话模式步骤的合成面板集成测试。

小合成面板（8 股 × 90 交易日，含每日重放撮合所需交易列）跑
data_import（loader 打桩）→ train_candidate → gate_eval 三步，验证：
  · 三步在小数据上跑通，report 结构完整；
  · staging 候选目录含 manifest.json + evaluation.json；
  · auto_promote=False：无论指标判定如何，champion 绝不被改动。

路径由 tests/conftest.py 统一处理。
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
from pipeline.context import StepContext
from pipeline import steps_train as T

FEATS = ["feature_00", "feature_01"]
PARAMS = {
    "ensemble_models": "gbm_l2,ic_linear",
    "n_seeds": 1,
    "incr_trees": 5,
    "rebuild_days": 20,
    "train_window": 20,
    "ic_window": 5,
    "eval_days": 10,
}


def _make_panel(n_dates: int = 90, n_syms: int = 8) -> pd.DataFrame:
    """合成面板：feature_00 带真实截面信号，含门控双回放撮合所需的交易列
    （OHLC/pre_close/pct_chg/limit_flag/amount + 布林/RSI meta）。"""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    rows = []
    for i in range(n_syms):
        f0 = rng.normal(0, 1, n_dates)
        ret = rng.normal(0, 0.01, n_dates)
        close = 10.0 * np.cumprod(1.0 + ret)
        pre_close = np.concatenate([[close[0] / (1.0 + ret[0])], close[:-1]])
        for j, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "symbol": f"S{i:03d}",
                    "close": close[j],
                    "adj_close": close[j],
                    "open": pre_close[j],
                    "high": max(close[j], pre_close[j]) * 1.001,
                    "low": min(close[j], pre_close[j]) * 0.999,
                    "pre_close": pre_close[j],
                    "pct_chg": ret[j] * 100.0,
                    "limit_flag": 0.0,
                    "amount": 50000.0,
                    "return_1d": ret[j],
                    "bb_upper_20": close[j] * 1.05,
                    "bb_lower_20": close[j] * 0.95,
                    "rsi_14": 50.0,
                    # feature_00 对前瞻收益有真实截面信号
                    C.TARGET: 0.005 * f0[j] + rng.normal(0, 0.02),
                    C.WEIGHT: 1.0,
                    "feature_00": f0[j],
                    "feature_01": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def train_ctx(tmp_path, monkeypatch):
    """小配方 + 单 seed + lgb 钉 n_jobs=1（容器内 OpenMP 全核训练会挂死，
    同 test_steps_walkforward 的收口方式）；MODEL_DIR 重定向到 tmp。"""
    import lightgbm as lgb

    _orig = lgb.LGBMRegressor

    def _patched(**kw):
        kw["n_jobs"] = 1
        return _orig(**kw)

    monkeypatch.setattr(lgb, "LGBMRegressor", _patched)
    monkeypatch.setattr(C, "MODEL_DIR", str(tmp_path / "models"))
    ctx = StepContext(session_dir=str(tmp_path), params=dict(PARAMS))
    ctx.shared["panel"] = _make_panel()
    ctx.shared["factors"] = list(FEATS)
    return ctx


def _make_champion(ctx) -> dict:
    """把当前候选直接晋升为 champion（同配方），返回 champion manifest。"""
    ens = ctx.shared["candidate_ens"]
    sdir, manifest = model_store.stage_candidate(ens, FEATS, ctx.shared["cand_recipe"])
    model_store.promote(manifest["model_id"], promotion_record={"reason": "t", "decided_at": "d0"})
    return model_store.load_champion_manifest()


# ---------------------------------------------------------------------------
def test_data_import(train_ctx, monkeypatch):
    """data_import：loader 打桩为合成面板，panel/factors 入 shared，report 完整。"""
    panel = train_ctx.shared["panel"]
    monkeypatch.setattr(T, "load_factor_panel", lambda **kw: panel)
    rep = T.data_import(train_ctx)
    assert rep["step"] == "data_import"
    assert rep["kpis"] and rep["figures"]
    assert train_ctx.shared["panel"] is panel
    assert {"feature_00", "feature_01"} <= set(train_ctx.shared["factors"])


def test_train_candidate(train_ctx):
    """train_candidate：产出训练好的 ensemble + 归一化配方，report 完整且说人话。"""
    rep = T.train_candidate(train_ctx)
    assert rep["step"] == "train_candidate"
    assert rep["kpis"]
    # 用户可见文案：摘要用「子模型」，KPI/摘要不出现裸模型键与「腿」
    assert "个子模型" in rep["summary"] and "腿" not in rep["summary"]
    kpi_txt = " ".join(f"{k['label']} {k['value']} {k['hint']}" for k in rep["kpis"])
    assert "子模型" in kpi_txt and "AI 树模型" in kpi_txt and "因子加权模型" in kpi_txt
    assert "gbm_l2" not in kpi_txt and "ic_linear" not in kpi_txt
    fig = rep["figures"][0]
    assert "gbm_l2" not in " ".join(map(str, fig["x"]))
    ens = train_ctx.shared["candidate_ens"]
    assert isinstance(ens, IncrementalEnsemble)
    assert ens._last_trained_date is not None
    recipe = train_ctx.shared["cand_recipe"]
    for k in (
        "ensemble_models",
        "n_seeds",
        "incr_trees",
        "rebuild_days",
        "train_window",
        "ic_window",
    ):
        assert k in recipe, f"recipe 缺 {k}"
    assert recipe["train_window"] == PARAMS["train_window"]


def test_gate_eval_no_champion(train_ctx):
    """无 champion：候选留档 staging（manifest + evaluation.json），不晋升、不评估。"""
    T.train_candidate(train_ctx)
    rep = T.gate_eval(train_ctx)
    assert rep["step"] == "gate_eval"
    res = train_ctx.shared["gate_eval"]
    assert res["should_promote"] is False and res["promoted"] is False
    assert res["champion_id"] is None and res["candidate"] is None
    # 无 champion 时 champion 依然不存在（绝不被创建/改动）
    assert model_store.load_champion_manifest() is None
    sdir = model_store.staging_dir_for(res["staging_id"])
    assert os.path.isfile(os.path.join(sdir, "manifest.json"))
    assert os.path.isfile(os.path.join(sdir, "evaluation.json"))
    saved = json.loads(open(os.path.join(sdir, "evaluation.json"), encoding="utf-8").read())
    assert saved["decision"] == "staged"


def test_gate_eval_with_champion_never_promotes(train_ctx):
    """有 champion：双回放评估 + staging 留档，champion 原封不动（auto_promote=False）。"""
    T.train_candidate(train_ctx)
    champ_before = _make_champion(train_ctx)

    rep = T.gate_eval(train_ctx)
    assert rep["step"] == "gate_eval"
    res = train_ctx.shared["gate_eval"]
    assert res["promoted"] is False, "train 模式绝不自动晋升"
    assert res["evaluation"] is not None
    for side in ("candidate", "champion"):
        for k in ("sharpe", "max_drawdown", "alpha_ann", "n_days"):
            assert k in res["evaluation"][side], f"evaluation.{side} 缺 {k}"
    # 判定与指标一致（should_promote 只是指标判定，不代表已晋升）
    from model_gate import should_promote

    ok_expected, _ = should_promote(res["candidate"], res["champion"])
    assert res["should_promote"] == ok_expected

    # staging 留档：manifest.json + evaluation.json
    sdir = model_store.staging_dir_for(res["staging_id"])
    assert os.path.isfile(os.path.join(sdir, "manifest.json"))
    assert os.path.isfile(os.path.join(sdir, "evaluation.json"))
    saved = json.loads(open(os.path.join(sdir, "evaluation.json"), encoding="utf-8").read())
    assert saved["decision"] in ("staged", "rejected")
    assert saved["candidate_id"] == res["staging_id"]
    assert saved["champion_id"] == champ_before["model_id"]

    # champion 未被改动：model_id / train_end_date / promotion 记录不变
    champ_after = model_store.load_champion_manifest()
    assert champ_after["model_id"] == champ_before["model_id"]
    assert champ_after["train_end_date"] == champ_before["train_end_date"]
    assert champ_after["promotion"] == champ_before["promotion"]
    assert "refreshed_at" not in champ_after, "champion 不得被原位刷新"

    # report：对比表 + 判定文案
    assert rep["tables"], "缺候选 vs champion 对比表"
    cols = rep["tables"][0]["columns"]
    assert cols == ["指标", "候选", "champion"]
    notes = " ".join(rep["notes"])
    assert "不自动晋升" in notes
    labels = [k["label"] for k in rep["kpis"]]
    assert any("staging" in lb for lb in labels)
    assert any("门控判定" in lb for lb in labels)
