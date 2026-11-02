"""信号评估（src/signal_eval + pipeline/steps_modeleval）的合成面板测试。

小合成面板（30 股 × 130 交易日，feature_00 带真实截面信号）验证：
  - rolling 路径产出 pred_ic_mean / pred_icir / layer_returns / 逐窗口序列等字段；
  - progress_cb 的 current 单调不减；
  - 冻结模型路径（monkeypatch model_store.load_champion 注入 ic_linear 集成）；
  - steps_modeleval.signal_eval / eval_report 的报告结构与 artifact 落盘重建。

路径由 tests/conftest.py 统一处理。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config as C
from pipeline.context import StepContext
from pipeline import steps_modeleval as M
from signal_eval import run_signal_eval


def _make_panel(n_dates: int = 130, n_syms: int = 30) -> pd.DataFrame:
    """合成面板：feature_00 对前瞻收益有真实截面信号（同 test_steps_walkforward）。"""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rows = []
    for i in range(n_syms):
        f0 = rng.normal(0, 1, n_dates)
        ret = rng.normal(0, 0.01, n_dates)
        close = 10.0 * np.cumprod(1.0 + ret)
        for j, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "symbol": f"S{i:03d}",
                    "close": close[j],
                    "return_1d": ret[j],
                    C.TARGET: 0.005 * f0[j] + rng.normal(0, 0.02),
                    "feature_00": f0[j],
                    "feature_01": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def small_panel(monkeypatch):
    """缩小滚动窗口 + 单 seed；LGBMRegressor 钉 n_jobs=1（本容器 n_jobs=-1 会挂死，
    与 test_steps_walkforward 同一收口方式）。"""
    import lightgbm as lgb

    _orig = lgb.LGBMRegressor

    def _patched(**kw):
        kw["n_jobs"] = 1
        return _orig(**kw)

    monkeypatch.setattr(lgb, "LGBMRegressor", _patched)
    monkeypatch.setattr(C, "N_SEEDS", 1)
    return _make_panel()


_FEATS = ["feature_00", "feature_01"]


def test_rolling_signal_eval(small_panel):
    """rolling 路径：字段齐全、PredIC 显著为正（feature_00 有真实信号）、进度单调。"""
    calls = []

    def cb(**kw):
        calls.append(kw)

    res = run_signal_eval(
        small_panel,
        _FEATS,
        model_ref="rolling",
        lookback_days=70,
        train_window=40,
        test_window=15,
        wf_step=15,
        n_quantiles=5,
        progress_cb=cb,
    )
    assert res["model_ref"] == "rolling"
    # 窗口切分：start=40+embargo(20)=60，15 步长、start+15<=130 → 4 窗口覆盖 60 天
    assert res["n_windows"] == 4
    assert res["windows"] is not None and len(res["windows"]) == 4
    assert res["n_dates"] == 60
    for key in ("pred_ic_mean", "pred_icir", "ic_positive_ratio", "long_short_return"):
        assert res[key] is not None and np.isfinite(res[key]), f"{key} 应为有限数"
    assert res["pred_ic_mean"] > 0, "feature_00 带真实信号，PredIC 应为正"
    assert len(res["layer_returns"]) == 5
    assert res["monotonicity"] > 0.5, "真实信号下分层应基本单调"
    # 逐日 IC 序列
    assert len(res["daily_ic"]) == res["n_dates"]
    assert {"date", "rank_ic"} <= set(res["daily_ic"].columns)
    # 逐窗口 PredIC 与整体口径一致
    assert "PredIC" in res["windows"].columns
    # progress_cb：按窗口上报，current 单调不减且不越界
    win_calls = [c for c in calls if c["phase"] == "滚动预测中"]
    assert len(win_calls) == 4
    curs = [c["current"] for c in win_calls]
    assert curs == sorted(curs) and curs[-1] == win_calls[-1]["total"] == 4


def test_frozen_champion_signal_eval(small_panel, monkeypatch):
    """冻结路径：monkeypatch load_champion 注入训练好的 ic_linear 集成。"""
    from ensemble import IncrementalEnsemble
    import model_store

    dates = np.sort(small_panel["date"].unique())
    train = small_panel[small_panel["date"] < dates[-30]].copy()
    train[C.WEIGHT] = np.float32(1.0)
    ens = IncrementalEnsemble(_FEATS, models=("ic_linear",), n_seeds=1)
    ens.update(train, force_full=True)
    manifest = {"model_id": "test_champ", "train_end_date": str(pd.Timestamp(dates[-31]).date())}

    monkeypatch.setattr(model_store, "load_champion", lambda: (ens, manifest))

    calls = []
    res = run_signal_eval(
        small_panel,
        _FEATS,
        model_ref="champion",
        lookback_days=30,
        progress_cb=lambda **kw: calls.append(kw),
    )
    assert res["model_ref"] == "champion"
    assert res["n_windows"] is None and res["windows"] is None
    assert res["n_dates"] == 30
    assert res["model_info"]["model_id"] == "test_champ"
    assert np.isfinite(res["pred_ic_mean"]) and res["pred_ic_mean"] > 0
    assert len(res["layer_returns"]) == 5
    # progress_cb：按日上报，current 单调不减
    day_calls = [c for c in calls if c["phase"] == "逐日打分"]
    curs = [c["current"] for c in day_calls]
    assert curs == sorted(curs) and curs[-1] == 30


def test_modeleval_steps(small_panel, tmp_path):
    """signal_eval → eval_report：报告结构 + artifact 落盘 + 仅凭 artifact 重建。"""
    ctx = StepContext(
        session_dir=str(tmp_path),
        params={
            "model_ref": "rolling",
            "lookback_days": 70,
            "train_window": 40,
            "test_window": 15,
            "wf_step": 15,
            "n_quantiles": 5,
        },
    )
    ctx.shared["panel"] = small_panel
    ctx.shared["factors"] = _FEATS

    rep_run = M.signal_eval(ctx)
    assert rep_run["step"] == "signal_eval"
    assert rep_run["kpis"]
    assert ctx.shared["signal_eval"]["n_windows"] == 4

    art = tmp_path / "artifacts"
    assert (art / "daily_ic.parquet").exists()
    assert (art / "windows.parquet").exists()
    meta = json.loads((art / "eval_summary.json").read_text(encoding="utf-8"))
    assert meta["model_ref"] == "rolling"
    assert np.isfinite(meta["pred_ic_mean"])

    rep = M.eval_report(ctx)
    assert rep["step"] == "eval_report"
    labels = [k["label"] for k in rep["kpis"]]
    for want in ("PredIC", "ICIR", "IC>0 占比", "多空收益", "分层单调性", "评估天数"):
        assert any(want in lb for lb in labels), f"缺 KPI: {want}"
    # 图：逐窗口 PredIC(bar) + 逐日 RankIC(line) + 分位收益(bar)
    types = [f["type"] for f in rep["figures"]]
    assert types == ["bar", "line", "bar"], types
    assert rep["tables"], "缺分位/窗口表"

    # shared 缺失时仅凭 artifact 重建（跨进程回读）
    ctx.shared.pop("signal_eval")
    rep2 = M.eval_report(ctx)
    assert [f["type"] for f in rep2["figures"]] == ["bar", "line", "bar"]
    assert rep2["kpis"]
