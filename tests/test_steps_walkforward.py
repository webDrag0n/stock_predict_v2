"""walkforward 会话模式步骤的合成面板集成测试。

小合成面板（30 股 × 130 交易日，feature_00 带真实信号）跑 wf_run + wf_report，
验证：报告结构（KPI 键 / 4 图 / 窗口表）与 artifact 落盘。wf_data_import 依赖
真实数据源，不在此测试（与 steps_backtest.data_import 同构）。

路径由 tests/conftest.py 统一处理。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config as C
from pipeline.context import StepContext
from pipeline import steps_walkforward as W


def _make_panel(n_dates: int = 130, n_syms: int = 30) -> pd.DataFrame:
    """合成面板：feature_00 带真实截面信号，且含每日重放撮合所需的交易列
    （OHLC/pre_close/pct_chg/limit_flag/amount——价格路径由 return_1d 累乘重建，
    次日 open=当日 close 的 gapless 口径）。"""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
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
                    "open": pre_close[j],
                    "high": max(close[j], pre_close[j]) * 1.001,
                    "low": min(close[j], pre_close[j]) * 0.999,
                    "pre_close": pre_close[j],
                    "pct_chg": ret[j] * 100.0,
                    "limit_flag": 0.0,
                    "amount": 50000.0,
                    "return_1d": ret[j],
                    # feature_00 对前瞻收益有真实截面信号
                    C.TARGET: 0.005 * f0[j] + rng.normal(0, 0.02),
                    "feature_00": f0[j],
                    "feature_01": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def wf_ctx(tmp_path, monkeypatch):
    """缩小滚动窗口 + 单 seed，保证秒级跑完。

    另把 LGBMRegressor 的 n_jobs 钉为 1：本测试容器里 lightgbm 以 n_jobs=-1
    （OpenMP 全核）训练会挂死（basic.py update 处），现有引擎测试均走 n_jobs=1
    路径绕开了它；src/walkforward.py 硬编码 n_jobs=-1，只能在测试侧收口。
    """
    import lightgbm as lgb

    _orig = lgb.LGBMRegressor

    def _patched(**kw):
        kw["n_jobs"] = 1
        return _orig(**kw)

    monkeypatch.setattr(lgb, "LGBMRegressor", _patched)
    monkeypatch.setattr(C, "TRAIN_WINDOW", 40)
    monkeypatch.setattr(C, "TEST_WINDOW", 15)
    monkeypatch.setattr(C, "WF_STEP", 15)
    monkeypatch.setattr(C, "N_SEEDS", 1)
    # canonical 目录（wf_run 额外落盘 outputs/walkforward/）重定向到 tmp，
    # 避免合成面板的测试产物污染真实 CLI 结果
    monkeypatch.setattr(C, "OUTPUT_DIR", str(tmp_path / "outputs"))
    ctx = StepContext(
        session_dir=str(tmp_path),
        params={"select_features": False, "strategy": "long_topk", "capital": 1000000},
    )
    ctx.shared["panel"] = _make_panel()
    ctx.shared["factors"] = ["feature_00", "feature_01"]
    return ctx


def test_wf_run_and_report(wf_ctx, tmp_path):
    rep_run = W.wf_run(wf_ctx)
    assert rep_run["step"] == "wf_run"
    assert rep_run["kpis"], "wf_run 应返回 KPI"
    assert wf_ctx.shared["wf"]["n_windows"] >= 2

    # artifact 落盘
    art = tmp_path / "artifacts"
    assert (art / "oos_curve.parquet").exists()
    assert (art / "window_metrics.parquet").exists()
    assert (art / "oos_bench.parquet").exists()

    # canonical 目录（fixture 已把 C.OUTPUT_DIR 重定向到 tmp_path/outputs）
    canon = tmp_path / "outputs" / "walkforward"
    assert (canon / "oos_curve.parquet").exists()
    assert (canon / "window_metrics.parquet").exists()
    assert (canon / "oos_bench.parquet").exists()
    meta = json.loads((canon / "meta.json").read_text(encoding="utf-8"))
    assert meta["generated_at"]
    assert meta["n_windows"] >= 2
    assert meta["config"]["train_window"] == 40
    assert meta["select_features"] is False
    assert "sharpe" in meta["metrics"]
    # 「为正窗口占比」门禁字段（P1-6）
    for key in ("sharpe_pos_ratio", "predic_pos_ratio"):
        assert key in meta, f"meta.json 缺 {key}"
        assert 0.0 <= meta[key] <= 1.0

    rep = W.wf_report(wf_ctx)
    assert rep["step"] == "wf_report"

    # KPI 键齐全
    labels = [k["label"] for k in rep["kpis"]]
    for want in (
        "Sharpe",
        "Sortino",
        "Calmar",
        "最大回撤",
        "日胜率",
        "总 PnL",
        "成本合计",
        "PredIC",
        "Sharpe>0 窗口占比",
        "PredIC>0 窗口占比",
    ):
        assert any(want in lb for lb in labels), f"缺 KPI: {want}"

    # 4 图：净值 vs 基准(line) / 累计 alpha(line) / 逐窗口 Sharpe(bar) / 回撤(line)
    types = [f["type"] for f in rep["figures"]]
    assert types == ["line", "line", "bar", "line"], types
    fig_a = rep["figures"][0]
    assert len(fig_a["series"]) == 2, "图A 应为 策略+基准 双系列"
    assert fig_a["x_is_date"] is True

    # 窗口表
    assert rep["tables"], "缺逐窗口指标表"
    cols = rep["tables"][0]["columns"]
    for c in ("window", "Sharpe", "PnL", "hit"):
        assert c in cols
    assert rep["tables"][0]["n_total"] >= 2


def test_wf_report_from_artifacts_only(wf_ctx):
    """shared 缺失时 wf_report 可仅凭 artifact 重建（模拟跨进程回读）。"""
    W.wf_run(wf_ctx)
    wf_ctx.shared.pop("wf")
    rep = W.wf_report(wf_ctx)
    types = [f["type"] for f in rep["figures"]]
    assert types == ["line", "line", "bar", "line"], types
    # 重建路径没有 metrics → KPI 容错为占位符而非炸掉
    assert rep["kpis"]


def test_registry_walkforward_wired():
    from pipeline import registry

    assert [s[0] for s in registry.STEPS["walkforward"]] == [
        "wf_data_import",
        "wf_run",
        "wf_report",
    ]
    assert registry.MODE_LABELS["walkforward"] == "滚动Walk-Forward"
    keys = {p["key"] for p in registry.PARAM_SCHEMA["walkforward"]}
    assert {
        "capital",
        "train_window",
        "test_window",
        "wf_step",
        "select_features",
        "strategy",
    } <= keys


def test_wf_run_reports_window_progress(wf_ctx):
    """wf_run 把引擎的 progress_cb 桥接进 ctx.report_progress：逐窗口 current 递增、
    最后一次 current==total，extras 带窗口 OOS 起始日(date)与 pred_ic。"""
    payloads = []
    wf_ctx.on_progress = payloads.append
    W.wf_run(wf_ctx)

    win = [p for p in payloads if p["phase"] == "windows"]
    assert win, "应收到 windows 阶段进度"
    total = win[0]["total"]
    assert total >= 2
    assert [p["current"] for p in win] == list(range(1, total + 1))
    assert all(p["total"] == total for p in win)
    assert win[-1]["current"] == win[-1]["total"]
    for p in win:
        assert p["message"]
        assert "date" in p["extras"], "extras 缺窗口 OOS 起始日"
        assert "pred_ic" in p["extras"], "extras 缺窗口 pred_ic"

    tail = [p for p in payloads if p["phase"] == "final_replay"]
    assert tail, "应收到 final_replay 阶段进度"
    assert tail[-1]["current"] == tail[-1]["total"]


def test_run_walkforward_progress_cb_never_breaks(wf_ctx):
    """回调异常必须被引擎吞掉：progress 永远不能影响主流程。"""
    from walkforward import run_walkforward

    def _boom(**kw):
        raise RuntimeError("progress callback exploded")

    res = run_walkforward(
        wf_ctx.shared["panel"],
        wf_ctx.shared["factors"],
        save=False,
        progress_cb=_boom,
    )
    assert res["n_windows"] >= 2
