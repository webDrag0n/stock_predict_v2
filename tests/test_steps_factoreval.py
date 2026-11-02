"""factoreval 会话模式步骤的单元/集成测试。

- merge_factor_eval_csv：tmp_path 造 csv，验证 subset 合并写语义
  （替换本次因子行、保留其余行、按 |IC_t| 重排）。
- eval_factors/eval_report：小合成面板（60 交易日 × 30 股，fac_signal 带
  真实截面信号）跑 smoke，C.OUTPUT_DIR monkeypatch 到 tmp_path，不污染真实产物。
- data_import 依赖真实数据源，只验证日期参数解析（load_factor_panel 打桩）。

路径由 tests/conftest.py 统一处理。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config as C
from pipeline.context import StepContext
from pipeline import steps_factoreval as S


def _make_panel(n_dates: int = 60, n_syms: int = 30) -> pd.DataFrame:
    """合成面板：fac_signal 对前瞻收益有真实截面信号，fac_noise 为纯噪声。"""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    rows = []
    for d in dates:
        sig = rng.normal(0, 1, n_syms)
        rows.append(
            pd.DataFrame(
                {
                    "date": d,
                    "symbol": [f"S{i:03d}" for i in range(n_syms)],
                    "fac_signal": sig,
                    "fac_noise": rng.normal(0, 1, n_syms),
                    C.TARGET: 0.02 * sig + rng.normal(0, 0.02, n_syms),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _eval_row(factor: str, ic_t: float, rank_icir: float) -> dict:
    """一行 factor_eval.csv 记录（列结构与 factor_eval.reports_to_table 一致）。"""
    return {
        "factor": factor,
        "RankIC": 0.05,
        "RankICIR": rank_icir,
        "RankICIR_ann": rank_icir * 3.5,
        "IC": 0.04,
        "ICIR": 0.3,
        "IC_t": ic_t,
        "IC>0占比": 0.6,
        "多空收益": 0.1,
        "单调性": 0.8,
        "换手": 0.2,
        "准入": "✓",
    }


# ---------------------------------------------------------------------------
# merge_factor_eval_csv / merge_eval_frames
# ---------------------------------------------------------------------------
def test_merge_csv_replaces_and_keeps(tmp_path):
    path = tmp_path / "factor_eval.csv"
    old = pd.DataFrame(
        [_eval_row("a", 5.0, 0.5), _eval_row("b", 3.0, 0.3), _eval_row("c", 1.0, 0.1)]
    )
    old.to_csv(path, index=False)

    new = pd.DataFrame([_eval_row("b", 9.0, 0.9), _eval_row("d", 2.0, 0.2)])
    merged = S.merge_factor_eval_csv(str(path), new)

    # b 被替换为新值，a/c 原样保留，d 插入
    assert sorted(merged["factor"]) == ["a", "b", "c", "d"]
    assert float(merged.loc[merged["factor"] == "b", "IC_t"].iloc[0]) == 9.0
    assert float(merged.loc[merged["factor"] == "a", "RankICIR"].iloc[0]) == 0.5
    # 按 |IC_t| 降序
    assert merged["factor"].tolist() == ["b", "a", "d", "c"]
    # 文件已写回，回读一致
    reread = pd.read_csv(path)
    assert reread["factor"].tolist() == ["b", "a", "d", "c"]


def test_merge_csv_missing_file_creates(tmp_path):
    path = tmp_path / "sub" / "factor_eval.csv"
    new = pd.DataFrame([_eval_row("x", 4.0, 0.4)])
    merged = S.merge_factor_eval_csv(str(path), new)
    assert merged["factor"].tolist() == ["x"]
    assert path.exists()


def test_merge_frames_empty_old():
    new = pd.DataFrame([_eval_row("x", 4.0, 0.4)])
    merged = S.merge_eval_frames(pd.DataFrame(), new)
    assert merged["factor"].tolist() == ["x"]


# ---------------------------------------------------------------------------
# data_import 日期参数解析（load_factor_panel 打桩）
# ---------------------------------------------------------------------------
def test_data_import_date_params(tmp_path, monkeypatch):
    panel = _make_panel()
    seen = {}

    def _fake_load(start_date=None, end_date=None, **kw):
        seen.update(start_date=start_date, end_date=end_date)
        return panel

    monkeypatch.setattr(S, "load_factor_panel", _fake_load)

    ctx = StepContext(
        session_dir=str(tmp_path),
        params={"start_date": "2025-01-01", "end_date": "2025-06-30"},
    )
    rep = S.data_import(ctx)
    assert rep["step"] == "data_import"
    assert seen == {"start_date": "2025-01-01", "end_date": "2025-06-30"}
    assert ctx.shared["factors"] == ["fac_signal", "fac_noise"]

    # 未给显式日期时按 lookback_days 推导
    ctx2 = StepContext(session_dir=str(tmp_path), params={"lookback_days": 90})
    S.data_import(ctx2)
    assert seen["end_date"] is None
    from datetime import datetime

    delta = (pd.Timestamp(datetime.now().date()) - pd.Timestamp(seen["start_date"])).days
    assert 85 <= delta <= 95


# ---------------------------------------------------------------------------
# eval_factors + eval_report smoke（小合成面板）
# ---------------------------------------------------------------------------
@pytest.fixture
def fe_ctx(tmp_path, monkeypatch):
    """C.OUTPUT_DIR 重定向到 tmp_path，避免污染真实 outputs/ 产物。"""
    monkeypatch.setattr(C, "OUTPUT_DIR", str(tmp_path / "outputs"))
    ctx = StepContext(
        session_dir=str(tmp_path),
        params={"factor_subset": "", "n_layers": 5, "dump_layers": True, "with_diagnostics": False},
    )
    ctx.shared["panel"] = _make_panel()
    ctx.shared["factors"] = ["fac_signal", "fac_noise"]
    return ctx


def test_eval_factors_full_mode(fe_ctx, tmp_path):
    rep = S.eval_factors(fe_ctx)
    assert rep["step"] == "eval_factors"

    # canonical 产物：覆盖写 csv
    csv_path = tmp_path / "outputs" / "factor_eval.csv"
    assert csv_path.exists()
    tbl = pd.read_csv(csv_path)
    assert sorted(tbl["factor"]) == ["fac_noise", "fac_signal"]
    for col in ("RankIC", "RankICIR", "IC_t", "单调性", "换手", "准入"):
        assert col in tbl.columns
    # 带真实信号的因子应通过准入
    assert tbl.loc[tbl["factor"] == "fac_signal", "准入"].iloc[0] == "✓"

    # dump_layers：准入通过因子的分层净值 + 逐日 RankIC parquet
    assert (tmp_path / "outputs" / "factor_layer_fac_signal.parquet").exists()
    assert (tmp_path / "outputs" / "factor_ic_daily_fac_signal.parquet").exists()

    # 会话 artifacts（eval_report 重建依据）
    assert (tmp_path / "artifacts" / "eval_table.parquet").exists()
    meta = json.loads((tmp_path / "artifacts" / "eval_meta.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "full"
    assert meta["n_evaluated"] == 2
    assert meta["failed"] == []

    # eval_report 结构
    rep2 = S.eval_report(fe_ctx)
    assert rep2["step"] == "eval_report"
    titles = [t["title"] for t in rep2["tables"]]
    assert "RankICIR Top10" in titles and "RankICIR Bottom10" in titles
    assert "产物路径清单" in titles
    types = [f["type"] for f in rep2["figures"]]
    assert types == ["bar", "histogram"]


def test_eval_report_from_artifacts_only(fe_ctx):
    """shared 缺失时 eval_report 可仅凭 artifact 重建（模拟跨进程回读）。"""
    S.eval_factors(fe_ctx)
    fe_ctx.shared.clear()
    rep = S.eval_report(fe_ctx)
    assert rep["step"] == "eval_report"
    assert rep["kpis"] and rep["tables"]


def test_eval_factors_subset_merges_csv(fe_ctx, tmp_path):
    """subset 模式合并写：预置的无关因子行保留，本次评估行替换/插入。"""
    out = tmp_path / "outputs"
    out.mkdir(parents=True)
    pd.DataFrame([_eval_row("legacy_factor", 8.0, 0.8)]).to_csv(
        out / "factor_eval.csv", index=False
    )

    fe_ctx.params["factor_subset"] = "fac_signal, ghost_factor"
    rep = S.eval_factors(fe_ctx)
    assert rep["step"] == "eval_factors"

    tbl = pd.read_csv(out / "factor_eval.csv")
    # legacy 行保留 + 本次 fac_signal 行插入
    assert sorted(tbl["factor"]) == ["fac_signal", "legacy_factor"]
    # 不存在的因子记入失败清单
    meta = json.loads((tmp_path / "artifacts" / "eval_meta.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "subset"
    assert [f["factor"] for f in meta["failed"]] == ["ghost_factor"]
    assert "不存在" in meta["failed"][0]["error"]

    rep2 = S.eval_report(fe_ctx)
    titles = [t["title"] for t in rep2["tables"]]
    assert "失败/跳过因子及原因" in titles


def test_eval_factors_diagnostics_wiring(fe_ctx, monkeypatch):
    """with_diagnostics=True 时调用诊断生成并记入 report（诊断本身打桩，避免耗时）。"""
    called = {}

    def _fake_diag(panel, factors, out_dir, with_model, ctx=None):
        called.update(n=len(factors), with_model=with_model, out_dir=out_dir)
        return {"files": [f"{out_dir}/diag_factor_corr.csv"], "elapsed_sec": 0.1, "error": ""}

    monkeypatch.setattr(S, "_run_diagnostics", _fake_diag)
    fe_ctx.params["with_diagnostics"] = True
    rep = S.eval_factors(fe_ctx)
    assert called["n"] == 2
    assert called["with_model"] is False
    assert called["out_dir"] == C.OUTPUT_DIR
    labels = [k["label"] for k in rep["kpis"]]
    assert "诊断产物" in labels
