"""步骤报告渲染：scripts/run_pipeline.py 落盘的 reports/*.json → 运行详情页。

用 pipeline.context 的真实构造器（kpi/table/fig_*）造报告，保证渲染器与生产者的契约一致，
而不是手写一份「看起来像」的 JSON。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import run_registry as rr
from pipeline.context import (
    fig_area_stacked,
    fig_bar,
    fig_heatmap,
    fig_hist,
    fig_line,
    fig_scatter,
    kpi,
    report,
    table,
)
from webapp2 import data
from webapp2 import report_view as rv
from webapp2.pages import runs

from tests.test_webapp2_pages import text_of  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr(data, "ROOT", str(tmp_path))
    monkeypatch.setattr(data, "OUTPUTS", str(out))
    data.refresh()
    yield tmp_path
    data.refresh()


def put_report(sandbox, run_id, name, rep):
    d = sandbox / "outputs" / "runs" / run_id / "reports"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(rep, ensure_ascii=False))
    return d


def real_report():
    x = list(range(10))
    return report(
        "eval_factors",
        "因子评估",
        summary="评估 3 个因子",
        kpis=[kpi("评估因子数", "3", "候选 3"), kpi("准入通过", "2")],
        figures=[
            fig_line(
                "折线",
                x,
                [
                    {
                        "name": "gbm_l2",
                        "y": list(np.arange(10.0)),
                        "band": {
                            "upper": list(np.arange(10.0) + 1),
                            "lower": list(np.arange(10.0) - 1),
                        },
                    }
                ],
            ),
            fig_bar("柱状", ["a", "b"], y=[1.0, -1.0], diverging=True),
            fig_bar(
                "分组",
                ["a", "b"],
                series=[{"name": "s1", "y": [1, 2]}, {"name": "s2", "y": [2, 1]}],
            ),
            fig_scatter("散点", x, [float(i) for i in x], trendline=True),
            fig_hist("直方", np.random.default_rng(0).normal(size=500)),
            fig_area_stacked(
                "堆叠", x, [{"name": "a", "y": [1.0] * 10}, {"name": "b", "y": [2.0] * 10}]
            ),
            fig_heatmap("热力", [[1, -1], [-1, 1]], x=["p", "q"], y=["p", "q"]),
        ],
        tables=[table("明细", pd.DataFrame({"factor": ["ep", "bp"], "RankIC": [0.05, 0.02]}))],
        notes=["⚠️ 提示一条", "❌ 步骤失败：\nTraceback..."],
    )


def test_every_figure_type_the_producer_can_emit_is_renderable():
    rep = real_report()
    assert len(rep["figures"]) == 7
    for spec in rep["figures"]:
        fig = rv.figure_from_spec(spec)
        assert len(fig.data) >= 1, spec["type"]  # 不是「未知类型」的空图
        assert not fig.layout.annotations or "无法渲染" not in str(fig.layout.annotations)


def test_gbm_keys_are_humanised_in_series_names():
    fig = rv.figure_from_spec(real_report()["figures"][0])
    names = [t.name for t in fig.data if t.name]
    assert names and "gbm_l2" not in names and "AI 树模型" in names


def test_unknown_or_broken_specs_degrade_to_an_annotated_empty_figure():
    unknown = rv.figure_from_spec({"type": "voronoi", "title": "x"})
    assert "未知图表类型" in unknown.layout.annotations[0].text
    broken = rv.figure_from_spec({"type": "line", "title": "x"})  # 缺 series/x
    assert "无法渲染" in broken.layout.annotations[0].text


def test_render_report_shows_kpis_tables_notes_and_figures():
    t = text_of(rv.render_report(real_report() | {"status": "done", "elapsed": 12.3}))
    for needle in (
        "因子评估",
        "评估 3 个因子",
        "评估因子数",
        "准入通过",
        "明细",
        "ep",
        "0.05",
        "提示一条",
        "Traceback",
        "12.3s",
        "完成",
    ):
        assert needle in t, needle


def test_failed_report_is_marked_failed():
    assert "失败" in text_of(rv.render_report({"title": "x", "status": "failed", "elapsed": 1}))


def test_table_truncation_note():
    spec = table("大表", pd.DataFrame({"a": range(200)}), max_rows=50)
    assert "显示前 50 / 共 200 行" in text_of(rv._table(spec))


# ----------------------------------------------------------------------------- 读取
def test_read_reports_in_step_order_and_tolerates_bad_files(sandbox):
    put_report(sandbox, "r1", "01_eval.json", {"title": "第二步", "summary": "b"})
    d = put_report(sandbox, "r1", "00_load.json", {"title": "第一步", "summary": "a"})
    (d / "02_bad.json").write_text("{broken")
    (d / "03_list.json").write_text("[1,2,3]")  # 合法 JSON 但不是报告
    (d / "notes.txt").write_text("ignore me")
    reps = rv.read_reports("r1", str(sandbox / "outputs" / "runs"))
    assert [r["title"] for r in reps[:2]] == ["第一步", "第二步"]  # 文件名排序 = 步骤顺序
    assert any(r.get("_skipped") and "损坏" in r["summary"] for r in reps)
    assert len(reps) == 3  # 列表型被丢弃，坏文件留占位，txt 忽略


def test_oversized_report_is_skipped_not_loaded(sandbox, monkeypatch):
    """历史上有 162MB 的报告把前端卡死；超限的不读。"""
    monkeypatch.setattr(rv, "_MAX_REPORT_BYTES", 100)
    put_report(sandbox, "r2", "00_big.json", {"title": "大", "summary": "x" * 500})
    reps = rv.read_reports("r2", str(sandbox / "outputs" / "runs"))
    assert reps[0]["_skipped"] and "过大" in reps[0]["summary"]
    assert "过大" in text_of(rv.render_report(reps[0]))


def test_read_reports_rejects_unsafe_run_ids_and_missing_dirs(sandbox):
    root = str(sandbox / "outputs" / "runs")
    assert rv.read_reports("../../etc", root) == [] and rv.read_reports("nope", root) == []
    assert rv.render_reports([]) is None


# ----------------------------------------------------------------------------- 运行详情页
def test_run_detail_page_includes_the_step_reports(sandbox):
    root = str(sandbox / "outputs" / "runs")
    with rr.RunReporter("factoreval", label="因子评估", run_id="fe1", root=root) as r:
        r.step("eval_factors", "因子评估")
        r.step_done("eval_factors", "ok")
    put_report(
        sandbox, "fe1", "00_eval_factors.json", real_report() | {"status": "done", "elapsed": 3}
    )
    t = text_of(runs.detail_view("fe1"))
    assert "评估 3 个因子" in t and "准入通过" in t and "明细" in t
    # 没有报告目录的运行（夜间/回测）不受影响
    with rr.RunReporter("nightly", label="夜间", run_id="n1", root=root):
        pass
    assert "夜间" in text_of(runs.detail_view("n1"))
