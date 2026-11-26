"""把 pipeline 步骤报告（scripts/run_pipeline.py 落盘到 outputs/runs/<id>/reports/*.json）渲染成页面。

报告是自描述的：{title, summary, kpis[], tables[], figures[], notes[]}，新增流水线步骤不需要写前端，
只要返回 pipeline.context 的 fig_* / table / kpi 构造器产物，这里就能显示。
图规格 type ∈ line|bar|scatter|histogram|area_stacked|heatmap（与旧前端 render.py 同一契约）。
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html

import run_registry as rr

from . import ui

_MAX_REPORT_BYTES = 8_000_000  # 单个报告超过它就不读（曾有 162MB 的报告把前端卡死）


def read_reports(run_id: str, root: str) -> list[dict]:
    """按步骤顺序读取运行的全部报告；损坏/过大的文件跳过并留一条占位说明，不抛异常。"""
    if not rr.valid_run_id(run_id):
        return []
    d = os.path.join(root, run_id, "reports")
    try:
        names = sorted(n for n in os.listdir(d) if n.endswith(".json"))
    except OSError:
        return []
    out = []
    for n in names:
        path = os.path.join(d, n)
        try:
            if os.path.getsize(path) > _MAX_REPORT_BYTES:
                out.append({"title": n, "summary": "报告文件过大，未加载", "_skipped": True})
                continue
            with open(path, encoding="utf-8") as f:
                rep = json.load(f)
            if isinstance(rep, dict):
                out.append(rep)
        except Exception:
            out.append({"title": n, "summary": "报告文件损坏，无法读取", "_skipped": True})
    return out


def _humanize(text) -> str:
    try:
        from model_display import humanize_model_text

        return humanize_model_text(str(text))
    except Exception:
        return str(text)


def figure_from_spec(spec: dict) -> go.Figure:
    t = spec.get("type")
    f = ui.fig(spec.get("title", ""), height=spec.get("height", 300))
    try:
        if t == "line":
            for s in spec["series"]:
                band = s.get("band")
                if band:
                    f.add_scatter(
                        x=spec["x"],
                        y=band["upper"],
                        mode="lines",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                    f.add_scatter(
                        x=spec["x"],
                        y=band["lower"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(8,145,178,0.15)",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                f.add_scatter(
                    x=spec["x"],
                    y=s["y"],
                    mode="lines",
                    name=_humanize(s.get("name", "")),
                    opacity=s.get("opacity"),
                    line=dict(width=s.get("width", 2), color=s.get("color"), dash=s.get("dash")),
                )
        elif t == "bar":
            if spec.get("series"):
                for s in spec["series"]:
                    f.add_bar(
                        x=spec["x"],
                        y=s["y"],
                        name=_humanize(s.get("name", "")),
                        marker_color=s.get("color"),
                        opacity=s.get("opacity"),
                    )
                f.update_layout(barmode="group")
            else:
                y = spec["y"]
                colors = (
                    [ui.C["ok"] if (v is not None and v >= 0) else ui.C["bad"] for v in y]
                    if spec.get("diverging")
                    else (spec.get("color") or ui.C["accent"])
                )
                f.add_bar(x=spec["x"], y=y, marker_color=colors)
        elif t == "scatter":
            f.add_scattergl(
                x=spec["x"],
                y=spec["y"],
                mode="markers",
                marker=dict(size=4, color=ui.C["violet"], opacity=0.4),
                text=spec.get("text"),
                name="点",
            )
            if spec.get("trendline") and len(spec["x"]) > 2:
                x, yv = np.asarray(spec["x"], float), np.asarray(spec["y"], float)
                ok = np.isfinite(x) & np.isfinite(yv)
                if ok.sum() > 2:
                    b = np.polyfit(x[ok], yv[ok], 1)
                    xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
                    f.add_scatter(
                        x=xs,
                        y=np.polyval(b, xs),
                        mode="lines",
                        line=dict(color=ui.C["bad"], width=2),
                        name="拟合",
                    )
        elif t == "histogram":
            if "values" in spec:
                f.add_histogram(
                    x=spec["values"], nbinsx=spec.get("nbins", 60), marker_color=ui.C["violet"]
                )
            else:
                f.add_bar(
                    x=spec["centers"],
                    y=spec["counts"],
                    marker_color=ui.C["violet"],
                    width=spec.get("bin_width"),
                )
            f.update_layout(bargap=0, hovermode="x")
            f.update_yaxes(title_text="频数")
        elif t == "area_stacked":
            for s in spec["series"]:
                f.add_scatter(
                    x=spec["x"],
                    y=s["y"],
                    mode="lines",
                    name=_humanize(s.get("name", "")),
                    line=dict(width=0.5, color=s.get("color")),
                    stackgroup="one",
                    fillcolor=s.get("color"),
                )
        elif t == "heatmap":
            f.add_heatmap(
                z=spec["z"], x=spec.get("x"), y=spec.get("y"), colorscale="RdBu_r", zmid=0
            )
        else:
            return ui.empty_fig(f"未知图表类型：{t}", 240)
    except Exception as exc:  # 单张图的规格坏了不能拖垮整份报告
        return ui.empty_fig(f"图表规格无法渲染：{type(exc).__name__}", 240)
    if spec.get("yaxis"):
        f.update_yaxes(title_text=spec["yaxis"])
    if spec.get("xaxis"):
        f.update_xaxes(title_text=spec["xaxis"])
    return f


def _table(spec: dict):
    cols = [str(c) for c in spec.get("columns", [])]
    rows = spec.get("rows", [])
    if not cols:
        return None
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    n_total, cap = spec.get("n_total", len(df)), spec.get("max_rows", 50)
    extra = f"（显示前 {cap} / 共 {n_total} 行）" if n_total > cap else ""
    return html.Div(
        [
            html.Div(_humanize(spec.get("title", "")) + extra, className="why-sec"),
            ui.table(df, [(c, c, "") for c in cols], max_rows=60),
        ],
        style={"marginTop": "8px"},
    )


def render_report(rep: dict):
    """单份步骤报告 → 卡片。"""
    if rep.get("_skipped"):
        return ui.card(
            rep.get("title", ""), html.Div(rep.get("summary", ""), className="warn"), tone="warn"
        )
    kpis = rep.get("kpis") or []
    parts = []
    if rep.get("summary"):
        parts.append(html.Div(_humanize(rep["summary"]), className="hint"))
    if kpis:
        parts.append(
            html.Div(
                [
                    ui.kpi(
                        _humanize(k.get("value", "")),
                        _humanize(k.get("label", "")),
                        hint=_humanize(k.get("hint", "")),
                    )
                    for k in kpis
                ],
                className="grid g4",
            )
        )
    for note in rep.get("notes") or []:
        is_err = str(note).startswith("❌")
        parts.append(
            html.Pre(
                _humanize(note),
                className="log" if is_err else "events",
                style={"whiteSpace": "pre-wrap", "marginTop": "8px"},
            )
        )
    figs = rep.get("figures") or []
    if figs:
        from . import ui as _ui

        parts.append(
            html.Div(
                [_ui.card("", _ui.graph(figure_from_spec(s))) for s in figs], className="grid g2"
            )
        )
    for t in rep.get("tables") or []:
        tv = _table(t)
        if tv is not None:
            parts.append(tv)
    status = rep.get("status", "done")
    head = html.Span(
        [
            ui.badge(
                {"done": "完成", "failed": "失败"}.get(status, status),
                "done" if status == "done" else "failed",
            ),
            html.Span(f"  {rep.get('elapsed', 0)}s", className="muted num"),
        ]
    )
    return ui.card(rep.get("title") or rep.get("step", ""), head, *parts)


def render_reports(reports: list[dict]):
    if not reports:
        return None
    return html.Div([render_report(r) for r in reports])
