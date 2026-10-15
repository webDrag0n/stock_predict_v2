"""通用渲染器 —— 把自描述的 step report 转成 Dash 组件。

新增流水线模块无需写前端：只要返回 context.fig_* 构造的 figure spec，
这里就能渲染。所有图统一套用 theme.py 的浅色模板与调色板。
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from .theme import base_fig, empty_fig, PALETTE, DIVERGING

_STATUS_COLOR = {"running": "warning", "done": "success",
                 "failed": "danger", "killed": "secondary",
                 "queued": "info", "pending": "secondary"}


def _diverging_colors(y):
    return [PALETTE["strat"] if (v is not None and v >= 0) else PALETTE["loss"]
            for v in y]


def _rgba(color, alpha: float) -> str:
    """hex 主色转带 alpha 的 rgba 字符串；非 hex（或未指定）退化为中性灰。"""
    if isinstance(color, str) and color.startswith("#"):
        h = color.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(100,116,139,{alpha})"


def figure_from_spec(spec: dict) -> go.Figure:
    """按 spec["type"] 分派构图。"""
    t = spec.get("type")
    title = spec.get("title", "")
    fig = base_fig(title=title, height=spec.get("height", 320))

    if t == "line":
        for s in spec["series"]:
            band = s.get("band")
            if band:
                base_color = s.get("color")
                fig.add_trace(go.Scatter(
                    x=spec["x"], y=band["upper"], mode="lines",
                    line=dict(width=0, color=base_color),
                    hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(
                    x=spec["x"], y=band["lower"], mode="lines",
                    line=dict(width=0, color=base_color),
                    fill="tonexty", fillcolor=_rgba(base_color, 0.15),
                    hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=spec["x"], y=s["y"], mode="lines", name=s.get("name", ""),
                opacity=s.get("opacity"),
                line=dict(width=s.get("width", 2),
                          color=s.get("color"),
                          dash=s.get("dash"))))
        fig.update_yaxes(title_text=spec.get("yaxis", ""))
        fig.update_xaxes(title_text=spec.get("xaxis", ""))

    elif t == "bar":
        if spec.get("series"):
            for s in spec["series"]:
                fig.add_trace(go.Bar(
                    x=spec["x"], y=s["y"], name=s.get("name", ""),
                    marker_color=s.get("color"), opacity=s.get("opacity")))
            fig.update_layout(barmode="group")
        else:
            y = spec["y"]
            colors = (_diverging_colors(y) if spec.get("diverging")
                      else (spec.get("color") or PALETTE["strat"]))
            fig.add_trace(go.Bar(x=spec["x"], y=y, marker_color=colors))
        fig.update_yaxes(title_text=spec.get("yaxis", ""))
        fig.update_xaxes(title_text=spec.get("xaxis", ""))

    elif t == "scatter":
        fig.add_trace(go.Scattergl(
            x=spec["x"], y=spec["y"], mode="markers",
            marker=dict(size=4, color=PALETTE["commission"], opacity=0.4),
            text=spec.get("text"), name="点"))
        if spec.get("trendline") and len(spec["x"]) > 2:
            x = np.asarray(spec["x"], float); yv = np.asarray(spec["y"], float)
            ok = np.isfinite(x) & np.isfinite(yv)
            if ok.sum() > 2:
                b = np.polyfit(x[ok], yv[ok], 1)
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
                fig.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines",
                                         line=dict(color=PALETTE["loss"], width=2),
                                         name="拟合"))
        fig.update_yaxes(title_text=spec.get("yaxis", ""))
        fig.update_xaxes(title_text=spec.get("xaxis", ""))

    elif t == "histogram":
        # 服务端已预分箱（np.histogram）→ 只传 centers/counts，前端画 Bar。
        # 避免把数百万行原始值塞进响应（曾导致 data_import 报告 162MB）。
        if "values" in spec:                      # 向后兼容旧报告（原始值）
            fig.add_trace(go.Histogram(x=spec["values"], nbinsx=spec.get("nbins", 60),
                                       marker_color=PALETTE["commission"]))
        else:
            fig.add_trace(go.Bar(x=spec["centers"], y=spec["counts"],
                                  marker_color=PALETTE["commission"],
                                  width=spec.get("bin_width")))
        fig.update_xaxes(title_text=spec.get("xaxis", ""))
        fig.update_yaxes(title_text="频数")
        fig.update_layout(hovermode="x", bargap=0)

    elif t == "area_stacked":
        for s in spec["series"]:
            fig.add_trace(go.Scatter(
                x=spec["x"], y=s["y"], mode="lines", name=s.get("name", ""),
                opacity=s.get("opacity"),
                line=dict(width=0.5, color=s.get("color")),
                stackgroup="one", fillcolor=s.get("color")))
        fig.update_yaxes(title_text=spec.get("yaxis", ""))

    elif t == "heatmap":
        fig.add_trace(go.Heatmap(z=spec["z"], x=spec.get("x"), y=spec.get("y"),
                                 colorscale=DIVERGING, zmid=0))
        fig.update_yaxes(title_text=spec.get("yaxis", ""))
        fig.update_xaxes(title_text=spec.get("xaxis", ""))

    else:
        return empty_fig(f"未知图表类型: {t}")

    return fig


def _kpi_card(k: dict):
    return dbc.Col(html.Div([
        html.Div(str(k.get("label", "")), style={"fontSize": "12px",
                 "color": PALETTE["muted"]}),
        html.Div(str(k.get("value", "")), style={"fontSize": "22px",
                 "fontWeight": 700, "color": PALETTE["ink"], "lineHeight": "1.3"}),
        html.Div(str(k.get("hint", "")), style={"fontSize": "11px",
                 "color": PALETTE["muted"]}),
    ], className="control-card", style={"height": "100%"}),
        xs=6, md=3, lg=2, style={"marginBottom": "8px"})


def _table(spec: dict):
    cols = [{"name": c, "id": c} for c in spec["columns"]]
    data = [dict(zip(spec["columns"], row)) for row in spec["rows"]]
    extra = ""
    if spec.get("n_total", 0) > spec.get("max_rows", 50):
        extra = f"（显示前 {spec['max_rows']} / 共 {spec['n_total']} 行）"
    return html.Div([
        html.Div(spec.get("title", "") + extra, style={"fontWeight": 600,
                 "fontSize": "14px", "margin": "8px 0 6px"}),
        dash_table.DataTable(
            columns=cols, data=data, page_size=15,
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "12px", "fontFamily": "Inter, sans-serif",
                        "padding": "6px 10px", "textAlign": "right"},
            style_header={"fontWeight": 600, "backgroundColor": "#f3f4f6"},
            style_data={"borderBottom": "1px solid #eee"},
        ),
    ])


def render_step(report: dict):
    """渲染单个 step report（标题/状态/KPI/图表/表格/notes）。"""
    status = report.get("status", "done")
    children = [
        html.Div([
            html.Span(report.get("title", report.get("step", "")),
                      style={"fontWeight": 700, "fontSize": "17px"}),
            dbc.Badge(status, color=_STATUS_COLOR.get(status, "secondary"),
                      className="ms-2"),
            html.Span(f"  {report.get('elapsed', 0)}s",
                      style={"color": PALETTE["muted"], "fontSize": "12px",
                             "marginLeft": "8px"}),
            html.Div(report.get("summary", ""), style={"color": PALETTE["muted"],
                     "fontSize": "13px", "marginTop": "2px"}),
        ], style={"marginBottom": "12px"}),
    ]
    if report.get("kpis"):
        children.append(dbc.Row([_kpi_card(k) for k in report["kpis"]],
                                className="g-2", style={"marginBottom": "10px"}))
    for note in report.get("notes", []):
        is_err = str(note).startswith("❌")
        children.append(html.Pre(note, style={
            "whiteSpace": "pre-wrap", "fontSize": "12px",
            "background": "#fef2f2" if is_err else "#fffbeb",
            "color": "#991b1b" if is_err else "#92400e",
            "padding": "10px", "borderRadius": "6px", "marginBottom": "8px"}))
    figs = report.get("figures", [])
    fig_cols = []
    for spec in figs:
        fig_cols.append(dbc.Col(dcc.Graph(figure=figure_from_spec(spec),
                                          config={"displayModeBar": False}),
                                md=6, style={"marginBottom": "10px"}))
    if fig_cols:
        children.append(dbc.Row(fig_cols, className="g-3"))
    for tbl in report.get("tables", []):
        children.append(_table(tbl))
    return html.Div(children, className="chart-card",
                    style={"marginBottom": "16px"})
