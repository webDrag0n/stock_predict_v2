"""UI 基础件：纯函数，输入数据、输出 dash 组件。页面只做组合，不重复造轮子。"""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

# --- 图表主题：与 assets/app.css 同源 -------------------------------------------
C = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#eef2f7",
    "accent": "#0891b2",
    "ok": "#16a34a",
    "bad": "#e11d48",
    "warn": "#d97706",
    "run": "#0ea5e9",
    "violet": "#6366f1",
}


def fig(title: str = "", height: int = 300) -> go.Figure:
    f = go.Figure()
    f.update_layout(
        title=dict(text=title, font=dict(size=12.5, color=C["ink"]), x=0.01),
        height=height,
        margin=dict(l=48, r=16, t=40, b=34),
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        font=dict(family="ui-monospace, Menlo, monospace", size=11, color=C["muted"]),
        xaxis=dict(gridcolor=C["grid"], linecolor="#cbd5e1", zeroline=False),
        yaxis=dict(gridcolor=C["grid"], linecolor="#cbd5e1", zerolinecolor="#cbd5e1"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return f


def empty_fig(msg: str = "暂无数据", height: int = 260) -> go.Figure:
    f = fig(height=height)
    f.add_annotation(
        text=msg,
        showarrow=False,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        font=dict(size=13, color="#94a3b8"),
    )
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False)
    return f


def graph(figure: go.Figure, gid: str | None = None):
    return (
        dcc.Graph(figure=figure, id=gid, config={"displayModeBar": False})
        if gid
        else dcc.Graph(figure=figure, config={"displayModeBar": False})
    )


# --- 格式化 -------------------------------------------------------------------------
def fnum(x, fmt: str = "{:+.3f}", na: str = "—") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    return fmt.format(v) if v == v and abs(v) != float("inf") else na


def money(x, na: str = "—") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    if v != v:
        return na
    return f"¥{v:,.0f}"


def pct(x, digits: int = 1, na: str = "—") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    return f"{v * 100:.{digits}f}%" if v == v else na


def dur(sec) -> str:
    try:
        s = float(sec)
    except (TypeError, ValueError):
        return "—"
    if s != s or s < 0:
        return "—"
    if s < 90:
        return f"{s:.0f}秒"
    if s < 5400:
        return f"{s / 60:.0f}分"
    return f"{s / 3600:.1f}小时"


def ago(ts) -> str:
    try:
        d = time.time() - float(ts)
    except (TypeError, ValueError):
        return "—"
    return dur(d) + "前" if d >= 0 else "刚刚"


def clock(ts) -> str:
    try:
        return time.strftime("%m-%d %H:%M:%S", time.localtime(float(ts)))
    except (TypeError, ValueError, OSError):
        return "—"


# --- 组件 -----------------------------------------------------------------------------
def page(eyebrow: str, title: str, subtitle: str, *children):
    return html.Div(
        [
            html.Div(eyebrow, className="eyebrow"),
            html.H1(title, className="title"),
            html.Div(subtitle, className="subtitle"),
            *children,
        ],
        className="page",
    )


def card(title: str, *children, tone: str = ""):
    return html.Div(
        [html.H3(title) if title else None, *children], className=f"card {tone}".strip()
    )


def kpi(value, label: str, tone: str = "", hint: str = ""):
    return html.Div(
        [
            html.Div(value, className=f"v {tone}".strip()),
            html.Div(label, className="l"),
            html.Div(hint, className="hint") if hint else None,
        ],
        className="card kpi",
    )


def badge(text: str, kind: str):
    return html.Span(text, className=f"badge {kind}")


def empty(msg: str):
    return html.Div(msg, className="empty")


def bar(frac: float | None):
    if frac is None:
        return html.Div(html.I(), className="bar indet")
    return html.Div(
        html.I(style={"width": f"{max(0.0, min(1.0, frac)) * 100:.1f}%"}), className="bar"
    )


def table(
    df: pd.DataFrame,
    cols: list[tuple[str, str, str]] | None = None,
    max_rows: int = 200,
    row_class=None,
    empty_msg: str = "暂无数据",
):
    """cols = [(列名, 表头, 类型)]，类型 ∈ ""|n（数字右对齐）；None → 全部列。"""
    if df is None or df.empty:
        return empty(empty_msg)
    cols = cols or [(c, c, "") for c in df.columns]
    cols = [c for c in cols if c[0] in df.columns]
    head = html.Tr([html.Th(h, className="n" if k == "n" else "") for _, h, k in cols])
    rows = []
    for _, r in df.head(max_rows).iterrows():
        cls = row_class(r) if row_class else ""
        rows.append(
            html.Tr(
                [
                    html.Td("—" if pd.isna(r[c]) else str(r[c]), className="n" if k == "n" else "")
                    for c, _, k in cols
                ],
                className=cls,
            )
        )
    note = (
        html.Div(f"仅显示前 {max_rows} 行，共 {len(df)} 行", className="hint")
        if len(df) > max_rows
        else None
    )
    return html.Div([html.Table([html.Thead(head), html.Tbody(rows)], className="t"), note])


def kv_table(rows: list[tuple[str, object]]):
    return html.Table(
        [
            html.Tbody(
                [
                    html.Tr([html.Td(k, className="muted"), html.Td(v, className="n")])
                    for k, v in rows
                ]
            )
        ],
        className="t",
    )
