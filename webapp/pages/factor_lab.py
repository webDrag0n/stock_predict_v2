"""Factor Lab — single-factor IC diagnostics from the cached factor-eval outputs.

Surfaces the leak-free OOS factor diagnostics: IC-decay table (sortable),
factors ranked by |ICIR|, an interactive factor correlation heatmap (find
redundant clusters), and a per-window IC decay line for a selected factor.
Data comes from outputs/diag_factor_ic_decay.csv + diag_factor_corr.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from .. import dataio
from ..theme import PALETTE, base_fig, empty_fig

_IC = dataio.output_csv("diag_factor_ic_decay.csv")
_CORR = dataio.output_csv("diag_factor_corr.csv")
_WCOLS = [c for c in _IC.columns if c.startswith("w")] if not _IC.empty else []


def layout():
    if _IC.empty:
        return dbc.Container([
            html.H4("因子实验室 Factor Lab", className="page-title"),
            dbc.Alert("未找到因子诊断 (outputs/diag_factor_ic_decay.csv)。"
                      "请先运行 scripts/diagnose_factors.py。", color="warning"),
        ], fluid=True)
    factors = _IC["factor"].tolist()
    return dbc.Container([
        html.H4("因子实验室 Factor Lab", className="page-title"),
        html.P("单因子防泄露 OOS 诊断：IC 衰减、ICIR 排序、相关矩阵（找冗余簇）、"
               "逐窗口 IC。数据源：全市场 ~5700 股诊断面板（99 因子含 Alpha158，7 窗口）。",
               className="page-sub"),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(id="fl-icir-bar"), className="chart-card"), md=7),
            dbc.Col([
                html.Div([
                    html.Div("选择因子看逐窗口 IC", className="control-label"),
                    dcc.Dropdown(id="fl-factor", options=factors, value=factors[0],
                                 clearable=False),
                ], className="control-card"),
                html.Div(dcc.Graph(id="fl-decay"), className="chart-card"),
            ], md=5),
        ], className="g-0"),
        html.Div(dcc.Graph(id="fl-heatmap"), className="chart-card"),
        html.Div([
            html.Div("单因子 IC 汇总表（按 |ICIR| 排序）",
                     style={"fontWeight": 600, "fontSize": 15, "padding": "6px 4px 10px"}),
            dash_table.DataTable(
                id="fl-table",
                data=_table_data(), columns=_table_cols(),
                sort_action="native", page_size=20,
                style_cell={"textAlign": "center", "fontSize": 12.5, "padding": "6px",
                            "fontFamily": "Inter"},
                style_header={"backgroundColor": PALETTE["strat"], "color": "white",
                              "fontWeight": 700},
                style_data_conditional=[
                    {"if": {"filter_query": "{ICIR} > 0", "column_id": "ICIR"},
                     "color": PALETTE["strat"]},
                    {"if": {"filter_query": "{ICIR} < 0", "column_id": "ICIR"},
                     "color": PALETTE["loss"]}],
            ),
        ], className="chart-card"),
    ], fluid=True)


def _table_data():
    df = _IC.copy()
    for c in ["IC_mean", "IC_std", "ICIR"]:
        df[c] = df[c].round(4)
    keep = ["factor", "IC_mean", "IC_std", "ICIR", "pos_windows", "n_windows"]
    df["abs_icir"] = df["ICIR"].abs()
    df = df.sort_values("abs_icir", ascending=False).drop(columns="abs_icir")
    return df[keep].to_dict("records")


def _table_cols():
    keep = ["factor", "IC_mean", "IC_std", "ICIR", "pos_windows", "n_windows"]
    return [{"name": c, "id": c} for c in keep]


def _icir_bar():
    df = _IC.copy()
    df["abs"] = df["ICIR"].abs()
    df = df.sort_values("abs", ascending=False).head(25).iloc[::-1]
    colors = [PALETTE["strat"] if v >= 0 else PALETTE["loss"] for v in df["ICIR"]]
    fig = base_fig("因子 ICIR 排名 (Top 25 by |ICIR|，绿正红负)", height=560)
    fig.add_trace(go.Bar(y=df["factor"], x=df["ICIR"], orientation="h",
                  marker_color=colors,
                  text=[f"{v:+.2f}" for v in df["ICIR"]], textposition="outside"))
    fig.add_vline(x=0, line=dict(color="grey", width=0.8))
    fig.update_xaxes(title="ICIR (IC均值/IC标准差)")
    fig.update_layout(margin=dict(l=140))
    return fig


def _decay_fig(factor):
    row = _IC[_IC["factor"] == factor]
    if row.empty or not _WCOLS:
        return empty_fig("无该因子数据")
    vals = row[_WCOLS].to_numpy().ravel().astype(float)
    x = list(range(len(vals)))
    colors = [PALETTE["strat"] if v >= 0 else PALETTE["loss"] for v in vals]
    fig = base_fig(f"{factor} · 各 OOS 窗口 RankIC", height=300)
    fig.add_trace(go.Bar(x=x, y=vals, marker_color=colors,
                  text=[f"{v:+.3f}" for v in vals], textposition="outside"))
    mean_ic = float(row["IC_mean"].iloc[0])
    fig.add_hline(y=mean_ic, line=dict(color=PALETTE["alpha"], width=1.2, dash="dash"),
                  annotation_text=f"均值 {mean_ic:+.3f}")
    fig.add_hline(y=0, line=dict(color="grey", width=0.7))
    fig.update_xaxes(title="窗口 #", tickvals=x)
    fig.update_yaxes(title="RankIC")
    fig.update_layout(showlegend=False)
    return fig


def _heatmap_fig():
    if _CORR.empty:
        return empty_fig("无相关矩阵")
    corr = _CORR.set_index(_CORR.columns[0]) if _CORR.columns[0] == "" or \
        _CORR.columns[0].startswith("Unnamed") else _CORR
    # the index col was read as first column already via output_csv (no index_col)
    m = _CORR.copy()
    first = m.columns[0]
    m = m.set_index(first)
    labels = list(m.columns)
    z = m.to_numpy(dtype=float)
    fig = base_fig("因子相关矩阵 (按日截面 z-score 后，|高相关|=冗余簇)", height=720)
    fig.add_trace(go.Heatmap(z=z, x=labels, y=labels, colorscale="RdBu_r",
                  zmid=0, zmin=-1, zmax=1, colorbar=dict(title="corr")))
    fig.update_xaxes(tickfont=dict(size=7), tickangle=90)
    fig.update_yaxes(tickfont=dict(size=7), autorange="reversed")
    fig.update_layout(margin=dict(l=110, b=110))
    return fig


def register(app):
    @app.callback(Output("fl-decay", "figure"), Input("fl-factor", "value"))
    def _upd_decay(factor):
        return _decay_fig(factor)

    @app.callback(Output("fl-icir-bar", "figure"), Output("fl-heatmap", "figure"),
                  Input("fl-factor", "value"))
    def _upd_static(_):
        return _icir_bar(), _heatmap_fig()
