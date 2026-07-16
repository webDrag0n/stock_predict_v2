"""Predictions / Live — next-day orders, holdings, cost estimate.

Reads the latest outputs/orders_*.csv and holdings.csv produced by
predict_tomorrow.py. Generating fresh orders needs training on current data, so
that button launches a background job (see Jobs page) rather than blocking a
callback.
"""
from __future__ import annotations

import os
import glob

import pandas as pd
from dash import dcc, html, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc

from .. import jobs
from ..theme import PALETTE, money

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS = os.path.join(ROOT, "outputs")


def _latest_orders() -> tuple[str, pd.DataFrame]:
    files = sorted(glob.glob(os.path.join(OUTPUTS, "orders_*.csv")))
    if not files:
        return "", pd.DataFrame()
    f = files[-1]
    try:
        return os.path.basename(f), pd.read_csv(f, dtype={"代码": str})
    except Exception:
        return os.path.basename(f), pd.DataFrame()


def _holdings() -> pd.DataFrame:
    p = os.path.join(OUTPUTS, "holdings.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    try:
        return pd.read_csv(p, dtype={"代码": str})
    except Exception:
        return pd.DataFrame()


def layout():
    fname, orders = _latest_orders()
    holdings = _holdings()
    return dbc.Container([
        html.H4("预测 / 实盘 Predictions", className="page-title"),
        html.P("次日可执行订单（收盘后生成、次日开盘执行，无未来信息）、当前持仓、成本估算。",
               className="page-sub"),
        html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("生成次日订单", className="control-label"),
                    html.Div("基于截至今天的数据训练并选股 → 后台任务（约数分钟）。",
                             style={"fontSize": "12px", "color": PALETTE["muted"],
                                    "marginBottom": "8px"}),
                    dbc.Button("▶ 生成次日订单 (后台)", id="pred-gen", color="success"),
                    html.Span(id="pred-gen-status", style={"marginLeft": "12px",
                              "fontSize": "13px", "color": PALETTE["muted"]}),
                ], className="control-card"), md=8),
                dbc.Col(html.Div([
                    html.Div("当前文件", className="control-label"),
                    html.Div(fname or "（暂无订单文件）", id="pred-fname",
                             style={"fontWeight": 600}),
                    html.Div(f"持仓 {len(holdings)} 只" if not holdings.empty
                             else "空仓", style={"fontSize": "12px",
                             "color": PALETTE["muted"]}),
                ], className="control-card"), md=4),
            ], className="g-3"),
        ]),
        dbc.Row([
            dbc.Col(html.Div(id="pred-orders-card", children=_orders_card(orders),
                             className="chart-card"), md=8),
            dbc.Col(html.Div(id="pred-holdings-card", children=_holdings_card(holdings),
                             className="chart-card"), md=4),
        ], className="g-0"),
        dcc.Interval(id="pred-poll", interval=4000, disabled=True),
        dcc.Store(id="pred-job-id"),
    ], fluid=True)


def _orders_card(orders):
    if orders.empty:
        return html.Div([
            html.Div("次日订单", style={"fontWeight": 600, "fontSize": 15,
                     "padding": "6px 4px 10px"}),
            dbc.Alert("暂无订单。点击「生成次日订单」运行后台任务。", color="secondary"),
        ])
    buys = orders[orders["方向"] == "买入"] if "方向" in orders else orders
    sells = orders[orders["方向"] == "卖出"] if "方向" in orders else orders.iloc[:0]
    summary = html.Div([
        dbc.Badge(f"买入 {len(buys)}", color="success", className="me-2"),
        dbc.Badge(f"卖出 {len(sells)}", color="danger", className="me-2"),
        dbc.Badge(f"共 {len(orders)} 笔", color="secondary"),
    ], style={"marginBottom": "10px"})
    cols = [{"name": c, "id": c} for c in orders.columns]
    return html.Div([
        html.Div("次日订单（卖在前、买在后）", style={"fontWeight": 600,
                 "fontSize": 15, "padding": "6px 4px 10px"}),
        summary,
        dash_table.DataTable(
            data=orders.to_dict("records"), columns=cols,
            page_size=15, sort_action="native", filter_action="native",
            style_cell={"textAlign": "center", "fontSize": 12, "padding": "5px",
                        "fontFamily": "Inter"},
            style_header={"backgroundColor": PALETTE["strat"], "color": "white",
                          "fontWeight": 700},
            style_data_conditional=[
                {"if": {"filter_query": '{方向} = "买入"', "column_id": "方向"},
                 "color": PALETTE["strat"], "fontWeight": 700},
                {"if": {"filter_query": '{方向} = "卖出"', "column_id": "方向"},
                 "color": PALETTE["loss"], "fontWeight": 700}],
        ),
    ])


def _holdings_card(holdings):
    if holdings.empty:
        return html.Div([
            html.Div("当前持仓", style={"fontWeight": 600, "fontSize": 15,
                     "padding": "6px 4px 10px"}),
            dbc.Alert("空仓。", color="secondary"),
        ])
    return html.Div([
        html.Div(f"当前持仓 ({len(holdings)} 只)", style={"fontWeight": 600,
                 "fontSize": 15, "padding": "6px 4px 10px"}),
        dash_table.DataTable(
            data=holdings.to_dict("records"),
            columns=[{"name": c, "id": c} for c in holdings.columns],
            page_size=20, style_cell={"textAlign": "center", "fontSize": 12,
                        "padding": "5px", "fontFamily": "Inter"},
            style_header={"backgroundColor": "#475569", "color": "white",
                          "fontWeight": 700}),
    ])


def register(app):
    @app.callback(
        [Output("pred-gen-status", "children"), Output("pred-poll", "disabled"),
         Output("pred-job-id", "data")],
        Input("pred-gen", "n_clicks"), prevent_initial_call=True)
    def _launch(n):
        job = jobs.launch("orders")
        return f"已启动后台任务 {job.job_id}（在「后台任务」页看日志）", False, job.job_id

    @app.callback(
        [Output("pred-orders-card", "children"),
         Output("pred-holdings-card", "children"),
         Output("pred-fname", "children"),
         Output("pred-gen-status", "children", allow_duplicate=True),
         Output("pred-poll", "disabled", allow_duplicate=True)],
        Input("pred-poll", "n_intervals"), State("pred-job-id", "data"),
        prevent_initial_call=True)
    def _poll(_n, job_id):
        status_txt = no_update
        stop = no_update
        if job_id:
            for j in jobs.all_jobs():
                if j["job_id"] == job_id:
                    if j["status"] != "running":
                        status_txt = f"任务 {job_id} → {j['status']}"
                        stop = True
                    break
        fname, orders = _latest_orders()
        holdings = _holdings()
        return (_orders_card(orders), _holdings_card(holdings),
                fname or "（暂无订单文件）", status_txt, stop)
