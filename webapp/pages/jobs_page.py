"""Jobs — launch and monitor heavy background tasks.

Heavy work (training, whole-market backfill, multi-seed, next-day orders) can't
run in a callback, so the Jobs page launches them as detached subprocesses via
webapp/jobs.py and polls their status + log tail with a dcc.Interval.
"""
from __future__ import annotations

from dash import dcc, html, Input, Output, State, ctx, no_update, ALL
import dash_bootstrap_components as dbc

from .. import jobs
from ..theme import PALETTE

_STATUS_COLOR = {"running": "warning", "done": "success",
                 "failed": "danger", "killed": "secondary"}


def _spec_cards():
    cards = []
    for sid, spec in jobs.JOB_SPECS.items():
        cards.append(dbc.Col(html.Div([
            html.Div(spec["label"], style={"fontWeight": 600, "fontSize": 14,
                     "marginBottom": "4px"}),
            html.Div(spec["desc"], style={"fontSize": 12, "color": PALETTE["muted"],
                     "minHeight": "48px", "marginBottom": "10px"}),
            dbc.Button("▶ 启动", id={"type": "job-launch", "spec": sid},
                       color="success", size="sm", className="w-100"),
        ], className="control-card"), md=3))
    return dbc.Row(cards, className="g-3")


def layout():
    return dbc.Container([
        html.H4("后台任务 Jobs", className="page-title"),
        html.P("启动重型任务（约数分钟~小时级），实时查看状态与日志。任务以独立进程运行，"
               "不阻塞界面。", className="page-sub"),
        _spec_cards(),
        html.Div([
            html.Div("任务列表", style={"fontWeight": 600, "fontSize": 15,
                     "padding": "12px 4px 8px"}),
            html.Div(id="jobs-list"),
        ], className="chart-card", style={"marginTop": "16px"}),
        html.Div([
            dbc.Row([
                dbc.Col(html.Div("实时日志", style={"fontWeight": 600, "fontSize": 15}), md=8),
                dbc.Col(dcc.Dropdown(id="jobs-logsel", placeholder="选择任务看日志",
                        clearable=True), md=4),
            ]),
            html.Pre(id="jobs-log", style={
                "background": "#0f172a", "color": "#e2e8f0", "padding": "14px",
                "borderRadius": "8px", "fontSize": "12px", "maxHeight": "420px",
                "overflow": "auto", "marginTop": "10px",
                "fontFamily": "Menlo, Consolas, monospace"}),
        ], className="chart-card"),
        dcc.Interval(id="jobs-poll", interval=3000),
    ], fluid=True)


def _jobs_table():
    rows = jobs.all_jobs()
    if not rows:
        return dbc.Alert("暂无任务。点击上方卡片启动。", color="secondary")
    items = []
    for j in rows:
        el = int(j["elapsed"])
        items.append(dbc.Row([
            dbc.Col(html.Code(j["job_id"]), md=4),
            dbc.Col(dbc.Badge(j["status"], color=_STATUS_COLOR.get(j["status"], "secondary")), md=2),
            dbc.Col(f"{el//60}m{el%60}s", md=2),
            dbc.Col(html.Span(f"PID {j['pid']}", style={"fontSize": 12,
                    "color": PALETTE["muted"]}), md=2),
            dbc.Col(dbc.Button("终止", id={"type": "job-kill", "jid": j["job_id"]},
                    color="danger", size="sm", outline=True,
                    disabled=(j["status"] != "running")), md=2),
        ], className="g-2", style={"padding": "6px 0", "borderBottom": "1px solid #eee",
                                   "alignItems": "center"}))
    return html.Div(items)


def register(app):
    @app.callback(Output("jobs-list", "children"),
                  Output("jobs-logsel", "options"),
                  Input("jobs-poll", "n_intervals"),
                  Input({"type": "job-launch", "spec": ALL}, "n_clicks"),
                  Input({"type": "job-kill", "jid": ALL}, "n_clicks"),
                  prevent_initial_call=False)
    def _refresh(_n, _launch, _kill):
        trig = ctx.triggered_id
        if isinstance(trig, dict):
            if trig.get("type") == "job-launch":
                jobs.launch(trig["spec"])
            elif trig.get("type") == "job-kill":
                jobs.kill(trig["jid"])
        all_j = jobs.all_jobs()
        opts = [{"label": j["job_id"], "value": j["job_id"]} for j in all_j]
        return _jobs_table(), opts

    @app.callback(Output("jobs-log", "children"),
                  Input("jobs-poll", "n_intervals"), Input("jobs-logsel", "value"))
    def _log(_n, job_id):
        if not job_id:
            return "（选择一个任务查看实时日志）"
        return jobs.tail_log(job_id, n=120) or "（暂无日志输出）"
