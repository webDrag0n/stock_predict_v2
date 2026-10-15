"""会话列表页（主页 /）—— 创建会话 + 并发设置 + 会话列表。

· 新建会话：选模式(回测/次日/滚动Walk-Forward/多Seed稳健性, tab 由 registry 驱动)
  → 按 registry PARAM_SCHEMA 动态渲染参数表单 → 创建。
· 并发设置：数字输入写 sessions.MAX_CONCURRENT。
· 列表：状态徽章 + 进度(当前步/总步)。运行中/排队「查看」禁用（满足运行时不可查看）。
· dcc.Interval 周期调用 sessions.tick() 推进调度并刷新列表。
"""
from __future__ import annotations

from dash import dcc, html, Input, Output, State, ctx, no_update, ALL
import dash_bootstrap_components as dbc

from .. import sessions
from ..theme import PALETTE
from pipeline import registry

_STATUS_COLOR = {"running": "warning", "done": "success",
                 "failed": "danger", "killed": "secondary", "queued": "info"}
_STATUS_ZH = {"running": "运行中", "done": "完成", "failed": "失败",
              "killed": "已终止", "queued": "排队中"}


def _param_inputs(mode: str):
    """按 schema 渲染某模式的参数输入控件。"""
    rows = []
    for p in registry.PARAM_SCHEMA[mode]:
        cid = {"type": "sess-param", "mode": mode, "key": p["key"]}
        if p["type"] == "number":
            ctrl = dbc.Input(type="number", value=p["default"], id=cid,
                             min=p.get("min"), max=p.get("max"),
                             step=p.get("step"), size="sm")
        elif p["type"] == "select":
            ctrl = dcc.Dropdown(id=cid, value=p["default"], clearable=False,
                                options=[{"label": o, "value": o} for o in p["options"]])
        elif p["type"] == "bool":
            ctrl = dbc.Checklist(id=cid, value=([p["key"]] if p["default"] else []),
                                 options=[{"label": "", "value": p["key"]}],
                                 switch=True)
        elif p["type"] == "textarea":
            ctrl = dbc.Textarea(id=cid, value=p.get("default", ""),
                                placeholder=p.get("placeholder", ""),
                                style={"height": "80px", "fontSize": "12px",
                                       "fontFamily": "monospace"})
        elif p["type"] == "text":
            ctrl = dbc.Input(type="text", value=p.get("default", ""), id=cid,
                             placeholder=p.get("placeholder", ""), size="sm")
        else:
            ctrl = dbc.Input(value=p["default"], id=cid, size="sm")
        rows.append(dbc.Col([
            html.Div(p["label"], className="control-label"),
            ctrl,
            html.Div(p.get("hint", ""), style={"fontSize": "11px",
                     "color": PALETTE["muted"]}),
        ], md=3, style={"marginBottom": "10px"}))
    return dbc.Row(rows, className="g-2")


def _create_card():
    # 模式 tab 由 registry 驱动：新增模式只改 pipeline/registry.py
    mode_tabs = dbc.Tabs([
        dbc.Tab(html.Div(_param_inputs(m), style={"paddingTop": "12px"}),
                label=registry.MODE_LABELS.get(m, m), tab_id=m)
        for m in registry.STEPS
    ], id="sess-mode-tabs", active_tab="backtest")
    return html.Div([
        html.Div("新建会话 · 模块化流水线", className="control-label",
                 style={"fontSize": "15px", "fontWeight": 600, "marginBottom": "8px"}),
        mode_tabs,
        dbc.Row([
            dbc.Col([html.Div("会话名称（可选）", className="control-label"),
                     dbc.Input(id="sess-label", placeholder="留空自动命名", size="sm")], md=6),
            dbc.Col(dbc.Button("＋ 创建并排队", id="sess-create", color="success",
                               className="w-100", style={"marginTop": "22px"}), md=3),
        ], className="g-2", style={"marginTop": "8px"}),
    ], className="chart-card")


def layout():
    return dbc.Container([
        html.H4("会话 Sessions · 模块化流水线", className="page-title"),
        html.P("每次运行是一条独立流水线（独立进程 + 独立存档）。运行中不可查看，"
               "完成/失败后可查看每步可视化分析。", className="page-sub"),
        _create_card(),
        html.Div([
            dbc.Row([
                dbc.Col(html.Div("会话列表", style={"fontWeight": 600, "fontSize": 15}), md=8),
                dbc.Col([html.Span("并发上限 ", style={"fontSize": 13}),
                         dbc.Input(id="sess-concurrency", type="number", value=sessions.MAX_CONCURRENT,
                                   min=1, max=8, step=1, size="sm",
                                   style={"width": "70px", "display": "inline-block"})],
                        md=4, style={"textAlign": "right"}),
            ], style={"alignItems": "center", "padding": "12px 4px 8px"}),
            html.Div(id="sess-list"),
        ], className="chart-card", style={"marginTop": "16px"}),
        dcc.Interval(id="sess-poll", interval=2500),
        dcc.Location(id="sess-redirect", refresh=True),
    ], fluid=True)


def _list():
    rows = sessions.list_sessions()
    if not rows:
        return dbc.Alert("暂无会话。用上方表单创建一条流水线。", color="secondary")
    items = [dbc.Row([
        dbc.Col(html.B("会话"), md=3), dbc.Col(html.B("模式"), md=1),
        dbc.Col(html.B("状态"), md=2), dbc.Col(html.B("进度"), md=2),
        dbc.Col(html.B("耗时"), md=2), dbc.Col(html.B("操作"), md=2),
    ], style={"borderBottom": "2px solid #e5e7eb", "padding": "6px 0",
              "fontSize": 13})]
    for m in rows:
        st = m.get("status", "queued")
        nsteps = len(m.get("steps", []))
        done = sum(1 for s in m.get("steps", []) if s["status"] == "done")
        cur = m.get("current_step", -1)
        prog = f"{done}/{nsteps}"
        if st == "running" and 0 <= cur < nsteps:
            prog = f"{cur+1}/{nsteps} · {m['steps'][cur]['title']}"
        el = int(sessions.elapsed(m))
        viewable = st in ("done", "failed")
        items.append(dbc.Row([
            dbc.Col([html.Code(m["id"]), html.Br(),
                     html.Span(m.get("label", ""), style={"fontSize": 12,
                               "color": PALETTE["muted"]})], md=3),
            dbc.Col(registry.MODE_LABELS.get(m["mode"], m["mode"]), md=1,
                    style={"fontSize": 13}),
            dbc.Col(dbc.Badge(_STATUS_ZH.get(st, st),
                              color=_STATUS_COLOR.get(st, "secondary")), md=2),
            dbc.Col(html.Span(prog, style={"fontSize": 12}), md=2),
            dbc.Col(f"{el//60}m{el%60}s", md=2, style={"fontSize": 13}),
            dbc.Col([
                dbc.Button("查看", id={"type": "sess-view", "sid": m["id"]},
                           color="primary", size="sm", outline=True,
                           disabled=(not viewable),
                           title=("运行中不可查看" if not viewable else ""),
                           style={"marginRight": "6px"}),
                dbc.Button("删除", id={"type": "sess-del", "sid": m["id"]},
                           color="danger", size="sm", outline=True,
                           disabled=(st == "running"),
                           title=("运行中不可删除，请先等待结束" if st == "running" else "")),
            ], md=2),
        ], style={"padding": "8px 0", "borderBottom": "1px solid #f0f0f0",
                  "alignItems": "center"}))
    return html.Div(items)


def register(app):
    @app.callback(
        Output("sess-list", "children"),
        Output("sess-redirect", "pathname"),
        Input("sess-poll", "n_intervals"),
        Input("sess-create", "n_clicks"),
        Input({"type": "sess-del", "sid": ALL}, "n_clicks"),
        Input({"type": "sess-view", "sid": ALL}, "n_clicks"),
        State("sess-mode-tabs", "active_tab"),
        State("sess-label", "value"),
        State({"type": "sess-param", "mode": ALL, "key": ALL}, "value"),
        State({"type": "sess-param", "mode": ALL, "key": ALL}, "id"),
        prevent_initial_call=False,
    )
    def _refresh(_n, _create, _del, _view, mode, label, pvals, pids):
        trig = ctx.triggered_id
        redirect = no_update
        if trig == "sess-create":
            params = _collect_params(mode, pvals, pids)
            sessions.create(mode, label, params)
        elif isinstance(trig, dict):
            if trig.get("type") == "sess-del":
                sessions.delete(trig["sid"])
            elif trig.get("type") == "sess-view":
                redirect = f"/session/{trig['sid']}"
        # 每次心跳推进调度
        sessions.tick()
        return _list(), redirect

    @app.callback(Output("sess-poll", "disabled"),
                  Input("sess-concurrency", "value"))
    def _set_concurrency(v):
        if v and int(v) >= 1:
            sessions.MAX_CONCURRENT = int(v)
        return False


def _collect_params(mode: str, pvals, pids) -> dict:
    """只取当前激活模式的参数；bool 用 checklist 是否含 key 判定。"""
    params = registry.default_params(mode)
    for val, pid in zip(pvals, pids):
        if pid.get("mode") != mode:
            continue
        key = pid["key"]
        schema = next((p for p in registry.PARAM_SCHEMA[mode] if p["key"] == key), None)
        if schema and schema["type"] == "bool":
            params[key] = (key in (val or []))
        elif val is not None and val != "":
            params[key] = val
    return params
