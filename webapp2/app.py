"""QuantLab v2 应用外壳：侧栏 + 顶部状态条 + 路由。

只读 outputs/ 与 models/，写操作经 webapp2.actions。与 CLI 互不依赖：没有 webapp 时 CLI
照常运行并落盘，本应用后启动也能看到全部历史与进行中的运行。

    uv run python run_webapp2.py          # http://127.0.0.1:8060
"""

from __future__ import annotations

import os

import dash
from dash import Input, Output, dcc, html

import run_registry as rr

from . import data, ui
from .pages import backtest, factors, market, models, overview, runs, trade

NAV = [
    ("/", "总览"),
    ("/trade", "交易"),
    ("/models", "模型"),
    ("/backtest", "回测"),
    ("/factors", "因子"),
    ("/market", "行情"),
    ("/runs", "运行"),
]

app = dash.Dash(
    __name__,
    title="QuantLab",
    assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"),
    # 页面是动态布局：按钮只在对应视图里存在，回调需要允许「输入暂不存在」
    suppress_callback_exceptions=True,
)
server = app.server

PAGES: dict[str, object] = {
    "/": None,
    "/trade": trade,
    "/models": models,
    "/backtest": backtest,
    "/runs": runs,
}


def _sidebar():
    return html.Div(
        [
            html.Div("QUANTLAB", className="brand"),
            html.Div(id="nav", className="nav"),
        ],
        className="side",
    )


def nav_active(pathname: str, href: str) -> bool:
    if href == "/":
        return pathname in ("", "/")
    return pathname == href or pathname.startswith(href + "/")


def status_chips() -> list:
    """顶部状态条：每一项都由文件即时算出，与 CLI 是否在跑无关。"""
    chips = []
    active = rr.active_runs(root=os.path.join(data.OUTPUTS, "runs"))
    if active:
        a = active[0]
        prog = a.get("progress") or {}
        txt = f"运行中 · {a.get('label')}" + (
            f" · {prog.get('phase')}" if prog.get("phase") else ""
        )
        chips.append(
            html.A(
                [html.Span(className="dot pulse"), txt],
                className="chip run",
                href=f"/runs/{a['run_id']}",
            )
        )
    else:
        chips.append(html.Span("空闲", className="chip"))
    champ = data.champion_manifest()
    rep = data.tracking_report() or {}
    age = (rep.get("champion") or {}).get("age_days")
    if champ is None:
        chips.append(html.Span("无 champion", className="chip bad"))
    else:
        tone = "bad" if (age or 0) > 60 else ("warn" if (age or 0) > 48 else "ok")
        chips.append(
            html.Span(
                f"champion {champ.get('model_id', '?')[:14]} · {age if age is not None else '?'}天",
                className=f"chip {tone}",
            )
        )
    acc = data.account_status()
    chips.append(
        html.Span(
            "现金 "
            + (ui.money(acc["cash"]) if acc["cash"] is not None else "未维护")
            + ("（已过期）" if acc["stale"] else ""),
            className="chip " + ("warn" if acc["stale"] else "ok"),
        )
    )
    _, day = data.latest_orders()
    chips.append(html.Span(f"订单 {day or '无'}", className="chip"))
    return chips


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Interval(id="status-tick", interval=5000),
        html.Div(
            [
                _sidebar(),
                html.Div(
                    [html.Div(id="status-bar", className="statusbar"), html.Div(id="page-content")],
                    className="main",
                ),
            ],
            className="shell",
        ),
    ]
)


def route(pathname: str | None, search: str | None = None):
    p = pathname or "/"
    if p == "/runs" or p.startswith("/runs/"):
        return runs.layout(p)
    if p == "/trade":
        return trade.layout()
    if p == "/models":
        return models.layout()
    if p == "/backtest":
        return backtest.layout()
    if p == "/factors" or p.startswith("/factors/"):
        return factors.layout(p)
    if p == "/market":
        return market.layout(search)
    return overview.layout()


@app.callback(Output("page-content", "children"), Input("url", "pathname"), Input("url", "search"))
def _route(pathname, search):
    return route(pathname, search)


@app.callback(Output("nav", "children"), Input("url", "pathname"))
def _nav(pathname):
    p = pathname or "/"
    return [
        dcc.Link(label, href=href, className="on" if nav_active(p, href) else "")
        for href, label in NAV
    ]


@app.callback(
    Output("status-bar", "children"), Input("status-tick", "n_intervals"), Input("url", "pathname")
)
def _status(_n, _p):
    try:
        return status_chips()
    except Exception as exc:  # 状态条不得拖垮页面
        return [html.Span(f"状态读取失败：{type(exc).__name__}", className="chip bad")]


runs.register(app)
overview.register(app)
trade.register(app)
models.register(app)
backtest.register(app)
factors.register(app)
market.register(app)
