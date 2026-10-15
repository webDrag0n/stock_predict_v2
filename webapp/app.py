"""Dash app entry: light theme, top navbar, multi-page router.

Pages live in webapp/pages/*.py and each exposes `layout` (a component or a
zero-arg function) and a `register(app)` to wire its callbacks. The app warm-loads
the light caches at import so the first interaction is snappy.
"""
from __future__ import annotations

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from . import dataio
from . import sessions as sessions_mgr
from .theme import PALETTE
from .pages import sessions as sessions_page
from .pages import session_detail, market_explorer, factor_lab

# --- app -------------------------------------------------------------------
app = dash.Dash(
    __name__,
    title="QuantLab — A股量化分析平台",
    external_stylesheets=[dbc.themes.FLATLY,
                          "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"],
    suppress_callback_exceptions=True,
    update_title=None,
)
server = app.server

# warm-load light caches once at startup
_STATUS = dataio.warm_load()

# 启动对账：把磁盘上 pid 已死的僵尸 running 会话标记 failed（防阻塞队列）
_N_RECONCILED = sessions_mgr.reconcile_on_startup()

# 重型首次 IO（consolidated_bars / market_breadth）后台线程预热，不阻塞启动
dataio.preheat_async()

NAV_ITEMS = [
    ("/", "会话", "Sessions"),
    ("/market", "行情浏览器", "Market Explorer"),
    ("/factor", "因子实验室", "Factor Lab"),
]


def _navbar():
    links = [
        dbc.NavItem(dcc.Link(
            html.Span([html.Span(zh, className="nav-zh"),
                       html.Span(en, className="nav-en")]),
            href=href, className="nav-link-custom", id=f"nav-{href}"))
        for href, zh, en in NAV_ITEMS
    ]
    status = (f"{_STATUS['n_symbols']:,} 股 · "
              f"{_STATUS['n_daily_files']:,} 交易日 · "
              f"{len(_STATUS['seeds'])} seeds · "
              f"{_STATUS['n_factors']} 因子")
    return dbc.Navbar(
        dbc.Container([
            html.Div([
                html.Span("📈", style={"fontSize": "22px", "marginRight": "8px"}),
                html.Span("QuantLab", className="brand-title"),
                html.Span("A股全市场量化分析平台", className="brand-sub"),
            ], className="brand-wrap"),
            dbc.Nav(links, className="ms-auto", navbar=True),
            html.Div(status, className="status-chip"),
        ], fluid=True),
        color="white", sticky="top", className="qnavbar",
    )


app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    _navbar(),
    html.Div(id="page-content", className="page-wrap"),
], className="app-root")


PAGES = {
    "/": sessions_page,
    "/market": market_explorer,
    "/factor": factor_lab,
}


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def _route(pathname):
    pathname = pathname or "/"
    # 动态会话详情路由 /session/<id>
    if pathname.startswith("/session/"):
        lay = session_detail.layout
        return lay() if callable(lay) else lay
    mod = PAGES.get(pathname, sessions_page)
    lay = mod.layout
    return lay() if callable(lay) else lay


# register each page's callbacks
for _mod in (sessions_page, session_detail, market_explorer, factor_lab):
    if hasattr(_mod, "register"):
        _mod.register(app)
