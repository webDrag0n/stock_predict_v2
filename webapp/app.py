"""Dash app entry: 左侧边栏导航 + 顶部状态条 + 多页路由（前端重构 P0 骨架）。

信息架构见 docs/frontend_redesign.html 第 2 节：
  /            交易台 Trading Desk（默认首页）
  /models      模型实验室 Model Lab
  /factors     因子工作台 Factor Workbench
  /market      行情观察（保留现有实现）
  /runs/<id>   会话详情（通用 step-report 渲染，原 /session/<id>）
旧路由 /session/<id> 重定向到 /runs/<id>（dcc.Location refresh 实现）。

Pages live in webapp/pages/*.py and each exposes `layout` (a component or a
zero-arg function) and a `register(app)` to wire its callbacks. The app warm-loads
the light caches at import so the first interaction is snappy.
"""

from __future__ import annotations

import os

import dash
from dash import dcc, html, Input, Output, ALL
import dash_bootstrap_components as dbc
import pandas as pd

from . import dataio
from . import sessions as sessions_mgr
from .pages import trading_desk, model_lab, factor_workbench, session_detail, market_explorer

# --- app -------------------------------------------------------------------
app = dash.Dash(
    __name__,
    title="QuantLab — A股量化分析平台",
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    update_title=None,
)
server = app.server

# warm-load light caches once at startup
_STATUS = dataio.warm_load()

# 启动对账：把磁盘上 pid 已死的僵尸 running 会话标记 failed（防阻塞队列）。
# debug 模式下 werkzeug reloader 会以双进程导入本模块：父进程（文件监听）不
# 服务请求，只有携带 WERKZEUG_RUN_MAIN=true 的子进程真正 serving；对账有磁盘
# 副作用、预热有重型 IO，都只在 serving 进程执行一次，非 debug 单进程启动不受影响。
_IN_RELOADER_PARENT = (
    os.environ.get("DEBUG", "0") == "1" and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
)
if _IN_RELOADER_PARENT:
    _N_RECONCILED = 0
else:
    _N_RECONCILED = sessions_mgr.reconcile_on_startup()
    # 重型首次 IO（consolidated_bars / market_breadth）后台线程预热，不阻塞启动
    dataio.preheat_async()

# --- 侧边栏导航 --------------------------------------------------------------
NAV_ITEMS = [
    ("/", "交易台", "每日出单与执行"),
    ("/models", "模型实验室", "训练·评估·回测"),
    ("/factors", "因子工作台", "因子研发与体检"),
    ("/market", "行情观察", "个股与全市场行情"),
]


def _nav_active(pathname: str, href: str) -> bool:
    """当前页高亮：/ 精确匹配，其余前缀匹配；会话详情 /runs/、/session/ 归模型实验室。"""
    if href == "/":
        return pathname == "/"
    if href == "/models" and (pathname.startswith("/runs/") or pathname.startswith("/session/")):
        return True
    return pathname.startswith(href)


def _sidebar():
    links = [
        dbc.NavLink(
            html.Div(
                [
                    html.Span(zh, className="side-zh"),
                    html.Span(en, className="side-en"),
                ]
            ),
            href=href,
            active=_nav_active("/", href),  # 初始渲染高亮首页；之后由回调维护
            id={"type": "nav-link", "href": href},
            className="side-link",
        )
        for href, zh, en in NAV_ITEMS
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Span("📈", style={"fontSize": "22px", "marginRight": "8px"}),
                    html.Span("QuantLab", className="brand-title"),
                ],
                className="brand-wrap",
                style={"padding": "18px 18px 4px"},
            ),
            html.Div(
                "A股全市场量化分析平台",
                className="brand-sub",
                style={"padding": "0 18px 14px"},
            ),
            dbc.Nav(links, vertical=True, pills=True, className="side-nav"),
        ],
        className="sidebar",
    )


# --- 顶部状态条 ---------------------------------------------------------------
def _chip(chip_id: str, txt: str, tip: str) -> html.Span:
    """一枚状态 chip + 悬停说明 Tooltip（target 用固定 id，同一时刻状态条只渲染一份）。"""
    return html.Span(
        [
            html.Span(txt, className="status-chip", id=chip_id),
            dbc.Tooltip(tip, target=chip_id, placement="bottom"),
        ]
    )


def _status_bar() -> list:
    """数据新鲜度 · champion 模型年龄 · 调度并发。全部从磁盘读，逐项容错。"""
    chips = []
    # 数据新鲜度：最新 daily 缓存日期
    tip = "最新行情缓存日期，daily_run 后更新"
    try:
        dts = dataio.trade_dates()
        latest = dts[-1] if dts else None
        txt = f"数据 {latest[:4]}-{latest[4:6]}-{latest[6:]}" if latest else "数据 无缓存"
    except Exception:
        txt = "数据 读取失败"
    chips.append(_chip("chip-data", txt, tip))
    # champion 模型年龄
    tip = "当前线上 champion 模型与其年龄"
    try:
        man = dataio.champion_manifest()
        if not man:
            txt = "无 champion"
        else:
            created = pd.to_datetime(man.get("created_at"), errors="coerce")
            if pd.isna(created):
                txt = f"champion {man.get('model_id', '?')}"
            else:
                age = (pd.Timestamp.now() - created.tz_localize(None)).days
                txt = f"champion {man.get('model_id', '?')} · {age}天"
    except Exception:
        txt = "champion 读取失败"
    chips.append(_chip("chip-model", txt, tip))
    # 调度并发：运行中/上限
    tip = "运行中会话数 / 调度并发上限"
    try:
        n_running, n_max = sessions_mgr.concurrency()
        txt = f"并发 {n_running}/{n_max}"
    except Exception:
        txt = "并发 读取失败"
    chips.append(_chip("chip-conc", txt, tip))
    return chips


# --- 布局 ---------------------------------------------------------------------
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(
            [
                _sidebar(),
                html.Div(
                    [
                        html.Div(id="status-bar", className="status-bar"),
                        html.Div(id="page-content", className="page-wrap"),
                    ],
                    className="main-col",
                ),
            ],
            className="shell",
        ),
    ],
    className="app-root",
)


PAGES = {
    "/": trading_desk,
    "/models": model_lab,
    "/factors": factor_workbench,
    "/market": market_explorer,
}


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def _route(pathname):
    pathname = pathname or "/"
    # 旧路由 /session/<id> 重定向到 /runs/<id>（产物同构，仅路径迁移）
    if pathname.startswith("/session/"):
        sid = pathname.split("/session/", 1)[1].strip("/")
        return dcc.Location(id="legacy-redirect", href=f"/runs/{sid}", refresh=True)
    # 会话详情 /runs/<id>（通用 step-report 渲染器）
    if pathname.startswith("/runs/"):
        lay = session_detail.layout
        return lay() if callable(lay) else lay
    # 因子工作台子路由 /factors/<id>（详情）与 /factors/new（编辑器）：
    # 前缀匹配后委派给页面模块的 route_content（纯函数）。必须在 PAGES
    # 精确匹配之前——否则子路径落入兜底页（曾用 allow_duplicate 重复输出
    # 回调实现，客户端去重与服务端 callback_map 分派规则不一致，路由结果
    # 依进入方式不同而随机出错，已移除该 hack）。
    if pathname.startswith("/factors/"):
        content = factor_workbench.route_content(pathname)
        return content if content is not None else factor_workbench.layout()
    mod = PAGES.get(pathname, trading_desk)
    lay = mod.layout
    return lay() if callable(lay) else lay


@app.callback(Output("status-bar", "children"), Input("url", "pathname"))
def _refresh_status(_pathname):
    """每次导航刷新状态条（lru_cache 保证重复读取廉价）。"""
    return _status_bar()


@app.callback(
    Output({"type": "nav-link", "href": ALL}, "active"),
    Input("url", "pathname"),
)
def _highlight_nav(pathname):
    pathname = pathname or "/"
    return [_nav_active(pathname, href) for href, _, _ in NAV_ITEMS]


# register each page's callbacks
for _mod in (trading_desk, model_lab, factor_workbench, session_detail, market_explorer):
    if hasattr(_mod, "register"):
        _mod.register(app)
