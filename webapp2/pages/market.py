"""行情：个股 K 线 + 技术指标，全市场涨跌宽度与涨跌停。只读。

全市场宽度首次计算要几秒，在后台线程里算；算完前图表区显示「计算中」，页面不被卡住，
计算完成后下一次轮询自动换成图表。涨跌停按板块幅度统计（口径见 market_logic 顶部）。
"""

from __future__ import annotations

from urllib.parse import parse_qs

import numpy as np
import pandas as pd
from dash import Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots

from .. import market_logic as ml
from .. import ui

UP, DOWN = "#e11d48", "#16a34a"  # A 股惯例：红涨绿跌
WINDOWS = [("近 1 年", "250"), ("近 2 年", "500"), ("全部", "0")]
OVERLAYS = [("MA20/60", "ma"), ("布林带", "boll"), ("RSI", "rsi"), ("MACD", "macd")]


def symbol_from_search(search: str | None) -> str | None:
    """'?symbol=000001.SZ' → 有效代码；无/非法/不在最新交易日 universe → None。"""
    if not search:
        return None
    sym = (parse_qs(search.lstrip("?")).get("symbol") or [None])[0]
    return sym if sym in set(ml.universe()) else None


def candle_fig(symbol: str, overlays: list[str], window: int):
    bars = ml.symbol_bars(symbol, window)
    if bars.empty:
        return ui.empty_fig(f"{symbol} 没有行情数据", 360)
    bars = bars.reset_index(drop=True)
    overlays = overlays or []
    n_extra = int("rsi" in overlays) + int("macd" in overlays)
    rows = 2 + n_extra
    f = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.56, 0.16] + [0.14] * n_extra,
        subplot_titles=[f"{symbol} K 线", "成交量"]
        + (["RSI(14)"] if "rsi" in overlays else [])
        + (["MACD"] if "macd" in overlays else []),
    )
    x, c = bars["date"], bars["close"]
    f.add_candlestick(
        x=x,
        open=bars["open"],
        high=bars["high"],
        low=bars["low"],
        close=c,
        name="K 线",
        increasing_line_color=UP,
        decreasing_line_color=DOWN,
        increasing_fillcolor=UP,
        decreasing_fillcolor=DOWN,
        row=1,
        col=1,
    )
    if "ma" in overlays:
        for w, col in ((20, "#2563eb"), (60, "#c026d3")):
            f.add_scatter(
                x=x,
                y=c.rolling(w).mean(),
                line=dict(color=col, width=1.2),
                name=f"MA{w}",
                row=1,
                col=1,
            )
    if "boll" in overlays:
        _, up, dn = ml.bollinger(c)
        f.add_scatter(
            x=x, y=up, line=dict(color="#94a3b8", width=0.8), name="布林上轨", row=1, col=1
        )
        f.add_scatter(
            x=x,
            y=dn,
            line=dict(color="#94a3b8", width=0.8),
            fill="tonexty",
            fillcolor="rgba(148,163,184,0.08)",
            name="布林下轨",
            row=1,
            col=1,
        )
    f.add_bar(
        x=x,
        y=bars["volume"],
        marker_color=np.where(c >= bars["open"], UP, DOWN),
        name="成交量",
        showlegend=False,
        row=2,
        col=1,
    )
    r = 3
    if "rsi" in overlays:
        f.add_scatter(
            x=x, y=ml.rsi(c), line=dict(color="#7c3aed", width=1.2), name="RSI", row=r, col=1
        )
        f.add_hline(y=70, line=dict(color=UP, width=0.6, dash="dot"), row=r, col=1)
        f.add_hline(y=30, line=dict(color=DOWN, width=0.6, dash="dot"), row=r, col=1)
        r += 1
    if "macd" in overlays:
        dif, dea, hist = ml.macd(c)
        f.add_bar(
            x=x,
            y=hist,
            marker_color=np.where(hist >= 0, UP, DOWN),
            name="MACD 柱",
            showlegend=False,
            row=r,
            col=1,
        )
        f.add_scatter(x=x, y=dif, line=dict(color="#2563eb", width=1), name="DIF", row=r, col=1)
        f.add_scatter(x=x, y=dea, line=dict(color="#f59e0b", width=1), name="DEA", row=r, col=1)
    f.update_layout(
        height=300 + 130 * rows,
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        hovermode="x unified",
        margin=dict(l=55, r=20, t=40, b=30),
        legend=dict(orientation="h", y=1.04),
        font=dict(family="ui-monospace, Menlo, monospace", size=11, color=ui.C["muted"]),
    )
    f.update_xaxes(rangeslider_visible=False, gridcolor=ui.C["grid"])
    f.update_yaxes(gridcolor=ui.C["grid"])
    return f


def _breadth_cards():
    b = ml.breadth()
    if b is None:
        return ui.card("全市场宽度", ui.empty("正在计算全市场涨跌宽度（首次约 5 秒）…"))
    if b.empty:
        return ui.card(
            "全市场宽度", ui.empty("没有行情缓存（data_cache/tushare/daily_*.parquet）。")
        )
    x = b["date"]
    f1 = ui.fig("涨跌家数", height=280)
    f1.add_scatter(
        x=x,
        y=b["advancers"],
        name="上涨",
        line=dict(color=UP, width=1),
        fill="tozeroy",
        fillcolor="rgba(225,29,72,0.10)",
    )
    f1.add_scatter(
        x=x,
        y=b["decliners"],
        name="下跌",
        line=dict(color=DOWN, width=1),
        fill="tozeroy",
        fillcolor="rgba(22,163,74,0.10)",
    )
    f2 = ui.fig("涨停 / 跌停家数（按板块幅度）", height=280)
    f2.add_scatter(x=x, y=b["limit_up"], name="涨停", line=dict(color=UP, width=1.2))
    f2.add_scatter(x=x, y=b["limit_down"], name="跌停", line=dict(color=DOWN, width=1.2))
    f3 = ui.fig(
        f"全市场平均 / 中位数涨跌幅（%）· 平均已将单只截断在 ±{ml.MEAN_CLIP_PCT:.0f}%", height=280
    )
    f3.add_scatter(x=x, y=b["mean_ret"], name="平均", line=dict(color=ui.C["accent"], width=1))
    f3.add_scatter(x=x, y=b["median_ret"], name="中位数", line=dict(color=ui.C["warn"], width=1))
    last = b.iloc[-1]
    return html.Div(
        [
            html.Div(
                [
                    ui.kpi(str(pd.Timestamp(last["date"]).date()), "最新交易日"),
                    ui.kpi(f"{int(last['advancers'])} / {int(last['decliners'])}", "上涨 / 下跌"),
                    ui.kpi(
                        f"{int(last['limit_up'])} / {int(last['limit_down'])}",
                        "涨停 / 跌停",
                        hint="按板块幅度",
                    ),
                    ui.kpi(f"{last['median_ret']:+.2f}%", "中位数涨跌幅"),
                ],
                className="grid g4",
            ),
            html.Div([ui.card("", ui.graph(f1)), ui.card("", ui.graph(f2))], className="grid g2"),
            ui.card("", ui.graph(f3)),
        ]
    )


OPTION_LIMIT = 60


def symbol_options(selected: str | None, query: str | None = None) -> list[str]:
    """下拉框选项：搜索结果的前 OPTION_LIMIT 个，且一定包含当前选中的代码。

    全市场 5500+ 只，不能全塞进下拉框。选中的代码若不在选项里（例如从其他页面带 ?symbol= 跳来），
    Dash 会把无效的 value 清成 None，页面就变成空白——所以必须把它并进选项。
    """
    opts = ml.search_symbols(query, OPTION_LIMIT)
    return opts if not selected or selected in opts else [selected] + opts


def stock_view(symbol: str | None, overlays, window) -> html.Div:
    if not symbol:
        return ui.empty("请选择一只股票。")
    win = int(window or 250)
    note = ml.stale_note(ml.bars_last_date(symbol), ml.latest_day())
    return html.Div(
        [
            html.Div(note, className="warn") if note else None,
            ui.graph(candle_fig(symbol, list(overlays or []), win)),
        ]
    )


def layout(search: str | None = None):
    syms = ml.universe()
    if not syms:
        return ui.page(
            "MARKET",
            "行情",
            "个股 K 线与全市场宽度。",
            ui.card(
                "",
                ui.empty(
                    "没有行情缓存（data_cache/tushare/daily_*.parquet）。先运行一次夜间任务拉取数据。"
                ),
            ),
        )
    sym = symbol_from_search(search) or syms[0]
    return ui.page(
        "MARKET",
        "行情",
        f"个股 K 线与全市场宽度 · 数据到 {ml.latest_day()}",
        dcc.Interval(id="mk-tick", interval=2000),
        dcc.Store(id="mk-ready", data=False),
        ui.card(
            "个股",
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("代码（输入搜索）", className="lb"),
                            dcc.Dropdown(
                                id="mk-symbol",
                                options=symbol_options(sym),
                                value=sym,
                                clearable=False,
                                searchable=True,
                            ),
                        ],
                        style={"minWidth": "220px"},
                    ),
                    html.Div(
                        [
                            html.Label("叠加指标", className="lb"),
                            dcc.Checklist(
                                id="mk-overlays",
                                options=[{"label": f" {a}", "value": b} for a, b in OVERLAYS],
                                value=["ma"],
                                inline=True,
                                inputStyle={"marginLeft": "12px", "marginRight": "4px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("窗口", className="lb"),
                            dcc.Dropdown(
                                id="mk-window",
                                options=[{"label": a, "value": b} for a, b in WINDOWS],
                                value="250",
                                clearable=False,
                            ),
                        ],
                        style={"minWidth": "140px"},
                    ),
                ],
                className="row",
            ),
            html.Div(stock_view(sym, ["ma"], "250"), id="mk-stock", style={"marginTop": "8px"}),
        ),
        html.Div(_breadth_cards(), id="mk-breadth"),
    )


def register(app):
    @app.callback(
        Output("mk-symbol", "options"),
        Input("mk-symbol", "search_value"),
        State("mk-symbol", "value"),
        prevent_initial_call=True,
    )
    def _search(q, current):
        return symbol_options(current, q)

    @app.callback(
        Output("mk-stock", "children"),
        Input("mk-symbol", "value"),
        Input("mk-overlays", "value"),
        Input("mk-window", "value"),
        prevent_initial_call=True,
    )
    def _stock(symbol, overlays, window):
        return stock_view(symbol, overlays, window)

    @app.callback(
        Output("mk-breadth", "children"),
        Output("mk-ready", "data"),
        Input("mk-tick", "n_intervals"),
        State("mk-ready", "data"),
        prevent_initial_call=True,
    )
    def _poll_breadth(_n, ready):
        """宽度在后台算：算完前每 2 秒问一次，算完换成图表后不再重绘（缩放/悬停状态不被冲掉）。"""
        if ready:
            return no_update, no_update
        if ml.breadth() is None:
            return no_update, False
        return _breadth_cards(), True
