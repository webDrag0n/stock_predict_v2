"""Market Explorer — per-stock candlestick + overlays, and whole-market breadth.

Per-symbol OHLCV comes from the consolidated bars cache (instant). Technical
overlays (MA / RSI / MACD / Bollinger) are computed on the fly. Whole-market
views (breadth, return distribution, limit up/down) come from the precomputed
breadth table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import dataio
from ..theme import PALETTE, base_fig, empty_fig

_UNIV = dataio.universe()
_SYMBOLS = _UNIV["symbol"].tolist() if not _UNIV.empty else []


def layout():
    if not _SYMBOLS:
        return dbc.Container([
            html.H4("行情浏览器 Market Explorer", className="page-title"),
            dbc.Alert("未找到行情缓存 (data_cache/tushare/daily_*.parquet)。", color="warning"),
        ], fluid=True)
    return dbc.Container([
        html.H4("行情浏览器 Market Explorer", className="page-title"),
        html.P("个股 K 线 + 成交量 + 技术指标叠加；全市场涨跌宽度、收益分布、涨跌停统计。",
               className="page-sub"),
        # --- single-stock section ---
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div("选择股票", className="control-label"),
                    dcc.Dropdown(id="me-symbol", options=_SYMBOLS[:1500],
                                 value=_SYMBOLS[0], clearable=False,
                                 placeholder="搜索代码…"),
                ], md=3),
                dbc.Col([
                    html.Div("技术指标叠加", className="control-label"),
                    dcc.Checklist(
                        id="me-overlays",
                        options=[{"label": " MA20/60", "value": "ma"},
                                 {"label": " 布林带", "value": "boll"},
                                 {"label": " RSI", "value": "rsi"},
                                 {"label": " MACD", "value": "macd"}],
                        value=["ma"], inline=True,
                        inputStyle={"marginRight": "4px", "marginLeft": "12px"}),
                ], md=6),
                dbc.Col([
                    html.Div("时间窗口", className="control-label"),
                    dcc.Dropdown(id="me-window", clearable=False, value="250",
                                 options=[{"label": "近 1 年", "value": "250"},
                                          {"label": "近 2 年", "value": "500"},
                                          {"label": "全部", "value": "0"}]),
                ], md=3),
            ], className="g-3"),
        ], className="control-card"),
        dcc.Loading(html.Div(dcc.Graph(id="me-candle"), className="chart-card")),
        # --- whole-market section ---
        html.H5("全市场视图 Whole-Market", className="page-title",
                style={"fontSize": "18px", "marginTop": "8px"}),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(id="me-breadth"), className="chart-card"), md=7),
            dbc.Col(html.Div(dcc.Graph(id="me-limit"), className="chart-card"), md=5),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(id="me-retdist"), className="chart-card"), md=6),
            dbc.Col(html.Div(dcc.Graph(id="me-meanret"), className="chart-card"), md=6),
        ], className="g-0"),
    ], fluid=True)


# --- indicator helpers -----------------------------------------------------
def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / (dn + 1e-12)
    return 100 - 100 / (1 + rs)


def _candle_fig(symbol, overlays, window):
    bars = dataio.symbol_bars(symbol)
    if bars.empty:
        return empty_fig(f"{symbol} 无数据")
    if window and int(window) > 0:
        bars = bars.tail(int(window))
    bars = bars.reset_index(drop=True)
    has_macd = "macd" in overlays
    has_rsi = "rsi" in overlays
    n_extra = int(has_rsi) + int(has_macd)
    rows = 2 + n_extra
    heights = [0.56, 0.16] + [0.14] * n_extra
    titles = [f"{symbol} K线", "成交量"] + (["RSI(14)"] if has_rsi else []) + \
             (["MACD"] if has_macd else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=heights,
                        subplot_titles=titles)
    x = bars["date"]
    fig.add_trace(go.Candlestick(
        x=x, open=bars["open"], high=bars["high"], low=bars["low"],
        close=bars["close"], name="K线",
        increasing_line_color=PALETTE["up"], decreasing_line_color=PALETTE["down"],
        increasing_fillcolor=PALETTE["up"], decreasing_fillcolor=PALETTE["down"]),
        row=1, col=1)
    if "ma" in overlays:
        for w, col in [(20, "#2563eb"), (60, "#c026d3")]:
            fig.add_trace(go.Scatter(x=x, y=bars["close"].rolling(w).mean(),
                          line=dict(color=col, width=1.2), name=f"MA{w}"), row=1, col=1)
    if "boll" in overlays:
        ma = bars["close"].rolling(20).mean(); sd = bars["close"].rolling(20).std()
        fig.add_trace(go.Scatter(x=x, y=ma + 2*sd, line=dict(color="#94a3b8", width=0.8),
                      name="布林上轨"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=ma - 2*sd, line=dict(color="#94a3b8", width=0.8),
                      fill="tonexty", fillcolor="rgba(148,163,184,0.08)",
                      name="布林下轨"), row=1, col=1)
    vol_colors = np.where(bars["close"] >= bars["open"], PALETTE["up"], PALETTE["down"])
    fig.add_trace(go.Bar(x=x, y=bars["volume"], marker_color=vol_colors,
                  name="成交量", showlegend=False), row=2, col=1)
    r = 3
    if has_rsi:
        fig.add_trace(go.Scatter(x=x, y=_rsi(bars["close"]),
                      line=dict(color="#7b3fb5", width=1.2), name="RSI"), row=r, col=1)
        fig.add_hline(y=70, line=dict(color="#d62728", width=0.6, dash="dot"), row=r, col=1)
        fig.add_hline(y=30, line=dict(color="#16a34a", width=0.6, dash="dot"), row=r, col=1)
        r += 1
    if has_macd:
        ema12 = bars["close"].ewm(span=12).mean(); ema26 = bars["close"].ewm(span=26).mean()
        macd = ema12 - ema26; sig = macd.ewm(span=9).mean(); hist = macd - sig
        fig.add_trace(go.Bar(x=x, y=hist, marker_color=np.where(hist >= 0, PALETTE["up"], PALETTE["down"]),
                      name="MACD柱", showlegend=False), row=r, col=1)
        fig.add_trace(go.Scatter(x=x, y=macd, line=dict(color="#2563eb", width=1), name="DIF"), row=r, col=1)
        fig.add_trace(go.Scatter(x=x, y=sig, line=dict(color="#f59e0b", width=1), name="DEA"), row=r, col=1)
    fig.update_layout(template="quantlight", height=300 + 130 * rows,
                      xaxis_rangeslider_visible=False, hovermode="x unified",
                      margin=dict(l=55, r=20, t=40, b=30), legend=dict(orientation="h", y=1.04))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def _breadth_fig():
    b = dataio.market_breadth()
    if b.empty:
        return empty_fig("无宽度数据")
    x = pd.to_datetime(b["date"])
    fig = base_fig("全市场涨跌家数 (Advancers vs Decliners)", height=320)
    fig.add_trace(go.Scatter(x=x, y=b["advancers"], line=dict(color=PALETTE["up"], width=1),
                  fill="tozeroy", fillcolor="rgba(22,163,74,0.10)", name="上涨家数"))
    fig.add_trace(go.Scatter(x=x, y=-b["decliners"], line=dict(color=PALETTE["down"], width=1),
                  fill="tozeroy", fillcolor="rgba(220,38,38,0.10)", name="下跌家数"))
    fig.update_yaxes(title="家数 (上正/下负)"); fig.update_xaxes(title="")
    return fig


def _limit_fig():
    b = dataio.market_breadth()
    if b.empty:
        return empty_fig("无数据")
    x = pd.to_datetime(b["date"])
    fig = base_fig("每日涨停 / 跌停家数", height=320)
    fig.add_trace(go.Scatter(x=x, y=b["limit_up"], line=dict(color=PALETTE["up"], width=1.2),
                  name="涨停"))
    fig.add_trace(go.Scatter(x=x, y=b["limit_down"], line=dict(color=PALETTE["down"], width=1.2),
                  name="跌停"))
    fig.update_yaxes(title="家数"); fig.update_xaxes(title="")
    return fig


def _retdist_fig():
    files = dataio.daily_files()
    if not files:
        return empty_fig("无数据")
    df = pd.read_parquet(files[-1], columns=["pct_chg", "date"])
    d = pd.to_datetime(df["date"].iloc[0]).strftime("%Y-%m-%d") if not df.empty else ""
    fig = base_fig(f"最新交易日个股收益分布 ({d})", height=320)
    fig.add_trace(go.Histogram(x=df["pct_chg"].clip(-11, 11), nbinsx=60,
                  marker_color=PALETTE["strat"], opacity=0.8, name="个股涨跌幅"))
    fig.add_vline(x=0, line=dict(color="grey", width=0.8, dash="dash"))
    fig.update_xaxes(title="涨跌幅 (%)"); fig.update_yaxes(title="家数")
    fig.update_layout(showlegend=False)
    return fig


def _meanret_fig():
    b = dataio.market_breadth()
    if b.empty:
        return empty_fig("无数据")
    x = pd.to_datetime(b["date"])
    cum = (1 + b["mean_ret"] / 100.0).cumprod()
    fig = base_fig("全市场等权指数 (累计平均日收益)", height=320)
    fig.add_trace(go.Scatter(x=x, y=cum, line=dict(color=PALETTE["alpha"], width=1.6),
                  fill="tozeroy", fillcolor="rgba(123,63,181,0.06)", name="等权指数"))
    fig.update_yaxes(title="净值 (起点=1)"); fig.update_xaxes(title="")
    fig.update_layout(showlegend=False)
    return fig


def register(app):
    @app.callback(Output("me-candle", "figure"),
                  [Input("me-symbol", "value"), Input("me-overlays", "value"),
                   Input("me-window", "value")])
    def _upd_candle(symbol, overlays, window):
        return _candle_fig(symbol, overlays or [], window)

    @app.callback([Output("me-breadth", "figure"), Output("me-limit", "figure"),
                   Output("me-retdist", "figure"), Output("me-meanret", "figure")],
                  Input("me-symbol", "value"))   # any trigger; breadth is global
    def _upd_market(_):
        return _breadth_fig(), _limit_fig(), _retdist_fig(), _meanret_fig()
