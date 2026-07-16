"""Backtest Lab — interactive multi-seed backtest on cached whole-market OOS.

Move a slider (TOPK / rebalance days / impact-k / seeds) and the periodic backtest
re-runs in seconds on the cached per-seed OOS predictions. Charts draw each seed
faintly and their mean boldly, so robustness is visible at a glance. Numbers match
the CLI multiseed result (Sharpe ~1.33, alpha ~15.4% with impact cost).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import dataio, compute
from ..theme import PALETTE, base_fig, empty_fig, money

_SEEDS = dataio.available_seeds()


def _control_card():
    return html.Div([
        html.Div("策略参数 · 实时重算", className="control-label",
                 style={"fontSize": "14px", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([
                html.Div("持仓比例 TOPK_RATIO", className="control-label"),
                dcc.Slider(0.05, 0.40, 0.05, value=0.15, id="bl-topk",
                           marks={v/100: f"{v}%" for v in range(5, 41, 5)},
                           tooltip={"placement": "bottom"}),
            ], md=4),
            dbc.Col([
                html.Div("调仓周期 (交易日)", className="control-label"),
                dcc.Slider(5, 60, 5, value=20, id="bl-rebal",
                           marks={v: str(v) for v in range(5, 61, 5)},
                           tooltip={"placement": "bottom"}),
            ], md=4),
            dbc.Col([
                html.Div("冲击成本系数 IMPACT_K", className="control-label"),
                dcc.Slider(0.0, 0.3, 0.05, value=0.1, id="bl-impactk",
                           marks={v/100: f"{v/100:.2f}" for v in range(0, 31, 5)},
                           tooltip={"placement": "bottom"}),
            ], md=4),
        ], className="g-3"),
        dbc.Row([
            dbc.Col([
                html.Div("随机种子 (多 seed 复核)", className="control-label"),
                dcc.Checklist(
                    options=[{"label": f" seed {s}", "value": s} for s in _SEEDS],
                    value=list(_SEEDS), id="bl-seeds", inline=True,
                    inputStyle={"marginRight": "5px", "marginLeft": "12px"}),
            ], md=6),
            dbc.Col([
                html.Div("流动性冲击成本模型", className="control-label"),
                dbc.Switch(id="bl-useimpact", value=True,
                           label="启用平方根冲击成本 (更真实)"),
            ], md=4),
            dbc.Col([
                dbc.Button("重新回测", id="bl-run", color="success",
                           className="w-100", style={"marginTop": "22px"}),
            ], md=2),
        ], className="g-3", style={"marginTop": "6px"}),
    ], className="control-card")


def _kpi_row():
    ids = ["sharpe", "alpha", "nwt", "beta", "maxdd", "calmar"]
    labels = ["Sharpe", "年化 Alpha", "NW-t (显著性)", "Beta", "最大回撤", "Calmar"]
    return dbc.Row([
        dbc.Col(html.Div([
            html.Div(lab, className="kpi-label"),
            html.Div("—", className="kpi-value", id=f"bl-kpi-{i}"),
            html.Div("", className="kpi-sub", id=f"bl-kpi-{i}-sub"),
        ], className="kpi-card"), md=2)
        for i, lab in zip(ids, labels)
    ], className="g-2", style={"marginBottom": "18px"})


def layout():
    if not _SEEDS:
        return dbc.Container([
            html.H4("回测实验室 Backtest Lab", className="page-title"),
            dbc.Alert("未找到缓存的 OOS 预测 (data_cache/seed_oos_*.parquet)。"
                      "请先在「后台任务」运行多 seed 复核。", color="warning"),
        ], fluid=True)
    return dbc.Container([
        html.H4("回测实验室 Backtest Lab", className="page-title"),
        html.P("全市场 ~5700 股 · ret_weighted 预测器 · 周期调仓 + 流动性成本模型。"
               "拖动参数即时重算，淡线=各 seed，浓线=均值。",
               className="page-sub"),
        _control_card(),
        dcc.Loading(html.Div([
            _kpi_row(),
            html.Div(dcc.Graph(id="bl-equity", config={"displModeBar": False}),
                     className="chart-card"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="bl-alpha"), className="chart-card"), md=6),
                dbc.Col(html.Div(dcc.Graph(id="bl-rollsharpe"), className="chart-card"), md=6),
            ], className="g-0"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="bl-drawdown"), className="chart-card"), md=6),
                dbc.Col(html.Div(dcc.Graph(id="bl-cost"), className="chart-card"), md=6),
            ], className="g-0"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="bl-monthly"), className="chart-card"), md=6),
                dbc.Col(html.Div(id="bl-table", className="chart-card"), md=6),
            ], className="g-0"),
        ]), type="default"),
    ], fluid=True)


# --- chart builders --------------------------------------------------------
CAP0 = 1_000_000.0


def _fig_equity(agg):
    runs = agg["runs"]; T = agg["T"]; x = np.arange(T)
    fig = base_fig("A. 账户净值 — 各 seed (淡) vs 均值 (浓) vs 等权基准", height=420)
    for i, sd in enumerate(agg["seeds"]):
        net = runs[sd]["net"][:T]
        acct = CAP0 * np.cumprod(1 + net)
        fig.add_trace(go.Scatter(x=x, y=acct, mode="lines",
                      line=dict(color=PALETTE["strat"], width=1),
                      opacity=0.28, name=f"seed {sd}", showlegend=False,
                      hoverinfo="skip"))
    mean_acct = CAP0 * np.cumprod(1 + agg["mean_net"])
    bench_acct = CAP0 * (1 + agg["bench_cum_pct"] / 100.0)
    fig.add_trace(go.Scatter(x=x, y=mean_acct, mode="lines",
                  line=dict(color=PALETTE["strat"], width=3), name="均值策略"))
    fig.add_trace(go.Scatter(x=x, y=bench_acct, mode="lines",
                  line=dict(color=PALETTE["bench"], width=2, dash="dash"),
                  name="等权基准"))
    fig.add_hline(y=CAP0, line=dict(color="grey", width=0.7, dash="dot"))
    fig.update_yaxes(title="账户 (¥)", tickformat=".2s")
    fig.update_xaxes(title="交易日")
    fig.update_layout(legend=dict(x=0.01, y=0.99))
    return fig


def _fig_alpha(agg):
    T = agg["T"]; x = np.arange(T)
    alpha_s = agg["alpha_s"]
    mean_a = alpha_s.mean(axis=0); std_a = alpha_s.std(axis=0)
    fig = base_fig("B. 累计 Alpha (策略−基准) · ±1σ seed 离散带", height=340)
    for i in range(alpha_s.shape[0]):
        fig.add_trace(go.Scatter(x=x, y=alpha_s[i], mode="lines",
                      line=dict(color=PALETTE["alpha"], width=0.8), opacity=0.22,
                      showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                  y=np.concatenate([mean_a + std_a, (mean_a - std_a)[::-1]]),
                  fill="toself", fillcolor="rgba(123,63,181,0.15)",
                  line=dict(width=0), name="±1σ", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=mean_a, mode="lines",
                  line=dict(color=PALETTE["alpha"], width=2.6), name="均值 Alpha"))
    fig.add_hline(y=0, line=dict(color="grey", width=0.8, dash="dash"))
    fig.update_yaxes(title="Alpha (%)")
    fig.update_xaxes(title="交易日")
    return fig


def _fig_rollsharpe(agg, win=63):
    net = pd.Series(agg["mean_net"])
    rs = (net.rolling(win).mean() / net.rolling(win).std() * np.sqrt(252)).to_numpy()
    x = np.arange(len(rs))
    fig = base_fig(f"C. 滚动 {win} 日年化 Sharpe (均值)", height=340)
    pos = np.where(rs >= 0, rs, np.nan)
    neg = np.where(rs < 0, rs, np.nan)
    fig.add_trace(go.Scatter(x=x, y=pos, mode="lines", fill="tozeroy",
                  line=dict(color=PALETTE["strat"], width=1.4),
                  fillcolor="rgba(26,127,55,0.12)", name="Sharpe>0"))
    fig.add_trace(go.Scatter(x=x, y=neg, mode="lines", fill="tozeroy",
                  line=dict(color=PALETTE["loss"], width=1.4),
                  fillcolor="rgba(214,39,40,0.12)", name="Sharpe<0"))
    fig.add_hline(y=1, line=dict(color="#16a34a", width=0.7, dash="dot"))
    fig.add_hline(y=0, line=dict(color="grey", width=0.7))
    fig.update_yaxes(title="Sharpe"); fig.update_xaxes(title="交易日")
    fig.update_layout(showlegend=False)
    return fig


def _fig_drawdown(agg):
    runs = agg["runs"]; T = agg["T"]; x = np.arange(T)
    dd_s = []
    for sd in agg["seeds"]:
        acct = np.cumprod(1 + runs[sd]["net"][:T])
        peak = np.maximum.accumulate(acct)
        dd_s.append((acct - peak) / peak * 100)
    dd_s = np.vstack(dd_s)
    mean_acct = np.cumprod(1 + agg["mean_net"])
    mean_peak = np.maximum.accumulate(mean_acct)
    mean_dd = (mean_acct - mean_peak) / mean_peak * 100
    fig = base_fig("D. 回撤 (% 距峰值) · 均值 + seed 范围", height=340)
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                  y=np.concatenate([dd_s.max(axis=0), dd_s.min(axis=0)[::-1]]),
                  fill="toself", fillcolor="rgba(214,39,40,0.12)",
                  line=dict(width=0), name="seed 范围", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=mean_dd, mode="lines",
                  line=dict(color=PALETTE["loss"], width=1.8), name="均值回撤"))
    fig.update_yaxes(title="回撤 (%)"); fig.update_xaxes(title="交易日")
    return fig


def _fig_cost(agg):
    T = agg["T"]; x = np.arange(T)
    comm = CAP0 * np.cumsum(agg["tc_comm_mean"])
    impact = CAP0 * np.cumsum(agg["tc_impact_mean"])
    fig = base_fig("E. 累计交易成本 — 佣金 vs 流动性冲击", height=340)
    fig.add_trace(go.Scatter(x=x, y=comm, mode="lines", fill="tozeroy",
                  line=dict(color=PALETTE["commission"], width=1.8),
                  fillcolor="rgba(37,99,235,0.10)", name="佣金+印花税"))
    fig.add_trace(go.Scatter(x=x, y=impact, mode="lines", fill="tozeroy",
                  line=dict(color=PALETTE["impact"], width=1.8),
                  fillcolor="rgba(194,65,12,0.10)", name="冲击成本 (√模型)"))
    fig.update_yaxes(title="累计成本 (¥)", tickformat=".2s")
    fig.update_xaxes(title="交易日")
    return fig


def _fig_monthly(agg):
    net = agg["mean_net"]; T = len(net)
    blocks = [np.prod(1 + net[i:i+21]) - 1.0 for i in range(0, T, 21)]
    blocks = np.array(blocks) * 100
    ncol = 6
    nrow = int(np.ceil(len(blocks) / ncol))
    grid = np.full((nrow, ncol), np.nan)
    for i, v in enumerate(blocks):
        grid[i // ncol, i % ncol] = v
    vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1
    fig = base_fig("F. ~月度收益 (21 日块, %)", height=340)
    fig.add_trace(go.Heatmap(z=grid, colorscale="RdYlGn", zmid=0, zmin=-vmax, zmax=vmax,
                  text=[[f"{v:+.1f}" if np.isfinite(v) else "" for v in row] for row in grid],
                  texttemplate="%{text}", textfont={"size": 10},
                  showscale=True, colorbar=dict(title="%")))
    fig.update_yaxes(title="", autorange="reversed",
                     tickvals=list(range(nrow)),
                     ticktext=[f"M{i*ncol+1}-{i*ncol+ncol}" for i in range(nrow)])
    fig.update_xaxes(visible=False)
    return fig


def _table(agg):
    rows = []
    for sd in agg["seeds"]:
        r = agg["runs"][sd]
        rows.append({
            "Seed": str(sd), "Sharpe": f"{r['sharpe']:.2f}",
            "Calmar": f"{r['calmar']:.2f}", "MaxDD": f"{r['maxdd']*100:.1f}%",
            "Alpha年化": f"{r['alpha_ann']*100:+.1f}%", "NW-t": f"{r['nw_t']:.2f}",
            "Beta": f"{r['beta']:.2f}", "胜率": f"{r['hit']*100:.0f}%",
            "佣金¥": money(r['cost_comm'] * CAP0), "冲击¥": money(r['cost_impact'] * CAP0),
        })
    rows.append({
        "Seed": "均值±SD", "Sharpe": f"{agg['sharpe_mean']:.2f}±{agg['sharpe_std']:.2f}",
        "Calmar": f"{agg['calmar_mean']:.2f}", "MaxDD": f"{agg['maxdd_mean']*100:.1f}%",
        "Alpha年化": f"{agg['alpha_ann_mean']*100:+.1f}%±{agg['alpha_ann_std']*100:.1f}",
        "NW-t": f"{agg['nw_t_mean']:.2f}", "Beta": f"{agg['beta_mean']:.2f}",
        "胜率": f"{agg['hit_mean']*100:.0f}%",
        "佣金¥": money(agg['cost_comm_mean'] * CAP0),
        "冲击¥": money(agg['cost_impact_mean'] * CAP0),
    })
    cols = list(rows[0].keys())
    return html.Div([
        html.Div("G. 各 seed 与均值汇总", style={"fontWeight": 600, "fontSize": 15,
                 "padding": "6px 4px 10px"}),
        dash_table.DataTable(
            data=rows, columns=[{"name": c, "id": c} for c in cols],
            style_cell={"textAlign": "center", "fontSize": 12.5, "padding": "7px",
                        "fontFamily": "Inter"},
            style_header={"backgroundColor": PALETTE["strat"], "color": "white",
                          "fontWeight": 700},
            style_data_conditional=[{
                "if": {"row_index": len(rows) - 1},
                "backgroundColor": "#dcfce7", "fontWeight": 700}],
        ),
    ])


def _kpi(val, pos_good=True, fmt="{:.2f}", sub=""):
    cls = "kpi-pos" if (val >= 0) == pos_good else "kpi-neg"
    return fmt.format(val), cls, sub


# --- callback --------------------------------------------------------------
def register(app):
    @app.callback(
        [Output("bl-equity", "figure"), Output("bl-alpha", "figure"),
         Output("bl-rollsharpe", "figure"), Output("bl-drawdown", "figure"),
         Output("bl-cost", "figure"), Output("bl-monthly", "figure"),
         Output("bl-table", "children"),
         Output("bl-kpi-sharpe", "children"), Output("bl-kpi-sharpe", "className"),
         Output("bl-kpi-alpha", "children"), Output("bl-kpi-alpha", "className"),
         Output("bl-kpi-nwt", "children"), Output("bl-kpi-nwt", "className"),
         Output("bl-kpi-beta", "children"),
         Output("bl-kpi-maxdd", "children"), Output("bl-kpi-maxdd", "className"),
         Output("bl-kpi-calmar", "children"),
         Output("bl-kpi-sharpe-sub", "children"),
         Output("bl-kpi-alpha-sub", "children"),
         Output("bl-kpi-nwt-sub", "children")],
        [Input("bl-run", "n_clicks"), Input("bl-topk", "value"),
         Input("bl-rebal", "value"), Input("bl-impactk", "value"),
         Input("bl-seeds", "value"), Input("bl-useimpact", "value")],
    )
    def _update(_n, topk, rebal, impactk, seeds, use_impact):
        seeds = tuple(sorted(seeds or _SEEDS))
        if not seeds:
            e = empty_fig("请至少选择一个 seed")
            return (e, e, e, e, e, e, html.Div(), "—", "kpi-value", "—",
                    "kpi-value", "—", "kpi-value", "—", "—", "kpi-value", "—",
                    "", "", "")
        agg = compute.multiseed_backtest(seeds, float(topk), int(rebal),
                                         float(impactk), bool(use_impact))
        if not agg.get("runs"):
            e = empty_fig("无数据")
            return (e, e, e, e, e, e, html.Div(), "—", "kpi-value", "—",
                    "kpi-value", "—", "kpi-value", "—", "—", "kpi-value", "—",
                    "", "", "")
        sh_v, sh_c, _ = _kpi(agg["sharpe_mean"], True, "{:.2f}")
        al_v = f"{agg['alpha_ann_mean']*100:+.1f}%"
        al_c = "kpi-pos" if agg["alpha_ann_mean"] >= 0 else "kpi-neg"
        nt_v = f"{agg['nw_t_mean']:.2f}"
        nt_c = "kpi-pos" if abs(agg["nw_t_mean"]) >= 2 else "kpi-neg"
        bt_v = f"{agg['beta_mean']:.2f}"
        dd_v = f"{agg['maxdd_mean']*100:.1f}%"
        cal_v = f"{agg['calmar_mean']:.2f}"
        sig = "显著 ✓ (≥2)" if abs(agg["nw_t_mean"]) >= 2 else "不显著 (<2)"
        return (_fig_equity(agg), _fig_alpha(agg), _fig_rollsharpe(agg),
                _fig_drawdown(agg), _fig_cost(agg), _fig_monthly(agg), _table(agg),
                sh_v, f"kpi-value {sh_c}", al_v, f"kpi-value {al_c}",
                nt_v, f"kpi-value {nt_c}", bt_v, dd_v, "kpi-value kpi-neg", cal_v,
                f"±{agg['sharpe_std']:.2f}", f"±{agg['alpha_ann_std']*100:.1f}%", sig)
