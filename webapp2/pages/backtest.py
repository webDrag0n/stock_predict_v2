"""回测：walk-forward 结果的查看与横向对比。

读 outputs/walkforward/（当前 canonical）与 outputs/experiments/**/walkforward/（各次实验）。
页面不重算回测，只展示 CLI 落盘的产物；唯一的例外是增量引擎的窗口 Sharpe：旧产物存档值
偏高（T17），所以按净值曲线复算并在不一致时明确标出（见 backtest_logic 顶部说明）。

轮询只更新独立的运行状态区。活动任务写产物时不重绘结果区，任务结束后才按文件签名更新；
产物下拉框只能手动刷新，轮询绝不写 options，避免关闭菜单或重置图表状态。
"""

from __future__ import annotations

import os

import pandas as pd
from dash import Input, Output, State, ctx, dcc, html, no_update

import run_registry as rr

from .. import actions, backtest_logic as bl, data, ui


def _runs_root() -> str:
    return os.path.join(data.OUTPUTS, "runs")


# ----------------------------------------------------------------------------- 签名 / 选项
def _files() -> list[str]:
    out = [os.path.join(data.OUTPUTS, "walkforward", "meta.json")]
    out += [os.path.join(r["dir"], "meta.json") for r in data.experiment_runs()]
    out += [
        os.path.join(data.OUTPUTS, "experiments", "trials.jsonl"),
        os.path.join(data.OUTPUTS, "experiments", "impact_sensitivity.csv"),
    ]
    return out


def signature() -> list[float | None]:
    """产物签名不包含 run.json：进度与心跳变化不得触发结果区重绘。"""
    return [data.mtime(p) for p in _files()]


def options() -> list[dict]:
    return [
        {
            "label": f"{r['label']} · {str(r['meta'].get('generated_at', ''))[:10]}",
            "value": r["dir"],
        }
        for r in data.experiment_runs()
    ]


def default_value() -> str | None:
    opts = options()
    return opts[0]["value"] if opts else None


# ----------------------------------------------------------------------------- 渲染
def _f(v, fmt="{:+.2f}", na="—") -> str:
    return ui.fnum(v, fmt, na)


def _kpis(meta: dict, wt: dict):
    m = meta.get("metrics") or {}
    cells = [
        ui.kpi(_f(m.get("sharpe")), "Sharpe（净）"),
        ui.kpi(ui.pct(m.get("ann_return")), "年化收益"),
        ui.kpi(
            ui.pct(m.get("max_drawdown")),
            "最大回撤",
            "bad" if (m.get("max_drawdown") or 0) < -0.15 else "",
        ),
        ui.kpi(_f(m.get("pred_ic"), "{:+.4f}"), "PredIC"),
        ui.kpi(_f(meta.get("nw_t")), "Newey–West t"),
        ui.kpi(ui.pct(meta.get("alpha_ann")), "年化 alpha"),
    ]
    return html.Div(cells, className="grid g6")


def _badges(meta: dict, wt: dict):
    items = [
        ui.badge(t, "done" if lv == "ok" else "failed")
        for t, lv in bl.trust_badges(meta, wt["pos_used"], wt["n"])
    ]
    return html.Div(items, className="row", style={"gap": "6px", "margin": "4px 0 8px"})


def _curve_card(meta: dict, curve: pd.DataFrame, bench: pd.DataFrame):
    if curve.empty:
        return ui.card("累计净值", ui.empty("没有 oos_curve.parquet"))
    f = ui.fig("OOS 累计净收益（相对初始资金）", height=300)
    x = pd.to_datetime(curve["date"])
    f.add_scatter(x=x, y=curve["equity"], name="策略", line=dict(color=ui.C["accent"], width=1.6))
    if not bench.empty and "bench_ret" in bench.columns:
        b = (1 + bench["bench_ret"].fillna(0.0)).cumprod() - 1
        f.add_scatter(
            x=pd.to_datetime(bench["date"]),
            y=b,
            name="全市场等权",
            line=dict(color="#94a3b8", dash="dash"),
        )
    cfg = meta.get("config") or {}
    ho = bl.holdout_rows(meta)
    if ho:
        import config as C

        if getattr(C, "HOLDOUT_START", ""):
            f.add_vline(x=pd.Timestamp(C.HOLDOUT_START), line_dash="dot", line_color=ui.C["warn"])
            f.add_annotation(
                x=pd.Timestamp(C.HOLDOUT_START),
                y=1,
                yref="paper",
                text="HOLDOUT 起点",
                showarrow=False,
                font=dict(size=10, color=ui.C["warn"]),
            )
    f.update_yaxes(tickformat=".0%")
    note = f"引擎 {cfg.get('engine', 'windowed')} · 初始资金 {ui.money(cfg.get('initial_capital'))}"
    return ui.card("累计净值", ui.graph(f), html.Div(note, className="hint"))


def _windows_card(wt: dict, meta: dict):
    df = wt["df"]
    if df is None or df.empty:
        return ui.card("窗口表", ui.empty("没有 window_metrics.parquet"))
    cols = [
        ("window", "窗口", ""),
        ("test", "测试区间", ""),
        ("n_feat", "特征", "n"),
        ("Sharpe", "Sharpe(存档)", "n"),
    ]
    if wt["recomputed"]:
        cols.append(("Sharpe_复算", "Sharpe(复算)", "n"))
    cols += [("MaxDD", "最大回撤", "n"), ("PredIC", "PredIC", "n")]
    show = df.copy()
    for c in ("Sharpe", "Sharpe_复算", "MaxDD", "PredIC"):
        if c in show.columns:
            show[c] = show[c].map(
                lambda v: (
                    "—" if pd.isna(v) else f"{v:+.2f}" if c.startswith("Sharpe") else f"{v:+.3f}"
                )
            )
    warn = None
    if wt["stale"]:
        warn = html.Div(
            f"⚠ 存档的窗口 Sharpe 与按净值曲线复算的不一致：存档 {wt['pos_stored']}/{wt['n']} 个窗口为正，"
            f"复算只有 {wt['pos_used']}/{wt['n']}。这份产物生成于 T17（窗口基准资金错误）修复之前，"
            "页面徽章与结论一律采用复算值；重跑一次即可得到修正后的存档。",
            className="warn",
            style={"margin": "6px 0"},
        )
    elif wt["recomputed"]:
        warn = html.Div("存档与复算的窗口 Sharpe 一致。", className="hint")
    elif not bl.is_incremental(meta):
        warn = html.Div(
            "滚动窗口引擎：每个窗口各自从空仓独立重放，窗口指标与上方拼接曲线的分段不会逐段相等"
            "（持仓不跨窗口延续）。头条 Sharpe 来自拼接后的连续重放。",
            className="hint",
            style={"margin": "6px 0"},
        )
    return ui.card("窗口表", warn, ui.table(show, cols, max_rows=40))


def _holdout_card(meta: dict):
    rows = bl.holdout_rows(meta)
    dsr = bl.dsr_note(meta)
    body = []
    if rows:
        df = pd.DataFrame(rows)
        df["sharpe"] = df["sharpe"].map(lambda v: _f(v))
        df["pred_ic"] = df["pred_ic"].map(lambda v: _f(v, "{:+.4f}"))
        body.append(
            ui.table(
                df,
                [
                    ("seg", "分段", ""),
                    ("n_days", "天数", "n"),
                    ("sharpe", "Sharpe", "n"),
                    ("pred_ic", "PredIC", "n"),
                ],
            )
        )
        body.append(
            html.Div(
                "HOLDOUT 只用于最终报告，不参与训练或选型；选型一律只看它之前。", className="hint"
            )
        )
    else:
        body.append(ui.empty("这份产物没有 HOLDOUT 分段（生成于 HOLDOUT 功能之前，重跑即有）。"))
    if dsr:
        body.append(
            html.Div(
                dsr[0], className="ok" if dsr[1] == "ok" else "warn", style={"marginTop": "6px"}
            )
        )
    else:
        body.append(
            html.Div(
                "没有 Deflated Sharpe（旧产物）。", className="hint", style={"marginTop": "6px"}
            )
        )
    return ui.card("HOLDOUT 与多重检验", *body)


def _compare_card():
    rows = bl.compare_rows(data.experiment_runs())
    if not rows:
        return ui.card("实验横向对比", ui.empty("没有实验产物。"))
    df = pd.DataFrame(rows)
    df["sharpe"] = df["sharpe"].map(lambda v: _f(v))
    df["ann_return"] = df["ann_return"].map(lambda v: ui.pct(v))
    df["max_drawdown"] = df["max_drawdown"].map(lambda v: ui.pct(v))
    df["pred_ic"] = df["pred_ic"].map(lambda v: _f(v, "{:+.4f}"))
    df["nw_t"] = df["nw_t"].map(lambda v: _f(v))
    df["pos"] = [f"{p}/{w}" if p is not None else "—" for p, w in zip(df["pos"], df["windows"])]
    df["capital"] = df["capital"].map(lambda v: ui.money(v))
    df["flag"] = ["复算" if s else "" for s in df["stale"]]
    cols = [
        ("label", "实验", ""),
        ("engine", "引擎", ""),
        ("capital", "本金", "n"),
        ("sharpe", "Sharpe", "n"),
        ("ann_return", "年化", "n"),
        ("max_drawdown", "回撤", "n"),
        ("pred_ic", "PredIC", "n"),
        ("nw_t", "NW-t", "n"),
        ("pos", "正窗口", "n"),
        ("flag", "", ""),
        ("oos", "OOS 区间", ""),
        ("generated_at", "生成日", ""),
    ]
    notes = [html.Div(bl.noise_note(), className="hint")]
    if bl.snapshot_mismatch(rows):
        notes.append(
            html.Div(
                "⚠ 这些实验的 OOS 区间不同（数据快照不同）：差异里混着快照的影响，不是纯粹的方法差异。",
                className="warn",
            )
        )
    if any(r["stale"] for r in rows):
        notes.append(
            html.Div(
                "「复算」= 该行正窗口数已按净值曲线复算（存档值偏高，见窗口表说明）。",
                className="hint",
            )
        )
    return ui.card("实验横向对比", ui.table(df, cols), *notes)


def _impact_card():
    df = data.impact_sensitivity()
    if df.empty:
        return ui.card("冲击成本敏感性", ui.empty("没有 impact_sensitivity.csv。"))
    f = ui.fig("Sharpe 随冲击系数 K 的变化", height=240)
    f.add_scatter(
        x=df["k"],
        y=df["sharpe"],
        mode="lines+markers",
        name="Sharpe",
        line=dict(color=ui.C["accent"]),
    )
    f.update_xaxes(title="K")
    show = df.copy()
    for c in ("sharpe",):
        show[c] = show[c].map(lambda v: f"{v:+.2f}")
    for c in ("ann_return", "max_drawdown", "total_cost"):
        show[c] = show[c].map(lambda v: ui.pct(v))
    return ui.card(
        "冲击成本敏感性",
        ui.graph(f),
        ui.table(
            show,
            [
                ("k", "K", "n"),
                ("sharpe", "Sharpe", "n"),
                ("ann_return", "年化", "n"),
                ("max_drawdown", "回撤", "n"),
                ("total_cost", "总成本", "n"),
            ],
        ),
        html.Div("K 越大冲击成本越高；K=0.1 是默认口径。", className="hint"),
    )


def _trials_card():
    rows = [t for t in data.trials(200) if t.get("kind") == "walkforward"]
    if not rows:
        return ui.card("试验记录", ui.empty("还没有 walk-forward 试验记录。"))
    out = []
    for t in reversed(rows[-15:]):
        m, c = t.get("metrics") or {}, t.get("config") or {}
        out.append(
            {
                "ts": str(t.get("ts", ""))[:16],
                "engine": c.get("engine"),
                "obj": c.get("model_objective"),
                "sharpe": _f(m.get("sharpe")),
                "hash": t.get("config_hash"),
            }
        )
    return ui.card(
        f"试验记录（walk-forward，共 {len(rows)} 条）",
        ui.table(
            pd.DataFrame(out),
            [
                ("ts", "时间", ""),
                ("engine", "引擎", ""),
                ("obj", "目标", ""),
                ("sharpe", "Sharpe", "n"),
                ("hash", "配置哈希", ""),
            ],
        ),
        html.Div(
            "试验次数进入 Deflated Sharpe 的多重检验折扣；相同配置哈希只算一次。", className="hint"
        ),
    )


def detail(run_dir: str | None):
    if not run_dir:
        return ui.empty("还没有任何 walk-forward 产物。点上方「启动 walk-forward」生成。")
    meta = data.jsonf(os.path.join(run_dir, "meta.json"))
    if not isinstance(meta, dict):
        return ui.empty("所选产物不存在或已损坏。")
    curve = data.walkforward_curve(os.path.join(run_dir, "oos_curve.parquet"))
    wins = data.walkforward_windows(os.path.join(run_dir, "window_metrics.parquet"))
    bench = data.parquet(os.path.join(run_dir, "oos_bench.parquet"))
    wt = bl.window_table(meta, wins, curve)
    uni = meta.get("universe") or {}
    sub = f"生成 {str(meta.get('generated_at', ''))[:16]} · {uni.get('n_symbols', '?')} 只 · OOS {str(uni.get('oos_start', ''))[:10]} ~ {str(uni.get('oos_end', ''))[:10]} · {meta.get('n_windows', '?')} 个窗口"
    return html.Div(
        [
            html.Div(sub, className="hint"),
            _badges(meta, wt),
            _kpis(meta, wt),
            html.Div([_curve_card(meta, curve, bench), _holdout_card(meta)], className="grid g2"),
            _windows_card(wt, meta),
        ]
    )


def lower_cards():
    return html.Div(
        [_compare_card(), html.Div([_impact_card(), _trials_card()], className="grid g2")]
    )


# ----------------------------------------------------------------------------- 轮询 / 骨架
def _active_walkforward() -> list[dict]:
    return [m for m in rr.active_runs(root=_runs_root()) if m.get("kind") == "walkforward"]


def _run_status():
    """唯一随每次轮询更新的区域，不包含下拉框、图表或可展开内容。"""
    active = _active_walkforward()
    if active:
        a = active[0]
        p = a.get("progress") or {}
        frac = p.get("pct") if p.get("total") else None
        count = f"{p.get('current', 0):g}/{p.get('total', 0):g}" if p.get("total") else "准备中"
        pct = ui.pct(frac, 0) if frac is not None else "—"
        return html.Div(
            [
                html.Div(
                    [
                        html.B("正在运行 walk-forward"),
                        html.Span(f"  {a['run_id']}", className="faint"),
                    ]
                ),
                ui.bar(frac),
                html.Div(
                    f"{p.get('phase', '启动中')} · {p.get('message', '')} · {count}（{pct}）",
                    className="hint",
                ),
                html.Div(
                    [
                        dcc.Link("查看实时进度 →", href=f"/runs/{a['run_id']}"),
                        html.Span("  结果区将在任务结束后更新", className="hint"),
                    ]
                ),
            ]
        )
    last = rr.list_runs(kind="walkforward", limit=1, root=_runs_root())
    if not last:
        return html.Div("还没有 walk-forward 运行记录。", className="hint")
    m = last[0]
    st = {"done": "完成", "failed": "失败", "killed": "已终止"}.get(m["status"], m["status"])
    if m["status"] == "running":
        st = "疑似卡住" if m.get("liveness") == "stale" else "已中断"
    return html.Div(
        [
            "最近一次 walk-forward：",
            ui.badge(st, {"完成": "done", "失败": "failed"}.get(st, "dead")),
            html.Span(f"  {ui.ago(m.get('started'))}  ", className="muted"),
            dcc.Link("详情", href=f"/runs/{m['run_id']}"),
        ]
    )


def refresh_decision(stored, run_dir: str | None, active: bool | None = None):
    """活动任务期间保留全部结果 DOM；无活动任务且产物变化时才返回新内容。"""
    sig = signature()
    if active is None:
        active = bool(_active_walkforward())
    if active or sig == stored:
        return None, None, stored
    return detail(run_dir), lower_cards(), sig


def layout():
    val = default_value()
    return ui.page(
        "BACKTEST",
        "回测",
        "walk-forward 样本外结果。增量引擎的窗口指标按净值曲线复算，不一致会明确标出。",
        dcc.Interval(id="bt-tick", interval=5000),
        dcc.Store(id="bt-sig", data=signature()),
        ui.card(
            "选择产物",
            html.Div(
                [
                    html.Div(
                        dcc.Dropdown(id="bt-run", options=options(), value=val, clearable=False),
                        style={"minWidth": "420px"},
                    ),
                    html.Button("刷新产物列表", id="bt-refresh", className="btn"),
                    html.Button(
                        "启动 walk-forward（增量）", id="bt-launch-inc", className="btn primary"
                    ),
                    html.Button(
                        "启动 walk-forward（滚动窗口）", id="bt-launch-win", className="btn"
                    ),
                ],
                className="row",
            ),
            html.Div(id="bt-launch-msg", className="msg"),
            html.Div(_run_status(), id="bt-run-status"),
            html.Div("长任务（数小时）；与终端运行同一条命令，关闭本页不影响。", className="hint"),
        ),
        html.Div(detail(val), id="bt-detail"),
        html.Div(lower_cards(), id="bt-lower"),
    )


def register(app):
    @app.callback(
        Output("bt-detail", "children"),
        Output("bt-lower", "children"),
        Output("bt-run", "options"),
        Output("bt-run-status", "children"),
        Output("bt-sig", "data"),
        Input("bt-run", "value"),
        Input("bt-tick", "n_intervals"),
        Input("bt-refresh", "n_clicks"),
        State("bt-sig", "data"),
        prevent_initial_call=True,
    )
    def _render(value, _n, _refresh, stored):
        trig = ctx.triggered_id
        if trig == "bt-refresh":
            return no_update, no_update, options(), _run_status(), no_update
        if trig != "bt-tick":
            return detail(value), no_update, no_update, no_update, no_update
        new_detail, new_lower, sig = refresh_decision(stored, value)
        return (
            new_detail if new_detail is not None else no_update,
            new_lower if new_lower is not None else no_update,
            no_update,
            _run_status(),
            sig,
        )

    @app.callback(
        Output("bt-launch-msg", "children"),
        Input("bt-launch-inc", "n_clicks"),
        Input("bt-launch-win", "n_clicks"),
        prevent_initial_call=True,
    )
    def _launch(n_inc, n_win):
        trig = ctx.triggered_id
        if not ((trig == "bt-launch-inc" and n_inc) or (trig == "bt-launch-win" and n_win)):
            return no_update
        args = ["--engine", "incremental" if trig == "bt-launch-inc" else "windowed"]
        ok, msg, rid = actions.launch_cli("walkforward", args)
        return html.Span(
            [msg + " ", dcc.Link("查看进度", href=f"/runs/{rid}") if rid else ""],
            className="ok" if ok else "bad",
        )
