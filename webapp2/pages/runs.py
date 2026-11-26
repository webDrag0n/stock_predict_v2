"""运行与进度：新契约（outputs/runs/）的主要消费者。

两个视图，同一页：/runs 列表；/runs/<id> 详情。详情页对「正在运行」的任务每 2 秒刷新，
终态后自动停止轮询。CLI 与本页互不依赖：没有 webapp 时 CLI 照常写 run.json，
webapp 后启动也能立刻看到历史与进行中的运行。
"""

from __future__ import annotations

import os

from dash import Input, Output, State, dcc, html, no_update

import run_registry as rr

from .. import actions, data, report_view, ui

KIND_LABEL = {
    "nightly": "夜间出单",
    "walkforward": "Walk-forward",
    "tracking": "偏差跟踪",
    "commit": "成交确认",
    "promote": "人工晋升",
    "impact": "冲击敏感性",
    "train": "训练候选",
    "modeleval": "信号评估",
    "factoreval": "因子评估",
}


def _root() -> str:
    return os.path.join(data.OUTPUTS, "runs")


def _live_badge(meta: dict):
    st, lv = meta.get("status"), meta.get("liveness")
    if st == "running" and lv == "dead":
        return ui.badge("已中断", "dead")
    if st == "running" and lv == "stale":
        return ui.badge("疑似卡住", "stale")
    return ui.badge(
        {"running": "运行中", "done": "完成", "failed": "失败", "killed": "已终止"}.get(st, st), st
    )


def _elapsed(meta: dict) -> float:
    end = meta.get("finished") or (
        meta.get("heartbeat") if meta.get("liveness") == "dead" else __import__("time").time()
    )
    return max(0.0, (end or 0) - (meta.get("started") or meta.get("created") or 0))


# ----------------------------------------------------------------------------- 列表
def list_view():
    runs = rr.list_runs(limit=60, root=_root())
    if not runs:
        return ui.page(
            "RUNS",
            "运行",
            "所有 CLI 运行的统一记录",
            # 空列表正是第一次使用的场景，启动区必须在
            _launch_card(),
            ui.card(
                "",
                ui.empty(
                    "还没有运行记录。运行 predict_tomorrow.py 或 scripts/run_walkforward.py 后这里会出现。"
                ),
            ),
        )
    rows = []
    for m in runs:
        prog = m.get("progress") or {}
        cur = next((s for s in m.get("steps", []) if s.get("status") == "running"), None)
        detail = ""
        if m.get("status") == "running" and m.get("liveness") in ("alive", "stale"):
            detail = (cur or {}).get("title", "") + (
                f" · {prog['phase']}" if prog.get("phase") else ""
            )
        elif m.get("status") == "failed":
            detail = (m.get("failure") or {}).get("exc_msg", "") or m.get("summary", "")
        else:
            detail = m.get("summary", "") or ""
        rows.append(
            html.Tr(
                [
                    html.Td(dcc.Link(m["run_id"], href=f"/runs/{m['run_id']}")),
                    html.Td(KIND_LABEL.get(m.get("kind"), m.get("kind"))),
                    html.Td(_live_badge(m)),
                    html.Td(ui.clock(m.get("started") or m.get("created")), className="n"),
                    html.Td(ui.dur(_elapsed(m)), className="n"),
                    html.Td(detail[:90], className="muted"),
                ]
            )
        )
    head = html.Tr([html.Th(h) for h in ("运行", "类型", "状态", "开始", "耗时", "说明")])
    return ui.page(
        "RUNS",
        "运行",
        "所有 CLI 运行的统一记录。数据来自 outputs/runs/，不依赖本页服务。",
        _launch_card(),
        ui.card("", html.Table([html.Thead(head), html.Tbody(rows)], className="t")),
    )


def _launch_card():
    return ui.card(
        "从这里启动（等价于在终端运行同一条命令）",
        html.Div(
            [
                html.Button("生成次日订单", id="launch-nightly", className="btn primary"),
                html.Button("偏差跟踪报告", id="launch-tracking", className="btn"),
                html.Button("walk-forward（增量）", id="launch-wf", className="btn"),
            ],
            className="row",
        ),
        html.Div(id="launch-msg", className="msg"),
        html.Div("CLI 自己登记运行，关闭本页不影响运行。", className="hint"),
    )


# ----------------------------------------------------------------------------- 详情
def _progress_block(meta: dict):
    if meta.get("status") != "running":
        return None
    p = meta.get("progress") or {}
    steps = meta.get("steps") or []
    done = sum(s.get("status") in ("done", "skipped") for s in steps)
    current_step = next((s for s in steps if s.get("status") == "running"), None)
    frac = p.get("pct") if p.get("total") else None
    phase_progress = (
        f"{p.get('current', 0):g}/{p.get('total', 0):g}（{ui.pct(frac, 0)}）"
        if frac is not None
        else "等待进度上报"
    )
    eta = f" · 阶段预计剩余 {ui.dur(p['eta_sec'])}" if p.get("eta_sec") else ""
    phase_elapsed = (
        f" · 本阶段耗时 {ui.dur(p['elapsed_sec'])}" if p.get("elapsed_sec") is not None else ""
    )
    extras = p.get("extras") or {}
    ex = " · ".join(
        f"{k}={ui.fnum(v, '{:.3f}') if isinstance(v, float) else v}"
        for k, v in list(extras.items())[:4]
    )
    phase = p.get("phase") or (current_step or {}).get("title") or "等待进度上报"
    return ui.card(
        "当前进度",
        html.Div([html.B(phase), html.Span(f"  {p.get('message', '')}", className="muted")]),
        html.Div(f"当前步骤：{(current_step or {}).get('title', '准备中')}", className="hint"),
        ui.bar(frac),
        html.Div(
            f"任务步骤 {done}/{len(steps)} · 阶段进度 {phase_progress}"
            f" · 已运行 {ui.dur(_elapsed(meta))}{phase_elapsed}{eta}" + (f" · {ex}" if ex else ""),
            className="hint num",
        ),
        tone="hot",
    )


def _steps_block(meta: dict):
    steps = meta.get("steps") or []
    if not steps:
        return ui.card("步骤", ui.empty("此运行没有登记步骤"))
    items = []
    for s in steps:
        st = s.get("status", "pending")
        items.append(
            html.Div(
                [
                    html.Div(className="mark"),
                    html.Div(
                        [
                            html.Div(s.get("title") or s["id"], className="tt"),
                            html.Div(s.get("summary", ""), className="ss"),
                        ]
                    ),
                    html.Div(
                        ui.dur(s["elapsed"]) if s.get("elapsed") is not None else "", className="el"
                    ),
                ],
                className=f"step {st}",
            )
        )
    return ui.card("步骤", html.Div(items, className="steps"))


def _events_block(run_id: str):
    ev = rr.read_events(run_id, tail=80, root=_root())
    if not ev:
        return ui.card("事件", ui.empty("暂无事件"))
    lines = []
    for e in ev[::-1]:
        t = e.get("type")
        if t == "note":
            cls, text = e.get("level", "info"), e.get("message", "")
        elif t == "error":
            cls, text = "error", f"{e.get('exc_type')}: {e.get('exc_msg')}"
        elif t == "step_start":
            cls, text = "info", f"▶ {e.get('title') or e.get('step')}"
        elif t == "step_done":
            cls, text = "info", f"✔ {e.get('step')} {e.get('summary', '')}"
        elif t == "artifact":
            cls, text = "info", f"产物 {e.get('name')} → {e.get('path')}"
        else:
            cls, text = "info", f"{t} {e.get('status', '')}".strip()
        lines.append(
            html.Div(
                [
                    html.Span(ui.clock(e.get("ts")) + "  ", className="ts"),
                    html.Span(text, className=cls),
                ]
            )
        )
    return ui.card("事件", html.Div(lines, className="events"))


def _failure_block(meta: dict):
    f = meta.get("failure")
    if not f:
        return None
    return ui.card(
        "失败",
        html.Div([html.B(f"{f.get('exc_type')}: "), html.Span(f.get("exc_msg", ""))]),
        html.Div(f"步骤 {f.get('step')} · 阶段 {f.get('phase') or '—'}", className="hint"),
        (
            html.Details(
                [html.Summary("堆栈"), html.Pre(f.get("traceback") or "（无）", className="log")],
                className="d",
            )
            if f.get("traceback")
            else None
        ),
        tone="bad",
    )


def _artifacts_block(meta: dict):
    arts = meta.get("artifacts") or []
    if not arts:
        return None
    rows = []
    for a in arts:
        exists = os.path.exists(os.path.join(data.ROOT, a["path"]))
        extra = "  ".join(f"{k}={v}" for k, v in a.items() if k not in ("name", "path", "kind"))
        rows.append(
            html.Tr(
                [
                    html.Td(a["name"]),
                    html.Td(a["path"], className="mono"),
                    html.Td(ui.badge("存在" if exists else "缺失", "ok" if exists else "warn")),
                    html.Td(extra, className="muted"),
                ]
            )
        )
    return ui.card(
        "产物",
        html.Table(
            [
                html.Thead(html.Tr([html.Th(h) for h in ("名称", "路径", "状态", "说明")])),
                html.Tbody(rows),
            ],
            className="t",
        ),
    )


def _result_block(meta: dict):
    res = {
        k: v
        for k, v in (meta.get("params") or {}).items()
        if k.startswith("result_") and isinstance(v, dict)
    }
    if not res:
        return None
    cards = []
    for k, v in res.items():
        cards.append(
            ui.card(
                k.replace("result_", ""),
                ui.kv_table(
                    [
                        ("Sharpe", ui.fnum(v.get("sharpe"), "{:.2f}")),
                        ("PredIC", ui.fnum(v.get("pred_ic"), "{:.4f}")),
                        ("NW-t", ui.fnum(v.get("nw_t"), "{:.2f}")),
                        ("窗口数", v.get("n_windows") or "—"),
                        ("为正窗口占比", ui.pct(v.get("sharpe_pos_ratio"), 0)),
                    ]
                ),
            )
        )
    return html.Div(cards, className="grid g3")


def detail_view(run_id: str):
    meta = rr.read_run(run_id, root=_root())
    if meta is None:
        return ui.page(
            "RUN",
            "找不到该运行",
            run_id,
            ui.card(
                "", ui.empty("run.json 不存在或已损坏。"), dcc.Link("← 返回运行列表", href="/runs")
            ),
        )
    live = meta.get("status") == "running" and meta.get("liveness") in ("alive", "stale")
    warn = None
    if meta.get("status") == "running" and meta.get("liveness") == "dead":
        warn = ui.card(
            "",
            html.B("进程已不存在，但没有写入终态。"),
            html.Div(
                f"pid {meta.get('pid')} 已退出，最后心跳 {ui.ago(meta.get('heartbeat'))}。"
                "通常是被 kill -9 或机器重启。",
                className="hint",
            ),
            tone="bad",
        )
    elif meta.get("liveness") == "stale":
        warn = ui.card("", html.B("心跳超过 3 分钟没有更新，疑似卡住。"), tone="warn")
    head = html.Div(
        [
            _live_badge(meta),
            html.Span(f"  {KIND_LABEL.get(meta.get('kind'), meta.get('kind'))}", className="muted"),
            html.Span(
                f"  pid {meta.get('pid')} · 开始 {ui.clock(meta.get('started'))} · 用时 {ui.dur(_elapsed(meta))}",
                className="hint num",
            ),
            (
                html.Button(
                    "终止", id="stop-run", className="btn danger", style={"marginLeft": "14px"}
                )
                if live
                else None
            ),
        ],
    )
    params = {k: v for k, v in (meta.get("params") or {}).items() if not k.startswith("result_")}
    return ui.page(
        "RUN",
        meta.get("label") or run_id,
        run_id,
        dcc.Link("← 返回运行列表", href="/runs"),
        html.Div(head, style={"margin": "10px 0"}),
        html.Div(id="stop-msg", className="msg"),
        warn,
        _failure_block(meta),
        _progress_block(meta),
        _result_block(meta),
        html.Div(
            [_steps_block(meta), _artifacts_block(meta) or ui.card("产物", ui.empty("暂无产物"))],
            className="grid g2",
        ),
        report_view.render_reports(report_view.read_reports(run_id, _root())),
        _events_block(run_id),
        html.Details(
            [html.Summary("参数"), ui.kv_table([(k, str(v)[:140]) for k, v in params.items()])],
            className="d",
        ),
        html.Details(
            [
                html.Summary("原始日志（末尾）"),
                html.Pre(
                    rr.read_log_tail(run_id, 150, root=_root()) or "（无 run.log）", className="log"
                ),
            ],
            className="d",
        ),
        dcc.Store(id="run-id", data=run_id),
    )


def _rid_of(pathname: str | None) -> str | None:
    p = pathname or ""
    return p.split("/runs/", 1)[1].strip("/") if p.startswith("/runs/") else None


def _body(pathname: str | None):
    rid = _rid_of(pathname)
    return detail_view(rid) if rid else list_view()


def layout(pathname: str = "/runs"):
    """外壳固定：轮询组件、路径 Store 与按钮不随刷新重建；只有 runs-body 的内容被替换。

    这样计时器不会因重绘而重置，用户在启动区填的内容也不会被清掉。
    stop-run 与三个 launch 按钮始终存在于外壳，避免 Dash 回调因输入组件缺失而报错。
    """
    return html.Div(
        [
            dcc.Interval(id="runs-tick", interval=2000),
            dcc.Store(id="runs-path", data=pathname),
            html.Div(_body(pathname), id="runs-body"),
        ]
    )


def register(app):
    @app.callback(
        Output("runs-body", "children"),
        Input("runs-tick", "n_intervals"),
        Input("runs-path", "data"),
        prevent_initial_call=True,
    )
    def _refresh(_n, pathname):
        return _body(pathname)

    @app.callback(
        Output("launch-msg", "children"),
        Input("launch-nightly", "n_clicks"),
        Input("launch-tracking", "n_clicks"),
        Input("launch-wf", "n_clicks"),
        prevent_initial_call=True,
    )
    def _launch(n1, n2, n3):
        from dash import ctx

        spec = {
            "launch-nightly": ("nightly", actions.NIGHTLY_PRESET),
            "launch-tracking": ("tracking", []),
            "launch-wf": ("walkforward", ["--engine", "incremental"]),
        }.get(ctx.triggered_id)
        if spec is None or not any((n1, n2, n3)):
            return no_update
        ok, msg, rid = actions.launch_cli(*spec)
        return html.Span(
            [msg + " ", dcc.Link("查看", href=f"/runs/{rid}") if rid else ""],
            className="ok" if ok else "bad",
        )

    @app.callback(
        Output("stop-msg", "children"),
        Input("stop-run", "n_clicks"),
        State("run-id", "data"),
        prevent_initial_call=True,
    )
    def _stop(n, rid):
        if not n:
            return no_update
        ok, msg = actions.stop_run(rid)
        return html.Span(msg, className="ok" if ok else "bad")
