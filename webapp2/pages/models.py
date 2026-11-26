"""模型：champion / staging / archive 的现状、两模型对比、晋升与删除。

骨架分层（与交易页同一套路）：
  · 静态区（永不被轮询替换）：对比下拉框、确认弹层（理由输入框）。
  · models-body：只读内容 + 带 id 的「晋升/删除」按钮；仅当磁盘签名变化才重绘，
    避免用户展开的配方详情、正在点的按钮被定时刷新打断。
晋升与删除都要二次确认；未达门控或评估可疑的候选，晋升必须填写理由（写入审计）。
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

import run_registry as rr

from .. import actions, data, model_logic as ml, ui

_AGE_WARN, _AGE_BAD = 48, 60


def _runs_root() -> str:
    return os.path.join(data.OUTPUTS, "runs")


# ----------------------------------------------------------------------------- 签名
def signature() -> list:
    busy = [m["run_id"] for m in rr.active_runs(root=_runs_root()) if m.get("kind") == "nightly"]
    return [
        data.models_signature(),
        busy,
        data.mtime(os.path.join(data.OUTPUTS, "tracking", "tracking_report.json")),
    ]


def refresh_decision(stored) -> tuple[object | None, list]:
    sig = signature()
    return (None, sig) if stored == sig else (body(), sig)


# ----------------------------------------------------------------------------- 渲染
def _fmt(v, key: str = "") -> str:
    if not ml.number(v):
        return "—" if v is None else str(v)
    if key in ("n_days", "n_features"):
        return f"{int(v):,}"
    if key == "hit_rate":
        return f"{v:.1%}" if abs(v) <= 1 else f"{v:.1f}%"
    return f"{v:+.2f}" if abs(v) < 100 else f"{v:,.0f}"


def _age_days() -> int | None:
    return ((data.tracking_report() or {}).get("champion") or {}).get("age_days")


def _guidance_bar():
    g = ml.guidance(data.champion_manifest(), data.staging_models())
    return html.Div(g, className="hint", style={"marginBottom": "6px"})


def _champion_card():
    man = data.champion_manifest()
    if not man:
        return ui.card(
            "champion（线上模型）",
            ui.empty("还没有 champion。运行一次夜间任务会自动 bootstrap，或在下方晋升一个候选。"),
            tone="bad",
        )
    age = _age_days()
    age_txt = "—" if age is None else f"{age} / {_AGE_BAD} 交易日"
    tone = "bad" if (age or 0) > _AGE_BAD else "warn" if (age or 0) > _AGE_WARN else ""
    metrics, src = ml.champion_metrics(man, data.champion_evaluation())
    promo = man.get("promotion") or {}
    rows = [
        ("model_id", man.get("model_id")),
        ("指纹", man.get("fingerprint", "—")),
        ("训练截止", man.get("train_end_date", "—")),
        ("年龄", age_txt),
        ("晋升时间", promo.get("decided_at", "—")),
    ]
    mrows = [
        (label, _fmt(metrics.get(k), k))
        for k, label, _ in ml.METRIC_KEYS
        if metrics.get(k) is not None
    ]
    return ui.card(
        "champion（线上模型）",
        ui.kv_table(rows),
        html.Div(ml.recipe_summary(man.get("config_snapshot") or {}), className="hint"),
        html.Div(
            [
                html.Div("晋升理由", className="why-sec"),
                html.Div(promo.get("reason", "—"), className="why-line"),
            ],
            style={"marginTop": "8px"},
        ),
        html.Div(f"指标来源：{src}", className="hint", style={"marginTop": "8px"}),
        ui.kv_table(mrows) if mrows else html.Div("没有可用的评估指标。", className="muted"),
        html.Details(
            [
                html.Summary("全量配方 JSON"),
                html.Pre(
                    json.dumps(man.get("config_snapshot") or {}, ensure_ascii=False, indent=2),
                    className="log",
                ),
            ],
            className="d",
        ),
        tone=tone,
    )


def _staging_item(e: dict):
    s = ml.staging_summary(e)
    if s["is_champion"]:
        verdict = [
            ui.badge("当前 champion", "done"),
            html.Span("  晋升后留档的副本，不是待决策的候选", className="muted"),
        ]
    elif not s["has_eval"]:
        verdict = [
            ui.badge("无评估", "pending"),
            html.Span("  evaluation.json 缺失", className="muted"),
        ]
    else:
        ds = ui.fnum(s["d_sharpe"], "{:+.2f}")
        dd = ui.fnum(s["d_dd"], "{:+.1f}pp")
        verdict = [
            ui.badge(
                "达门槛" if s["eligible"] else "未达门槛",
                "done" if s["eligible"] else "warn",
            ),
            html.Span(f"  ΔSharpe {ds} · Δ回撤 {dd}", className="num"),
        ]
        if s["eval_days"]:
            win = f"{s['window'][0]} ~ {s['window'][1]}" if s["window"] else ""
            verdict.append(html.Span(f"  · {s['eval_days']} 日 {win}", className="muted"))
    extra = []
    if s["suspect"] and not s["is_champion"]:
        extra.append(
            html.Div("⚠ 两侧指标逐项相同，这次评估不可信（同一套特征重放两次）", className="warn")
        )
    if s["missing_cols"]:
        extra.append(
            html.Div(f"⚠ champion 有 {len(s['missing_cols'])} 个特征列缺失", className="warn")
        )
    buttons = [html.Button("删除", id={"type": "m-delete", "id": s["id"]}, className="btn danger")]
    if not s["is_champion"]:
        buttons.insert(
            0,
            html.Button("晋升", id={"type": "m-promote", "id": s["id"]}, className="btn primary"),
        )
    return html.Div(
        [
            html.Div([html.Code(s["id"]), html.Div(verdict, style={"marginTop": "2px"})]),
            html.Div(buttons, className="row", style={"gap": "6px", "flexWrap": "nowrap"}),
            html.Div(
                [html.Div(s["reason"], className="hint"), *extra],
                style={"gridColumn": "1 / -1"},
            ),
        ],
        className="srow",
    )


def _staging_card():
    items = data.staging_models()
    if not items:
        return ui.card("staging 候选", ui.empty("没有候选。夜间门控未通过的新配方会留档在这里。"))
    pending = len(ml.pending_candidates(items))
    return ui.card(
        f"staging 候选（待决策 {pending} / 共 {len(items)}）",
        *[_staging_item(e) for e in items],
        html.Div(
            "晋升：替换 champion，旧 champion 归档到 models/archive/ 可回滚。"
            "删除：永久删除留档目录。两者都会二次确认并写入 models/promotions.log。",
            className="hint",
            style={"marginTop": "8px"},
        ),
    )


def _archive_card():
    items = [e for e in data.archived_models() if e["manifest"]]
    if not items:
        return ui.card("archive（已退役）", ui.empty("没有已退役的模型。"))
    rows = []
    for e in items:
        m = e["manifest"]
        promo = m.get("promotion") or {}
        rows.append(
            {
                "id": e["id"],
                "train_end": m.get("train_end_date", "—"),
                "promoted": str(promo.get("decided_at", "—"))[:10],
                "recipe": ml.recipe_summary(m.get("config_snapshot") or {})[:70],
            }
        )
    return ui.card(
        f"archive（已退役，{len(items)}）",
        ui.table(
            pd.DataFrame(rows),
            [
                ("id", "model_id", ""),
                ("train_end", "训练截止", ""),
                ("promoted", "晋升日", ""),
                ("recipe", "配方", ""),
            ],
            max_rows=30,
        ),
        html.Div("可在下方对比区选择 archive 模型与现任对比。", className="hint"),
    )


def _audit_card():
    lines = data.promotion_log(12)
    if not lines:
        return ui.card("操作审计", ui.empty("还没有晋升/删除记录。"))
    return ui.card(
        "操作审计（models/promotions.log）",
        html.Pre("\n".join(lines), className="log"),
    )


def body():
    return html.Div(
        [
            _guidance_bar(),
            html.Div([_champion_card(), _staging_card()], className="grid g2"),
            html.Div([_archive_card(), _audit_card()], className="grid g2"),
        ]
    )


# ----------------------------------------------------------------------------- 对比
def compare_view(a: str | None, b: str | None):
    if not a or not b or a == b:
        return ui.empty("选择两个不同的模型开始对比。")
    ra, rb = ml.model_record(a), ml.model_record(b)
    if not ra or not rb:
        return ui.empty("所选模型不存在（可能刚被删除或晋升）。")
    rows = ml.metric_rows(ra, rb)
    head = html.Tr(
        [
            html.Th("指标"),
            html.Th("A", className="n"),
            html.Th("B", className="n"),
            html.Th("Δ (B−A)", className="n"),
        ]
    )
    body_rows = []
    for r in rows:
        tone = {"b": "ok", "a": "bad"}.get(r["better"], "")
        body_rows.append(
            html.Tr(
                [
                    html.Td(r["label"]),
                    html.Td(_fmt(r["a"], r["key"]), className="n"),
                    html.Td(_fmt(r["b"], r["key"]), className="n"),
                    html.Td(
                        _fmt(r["delta"], r["key"]) if r["delta"] is not None else "—",
                        className=f"n {tone}",
                    ),
                ]
            )
        )
    diffs = ml.recipe_diff(ra["recipe"], rb["recipe"])
    changed = [d for d in diffs if not d["same"]]
    fd = ml.feat_diff(ra["recipe"], rb["recipe"])
    recipe_rows = [
        html.Tr(
            [html.Td(html.Code(d["key"])), html.Td(str(d["a"])), html.Td(str(d["b"]))],
        )
        for d in changed
    ]
    notes = []
    for tag, r in (("A", ra), ("B", rb)):
        notes.append(html.Div(f"{tag} = {r['id']} · 指标来源：{r['src']}", className="hint"))
    return html.Div(
        [
            html.Div(
                [
                    ui.card(
                        "效果指标",
                        (
                            html.Table(
                                [html.Thead(head), html.Tbody(body_rows)], className="t nowrap1"
                            )
                            if body_rows
                            else ui.empty("两个模型都没有可比的指标。")
                        ),
                        *notes,
                    ),
                    ui.card(
                        f"配方差异（{len(changed)} 项不同 / 共 {len(diffs)} 项）",
                        (
                            html.Table(
                                [
                                    html.Thead(
                                        html.Tr([html.Th("参数"), html.Th("A"), html.Th("B")])
                                    ),
                                    html.Tbody(recipe_rows),
                                ],
                                className="t",
                            )
                            if recipe_rows
                            else html.Div("配方参数完全一致。", className="muted")
                        ),
                        html.Div(
                            f"特征：A {fd['n_a']} 个 / B {fd['n_b']} 个 · 仅 A 有 {len(fd['only_a'])}"
                            f" · 仅 B 有 {len(fd['only_b'])}"
                            + (f"（{', '.join(fd['only_b'][:6])}…）" if fd["only_b"] else ""),
                            className="hint",
                            style={"marginTop": "6px"},
                        ),
                    ),
                ],
                className="grid g2",
            ),
        ]
    )


def _static_area():
    opts = ml.model_options()
    staging = [o["value"] for o in opts if o["value"] not in ("champion",)]
    default_a = "champion" if data.champion_manifest() else (staging[0] if staging else None)
    default_b = next((v for v in staging if v != default_a), None)
    return html.Div(
        [
            ui.card(
                "模型对比",
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("A（基准）", className="lb"),
                                dcc.Dropdown(
                                    id="m-cmp-a", options=opts, value=default_a, clearable=False
                                ),
                            ],
                            style={"minWidth": "280px"},
                        ),
                        html.Div(
                            [
                                html.Label("B（对照）", className="lb"),
                                dcc.Dropdown(
                                    id="m-cmp-b", options=opts, value=default_b, clearable=False
                                ),
                            ],
                            style={"minWidth": "280px"},
                        ),
                    ],
                    className="row",
                ),
                html.Div(id="m-cmp-out", children=compare_view(default_a, default_b)),
            ),
            html.Div(
                html.Div(
                    [
                        html.Div(id="m-confirm-title", style={"fontWeight": 600}),
                        html.Div(id="m-confirm-body", style={"margin": "8px 0"}),
                        html.Div(
                            [
                                html.Label(
                                    "理由（未达门槛或评估可疑时必填，写入审计）", className="lb"
                                ),
                                dcc.Input(
                                    id="m-reason", type="text", className="in", debounce=False
                                ),
                            ],
                            id="m-reason-wrap",
                            style={"display": "none", "margin": "6px 0"},
                        ),
                        html.Div(
                            [
                                html.Button("确认", id="m-confirm-run", className="btn primary"),
                                html.Button("取消", id="m-confirm-cancel", className="btn"),
                            ],
                            className="row",
                        ),
                    ],
                    className="modal-box",
                ),
                id="m-confirm-panel",
                className="modal-wrap",
                style={"display": "none"},
            ),
            dcc.Store(id="m-pending"),
            html.Div(id="m-toast", className="toast"),
        ]
    )


def _step_counts(meta: dict) -> tuple[int, int, str]:
    steps = meta.get("steps") or []
    done = sum(s.get("status") in ("done", "skipped") for s in steps)
    current = next((s for s in steps if s.get("status") == "running"), None)
    title = (current or {}).get("title") or meta.get("current_step") or "准备中"
    return done, len(steps), title


def _task_status(kind: str = "train"):
    """按任务类型展示独立实时状态；只有整个 run 结束才显示完成。"""
    label = "训练" if kind == "train" else "信号评估"
    active = [m for m in rr.active_runs(root=_runs_root()) if m.get("kind") == kind]
    if active:
        a = active[0]
        p = a.get("progress") or {}
        done, total_steps, current_title = _step_counts(a)
        frac = p.get("pct") if p.get("total") else None
        phase_progress = (
            f"{p.get('current', 0):g}/{p.get('total', 0):g}（{ui.pct(frac, 0)}）"
            if frac is not None
            else "等待进度上报"
        )
        elapsed = max(0.0, time.time() - (a.get("started") or a.get("created") or time.time()))
        eta = f" · 阶段预计剩余 {ui.dur(p['eta_sec'])}" if p.get("eta_sec") else ""
        stale_note = (
            html.Div("进度已超过 3 分钟未上报；总耗时仍按任务开始时间计算。", className="warn")
            if a.get("liveness") == "stale"
            else None
        )
        return html.Div(
            [
                html.Div(
                    [
                        html.B(f"正在运行 · {a.get('label') or a.get('kind')}"),
                        ui.badge("运行中", "running"),
                        html.Span(f"  {a['run_id']}", className="faint"),
                    ],
                    className="row",
                ),
                html.Div(
                    f"当前步骤：{current_title} · 任务步骤 {done}/{total_steps}",
                    className="hint",
                ),
                ui.bar(frac),
                html.Div(
                    f"阶段进度 {phase_progress} · 已运行 {ui.dur(elapsed)}{eta}",
                    className="hint num",
                ),
                html.Div(
                    f"{p.get('phase', '等待进度上报')} · {p.get('message', '')}", className="hint"
                ),
                stale_note,
                dcc.Link("查看完整进度与日志 →", href=f"/runs/{a['run_id']}"),
            ]
        )
    last = rr.list_runs(kind=kind, limit=1, root=_runs_root())
    if not last:
        return html.Div(f"还没有{label}运行记录。", className="hint")
    m = last[0]
    done, total_steps, _ = _step_counts(m)
    st = {"done": "完成", "failed": "失败", "killed": "已终止"}.get(m["status"], m["status"])
    if m["status"] == "running":
        st = "疑似卡住" if m.get("liveness") == "stale" else "已中断"
    end = m.get("finished") or m.get("heartbeat") or m.get("started") or time.time()
    elapsed = max(0.0, end - (m.get("started") or m.get("created") or end))
    return html.Div(
        [
            html.Div(
                [
                    html.B(f"最近一次{label} · {m.get('label') or m['run_id']}"),
                    ui.badge(st, {"完成": "done", "失败": "failed"}.get(st, "dead")),
                ],
                className="row",
            ),
            html.Div(f"任务步骤 {done}/{total_steps} · 总耗时 {ui.dur(elapsed)}", className="hint"),
            html.Div(m.get("summary") or "", className="hint"),
            dcc.Link("查看运行详情 →", href=f"/runs/{m['run_id']}"),
        ]
    )


def _training_card():
    return ui.card(
        "训练区",
        html.Div(
            "训练新候选并进入 staging；依次完成数据导入、候选训练与门控评估，"
            "不会自动替换 champion。",
            className="hint",
        ),
        html.Div(
            html.Button("训练候选模型 → staging", id="m-train", className="btn primary"),
            style={"margin": "8px 0"},
        ),
        html.Div(id="m-train-msg", className="msg"),
        html.Div(_task_status("train"), id="m-train-status"),
    )


def _evaluation_card():
    return ui.card(
        "评估区",
        html.Div(
            [
                html.Label("信号评估的模型", className="lb"),
                dcc.Dropdown(
                    id="m-eval-ref",
                    options=[
                        {"label": "rolling（滚动重训，不依赖线上模型）", "value": "rolling"},
                        {"label": "champion（线上模型冻结工件）", "value": "champion"},
                    ],
                    value="rolling",
                    clearable=False,
                ),
            ],
            style={"minWidth": "280px"},
        ),
        html.Div(
            html.Button("开始信号评估（IC / 分位数收益）", id="m-eval", className="btn"),
            style={"margin": "8px 0"},
        ),
        html.Div(id="m-eval-msg", className="msg"),
        html.Div(_task_status("modeleval"), id="m-eval-status"),
        html.Div("只评估预测排序力，不做订单撮合或交易成本回测。", className="hint"),
    )


def layout():
    return ui.page(
        "MODELS",
        "模型",
        "线上 champion、待决策的 staging 候选、已退役的 archive。晋升与删除都在这里完成。",
        dcc.Interval(id="m-tick", interval=3000),
        dcc.Store(id="m-sig", data=signature()),
        html.Div([_training_card(), _evaluation_card()], className="grid g2"),
        html.Div(body(), id="m-body"),
        _static_area(),
    )


# ----------------------------------------------------------------------------- 回调
def open_confirm(op: str, model_id: str) -> dict:
    """打开确认弹层所需的内容。纯函数，便于单测。"""
    if op == "delete":
        return {
            "title": f"删除 staging 候选 {model_id}？",
            "lines": ["将永久删除 models/staging/ 下的该目录，无法恢复。"],
            "need_reason": False,
            "can": data.mtime(os.path.join(data.MODELS, "staging", model_id)) is not None,
        }
    plan = ml.promote_plan(model_id)
    lines = list(plan["warnings"]) or ["门控已通过。"]
    if plan["can"]:
        lines.append("晋升后旧 champion 归档到 models/archive/，可回滚。")
    else:
        lines = [plan["why"]]
    return {
        "title": f"晋升 {model_id} 为 champion？",
        "lines": lines,
        "need_reason": plan["need_reason"],
        "can": plan["can"],
    }


def register(app):
    @app.callback(
        Output("m-body", "children"),
        Output("m-sig", "data"),
        Output("m-train-status", "children"),
        Output("m-eval-status", "children"),
        Input("m-tick", "n_intervals"),
        State("m-sig", "data"),
        prevent_initial_call=True,
    )
    def _refresh(_n, stored):
        new_body, sig = refresh_decision(stored)
        if new_body is None:
            return no_update, no_update, _task_status("train"), _task_status("modeleval")
        return new_body, sig, _task_status("train"), _task_status("modeleval")

    @app.callback(
        Output("m-train-msg", "children"),
        Output("m-eval-msg", "children"),
        Input("m-train", "n_clicks"),
        Input("m-eval", "n_clicks"),
        State("m-eval-ref", "value"),
        prevent_initial_call=True,
    )
    def _launch(n_train, n_eval, ref):
        trig = ctx.triggered_id
        if not ((trig == "m-train" and n_train) or (trig == "m-eval" and n_eval)):
            return no_update, no_update
        if trig == "m-train":
            ok, msg, rid = actions.launch_cli("train", [])
        else:
            ok, msg, rid = actions.launch_cli(
                "modeleval", ["--set", f"model_ref={ref or 'rolling'}"]
            )
        message = html.Span(
            [msg + " ", dcc.Link("查看进度", href=f"/runs/{rid}") if rid else ""],
            className="ok" if ok else "bad",
        )
        return (message, no_update) if trig == "m-train" else (no_update, message)

    @app.callback(
        Output("m-cmp-a", "options"),
        Output("m-cmp-b", "options"),
        Input("m-sig", "data"),
        prevent_initial_call=True,
    )
    def _options(_sig):
        opts = ml.model_options()
        return opts, opts

    @app.callback(
        Output("m-cmp-out", "children"),
        Input("m-cmp-a", "value"),
        Input("m-cmp-b", "value"),
        Input("m-sig", "data"),
        prevent_initial_call=True,
    )
    def _compare(a, b, _sig):
        return compare_view(a, b)

    @app.callback(
        Output("m-pending", "data"),
        Output("m-confirm-panel", "style"),
        Output("m-confirm-title", "children"),
        Output("m-confirm-body", "children"),
        Output("m-reason-wrap", "style"),
        Output("m-reason", "value"),
        Input({"type": "m-promote", "id": ALL}, "n_clicks"),
        Input({"type": "m-delete", "id": ALL}, "n_clicks"),
        Input("m-confirm-cancel", "n_clicks"),
        Input("m-confirm-run", "n_clicks"),
        prevent_initial_call=True,
    )
    def _open(_p, _d, _cancel, _run):
        trig = ctx.triggered_id
        hide = {"display": "none"}
        if trig in ("m-confirm-cancel", "m-confirm-run"):
            return None, hide, "", "", hide, ""
        if not isinstance(trig, dict) or not (ctx.triggered and ctx.triggered[0].get("value")):
            return (no_update,) * 6  # 按钮被重建时的伪触发，忽略
        op = "promote" if trig["type"] == "m-promote" else "delete"
        info = open_confirm(op, trig["id"])
        return (
            {"op": op, "id": trig["id"]},
            {"display": "flex"},
            info["title"],
            [
                html.Div(
                    ln,
                    className=(
                        "warn" if ln.startswith(("门控结论", "没有", "候选", "champion")) else ""
                    ),
                )
                for ln in info["lines"]
            ],
            {"display": "block"} if info["need_reason"] else hide,
            "",
        )

    @app.callback(
        Output("m-toast", "children"),
        Output("m-sig", "data", allow_duplicate=True),
        Output("m-body", "children", allow_duplicate=True),
        Input("m-confirm-run", "n_clicks"),
        State("m-pending", "data"),
        State("m-reason", "value"),
        prevent_initial_call=True,
    )
    def _run(n, pending, reason):
        if not n or not pending:
            return no_update, no_update, no_update
        if pending["op"] == "delete":
            ok, msg = actions.delete_staging(pending["id"])
        else:
            ok, msg = actions.promote_staging(pending["id"], reason or "")
        return html.Span(msg, className="ok" if ok else "bad"), signature(), body()
