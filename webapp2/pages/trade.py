"""交易：今天要做什么 → 照单执行 → 成交确认 → 现金。

页面骨架分三层，保证输入不被轮询冲掉：
  · 外壳（trade-tick + 各表单/按钮）永不重建；
  · trade-body 每 3 秒整体替换（只含只读内容与带 id 的执行按钮）；
  · 执行/拒绝按钮用模式匹配 id，点击 → 写 execution_state_<今天>.json → 下一次轮询呈现。

写操作全部走 webapp2.actions（与 CLI 共用 execution.* 实现）：
  执行/拒绝勾选、成交确认（含二次确认）、现金维护、发起「生成次日订单」。
"""

from __future__ import annotations

import os

import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

import run_registry as rr

from .. import actions, data, ui
from ..explain import explain_context, order_explain
from ..trade_logic import (
    account_snapshot,
    confirm_plan,
    exec_progress,
    flow_steps,
    order_amounts,
    order_key,
    orders_fresh,
    rejected_keys,
    risk_layers,
    target_vs_current,
)

_STATE_BADGE = {"ok": "done", "warn": "warn", "off": "pending"}
_RISK_BADGE = {
    "ok": ("正常", "done"),
    "warn": ("降仓", "warn"),
    "danger": ("重度降仓", "failed"),
    "unknown": ("样本不足", "pending"),
    "disabled": ("已关闭", "pending"),
}


def _runs_root() -> str:
    return os.path.join(data.OUTPUTS, "runs")


# ----------------------------------------------------------------------------- 取数
def snapshot() -> dict:
    """本页需要的全部数据，一次读完。页面渲染与测试共用。"""
    orders, day = data.latest_orders()
    data_day = data.latest_data_day()
    today = actions.today_str()
    state = data.execution_state(today)
    acc = data.account_status()
    prev_px, last_px, px_day = data.close_maps()
    h = data.holdings()
    snap = account_snapshot(acc["cash"], h, last_px, prev_px)
    fresh = orders_fresh(day, data_day)
    confirmed = actions.already_confirmed(day) if day else False
    return {
        "orders": orders,
        "day": day,
        "data_day": data_day,
        "today": today,
        "state": state,
        "account": acc,
        "holdings": h,
        "last_px": last_px,
        "prev_px": prev_px,
        "px_day": px_day,
        "snap": snap,
        "fresh": fresh,
        "confirmed": confirmed,
        "risk": risk_layers(data.risk_state(), *_iv()),
        "tracking": data.tracking_report(),
        "live": data.live_daily(),
    }


def _iv() -> tuple[float | None, str]:
    status, scale = data.iv_scale_now()
    return scale, status


def _latest_nightly_params() -> dict:
    """最近一次夜间运行的结构化参数（模型权重等），用于订单「分析」。"""
    for m in rr.list_runs(kind="nightly", limit=10, root=_runs_root()):
        if m.get("status") == "done" and (m.get("params") or {}).get("model_weights"):
            return m["params"]
    return {}


# ----------------------------------------------------------------------------- 渲染
def _flow_card(s: dict):
    progress = exec_progress(s["orders"], s["state"])
    steps = flow_steps(
        orders_day=s["day"],
        data_day=s["data_day"],
        progress=progress,
        has_orders=not s["orders"].empty,
        orders_mtime=data.orders_mtime(s["day"]),
        holdings_mtime=data.holdings_mtime(),
        account=s["account"],
    )
    cells = [
        html.Div(
            [
                html.Div(
                    [
                        html.Span(f"{i}", className="num faint"),
                        html.B(f"  {st['title']}"),
                        ui.badge(
                            {"ok": "完成", "warn": "待办", "off": "未到"}[st["state"]],
                            _STATE_BADGE[st["state"]],
                        ),
                    ]
                ),
                html.Div(st["text"], className="hint", style={"marginTop": "4px"}),
            ],
            className="card",
            style={"margin": 0},
        )
        for i, st in enumerate(steps, 1)
    ]
    return html.Div(cells, className="grid g4")


def _live_status():
    """夜间任务的实时状态（只读，随轮询刷新）。按钮在静态区，不在这里。"""
    active = [m for m in rr.active_runs(root=_runs_root()) if m.get("kind") == "nightly"]
    if active:
        a = active[0]
        p = a.get("progress") or {}
        frac = p.get("pct") if p.get("total") else None
        return html.Div(
            [
                html.Div(
                    [html.B("正在生成订单"), html.Span(f"  {a.get('run_id')}", className="faint")]
                ),
                ui.bar(frac),
                html.Div(
                    f"{p.get('phase', '启动中')} · {p.get('message', '')}"
                    + (f" · 已用 {ui.dur(p.get('elapsed_sec'))}" if p.get("elapsed_sec") else ""),
                    className="hint",
                ),
                dcc.Link("查看实时进度 →", href=f"/runs/{a['run_id']}"),
            ]
        )
    last = rr.list_runs(kind="nightly", limit=1, root=_runs_root())
    if not last:
        return html.Div("还没有夜间运行记录。", className="hint")
    m = last[0]
    st = {"done": "完成", "failed": "失败", "killed": "已终止"}.get(m["status"], m["status"])
    if m["status"] == "running":
        st = "已中断"
    return html.Div(
        [
            "最近一次夜间运行：",
            ui.badge(st, {"完成": "done", "失败": "failed"}.get(st, "dead")),
            html.Span(f"  {ui.ago(m.get('started'))}", className="muted"),
            "  ",
            dcc.Link("详情", href=f"/runs/{m['run_id']}"),
        ]
    )


def _orders_summary(s: dict):
    stats = order_amounts(s["orders"])
    if not s["day"] or s["orders"].empty:
        return html.Div("没有订单。非调仓日或尚未运行夜间任务时属正常。", className="muted")
    stale = "" if s["fresh"] else "（已过期，勿执行）"
    return html.Div(
        f"订单 {s['day']}{stale} · 卖 {stats['n_sell']} 笔 {ui.money(stats['sell_amt'])}"
        f" · 买 {stats['n_buy']} 笔 {ui.money(stats['buy_amt'])} · 拦截 {stats['n_blocked']} 笔",
        className="hint num",
    )


def _why_panel(row: dict, ctx_: dict):
    body = []
    for sec in order_explain(row, ctx_):
        body.append(html.Div(sec["title"], className="why-sec"))
        body.extend(html.Div(ln, className="why-line") for ln in sec["lines"])
    return html.Details(
        [html.Summary("分析"), html.Div(body, className="why-panel")], className="why"
    )


def _order_row(row: dict, s: dict, ctx_: dict):
    blocked = str(row.get("blocked")).lower() in ("true", "1")
    key = order_key(row["side"], row["symbol"])
    mark = s["state"].get(key)
    done, rejected = mark in ("done", True), mark == "rejected"
    shares = row.get("shares") or 0
    px = row.get("limit_price") if pd.notna(row.get("limit_price")) else row.get("ref_close")
    cls = "orow" + (" blocked" if blocked else " done" if done else " rejected" if rejected else "")
    if blocked:
        actions_cell = html.Span("拦截 · 不执行", className="faint")
    else:
        actions_cell = html.Span(
            [
                html.Button(
                    "✓ 已执行" if done else "执行",
                    id={"type": "mark", "key": key, "to": "done"},
                    className="btn primary" if done else "btn",
                ),
                html.Button(
                    "✗ 已拒绝" if rejected else "拒绝",
                    id={"type": "mark", "key": key, "to": "rejected"},
                    className="btn danger" if rejected else "btn",
                ),
            ],
            className="row",
            style={"gap": "6px"},
        )
    return html.Div(
        [
            html.Div(
                [
                    html.Span(row["symbol"], className="num", style={"fontWeight": 600}),
                    html.Span(f"  {int(shares):,} 股", className="num"),
                    html.Span(f"  @ {px:.3f}" if pd.notna(px) else "", className="num muted"),
                    html.Span(
                        f"  ≈{ui.money(float(shares) * float(px))}" if pd.notna(px) else "",
                        className="num muted",
                    ),
                    html.Span(f"  {row.get('reason', '')}", className="muted"),
                ],
                className="o-main",
            ),
            html.Div(actions_cell, className="o-act"),
            html.Div(_why_panel(row, ctx_), className="o-why"),
        ],
        className=cls,
    )


def _orders_block(s: dict):
    orders = s["orders"]
    if orders.empty:
        return ui.card("订单执行单", ui.empty("没有订单。到「运行」页或下方按钮生成次日订单。"))
    ctx_ = explain_context(orders, weights=_latest_nightly_params().get("model_weights"))
    recs = orders.to_dict("records")
    sells = [r for r in recs if r.get("side") == "sell"]
    buys = [r for r in recs if r.get("side") == "buy"]

    def section(title, rows, tone):
        return html.Div(
            [
                html.Div(f"{title}（{len(rows)} 笔）", className=f"osec {tone}"),
                *(
                    [_order_row(r, s, ctx_) for r in rows]
                    if rows
                    else [html.Div("（无）", className="faint")]
                ),
            ]
        )

    return ui.card(
        "订单执行单" + ("" if s["fresh"] else " · 已过期，勿执行"),
        _orders_summary(s),
        section("卖单：先卖腾现金", sells, "sell"),
        section("买单：后买", buys, "buy"),
        html.Div(
            "点「执行」或「拒绝」立即写入 outputs/execution_state_<今日>.json，刷新不丢；"
            "拦截单与已拒绝单在成交确认时被跳过。再点一次同一个按钮可撤销。",
            className="hint",
            style={"marginTop": "8px"},
        ),
        tone="warn" if not s["fresh"] else "",
    )


def _confirm_card(s: dict):
    plan = confirm_plan(s["orders"], s["state"], s["fresh"], s["confirmed"])
    if plan["can"]:
        text = (
            f"将假设 {plan['n_sell'] + plan['n_buy']} 笔订单全部成交"
            f"（卖 {plan['n_sell']} / 买 {plan['n_buy']}），写回 holdings.csv。"
            f"拦截 {plan['n_blocked']} 笔、拒绝 {plan['n_rej']} 笔不计入。"
        )
    else:
        text = f"当前不可确认：{plan['why']}"
    return ui.card(
        "成交确认",
        html.Div(text, className="" if plan["can"] else "warn"),
        html.Div(
            "与 ./daily_run.sh commit 等价。按券商实际成交核对后再确认；确认后请维护现金。",
            className="hint",
            style={"margin": "4px 0 8px"},
        ),
        html.Div(
            [
                html.Button(
                    "确认成交，写回持仓",
                    id="confirm-open",
                    className="btn primary",
                    disabled=not plan["can"],
                ),
            ],
            className="row",
        ),
    )


def _risk_card(s: dict):
    r = s["risk"]
    if r is None:
        return ui.card(
            "风控五层",
            ui.empty("还没有净值历史（live_risk_state.json）。首次夜间运行后开始积累。"),
        )
    rows = []
    for layer in r["layers"]:
        txt, kind = _RISK_BADGE[layer["status"]]
        rows.append(
            html.Tr(
                [
                    html.Td(layer["name"]),
                    html.Td(ui.badge(txt, kind)),
                    html.Td(
                        "—" if layer["scale"] is None else f"×{layer['scale']:.2f}", className="n"
                    ),
                    html.Td(layer["reason"], className="muted"),
                ]
            )
        )
    total = r["total"]
    return ui.card(
        "风控五层",
        html.Div(
            [html.Span(f"总乘数 ×{total:.2f}", className="num", style={"fontSize": "18px"})],
        ),
        html.Table([html.Tbody(rows)], className="t"),
        html.Div(
            "乘数只作用于目标仓位；样本不足的层不参与叠乘。",
            className="hint",
        ),
    )


def _account_card(s: dict):
    snap, acc = s["snap"], s["account"]
    if snap is None:
        return ui.card(
            "账户快照", ui.empty("现金未维护，无法估值。在下方「现金」填写今日账户现金。")
        )
    pnl = snap["pnl"]
    return ui.card(
        "账户快照",
        ui.kv_table(
            [
                ("现金", ui.money(snap["cash"]) + ("（已过期）" if acc["stale"] else "")),
                ("持仓市值", ui.money(snap["mv"])),
                ("净值", ui.money(snap["nav"])),
                ("当日盈亏", "—" if pnl is None else f"{pnl:+,.0f}"),
                ("估值收盘日", s["px_day"] or "—"),
            ]
        ),
    )


def _target_card(s: dict):
    nav = (s["snap"] or {}).get("nav")
    tvc = target_vs_current(
        s["orders"] if s["fresh"] else pd.DataFrame(),
        s["holdings"],
        s["last_px"],
        nav,
        skip_keys=rejected_keys(s["state"]),
    )
    if tvc.empty or tvc["cur_w"].isna().all():
        return ui.card("目标仓位 vs 当前", ui.empty("无持仓且无有效订单，或净值未知。"))
    f = ui.fig("持仓权重：当前 vs 订单推导目标", height=280)
    f.add_bar(x=tvc["symbol"], y=tvc["cur_w"], name="当前", marker_color="#cbd5e1")
    f.add_bar(x=tvc["symbol"], y=tvc["tgt_w"], name="目标", marker_color=ui.C["accent"])
    f.update_layout(barmode="group")
    f.update_yaxes(tickformat=".0%")
    return ui.card("目标仓位 vs 当前", ui.graph(f))


def _holdings_card(s: dict):
    h = s["holdings"]
    if h.empty:
        return ui.card("持仓明细", ui.empty("当前空仓。"))
    rows = []
    for r in h.to_dict("records"):
        sh = float(r.get("shares") or 0)
        if sh <= 0:
            continue
        cost = r.get("ref_close")
        px = s["last_px"].get(str(r["symbol"]))
        pnl = (px - cost) * sh if (px and pd.notna(cost) and cost) else None
        rows.append(
            {
                "symbol": r["symbol"],
                "shares": int(sh),
                "entry_date": str(r.get("entry_date"))[:10],
                "cost": f"{cost:.2f}" if pd.notna(cost) else "—",
                "px": f"{px:.2f}" if px else "—",
                "pnl": "—" if pnl is None else f"{pnl:+,.0f} ({px / cost - 1:+.1%})",
            }
        )
    return ui.card(
        "持仓明细",
        ui.table(
            pd.DataFrame(rows),
            [
                ("symbol", "代码", ""),
                ("shares", "股数", "n"),
                ("entry_date", "建仓日", ""),
                ("cost", "成本", "n"),
                ("px", "现价", "n"),
                ("pnl", "浮盈", "n"),
            ],
        ),
        html.Div(f"现价取自 {s['px_day']} 收盘" if s["px_day"] else "现价不可得", className="hint"),
    )


def _track_card(s: dict):
    live, rep = s["live"], s["tracking"]
    body = []
    d = live.dropna(subset=["live_total_value"]) if "live_total_value" in live.columns else live
    if not d.empty:
        x = pd.to_datetime(d["date"])
        f = ui.fig("实盘净值 vs 等权基准（归一）", height=260)
        f.add_scatter(x=x, y=d["live_total_value"] / d["live_total_value"].iloc[0], name="实盘净值")
        if "bench_ret_1d" in d.columns:
            f.add_scatter(
                x=x,
                y=(1 + d["bench_ret_1d"].fillna(0.0)).cumprod(),
                name="等权基准",
                line=dict(dash="dash"),
            )
        body.append(ui.graph(f))
    elif not live.empty:
        body.append(
            html.Div(
                f"已有 {len(live)} 天预测跟踪记录，但没有真实净值：account.csv 需每日维护现金。",
                className="hint",
            )
        )
    if rep:
        for a in rep.get("alerts") or []:
            body.append(html.Div(a, className="warn", style={"padding": "2px 0"}))
        if not rep.get("alerts"):
            body.append(html.Div("无活跃告警。", className="ok"))
    if not body:
        body.append(ui.empty("还没有跟踪数据。"))
    return ui.card("净值与偏差", *body)


def body():
    s = snapshot()
    return html.Div(
        [
            _flow_card(s),
            _orders_block(s),
            html.Div([_confirm_card(s), _account_card(s)], className="grid g2"),
            html.Div([_risk_card(s), _track_card(s)], className="grid g2"),
            html.Div([_target_card(s), _holdings_card(s)], className="grid g2"),
        ]
    )


# ----------------------------------------------------------------------------- 刷新门控
def signature() -> list:
    """本页内容依赖的全部文件的 mtime。只在它变化时才重绘 trade-body。

    原因：每 3 秒无条件重绘会把用户刚展开的「分析」面板收回去，也可能在点击瞬间替换掉按钮。
    CLI 在另一个进程里写任何一个依赖文件，签名都会变，页面随之更新，所以不会漏刷。
    """
    day = (data.order_files() or [(None, None)])[0][0]
    today = actions.today_str()
    out_files = [
        f"orders_{day}.csv" if day else "",
        "holdings.csv",
        "account.csv",
        f"execution_state_{today}.json",
        "live_risk_state.json",
        "live_risk_state_pred_hist.parquet",
        os.path.join("tracking", "tracking_report.json"),
        os.path.join("tracking", "live_daily.parquet"),
    ]
    sig = [day, today, data.latest_data_day()]
    sig += [data.mtime(os.path.join(data.OUTPUTS, f)) if f else None for f in out_files]
    sig.append(data.mtime(os.path.join(data.MODELS, "champion", "manifest.json")))
    # nightly 运行结束会改 model_weights（订单「分析」用），按最近一次终态运行的 run_id 计入
    sig.append(
        [
            (m["run_id"], m.get("status"))
            for m in rr.list_runs(kind="nightly", limit=1, root=_runs_root())
        ]
    )
    return sig


def refresh_decision(stored) -> tuple[object | None, list]:
    """(新内容或 None, 新签名)。签名没变 → None（调用方返回 no_update，DOM 原样保留）。"""
    sig = signature()
    if stored == sig:
        return None, sig
    return body(), sig


# ----------------------------------------------------------------------------- 骨架
def _forms():
    """永不被轮询替换的表单：现金、发起生成订单、确认弹层。"""
    return html.Div(
        [
            ui.card(
                "生成次日订单",
                html.Div(
                    "等价于终端运行 ./daily_run.sh；资金配置只影响仓位、金额门槛和现金校验，不触发自动交易。",
                    className="hint",
                ),
                html.Div(
                    [
                        html.Label("资金配置", className="lb"),
                        dcc.Dropdown(
                            id="trade-capital-profile",
                            options=[
                                {"label": "standard（默认 3 万 / 6 仓）", "value": "standard"},
                                {"label": "small（2 万 / 4 仓，含现金校验）", "value": "small"},
                            ],
                            value=os.environ.get("CAPITAL_PROFILE", "standard"),
                            clearable=False,
                        ),
                    ],
                    style={"maxWidth": "360px", "margin": "8px 0"},
                ),
                html.Div(
                    html.Button("生成次日订单", id="trade-gen", className="btn primary"),
                    style={"margin": "8px 0"},
                ),
                html.Div(id="trade-gen-msg", className="msg"),
                html.Div(_live_status(), id="trade-live"),
            ),
            ui.card(
                "现金",
                html.Div(
                    [
                        dcc.Input(
                            id="trade-cash",
                            type="number",
                            min=0,
                            step="any",
                            placeholder="今日账户现金",
                            className="in",
                            style={"width": "170px"},
                        ),
                        html.Button("保存", id="trade-cash-save", className="btn primary"),
                    ],
                    className="row",
                ),
                html.Div(id="trade-cash-msg", className="msg"),
            ),
            # 确认弹层：固定定位的遮罩，点「确认成交」按钮的位置在页面中部，弹层必须在视口内
            html.Div(
                html.Div(
                    [
                        html.Div("确认成交，写回持仓？", style={"fontWeight": 600}),
                        html.Div(id="confirm-preview", className="hint", style={"margin": "8px 0"}),
                        html.Div(
                            [
                                html.Button("确认写回", id="confirm-run", className="btn primary"),
                                html.Button("取消", id="confirm-cancel", className="btn"),
                            ],
                            className="row",
                        ),
                    ],
                    className="modal-box",
                ),
                id="confirm-panel",
                className="modal-wrap",
                style={"display": "none"},
            ),
            html.Div(id="confirm-msg", className="toast"),
        ],
        className="grid g2",
    )


def layout():
    return ui.page(
        "TRADE",
        "交易",
        "收盘后出单，次日开盘前照单执行，成交后确认并维护现金。每一步的状态都来自磁盘文件。",
        dcc.Interval(id="trade-tick", interval=3000),
        dcc.Store(id="trade-sig", data=signature()),
        html.Div(body(), id="trade-body"),
        _forms(),
    )


# ----------------------------------------------------------------------------- 回调
def apply_mark(key: str, to: str, today: str | None = None) -> tuple[bool, str]:
    """点击切换：已是该标记 → 清除；否则设为该标记（执行/拒绝互斥）。纯逻辑，便于单测。"""
    day = today or actions.today_str()
    side, _, symbol = key.partition("-")
    if side not in ("buy", "sell") or not symbol:
        return False, f"非法订单键 {key!r}"
    cur = data.execution_state(day).get(key)
    target = {"done": "done", "rejected": "rejected"}.get(to)
    if target is None:
        return False, f"非法标记 {to!r}"
    already = (cur in ("done", True)) if target == "done" else (cur == "rejected")
    return actions.set_execution_mark(day, side, symbol, None if already else target)


def register(app):
    @app.callback(
        Output("trade-body", "children"),
        Output("trade-sig", "data"),
        Output("trade-live", "children"),
        Input("trade-tick", "n_intervals"),
        State("trade-sig", "data"),
        prevent_initial_call=True,
    )
    def _refresh(_n, stored):
        new_body, sig = refresh_decision(stored)
        if new_body is None:
            return no_update, no_update, _live_status()
        return new_body, sig, _live_status()

    @app.callback(
        Output("trade-body", "children", allow_duplicate=True),
        Output("trade-sig", "data", allow_duplicate=True),
        Input({"type": "mark", "key": ALL, "to": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _mark(clicks):
        trig = ctx.triggered_id
        # 模式匹配回调会在按钮被重建时以 n_clicks=None 触发，必须过滤掉，否则会误写状态
        if not isinstance(trig, dict) or not any(c for c in (clicks or []) if c):
            return no_update, no_update
        triggered = ctx.triggered[0] if ctx.triggered else {}
        if not triggered.get("value"):
            return no_update, no_update
        apply_mark(trig["key"], trig["to"])
        return body(), signature()

    @app.callback(
        Output("trade-gen-msg", "children"),
        Input("trade-gen", "n_clicks"),
        State("trade-capital-profile", "value"),
        prevent_initial_call=True,
    )
    def _gen(n, profile):
        if not n:
            return no_update
        ok, msg, rid = actions.launch_cli("nightly", actions.nightly_preset(profile))
        return html.Span(
            [msg + " ", dcc.Link("查看进度", href=f"/runs/{rid}") if rid else ""],
            className="ok" if ok else "bad",
        )

    @app.callback(
        Output("trade-cash-msg", "children"),
        Input("trade-cash-save", "n_clicks"),
        State("trade-cash", "value"),
        prevent_initial_call=True,
    )
    def _cash(n, value):
        if not n:
            return no_update
        ok, msg = actions.save_cash(value)
        return html.Span(msg, className="ok" if ok else "bad")

    @app.callback(
        Output("confirm-panel", "style"),
        Output("confirm-preview", "children"),
        Output("confirm-msg", "children"),
        Input("confirm-open", "n_clicks"),
        Input("confirm-cancel", "n_clicks"),
        Input("confirm-run", "n_clicks"),
        prevent_initial_call=True,
    )
    def _confirm(n_open, n_cancel, n_run):
        trig = ctx.triggered_id
        hide, show = {"display": "none"}, {"display": "flex"}
        if trig == "confirm-open" and n_open:
            s = snapshot()
            plan = confirm_plan(s["orders"], s["state"], s["fresh"], s["confirmed"])
            if not plan["can"]:
                return hide, "", html.Span(plan["why"], className="warn")
            return (
                show,
                f"订单 {s['day']}：卖 {plan['n_sell']} / 买 {plan['n_buy']} 笔将按全部成交写回；"
                f"拦截 {plan['n_blocked']}、拒绝 {plan['n_rej']} 笔跳过。写回后请按券商实际成交核对。",
                "",
            )
        if trig == "confirm-cancel":
            return hide, "", ""
        if trig == "confirm-run" and n_run:
            ok, msg = actions.confirm_fills()
            return hide, "", html.Span(msg, className="ok" if ok else "bad")
        return no_update, no_update, no_update
