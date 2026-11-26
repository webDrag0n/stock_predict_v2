"""总览：今天的状态与待办。每一块都直接来自磁盘文件，与 CLI 是否在跑无关。

读法：先看顶部「待办」（需要你动手的事，按紧迫度排），再看各块的现状。
"""

from __future__ import annotations

import os

import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

import run_registry as rr

from .. import actions, data, ui
from ..trade_logic import blocked_mask, order_key, orders_fresh


def _runs_root() -> str:
    return os.path.join(data.OUTPUTS, "runs")


def signature() -> list:
    """只在账本、订单或运行状态变化时重绘，输入中的表单不参与轮询。"""
    order_files = data.order_files()
    order_path = order_files[0][1] if order_files else ""
    names = [
        "holdings.csv",
        "holdings_history.csv",
        "account.csv",
        "live_risk_state.json",
        "live_rebalance_state.json",
        "live_risk_state_pred_hist.parquet",
        os.path.join("tracking", "tracking_report.json"),
    ]
    runs = [
        (m["run_id"], m.get("status"), m.get("liveness"))
        for m in rr.list_runs(limit=20, root=_runs_root())
    ]
    return [
        data.latest_data_day(),
        data.mtime(order_path) if order_path else None,
        [data.mtime(os.path.join(data.OUTPUTS, n)) for n in names],
        [(day, data.mtime(path)) for day, path in data.execution_state_files()],
        data.mtime(os.path.join(data.MODELS, "champion", "manifest.json")),
        runs,
    ]


def refresh_decision(stored):
    sig = signature()
    return (None, sig) if stored == sig else (body(), sig)


# ----------------------------------------------------------------------------- 待办
def todo_items(ov: dict, now_runs: list[dict]) -> list[tuple[str, str, str]]:
    """[(级别, 标题, 说明)]，级别 bad > warn > info。纯函数，便于单测。"""
    items: list[tuple[str, str, str]] = []
    acc = ov["account"]
    if not acc["exists"]:
        items.append(
            (
                "bad",
                "账户现金未维护",
                "outputs/account.csv 不存在，夜间体检会拒绝出单。请在下方「现金」填写。",
            )
        )
    elif acc["stale"]:
        items.append(
            (
                "bad",
                "账户现金已过期",
                f"{acc['age_days']:.0f} 天未更新（上限 {acc['max_age_days']} 天），夜间体检会以退出码 3 拦下，不出单。",
            )
        )
    if ov["champion"] is None:
        items.append(("bad", "没有 champion", "models/champion 不存在，夜间无法冻结预测。"))
    rep = ov["tracking"] or {}
    age = (rep.get("champion") or {}).get("age_days")
    if age is not None and age > 60:
        items.append(("bad", f"champion 超龄 {age} 个交易日", "超过 60 日强制重训上限。"))
    elif age is not None and age > 48:
        items.append(
            ("warn", f"champion 临近超龄（{age}/60）", "下一次夜间运行可能触发重训与门控。")
        )
    for a in (rep.get("alerts") or [])[:4]:
        items.append(("warn", "跟踪告警", str(a)))
    orders, day = ov["orders"], ov["orders_day"]
    if day and not orders.empty:
        data_day = ov.get("data_day")
        # 订单是「当晚生成、次日开盘执行」的：订单日期早于行情缓存最新日，说明之后又有了
        # 新交易日数据，这批订单已作废，不能再当「待执行」提示（与旧前端 _orders_fresh 同口径）。
        if data_day and str(day) < str(data_day):
            items.append(
                (
                    "warn",
                    f"最新订单已过期（{day}，行情已到 {data_day}）",
                    "这批订单用旧数据生成，不要执行；重新运行夜间任务生成新订单。",
                )
            )
        else:
            h = data.holdings()
            pending = int(
                (
                    ~orders.get("blocked", pd.Series(False, index=orders.index))
                    .astype(str)
                    .str.lower()
                    .isin(["true", "1"])
                ).sum()
            )
            confirmed = (
                not h.empty
                and "entry_date" in h.columns
                and "".join(str(h["entry_date"].max()).split("-")) >= str(day)
            )
            if pending and not confirmed:
                items.append(
                    (
                        "info",
                        f"有 {pending} 笔订单待执行（{day}）",
                        "按券商成交后在「交易」页确认，或用 --commit。",
                    )
                )
    failed = [r for r in now_runs if r.get("status") == "failed"][:1]
    for r in failed:
        items.append(
            (
                "warn",
                f"最近一次运行失败：{r.get('label')}",
                (r.get("failure") or {}).get("exc_msg", "")[:100],
            )
        )
    dead = [r for r in now_runs if r.get("status") == "running" and r.get("liveness") == "dead"]
    for r in dead[:2]:
        items.append(
            (
                "bad",
                f"运行意外中断：{r.get('label')}",
                "进程已不存在，没有写终态（可能被 kill 或机器重启）。",
            )
        )
    order = {"bad": 0, "warn": 1, "info": 2}
    return sorted(items, key=lambda x: order[x[0]])


def _todo_card(items):
    if not items:
        return ui.card("待办", html.Div("没有需要处理的事项。", className="ok"), tone="ok")
    rows = [
        html.Div(
            [
                ui.badge(
                    {"bad": "紧急", "warn": "注意", "info": "提示"}[k],
                    {"bad": "dead", "warn": "warn", "info": "pending"}[k],
                ),
                html.Span(f"  {t}  ", style={"fontWeight": 600}),
                html.Span(d, className="muted"),
            ],
            style={"padding": "5px 0"},
        )
        for k, t, d in items
    ]
    return ui.card("待办", *rows, tone="bad" if items[0][0] == "bad" else "warn")


# ----------------------------------------------------------------------------- 各块
def _orders_card(ov):
    orders, day = ov["orders"], ov["orders_day"]
    if not day or orders.empty:
        return ui.card("最新订单", ui.empty("没有订单。非调仓日或尚未运行夜间任务时属正常。"))
    nb = (
        orders.get("blocked", pd.Series(False, index=orders.index))
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )
    buy = int(((orders["side"] == "buy") & ~nb).sum())
    sell = int(((orders["side"] == "sell") & ~nb).sum())
    notional = (
        float((orders["shares"] * orders["limit_price"])[~nb].sum())
        if {"shares", "limit_price"} <= set(orders.columns)
        else 0.0
    )
    cols = [
        (c, h, k)
        for c, h, k in [
            ("symbol", "代码", ""),
            ("side", "方向", ""),
            ("shares", "股数", "n"),
            ("limit_price", "限价", "n"),
            ("ref_close", "参考价", "n"),
            ("reason", "原因", ""),
            ("rank_pct", "排名", "n"),
        ]
    ]
    stale = bool(ov.get("data_day")) and str(day) < str(ov["data_day"])
    return ui.card(
        f"最新订单 · {day}" + ("（已过期，勿执行）" if stale else ""),
        html.Div(
            f"买 {buy} · 卖 {sell} · 拦截 {int(nb.sum())} · 预计金额 {ui.money(notional)}",
            className="hint num",
        ),
        ui.table(
            orders,
            cols,
            row_class=lambda r: "blocked" if str(r.get("blocked")).lower() in ("true", "1") else "",
        ),
        tone="warn" if stale else "hot",
    )


def _champion_card(ov):
    m, rep = ov["champion"], ov["tracking"] or {}
    if m is None:
        return ui.card("champion", ui.empty("没有 champion"), tone="bad")
    ch = rep.get("champion") or {}
    exp = ch.get("backtest_expectation") or {}
    age = ch.get("age_days")
    return ui.card(
        "champion",
        ui.kv_table(
            [
                ("model_id", m.get("model_id")),
                ("训练截止", m.get("train_end_date")),
                ("年龄", f"{age} / 60 交易日" if age is not None else "—"),
                ("晋升", (m.get("promotion") or {}).get("decided_at", "—")),
                ("回测预期 Sharpe", ui.fnum(exp.get("sharpe"), "{:.2f}")),
                ("评估天数", exp.get("n_days", "—")),
                ("特征数", len((m.get("config_snapshot") or {}).get("feat_cols") or [])),
            ]
        ),
        html.Div((m.get("promotion") or {}).get("reason", "")[:140], className="hint"),
    )


def _risk_card(ov):
    rs, rb = ov["risk_state"], ov["rebalance_state"]
    live = (ov["tracking"] or {}).get("live") or {}
    rows = [
        ("净值点数", len((rs or {}).get("total_values") or [])),
        ("最新总值", ui.money(((rs or {}).get("total_values") or [None])[-1])),
        ("活体 IC 样本", live.get("n_ic", 0)),
        ("活体 IC 均值", ui.fnum(live.get("ic_mean"), "{:+.4f}")),
        ("上次调仓", (rb or {}).get("last_rebalance_date", "—")),
    ]
    note = None
    if (live.get("n_ic") or 0) < 3 or len((rs or {}).get("total_values") or []) < 20:
        note = html.Div(
            "样本不足：IC 降仓与回撤刹车要积累约 20 个交易日后才起作用。", className="hint"
        )
    return ui.card("风控与实盘跟踪", ui.kv_table(rows), note)


def holding_order_states(holdings, orders, execution_state) -> dict[str, str]:
    """把当前持仓映射到最新订单落实状态，供总览分组排序与测试共用。"""
    out = {}
    if holdings is None or holdings.empty:
        return out
    orders = orders if orders is not None else pd.DataFrame()
    for symbol in holdings["symbol"].astype(str):
        if orders.empty or "symbol" not in orders:
            out[symbol] = "无本批订单"
            continue
        sub = orders[orders["symbol"].astype(str) == symbol]
        if sub.empty:
            out[symbol] = "无本批订单"
            continue
        valid = sub[~blocked_mask(sub)]
        if valid.empty:
            out[symbol] = "已拦截"
            continue
        done = rejected = 0
        for row in valid.to_dict("records"):
            mark = (execution_state or {}).get(order_key(row["side"], row["symbol"]))
            done += mark in ("done", True)
            rejected += mark == "rejected"
        pending = len(valid) - done - rejected
        if done and (pending or rejected):
            status = "部分落实"
        elif done:
            status = "已执行"
        elif pending:
            status = "待执行"
        else:
            status = "已拒绝"
        out[symbol] = status
    return out


def _holdings_card(ov):
    h = data.holdings()
    if h.empty:
        return ui.card(
            "当前持仓 · 按订单落实状态",
            ui.empty("当前空仓（没有 holdings.csv）；可在下方精确持仓校正中新增。"),
        )
    _, state = data.latest_execution_state()
    statuses = holding_order_states(h, ov["orders"], state)
    show = h.copy()
    show["落实状态"] = show["symbol"].astype(str).map(statuses)
    rank = {"待执行": 0, "部分落实": 1, "已执行": 2, "已拒绝": 3, "无本批订单": 4, "已拦截": 5}
    show = show.assign(_rank=show["落实状态"].map(rank)).sort_values(["_rank", "symbol"])
    return ui.card(
        "当前持仓 · 按订单落实状态",
        ui.table(
            show,
            [
                ("落实状态", "落实状态", ""),
                ("symbol", "代码", ""),
                ("shares", "股数", "n"),
                ("entry_date", "建仓日", ""),
                ("ref_close", "参考价", "n"),
            ],
        ),
        html.Div(
            "状态来自最新订单与 execution_state；当前股数来自 holdings.csv。"
            "实际数量或价格有差异时使用下方精确校正。",
            className="hint",
        ),
    )


def _execution_card(ov):
    orders, day = ov["orders"], ov["orders_day"]
    if not day or orders.empty:
        return ui.card("订单落实状态", ui.empty("没有最新订单可编辑。"))
    _, state = data.latest_execution_state()
    fresh = orders_fresh(day, ov.get("data_day"))
    rows = []
    for row in orders.to_dict("records"):
        blocked = bool(blocked_mask(pd.DataFrame([row])).iloc[0])
        key = order_key(row["side"], row["symbol"])
        mark = state.get(key)
        done, rejected = mark in ("done", True), mark == "rejected"
        status = "已拦截" if blocked else "已执行" if done else "已拒绝" if rejected else "待执行"
        if blocked or not fresh:
            action = html.Span("拦截" if blocked else "订单已过期", className="faint")
        else:
            action = html.Span(
                [
                    html.Button(
                        "✓ 已执行" if done else "执行",
                        id={"type": "ov-mark", "key": key, "to": "done"},
                        className="btn primary" if done else "btn",
                    ),
                    html.Button(
                        "✗ 已拒绝" if rejected else "拒绝",
                        id={"type": "ov-mark", "key": key, "to": "rejected"},
                        className="btn danger" if rejected else "btn",
                    ),
                ],
                className="row",
                style={"gap": "6px"},
            )
        price = row.get("limit_price") if pd.notna(row.get("limit_price")) else row.get("ref_close")
        rows.append(
            html.Tr(
                [
                    html.Td(row["symbol"], className="mono"),
                    html.Td(row["side"]),
                    html.Td(f"{int(row.get('shares') or 0):,}", className="n"),
                    html.Td(f"{price:.3f}" if pd.notna(price) else "—", className="n"),
                    html.Td(
                        ui.badge(status, "done" if done else "warn" if rejected else "pending")
                    ),
                    html.Td(action),
                ],
                className="blocked" if blocked else "",
            )
        )
    head = html.Tr([html.Th(x) for x in ("代码", "方向", "股数", "价格", "状态", "再次编辑")])
    return ui.card(
        f"订单落实状态 · {day}" + ("" if fresh else " · 已过期"),
        html.Table([html.Thead(head), html.Tbody(rows)], className="t"),
        html.Div(
            "再次点击同一按钮可撤销，执行与拒绝互斥。这里只记录实际执行选择；"
            "部分成交的数量和价格请在下方精确校正，或用实际 fills CSV。",
            className="hint",
        ),
        tone="" if fresh else "warn",
    )


def _history_card():
    history = data.holdings_history(30)
    if history.empty:
        return ui.card(
            "持仓历史",
            ui.empty("还没有历史快照；下一次订单确认、实际成交写回或手动校正后生成。"),
        )
    show = history.copy()
    if "action" in show:
        show["action"] = (
            show["action"]
            .map(
                {
                    "orders_commit": "订单确认",
                    "fills_commit": "实际成交",
                    "manual_edit": "手动校正",
                }
            )
            .fillna(show["action"])
        )
    return html.Details(
        [
            html.Summary(f"持仓历史 · 最近 {len(show)} 行（按快照分组）"),
            html.Div(
                ui.table(
                    show,
                    [
                        ("recorded_at", "记录时间", ""),
                        ("effective_date", "生效日", ""),
                        ("action", "来源", ""),
                        ("symbol", "代码", ""),
                        ("shares", "股数", "n"),
                        ("entry_date", "建仓日", ""),
                        ("ref_close", "参考价", "n"),
                    ],
                ),
                className="b",
            ),
        ],
        className="d",
    )


def _runs_card(runs):
    if not runs:
        return ui.card(
            "最近运行", ui.empty("没有运行记录"), html.Div(dcc.Link("去运行页", href="/runs"))
        )
    rows = []
    for m in runs[:6]:
        st = {"running": "运行中", "done": "完成", "failed": "失败", "killed": "已终止"}.get(
            m.get("status"), m.get("status")
        )
        if m.get("status") == "running" and m.get("liveness") == "dead":
            st = "已中断"
        rows.append(
            html.Tr(
                [
                    html.Td(dcc.Link(m.get("label") or m["run_id"], href=f"/runs/{m['run_id']}")),
                    html.Td(
                        ui.badge(
                            st,
                            {
                                "运行中": "running",
                                "完成": "done",
                                "失败": "failed",
                                "已中断": "dead",
                            }.get(st, "pending"),
                        )
                    ),
                    html.Td(ui.ago(m.get("started") or m.get("created")), className="n"),
                ]
            )
        )
    return ui.card(
        "最近运行",
        html.Table([html.Tbody(rows)], className="t"),
        html.Div(dcc.Link("全部运行 →", href="/runs"), style={"marginTop": "6px"}),
    )


def _cash_status(ov):
    """现金的只读状态行：随轮询刷新。输入框与保存按钮在 _cash_form，不随轮询重建。"""
    acc = ov["account"]
    cur = ui.money(acc["cash"]) if acc["cash"] is not None else "未维护"
    age = ("%.1f 天" % acc["age_days"]) if acc["age_days"] is not None else "—"
    return html.Div(
        [
            html.Div(
                [
                    html.Span(cur, className="num", style={"fontSize": "20px", "fontWeight": 600}),
                    html.Span(
                        "  已过期" if acc["stale"] else "  新鲜",
                        className="warn" if acc["stale"] else "ok",
                    ),
                ]
            ),
            html.Div(f"文件 {os.path.basename(acc['path'])} · 距上次更新 {age}", className="hint"),
        ],
        id="cash-status",
    )


def _cash_form():
    """静态表单：不在任何会被轮询替换的容器里，所以输入中的内容不会被清掉。"""
    return ui.card(
        "现金",
        html.Div(_cash_status(data.overview()), id="cash-status-slot"),
        html.Div(
            [
                dcc.Input(
                    id="cash-input",
                    type="number",
                    min=0,
                    step="any",
                    placeholder="今日账户现金",
                    className="in",
                    style={"width": "170px"},
                ),
                html.Button("保存", id="cash-save", className="btn primary"),
            ],
            className="row",
            style={"marginTop": "8px"},
        ),
        html.Div(id="cash-msg", className="msg"),
    )


def _holdings_form():
    """固定持仓编辑器；只有用户点击重新载入或保存成功才改写文本。"""
    return ui.card(
        "精确持仓校正",
        html.Div(
            "每行一只：symbol,shares,entry_date,ref_close。股数填 0 可删除；"
            "仅保留表头即空仓。保存覆盖当前账本，但历史快照不会删除。",
            className="hint",
        ),
        dcc.Textarea(
            id="ov-holdings-input",
            value=actions.holdings_editor_text(),
            className="in",
            style={"height": "170px", "marginTop": "8px"},
        ),
        html.Div(
            [
                html.Button("保存精确持仓", id="ov-holdings-save", className="btn primary"),
                html.Button("重新载入当前账本", id="ov-holdings-reload", className="btn"),
            ],
            className="row",
            style={"marginTop": "8px"},
        ),
        html.Div(id="ov-holdings-msg", className="msg"),
    )


def _settings_form():
    values = actions.trading_preferences()

    def number_field(label, key, hint):
        return html.Div(
            [
                html.Label(label, className="lb"),
                dcc.Input(
                    id=f"ov-setting-{key}",
                    type="number",
                    min=0,
                    step="any",
                    value=values[key],
                    className="in",
                ),
                html.Div(hint, className="hint"),
            ],
            style={"minWidth": "190px"},
        )

    return ui.card(
        "交易费用与撮合设置",
        html.Div(
            "保存到 outputs/trading_settings.json。新启动的出单和回测生效；"
            "进行中的任务不变，环境变量优先。",
            className="hint",
        ),
        html.Div(
            [
                number_field("佣金率", "commission_rate", "0.0002 = 0.02%，买卖双边"),
                number_field("印花税率", "stamp_duty_rate", "0.001 = 0.1%，仅卖出"),
                number_field("最低佣金（元/笔）", "min_commission", "0 表示无最低佣金"),
                number_field("冲击系数 K", "impact_k", "平方根冲击模型，默认 0.1"),
                number_field("参与率上限", "max_participation", "0.1 = 10%；0 表示不限量"),
            ],
            className="row",
            style={"marginTop": "8px"},
        ),
        dcc.Checklist(
            id="ov-setting-use-impact",
            options=[{"label": " 启用冲击成本", "value": "on"}],
            value=["on"] if values["use_impact_cost"] else [],
            inline=True,
            style={"marginTop": "8px"},
        ),
        html.Div(
            html.Button("保存交易设置", id="ov-settings-save", className="btn primary"),
            style={"marginTop": "8px"},
        ),
        html.Div(id="ov-settings-msg", className="msg"),
    )


def body():
    ov = data.overview()
    runs = rr.list_runs(limit=20, root=_runs_root())
    ph = data.pred_hist()
    snap = (
        ui.kpi("—", "最新预测日")
        if ph.empty
        else ui.kpi(
            str(ph["date"].max())[:10], "最新预测日", "", f"{ph['date'].nunique()} 天已累积"
        )
    )
    return html.Div(
        [
            _todo_card(todo_items(ov, runs)),
            html.Div(
                [
                    ui.kpi(ov["orders_day"] or "—", "最新订单日"),
                    snap,
                    ui.kpi(
                        str(((ov["tracking"] or {}).get("champion") or {}).get("age_days", "—")),
                        "champion 年龄(交易日)",
                    ),
                    ui.kpi(str(len(data.holdings())), "持仓只数"),
                ],
                className="grid g4",
            ),
            html.Div([_orders_card(ov), _champion_card(ov)], className="grid g2"),
            html.Div([_risk_card(ov), _holdings_card(ov), _runs_card(runs)], className="grid g3"),
            _execution_card(ov),
            _history_card(),
        ]
    )


def layout():
    """动态账本按签名刷新；现金、持仓和费用表单始终保留用户输入。"""
    return ui.page(
        "OVERVIEW",
        "总览",
        "当前持仓、订单落实状态、历史快照与账户设置。所有数据来自磁盘文件。",
        dcc.Interval(id="ov-tick", interval=5000),
        dcc.Store(id="ov-sig", data=signature()),
        html.Div(body(), id="ov-body"),
        html.Div(id="ov-order-msg", className="msg"),
        html.Div([_cash_form(), _holdings_form(), _settings_form()], className="grid g2"),
    )


def register(app):
    @app.callback(
        Output("ov-body", "children"),
        Output("cash-status-slot", "children"),
        Output("ov-sig", "data"),
        Input("ov-tick", "n_intervals"),
        State("ov-sig", "data"),
        prevent_initial_call=True,
    )
    def _refresh(_n, stored):
        new_body, sig = refresh_decision(stored)
        if new_body is None:
            return no_update, no_update, sig
        return new_body, _cash_status(data.overview()), sig

    @app.callback(
        Output("cash-msg", "children"),
        Output("cash-status-slot", "children", allow_duplicate=True),
        Output("ov-body", "children", allow_duplicate=True),
        Output("ov-sig", "data", allow_duplicate=True),
        Input("cash-save", "n_clicks"),
        State("cash-input", "value"),
        prevent_initial_call=True,
    )
    def _save_cash(n, value):
        if not n:
            return no_update, no_update, no_update, no_update
        ok, msg = actions.save_cash(value)
        return (
            html.Span(msg, className="ok" if ok else "bad"),
            _cash_status(data.overview()),
            body(),
            signature(),
        )

    @app.callback(
        Output("ov-body", "children", allow_duplicate=True),
        Output("ov-sig", "data", allow_duplicate=True),
        Output("ov-order-msg", "children"),
        Input({"type": "ov-mark", "key": ALL, "to": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _mark(clicks):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not any(c for c in (clicks or []) if c):
            return no_update, no_update, no_update
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update, no_update, no_update
        side, _, symbol = trig["key"].partition("-")
        target = trig["to"]
        day = actions.today_str()
        current = data.execution_state(day).get(trig["key"])
        already = current in ("done", True) if target == "done" else current == "rejected"
        ok, msg = actions.set_execution_mark(day, side, symbol, None if already else target)
        return body(), signature(), html.Span(msg, className="ok" if ok else "bad")

    @app.callback(
        Output("ov-holdings-msg", "children"),
        Output("ov-holdings-input", "value"),
        Output("ov-body", "children", allow_duplicate=True),
        Output("ov-sig", "data", allow_duplicate=True),
        Input("ov-holdings-save", "n_clicks"),
        State("ov-holdings-input", "value"),
        prevent_initial_call=True,
    )
    def _save_holdings(n, text):
        if not n:
            return no_update, no_update, no_update, no_update
        ok, msg = actions.save_holdings_editor(text or "")
        if not ok:
            return html.Span(msg, className="bad"), no_update, no_update, no_update
        return (
            html.Span(msg, className="ok"),
            actions.holdings_editor_text(),
            body(),
            signature(),
        )

    @app.callback(
        Output("ov-holdings-input", "value", allow_duplicate=True),
        Input("ov-holdings-reload", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reload_holdings(n):
        return actions.holdings_editor_text() if n else no_update

    @app.callback(
        Output("ov-settings-msg", "children"),
        Input("ov-settings-save", "n_clicks"),
        State("ov-setting-commission_rate", "value"),
        State("ov-setting-stamp_duty_rate", "value"),
        State("ov-setting-min_commission", "value"),
        State("ov-setting-impact_k", "value"),
        State("ov-setting-use-impact", "value"),
        State("ov-setting-max_participation", "value"),
        prevent_initial_call=True,
    )
    def _save_settings(n, commission, stamp, minimum, impact_k, impact_on, participation):
        if not n:
            return no_update
        values = {
            "commission_rate": commission,
            "stamp_duty_rate": stamp,
            "min_commission": minimum,
            "impact_k": impact_k,
            "use_impact_cost": "on" in (impact_on or []),
            "max_participation": participation,
        }
        ok, msg, _ = actions.save_trading_preferences(values)
        return html.Span(msg, className="ok" if ok else "bad")
