"""流水线 Pipeline（/pipeline）—— 可视化工作流驾驶舱。

设计动机（对应 docs/frontend_redesign.html 的信息架构延伸）：
把「数据 → 因子 → 训练 → 验证 → 门控 → 交易」画成一条可点击的流水线，
每个节点实时显示健康状态；点击节点在下方展开该阶段的操作面板——
状态详情、可调参数（pipeline/stages.py::stage_param_schema 驱动）、
一键发起对应会话（scheduler.create）、跳转专属工作台。

阶段元数据来自 pipeline/stages.py（纯静态，无 IO）；状态灯由本页用
dataio/scheduler 现算。新增阶段只改 stages.py，本页不用动。
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc

from .. import dataio
from ..components import page_header, svg_icon
from ..theme import PALETTE
from pipeline import registry, scheduler, stages as pipe_stages

# --- 阶段图标（16px stroke SVG，与 app.py 侧边栏同风格） --------------------------
_ICONS = {
    "data": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>',
    "factor": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M18 4H6l6 8-6 8h12"/></svg>',
    "train": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6M10 3v6.3L4.7 18a2 2 0 0 0 1.8 3h11a2 2 0 0 0 1.8-3L14 9.3V3"/><path d="M7.5 14h9"/></svg>',
    "validate": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-6 4 3 6-8 3 4"/><path d="M3 21h18"/></svg>',
    "gate": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 4v5c0 5-3.5 8-8 9-4.5-1-8-4-8-9V7z"/><path d="M9 12l2 2 4-4"/></svg>',
    "trade": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 10v10"/></svg>',
}

# 状态灯三色：ok 绿 / warn 黄 / bad 红 / idle 灰
_LAMP_CLASS = {"ok": "pipe-lamp-ok", "warn": "pipe-lamp-warn", "bad": "pipe-lamp-bad", "idle": "pipe-lamp-idle"}


def _fmt_date8(d: str | None) -> str:
    """YYYYMMDD → YYYY-MM-DD；非法输入原样返回。"""
    if d and len(d) == 8 and d.isdigit():
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return d or "-"


# --- 各阶段状态灯：返回 (level, 一行摘要) ------------------------------------------
def _status_data() -> tuple[str, str]:
    """数据缓存新鲜度：最新 daily 缓存距今 ≤4 天绿（含周末）、≤7 天黄、否则红。"""
    try:
        dts = dataio.trade_dates()
        if not dts:
            return "bad", "无行情缓存"
        latest = dts[-1]
        age = (pd.Timestamp.now().normalize() - pd.Timestamp(_fmt_date8(latest))).days
        level = "ok" if age <= 4 else ("warn" if age <= 7 else "bad")
        return level, f"缓存至 {_fmt_date8(latest)}（{age} 天前）"
    except Exception:
        return "bad", "缓存读取失败"


def _status_factor() -> tuple[str, str]:
    """因子库规模 + 最近一次全库评估时间。"""
    try:
        lib = dataio.factor_library()
        n = int(lib["enabled"].sum()) if not lib.empty else 0
        ev = dataio.factor_eval_table()
        if ev.empty:
            return "warn", f"{n} 个启用因子 · 尚未评估"
        mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(dataio.OUTPUTS, "factor_eval.csv")))
        age = (pd.Timestamp.now() - pd.Timestamp(mtime)).days
        level = "ok" if age <= 14 else "warn"
        return level, f"{n} 个启用因子 · 评估于 {age} 天前"
    except Exception:
        return "bad", "因子库读取失败"


def _status_train() -> tuple[str, str]:
    """staging 候选数：有待决策的候选亮黄（提示该去门控对比）。"""
    try:
        n = len(dataio.staging_manifests())
        if n:
            return "warn", f"{n} 个候选待门控决策"
        return "ok", "无待晋升候选"
    except Exception:
        return "idle", "状态未知"


def _status_validate() -> tuple[str, str]:
    """canonical walkforward：有产物显示 OOS Sharpe，缺失亮黄。"""
    try:
        meta = dataio._read_json(os.path.join(dataio.OUTPUTS, "walkforward", "meta.json"))
        if not meta:
            return "warn", "尚无 walkforward 产物"
        sh = (meta.get("metrics") or {}).get("sharpe")
        sh_txt = f"OOS Sharpe {sh:+.2f}" if isinstance(sh, (int, float)) else "OOS Sharpe -"
        gen = pd.to_datetime(meta.get("generated_at"), errors="coerce")
        age = "" if pd.isna(gen) else f" · {(pd.Timestamp.now() - gen.tz_localize(None)).days} 天前"
        return ("ok" if isinstance(sh, (int, float)) and sh > 0 else "warn"), sh_txt + age
    except Exception:
        return "idle", "状态未知"


def _status_gate() -> tuple[str, str]:
    """champion 年龄：≥84 自然日黄灯（与模型实验室口径一致）。"""
    try:
        man = dataio.champion_manifest()
        if not man:
            return "bad", "无 champion"
        created = pd.to_datetime(man.get("created_at"), errors="coerce")
        if pd.isna(created):
            return "idle", f"champion {man.get('model_id', '?')}"
        age = (pd.Timestamp.now() - created.tz_localize(None)).days
        return ("warn" if age >= 84 else "ok"), f"champion 已服役 {age} 天"
    except Exception:
        return "idle", "状态未知"


def _status_trade() -> tuple[str, str]:
    """最新订单日期 vs 最新数据日期：一致绿，落后黄。"""
    try:
        orders, meta = dataio.latest_orders_with_meta()
        if orders.empty:
            return "warn", "尚无订单"
        odate = meta.get("date") or ""
        dts = dataio.trade_dates()
        latest = _fmt_date8(dts[-1]) if dts else ""
        if odate and latest and odate >= latest:
            return "ok", f"订单已覆盖 {odate}"
        return "warn", f"最新订单 {odate or '-'} · 数据至 {latest or '-'}"
    except Exception:
        return "idle", "状态未知"


_STAGE_STATUS = {
    "data": _status_data,
    "factor": _status_factor,
    "train": _status_train,
    "validate": _status_validate,
    "gate": _status_gate,
    "trade": _status_trade,
}


# --- 流水线节点 -------------------------------------------------------------------
def _node(st: dict, selected: str | None) -> html.Button:
    """单个流水线节点卡片（可点击）。选中态加 .pipe-node-active。"""
    level, summary = _STAGE_STATUS[st["id"]]()
    cls = "pipe-node" + (" pipe-node-active" if st["id"] == selected else "")
    return html.Button(
        [
            html.Div(
                [
                    svg_icon(_ICONS[st["id"]], "pipe-icon", size=20),
                    html.Span("", className=f"pipe-lamp {_LAMP_CLASS[level]}", title=st["status_hint"]),
                ],
                className="pipe-node-top",
            ),
            html.Div(st["title"], className="pipe-node-title"),
            html.Div(st["en"], className="pipe-node-en"),
            html.Div(summary, className="pipe-node-status"),
        ],
        id={"type": "pipe-node", "stage": st["id"]},
        className=cls,
        n_clicks=0,
    )


def _pipeline_row(selected: str | None) -> html.Div:
    """六节点横向流水线：节点 + 箭头连接器交替排列。"""
    children = []
    for i, st in enumerate(pipe_stages.PIPELINE_STAGES):
        if i:
            children.append(html.Div("›", className="pipe-connector"))
        children.append(_node(st, selected))
    return html.Div(children, className="pipe-row")


# --- 阶段展开面板 -------------------------------------------------------------------
def _param_controls(stage_id: str):
    """按 stages.stage_param_schema 渲染参数控件（紧凑两列）。"""
    schema = pipe_stages.stage_param_schema(stage_id)
    items = []
    for p in schema:
        cid = {"type": "pipe-param", "key": p["key"]}
        if p["type"] == "number":
            ctrl = dbc.Input(
                type="number", value=p["default"], id=cid,
                min=p.get("min"), max=p.get("max"), step=p.get("step"), size="sm",
            )
        elif p["type"] == "select":
            ctrl = dcc.Dropdown(
                id=cid, value=p["default"], clearable=False,
                options=[{"label": o, "value": o} for o in p["options"]],
            )
        elif p["type"] == "bool":
            ctrl = dbc.Checklist(
                id=cid, value=([p["key"]] if p["default"] else []),
                options=[{"label": "", "value": p["key"]}], switch=True,
            )
        else:
            ctrl = dbc.Input(
                type="text", value=p.get("default", ""), id=cid,
                placeholder=p.get("placeholder", ""), size="sm",
            )
        items.append(
            dbc.Col(
                [
                    html.Div(p["label"], className="control-label"),
                    ctrl,
                    html.Div(p.get("hint", ""), style={"fontSize": "11px", "color": PALETTE["muted"]}),
                ],
                md=6,
            )
        )
    return dbc.Row(items, className="g-3") if items else None


def _stage_detail_rows(stage_id: str) -> list:
    """面板左侧的状态详情行（比节点上的一行摘要更详细）。"""
    rows = []
    if stage_id == "data":
        try:
            dts = dataio.trade_dates()
            uni = dataio.universe()
            rows = [
                f"行情缓存：{_fmt_date8(dts[0])} ~ {_fmt_date8(dts[-1])}（{len(dts)} 个交易日）" if dts else "无行情缓存",
                f"股票池：{len(uni)} 只",
                "更新方式：命令行跑 ./daily_run.sh（数据导入也会随各会话首步自动执行）",
            ]
        except Exception as e:
            rows = [f"读取失败：{e}"]
    elif stage_id == "gate":
        man = dataio.champion_manifest()
        stg = dataio.staging_manifests()
        if man:
            rows.append(f"champion：{man.get('model_id', '?')}（创建 {str(man.get('created_at', ''))[:10]}）")
        for e in stg:
            ev = e.get("evaluation") or {}
            dec = ev.get("decision")
            tag = {"promoted": "已达晋升门槛", "rejected": "未达门槛"}.get(dec, "未评估")
            rows.append(f"候选 {e['id']}：{tag}（{ev.get('reason', '')[:60]}）")
        if not stg:
            rows.append("无 staging 候选。")
        rows.append("晋升/淘汰操作在模型实验室的「模型版本」区完成。")
    return rows


def _stage_panel(stage_id: str | None):
    """选中阶段的展开面板：状态详情 + 参数编辑 + 发起会话 + 工作台链接。"""
    if not stage_id:
        return html.Div("点击上方任一阶段节点，在此展开它的状态与操作。", className="pipe-panel-empty")
    st = pipe_stages.stage(stage_id)
    if not st:
        return html.Div("未知阶段", className="pipe-panel-empty")
    level, summary = _STAGE_STATUS[stage_id]()
    mode = st.get("mode")
    body = [
        html.Div(
            [
                html.Span("", className=f"pipe-lamp {_LAMP_CLASS[level]}", style={"marginRight": "8px"}),
                html.B(f"{st['title']} · {st['en']}", style={"fontSize": "15px"}),
                html.Span(summary, style={"marginLeft": "10px", "fontSize": "12px", "color": PALETTE["muted"]}),
                dcc.Link("打开工作台 →", href=st["page"], className="pipe-panel-link"),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
        ),
        html.Div(st["desc"], style={"fontSize": "13px", "color": PALETTE["muted"], "marginBottom": "10px"}),
    ]
    detail = _stage_detail_rows(stage_id)
    if detail:
        body.append(
            html.Ul([html.Li(r, style={"fontSize": "12.5px"}) for r in detail], className="mb-2")
        )
    form = _param_controls(stage_id)
    if form:
        body.append(form)
    if mode:
        eta = {
            "walkforward": "预计 ≈5 小时（长任务）",
            "train": "分钟~小时级",
            "factoreval": "十分钟级起",
            "nextday": "分钟级",
        }.get(mode, "")
        body.append(
            html.Div(
                [
                    dbc.Button(
                        f"▶ 发起「{registry.MODE_LABELS.get(mode, mode)}」",
                        id={"type": "pipe-launch", "stage": stage_id},
                        color="primary", size="sm",
                    ),
                    html.Span(eta, style={"fontSize": "11px", "color": PALETTE["warn" if False else "muted"], "marginLeft": "10px"}),
                ],
                style={"marginTop": "10px"},
            )
        )
    return html.Div(body)


def _collect_params(stage_id: str, pvals, pids) -> dict:
    """面板参数 → 会话 params：默认参数上覆盖面板里编辑过的子集。"""
    st = pipe_stages.stage(stage_id)
    mode = st.get("mode") if st else None
    if not mode:
        return {}
    params = registry.default_params(mode)
    schema = {p["key"]: p for p in pipe_stages.stage_param_schema(stage_id)}
    for val, pid in zip(pvals, pids):
        key = pid.get("key")
        if key not in schema:
            continue
        if schema[key]["type"] == "bool":
            params[key] = key in (val or [])
        elif val is not None and val != "":
            params[key] = val
    return params


# --- layout ----------------------------------------------------------------------
def layout():
    return dbc.Container(
        [
            page_header(
                "流水线 Pipeline",
                "数据 → 因子 → 训练 → 验证 → 门控 → 交易：点击任一阶段展开操作面板",
                flow_step=None,
            ),
            html.Div(id="pipe-row"),
            html.Div(id="pipe-notice", style={"marginTop": "10px"}),
            html.Div(id="pipe-panel", className="chart-card", style={"marginTop": "12px"}),
            dcc.Store(id="pipe-selected", data=None),
            dcc.Interval(id="pipe-poll", interval=10000),
            dcc.Location(id="pipe-redirect", refresh=True),
        ],
        fluid=True,
    )


# --- callbacks --------------------------------------------------------------------
def register(app):
    @app.callback(
        Output("pipe-selected", "data"),
        Input({"type": "pipe-node", "stage": ALL}, "n_clicks"),
        State("pipe-selected", "data"),
        prevent_initial_call=True,
    )
    def _select(n_clicks, current):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not any(n_clicks or []):
            return no_update
        # 再点同一节点 = 收起面板
        return None if trig["stage"] == current else trig["stage"]

    @app.callback(
        Output("pipe-row", "children"),
        Output("pipe-panel", "children"),
        Output("pipe-notice", "children"),
        Output("pipe-redirect", "pathname"),
        Input("pipe-poll", "n_intervals"),
        Input("pipe-selected", "data"),
        Input({"type": "pipe-launch", "stage": ALL}, "n_clicks"),
        State({"type": "pipe-param", "key": ALL}, "value"),
        State({"type": "pipe-param", "key": ALL}, "id"),
    )
    def _refresh(_n, selected, _launch, pvals, pids):
        notice, redirect = no_update, no_update
        trig = ctx.triggered_id
        if isinstance(trig, dict) and trig.get("type") == "pipe-launch":
            stage_id = trig["stage"]
            st = pipe_stages.stage(stage_id)
            mode = st.get("mode") if st else None
            if mode:
                params = _collect_params(stage_id, pvals, pids)
                sid = scheduler.create(mode, f"流水线·{st['title']}", params)
                scheduler.tick()
                notice = dbc.Alert(
                    [
                        f"已创建会话 {sid}（{registry.MODE_LABELS.get(mode, mode)}）。",
                        dcc.Link("查看详情 →", href=f"/runs/{sid}", style={"marginLeft": "8px"}),
                    ],
                    color="success", className="py-2",
                )
        scheduler.tick()
        return _pipeline_row(selected), _stage_panel(selected), notice, redirect
