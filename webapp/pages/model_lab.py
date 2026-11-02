"""模型实验室 Model Lab（/models）。

对应设计文档 docs/frontend_redesign.html 第 4 节（页面二 · 模型实验室）。
布局自上而下：

  · 页面头（components.page_header，工作流面包屑第 2 步「模型训练与评估」）
  · 引导条（_guidance 按 champion/staging 数据状态给唯一建议动作）
  · 模型版本：Champion 卡片（models/champion/manifest.json：配方摘要 / 晋升评估 /
    模型年龄灯）+ Staging 候选列表（models/staging/*/manifest.json + evaluation.json，
    晋升/删除）
  · 新建任务面板（整行；四种研究模式 walkforward/train/modeleval/factoreval：
    pipeline/registry.py::PARAM_SCHEMA 动态两列表单 + 成本预告 + 参数diff）
  · 验证历史时间线（完成的研究类会话 + 历史 backtest/robustness 会话 + canonical
    outputs/walkforward/meta.json 的 Sharpe 点图，±0.4 噪声带，点击跳 /runs/<id>）
  · Runs 任务表（状态徽章/进度/耗时/产物时间/删除；默认只显示最近 _RUNS_PAGE_SIZE
    条会话，「显示全部」切换；canonical walkforward CLI 置顶行始终显示）

数据 loader 走 webapp/dataio；会话操作走 pipeline/scheduler（MAX_CONCURRENT
必须写 pipeline.scheduler 的模块变量，webapp.sessions 的同名属性只是拷贝）。
dcc.Interval 2.5s 心跳，每次轮询调 scheduler.tick() 推进调度。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc

from .. import dataio
from ..components import model_label, page_header, term
from ..theme import PALETTE, base_fig, empty_fig
from pipeline import registry, scheduler

# --- 常量 --------------------------------------------------------------------
# 模型年龄黄灯阈值：60 交易日 ≈ 84 自然日（MAX_MODEL_AGE_DAYS 的自然日口径）
_AGE_WARN_DAYS = 84
# 同配置随机波动带（设计文档 4.3：防止把噪声当改进）
_NOISE_BAND = 0.4
# Runs 任务表默认只显示最近 N 条会话（「显示全部」切换按钮展开）
_RUNS_PAGE_SIZE = 20

_STATUS_COLOR = {
    "running": "warning",
    "done": "success",
    "failed": "danger",
    "killed": "secondary",
    "queued": "info",
}
_STATUS_ZH = {
    "running": "运行中",
    "done": "完成",
    "failed": "失败",
    "killed": "已终止",
    "queued": "排队中",
}
_MODE_COLOR = {
    "walkforward": PALETTE["strat"],
    "train": PALETTE["impact"],
    "modeleval": PALETTE["bench"],
    "factoreval": PALETTE["up"],
    # 历史模式：已从创建入口移除，仅为旧会话的时间线显示保留配色
    "backtest": PALETTE["commission"],
    "robustness": PALETTE["alpha"],
}
# 成本预告（设计文档 4.4：防止误触长任务）；仅覆盖「新建任务」面板的研究类模式
_ETA_HINT = {
    "walkforward": "预计耗时 ≈ 5 小时（全市场滚动重训，长任务，请确认后再提交）",
    "train": "预计耗时 分钟~小时级（训练 + 门控双回放）",
    "modeleval": "rolling 路径含滚动重训，预计十分钟~小时级；冻结模型逐日打分较快",
    "factoreval": "全库评估预计十分钟级起；勾选诊断更久",
}
# 时间线散点里的 canonical 占位 sid（无会话，不链 /runs/<id>）
_CANONICAL_SID = "__canonical__"

_NUM_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


# --- 通用小工具 ----------------------------------------------------------------
def _fmt_ts(ts) -> str:
    """epoch 秒 / ISO 字符串 → 'YYYY-MM-DD HH:MM'，非法输入返回 '-'。"""
    if not ts:
        return "-"
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        t = pd.to_datetime(ts, errors="coerce")
        return "-" if pd.isna(t) else t.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "-"


def _parse_first_num(v) -> float | None:
    """从 KPI 值（可能是 '+2.48' / '+1.20 ± 0.30' 这类格式化字符串）取首个数字。"""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    m = _NUM_RE.search(str(v))
    return float(m.group()) if m else None


def _audit(action: str) -> None:
    """晋升/删除等敏感操作留审计痕迹：models/promotions.log 追加一行。"""
    try:
        os.makedirs(dataio.MODELS, exist_ok=True)
        with open(os.path.join(dataio.MODELS, "promotions.log"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {action}\n")
    except Exception:
        pass  # 审计失败不阻塞主操作


def _guidance(champion: dict | None, stagings) -> str | None:
    """顶部引导条文案：按模型版本数据状态给唯一建议动作（三态互斥）。

    无 champion → 先训练候选；有 champion 且无 staging → 先跑 walkforward 验证；
    有 staging → 去「模型版本」区对比并决定晋升。返回 None 表示不显示引导条。
    """
    if not champion:
        return "还没有线上模型：先在下方创建「训练候选模型」任务"
    if stagings:
        return "有待晋升候选：在下方「模型版本」区对比并决定晋升"
    return "线上模型运行中：想验证它，先跑一次「滚动Walk-Forward」"


# --- A. Champion 卡片 -----------------------------------------------------------
def _age_badge(man: dict):
    """模型年龄灯：创建至今自然日数，>= _AGE_WARN_DAYS 亮黄灯。"""
    created = pd.to_datetime(man.get("created_at"), errors="coerce")
    if pd.isna(created):
        return dbc.Badge("年龄未知", color="secondary")
    age = (pd.Timestamp.now() - created.tz_localize(None)).days
    if age >= _AGE_WARN_DAYS:
        return dbc.Badge(
            f"{age} 天 · 超龄", color="warning", title=f"超过 {_AGE_WARN_DAYS} 自然日（≈60 交易日）"
        )
    return dbc.Badge(f"{age} 天", color="success")


def _recipe_summary(snapshot: dict) -> str:
    """config_snapshot → 一行配方摘要。"""
    if not snapshot:
        return "无配方快照"
    feats = snapshot.get("feat_cols") or []
    models_txt = "、".join(
        model_label(m) for m in str(snapshot.get("ensemble_models", "?")).split(",") if m
    )
    return (
        f"特征 {len(feats)} 个 · 前瞻 {snapshot.get('fw', '?')}d · "
        f"子模型 [{models_txt}] × {snapshot.get('n_seeds', '?')} seeds · "
        f"训练窗 {snapshot.get('train_window', '?')} · 重建 {snapshot.get('rebuild_days', '?')}d · "
        f"增量树 {snapshot.get('incr_trees', '?')} · IC窗 {snapshot.get('ic_window', '?')}"
    )


def _promotion_block(man: dict):
    """manifest.promotion → 「最近评估结论」块；无记录给空态文案。"""
    promo = man.get("promotion")
    if not promo:
        return html.Div(
            "最近评估：无晋升评估记录（首次引导或原位刷新）。",
            style={"fontSize": "12px", "color": PALETTE["muted"]},
        )
    lines = [
        html.Div(
            f"最近评估：{promo.get('reason', '-')}",
            style={"fontSize": "12px"},
        ),
        html.Div(
            f"判定时间 {_fmt_ts(promo.get('decided_at'))}",
            style={"fontSize": "11px", "color": PALETTE["muted"]},
        ),
    ]
    cm, hm = promo.get("candidate_metrics"), promo.get("champion_metrics")

    def _num(v, fmt="{:+.2f}"):
        return fmt.format(v) if isinstance(v, (int, float)) else str(v if v is not None else "-")

    if cm and hm:
        lines.append(
            html.Div(
                f"候选 Sharpe {_num(cm.get('sharpe'))} vs 冠军 {_num(hm.get('sharpe'))} · "
                f"回撤 {_num(cm.get('max_drawdown'))}% vs {_num(hm.get('max_drawdown'))}%",
                style={"fontSize": "12px", "color": PALETTE["muted"]},
            )
        )
    return html.Div(lines)


def _champion_card(man: dict | None):
    """A. Champion 卡片。无 champion（models/ 为空）→ 首次训练引导空态。"""
    if not man:
        return dbc.Alert(
            [
                html.B("暂无 champion 模型。"),
                " 到 ",
                dcc.Link("交易台", href="/"),
                " 点「生成次日订单」创建出单会话，首个 champion 会自动完成 bootstrap 晋升；"
                "或用下方「新建任务」跑「训练候选模型」后在右侧 staging 手动晋升。",
                html.Div(
                    "建议先跑一次 walkforward（≈5h）验证配方可信度。",
                    style={"fontSize": "12px", "color": PALETTE["muted"]},
                ),
            ],
            color="secondary",
            className="mb-0",
        )
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(html.Code(man.get("model_id", "?"), style={"fontSize": "15px"}), md=8),
                    dbc.Col(_age_badge(man), md=4, style={"textAlign": "right"}),
                ],
                style={"alignItems": "center"},
            ),
            html.Div(
                f"创建 {_fmt_ts(man.get('created_at'))} · 指纹 {man.get('fingerprint', '-')}",
                style={"fontSize": "12px", "color": PALETTE["muted"], "marginTop": "4px"},
            ),
            html.Div(
                _recipe_summary(man.get("config_snapshot") or {}),
                style={"fontSize": "12px", "marginTop": "8px"},
            ),
            html.Details(
                [
                    html.Summary(
                        "全量配方 JSON",
                        style={"fontSize": "12px", "color": PALETTE["commission"]},
                    ),
                    html.Pre(
                        json.dumps(man.get("config_snapshot") or {}, ensure_ascii=False, indent=2),
                        style={"fontSize": "11px", "maxHeight": "220px", "overflowY": "auto"},
                    ),
                ],
                style={"marginTop": "4px"},
            ),
            html.Hr(style={"margin": "10px 0"}),
            _promotion_block(man),
        ]
    )


# --- B. Staging 候选列表 ---------------------------------------------------------
def _staging_summary(entry: dict) -> dict:
    """staging 条目的 evaluation.json → 对比结论（Sharpe 差/回撤差/是否达晋升门槛）。

    evaluation 契约见 src/model_gate.py::run_gate：{candidate{sharpe,max_drawdown,...},
    champion{...}, decision: promoted|rejected, reason, decided_at}。
    """
    ev = entry.get("evaluation") or {}
    cand, champ = ev.get("candidate") or {}, ev.get("champion") or {}
    cs, hs = cand.get("sharpe"), champ.get("sharpe")
    cd, hd = cand.get("max_drawdown"), champ.get("max_drawdown")
    return {
        "id": entry.get("id", "?"),
        "has_eval": bool(ev),
        "d_sharpe": (cs - hs) if (cs is not None and hs is not None) else None,
        "d_dd": (cd - hd) if (cd is not None and hd is not None) else None,
        "cand_sharpe": cs,
        "champ_sharpe": hs,
        "decision": ev.get("decision"),
        "eligible": ev.get("decision") == "promoted",
        "reason": ev.get("reason", ""),
        "decided_at": ev.get("decided_at"),
    }


def _staging_list(entries: tuple[dict, ...]):
    """B. Staging 候选卡片列表；空目录 → 空态。"""
    if not entries:
        return dbc.Alert(
            "暂无 staging 候选。 nightly 门控（model_gate.run_gate）未通过的新配方会留档在这里。",
            color="secondary",
            className="mb-0",
        )
    cards = []
    for e in entries:
        s = _staging_summary(e)
        if s["has_eval"]:
            ds = f"{s['d_sharpe']:+.2f}" if s["d_sharpe"] is not None else "-"
            dd = f"{s['d_dd']:+.1f}pp" if s["d_dd"] is not None else "-"
            gate = (
                dbc.Badge("已达晋升门槛", color="success")
                if s["eligible"]
                else dbc.Badge("未达门槛（可人工强制晋升）", color="warning")
            )
            body = [
                html.Div(
                    [
                        f"ΔSharpe {ds} · Δ回撤 {dd} · ",
                        gate,
                    ],
                    style={"fontSize": "12px"},
                ),
                html.Div(
                    s["reason"],
                    style={"fontSize": "11px", "color": PALETTE["muted"], "marginTop": "4px"},
                ),
                html.Div(
                    f"评估于 {_fmt_ts(s['decided_at'])}",
                    style={"fontSize": "11px", "color": PALETTE["muted"]},
                ),
            ]
        else:
            body = [
                html.Div(
                    "尚无门控评估（evaluation.json 缺失）。",
                    style={"fontSize": "12px", "color": PALETTE["muted"]},
                )
            ]
        cards.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(html.Code(s["id"], style={"fontSize": "13px"}), md=7),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "晋升",
                                            id={"type": "mlab-promote", "mid": s["id"]},
                                            color="success",
                                            size="sm",
                                            outline=True,
                                            style={"marginRight": "6px"},
                                            title="人工晋升：替换 champion（旧冠军归档 archive/）",
                                        ),
                                        dbc.Button(
                                            "删除",
                                            id={"type": "mlab-del-staging", "mid": s["id"]},
                                            color="danger",
                                            size="sm",
                                            outline=True,
                                            title="删除该 staging 留档目录",
                                        ),
                                    ],
                                    md=5,
                                    style={"textAlign": "right"},
                                ),
                            ],
                            style={"alignItems": "center"},
                        ),
                        html.Div(body, style={"marginTop": "6px"}),
                    ]
                ),
                className="mb-2",
            )
        )
    return html.Div(cards)


# --- canonical walkforward 产物 -------------------------------------------------
def _canonical_wf() -> dict | None:
    """outputs/walkforward/meta.json；pos_ratio 缺失时从 window_metrics.parquet 兜底现算。

    meta 契约见 src/walkforward.py::save_walkforward_result：metrics{sharpe,...}
    + sharpe_pos_ratio/predic_pos_ratio（为正窗口占比门禁，可能缺失）。
    """
    wdir = os.path.join(dataio.OUTPUTS, "walkforward")
    meta = dataio._read_json(os.path.join(wdir, "meta.json"))
    if not meta:
        return None
    meta = dict(meta)
    need = {k for k in ("sharpe_pos_ratio", "predic_pos_ratio") if meta.get(k) is None}
    wm_path = os.path.join(wdir, "window_metrics.parquet")
    if need and os.path.exists(wm_path):
        try:
            wm = pd.read_parquet(wm_path)
            if "sharpe_pos_ratio" in need and "Sharpe" in wm.columns and len(wm):
                meta["sharpe_pos_ratio"] = float((wm["Sharpe"] > 0).mean())
            if "predic_pos_ratio" in need and "PredIC" in wm.columns and len(wm):
                meta["predic_pos_ratio"] = float((wm["PredIC"] > 0).mean())
        except Exception:
            pass
    return meta


def _trust_badges(meta: dict) -> list:
    """可信度徽章（设计文档 4.4）：Sharpe>0 窗口占比 <60% 红，PredIC>0 占比 <50% 红。"""
    badges = []
    spr, ppr = meta.get("sharpe_pos_ratio"), meta.get("predic_pos_ratio")
    if spr is not None:
        badges.append(
            dbc.Badge(
                f"Sharpe>0 窗口 {spr:.0%}",
                color="success" if spr >= 0.6 else "danger",
                className="me-1",
            )
        )
    if ppr is not None:
        badges.append(
            dbc.Badge(
                f"PredIC>0 窗口 {ppr:.0%}",
                color="success" if ppr >= 0.5 else "danger",
                className="me-1",
            )
        )
    return badges


def _canonical_modal_body(meta: dict | None):
    """canonical walkforward 的 meta 摘要（CLI 置顶行「摘要」弹层）。"""
    if not meta:
        return dbc.Alert(
            "无 canonical walkforward 产物（outputs/walkforward/）。", color="secondary"
        )
    m = meta.get("metrics") or {}

    def _f(v, fmt="{:+.2f}"):
        return fmt.format(v) if isinstance(v, (int, float)) else "-"

    return html.Div(
        [
            html.Div(_trust_badges(meta), style={"marginBottom": "8px"}),
            html.Div(
                f"生成 {_fmt_ts(meta.get('generated_at'))} · {meta.get('n_windows', '?')} 个 OOS 窗口"
            ),
            html.Ul(
                [
                    html.Li(
                        f"Sharpe {_f(m.get('sharpe'))} · 年化 {_f(m.get('ann_return'), '{:.1%}')}"
                        f" · 最大回撤 {_f(m.get('max_drawdown'), '{:.1%}')}"
                    ),
                    html.Li(
                        f"总 PnL {_f(m.get('total_pnl'), '{:+.4f}')} · PredIC "
                        f"{_f(m.get('pred_ic'), '{:+.4f}')}"
                    ),
                ],
                style={"fontSize": "13px"},
            ),
            html.Div(
                f"±{_NOISE_BAND} 为同配置随机波动带，带内差异不构成改进证据。",
                style={"fontSize": "12px", "color": PALETTE["muted"]},
            ),
        ]
    )


# --- C. 验证历史时间线 -----------------------------------------------------------
def _session_sharpe(sid: str) -> float | None:
    """从会话 step reports 的 KPI 提取 Sharpe：优先精确标签，其次含 'sharpe' 的标签。"""
    try:
        reps = scheduler.step_reports(sid)
    except Exception:
        return None
    kpis = [k for r in reps for k in (r.get("kpis") or [])]
    for pred in (lambda lb: lb == "sharpe", lambda lb: "sharpe" in lb):
        for k in kpis:
            if pred(str(k.get("label", "")).lower()):
                v = _parse_first_num(k.get("value"))
                if v is not None:
                    return v
    return None


def _timeline_points() -> list[dict]:
    """收集时间线散点：完成的研究类会话（_MODE_COLOR 覆盖，含历史 backtest/robustness）
    + canonical CLI 产物。

    每点 {t: datetime, sharpe, mode, sid, canonical}；canonical 点 sid=_CANONICAL_SID。
    """
    points = []
    for s in scheduler.list_sessions():
        mode = s.get("mode")
        if mode not in _MODE_COLOR or s.get("status") != "done":
            continue
        sh = _session_sharpe(s["id"])
        if sh is None:
            continue
        ts = s.get("finished") or s.get("created")
        try:
            t = datetime.fromtimestamp(float(ts))
        except Exception:
            continue
        points.append({"t": t, "sharpe": sh, "mode": mode, "sid": s["id"], "canonical": False})
    meta = _canonical_wf()
    if meta:
        sh = (meta.get("metrics") or {}).get("sharpe")
        t = pd.to_datetime(meta.get("generated_at"), errors="coerce")
        if sh is not None and not pd.isna(t):
            points.append(
                {
                    "t": t.to_pydatetime(),
                    "sharpe": float(sh),
                    "mode": "walkforward",
                    "sid": _CANONICAL_SID,
                    "canonical": True,
                }
            )
    return sorted(points, key=lambda p: p["t"])


def _timeline_fig(points: list[dict]) -> go.Figure:
    """C. 验证历史时间线：按模式分色的 Sharpe 散点 + ±0.4 随机波动带。

    canonical 用大号菱形；波动带锚定最近一个点（同一配置的连续验证应落在带内）。
    """
    if not points:
        return empty_fig("暂无验证记录：完成的研究类会话（含历史回测/稳健性）会在此打点", 340)
    fig = base_fig("验证历史 · Sharpe 时间线", height=340)
    for mode, color in _MODE_COLOR.items():
        pts = [p for p in points if p["mode"] == mode and not p["canonical"]]
        if not pts:
            continue
        fig.add_trace(
            go.Scatter(
                x=[p["t"] for p in pts],
                y=[p["sharpe"] for p in pts],
                mode="markers",
                name=registry.MODE_LABELS.get(mode, mode),
                marker=dict(color=color, size=9),
                customdata=[[p["sid"]] for p in pts],
                hovertemplate="%{x|%Y-%m-%d}<br>Sharpe %{y:+.2f}<br>%{customdata}<extra></extra>",
            )
        )
    canon = [p for p in points if p["canonical"]]
    for p in canon:
        fig.add_trace(
            go.Scatter(
                x=[p["t"]],
                y=[p["sharpe"]],
                mode="markers",
                name="canonical（CLI）",
                marker=dict(
                    color=PALETTE["bench"],
                    size=16,
                    symbol="diamond",
                    line=dict(color=PALETTE["ink"], width=1),
                ),
                customdata=[[_CANONICAL_SID]],
                hovertemplate="%{x|%Y-%m-%d}<br>Sharpe %{y:+.2f}<br>canonical walkforward（CLI）"
                "<extra></extra>",
            )
        )
    anchor = points[-1]["sharpe"]
    fig.add_hrect(
        y0=anchor - _NOISE_BAND,
        y1=anchor + _NOISE_BAND,
        fillcolor=PALETTE["faint"],
        opacity=0.12,
        line_width=0,
        annotation_text=f"±{_NOISE_BAND} 随机波动带",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color=PALETTE["muted"],
    )
    fig.update_yaxes(title="Sharpe")
    fig.update_layout(clickmode="event")
    return fig


# --- D. 新建任务面板 -------------------------------------------------------------
def _param_inputs(mode: str):
    """按 registry.PARAM_SCHEMA[mode] 动态渲染参数输入控件（整行面板内两列排布）。"""
    items = []
    for p in registry.PARAM_SCHEMA[mode]:
        cid = {"type": "mlab-param", "mode": mode, "key": p["key"]}
        if p["type"] == "number":
            ctrl = dbc.Input(
                type="number",
                value=p["default"],
                id=cid,
                min=p.get("min"),
                max=p.get("max"),
                step=p.get("step"),
                size="sm",
            )
        elif p["type"] == "select":
            ctrl = dcc.Dropdown(
                id=cid,
                value=p["default"],
                clearable=False,
                options=[{"label": o, "value": o} for o in p["options"]],
            )
        elif p["type"] == "bool":
            ctrl = dbc.Checklist(
                id=cid,
                value=([p["key"]] if p["default"] else []),
                options=[{"label": "", "value": p["key"]}],
                switch=True,
            )
        elif p["type"] == "textarea":
            ctrl = dbc.Textarea(
                id=cid,
                value=p.get("default", ""),
                placeholder=p.get("placeholder", ""),
                style={"height": "80px", "fontSize": "12px", "fontFamily": "monospace"},
            )
        elif p["type"] == "text":
            ctrl = dbc.Input(
                type="text",
                value=p.get("default", ""),
                id=cid,
                placeholder=p.get("placeholder", ""),
                size="sm",
            )
        else:
            ctrl = dbc.Input(value=p["default"], id=cid, size="sm")
        items.append(
            html.Div(
                [
                    html.Div(p["label"], className="control-label"),
                    ctrl,
                    html.Div(
                        p.get("hint", ""),
                        style={"fontSize": "11px", "color": PALETTE["muted"]},
                    ),
                ],
                style={"marginBottom": "10px"},
            )
        )
    half = (len(items) + 1) // 2
    return dbc.Row(
        [dbc.Col(items[:half], md=6), dbc.Col(items[half:], md=6)],
        className="g-3",
    )


def _collect_params(mode: str, pvals, pids) -> dict:
    """只取当前激活模式的参数；bool 用 checklist 是否含 key 判定。"""
    params = registry.default_params(mode)
    for val, pid in zip(pvals, pids):
        if pid.get("mode") != mode:
            continue
        key = pid["key"]
        schema = next((p for p in registry.PARAM_SCHEMA[mode] if p["key"] == key), None)
        if schema and schema["type"] == "bool":
            params[key] = key in (val or [])
        elif val is not None and val != "":
            params[key] = val
    return params


def _recipe_diff_lines(mode: str, params: dict, man: dict | None) -> list[str]:
    """参数diff预览：本次提交参数 vs champion 配方快照（仅配方重叠键）。"""
    snap = (man or {}).get("config_snapshot") or {}
    lines = []
    for k in ("train_window", "rebuild_days", "incr_trees", "ic_window", "n_seeds"):
        if k in params and k in snap and params[k] != snap[k]:
            lines.append(f"{k}: champion={snap[k]} → 本次={params[k]}")
    return lines


def _create_notice(mode: str, sid: str, params: dict, man: dict | None) -> dbc.Alert:
    """创建成功 Alert：会话 id + 成本预告 + 参数diff + 「查看详情 →」链接。"""
    diffs = _recipe_diff_lines(mode, params, man)
    diff_txt = "；".join(diffs) if diffs else "与 champion 配方无差异（或无 champion）"
    return dbc.Alert(
        [
            html.Span(f"已创建会话 {sid}（{_ETA_HINT.get(mode, '')}）。参数diff：{diff_txt}。"),
            dcc.Link("查看详情 →", href=f"/runs/{sid}", style={"marginLeft": "8px"}),
        ],
        color="success",
    )


def _create_card():
    """D. 新建任务面板：研究类四模式 tab + 动态表单 + 成本预告 + 并发上限控件。"""
    research_modes = [m for m in registry.STEPS if registry.MODE_CATEGORY.get(m) == "research"]
    tabs = dbc.Tabs(
        [
            dbc.Tab(
                html.Div(
                    [
                        dbc.Alert(
                            _ETA_HINT.get(m, ""),
                            color="warning" if m in ("walkforward", "train") else "light",
                            className="py-1 px-2 mt-2 mb-1",
                            style={"fontSize": "12px"},
                        ),
                        _param_inputs(m),
                    ],
                    style={"paddingTop": "4px"},
                ),
                label=registry.MODE_LABELS.get(m, m),
                tab_id=m,
            )
            for m in research_modes
        ],
        id="mlab-mode-tabs",
        active_tab="walkforward",
    )
    return html.Div(
        [
            html.Div(
                "新建任务 · 模块化流水线",
                className="control-label",
                style={"fontSize": "15px", "fontWeight": 600, "marginBottom": "8px"},
            ),
            tabs,
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("会话名称（可选）", className="control-label"),
                            dbc.Input(id="mlab-label", placeholder="留空自动命名", size="sm"),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Div("并发上限", className="control-label"),
                            dbc.Input(
                                id="mlab-concurrency",
                                type="number",
                                value=scheduler.MAX_CONCURRENT,
                                min=1,
                                max=8,
                                step=1,
                                size="sm",
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "＋ 创建并排队",
                            id="mlab-create",
                            color="success",
                            className="w-100",
                            style={"marginTop": "22px"},
                        ),
                        md=3,
                    ),
                ],
                className="g-2",
                style={"marginTop": "8px"},
            ),
        ],
        className="chart-card",
    )


# --- E. Runs 任务表 ---------------------------------------------------------------
def _fmt_eta(sec) -> str:
    """eta_sec 秒 → 可读文本（'42s' / '7m' / '1.5h'）；非法/非正值返回 ''。"""
    try:
        sec = float(sec)
    except (TypeError, ValueError):
        return ""
    if sec <= 0:
        return ""
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.0f}m"
    return f"{sec / 3600:.1f}h"


def _overall_pct(m: dict) -> float:
    """整体进度 0-100：done 步数 + 当前步 pct 加权（口径同 session_detail）。

    current_progress 缺省/字段非法时按当前步 0% 容错；nsteps=0 返回 0。
    """
    steps = m.get("steps") or []
    nsteps = len(steps)
    if not nsteps:
        return 0.0
    done = sum(1 for s in steps if s.get("status") == "done")
    cur = m.get("current_step", -1)
    sub = 0.0
    if 0 <= cur < nsteps:
        cur_prog = m.get("current_progress") or {}
        try:
            sub = float(cur_prog.get("pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            sub = 0.0
        sub = max(0.0, min(1.0, sub))
    return max(0.0, min(100.0, (done + sub) / nsteps * 100))


def _progress_cell(m: dict):
    """进度列：running → Progress 条 + phase · message · ETA 小字；queued → 排队中；
    终态 → done数/总步数。"""
    st = m.get("status", "queued")
    steps = m.get("steps") or []
    nsteps = len(steps)
    done = sum(1 for s in steps if s.get("status") == "done")
    if st == "running":
        pct = _overall_pct(m)
        cur_prog = m.get("current_progress") or {}
        eta = _fmt_eta(cur_prog.get("eta_sec"))
        sub = " · ".join(
            x
            for x in (
                cur_prog.get("phase", ""),
                cur_prog.get("message", ""),
                f"ETA {eta}" if eta else "",
            )
            if x
        )
        return html.Div(
            [
                dbc.Progress(
                    value=round(pct, 1),
                    label=f"{pct:.0f}%",
                    color="warning",
                    striped=True,
                    animated=True,
                    style={"height": "14px", "fontSize": "10px"},
                ),
                html.Div(
                    sub or "等待子阶段上报…",
                    style={
                        "fontSize": 11,
                        "color": PALETTE["muted"],
                        "marginTop": "2px",
                        "whiteSpace": "pre-wrap",
                    },
                ),
            ]
        )
    if st == "queued":
        return html.Span("排队中", style={"fontSize": 12, "color": PALETTE["muted"]})
    return html.Span(f"{done}/{nsteps}", style={"fontSize": 12})


def _runs_table(canonical: dict | None, show_all: bool = False):
    """Runs 任务表：canonical walkforward 置顶 CLI 行（始终显示）+ 会话行。

    会话按创建时间倒序（list_sessions 已排序），默认只渲染最近 _RUNS_PAGE_SIZE 条，
    show_all=True 渲染全部；底部「显示全部」切换按钮在 layout 静态区，文案/显隐
    由 _refresh 根据会话总数控制。
    """
    header = dbc.Row(
        [
            dbc.Col(html.B("ID"), md=3),
            dbc.Col(html.B("模式"), md=1),
            dbc.Col(html.B("状态"), md=1),
            dbc.Col(html.B("进度"), md=2),
            dbc.Col(html.B("耗时"), md=1),
            dbc.Col(html.B("产物时间"), md=2),
            dbc.Col(html.B("操作"), md=2),
        ],
        style={"borderBottom": "2px solid #e5e7eb", "padding": "6px 0", "fontSize": 13},
    )
    rows = [header]
    if canonical:
        m = canonical.get("metrics") or {}
        sh = m.get("sharpe")
        rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Code("outputs/walkforward", style={"fontSize": "12px"}),
                            html.Br(),
                            dbc.Badge("CLI", color="dark", className="mt-1"),
                        ],
                        md=3,
                    ),
                    dbc.Col("滚动Walk-Forward", md=1, style={"fontSize": 13}),
                    dbc.Col(dbc.Badge("canonical", color="primary"), md=1),
                    dbc.Col(
                        html.Span(
                            f"Sharpe {sh:+.2f}" if isinstance(sh, (int, float)) else "-",
                            style={"fontSize": 12},
                        ),
                        md=2,
                    ),
                    dbc.Col("-", md=1),
                    dbc.Col(_fmt_ts(canonical.get("generated_at")), md=2, style={"fontSize": 13}),
                    dbc.Col(
                        dbc.Button(
                            "摘要",
                            # pattern-matching：该按钮由 _runs_table 动态渲染，
                            # 字符串 id 作 Input 会在组件缺失时让整页回调报错
                            id={"type": "mlab-cli-btn", "role": "open"},
                            color="primary",
                            size="sm",
                            outline=True,
                        ),
                        md=2,
                    ),
                ],
                style={
                    "padding": "8px 0",
                    "borderBottom": "1px solid #f0f0f0",
                    "alignItems": "center",
                    "background": PALETTE["bg"],
                },
            )
        )
    sessions = scheduler.list_sessions()
    if not sessions and not canonical:
        return dbc.Alert("暂无任务。用上方「新建任务」面板创建一条流水线。", color="secondary")
    for m in sessions if show_all else sessions[:_RUNS_PAGE_SIZE]:
        st = m.get("status", "queued")
        el = int(scheduler.elapsed(m))
        viewable = st in ("done", "failed", "killed")
        rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Code(m["id"], style={"fontSize": "12px"}),
                            html.Br(),
                            html.Span(
                                m.get("label", ""),
                                style={"fontSize": 11, "color": PALETTE["muted"]},
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        registry.MODE_LABELS.get(m.get("mode"), m.get("mode", "-")),
                        md=1,
                        style={"fontSize": 13},
                    ),
                    dbc.Col(
                        dbc.Badge(_STATUS_ZH.get(st, st), color=_STATUS_COLOR.get(st, "secondary")),
                        md=1,
                    ),
                    dbc.Col(_progress_cell(m), md=2),
                    dbc.Col(f"{el // 60}m{el % 60}s", md=1, style={"fontSize": 13}),
                    dbc.Col(
                        _fmt_ts(m.get("finished") or m.get("created")), md=2, style={"fontSize": 13}
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                "查看",
                                id={"type": "mlab-view", "sid": m["id"]},
                                color="primary",
                                size="sm",
                                outline=True,
                                disabled=(not viewable),
                                title=("运行中/排队不可查看" if not viewable else ""),
                                style={"marginRight": "6px"},
                            ),
                            dbc.Button(
                                "删除",
                                id={"type": "mlab-del-run", "sid": m["id"]},
                                color="danger",
                                size="sm",
                                outline=True,
                                disabled=(st == "running"),
                                title=("运行中不可删除" if st == "running" else ""),
                            ),
                        ],
                        md=2,
                    ),
                ],
                style={
                    "padding": "8px 0",
                    "borderBottom": "1px solid #f0f0f0",
                    "alignItems": "center",
                },
            )
        )
    return html.Div(rows)


# --- 敏感操作 ---------------------------------------------------------------------
def _do_promote(mid: str) -> tuple[bool, str]:
    """人工晋升 staging 候选为 champion（webapp 进程内直接 import src/model_store）。

    run_webapp.py 已把 src/ 注入 sys.path，与 predict_tomorrow.py --promote 同路径；
    候选缺工件（ensemble_state.pkl）等异常按失败返回，不崩回调。
    """
    if not scheduler.valid_sid(mid):
        return False, f"非法 model_id: {mid!r}"
    try:
        import model_store

        record = {
            "reason": "webapp 人工晋升（模型实验室）",
            "decided_at": datetime.now().isoformat(timespec="seconds"),
        }
        model_store.promote(mid, promotion_record=record)
    except Exception as e:
        return False, f"晋升失败：{type(e).__name__}: {e}"
    _audit(f"PROMOTE {mid} by=webapp-model-lab")
    dataio.refresh_caches()
    return True, f"已晋升 {mid} 为 champion（旧冠军归档 archive/，审计见 models/promotions.log）"


def _do_delete_staging(mid: str) -> tuple[bool, str]:
    """删除 staging 留档目录。"""
    if not scheduler.valid_sid(mid):
        return False, f"非法 model_id: {mid!r}"
    d = os.path.join(dataio.MODELS, "staging", mid)
    if not os.path.isdir(d):
        return False, f"staging 候选不存在: {mid}"
    try:
        shutil.rmtree(d)
    except Exception as e:
        return False, f"删除失败：{type(e).__name__}: {e}"
    _audit(f"DELETE_STAGING {mid} by=webapp-model-lab")
    dataio.refresh_caches()
    return True, f"已删除 staging 候选 {mid}（审计见 models/promotions.log）"


# --- layout ----------------------------------------------------------------------
def layout():
    return dbc.Container(
        [
            page_header(
                "模型实验室 Model Lab",
                "线上用的是哪个模型？新配方值得晋升吗？回测可信吗？",
                flow_step=1,
            ),
            html.Div(id="mlab-guidance"),
            html.Div(id="mlab-notice"),
            html.Div(
                [
                    html.Div(
                        "模型版本",
                        className="control-label",
                        style={"fontSize": "15px", "fontWeight": 600},
                    ),
                    html.Div(
                        [
                            term("champion", id_prefix="mlab-ver-champion"),
                            " = 线上在用的生产模型（订单只出自它）；",
                            term("staging", id_prefix="mlab-ver-staging"),
                            " = 待晋升候选，与 champion 做门控对比后决定是否替换。",
                        ],
                        style={
                            "fontSize": "12px",
                            "color": PALETTE["muted"],
                            "marginBottom": "8px",
                        },
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(
                                    "Champion 生产模型",
                                    className="control-label",
                                    style={
                                        "fontSize": "15px",
                                        "fontWeight": 600,
                                        "marginBottom": "8px",
                                    },
                                ),
                                html.Div(id="mlab-champion"),
                            ],
                            className="chart-card",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(
                                    "Staging 候选（vs champion 门控对比）",
                                    className="control-label",
                                    style={
                                        "fontSize": "15px",
                                        "fontWeight": 600,
                                        "marginBottom": "8px",
                                    },
                                ),
                                html.Div(id="mlab-staging"),
                            ],
                            className="chart-card",
                        ),
                        md=6,
                    ),
                ],
                className="g-3",
            ),
            dbc.Row(
                dbc.Col(_create_card(), md=12),
                className="g-3",
                style={"marginTop": "4px"},
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.Div(
                                "验证历史时间线（点击点跳 /runs/<id>）",
                                className="control-label",
                                style={
                                    "fontSize": "15px",
                                    "fontWeight": 600,
                                    "marginBottom": "8px",
                                },
                            ),
                            dcc.Graph(id="mlab-timeline", config={"displayModeBar": False}),
                            html.Div(
                                [
                                    "同配方 Sharpe 逐次下滑 = 信号衰减；±0.4 为",
                                    term("噪声带", id_prefix="mlab-tl-noise"),
                                    "，带内差异是噪声，不算改进。预测质量看 ",
                                    term("PredIC", id_prefix="mlab-tl-predic"),
                                    "，候选能否上线由",
                                    term("门控", id_prefix="mlab-tl-gate"),
                                    "判定。",
                                ],
                                style={"fontSize": "11px", "color": PALETTE["muted"]},
                            ),
                        ],
                        className="chart-card",
                    ),
                    md=12,
                ),
                className="g-3",
                style={"marginTop": "4px"},
            ),
            html.Div(
                [
                    html.Div(
                        "Runs 任务表",
                        className="control-label",
                        style={"fontSize": "15px", "fontWeight": 600, "marginBottom": "8px"},
                    ),
                    html.Div(id="mlab-runs"),
                    dbc.Button(
                        "",
                        id="mlab-runs-toggle",
                        color="link",
                        size="sm",
                        className="px-0",
                        style={"display": "none"},
                    ),
                ],
                className="chart-card",
                style={"marginTop": "16px"},
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("canonical walkforward（CLI 产物）")),
                    dbc.ModalBody(id="mlab-cli-body"),
                    dbc.ModalFooter(
                        dbc.Button("关闭", id="mlab-cli-close", size="sm", outline=True)
                    ),
                ],
                id="mlab-cli-modal",
                is_open=False,
            ),
            dcc.Store(id="mlab-runs-show-all", data=False),
            dcc.Interval(id="mlab-poll", interval=2500),
            dcc.Location(id="mlab-redirect", refresh=True),
        ],
        fluid=True,
    )


# --- callbacks --------------------------------------------------------------------
def register(app):
    @app.callback(
        Output("mlab-champion", "children"),
        Output("mlab-staging", "children"),
        Output("mlab-timeline", "figure"),
        Output("mlab-runs", "children"),
        Output("mlab-notice", "children"),
        Output("mlab-redirect", "pathname"),
        Output("mlab-cli-modal", "is_open"),
        Output("mlab-cli-body", "children"),
        Output("mlab-guidance", "children"),
        Output("mlab-runs-toggle", "children"),
        Output("mlab-runs-toggle", "style"),
        Input("mlab-poll", "n_intervals"),
        Input("mlab-create", "n_clicks"),
        Input({"type": "mlab-view", "sid": ALL}, "n_clicks"),
        Input({"type": "mlab-del-run", "sid": ALL}, "n_clicks"),
        Input({"type": "mlab-promote", "mid": ALL}, "n_clicks"),
        Input({"type": "mlab-del-staging", "mid": ALL}, "n_clicks"),
        Input({"type": "mlab-cli-btn", "role": ALL}, "n_clicks"),
        Input("mlab-cli-close", "n_clicks"),
        Input("mlab-timeline", "clickData"),
        Input("mlab-runs-show-all", "data"),
        State("mlab-mode-tabs", "active_tab"),
        State("mlab-label", "value"),
        State({"type": "mlab-param", "mode": ALL, "key": ALL}, "value"),
        State({"type": "mlab-param", "mode": ALL, "key": ALL}, "id"),
        prevent_initial_call=False,
    )
    def _refresh(
        _n,
        _create,
        _view,
        _del_run,
        _promote,
        _del_stg,
        _cli_open,
        _cli_close,
        click_data,
        show_all,
        mode,
        label,
        pvals,
        pids,
    ):
        trig = ctx.triggered_id
        notice, redirect, modal_open, modal_body = no_update, no_update, no_update, no_update
        if trig == "mlab-create":
            params = _collect_params(mode, pvals, pids)
            sid = scheduler.create(mode, label, params)
            notice = _create_notice(mode, sid, params, dataio.champion_manifest())
        elif isinstance(trig, dict):
            t = trig.get("type")
            if t == "mlab-view":
                redirect = f"/runs/{trig['sid']}"
            elif t == "mlab-del-run":
                ok = scheduler.delete(trig["sid"])
                notice = dbc.Alert(
                    f"已删除会话 {trig['sid']}" if ok else f"删除失败 {trig['sid']}",
                    color="info" if ok else "danger",
                )
            elif t == "mlab-promote":
                ok, msg = _do_promote(trig["mid"])
                notice = dbc.Alert(msg, color="success" if ok else "danger")
            elif t == "mlab-del-staging":
                ok, msg = _do_delete_staging(trig["mid"])
                notice = dbc.Alert(msg, color="info" if ok else "danger")
            elif t == "mlab-cli-btn":
                modal_open, modal_body = True, _canonical_modal_body(_canonical_wf())
        elif trig == "mlab-cli-close":
            modal_open = False
        elif trig == "mlab-timeline" and click_data:
            try:
                sid = click_data["points"][0]["customdata"][0]
            except Exception:
                sid = None
            if sid == _CANONICAL_SID:
                modal_open, modal_body = True, _canonical_modal_body(_canonical_wf())
            elif sid and scheduler.valid_sid(sid):
                redirect = f"/runs/{sid}"
        # 每次心跳推进调度（queued → running、回收退出码）
        scheduler.tick()
        man = dataio.champion_manifest()
        stagings = dataio.staging_manifests()
        g = _guidance(man, stagings)
        guidance = dbc.Alert(g, color="info", className="py-2") if g else None
        n_sessions = len(scheduler.list_sessions())
        if n_sessions <= _RUNS_PAGE_SIZE:
            toggle_children, toggle_style = "", {"display": "none"}
        elif show_all:
            toggle_children, toggle_style = f"收起（只看最近 {_RUNS_PAGE_SIZE} 条）", {}
        else:
            toggle_children, toggle_style = f"显示全部（共 {n_sessions} 条）", {}
        return (
            _champion_card(man),
            _staging_list(stagings),
            _timeline_fig(_timeline_points()),
            _runs_table(_canonical_wf(), show_all=bool(show_all)),
            notice,
            redirect,
            modal_open,
            modal_body,
            guidance,
            toggle_children,
            toggle_style,
        )

    @app.callback(
        Output("mlab-runs-show-all", "data"),
        Input("mlab-runs-toggle", "n_clicks"),
        State("mlab-runs-show-all", "data"),
        prevent_initial_call=True,
    )
    def _toggle_runs(_n, show_all):
        return not show_all

    @app.callback(
        Output("mlab-poll", "disabled"),
        Input("mlab-concurrency", "value"),
    )
    def _set_concurrency(v):
        # 并发上限必须写 pipeline.scheduler 的模块变量：webapp/sessions.py 的
        # MAX_CONCURRENT 只是 import 时的拷贝，写它不生效。
        if v and int(v) >= 1:
            scheduler.MAX_CONCURRENT = int(v)
        return False
