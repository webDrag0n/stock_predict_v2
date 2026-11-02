"""会话详情页（/runs/<id>；旧路由 /session/<id> 由 app.py 重定向至此）。

· 顶部「返回」链接按会话模式指回来源页（_back_target：nextday→交易台、
  factoreval→因子工作台、其余→模型实验室；历史 backtest/robustness 容错归模型实验室）。
· queued/running：进度条（步级 + 子阶段加权）+ 当前 phase/ETA/extras +
  步骤清单（当前步骤高亮副标题）+ 日志尾巴（**不渲染步骤详情**，满足运行时不可查看）。
· done：header 卡下方渲染「下一步去哪」指引条（next_step_alert，按模式给文案），
  然后逐步渲染 step report（KPI/图表/表格）。
· failed：失败 Alert（异常类型+消息+失败时所在 phase + 「以相同参数重试」按钮，
  已下线的历史模式禁用并提示），traceback 默认折叠，+ 已完成步骤报告 + run.log 尾巴。

布局是 zero-arg 函数；真正内容由 dcc.Interval 驱动的回调按 url 填充。
"""

from __future__ import annotations

from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from .. import sessions
from ..components import humanize_model_text, next_step_alert
from ..render import render_step
from ..theme import PALETTE
from pipeline import registry

_STATUS_ZH = {
    "running": "运行中",
    "done": "完成",
    "failed": "失败",
    "killed": "已终止",
    "queued": "排队中",
}
_STATUS_COLOR = {
    "running": "warning",
    "done": "success",
    "failed": "danger",
    "killed": "secondary",
    "queued": "info",
}

_LOG_STYLE = {
    "background": "#0f172a",
    "color": "#e2e8f0",
    "padding": "14px",
    "borderRadius": "8px",
    "fontSize": "12px",
    "maxHeight": "380px",
    "overflow": "auto",
    "marginTop": "10px",
    "fontFamily": "Menlo, Consolas, monospace",
    "whiteSpace": "pre-wrap",
}

# 完成态「下一步去哪」指引：mode → (文案, 目标路由)。历史模式（backtest/robustness）
# 不在其中 —— 不显示指引条。
_NEXT_STEP = {
    "factoreval": ("评估结果已写入因子工作台，去看看因子的 RankIC/ICIR 与分层表现", "/factors"),
    "train": ("候选已在 staging，去模型实验室对比并决定是否晋升", "/models"),
    "walkforward": ("回模型实验室看验证时间线，与历史结果对比", "/models"),
    "modeleval": ("去模型实验室继续评估或发起 Walk-Forward 回测验证", "/models"),
    "nextday": ("订单已生成，回交易台执行", "/"),
}


def _back_target(mode: str | None) -> tuple[str, str]:
    """顶部返回链接的 (href, 页面名)：按会话模式指回它的来源工作台。

    nextday → 交易台；factoreval → 因子工作台；walkforward/train/modeleval
    以及未知/历史模式（backtest/robustness）→ 模型实验室。
    """
    if mode == "nextday":
        return "/", "交易台"
    if mode == "factoreval":
        return "/factors", "因子工作台"
    return "/models", "模型实验室"


def _sid_from_path(pathname: str | None) -> str | None:
    """从 url 提取 sid：兼容 /runs/<id>（新）与 /session/<id>（旧，app.py 会重定向）。"""
    for prefix in ("/runs/", "/session/"):
        if pathname and pathname.startswith(prefix):
            return pathname.split(prefix, 1)[1].strip("/")
    return None


def retry_session(sid: str | None) -> str | None:
    """以相同参数重试：创建同 mode/label/params 的新会话并推进调度，返回新 sid。

    无效 sid、会话不存在、或模式已下线（不在 registry.STEPS，如历史
    backtest/robustness）时返回 None，不做任何副作用。
    """
    if not sid or not sessions.valid_sid(sid):
        return None
    meta = sessions.get(sid)
    if not meta:
        return None
    mode = meta.get("mode")
    if mode not in registry.STEPS:
        return None
    new_sid = sessions.create(mode, meta.get("label", ""), meta.get("params") or {})
    sessions.tick()
    return new_sid


def layout():
    return dbc.Container(
        [
            html.Div(id="sd-back"),
            html.Div(id="sd-header"),
            html.Div(id="sd-body"),
            dcc.Interval(id="sd-poll", interval=2000),
            dcc.Location(id="sd-redirect", refresh=True),
        ],
        fluid=True,
    )


def _params_summary(meta: dict) -> str:
    p = meta.get("params", {})
    return humanize_model_text(" · ".join(f"{k}={v}" for k, v in p.items()))


def _fmt_extra(extras: dict) -> str:
    """把 extras 字典格式化成人类可读的小尾巴。"""
    if not extras:
        return ""
    parts = []
    if "date" in extras:
        parts.append(f"@{extras['date']}")
    if "total_value" in extras:
        parts.append(f"总值¥{extras['total_value']:,.0f}")
    if "n_positions" in extras:
        parts.append(f"持仓{int(extras['n_positions'])}")
    if "n_fill" in extras:
        parts.append(f"成交{int(extras['n_fill'])}/{int(extras.get('n_reject', 0))}")
    if "n_train_rows" in extras:
        parts.append(f"训练行{int(extras['n_train_rows']):,}")
    if "n_score_days" in extras:
        parts.append(f"IC打分{int(extras['n_score_days'])}天")
    if "n_today_rows" in extras:
        parts.append(f"预测{int(extras['n_today_rows']):,}")
    if "n_decision_days" in extras:
        parts.append(f"{int(extras['n_decision_days'])} 决策日")
    if "n_days" in extras:
        parts.append(f"{int(extras['n_days'])} 日")
    if "n_trades" in extras:
        parts.append(f"{int(extras['n_trades']):,} 笔")
    if "date_range" in extras:
        parts.append(str(extras["date_range"]))
    return "  ·  ".join(parts)


def _progress_subline(cur_prog: dict) -> str:
    """当前 phase + elapsed/ETA + extras 关键指标，一行小字。"""
    if not cur_prog:
        return ""
    phase = cur_prog.get("phase", "")
    cur = cur_prog.get("current", 0)
    tot = cur_prog.get("total", 1)
    el = cur_prog.get("elapsed_sec", 0)
    eta = cur_prog.get("eta_sec", 0)
    msg = cur_prog.get("message", "")
    base = f"▸ {phase}  ({cur}/{tot} · {cur_prog.get('pct',0):.0%})  ·  已用 {el:.0f}s  ·  预计余 {eta:.0f}s"
    if msg:
        base += f"  ·  {msg}"
    extra = _fmt_extra(cur_prog.get("extras") or {})
    if extra:
        base += f"\n  {extra}"
    return base


def _retry_block(mode: str | None):
    """失败 Alert 内的重试运行：按钮 + 历史模式（已下线）的禁用提示。"""
    offline = mode not in registry.STEPS
    children = [
        dbc.Button(
            "以相同参数重试",
            id="sd-retry",
            color="danger",
            outline=True,
            size="sm",
            disabled=offline,
            title=(
                "该模式已下线，无法重试"
                if offline
                else "按本会话的 mode/label/params 创建新会话并排队"
            ),
        ),
    ]
    if offline:
        children.append(
            html.Span(
                f"「{registry.MODE_LABELS.get(mode, mode)}」模式已下线，请改用现有模式重建任务。",
                style={"fontSize": "12px", "color": "#7f1d1d", "marginLeft": "8px"},
            )
        )
    return html.Div(children, style={"marginTop": "8px"})


def _render_detail(sid: str | None):
    """按 sid 渲染 (返回链接, header, body, 是否停止轮询)；回调与测试共用的主体。"""
    if sid is None:
        return html.Div(), html.Div(), html.Div(), True
    if not sessions.valid_sid(sid):
        return html.Div(), dbc.Alert("非法会话 ID。", color="danger"), html.Div(), True
    # 详情页轮询同样推进调度：只停留在详情页时队列也不能停摆
    sessions.tick()
    meta = sessions.get(sid)
    if meta is None:
        return html.Div(), dbc.Alert("会话不存在或已删除。", color="warning"), html.Div(), True

    mode = meta.get("mode")
    href, page_name = _back_target(mode)
    back = dcc.Link(f"← 返回{page_name}", href=href, className="page-sub")

    st = meta.get("status", "queued")
    nsteps = len(meta.get("steps", []))
    done = sum(1 for s in meta.get("steps", []) if s["status"] == "done")
    header = html.Div(
        [
            html.Div(
                [
                    html.Span(
                        meta.get("label", sid), style={"fontWeight": 700, "fontSize": "20px"}
                    ),
                    dbc.Badge(
                        _STATUS_ZH.get(st, st),
                        color=_STATUS_COLOR.get(st, "secondary"),
                        className="ms-2",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Code(sid),
                    html.Span(
                        f"  ·  {registry.MODE_LABELS.get(mode, mode)}  ·  {_params_summary(meta)}",
                        style={"color": PALETTE["muted"], "fontSize": "12px"},
                    ),
                ]
            ),
        ],
        className="chart-card",
        style={"marginTop": "10px", "marginBottom": "12px"},
    )

    # 运行中/排队：进度条（步级 + 子阶段加权）+ phase/extras + 步骤清单 + 日志
    if st in ("queued", "running"):
        cur = meta.get("current_step", -1)
        cur_prog = meta.get("current_progress") or {}
        sub_pct = float(cur_prog.get("pct", 0.0)) if cur >= 0 else 0.0
        overall_pct = ((done + sub_pct) / nsteps * 100) if nsteps else 0
        overall_pct = max(0, min(100, overall_pct))

        step_items = []
        for i, s in enumerate(meta.get("steps", [])):
            icon = {"done": "✅", "running": "⏳", "failed": "❌"}.get(s["status"], "⚪")
            line = f"{icon} {i+1}. {s['title']}"
            if s.get("summary"):
                line += f"  — {humanize_model_text(s['summary'])}"
            is_cur = i == cur and s["status"] == "running"
            sub = ""
            if is_cur and cur_prog.get("phase"):
                sub = html.Div(
                    f"   ↳ {_progress_subline(cur_prog)}",
                    style={
                        "color": PALETTE["muted"],
                        "fontSize": "11px",
                        "whiteSpace": "pre-wrap",
                        "marginTop": "2px",
                    },
                )
            step_items.append(
                html.Div(
                    [
                        html.Div(
                            line, style={"fontSize": 13, "fontWeight": (600 if is_cur else 400)}
                        ),
                        sub if sub else html.Div(),
                    ],
                    style={"padding": "4px 0"},
                )
            )

        body = html.Div(
            [
                html.Div(
                    "流水线进行中 · 完成后可查看每步可视化分析",
                    style={"fontWeight": 600, "marginBottom": "8px"},
                ),
                dbc.Progress(
                    value=round(overall_pct, 1),
                    label=f"{overall_pct:.0f}%",
                    striped=True,
                    animated=True,
                    color="warning",
                    style={"marginBottom": "6px"},
                ),
                html.Div(
                    _progress_subline(cur_prog) or "等待子阶段上报…",
                    style={
                        "color": PALETTE["muted"],
                        "fontSize": "12px",
                        "whiteSpace": "pre-wrap",
                        "marginBottom": "12px",
                    },
                ),
                html.Div(step_items),
                html.Div("实时日志", style={"fontWeight": 600, "marginTop": "14px"}),
                html.Pre(sessions.run_log(sid, n=120) or "（暂无输出）", style=_LOG_STYLE),
            ],
            className="chart-card",
        )
        return back, header, body, False  # keep polling

    # done / failed：渲染步骤报告
    reps = sessions.step_reports(sid)
    blocks = [render_step(r) for r in reps]
    if st == "done":
        # 「下一步去哪」指引条：历史模式（不在 _NEXT_STEP）不显示
        nxt = _NEXT_STEP.get(mode)
        if nxt:
            blocks.insert(0, next_step_alert(nxt[0], nxt[1]))
    if st == "failed":
        failure = meta.get("failure") or {}
        if failure:
            pos = failure.get("progress_position") or failure.get("step_title") or ""
            detail = failure.get("progress_detail", "")
            exc_type = failure.get("exc_type", "Exception")
            exc_msg = failure.get("exc_msg", "")
            tb = failure.get("traceback", "")
            where = f"在「{pos}」" + (f"  ·  {detail}" if detail else "") if pos else ""
            blocks.insert(
                0,
                dbc.Alert(
                    [
                        html.Div(
                            [
                                html.Strong(f"❌ 失败：{exc_type}"),
                                html.Span(f"  ·  {exc_msg}", style={"color": "#7f1d1d"}),
                            ],
                            style={"marginBottom": "6px"},
                        ),
                        html.Div(
                            where,
                            style={
                                "fontSize": "12px",
                                "color": "#7f1d1d",
                                "marginBottom": "6px",
                            },
                        ),
                        html.Details(
                            [
                                html.Summary(
                                    "展开 traceback",
                                    style={"cursor": "pointer", "fontSize": "12px"},
                                ),
                                html.Pre(tb, style={**_LOG_STYLE, "maxHeight": "300px"}),
                            ]
                        ),
                        _retry_block(mode),
                    ],
                    color="danger",
                ),
            )
        blocks.append(
            html.Div(
                [
                    html.Div("运行日志（失败排查）", style={"fontWeight": 600}),
                    html.Pre(sessions.run_log(sid, n=160), style=_LOG_STYLE),
                ],
                className="chart-card",
            )
        )
    if not blocks:
        blocks = [dbc.Alert("无步骤报告。", color="secondary")]
    return back, header, html.Div(blocks), True  # stop polling


def register(app):
    @app.callback(
        Output("sd-back", "children"),
        Output("sd-header", "children"),
        Output("sd-body", "children"),
        Output("sd-poll", "disabled"),
        Input("sd-poll", "n_intervals"),
        Input("url", "pathname"),
    )
    def _render(_n, pathname):
        return _render_detail(_sid_from_path(pathname))

    @app.callback(
        Output("sd-redirect", "pathname"),
        Input("sd-retry", "n_clicks"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def _retry(n_clicks, pathname):
        if not n_clicks:
            return no_update
        new_sid = retry_session(_sid_from_path(pathname))
        return f"/runs/{new_sid}" if new_sid else no_update
