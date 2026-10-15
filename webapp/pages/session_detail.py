"""会话详情页（/session/<id>）。

· queued/running：进度条（步级 + 子阶段加权）+ 当前 phase/ETA/extras +
  步骤清单（当前步骤高亮副标题）+ 日志尾巴（**不渲染步骤详情**，满足运行时不可查看）。
· done：逐步渲染 step report（KPI/图表/表格）。
· failed：失败 Alert（异常类型+消息+失败时所在 phase），traceback 默认折叠，
  + 已完成步骤报告 + run.log 尾巴。

布局是 zero-arg 函数；真正内容由 dcc.Interval 驱动的回调按 url 填充。
"""
from __future__ import annotations

from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from .. import sessions
from ..render import render_step
from ..theme import PALETTE
from pipeline import registry

_STATUS_ZH = {"running": "运行中", "done": "完成", "failed": "失败",
              "killed": "已终止", "queued": "排队中"}
_STATUS_COLOR = {"running": "warning", "done": "success",
                 "failed": "danger", "killed": "secondary", "queued": "info"}

_LOG_STYLE = {"background": "#0f172a", "color": "#e2e8f0", "padding": "14px",
              "borderRadius": "8px", "fontSize": "12px", "maxHeight": "380px",
              "overflow": "auto", "marginTop": "10px",
              "fontFamily": "Menlo, Consolas, monospace", "whiteSpace": "pre-wrap"}


def layout():
    return dbc.Container([
        dcc.Link("← 返回会话列表", href="/", className="page-sub"),
        html.Div(id="sd-header"),
        html.Div(id="sd-body"),
        dcc.Interval(id="sd-poll", interval=2000),
    ], fluid=True)


def _params_summary(meta: dict) -> str:
    p = meta.get("params", {})
    return " · ".join(f"{k}={v}" for k, v in p.items())


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
    cur = cur_prog.get("current", 0); tot = cur_prog.get("total", 1)
    el = cur_prog.get("elapsed_sec", 0); eta = cur_prog.get("eta_sec", 0)
    msg = cur_prog.get("message", "")
    base = f"▸ {phase}  ({cur}/{tot} · {cur_prog.get('pct',0):.0%})  ·  已用 {el:.0f}s  ·  预计余 {eta:.0f}s"
    if msg:
        base += f"  ·  {msg}"
    extra = _fmt_extra(cur_prog.get("extras") or {})
    if extra:
        base += f"\n  {extra}"
    return base


def register(app):
    @app.callback(
        Output("sd-header", "children"),
        Output("sd-body", "children"),
        Output("sd-poll", "disabled"),
        Input("sd-poll", "n_intervals"),
        Input("url", "pathname"),
    )
    def _render(_n, pathname):
        if not pathname or not pathname.startswith("/session/"):
            return html.Div(), html.Div(), True
        sid = pathname.split("/session/", 1)[1].strip("/")
        if not sessions.valid_sid(sid):
            return dbc.Alert("非法会话 ID。", color="danger"), html.Div(), True
        # 详情页轮询同样推进调度：只停留在详情页时队列也不能停摆
        sessions.tick()
        meta = sessions.get(sid)
        if meta is None:
            return dbc.Alert("会话不存在或已删除。", color="warning"), html.Div(), True

        st = meta.get("status", "queued")
        nsteps = len(meta.get("steps", []))
        done = sum(1 for s in meta.get("steps", []) if s["status"] == "done")
        header = html.Div([
            html.Div([
                html.Span(meta.get("label", sid), style={"fontWeight": 700,
                          "fontSize": "20px"}),
                dbc.Badge(_STATUS_ZH.get(st, st),
                          color=_STATUS_COLOR.get(st, "secondary"), className="ms-2"),
            ]),
            html.Div([html.Code(sid), html.Span(
                f"  ·  {registry.MODE_LABELS.get(meta['mode'], meta['mode'])}  ·  {_params_summary(meta)}",
                style={"color": PALETTE["muted"], "fontSize": "12px"})]),
        ], className="chart-card", style={"marginTop": "10px", "marginBottom": "12px"})

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
                    line += f"  — {s['summary']}"
                is_cur = (i == cur and s["status"] == "running")
                sub = ""
                if is_cur and cur_prog.get("phase"):
                    sub = html.Div(
                        f"   ↳ {_progress_subline(cur_prog)}",
                        style={"color": PALETTE["muted"], "fontSize": "11px",
                               "whiteSpace": "pre-wrap", "marginTop": "2px"})
                step_items.append(html.Div([
                    html.Div(line, style={"fontSize": 13,
                                          "fontWeight": (600 if is_cur else 400)}),
                    sub if sub else html.Div(),
                ], style={"padding": "4px 0"}))

            body = html.Div([
                html.Div("流水线进行中 · 完成后可查看每步可视化分析",
                         style={"fontWeight": 600, "marginBottom": "8px"}),
                dbc.Progress(value=round(overall_pct, 1),
                             label=f"{overall_pct:.0f}%",
                             striped=True, animated=True, color="warning",
                             style={"marginBottom": "6px"}),
                html.Div(_progress_subline(cur_prog) or "等待子阶段上报…",
                         style={"color": PALETTE["muted"], "fontSize": "12px",
                                "whiteSpace": "pre-wrap", "marginBottom": "12px"}),
                html.Div(step_items),
                html.Div("实时日志", style={"fontWeight": 600, "marginTop": "14px"}),
                html.Pre(sessions.run_log(sid, n=120) or "（暂无输出）", style=_LOG_STYLE),
            ], className="chart-card")
            return header, body, False    # keep polling

        # done / failed：渲染步骤报告
        reps = sessions.step_reports(sid)
        blocks = [render_step(r) for r in reps]
        if st == "failed":
            failure = meta.get("failure") or {}
            if failure:
                pos = failure.get("progress_position") or failure.get("step_title") or ""
                detail = failure.get("progress_detail", "")
                exc_type = failure.get("exc_type", "Exception")
                exc_msg = failure.get("exc_msg", "")
                tb = failure.get("traceback", "")
                where = f"在「{pos}」" + (f"  ·  {detail}" if detail else "") if pos else ""
                blocks.insert(0, dbc.Alert([
                    html.Div([
                        html.Strong(f"❌ 失败：{exc_type}"),
                        html.Span(f"  ·  {exc_msg}",
                                  style={"color": "#7f1d1d"}),
                    ], style={"marginBottom": "6px"}),
                    html.Div(where, style={"fontSize": "12px",
                                            "color": "#7f1d1d", "marginBottom": "6px"}),
                    html.Details([
                        html.Summary("展开 traceback", style={"cursor": "pointer",
                                                              "fontSize": "12px"}),
                        html.Pre(tb, style={**_LOG_STYLE, "maxHeight": "300px"}),
                    ]),
                ], color="danger"))
            blocks.append(html.Div([
                html.Div("运行日志（失败排查）", style={"fontWeight": 600}),
                html.Pre(sessions.run_log(sid, n=160), style=_LOG_STYLE),
            ], className="chart-card"))
        if not blocks:
            blocks = [dbc.Alert("无步骤报告。", color="secondary")]
        return header, html.Div(blocks), True   # stop polling
