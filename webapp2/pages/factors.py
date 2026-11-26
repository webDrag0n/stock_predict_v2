"""因子：因子库总览、单因子详情、公式编辑器。

  /factors        总览：筛选 + 主表（评估指标 × 衰减 sparkline × 冗余簇 × 模型内贡献）+ 相关热图
  /factors/<id>   详情：定义与公式 + 六面板（逐窗口 IC / 分层净值 / IC 分布 / 相关邻域 / 模型内贡献 / 覆盖面）
  /factors/new    编辑器：校验 → 保存草稿（custom.yaml，enabled=false）→ 一键评估

写操作：保存草稿（actions.save_factor_draft）、发起评估（CLI scripts/run_pipeline.py，登记到 outputs/runs/）。
评估一个公式因子走 --draft：默认公式因子不在因子面板里，需要在本次进程内现算；且 --draft 不读写
共享面板缓存，不会污染夜间任务（原因见 scripts/run_pipeline.py::apply_draft）。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, dcc, html, no_update

import run_registry as rr

from .. import actions, data, factor_logic as fl, ui

ROW_CAP = 120
_GATE_OPTS = ["全部", "已通过", "未通过", "未评估"]
_SRC_OPTS = ["全部", "已启用", "已停用", "原生因子"]


def _runs_root() -> str:
    return os.path.join(data.OUTPUTS, "runs")


# ----------------------------------------------------------------------------- 取数
def overview_frame() -> pd.DataFrame:
    contrib, scores = data.factor_contrib()
    return fl.build_overview(
        data.factor_library(),
        data.factor_eval_table(),
        data.factor_ic_decay(),
        data.factor_corr(),
        scores,
        data.panel_factor_columns(),
    )


def signature() -> list:
    base = os.path.join(data.OUTPUTS, "factor_eval.csv")
    diag = [
        os.path.join(data.OUTPUTS, n)
        for n in ("diag_factor_ic_decay.csv", "diag_factor_corr.csv", "diag_model_vs_factor.csv")
    ]
    runs = [
        (m["run_id"], m.get("status"))
        for m in rr.list_runs(kind="factoreval", limit=3, root=_runs_root())
    ]
    return [
        data.mtime(base),
        [data.mtime(p) for p in diag],
        data.library_signature(),
        data.mtime(os.path.join(data.CACHE, "factor_panel_cache.parquet")),
        data.models_signature(),
        runs,
    ]


# ----------------------------------------------------------------------------- 总览
def _f(v, fmt="{:+.3f}") -> str:
    return ui.fnum(v, fmt)


def _table(df: pd.DataFrame):
    if df.empty:
        return ui.empty("没有符合筛选条件的因子。")
    head = html.Tr(
        [
            html.Th(h, className="n" if k == "n" else "", title=fl._COLUMN_TOOLTIPS.get(c, ""))
            for c, h, k in _COLS
        ]
    )
    rows = []
    for r in df.head(ROW_CAP).to_dict("records"):
        cells = []
        for c, _, k in _COLS:
            v = r.get(c)
            if c == "因子":
                cells.append(html.Td(dcc.Link(str(v), href=f"/factors/{v}", className="mono")))
            elif c == "准入":
                cells.append(
                    html.Td(
                        ui.badge("通过", "done")
                        if v == "✓"
                        else (
                            ui.badge("未通过", "failed")
                            if pd.notna(r.get("RankIC"))
                            else ui.badge("未评估", "pending")
                        )
                    )
                )
            elif c in ("RankIC", "ICIR", "多空收益"):
                cells.append(html.Td(_f(v), className="n"))
            elif c == "换手":
                cells.append(html.Td(_f(v, "{:.2f}"), className="n"))
            elif c == "贡献分位":
                cells.append(html.Td(_f(v, "{:.0f}%"), className="n"))
            elif c == "近期衰减":
                cells.append(html.Td(str(v or ""), className="mono"))
            else:
                cells.append(
                    html.Td("" if pd.isna(v) else str(v), className="n" if k == "n" else "")
                )
        rows.append(html.Tr(cells))
    capped = len(df) > ROW_CAP
    note = f"显示 {min(len(df), ROW_CAP)} / {len(df)} 个" + (
        f"（仅前 {ROW_CAP} 个，用上方筛选缩小范围）" if capped else ""
    )
    return html.Div(
        [
            html.Table([html.Thead(head), html.Tbody(rows)], className="t"),
            html.Div(note, className="hint"),
        ]
    )


_COLS = [
    ("因子", "因子", ""),
    ("类别", "类别", ""),
    ("来源", "来源", ""),
    ("准入", "准入", ""),
    ("RankIC", "RankIC", "n"),
    ("ICIR", "ICIR", "n"),
    ("多空收益", "多空收益", "n"),
    ("换手", "换手", "n"),
    ("贡献分位", "贡献分位", "n"),
    ("近期衰减", "近期衰减", ""),
    ("冗余簇", "冗余簇", ""),
]


def _kpis(df: pd.DataFrame):
    n_eval = int(df["RankIC"].notna().sum()) if not df.empty else 0
    n_pass = int((df["准入"] == "✓").sum()) if not df.empty else 0
    n_red = int((df["冗余簇"] != "").sum()) if not df.empty else 0
    lib = data.factor_library()
    return html.Div(
        [
            ui.kpi(
                str(len(df)),
                "因子总数",
                hint=f"公式库 {len(lib)} · 原生 {int((df['来源'] == '原生').sum()) if not df.empty else 0}",
            ),
            ui.kpi(str(n_eval), "已评估"),
            ui.kpi(str(n_pass), "通过准入", "ok" if n_pass else ""),
            ui.kpi(str(n_red), "处于冗余簇", hint="|corr|≥0.8"),
        ],
        className="grid g4",
    )


def _heatmap_card():
    corr = data.factor_corr()
    if corr.empty:
        return ui.card(
            "因子相关矩阵",
            ui.empty("没有 diag_factor_corr.csv。点「运行全库评估」并勾选诊断后生成。"),
        )
    m = corr.copy()
    first = m.columns[0]
    if str(first).startswith("Unnamed") or first == "":
        m = m.set_index(first)
    labels = [str(c) for c in m.columns]
    f = ui.fig("相关矩阵（|corr|≥0.8 在主表标注同簇）", height=520)
    f.add_heatmap(
        z=m.to_numpy(dtype=float), x=labels, y=labels, colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1
    )
    f.update_xaxes(tickfont=dict(size=7), tickangle=90)
    f.update_yaxes(tickfont=dict(size=7), autorange="reversed")
    f.update_layout(hovermode="closest", margin=dict(l=100, b=100, t=40, r=10))
    return ui.card("因子相关矩阵", ui.graph(f))


def _model_vs_card():
    mvf = data.model_vs_factor()
    if mvf.empty:
        return ui.card(
            "模型 vs 最强单因子",
            ui.empty("没有 diag_model_vs_factor.csv（诊断三需要勾选「含模型对比」，更耗时）。"),
        )
    f = ui.fig("每个窗口：模型 IC vs 最强单因子 IC", height=300)
    label = mvf.columns[0]
    for c in [c for c in mvf.columns[1:] if pd.api.types.is_numeric_dtype(mvf[c])]:
        f.add_bar(x=mvf[label].astype(str), y=mvf[c], name=str(c))
    f.update_layout(barmode="group")
    return ui.card(
        "模型 vs 最强单因子",
        ui.graph(f),
        html.Div("模型 IC 长期低于最强单因子，说明模型没有增量。", className="hint"),
    )


def _eval_status():
    active = [m for m in rr.active_runs(root=_runs_root()) if m.get("kind") == "factoreval"]
    if active:
        a = active[0]
        p = a.get("progress") or {}
        frac = p.get("pct") if p.get("total") else None
        return html.Div(
            [
                html.B("正在评估"),
                html.Span(f"  {a['run_id']}", className="faint"),
                ui.bar(frac),
                html.Div(f"{p.get('phase', '启动中')} · {p.get('message', '')}", className="hint"),
                dcc.Link("查看实时进度 →", href=f"/runs/{a['run_id']}"),
            ]
        )
    last = rr.list_runs(kind="factoreval", limit=1, root=_runs_root())
    if not last:
        return html.Div("还没有评估运行记录。", className="hint")
    m = last[0]
    st = {"done": "完成", "failed": "失败", "killed": "已终止"}.get(m["status"], m["status"])
    if m["status"] == "running":
        st = "已中断"
    return html.Div(
        [
            "最近一次评估：",
            ui.badge(st, {"完成": "done", "失败": "failed"}.get(st, "dead")),
            html.Span(f"  {ui.ago(m.get('started'))}  ", className="muted"),
            dcc.Link("详情", href=f"/runs/{m['run_id']}"),
        ]
    )


def overview_kpis():
    return _kpis(overview_frame())


def overview_charts():
    return html.Div([_model_vs_card(), _heatmap_card()], className="grid")


def overview_layout():
    df = overview_frame()
    cats = ["全部"] + sorted(df["类别"].unique().tolist()) if not df.empty else ["全部"]
    return ui.page(
        "FACTORS",
        "因子",
        "哪个因子有效、哪个冗余、新写的因子能不能打：独立预测力 · 边际贡献 · 模型内贡献。",
        dcc.Interval(id="fw-tick", interval=5000),
        dcc.Store(id="fw-sig", data=signature()),
        ui.card(
            "评估",
            html.Div(
                [
                    html.Button("运行全库评估", id="fw-eval-all", className="btn primary"),
                    html.Button(
                        "全库评估 + 诊断（衰减/相关/模型对比，更耗时）",
                        id="fw-eval-diag",
                        className="btn",
                    ),
                    dcc.Link("＋ 新建公式因子", href="/factors/new", className="btn"),
                ],
                className="row",
            ),
            html.Div(id="fw-launch-msg", className="msg"),
            html.Div(_eval_status(), id="fw-status", style={"marginTop": "6px"}),
        ),
        html.Div(overview_kpis(), id="fw-kpis"),
        ui.card(
            "因子主表（点因子名进入详情）",
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("类别", className="lb"),
                            dcc.Dropdown(id="fw-cat", options=cats, value="全部", clearable=False),
                        ],
                        style={"minWidth": "160px"},
                    ),
                    html.Div(
                        [
                            html.Label("准入", className="lb"),
                            dcc.Dropdown(
                                id="fw-gate", options=_GATE_OPTS, value="全部", clearable=False
                            ),
                        ],
                        style={"minWidth": "130px"},
                    ),
                    html.Div(
                        [
                            html.Label("来源/状态", className="lb"),
                            dcc.Dropdown(
                                id="fw-src", options=_SRC_OPTS, value="全部", clearable=False
                            ),
                        ],
                        style={"minWidth": "130px"},
                    ),
                    html.Div(
                        [
                            html.Label("搜索", className="lb"),
                            dcc.Input(
                                id="fw-text",
                                type="text",
                                placeholder="因子 id / 名称",
                                className="in",
                            ),
                        ],
                        style={"minWidth": "200px"},
                    ),
                ],
                className="row",
            ),
            html.Div(_table(df), id="fw-table", style={"marginTop": "8px"}),
        ),
        html.Div(overview_charts(), id="fw-charts"),
    )


# ----------------------------------------------------------------------------- 详情
def _sign_text(sign) -> str:
    return {1: "预期方向 +", -1: "预期方向 −"}.get(sign, "预期方向 未知")


def _decay_fig(fid: str):
    decay = data.factor_ic_decay()
    if decay.empty or "factor" not in decay.columns:
        return ui.empty_fig("没有 IC 衰减数据（diag_factor_ic_decay.csv）", 280)
    import re

    wcols = sorted(
        [c for c in decay.columns if re.fullmatch(r"w\d+", str(c))], key=lambda c: int(str(c)[1:])
    )
    row = decay[decay["factor"] == fid]
    if row.empty or not wcols:
        return ui.empty_fig(f"{fid} 不在 IC 衰减诊断中", 280)
    vals = row[wcols].to_numpy().ravel().astype(float)
    f = ui.fig("逐窗口 RankIC：它还活着吗？", height=280)
    f.add_bar(
        x=list(range(len(vals))),
        y=vals,
        marker_color=[ui.C["ok"] if v >= 0 else ui.C["bad"] for v in vals],
        name="窗口 RankIC",
    )
    if len(vals) >= 2:
        f.add_scatter(
            x=list(range(len(vals))),
            y=np.cumsum(vals) / np.arange(1, len(vals) + 1),
            name="累计均值",
            line=dict(color=ui.C["accent"]),
        )
    f.add_hline(y=0, line=dict(color="#94a3b8", width=0.8))
    f.update_xaxes(title="窗口 #")
    return f


def _layer_fig(fid: str):
    eq = data.factor_layers(fid)
    if eq.empty:
        return ui.empty_fig("没有分层净值（评估通过准入的因子才落盘）", 280)
    f = ui.fig("分层净值：单调吗？头部还是尾部赚钱？", height=280)
    layers = sorted(eq["layer"].unique())
    for i, lyr in enumerate(layers):
        sub = eq[eq["layer"] == lyr]
        end = i in (0, len(layers) - 1)
        f.add_scatter(
            x=sub["date"],
            y=sub["equity"],
            name=str(lyr),
            mode="lines",
            line=dict(
                color=ui.C["ok"] if i == len(layers) - 1 else ui.C["bad"] if i == 0 else "#cbd5e1",
                width=2 if end else 1,
            ),
        )
    return f


def _ic_dist_fig(fid: str):
    ic = data.factor_daily_ic(fid)
    if ic.empty:
        return ui.empty_fig("没有逐日 IC（评估通过准入的因子才落盘）", 280)
    v = ic["rank_ic"].astype(float)
    f = ui.fig(f"逐日 RankIC 分布：正 IC 占比 {float((v > 0).mean()):.0%}", height=280)
    f.add_histogram(x=v, nbinsx=40, marker_color=ui.C["violet"], opacity=0.75)
    f.add_vline(x=0, line_color=ui.C["bad"])
    f.add_vline(
        x=float(v.mean()),
        line_dash="dash",
        line_color=ui.C["ok"],
        annotation_text=f"均值 {v.mean():+.3f}",
    )
    return f


def _corr_fig(fid: str):
    corr = data.factor_corr()
    if corr.empty:
        return ui.empty_fig("没有相关矩阵（diag_factor_corr.csv）", 280)
    m = corr.copy()
    if str(m.columns[0]).startswith("Unnamed") or m.columns[0] == "":
        m = m.set_index(m.columns[0])
    if fid not in m.index and fid not in m.columns:
        return ui.empty_fig(f"{fid} 不在相关矩阵中", 280)
    s = m.loc[fid] if fid in m.index else m[fid]
    s = pd.to_numeric(s, errors="coerce").drop(labels=[fid], errors="ignore").dropna()
    if s.empty:
        return ui.empty_fig("没有相关数据", 280)
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    f = ui.fig("相关邻域 Top10：独立信息还是重复？", height=280)
    f.add_bar(
        y=[str(i) for i in top.index],
        x=top.values,
        orientation="h",
        marker_color=[ui.C["ok"] if v >= 0 else ui.C["bad"] for v in top.values],
        text=[f"{v:+.2f}" for v in top.values],
        textposition="outside",
    )
    f.update_traces(cliponaxis=False)
    f.update_xaxes(range=[-1.15, 1.3])
    f.update_layout(margin=dict(l=170, r=30, t=40, b=30))
    return f


def _contrib_card(fid: str):
    contrib, scores = data.factor_contrib()
    mine = contrib[contrib["factor"] == fid] if not contrib.empty else pd.DataFrame()
    if mine.empty:
        return ui.card(
            "模型内贡献",
            ui.empty("champion 里没有这个因子（champion 不存在、未入库，或不在模型特征集）。"),
        )
    from model_display import model_label

    q = float((scores.rank(pct=True) * 100).get(fid, np.nan)) if len(scores) else np.nan
    tbl = mine.copy()
    tbl["source"] = tbl["source"].map(model_label)
    tbl["value"] = tbl["value"].astype(float).map(lambda v: f"{v:+.5f}")
    return ui.card(
        "模型内贡献：模型真的在用它吗？",
        html.Div(f"综合贡献分位 {_f(q, '{:.0f}%')}", className="hint"),
        ui.table(tbl, [("source", "来源", ""), ("kind", "口径", ""), ("value", "取值", "n")]),
    )


def _coverage_fig(fid: str):
    path = os.path.join(data.CACHE, "factor_panel_cache.parquet")
    if not os.path.exists(path):
        return ui.empty_fig("没有因子面板缓存", 280)
    try:
        cov = _coverage(path, data.mtime(path), fid)
    except Exception:
        cov = pd.DataFrame()
    if cov.empty:
        return ui.empty_fig("面板里没有该因子列", 280)
    f = ui.fig("覆盖面：覆盖全市场还是只有大盘？", height=280)
    f.add_scatter(x=cov["date"], y=cov["coverage"], line=dict(color=ui.C["accent"]))
    f.update_yaxes(tickformat=".0%", range=[0, 1.02])
    return f


import functools  # noqa: E402


@functools.lru_cache(maxsize=32)
def _coverage(path: str, _mt, fid: str) -> pd.DataFrame:
    """按日非空占比；只读 (date, 因子) 两列。"""
    df = pd.read_parquet(path, columns=["date", fid])
    return (
        df.groupby("date")[fid]
        .apply(lambda s: float(s.notna().mean()))
        .rename("coverage")
        .reset_index()
    )


def detail_layout(fid: str):
    lib = data.factor_library()
    hit = lib[lib["id"] == fid] if not lib.empty else lib
    meta = hit.iloc[0].to_dict() if not hit.empty else None
    if meta is None and fid not in set(data.panel_factor_columns()):
        return ui.page(
            "FACTORS",
            "因子不存在",
            "",
            dcc.Link("← 返回因子库", href="/factors"),
            ui.card(
                "", ui.empty(f"因子 {fid!r} 不在因子库（factor_library/*.yaml）也不在因子面板里。")
            ),
        )
    ev = data.factor_eval_table()
    row = (
        ev[ev["factor"] == fid].iloc[0].to_dict()
        if not ev.empty and (ev["factor"] == fid).any()
        else None
    )
    if meta:
        facts = [
            ("类别", fl._norm_category(meta["category"])),
            ("来源", f"公式库 · {meta['file']}"),
            (_sign_text(meta["expected_sign"]), ""),
            ("状态", "启用" if meta["enabled"] else "草稿/停用"),
            ("版本", meta["version"]),
        ]
        formula = html.Pre(meta["formula"], className="log")
        desc = meta["description"]
    else:
        facts = [("类别", fl.infer_category(fid)), ("来源", "面板原生因子")]
        formula, desc = html.Div("原生因子，代码在 src/data/factors/。", className="hint"), ""
    kp = (
        html.Div(
            [
                ui.kpi(_f(row.get("RankIC")), "RankIC"),
                ui.kpi(_f(row.get("RankICIR")), "RankICIR"),
                ui.kpi(_f(row.get("IC_t"), "{:+.1f}"), "IC t 值"),
                ui.kpi(_f(row.get("多空收益")), "多空收益"),
                ui.kpi(_f(row.get("换手"), "{:.2f}"), "换手"),
                ui.kpi(
                    "通过" if row.get("准入") == "✓" else "未通过",
                    "准入",
                    "ok" if row.get("准入") == "✓" else "bad",
                ),
            ],
            className="grid g6",
        )
        if row
        else ui.card("评估", ui.empty("还没有评估结果。点下方「评估该因子」。"))
    )
    return ui.page(
        "FACTOR",
        fid,
        (meta or {}).get("name") or "面板原生因子",
        dcc.Link("← 返回因子库", href="/factors"),
        html.Div(
            " · ".join(f"{a}{('：' + str(b)) if b != '' else ''}" for a, b in facts),
            className="hint",
            style={"margin": "6px 0"},
        ),
        kp,
        html.Div(
            [
                ui.card("公式", formula, html.Div(desc, className="hint")),
                ui.card(
                    "评估",
                    html.Button("评估该因子", id="fd-eval", className="btn primary"),
                    html.Div(id="fd-msg", className="msg"),
                    dcc.Store(id="fd-id", data=fid),
                    html.Div(_eval_status(), style={"marginTop": "6px"}),
                ),
            ],
            className="grid g2",
        ),
        html.Div(
            [
                ui.card("逐窗口 RankIC", ui.graph(_decay_fig(fid))),
                ui.card("分层净值", ui.graph(_layer_fig(fid))),
            ],
            className="grid g2",
        ),
        html.Div(
            [
                ui.card("IC 分布", ui.graph(_ic_dist_fig(fid))),
                ui.card("相关邻域", ui.graph(_corr_fig(fid))),
            ],
            className="grid g2",
        ),
        html.Div(
            [_contrib_card(fid), ui.card("覆盖面", ui.graph(_coverage_fig(fid)))],
            className="grid g2",
        ),
    )


# ----------------------------------------------------------------------------- 编辑器
def _validation_view(res: dict):
    out = []
    if res["errors"]:
        out.append(
            html.Div(
                [html.B("校验未通过"), html.Ul([html.Li(e) for e in res["errors"]])],
                className="bad",
            )
        )
    else:
        out.append(html.Div("校验通过。", className="ok"))
    if res["warnings"]:
        out.append(html.Ul([html.Li(w) for w in res["warnings"]], className="warn"))
    if res["deps"]:
        out.append(html.Div("依赖列：" + "、".join(res["deps"]), className="hint"))
    return html.Div(out)


def editor_layout():
    def field(label, comp):
        return html.Div([html.Label(label, className="lb"), comp], style={"minWidth": "220px"})

    return ui.page(
        "FACTORS",
        "新建公式因子",
        "用 DSL 写一个因子，校验后存为草稿；草稿不会自动进入模型，评估通过后再决定是否启用。",
        dcc.Link("← 返回因子库", href="/factors"),
        ui.card(
            "定义",
            html.Div(
                [
                    field(
                        "id（列名）",
                        dcc.Input(
                            id="fe-id", type="text", className="in", placeholder="my_factor_1"
                        ),
                    ),
                    field("名称", dcc.Input(id="fe-name", type="text", className="in")),
                    field(
                        "类别", dcc.Input(id="fe-cat", type="text", className="in", value="custom")
                    ),
                    field(
                        "预期方向",
                        dcc.Dropdown(
                            id="fe-sign",
                            options=[
                                {"label": "未知", "value": ""},
                                {"label": "+1（越大越好）", "value": "+1"},
                                {"label": "−1（越小越好）", "value": "-1"},
                            ],
                            value="",
                            clearable=False,
                        ),
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Label("公式", className="lb"),
                    dcc.Textarea(
                        id="fe-formula",
                        className="in",
                        style={"height": "110px", "fontFamily": "var(--mono)"},
                        placeholder="close / ts_delay(close, 5) - 1",
                    ),
                ],
                style={"marginTop": "8px"},
            ),
            html.Div(
                [
                    field("标签（逗号分隔）", dcc.Input(id="fe-tags", type="text", className="in")),
                    field("说明", dcc.Input(id="fe-desc", type="text", className="in")),
                ],
                className="row",
                style={"marginTop": "8px"},
            ),
            html.Div(
                [
                    html.Button("校验", id="fe-validate", className="btn"),
                    html.Button("保存草稿", id="fe-save", className="btn primary"),
                ],
                className="row",
                style={"marginTop": "10px"},
            ),
            html.Div(id="fe-result", className="msg"),
        ),
        ui.card(
            "可用函数与列",
            html.Div("DSL 原语：" + "、".join(sorted(_dsl_names())), className="hint"),
            html.Div(
                "可引用的行情/面板列共 %d 个（open/high/low/close/amount/pct_chg/adj_close/return_1d 等；注意行情缓存里有 volume，面板里没有 turnover_rate）。"
                % len(data.base_columns()),
                className="hint",
            ),
        ),
    )


def _dsl_names() -> set[str]:
    try:
        from data.factors.parser import _FUNCTION_NAMES

        return set(_FUNCTION_NAMES)
    except Exception:
        return set()


# ----------------------------------------------------------------------------- 路由
def layout(pathname: str):
    p = (pathname or "/factors").rstrip("/")
    if p == "/factors/new":
        return editor_layout()
    if p.startswith("/factors/") and len(p) > len("/factors/"):
        return detail_layout(p[len("/factors/") :])
    return overview_layout()


def filtered_table(cat, gate, src, text):
    return _table(fl.apply_filters(overview_frame(), cat, gate, src, text))


def register(app):
    @app.callback(
        Output("fw-table", "children"),
        Input("fw-cat", "value"),
        Input("fw-gate", "value"),
        Input("fw-src", "value"),
        Input("fw-text", "value"),
        prevent_initial_call=True,
    )
    def _filter(cat, gate, src, text):
        return filtered_table(cat, gate, src, text)

    @app.callback(
        Output("fw-kpis", "children"),
        Output("fw-charts", "children"),
        Output("fw-table", "children", allow_duplicate=True),
        Output("fw-sig", "data"),
        Output("fw-status", "children"),
        Input("fw-tick", "n_intervals"),
        State("fw-sig", "data"),
        State("fw-cat", "value"),
        State("fw-gate", "value"),
        State("fw-src", "value"),
        State("fw-text", "value"),
        prevent_initial_call=True,
    )
    def _tick(_n, stored, cat, gate, src, text):
        sig = signature()
        if sig == stored:
            return no_update, no_update, no_update, no_update, _eval_status()
        return (
            overview_kpis(),
            overview_charts(),
            filtered_table(cat, gate, src, text),
            sig,
            _eval_status(),
        )

    @app.callback(
        Output("fw-launch-msg", "children"),
        Input("fw-eval-all", "n_clicks"),
        Input("fw-eval-diag", "n_clicks"),
        prevent_initial_call=True,
    )
    def _eval_all(n1, n2):
        trig = ctx.triggered_id
        if not ((trig == "fw-eval-all" and n1) or (trig == "fw-eval-diag" and n2)):
            return no_update
        ok, msg, rid = actions.launch_full_factor_eval(with_diagnostics=trig == "fw-eval-diag")
        return html.Span(
            [msg + " ", dcc.Link("查看进度", href=f"/runs/{rid}") if rid else ""],
            className="ok" if ok else "bad",
        )

    @app.callback(
        Output("fd-msg", "children"),
        Input("fd-eval", "n_clicks"),
        State("fd-id", "data"),
        prevent_initial_call=True,
    )
    def _eval_one(n, fid):
        if not n:
            return no_update
        ok, msg, rid = actions.launch_factor_eval(fid)
        return html.Span(
            [msg + " ", dcc.Link("查看进度", href=f"/runs/{rid}") if rid else ""],
            className="ok" if ok else "bad",
        )

    def _validate(fid, name, cat, formula, tags, sign, desc):
        from factor_draft import validate_draft

        lib = data.factor_library()
        return validate_draft(
            fid,
            name,
            cat,
            formula,
            tags,
            sign,
            desc,
            existing_ids=set(lib["id"]) | set(data.panel_factor_columns()),
            known_columns=data.base_columns() | set(lib["id"]),
        )

    @app.callback(
        Output("fe-result", "children"),
        Input("fe-validate", "n_clicks"),
        Input("fe-save", "n_clicks"),
        State("fe-id", "value"),
        State("fe-name", "value"),
        State("fe-cat", "value"),
        State("fe-formula", "value"),
        State("fe-tags", "value"),
        State("fe-sign", "value"),
        State("fe-desc", "value"),
        prevent_initial_call=True,
    )
    def _editor(n_val, n_save, fid, name, cat, formula, tags, sign, desc):
        trig = ctx.triggered_id
        if trig == "fe-save" and n_save:
            ok, msg, res = actions.save_factor_draft(
                fid, name, cat, formula, tags or "", sign, desc or ""
            )
            if not ok:
                return html.Div([html.Div(msg, className="bad"), _validation_view(res)])
            return html.Div(
                [
                    html.Div(msg, className="ok"),
                    html.Div(
                        [
                            html.Button(
                                "立即评估该因子", id="fe-eval-now", className="btn primary"
                            ),
                            dcc.Link(
                                "在因子库中查看",
                                href=f"/factors/{res['record']['id']}",
                                className="btn",
                            ),
                        ],
                        className="row",
                        style={"marginTop": "8px"},
                    ),
                    dcc.Store(id="fe-saved-id", data=res["record"]["id"]),
                ]
            )
        if trig == "fe-validate" and n_val:
            return _validation_view(_validate(fid, name, cat, formula, tags, sign, desc))
        return no_update

    @app.callback(
        Output("fe-result", "children", allow_duplicate=True),
        Input("fe-eval-now", "n_clicks"),
        State("fe-saved-id", "data"),
        prevent_initial_call=True,
    )
    def _eval_now(n, fid):
        if not n:
            return no_update
        ok, msg, rid = actions.launch_factor_eval(fid)
        return html.Div(
            [
                html.Span(msg + " ", className="ok" if ok else "bad"),
                dcc.Link("查看进度", href=f"/runs/{rid}") if rid else "",
            ]
        )
