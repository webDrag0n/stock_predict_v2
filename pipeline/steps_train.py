"""train 模式的流水线步骤：训练候选 ensemble + 门控评估 → staging（不自动晋升）。

顺序：data_import → train_candidate → gate_eval

与 nextday 模式的差异：不生成订单、不跑自适应触发判定；候选训练
（live.train_candidate_ensemble，与实盘门控同一训练段）后调
model_gate.stage_and_evaluate(auto_promote=False) —— 候选只落盘
models/staging/<model_id>/（manifest.json + ensemble_state.pkl +
evaluation.json），**绝不自动晋升、绝不改动 champion**；晋升交给 webapp
模型实验室的人工操作（model_store.promote）。

ctx.params 读取的参数：ensemble_models / n_seeds / incr_trees / rebuild_days /
train_window / ic_window（训练配方），eval_days（门控评估窗口，默认 60），
以及 steps_nextday._decision_params 认得的决策参数（门控双回放与正式下单
同口径）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import model_store
from data import load_factor_panel, list_factor_columns
from live import train_candidate_ensemble
from model_display import model_label
from model_gate import stage_and_evaluate

from . import context as ctxmod
from . import steps_nextday as N
from .context import kpi, table, fig_line, fig_bar, report

# 门控对比表的指标顺序与中文标签（键 = perf_metrics.scalar_metrics 输出）
_METRIC_LABELS = (
    ("sharpe", "Sharpe"),
    ("sortino", "Sortino"),
    ("max_drawdown", "最大回撤(%)"),
    ("alpha_ann", "年化alpha"),
    ("t_nw", "NW-t"),
    ("beta", "Beta"),
    ("hit_rate", "日胜率"),
    ("total_return", "区间收益"),
    ("n_days", "交易日数"),
)


def _f(x, fmt: str = "{:+.3f}", na: str = "—") -> str:
    """容错格式化：None/NaN/inf → na。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    return fmt.format(v) if np.isfinite(v) else na


# ---------------------------------------------------------------------------
def data_import(ctx: ctxmod.StepContext) -> dict:
    """加载全市场命名因子面板（keep_trade_cols=True：门控双回放撮合需 OHLC 等交易列）。

    与 steps_walkforward.wf_data_import 同一加载口径（全量历史，训练 + 门控
    双回放都需要长历史），仅报告画像按 train 模式裁剪。
    """
    ctx.report_progress("数据加载中", 0, 2, "load_factor_panel")
    panel = load_factor_panel(keep_trade_cols=True, verbose=True)
    factors = list_factor_columns(panel)
    ctx.shared["panel"] = panel
    ctx.shared["factors"] = factors
    ctx.report_progress("数据画像中", 1, 2, f"{panel['symbol'].nunique():,} 只")

    dates = np.sort(panel["date"].unique())
    per_date = panel.groupby("date")["symbol"].nunique()
    n_label = int(panel[C.TARGET].notna().sum())

    figures = [
        fig_line(
            "每日横截面股票数",
            per_date.index.astype(str).tolist(),
            [{"name": "股票数", "y": per_date.to_numpy()}],
            yaxis="只",
            x_is_date=True,
        )
    ]
    kpis = [
        kpi("股票数", f"{panel['symbol'].nunique():,}", "全样本去重"),
        kpi(
            "交易日",
            f"{len(dates):,}",
            f"{pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()}",
        ),
        kpi("总行数", f"{len(panel):,}", "面板 (date×symbol)"),
        kpi("有标签行", f"{n_label:,}", "训练样本（未成熟标签不计）"),
        kpi("因子数", f"{len(factors)}", "命名因子"),
    ]
    notes = []
    if per_date.min() < 30:
        notes.append(
            f"⚠️ 部分交易日横截面股票数过少（最小 {int(per_date.min())} 只），" "排序策略可能不稳定"
        )
    if "amount" not in panel.columns:
        notes.append("⚠️ 面板缺 amount 列，门控回放的市场冲击成本将被禁用（仅佣金+印花税）")
    return report(
        "data_import",
        "数据导入",
        summary=(
            f"{panel['symbol'].nunique():,} 只 · {len(dates):,} 交易日 · " f"{len(factors)} 因子"
        ),
        kpis=kpis,
        figures=figures,
        notes=notes,
    )


# ---------------------------------------------------------------------------
def _ensemble_params(ctx: ctxmod.StepContext) -> dict:
    """训练配方参数（与 steps_nextday 同键集合，另加 ensemble_models 腿配置）。"""
    p = ctx.params
    out = {}
    for key in ("n_seeds", "incr_trees", "rebuild_days", "train_window", "ic_window"):
        if p.get(key) is not None:
            out[key] = int(p[key])
    if p.get("ensemble_models"):
        out["ensemble_models"] = str(p["ensemble_models"])
    return out


def train_candidate(ctx: ctxmod.StepContext) -> dict:
    """训练候选 ensemble（同配方时增量续训 champion，否则全量重建），结果存 ctx.shared。"""
    panel = ctx.shared["panel"]
    factors = ctx.shared["factors"]
    params = _ensemble_params(ctx)

    def _cb(phase, current, total, **extras):
        ctx.report_progress(phase, current, total, **extras)

    ctx.report_progress("训练启动", 0, 3, f"{len(factors)} 因子")
    ens, recipe = train_candidate_ensemble(panel, factors, params=params, progress_cb=_cb)
    ctx.shared["candidate_ens"] = ens
    ctx.shared["cand_recipe"] = recipe
    ctx.report_progress("训练完成", 3, 3, models=",".join(ens.models))

    fp = model_store.fingerprint_of(factors, recipe)
    train_end = str(ens._last_trained_date)[:10] if ens._last_trained_date is not None else "—"
    wtxt = " · ".join(f"{model_label(m)} {w:.0%}" for m, w in ens.weights.items())
    figures = []
    if ens.weights:
        figures.append(
            fig_bar(
                "多模型初始 IC 权重（无 OOS 历史时为等权）",
                [model_label(m) for m in ens.weights.keys()],
                list(ens.weights.values()),
                yaxis="权重",
            )
        )
    kpis = [
        kpi(
            "子模型",
            "、".join(model_label(m) for m in ens.models),
            f"{len(ens.models)} 个 · 权重 {wtxt}",
        ),
        kpi("训练截止日", train_end, "已成熟标签最后一日"),
        kpi("配方指纹", fp, "与 champion 不同 → 门控评估"),
        kpi(
            "训练窗口",
            f"{recipe['train_window']}",
            f"n_seeds={recipe['n_seeds']} · incr_trees={recipe['incr_trees']}",
        ),
    ]
    champ = model_store.load_champion_manifest()
    notes = []
    if champ is None:
        notes.append("ℹ️ 当前无 champion：gate_eval 将只留档候选，不做双回放对比")
    elif champ.get("fingerprint") == fp:
        notes.append(
            f"ℹ️ 配方指纹与 champion（{champ.get('model_id')}）相同——"
            "属同一配方滚动重训，本模式仍走 staging 留档 + 评估，不会原位刷新 champion"
        )
    return report(
        "train_candidate",
        "训练候选模型",
        summary=f"已训练 {len(ens.models)} 个子模型 · 训练截止 {train_end} · 指纹 {fp}",
        kpis=kpis,
        figures=figures,
        notes=notes,
    )


# ---------------------------------------------------------------------------
def gate_eval(ctx: ctxmod.StepContext) -> dict:
    """候选落盘 staging + 与 champion 双回放对比（auto_promote=False，绝不自动晋升）。"""
    panel = ctx.shared["panel"]
    factors = ctx.shared["factors"]
    ens = ctx.shared["candidate_ens"]
    recipe = ctx.shared["cand_recipe"]
    eval_days = int(ctx.params.get("eval_days") or 60)

    def _cb(phase, current, total, **extras):
        ctx.report_progress(phase, current, total, **extras)

    ctx.report_progress("门控评估启动", 0, 1, f"eval_days={eval_days}")
    res = stage_and_evaluate(
        ens,
        panel,
        factors,
        recipe,
        eval_days=eval_days,
        auto_promote=False,
        decision_params=N._decision_params(ctx),
        progress=False,
        progress_cb=_cb,
    )
    ctx.shared["gate_eval"] = res
    ctx.report_progress("门控评估完成", 1, 1, staging_id=res["staging_id"])

    tables = []
    figures = []
    cand, champ_m = res["candidate"], res["champion"]
    if cand is not None and champ_m is not None:
        rows = []
        for key, label in _METRIC_LABELS:
            if key in cand and key in champ_m:
                rows.append([label, _f(cand[key]), _f(champ_m[key])])
        tables.append(
            table(
                "候选 vs champion 门控对比",
                pd.DataFrame(rows, columns=["指标", "候选", "champion"]),
            )
        )
        figures.append(
            fig_bar(
                "Sharpe 对比",
                ["候选", "champion"],
                [cand["sharpe"], champ_m["sharpe"]],
                yaxis="Sharpe",
                diverging=True,
            )
        )

    verdict = "建议晋升（指标通过）" if res["should_promote"] else "不建议晋升"
    kpis = [
        kpi("staging 候选", res["staging_id"], "已留档 models/staging/，未晋升"),
        kpi(
            "评估窗口",
            f"{res['window'][0]} ~ {res['window'][1]}" if res["window"] else "—",
            f"最近 {res['eval_days']} 个决策日双回放",
        ),
        kpi("门控判定", verdict, "auto_promote=False：指标判定仅供参考"),
        kpi("champion", res["champion_id"] or "无", "本次评估后保持不变"),
    ]
    if cand is not None and champ_m is not None:
        kpis.append(
            kpi(
                "候选 Sharpe",
                _f(cand["sharpe"], "{:+.2f}"),
                f"vs champion {_f(champ_m['sharpe'], '{:+.2f}')} · "
                f"回撤 {_f(cand['max_drawdown'], '{:+.1f}')}% vs "
                f"{_f(champ_m['max_drawdown'], '{:+.1f}')}%",
            )
        )
    notes = [f"门控结论：{res['reason']}"]
    notes.append(
        "ℹ️ 本模式【不自动晋升】：如需上线请到 webapp 模型实验室对该 staging 候选执行手动晋升；"
        "champion 与既有 promote 流程不受本次会话影响。"
    )
    return report(
        "gate_eval",
        "门控评估（→ staging，不晋升）",
        summary=f"候选 {res['staging_id']} 已留档 · {verdict}",
        kpis=kpis,
        tables=tables,
        figures=figures,
        notes=notes,
    )
