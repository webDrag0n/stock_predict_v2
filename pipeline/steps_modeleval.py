"""信号评估（modeleval）模式的流水线步骤：PredIC/ICIR/分位收益，不做撮合回测。

引擎是 src/signal_eval.run_signal_eval：
  model_ref="rolling"      滚动重训（复用 walkforward 切分 + train_predict，跳过撮合）
  model_ref="champion"     冻结 champion 工件逐日截面打分
  model_ref="staging:<id>" 冻结 staging 候选逐日截面打分

顺序：data_import → signal_eval → eval_report

报告视图：
  rolling：逐窗口 PredIC（柱状）+ 逐日 RankIC（折线）
  冻结：  逐日 RankIC（折线）
  共用：  分位数分层收益（柱状）+ 分位/窗口表

产物落盘会话 artifacts（daily_ic.parquet / windows.parquet / eval_summary.json），
eval_report 可仅凭 artifact 重建（跨进程回读，不依赖同进程 shared）。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C
from data import load_factor_panel, list_factor_columns
from signal_eval import run_signal_eval

from . import context as ctxmod
from .context import _jsonable, kpi, table, fig_line, fig_bar, report


def _f(x: float, fmt: str, na: str = "—") -> str:
    """容错格式化：None/NaN/inf → na。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    return fmt.format(v) if np.isfinite(v) else na


# ---------------------------------------------------------------------------
def data_import(ctx: ctxmod.StepContext) -> dict:
    """加载命名因子面板并按 lookback 截断（信号评估无需撮合列，不取交易列）。

    rolling 路径还需训练历史：保留尾部 train_window + embargo + lookback 天；
    冻结模型只需尾部 lookback 天（逐日打分自包含）。
    """
    p = ctx.params
    lookback = int(p.get("lookback_days") or 120)
    model_ref = str(p.get("model_ref") or "rolling")

    ctx.report_progress("数据加载中", 0, 2, "load_factor_panel")
    panel = load_factor_panel(verbose=True)
    factors = list_factor_columns(panel)

    if model_ref == "rolling":
        tw = int(p.get("train_window") or C.TRAIN_WINDOW)
        need = lookback + tw + C.EMBARGO_DAYS
    else:
        need = lookback
    dates = np.sort(panel["date"].unique())
    truncated = len(dates) > need
    if truncated:
        panel = panel[panel["date"] >= dates[-need]].reset_index(drop=True)
        dates = dates[-need:]
    ctx.shared["panel"] = panel
    ctx.shared["factors"] = factors
    ctx.report_progress("数据画像中", 1, 2, f"{panel['symbol'].nunique():,} 只")

    per_date = panel.groupby("date")["symbol"].nunique()
    figures = [
        fig_line(
            "每日横截面股票数",
            per_date.index.astype(str).tolist(),
            [{"name": "股票数", "y": per_date.to_numpy()}],
            yaxis="只",
            x_is_date=True,
        )
    ]
    notes = []
    if truncated:
        notes.append(
            f"面板已截断到尾部 {len(dates)} 个交易日"
            f"（lookback={lookback}"
            + (
                f" + 训练历史 train={tw} + embargo={C.EMBARGO_DAYS}"
                if model_ref == "rolling"
                else ""
            )
            + "）"
        )
    if per_date.min() < 30:
        notes.append(
            f"⚠️ 部分交易日横截面股票数过少（最小 {int(per_date.min())} 只），排序指标可能不稳定"
        )
    if model_ref == "rolling":
        n_win = max(
            (len(dates) - tw - C.EMBARGO_DAYS - int(p.get("test_window") or C.TEST_WINDOW))
            // int(p.get("wf_step") or C.WF_STEP)
            + 1,
            0,
        )
        if n_win <= 0:
            notes.append(
                f"⚠️ 按当前 train={tw}/embargo={C.EMBARGO_DAYS}/"
                f"test={int(p.get('test_window') or C.TEST_WINDOW)} 配置不足以构成任何滚动窗口"
            )

    kpis = [
        kpi("模型引用", model_ref, "rolling=滚动重训"),
        kpi("股票数", f"{panel['symbol'].nunique():,}", "截断后去重"),
        kpi(
            "交易日",
            f"{len(dates):,}",
            f"{pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()}",
        ),
        kpi("总行数", f"{len(panel):,}", "面板 (date×symbol)"),
        kpi("因子数", f"{len(factors)}", "命名因子"),
        kpi("评估区间", f"{lookback} 天", "尾部 lookback_days 个交易日"),
    ]
    summary = (
        f"{model_ref} · {panel['symbol'].nunique():,} 只 · {len(dates):,} 交易日 · "
        f"{len(factors)} 因子"
    )
    return report(
        "data_import", "数据导入", summary=summary, kpis=kpis, figures=figures, notes=notes
    )


# ---------------------------------------------------------------------------
def signal_eval(ctx: ctxmod.StepContext) -> dict:
    """跑信号评估（rolling 按窗口 / 冻结按日上报进度），落盘逐日 IC 与汇总。"""
    panel = ctx.shared["panel"]
    factors = ctx.shared["factors"]
    p = ctx.params
    model_ref = str(p.get("model_ref") or "rolling")
    lookback = int(p.get("lookback_days") or 120)
    n_quantiles = int(p.get("n_quantiles") or 5)

    def _cb(phase, current, total, message="", **extras):
        ctx.report_progress(phase, current, total, message, **extras)

    res = run_signal_eval(
        panel,
        factors,
        model_ref=model_ref,
        lookback_days=lookback,
        train_window=int(p.get("train_window") or C.TRAIN_WINDOW),
        test_window=int(p.get("test_window") or C.TEST_WINDOW),
        wf_step=int(p.get("wf_step") or C.WF_STEP),
        n_quantiles=n_quantiles,
        progress_cb=_cb,
    )
    ctx.shared["signal_eval"] = res

    # 落盘 artifact：eval_report 可仅凭 artifact 重建
    res["daily_ic"].to_parquet(ctx.artifact_path("daily_ic.parquet"), index=False)
    if res["windows"] is not None:
        res["windows"].to_parquet(ctx.artifact_path("windows.parquet"), index=False)
    summary_json = {k: v for k, v in res.items() if k not in ("daily_ic", "windows")}
    with open(ctx.artifact_path("eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary_json), f, ensure_ascii=False, indent=2)

    kpis = [
        kpi("PredIC", _f(res["pred_ic_mean"], "{:+.4f}"), "逐日截面 RankIC 均值"),
        kpi("ICIR", _f(res["pred_icir"], "{:+.3f}"), "PredIC 均值/标准差"),
        kpi("IC>0 占比", _f(res["ic_positive_ratio"], "{:.0%}"), f"{res['n_dates']} 个评估日"),
        kpi(
            "多空收益",
            _f(res["long_short_return"], "{:+.4f}"),
            f"Q{n_quantiles}−Q1 平均下期收益",
        ),
    ]
    if res["n_windows"] is not None:
        wm = res["windows"]
        n_pos = int((wm["PredIC"] > 0).sum())
        kpis.append(kpi("滚动窗口", f"{res['n_windows']}", f"PredIC>0 的窗口 {n_pos}/{len(wm)}"))
    return report(
        "signal_eval",
        "信号评估",
        summary=(
            f"{model_ref} · {res['n_dates']} 天 · "
            f"PredIC {_f(res['pred_ic_mean'], '{:+.4f}')} · "
            f"ICIR {_f(res['pred_icir'], '{:+.3f}')}"
        ),
        kpis=kpis,
    )


# ---------------------------------------------------------------------------
def _load_eval(ctx: ctxmod.StepContext) -> dict:
    """取信号评估结果：优先 ctx.shared（同进程），否则从 artifact 重建。"""
    res = ctx.shared.get("signal_eval")
    if res is not None:
        return res
    with open(ctx.artifact_path("eval_summary.json"), "r", encoding="utf-8") as f:
        res = json.load(f)
    res["daily_ic"] = pd.read_parquet(ctx.artifact_path("daily_ic.parquet"))
    try:
        res["windows"] = pd.read_parquet(ctx.artifact_path("windows.parquet"))
    except Exception:
        res["windows"] = None
    return res


def build_eval_report(res: dict) -> dict:
    """信号评估报告（纯函数）：PredIC/ICIR KPI + 逐窗口/逐日 IC 图 + 分位收益图/表。

    eval_report(ctx)（会话步骤）从 _load_eval 取数后委托此构造函数。
    """
    n_q = int(res.get("n_quantiles") or 5)
    layer_ret = res.get("layer_returns") or []
    daily_ic = res.get("daily_ic")
    windows = res.get("windows")
    model_ref = res.get("model_ref") or "rolling"
    model_info = res.get("model_info") or {}

    figures = []
    # 逐窗口 PredIC（rolling）：稳健性视角
    if windows is not None and len(windows):
        figures.append(
            fig_bar(
                "逐窗口 PredIC（稳健性）",
                [f"W{int(w)}" for w in windows["window"]],
                windows["PredIC"].to_numpy(),
                yaxis="PredIC",
                xaxis="窗口",
                diverging=True,
            )
        )
    # 逐日 RankIC
    if daily_ic is not None and len(daily_ic):
        x = [str(pd.Timestamp(d).date()) for d in daily_ic["date"]]
        series = [{"name": "日 RankIC", "y": daily_ic["rank_ic"].to_numpy(), "opacity": 0.45}]
        roll = daily_ic["rank_ic"].rolling(20, min_periods=5).mean().to_numpy()
        series.append({"name": "20 日均值", "y": roll, "width": 2.5, "color": "#1a7f37"})
        figures.append(
            fig_line(
                "逐日截面 RankIC（prediction vs 前瞻收益）",
                x,
                series,
                yaxis="RankIC",
                x_is_date=True,
            )
        )
    # 分位数分层收益
    if layer_ret:
        figures.append(
            fig_bar(
                f"分位数分层平均下期收益（Q1 最低预测 ~ Q{n_q} 最高）",
                [f"Q{i + 1}" for i in range(len(layer_ret))],
                layer_ret,
                yaxis="平均下期收益",
                xaxis="分层",
                diverging=True,
            )
        )

    kpis = [
        kpi("PredIC", _f(res.get("pred_ic_mean"), "{:+.4f}"), "选股 alpha 成立性的核心指标"),
        kpi("ICIR", _f(res.get("pred_icir"), "{:+.3f}"), "信号稳定性（均值/标准差）"),
        kpi(
            "IC>0 占比",
            _f(res.get("ic_positive_ratio"), "{:.0%}"),
            f"{res.get('eval_start')} ~ {res.get('eval_end')}",
        ),
        kpi("多空收益", _f(res.get("long_short_return"), "{:+.4f}"), f"Q{n_q}−Q1"),
        kpi("分层单调性", _f(res.get("monotonicity"), "{:.2f}"), "层序与层收益的秩相关"),
        kpi("评估天数", f"{res.get('n_dates', '—')}", f"共 {res.get('n_rows', '—')} 行预测"),
    ]

    notes = [
        "PredIC = 每个交易日 prediction 与前瞻收益（responder_6）的截面 Spearman "
        "秩相关，再对时间取均值——与 walk-forward 报告中的 PredIC 同口径；"
        "此处不做撮合回测，分层/多空收益是【信号口径】（不含成本与成交约束）。",
    ]
    pic = res.get("pred_ic_mean")
    if pic is not None and np.isfinite(float(pic)) and float(pic) < 0:
        notes.append("⚠️ PredIC 为负——该模型在当前评估区间无选股 alpha")
    mono = res.get("monotonicity")
    if mono is not None and np.isfinite(float(mono)) and float(mono) < 0.5:
        notes.append("⚠️ 分层单调性不足 0.5——分位收益不随预测排序单调递增")
    if model_ref != "rolling":
        ted = model_info.get("train_end_date")
        notes.append(
            f"冻结模型 {model_info.get('model_id') or model_ref}"
            f"（train_end_date={ted}）按工件保存的权重原样打分，不刷新权重。"
        )
        if ted and res.get("eval_start") and str(res["eval_start"]) <= str(ted):
            notes.append(
                f"⚠️ 评估起点 {res['eval_start']} 早于模型训练截止 {ted}："
                "该段为样本内打分，PredIC 会虚高，请把 lookback 限制在训练截止之后解读。"
            )

    tables = []
    if layer_ret:
        ldf = pd.DataFrame(
            {
                "分层": [f"Q{i + 1}" for i in range(len(layer_ret))],
                "平均下期收益": [None if v is None else round(float(v), 6) for v in layer_ret],
            }
        )
        tables.append(table("分位数分层收益", ldf, max_rows=20))
    if windows is not None and len(windows):
        disp = windows.copy()
        disp["PredIC"] = disp["PredIC"].astype(float).round(4)
        tables.append(table("逐窗口 PredIC", disp, max_rows=100))

    return report(
        "eval_report",
        "信号评估报告",
        summary=(
            f"{model_ref} · PredIC {_f(res.get('pred_ic_mean'), '{:+.4f}')} · "
            f"ICIR {_f(res.get('pred_icir'), '{:+.3f}')} · "
            f"多空 {_f(res.get('long_short_return'), '{:+.4f}')}"
        ),
        kpis=kpis,
        figures=figures,
        tables=tables,
        notes=notes,
    )


def eval_report(ctx: ctxmod.StepContext) -> dict:
    """会话步骤：从 _load_eval 取数（shared 优先，否则 artifact 重建）后构造报告。"""
    ctx.report_progress("构造报告", 0, 1)
    rep = build_eval_report(_load_eval(ctx))
    ctx.report_progress("构造报告", 1, 1)
    return rep
