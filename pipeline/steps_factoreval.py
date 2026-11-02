"""factoreval 模式 —— 因子库评估（全库 / 单因子子集）的流水线步骤。

与 CLI ``scripts/eval_factors.py`` 共用同一套度量与产物契约：
``load_factor_panel``（命名因子面板）→ 逐因子 ``evaluate_factor``
（RankIC/ICIR/分层/换手/准入）→ ``outputs/factor_eval.csv`` +
``factor_layers.dump_layers`` 落盘 ``factor_layer_<id>.parquet`` /
``factor_ic_daily_<id>.parquet``。webapp/pages/factor_workbench.py 直接读
这些产物，因此**文件名与列结构不可变**。

顺序：data_import → eval_factors → eval_report

ctx.params（由 registry.PARAM_SCHEMA["factoreval"] 提供）::

    lookback_days    number 回看天数（默认 1825；与 start/end_date 二选一）
    start_date       text   面板起始日 YYYY-MM-DD（空 = 按 lookback_days 推）
    end_date         text   面板结束日 YYYY-MM-DD（空 = 今天）
    factor_subset    text   逗号分隔因子 id；空 = 面板内全部启用因子
    n_layers         number 分层回测层数（默认 5）
    dump_layers      bool   准入通过因子的分层净值/逐日RankIC落盘（默认 True）
    with_diagnostics bool   追加衰减表/相关矩阵诊断产物（默认 False，耗时）
    diag_with_model  bool   诊断3「模型 vs 最强单因子」（默认 False，更耗时）

写 csv 语义：全库模式**覆盖写** ``outputs/factor_eval.csv``；subset 模式
**合并写**——读现有 csv，替换/插入本次评估的因子行，其余行保留
（``merge_factor_eval_csv``，可单测）。
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config as C
from data import load_factor_panel, list_factor_columns
from factor_eval import (
    GATE_IC_T,
    GATE_MONOTONIC,
    GATE_RANKIC,
    GATE_RANKICIR,
    evaluate_factor,
    reports_to_table,
)

from . import context as ctxmod
from .context import kpi, table, fig_line, fig_bar, fig_hist, report

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 评估预过滤（与 factor_eval.evaluate_panel 一致）：有效样本过少或取值退化的
# 因子直接跳过并记入失败清单
_MIN_OBS = 100
_MIN_NUNIQUE = 5

# 诊断产物文件名（scripts/diagnose_factors.py 的 canonical 契约，
# webapp/pages/factor_workbench.py 直接消费）
_DIAG_FILES = ["diag_factor_ic_decay.csv", "diag_factor_corr.csv", "diag_model_vs_factor.csv"]


# ---------------------------------------------------------------------------
# 参数解析小工具
# ---------------------------------------------------------------------------
def _as_bool(v, default: bool = False) -> bool:
    """容错地把会话参数转成 bool（前端 bool 直传；字符串 '0/false/no' 视为假）。"""
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v)


def _parse_subset(text) -> list[str]:
    """factor_subset 文本 → 有序去重的因子 id 列表（支持逗号/空白/分号分隔）。"""
    if not text:
        return []
    parts = str(text).replace("；", ",").replace(";", ",").replace("\n", ",").split(",")
    out, seen = [], set()
    for p in parts:
        fid = p.strip()
        if fid and fid not in seen:
            seen.add(fid)
            out.append(fid)
    return out


def _resolve_date_range(params: dict) -> tuple[str | None, str | None]:
    """解析面板日期区间：显式 start/end_date 优先，否则按 lookback_days 回看。

    返回 (start_date, end_date)，均可为 None（None = load_factor_panel 自默认）。
    """
    start = str(params.get("start_date") or "").strip() or None
    end = str(params.get("end_date") or "").strip() or None
    if start is None:
        lookback = int(float(params.get("lookback_days") or 1825))
        end_dt = pd.Timestamp(end) if end else pd.Timestamp(datetime.now().date())
        start = (end_dt - timedelta(days=lookback)).strftime("%Y-%m-%d")
    return start, end


# ---------------------------------------------------------------------------
# factor_eval.csv 合并写（subset 模式）
# ---------------------------------------------------------------------------
def merge_eval_frames(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """纯函数：把本次评估行合并进既有汇总表。

    语义：以 ``factor`` 为主键——new 中出现的因子整行替换 old 对应行，old 中
    其余因子行原样保留；列结构以 new（本次评估表）为准，old 缺的列补 NaN、
    多出的列丢弃。结果按 |IC_t| 降序（与 evaluate_panel 的输出顺序一致）。
    """
    if old is None or old.empty or "factor" not in old.columns:
        merged = new.copy()
    else:
        kept = old[~old["factor"].isin(new["factor"])].reindex(columns=new.columns)
        merged = pd.concat([kept, new], ignore_index=True)
    if "IC_t" in merged.columns and len(merged):
        merged = merged.assign(_abs=merged["IC_t"].abs())
        merged = merged.sort_values("_abs", ascending=False).drop(columns="_abs")
    return merged.reset_index(drop=True)


def merge_factor_eval_csv(path: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """读现有 csv（没有则视为空）→ 按 factor 合并 new_df → 写回 path，返回合并结果。"""
    old = None
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
        except Exception as ex:  # noqa: BLE001 —— 旧表损坏不应拖垮本次评估
            logger.warning("factor_eval.csv 读取失败（%s），按空表合并", ex)
    merged = merge_eval_frames(old, new_df)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


# ---------------------------------------------------------------------------
# 诊断三件套（衰减表 / 相关矩阵 [/ 模型 vs 最强单因子]）
# ---------------------------------------------------------------------------
def _load_diagnose_module():
    """按文件路径加载 scripts/diagnose_factors.py（scripts/ 非包，不能常规 import）。

    其核心函数（factor_ic_decay / factor_correlation / model_vs_best_factor）
    是可 import 的纯计算，直接调用以复用本会话已加载的 panel，避免子进程
    重新 load_factor_panel。
    """
    path = os.path.join(_PROJECT_ROOT, "scripts", "diagnose_factors.py")
    spec = importlib.util.spec_from_file_location("diagnose_factors", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_diagnostics(
    panel: pd.DataFrame,
    factors: list[str],
    out_dir: str,
    with_model: bool,
    ctx: ctxmod.StepContext | None = None,
) -> dict:
    """生成诊断产物（写 outputs/diag_*.csv），返回 {files, elapsed_sec, error}。

    逐步容错：诊断是附加产物，任何失败只记 error，不向会话步骤抛异常。
    """
    res = {"files": [], "elapsed_sec": 0.0, "error": ""}
    t0 = time.time()
    try:
        mod = _load_diagnose_module()
        os.makedirs(out_dir, exist_ok=True)
        target = C.TARGET

        if ctx is not None:
            ctx.report_progress("诊断1/2 因子衰减表", 0, 2, "factor_ic_decay")
        decay = mod.factor_ic_decay(panel, factors, target)
        p_decay = os.path.join(out_dir, "diag_factor_ic_decay.csv")
        decay.to_csv(p_decay, index=False)
        res["files"].append(p_decay)

        if ctx is not None:
            ctx.report_progress("诊断2/2 相关矩阵", 1, 2, "factor_correlation")
        corr = mod.factor_correlation(panel, factors)
        p_corr = os.path.join(out_dir, "diag_factor_corr.csv")
        corr.to_csv(p_corr)
        res["files"].append(p_corr)

        if with_model:
            if ctx is not None:
                ctx.report_progress("诊断3 模型 vs 最强单因子", 2, 2, "model_vs_best_factor")
            mvb = mod.model_vs_best_factor(panel, factors, target)
            p_mvb = os.path.join(out_dir, "diag_model_vs_factor.csv")
            mvb.to_csv(p_mvb, index=False)
            res["files"].append(p_mvb)
    except Exception as ex:  # noqa: BLE001 —— 附加诊断失败不拖垮主评估
        logger.warning("因子诊断产物生成失败: %s", ex)
        res["error"] = f"{type(ex).__name__}: {ex}"
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------------------
# 步骤 1：数据导入
# ---------------------------------------------------------------------------
def data_import(ctx: ctxmod.StepContext) -> dict:
    """加载命名因子面板（默认全市场近 lookback_days 天，K 线走磁盘缓存）。

    与 scripts/eval_factors.py 同一入口（load_factor_panel 默认参数），面板
    存入 ctx.shared["panel"]，因子清单存入 ctx.shared["factors"]。
    """
    start, end = _resolve_date_range(ctx.params)
    ctx.report_progress("数据加载中", 0, 2, f"load_factor_panel {start} ~ {end or '最新'}")
    panel = load_factor_panel(start_date=start, end_date=end, verbose=True)
    factors = list_factor_columns(panel)
    ctx.shared["panel"] = panel
    ctx.shared["factors"] = factors
    ctx.report_progress("数据画像中", 1, 2, f"{panel['symbol'].nunique():,} 只")

    dates = np.sort(panel["date"].unique())
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
    if per_date.min() < 30:
        notes.append(
            f"⚠️ 部分交易日横截面股票数过少（最小 {int(per_date.min())} 只），"
            "单因子 RankIC 噪声大，准入判定不稳定"
        )
    notes.append(
        f"评估口径：预测窗口 {C.FORWARD_WINDOW} 日 · "
        f"目标 {'横截面超额' if C.MARKET_NEUTRAL_TARGET else '原始'}收益"
    )

    kpis = [
        kpi("股票数", f"{panel['symbol'].nunique():,}", "全样本去重"),
        kpi(
            "交易日",
            f"{len(dates):,}",
            f"{pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()}",
        ),
        kpi("总行数", f"{len(panel):,}", "面板 (date×symbol)"),
        kpi("因子数", f"{len(factors)}", "命名因子（启用家族）"),
        kpi("日期区间", f"{start} ~ {end or '最新'}", "lookback 或显式起止"),
    ]
    summary = (
        f"{panel['symbol'].nunique():,} 只 · {len(dates):,} 交易日 · {len(factors)} 个命名因子"
    )
    return report(
        "data_import", "数据导入", summary=summary, kpis=kpis, figures=figures, notes=notes
    )


# ---------------------------------------------------------------------------
# 步骤 2：因子评估
# ---------------------------------------------------------------------------
def eval_factors(ctx: ctxmod.StepContext) -> dict:
    """逐因子调 evaluate_factor 评估（逐因子上报进度），写 factor_eval.csv。

    全库模式（factor_subset 为空）覆盖写；subset 模式合并写（merge_factor_eval_csv，
    其余因子行保留）。dump_layers=True 时对准入通过的因子落盘分层净值与逐日
    RankIC（factor_layer_/factor_ic_daily_ parquet）。with_diagnostics=True 时
    追加衰减表/相关矩阵诊断产物（耗时）。
    """
    panel = ctx.shared["panel"]
    all_factors = ctx.shared.get("factors") or list_factor_columns(panel)
    p = ctx.params
    subset = _parse_subset(p.get("factor_subset"))
    n_layers = int(float(p.get("n_layers") or 5))
    do_dump = _as_bool(p.get("dump_layers"), default=True)
    do_diag = _as_bool(p.get("with_diagnostics"), default=False)
    diag_model = _as_bool(p.get("diag_with_model"), default=False)

    # subset 模式：按用户给定顺序，只评估面板里存在的因子；不存在的记入失败
    failed: list[dict] = []
    if subset:
        missing = [f for f in subset if f not in panel.columns]
        for f in missing:
            failed.append({"factor": f, "error": "面板中不存在该因子列（未启用或 id 拼写错误）"})
        candidates = [f for f in subset if f in panel.columns]
    else:
        candidates = list(all_factors)

    # 预过滤（与 evaluate_panel 同口径）：样本不足/取值退化的因子跳过
    valid = []
    for f in candidates:
        s = panel[f]
        if s.notna().sum() < _MIN_OBS or s.nunique() < _MIN_NUNIQUE:
            failed.append(
                {"factor": f, "error": f"有效样本不足或取值退化（非空 {int(s.notna().sum())} 行）"}
            )
        else:
            valid.append(f)

    # 逐因子评估（不调 evaluate_panel 整包：需要逐因子上报进度）
    n = len(valid)
    reports = []
    t0 = time.time()
    for i, f in enumerate(valid):
        ctx.report_progress("factors", i, n, f)
        try:
            reports.append(evaluate_factor(panel, factor=f, target=C.TARGET, n_layers=n_layers))
        except Exception as ex:  # noqa: BLE001 —— 单因子失败不拖垮整批
            logger.warning("因子 %s 评估失败: %s", f, ex)
            failed.append({"factor": f, "error": f"{type(ex).__name__}: {ex}"})
    ctx.report_progress("factors", n, n, "评估完成")

    table_df = reports_to_table(reports)
    eval_sec = round(time.time() - t0, 1)

    # 写 outputs/factor_eval.csv：全库覆盖写，subset 合并写
    out_dir = C.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "factor_eval.csv")
    if subset:
        merged = merge_factor_eval_csv(csv_path, table_df)
        csv_note = f"合并写（本次 {len(table_df)} 行，合并后共 {len(merged)} 行）"
    else:
        table_df.to_csv(csv_path, index=False)
        csv_note = f"覆盖写（{len(table_df)} 行）"

    # 准入通过因子的分层净值/逐日 RankIC 落盘（前端因子详情 P2/P3 数据源）
    dump_df = pd.DataFrame()
    if do_dump and not table_df.empty:
        passing = table_df[table_df["准入"] == "✓"]["factor"].tolist()
        if passing:
            from factor_layers import dump_layers

            ctx.report_progress("dump_layers", 0, len(passing), "分层净值落盘")
            dump_df = dump_layers(panel, passing, out_dir, n_layers=n_layers)
            ctx.report_progress(
                "dump_layers", len(passing), len(passing), f"落盘 {int(dump_df['ok'].sum())} 个"
            )

    # 可选：诊断三件套（衰减表 + 相关矩阵 [+ 模型对比]）
    diag = {"files": [], "elapsed_sec": 0.0, "error": ""}
    if do_diag:
        diag_factors = valid if subset else all_factors
        diag = _run_diagnostics(panel, diag_factors, out_dir, diag_model, ctx)

    # 会话 artifacts（eval_report 可仅凭 artifact 重建，不依赖同进程 shared）
    table_df.to_parquet(ctx.artifact_path("eval_table.parquet"), index=False)
    if not dump_df.empty:
        dump_df.to_parquet(ctx.artifact_path("eval_dump_summary.parquet"), index=False)
    meta = {
        "mode": "subset" if subset else "full",
        "subset": subset,
        "n_layers": n_layers,
        "n_evaluated": len(table_df),
        "n_passed": int((table_df["准入"] == "✓").sum()) if not table_df.empty else 0,
        "failed": failed,
        "eval_elapsed_sec": eval_sec,
        "csv_note": csv_note,
        "diagnostics": diag,
        "artifacts": _artifact_listing(out_dir, diag),
    }
    with open(ctx.artifact_path("eval_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    ctx.shared["eval_table"] = table_df
    ctx.shared["eval_failed"] = failed
    ctx.shared["eval_dump"] = dump_df
    ctx.shared["eval_meta"] = meta

    n_pass = meta["n_passed"]
    kpis = [
        kpi(
            "评估因子数", f"{len(table_df)}", f"候选 {len(candidates)} · 预过滤/失败 {len(failed)}"
        ),
        kpi(
            "准入通过",
            f"{n_pass}",
            f"通过率 {(n_pass / len(table_df)):.0%}" if len(table_df) else "—",
        ),
        kpi("评估耗时", f"{eval_sec}s", f"{n_layers} 层分层回测"),
        kpi(
            "分层落盘",
            f"{int(dump_df['ok'].sum())}/{len(dump_df)}" if len(dump_df) else "0",
            "准入通过因子" if do_dump else "dump_layers=关",
        ),
    ]
    if do_diag:
        kpis.append(
            kpi(
                "诊断产物",
                f"{len(diag['files'])} 个",
                f"耗时 {diag['elapsed_sec']}s"
                + (f" · ⚠️ {diag['error']}" if diag["error"] else ""),
            )
        )
    notes = [f"factor_eval.csv：{csv_note} -> {csv_path}"]
    if failed:
        notes.append(f"⚠️ {len(failed)} 个因子评估失败/被跳过（明细见评估报告步骤）")
    if diag.get("error"):
        notes.append(f"⚠️ 诊断产物生成失败：{diag['error']}")
    return report(
        "eval_factors",
        "因子评估",
        summary=(
            f"评估 {len(table_df)} 个因子 · 准入通过 {n_pass} · " f"耗时 {eval_sec}s · {csv_note}"
        ),
        kpis=kpis,
        notes=notes,
    )


def _artifact_listing(out_dir: str, diag: dict) -> list[dict]:
    """产物路径清单（eval_factors/eval_report 共用）。"""
    rows = [
        {"产物": "评估汇总表", "路径": os.path.join(out_dir, "factor_eval.csv")},
        {"产物": "分层净值 parquet（factor_layer_<id>）", "路径": out_dir},
        {"产物": "逐日 RankIC parquet（factor_ic_daily_<id>）", "路径": out_dir},
    ]
    for f in diag.get("files") or []:
        rows.append({"产物": f"诊断 {os.path.basename(f)}", "路径": f})
    return rows


# ---------------------------------------------------------------------------
# 步骤 3：评估报告
# ---------------------------------------------------------------------------
def _load_eval(ctx: ctxmod.StepContext):
    """取评估结果：优先 ctx.shared（同进程），否则从 artifact 重建。"""
    if "eval_table" in ctx.shared:
        return (
            ctx.shared["eval_table"],
            ctx.shared.get("eval_failed") or [],
            ctx.shared.get("eval_dump"),
            ctx.shared.get("eval_meta") or {},
        )
    table_df = pd.read_parquet(ctx.artifact_path("eval_table.parquet"))
    dump_df = None
    try:
        dump_df = pd.read_parquet(ctx.artifact_path("eval_dump_summary.parquet"))
    except Exception:
        pass
    meta = {}
    try:
        with open(ctx.artifact_path("eval_meta.json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        pass
    return table_df, meta.get("failed") or [], dump_df, meta


def eval_report(ctx: ctxmod.StepContext) -> dict:
    """评估报告：准入概览 KPI、RankICIR Top10/Bottom10、失败因子、产物清单。"""
    ctx.report_progress("构造报告", 0, 1)
    table_df, failed, dump_df, meta = _load_eval(ctx)

    if table_df is None or table_df.empty:
        return report(
            "eval_report",
            "评估报告",
            summary="无可展示因子（全部评估失败或被预过滤）",
            notes=["⚠️ 本次评估未产生任何有效因子行"],
        )

    n_pass = int((table_df["准入"] == "✓").sum())
    ricir_med = float(table_df["RankICIR"].median())
    kpis = [
        kpi("评估因子数", f"{len(table_df)}", f"模式：{meta.get('mode', '—')}"),
        kpi("准入通过", f"{n_pass}", f"通过率 {n_pass / len(table_df):.0%}"),
        kpi("RankICIR 中位数", f"{ricir_med:+.3f}", "全体评估因子"),
        kpi(
            "分层落盘",
            (
                f"{int(dump_df['ok'].sum())}/{len(dump_df)}"
                if dump_df is not None and len(dump_df)
                else "0"
            ),
            "factor_layer_/factor_ic_daily_ parquet",
        ),
    ]
    diag = meta.get("diagnostics") or {}
    if diag.get("files"):
        kpis.append(
            kpi("诊断产物", f"{len(diag['files'])} 个", f"耗时 {diag.get('elapsed_sec', 0)}s")
        )

    # 图：RankICIR Top10+Bottom10 合并发散柱状 + RankIC 分布直方图
    by_ricir = table_df.sort_values("RankICIR", ascending=False)
    top10, bot10 = by_ricir.head(10), by_ricir.tail(10)
    tb = pd.concat([top10, bot10]).drop_duplicates(subset="factor")
    figures = [
        fig_bar(
            "RankICIR Top10 / Bottom10",
            tb["factor"].tolist(),
            tb["RankICIR"].to_numpy(),
            yaxis="RankICIR",
            xaxis="因子",
            diverging=True,
        ),
        fig_hist("RankIC 分布（全体评估因子）", table_df["RankIC"].to_numpy(), nbins=40),
    ]

    disp_cols = ["factor", "RankIC", "RankICIR", "IC_t", "多空收益", "单调性", "换手", "准入"]
    tables = [
        table("RankICIR Top10", top10[disp_cols].round(4), max_rows=10),
        table("RankICIR Bottom10", bot10[disp_cols].round(4), max_rows=10),
    ]
    if failed:
        tables.append(table("失败/跳过因子及原因", pd.DataFrame(failed), max_rows=100))
    if dump_df is not None and len(dump_df) and int((~dump_df["ok"]).sum()):
        bad = dump_df[~dump_df["ok"]][["factor", "error"]]
        tables.append(table("分层落盘失败", bad, max_rows=50))
    artifacts = meta.get("artifacts") or _artifact_listing(C.OUTPUT_DIR, diag)
    tables.append(table("产物路径清单", pd.DataFrame(artifacts), max_rows=50))

    notes = [
        f"准入门槛：|IC_t|≥{GATE_IC_T} 且 |RankICIR|≥{GATE_RANKICIR} 且 "
        f"|RankIC|≥{GATE_RANKIC} 且 单调性≥{GATE_MONOTONIC}",
    ]
    if meta.get("mode") == "subset":
        notes.append(
            "subset 模式：factor_eval.csv 为合并写——仅本次评估的因子行被替换，"
            "其余因子行保留历史评估结果"
        )
    if n_pass == 0:
        notes.append("⚠️ 无因子通过准入——信号普遍偏弱，考虑扩因子库或调整预测窗口")
    if diag.get("error"):
        notes.append(f"⚠️ 诊断产物生成失败：{diag['error']}")

    ctx.report_progress("构造报告", 1, 1)
    return report(
        "eval_report",
        "评估报告",
        summary=(f"{len(table_df)} 因子 · 准入 {n_pass} · RankICIR 中位 {ricir_med:+.3f}"),
        kpis=kpis,
        tables=tables,
        figures=figures,
        notes=notes,
    )
