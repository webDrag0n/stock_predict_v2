"""滚动 Walk-Forward 模式的流水线步骤（严格防泄露的样本外验证）。

引擎是 src/walkforward.run_walkforward：每步用前 TRAIN_WINDOW 天训练、embargo
后预测 TEST_WINDOW 天，拼接成一条连续 OOS 净值。train_window/test_window/
wf_step 经 PARAM_ENV 在子进程启动前写入环境变量 → config 读取；select_features/
strategy/capital 由步骤函数从 ctx.params 直接读取。

顺序：wf_data_import → wf_run → wf_report

报告视图（替代已删除的 matplotlib plot_dashboard PNG）：
  图A 账户净值 vs 等权基准 · 图B 累计 alpha · 图C 逐窗口 OOS Sharpe · 图D 回撤
RMSE 面板不迁移：周期调仓引擎的 rmse_curve 恒为 None 占位。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import run_walkforward

from . import context as ctxmod
from .context import kpi, table, fig_line, fig_bar, report

_STRAT_COLOR = "#1a7f37"
_BENCH_COLOR = "#ff7f0e"
_ALPHA_COLOR = "#7b3fb5"
_DD_COLOR = "#d62728"


def _f(x: float, fmt: str, na: str = "—") -> str:
    """容错格式化：None/NaN/inf → na。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    return fmt.format(v) if np.isfinite(v) else na


# ---------------------------------------------------------------------------
def wf_data_import(ctx: ctxmod.StepContext) -> dict:
    """加载命名因子面板（keep_price=True，walkforward 需要 amount 做冲击成本）。"""
    ctx.report_progress("数据加载中", 0, 2, "load_factor_panel")
    panel = load_factor_panel(keep_price=True, verbose=True)
    factors = list_factor_columns(panel)
    ctx.shared["panel"] = panel
    ctx.shared["factors"] = factors
    ctx.report_progress("数据画像中", 1, 2,
                        f"{panel['symbol'].nunique():,} 只")

    dates = np.sort(panel["date"].unique())
    per_date = panel.groupby("date")["symbol"].nunique()

    figures = []
    ret_col = ("return_1d" if "return_1d" in panel.columns
               else ("pct_chg" if "pct_chg" in panel.columns else None))
    if ret_col is not None:
        daily_ret = panel.groupby("date")[ret_col].mean()
        if ret_col == "pct_chg":
            daily_ret = daily_ret / 100.0
        cum = (1.0 + daily_ret).cumprod() - 1.0
        figures.append(fig_line(
            "全市场等权累计收益（大盘走势）", cum.index.astype(str).tolist(),
            [{"name": "全市场等权", "y": (cum.to_numpy() * 100)}],
            yaxis="累计收益 %", x_is_date=True))
    figures.append(fig_line(
        "每日横截面股票数", per_date.index.astype(str).tolist(),
        [{"name": "股票数", "y": per_date.to_numpy()}],
        yaxis="只", x_is_date=True))

    n_win = max((len(dates) - C.TRAIN_WINDOW - C.EMBARGO_DAYS) // C.WF_STEP + 1, 0)
    notes = []
    if per_date.min() < 30:
        notes.append(f"⚠️ 部分交易日横截面股票数过少（最小 {int(per_date.min())} 只），"
                     "排序策略可能不稳定")
    if n_win <= 0:
        notes.append(f"⚠️ 按当前 train={C.TRAIN_WINDOW}/test={C.TEST_WINDOW}/"
                     f"step={C.WF_STEP} 配置不足以构成任何滚动窗口")
    if "amount" not in panel.columns:
        notes.append("⚠️ 面板缺 amount 列，市场冲击成本将被禁用（仅佣金+印花税）")

    kpis = [
        kpi("股票数", f"{panel['symbol'].nunique():,}", "全样本去重"),
        kpi("交易日", f"{len(dates):,}",
            f"{pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()}"),
        kpi("总行数", f"{len(panel):,}", "面板 (date×symbol)"),
        kpi("因子数", f"{len(factors)}", "命名因子"),
        kpi("滚动窗口配置", f"{C.TRAIN_WINDOW}/{C.TEST_WINDOW}/{C.WF_STEP}",
            f"训练/测试/步长 · embargo={C.EMBARGO_DAYS}"),
        kpi("预计窗口数", f"{n_win}", "OOS 窗口互不重叠时为顺推"),
    ]
    summary = (f"{panel['symbol'].nunique():,} 只 · {len(dates):,} 交易日 · "
               f"{len(factors)} 因子 · 预计 {n_win} 窗口")
    return report("wf_data_import", "数据导入", summary=summary, kpis=kpis,
                  figures=figures, notes=notes)


# ---------------------------------------------------------------------------
def wf_run(ctx: ctxmod.StepContext) -> dict:
    """跑滚动 walk-forward，落盘 OOS 净值曲线 / 逐窗口指标 / 基准日收益。"""
    panel = ctx.shared["panel"]
    factors = ctx.shared["factors"]
    p = ctx.params
    select_features = bool(p.get("select_features", False))
    strategy = str(p.get("strategy") or "long_topk")

    ctx.report_progress("滚动回测中", 0, 1,
                        f"train={C.TRAIN_WINDOW} test={C.TEST_WINDOW} "
                        f"step={C.WF_STEP} strategy={strategy} "
                        f"select_features={select_features}")
    res = run_walkforward(panel, factors, select_features=select_features,
                          strategy=strategy, verbose=True)
    ctx.shared["wf"] = res
    ctx.report_progress("滚动回测完成", 1, 1, n_windows=res["n_windows"])

    # 落盘 artifact：OOS 曲线 + 逐窗口指标 + 基准日收益（wf_report 可仅凭
    # artifact 重建，不依赖同进程 shared）
    curve = res["oos_result"]["curve"]
    wm = res["window_metrics"]
    curve.to_parquet(ctx.artifact_path("oos_curve.parquet"), index=False)
    wm.to_parquet(ctx.artifact_path("window_metrics.parquet"), index=False)
    oos = res.get("oos")
    if oos is not None and "return_1d" in oos.columns:
        bench = (oos.groupby("date")["return_1d"].mean().sort_index()
                 .reset_index().rename(columns={"return_1d": "bench_ret"}))
        bench.to_parquet(ctx.artifact_path("oos_bench.parquet"), index=False)

    m = res["oos_result"]["metrics"]
    n_pos_win = int((wm["Sharpe"] > 0).sum()) if len(wm) else 0
    kpis = [
        kpi("OOS 窗口数", f"{res['n_windows']}",
            f"Sharpe>0 的窗口 {n_pos_win}/{len(wm)}"),
        kpi("连续 OOS Sharpe", _f(m.get("sharpe"), "{:+.2f}"), "拼接全部窗口"),
        kpi("窗口 Sharpe 均值", _f(wm["Sharpe"].mean() if len(wm) else None, "{:+.2f}"),
            f"最差 {_f(wm['Sharpe'].min() if len(wm) else None, '{:+.2f}')}"),
        kpi("总 PnL", _f(m.get("total_pnl"), "{:+.4f}"), "占本金分数"),
        kpi("日胜率", _f(m.get("hit_rate"), "{:.1%}"), ""),
        kpi("PredIC", _f(m.get("pred_ic"), "{:+.4f}"),
            f"ICIR {_f(m.get('pred_icir'), '{:+.3f}')}"),
    ]
    return report("wf_run", "滚动 Walk-Forward",
                  summary=(f"{res['n_windows']} 窗口 · OOS Sharpe "
                           f"{_f(m.get('sharpe'), '{:+.2f}')} · "
                           f"为正窗口 {n_pos_win}/{len(wm)}"),
                  kpis=kpis)


# ---------------------------------------------------------------------------
def _load_wf(ctx: ctxmod.StepContext):
    """取 walkforward 结果：优先 ctx.shared（同进程），否则从 artifact 重建。"""
    res = ctx.shared.get("wf")
    if res is not None:
        curve = res["oos_result"]["curve"]
        wm = res["window_metrics"]
        oos = res.get("oos")
        bench = None
        if oos is not None and "return_1d" in oos.columns:
            bench = (oos.groupby("date")["return_1d"].mean()
                     .sort_index().to_numpy())
        dts = (np.sort(oos["date"].unique()) if oos is not None else None)
        return res["oos_result"]["metrics"], curve, wm, bench, dts
    # artifact 重建（shared 缺失时：benchmark/dates 从 oos_bench 恢复）
    curve = pd.read_parquet(ctx.artifact_path("oos_curve.parquet"))
    wm = pd.read_parquet(ctx.artifact_path("window_metrics.parquet"))
    bench = dts = None
    try:
        bdf = pd.read_parquet(ctx.artifact_path("oos_bench.parquet"))
        dts = bdf["date"].to_numpy()
        bench = bdf["bench_ret"].to_numpy()
    except Exception:
        pass
    return {}, curve, wm, bench, dts


def wf_report(ctx: ctxmod.StepContext) -> dict:
    """OOS 绩效报告：净值 vs 基准、累计 alpha、逐窗口 Sharpe、回撤、窗口表。"""
    ctx.report_progress("构造报告", 0, 1)
    m, curve, wm, bench, dts = _load_wf(ctx)
    cap0 = float(ctx.params.get("capital") or C.INITIAL_CAPITAL)

    net = curve["net"].to_numpy(dtype=np.float64)
    n = len(net)
    if n == 0:
        return report("wf_report", "OOS 绩效报告", summary="无 OOS 数据",
                      notes=["⚠️ walk-forward 未产生任何样本外交易日"])
    x = ([str(pd.Timestamp(d).date()) for d in dts[:n]]
         if dts is not None and len(dts) >= n else list(range(n)))
    x_is_date = dts is not None and len(dts) >= n

    account = cap0 * np.cumprod(1.0 + net)
    cum_ret_pct = (account / cap0 - 1.0) * 100.0
    peak = np.maximum.accumulate(account)
    dd_pct = (account - peak) / peak * 100.0

    bench_account = alpha_pct = None
    if bench is not None and len(bench) >= n:
        bench_ret = np.cumprod(1.0 + bench[:n])
        bench_account = cap0 * bench_ret
        alpha_pct = cum_ret_pct - (bench_ret - 1.0) * 100.0

    # 图A：净值 vs 基准（金额口径）
    series_a = [{"name": "策略", "y": account, "width": 2.5, "color": _STRAT_COLOR}]
    if bench_account is not None:
        series_a.append({"name": "等权基准", "y": bench_account,
                         "color": _BENCH_COLOR, "dash": "dash"})
    figures = [fig_line("账户净值 vs 等权基准", x, series_a,
                        yaxis="账户价值 (元)", x_is_date=x_is_date)]
    # 图B：累计 alpha
    if alpha_pct is not None:
        figures.append(fig_line("累计 alpha（策略 − 基准）", x,
                                [{"name": "alpha", "y": alpha_pct,
                                  "color": _ALPHA_COLOR}],
                                yaxis="累计超额 (百分点)", x_is_date=x_is_date))
    # 图C：逐窗口 OOS Sharpe
    if len(wm):
        figures.append(fig_bar("逐窗口 OOS Sharpe（稳健性）",
                               [f"W{int(w)}" for w in wm["window"]],
                               wm["Sharpe"].to_numpy(),
                               yaxis="Sharpe", xaxis="窗口", diverging=True))
    # 图D：回撤
    figures.append(fig_line("回撤", x,
                            [{"name": "回撤", "y": dd_pct, "color": _DD_COLOR}],
                            yaxis="回撤 %", x_is_date=x_is_date))

    cost_comm = m.get("total_cost_commission")
    cost_impact = m.get("total_cost_impact")
    cost_txt = None
    if (cost_comm is not None and cost_impact is not None
            and np.isfinite(float(cost_comm)) and np.isfinite(float(cost_impact))):
        cost_txt = f"¥{(cost_comm + cost_impact) * cap0:,.0f}"
    total_pnl = m.get("total_pnl")
    pnl_hint = (f"≈ ¥{float(total_pnl) * cap0:,.0f}"
                if total_pnl is not None and np.isfinite(float(total_pnl)) else "")
    kpis = [
        kpi("Sharpe", _f(m.get("sharpe"), "{:+.2f}"), "日收益年化"),
        kpi("Sortino", _f(m.get("sortino"), "{:+.2f}"), ""),
        kpi("Calmar", _f(m.get("calmar"), "{:+.2f}"), "年化收益/|MaxDD|"),
        kpi("最大回撤", _f(m.get("max_drawdown"), "{:+.2%}"), "占本金分数"),
        kpi("日胜率", _f(m.get("hit_rate"), "{:.1%}"), ""),
        kpi("总 PnL", _f(total_pnl, "{:+.4f}"), pnl_hint),
        kpi("成本合计", cost_txt or "—",
            (f"佣金 ¥{cost_comm * cap0:,.0f} + 冲击 ¥{cost_impact * cap0:,.0f}"
             if cost_txt else "")),
        kpi("PredIC", _f(m.get("pred_ic"), "{:+.4f}"),
            f"ICIR {_f(m.get('pred_icir'), '{:+.3f}')}"),
    ]

    notes = []
    if len(wm):
        n_pos = int((wm["Sharpe"] > 0).sum())
        notes.append(f"Sharpe>0 窗口占比 {n_pos}/{len(wm)} —— "
                     "衡量信号能否跨时间稳健复现；连续 OOS Sharpe 通常远低于"
                     "单窗口数字，这才是可信的真实水平。")
        if n_pos * 2 < len(wm):
            notes.append("⚠️ 过半窗口 OOS Sharpe ≤ 0，信号跨窗口稳健性不足")
    if alpha_pct is not None:
        notes.append(f"期末累计 alpha {alpha_pct[-1]:+.1f} 个百分点（vs 等权基准）")

    tables = []
    if len(wm):
        disp = wm.copy()
        for c in ("PnL", "Sharpe", "MaxDD", "hit", "PriceRMSE", "PredIC"):
            if c in disp.columns:
                disp[c] = disp[c].astype(float).round(4)
        tables.append(table("逐窗口 OOS 指标", disp, max_rows=100))

    ctx.report_progress("构造报告", 1, 1)
    return report("wf_report", "OOS 绩效报告",
                  summary=(f"Sharpe {_f(m.get('sharpe'), '{:+.2f}')} · "
                           f"最大回撤 {_f(m.get('max_drawdown'), '{:+.2%}')} · "
                           f"总 PnL {_f(m.get('total_pnl'), '{:+.4f}')}"),
                  kpis=kpis, figures=figures, notes=notes, tables=tables)
