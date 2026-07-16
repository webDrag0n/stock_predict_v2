"""回测模式的流水线步骤（每日策略 + 逐笔模拟重放）。

每个步骤是 `fn(ctx: StepContext) -> dict`（返回 step report）。步骤间通过
ctx.shared 传递内存对象（panel/factors/replay），通过 ctx.artifacts_dir 落盘
轻量产物（每日状态 / 逐笔成交 / 模型权重）。

顺序：data_import → factor_analysis → daily_replay → performance

新引擎（backtest_daily.run_daily_replay）取代旧的固定周期 walkforward 回测：
每个交易日调用共享每日决策核心（trading.decide_orders：ML 选池 + 布林/RSI 择时），
用次日真实行情逐笔模拟成交，维护现金+持仓的真实账户状态。可设初始资金/初始持仓，
记录每笔模拟交易与每日收盘状态，画出 总值/股票市值/现金 组合图、持仓曲线等。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

import config as C
from data import load_factor_panel, list_factor_columns
from factor_eval import evaluate_panel
from backtest_daily import run_daily_replay, compute_trade_stats
from trading import parse_initial_holdings
from measure_alpha import _newey_west_alpha_t

from . import context as ctxmod
from .context import (kpi, table, fig_line, fig_bar, fig_scatter, fig_hist,
                      fig_area_stacked, report)


# 回测默认窗口：留空 start_date 时用今天 - 92 天 (3 个月含容差)
BACKTEST_DEFAULT_LOOKBACK_DAYS = 92


def _backtest_start_date(ctx: ctxmod.StepContext) -> str | None:
    """用户填了 start_date 用填的；否则默认今天 - 92 天。空字符串等同 None。"""
    raw = (ctx.params.get("start_date") or "").strip()
    if raw:
        return raw
    return (date.today() - timedelta(days=BACKTEST_DEFAULT_LOOKBACK_DAYS)).isoformat()


def C_up():
    return "#16a34a"


# ---------------------------------------------------------------------------
def data_import(ctx: ctxmod.StepContext) -> dict:
    """加载命名因子面板（保留 OHLC + 布林带 meta 供逐笔撮合/择时）并产出数据画像。"""
    ctx.report_progress("数据加载中", 0, 2, "load_factor_panel")
    panel = load_factor_panel(keep_trade_cols=True, verbose=True)
    factors = list_factor_columns(panel)
    ctx.shared["panel"] = panel
    ctx.shared["factors"] = factors
    ctx.report_progress("数据画像中", 1, 2, f"{panel['symbol'].nunique():,} 只")

    panel = panel.sort_values("date")
    dates = np.sort(panel["date"].unique())
    per_date = panel.groupby("date")["symbol"].nunique()
    y = panel[C.TARGET].dropna()

    ret_col = ("return_1d" if "return_1d" in panel.columns
               else ("pct_chg" if "pct_chg" in panel.columns else None))
    figures = []
    if ret_col is not None:
        daily_ret = panel.groupby("date")[ret_col].mean()
        if ret_col == "pct_chg":
            daily_ret = daily_ret / 100.0
        cum = (1.0 + daily_ret).cumprod() - 1.0
        figures.append(fig_line(
            "全市场等权累计收益（大盘走势）", cum.index.astype(str).tolist(),
            [{"name": "全市场等权", "y": (cum.to_numpy() * 100)}],
            yaxis="累计收益 %", x_is_date=True))
        adv = panel.assign(_up=(panel[ret_col] > 0)).groupby("date")["_up"].mean()
        figures.append(fig_line(
            "每日上涨家数占比（市场广度）", adv.index.astype(str).tolist(),
            [{"name": "上涨占比", "y": (adv.to_numpy() * 100), "color": C_up()}],
            yaxis="上涨占比 %", x_is_date=True))

    figures.append(fig_line(
        "每日横截面股票数", per_date.index.astype(str).tolist(),
        [{"name": "股票数", "y": per_date.to_numpy()}],
        yaxis="只", x_is_date=True))
    figures.append(fig_hist("标签(responder_6)分布", y.to_numpy(), nbins=80,
                            xaxis="前瞻收益"))

    notes = []
    if per_date.min() < 30:
        notes.append(f"⚠️ 部分交易日横截面股票数过少（最小 {int(per_date.min())} 只），排序策略可能不稳定")
    has_ohlc = all(c in panel.columns for c in ("open", "high", "low"))
    has_bb = "bb_lower_20" in panel.columns and "bb_upper_20" in panel.columns
    if not has_ohlc:
        notes.append("⚠️ 面板缺 OHLC 列，次日撮合将退化为按 close 估值")
    if not has_bb:
        notes.append("⚠️ 面板缺布林带列，将仅按 ML 排序选股（无指标择时）")

    kpis = [
        kpi("股票数", f"{panel['symbol'].nunique():,}", "全样本去重"),
        kpi("交易日", f"{len(dates):,}", f"{pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()}"),
        kpi("总行数", f"{len(panel):,}", "面板 (date×symbol)"),
        kpi("因子数", f"{len(factors)}", "命名因子（已排除 OHLC/布林带 meta）"),
        kpi("日均横截面", f"{per_date.mean():.0f}", f"最少 {int(per_date.min())}"),
        kpi("撮合/择时列", f"{'OHLC✓' if has_ohlc else 'OHLC✗'} {'BB✓' if has_bb else 'BB✗'}",
            "逐笔模拟所需 meta"),
    ]
    summary = (f"{panel['symbol'].nunique():,} 只 · {len(dates):,} 交易日 · "
               f"{len(factors)} 因子")
    return report("data_import", "数据导入", summary=summary, kpis=kpis,
                  figures=figures, notes=notes)


# ---------------------------------------------------------------------------
def factor_analysis(ctx: ctxmod.StepContext) -> dict:
    """单因子 IC/ICIR/分层评估，排序展示因子强弱。"""
    panel = ctx.shared["panel"]
    factors = ctx.shared["factors"]
    ctx.report_progress("因子评估中", 0, 1, f"{len(factors)} 因子")
    tbl = evaluate_panel(panel, factors=factors)
    if tbl.empty:
        return report("factor_analysis", "因子分析",
                      summary="无可评估因子", notes=["⚠️ evaluate_panel 返回空表"])

    n_pass = int((tbl["准入"] == "✓").sum())
    top = tbl.head(20)
    figures = [
        fig_bar("Top20 因子 RankIC（按显著性排序）",
                top["factor"].tolist(), top["RankIC"].to_numpy(),
                yaxis="RankIC", diverging=True),
        fig_bar("Top20 因子 RankICIR",
                top["factor"].tolist(), top["RankICIR"].to_numpy(),
                yaxis="RankICIR", diverging=True),
    ]
    disp = tbl.copy()
    for c in ("RankIC", "RankICIR", "RankICIR_ann", "IC", "ICIR", "IC_t",
              "IC>0占比", "多空收益", "单调性", "换手"):
        if c in disp.columns:
            disp[c] = disp[c].astype(float).round(4)
    kpis = [
        kpi("评估因子数", f"{len(tbl)}", "横截面≥100行且≥5取值"),
        kpi("通过准入", f"{n_pass}", "IC/ICIR/单调性门槛"),
        kpi("最强因子", f"{tbl.iloc[0]['factor']}", f"IC_t={tbl.iloc[0]['IC_t']:+.2f}"),
    ]
    return report("factor_analysis", "因子分析",
                  summary=f"{len(tbl)} 因子评估 · {n_pass} 个通过准入",
                  kpis=kpis, figures=figures,
                  tables=[table("因子评估汇总（按 |IC_t| 排序）", disp, max_rows=80)])


# ---------------------------------------------------------------------------
def _replay_params(ctx: ctxmod.StepContext) -> dict:
    """从会话参数组装 run_daily_replay 的 params 覆盖。"""
    p = ctx.params
    out = {}
    # 决策/择时
    for key, pk in (("topk", "topk"), ("hold_ratio", "hold_ratio"),
                    ("max_positions", "max_positions"),
                    ("max_weight", "max_weight"),
                    ("bb_buy_k", "bb_buy_k"), ("bb_sell_k", "bb_sell_k"),
                    ("rsi_buy_max", "rsi_buy_max"),
                    ("rsi_sell_min", "rsi_sell_min"),
                    ("min_notional", "min_notional"),
                    ("dynamic_topk_ic_full", "dynamic_topk_ic_full"),
                    ("dynamic_topk_ic_zero", "dynamic_topk_ic_zero"),
                    ("dynamic_topk_low", "dynamic_topk_low"),
                    ("dynamic_topk_high", "dynamic_topk_high"),
                    ("max_portfolio_beta", "max_portfolio_beta")):
        if p.get(pk) is not None:
            out[key] = float(p[pk])
    if p.get("use_bb_timing") is not None:
        out["use_bb_timing"] = bool(p["use_bb_timing"])
    if p.get("use_dynamic_topk") is not None:
        out["use_dynamic_topk"] = bool(p["use_dynamic_topk"])
    if p.get("use_beta_constraint") is not None:
        out["use_beta_constraint"] = bool(p["use_beta_constraint"])
    if p.get("beta_lookback") is not None:
        out["beta_lookback"] = int(p["beta_lookback"])
    if p.get("order_price_mode"):
        out["order_price_mode"] = str(p["order_price_mode"])
    if p.get("use_impact") is not None:
        out["use_impact"] = bool(p["use_impact"])
    # 增量集成
    for key in ("n_seeds", "incr_trees", "rebuild_days", "train_window",
                "ic_window"):
        if p.get(key) is not None:
            out[key] = int(p[key])
    return out


def daily_replay(ctx: ctxmod.StepContext) -> dict:
    """逐日重放回测：每日 ML 选池 + 布林/RSI 择时 + 次日逐笔撮合。落盘每日状态/逐笔成交。"""
    panel = ctx.shared["panel"]
    factors = ctx.shared["factors"]
    p = ctx.params

    initial_cash = float(p.get("capital", C.INITIAL_CAPITAL))
    init_holdings = parse_initial_holdings(p.get("initial_holdings", ""))
    start_date = _backtest_start_date(ctx)
    end_date = (p.get("end_date") or "").strip() or None
    rp = _replay_params(ctx)

    # 把 progress 回调从 ctx 注入 run_daily_replay 的 params
    def _cb(phase, current, total, **extras):
        ctx.report_progress(phase, current, total, **extras)
    rp["progress_cb"] = _cb

    ctx.report_progress("回放启动", 0, 1, f"start_date={start_date or '未限'}",
                        n_models=len(rp.get("ensemble_models", []) or []) or 3)
    res = run_daily_replay(
        panel, factors, initial_cash=initial_cash,
        initial_holdings=init_holdings, params=rp,
        start_date=start_date, end_date=end_date, progress=True)
    ctx.shared["replay"] = res
    daily = res["daily"]; trades = res["trades"]; weights = res["weights"]
    ctx.report_progress("回放完成", 1, 1, n_days=len(daily),
                        n_trades=int(len(trades)))

    # 落盘轻量产物
    daily.to_parquet(ctx.artifact_path("daily_state.parquet"), index=False)
    if len(trades):
        trades.to_parquet(ctx.artifact_path("trades.parquet"), index=False)
    if len(weights):
        weights.to_parquet(ctx.artifact_path("model_weights.parquet"), index=False)

    n_days = len(daily)
    tv = daily["total_value"].to_numpy() if n_days else np.array([initial_cash])
    final_v = float(tv[-1]) if n_days else initial_cash
    total_ret = final_v / initial_cash - 1.0
    n_trades = int(len(trades))
    fill_rate = (float(daily["n_fill"].sum() /
                       max(daily["n_fill"].sum() + daily["n_reject"].sum(), 1))
                 if n_days else float("nan"))

    figures = []
    if n_days:
        dts = [str(pd.Timestamp(x).date()) for x in daily["date"]]
        figures.append(fig_line(
            "组合价值构成（总值 = 股票市值 + 现金）", dts,
            [{"name": "总价值", "y": tv, "width": 2.5},
             {"name": "股票市值", "y": daily["stock_value"].to_numpy(), "color": "#1a7f37"},
             {"name": "现金", "y": daily["cash"].to_numpy(), "color": "#ff7f0e"}],
            yaxis="价值 (元)", x_is_date=True))
        figures.append(fig_area_stacked(
            "现金/股票市值堆叠（=总价值）", dts,
            [{"name": "现金", "y": daily["cash"].to_numpy(), "color": "#fbbf24"},
             {"name": "股票市值", "y": daily["stock_value"].to_numpy(), "color": "#1a7f37"}],
            yaxis="价值 (元)", x_is_date=True))
        figures.append(fig_line(
            "持仓数量曲线", dts,
            [{"name": "持仓只数", "y": daily["n_positions"].to_numpy(), "color": "#7b3fb5"}],
            yaxis="只", x_is_date=True))
        figures.append(fig_line(
            "每日成交笔数（成交 / 拒绝）", dts,
            [{"name": "成交", "y": daily["n_fill"].to_numpy(), "color": "#1a7f37"},
             {"name": "拒绝", "y": daily["n_reject"].to_numpy(), "color": "#d62728"}],
            yaxis="笔", x_is_date=True))

        # --- Option IV macro feature visualization ---
        if "opt_vix_like" in panel.columns:
            iv_daily = panel.groupby("date")["opt_vix_like"].first().reindex(daily["date"])
            iv_arr = iv_daily.to_numpy()
            if np.isfinite(iv_arr).any():
                figures.append(fig_line(
                    "期权隐含波动率 (QVIX-like)", dts,
                    [{"name": "opt_vix_like", "y": iv_arr * 100, "color": "#7b3fb5"}],
                    yaxis="年化隐含波动率 %", x_is_date=True))
        if "risk_scale" in daily.columns:
            figures.append(fig_line(
                "风险缩放系数", dts,
                [{"name": "risk_scale", "y": daily["risk_scale"].to_numpy(), "color": "#d62728"}],
                yaxis="杠杆乘数", x_is_date=True))

    iv_coverage = 0.0
    if n_days and "opt_vix_like" in panel.columns:
        iv_daily = panel.groupby("date")["opt_vix_like"].first().reindex(daily["date"])
        iv_coverage = float(iv_daily.notna().mean())

    kpis = [
        kpi("初始资金", f"¥{initial_cash:,.0f}", f"初始持仓 {len(init_holdings)} 只"),
        kpi("期末总值", f"¥{final_v:,.0f}", f"总收益 {total_ret:+.1%}"),
        kpi("回测交易日", f"{n_days}", f"{res['meta']['elapsed_sec']}s"),
        kpi("逐笔成交", f"{n_trades:,}", f"成交率 {fill_rate:.0%}"),
        kpi("期末持仓", f"{len(res['state'].held_symbols())}", "只"),
        kpi("期末现金", f"¥{res['state'].cash:,.0f}", ""),
        kpi("IV 特征覆盖率", f"{iv_coverage:.1%}", "opt_vix_like 无缺失交易日占比"),
    ]
    notes = []
    if n_days == 0:
        notes.append("⚠️ 无回测交易日，请检查预热期/日期区间设置")
    return report("daily_replay", "每日重放回测",
                  summary=f"{n_days} 交易日 · 期末 ¥{final_v:,.0f} ({total_ret:+.1%}) · {n_trades:,} 笔成交",
                  kpis=kpis, figures=figures, notes=notes,
                  tables=[table("每日收盘状态（最近优先截断）",
                                _daily_disp(daily), max_rows=60)] if n_days else [])


def _daily_disp(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["date"] = d["date"].astype(str)
    for c in ("cash", "stock_value", "total_value", "turnover", "cost"):
        if c in d.columns:
            d[c] = d[c].round(0)
    cols = [c for c in ("date", "cash", "stock_value", "total_value",
                        "n_positions", "turnover", "cost", "n_fill", "n_reject")
            if c in d.columns]
    return d[cols]


# ---------------------------------------------------------------------------
def performance(ctx: ctxmod.StepContext) -> dict:
    """绩效分析：净值 vs 基准、alpha(Newey-West)、回撤、成本拆解、模型权重、
    逐笔交易统计（胜率/盈亏比/持有天数）、交易明细表。"""
    res = ctx.shared["replay"]
    panel = ctx.shared["panel"]
    daily = res["daily"]; trades = res["trades"]; weights = res["weights"]
    ctx.report_progress("绩效计算中", 0, 1, f"{len(daily)} 交易日")
    if len(daily) == 0:
        return report("performance", "绩效分析", summary="无回测数据",
                      notes=["⚠️ daily_replay 未产生任何交易日"])

    initial_cash = res["meta"]["initial_cash"]
    dts_raw = daily["date"].to_numpy()
    dts = [str(pd.Timestamp(x).date()) for x in dts_raw]
    tv = daily["total_value"].to_numpy(dtype=np.float64)
    # 策略日收益率（基于总值）
    strat_ret = np.zeros(len(tv))
    strat_ret[1:] = tv[1:] / tv[:-1] - 1.0
    strat_cum = (tv / initial_cash - 1.0) * 100.0

    # 基准：全市场等权日收益，对齐到回测交易日
    ret_col = ("return_1d" if "return_1d" in panel.columns
               else ("pct_chg" if "pct_chg" in panel.columns else None))
    bench_cum = None; alpha_ann = beta = t_nw = float("nan")
    if ret_col is not None:
        bser = panel.groupby("date")[ret_col].mean()
        if ret_col == "pct_chg":
            bser = bser / 100.0
        bench = bser.reindex(pd.Index(dts_raw)).fillna(0.0).to_numpy()
        bench_cum = (np.cumprod(1.0 + bench) - 1.0) * 100.0
        # CAPM + Newey-West（滞后≈调仓持有期，用 FORWARD_WINDOW 兜底）
        lags = int(ctx.params.get("rebuild_days", getattr(C, "REBALANCE_DAYS", 20)))
        a_d, beta, t_nw, _ = _newey_west_alpha_t(
            strat_ret[1:], bench[1:], max(1, min(lags, len(strat_ret) - 2)))
        alpha_ann = a_d * 252

    # 回撤（按总值）
    peak = np.maximum.accumulate(tv)
    dd = (tv - peak) / peak * 100.0
    max_dd = float(dd.min())

    # Sharpe / Sortino（日收益年化）
    rr = strat_ret[1:]
    sharpe = float(rr.mean() / rr.std() * np.sqrt(252)) if rr.std() > 1e-12 else 0.0
    downside = rr[rr < 0]
    sortino = (float(rr.mean() / downside.std() * np.sqrt(252))
               if len(downside) > 1 and downside.std() > 1e-12 else 0.0)
    hit = float((rr > 0).mean()) if len(rr) else float("nan")

    figures = []
    notes = []
    series = [{"name": "策略", "y": strat_cum, "width": 2.5}]
    if bench_cum is not None:
        series.append({"name": "等权基准", "y": bench_cum, "color": "#ff7f0e"})
    figures.append(fig_line("策略净值 vs 基准（累计收益）", dts, series,
                            yaxis="累计收益 %", x_is_date=True))
    if bench_cum is not None:
        figures.append(fig_line("超额收益 alpha 曲线", dts,
                                [{"name": "alpha", "y": (strat_cum - bench_cum),
                                  "color": "#7b3fb5"}],
                                yaxis="累计超额 %", x_is_date=True))
    figures.append(fig_line("回撤", dts,
                            [{"name": "回撤", "y": dd, "color": "#d62728"}],
                            yaxis="回撤 %", x_is_date=True))
    # 成本拆解（累计）
    if {"commission", "stamp", "impact"} <= set(daily.columns):
        figures.append(fig_area_stacked(
            "交易成本拆解（累计）", dts,
            [{"name": "佣金", "y": np.cumsum(daily["commission"].to_numpy()), "color": "#2563eb"},
             {"name": "印花税", "y": np.cumsum(daily["stamp"].to_numpy()), "color": "#9333ea"},
             {"name": "市场冲击", "y": np.cumsum(daily["impact"].to_numpy()), "color": "#c2410c"}],
            yaxis="累计成本 (元)", x_is_date=True))
    # 每日换手
    if "turnover" in daily.columns:
        figures.append(fig_line("每日换手额", dts,
                                [{"name": "换手额", "y": daily["turnover"].to_numpy(),
                                  "color": "#0891b2"}],
                                yaxis="成交额 (元)", x_is_date=True))
    # 模型权重随时间
    if len(weights):
        wdts = [str(pd.Timestamp(x).date()) for x in weights["date"]]
        wseries = [{"name": c[2:], "y": weights[c].to_numpy()}
                   for c in weights.columns if c.startswith("w_")]
        figures.append(fig_line("多模型 IC 权重随时间", wdts, wseries,
                                yaxis="权重", x_is_date=True))

    # IV regime return analysis
    if "opt_vix_like" in panel.columns and len(daily) >= 10:
        iv_daily = panel.groupby("date")["opt_vix_like"].first().reindex(daily["date"])
        valid = iv_daily.notna() & np.isfinite(strat_ret)
        if valid.sum() >= 10:
            tmp = pd.DataFrame({
                "iv": iv_daily,
                "ret": strat_ret,
            }).dropna()
            if len(tmp) >= 10:
                try:
                    tmp["regime"] = pd.qcut(tmp["iv"], 3, labels=["低 IV", "中 IV", "高 IV"])
                    regime_ret = tmp.groupby("regime")["ret"].agg(["mean", "std", "count"])
                    figures.append(fig_bar(
                        "IV Regime 日均收益", regime_ret.index.tolist(),
                        regime_ret["mean"].to_numpy(),
                        yaxis="日均收益", diverging=True))
                    notes.append(
                        f"IV Regime: 高 IV 日均收益={regime_ret.loc['高 IV', 'mean']:+.4f} "
                        f"(n={int(regime_ret.loc['高 IV', 'count'])}), "
                        f"低 IV 日均收益={regime_ret.loc['低 IV', 'mean']:+.4f} "
                        f"(n={int(regime_ret.loc['低 IV', 'count'])}). "
                        "正值说明高波动环境对策略有利，负值说明应降仓避险。"
                    )
                except Exception:
                    pass

    # 逐笔交易统计
    stats = compute_trade_stats(trades)
    rt = stats.get("round_trips", pd.DataFrame())
    if isinstance(rt, pd.DataFrame) and len(rt):
        figures.append(fig_hist("每笔已实现收益率分布（配对卖出）",
                                rt["ret"].to_numpy() * 100, nbins=60,
                                xaxis="收益率 %"))
        # 持有天数（按交易日序号近似）
        date_rank = {d: i for i, d in enumerate(np.sort(panel["date"].unique()))}
        hold_days = rt.apply(
            lambda r: date_rank.get(r["sell_date"], 0) - date_rank.get(r["buy_date"], 0),
            axis=1).to_numpy()
        figures.append(fig_hist("持有天数分布", hold_days, nbins=40, xaxis="交易日"))

    kpis = [
        kpi("Sharpe", f"{sharpe:+.2f}", "日收益年化"),
        kpi("Sortino", f"{sortino:+.2f}", ""),
        kpi("年化 alpha", (f"{alpha_ann:+.1%}" if alpha_ann == alpha_ann else "—"),
            f"NW-t={t_nw:+.2f}" if t_nw == t_nw else "无基准"),
        kpi("Beta", (f"{beta:+.2f}" if beta == beta else "—"), "vs 等权基准"),
        kpi("最大回撤", f"{max_dd:+.1f}%", ""),
        kpi("日胜率", (f"{hit:.1%}" if hit == hit else "—"), ""),
        kpi("交易胜率", (f"{stats['win_rate']:.1%}" if stats['win_rate'] == stats['win_rate'] else "—"),
            f"{stats['n_round_trips']} 配对"),
        kpi("盈亏比", (f"{stats['profit_factor']:.2f}" if stats['profit_factor'] == stats['profit_factor'] else "—"),
            f"已实现 ¥{stats['realized_pnl']:,.0f}"),
    ]
    if t_nw == t_nw and t_nw < 2.0:
        notes.append(f"⚠️ alpha 的 Newey-West t={t_nw:+.2f} < 2，统计上不显著，谨慎对待")

    tables = []
    if len(trades):
        tshow = trades.copy()
        tshow["date"] = tshow["date"].astype(str)
        keep = [c for c in ("date", "symbol", "name", "side", "fill_shares",
                            "fill_price", "notional", "cost", "status", "reason")
                if c in tshow.columns]
        tables.append(table("逐笔成交明细", tshow[keep], max_rows=200))

    return report("performance", "绩效分析",
                  summary=(f"Sharpe={sharpe:+.2f} · α年化="
                           + (f"{alpha_ann:+.1%}" if alpha_ann == alpha_ann else "—")
                           + (f" (NW-t={t_nw:+.2f})" if t_nw == t_nw else "")
                           + f" · 最大回撤{max_dd:+.1f}%"),
                  kpis=kpis, figures=figures, notes=notes, tables=tables)
