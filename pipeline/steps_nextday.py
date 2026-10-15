"""次日交易模式的流水线步骤（多模型集成 + 布林/RSI 择时逐笔订单）。

顺序：data_import → train_predict → generate_orders

复用 src/live.py 的新实盘路径（ensemble_predict_latest + make_orders_from_state），
与每日重放回测**同一套决策代码**（trading.decide_orders），保证回测=实盘同分布。
产出明日可执行、精确到 (代码,方向,股数,挂单价,触发原因) 的逐笔订单。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from data import load_factor_panel, list_factor_columns
from live import ensemble_predict_latest, make_orders_from_state, live_panel_start_date
from trading import parse_initial_holdings, estimate_order_cost

from . import context as ctxmod
from .context import kpi, table, fig_line, fig_bar, fig_hist, report


# ---------------------------------------------------------------------------
def data_import(ctx: ctxmod.StepContext) -> dict:
    """加载含【最新无标签日】+ OHLC + 布林带 meta 的实盘面板（窗口受限，减少 I/O）。"""
    ctx.report_progress("数据加载中", 0, 2, "load_factor_panel (keep_unlabeled)")
    train_window = ctx.params.get("train_window")
    # 门控启用时 buffer 需额外覆盖评估窗口（双回放要有 eval_days 个决策日）
    gate_on = C.USE_MODEL_GATE and bool(ctx.params.get("use_model_gate", True))
    eval_days = int(ctx.params.get("promote_eval_days") or C.PROMOTE_EVAL_DAYS)
    buffer = max(80, eval_days + 10) if gate_on else 80
    start_date = live_panel_start_date(train_window=train_window,
                                       buffer_trading_days=buffer)
    panel = load_factor_panel(keep_unlabeled=True, keep_trade_cols=True,
                              start_date=start_date, verbose=True)
    factors = list_factor_columns(panel)
    ctx.shared["panel"] = panel
    ctx.shared["factors"] = factors
    ctx.report_progress("数据画像中", 1, 2, f"{panel['symbol'].nunique():,} 只")

    latest = panel["date"].max()
    n_label = int(panel[C.TARGET].notna().sum())
    today_n = int((panel["date"] == latest).sum())
    per_date = panel.groupby("date")["symbol"].nunique()

    figures = [fig_line("每日横截面股票数", per_date.index.astype(str).tolist(),
                        [{"name": "股票数", "y": per_date.to_numpy()}],
                        yaxis="只", x_is_date=True)]
    has_bb = "bb_lower_20" in panel.columns and "bb_upper_20" in panel.columns
    kpis = [
        kpi("最新交易日", f"{pd.Timestamp(latest).date()}", "次日执行基准"),
        kpi("当日股票数", f"{today_n:,}", "可预测 universe"),
        kpi("有标签行", f"{n_label:,}", "训练样本"),
        kpi("因子数", f"{len(factors)}", "命名因子"),
        kpi("择时列", f"{'BB✓' if has_bb else 'BB✗'} {'RSI✓' if 'rsi_14' in panel.columns else 'RSI✗'}",
            "布林/RSI 择时所需"),
    ]
    notes = []
    if not has_bb:
        notes.append("⚠️ 面板缺布林带列，将仅按 ML 排序选股（无指标择时）")
    return report("data_import", "数据导入（实盘）",
                  summary=f"最新日 {pd.Timestamp(latest).date()} · {today_n:,} 只可预测",
                  kpis=kpis, figures=figures, notes=notes)


# ---------------------------------------------------------------------------
def _ensemble_params(ctx: ctxmod.StepContext) -> dict:
    p = ctx.params
    out = {}
    for key in ("n_seeds", "incr_trees", "rebuild_days", "train_window", "ic_window"):
        if p.get(key) is not None:
            out[key] = int(p[key])
    return out


def _decision_params(ctx: ctxmod.StepContext) -> dict:
    """下单/撮合参数（generate_orders 与模型门控双回放共用，保证两边一致）。"""
    p = ctx.params
    dp = {}
    for key, pk in (("topk", "topk"), ("max_positions", "max_positions"),
                    ("max_weight", "max_weight"), ("min_notional", "min_notional"),
                    ("dynamic_topk_ic_full", "dynamic_topk_ic_full"),
                    ("dynamic_topk_ic_zero", "dynamic_topk_ic_zero"),
                    ("dynamic_topk_low", "dynamic_topk_low"),
                    ("dynamic_topk_high", "dynamic_topk_high"),
                    ("max_portfolio_beta", "max_portfolio_beta")):
        if p.get(pk) is not None:
            dp[key] = float(p[pk]) if key != "max_positions" else int(p[pk])
    if p.get("use_bb_timing") is not None:
        dp["use_bb_timing"] = bool(p["use_bb_timing"])
    if p.get("use_dynamic_topk") is not None:
        dp["use_dynamic_topk"] = bool(p["use_dynamic_topk"])
    if p.get("use_beta_constraint") is not None:
        dp["use_beta_constraint"] = bool(p["use_beta_constraint"])
    if p.get("order_price_mode"):
        dp["order_price_mode"] = str(p["order_price_mode"])
    return dp


def train_predict(ctx: ctxmod.StepContext) -> dict:
    """多模型集成训练（全部成熟历史）+ 预测最新日，滚动 IC 定权。"""
    panel = ctx.shared["panel"]
    factors = ctx.shared["factors"]
    p = dict(_ensemble_params(ctx))
    # 把 progress 回调注入 live.ensemble_predict_latest
    def _cb(phase, current, total, **extras):
        ctx.report_progress(phase, current, total, **extras)
    p["progress_cb"] = _cb
    ctx.report_progress("训练+预测启动", 0, 3, f"{len(factors)} 因子")

    # 自适应重训触发：默认不重训，指标（IC 退化/收益落后/超龄）任一触发才训练。
    # 注意：webapp 会话的 risk_state/pred_hist 是会话级 artifact，无跨会话积累，
    # IC/收益触发通常因无数据跳过、年龄触发可用；完整触发以 CLI nightly 为准。
    gate = None
    fire, reasons = True, []
    adaptive = (C.USE_MODEL_GATE and C.USE_ADAPTIVE_RETRAIN
                and bool(ctx.params.get("use_model_gate", True)))
    if adaptive:
        import model_store
        from live_state import pred_hist_path_for, load_pred_hist
        from live_tracking import matured_ic_stats, nav_stats
        from retrain_trigger import should_retrain
        risk_path = ctx.artifact_path("live_risk_state.json")
        champ = model_store.load_champion()
        fire, reasons = should_retrain(
            champ[1] if champ else None,
            matured_ic_stats(panel, load_pred_hist(pred_hist_path_for(risk_path))),
            nav_stats(risk_path, panel), panel)
    if not fire:
        from live import champion_predict_latest
        ctx.report_progress("冻结 champion 预测", 1, 3, "未触发重训")
        today, info = champion_predict_latest(panel, progress_cb=_cb)
        ctx.shared["today"] = today
        ctx.shared["pred_info"] = info
        ctx.shared["gate"] = {"action": "frozen",
                              "reason": "未触发重训：" + ("；".join(reasons) if reasons else "各指标正常"),
                              "champion_id": info.get("champion_id"),
                              "candidate_id": info.get("champion_id"),
                              "use_candidate_for_live": True, "evaluation": None}
        gate = ctx.shared["gate"]
    else:
        today, info = ensemble_predict_latest(panel, factors, params=p)

        # 模型门控：候选配方指纹 vs champion，不同则 staging 双回放对比，
        # Sharpe 更高且回撤不劣才晋升；被拒时用冻结 champion 出信号。
        if (C.USE_MODEL_GATE and bool(ctx.params.get("use_model_gate", True))
                and info.get("ensemble") is not None):
            from live import champion_predict_latest
            from model_gate import run_gate
            eval_days = int(ctx.params.get("promote_eval_days") or C.PROMOTE_EVAL_DAYS)
            gate = run_gate(panel, factors, info["ensemble"], _ensemble_params(ctx),
                            decision_params=_decision_params(ctx),
                            eval_days=eval_days, progress=False, progress_cb=_cb)
            ctx.shared["gate"] = gate
            if not gate["use_candidate_for_live"]:
                ctx.report_progress("冻结 champion 预测", 2, 3,
                                    f"候选被拒，champion={gate['champion_id']}")
                today, info = champion_predict_latest(panel, progress_cb=_cb)

        ctx.shared["today"] = today
        ctx.shared["pred_info"] = info

    score = today["score"].to_numpy()
    weights = info["weights"]; ics = info["ic_means"]
    figures = [
        fig_hist("融合排序分分布（最新日全 universe）", score, nbins=60, xaxis="score"),
        fig_bar("多模型当前 IC 权重",
                list(weights.keys()), list(weights.values()), yaxis="权重"),
    ]
    # 各模型 IC（诊断）
    ic_items = [(m, v) for m, v in ics.items() if v == v]
    if ic_items:
        figures.append(fig_bar("各模型近期滚动 IC",
                               [m for m, _ in ic_items], [v for _, v in ic_items],
                               yaxis="rank-IC", diverging=True))

    top = today.head(15)[["symbol", "score", "rank_pct"]].copy()
    if "name" in today.columns:
        top.insert(1, "name", today.head(15)["name"].to_numpy())
    top["score"] = top["score"].round(4)
    top["rank_pct"] = (top["rank_pct"] * 100).round(1)
    wtxt = " · ".join(f"{m}={w:.2f}" for m, w in weights.items())
    kpis = [
        kpi("预测股票数", f"{len(today):,}", "最新日全 universe"),
        kpi("最看好", f"{today.iloc[0]['symbol']}", f"score={today.iloc[0]['score']:+.4f}"),
        kpi("模型数", f"{len(weights)}", wtxt),
    ]
    notes = []
    if gate is not None:
        _GATE_LABEL = {"bootstrap": "首个 champion", "refreshed": "同配方刷新",
                       "promoted": "PROMOTED 已晋升", "rejected": "REJECTED 保留冠军",
                       "frozen": "冻结（未触发重训）"}
        kpis.append(kpi("模型门控", _GATE_LABEL.get(gate["action"], gate["action"]),
                        f"champion={gate['champion_id']}"))
        notes.append(f"模型门控：{gate['reason']}")
        ev = gate.get("evaluation")
        if ev:
            c_, h_ = ev["candidate"], ev["champion"]
            notes.append(
                f"门控评估（{ev['window'][0]} ~ {ev['window'][1]}）："
                f"候选 Sharpe {c_['sharpe']:+.2f}/回撤 {c_['max_drawdown']:+.1f}% "
                f"vs 冠军 Sharpe {h_['sharpe']:+.2f}/回撤 {h_['max_drawdown']:+.1f}%")
    return report("train_predict", "训练 + 预测（多模型集成）",
                  summary=f"已预测 {len(today):,} 只 · 权重 {wtxt}",
                  kpis=kpis, figures=figures, notes=notes,
                  tables=[table("融合排序 Top15（最看好）", top, max_rows=15)])


# ---------------------------------------------------------------------------
def generate_orders(ctx: ctxmod.StepContext) -> dict:
    """共享 decide_orders 生成明日逐笔订单（ML 选池 + 布林/RSI 择时）。订单 csv 落盘。"""
    today = ctx.shared["today"]
    p = ctx.params
    capital = float(p.get("capital", C.INITIAL_CAPITAL))
    current = parse_initial_holdings(p.get("initial_holdings", ""))

    dp = _decision_params(ctx)

    score = today["score"].to_numpy()
    ctx.report_progress("下单决策中", 0, 2, f"capital={capital:,.0f}")
    orders = make_orders_from_state(
        today, score, current, capital, params=dp,
        price_col="adj_close",
        risk_state_path=ctx.artifact_path("live_risk_state.json"),
        panel=ctx.shared.get("panel"),
    )
    orders.to_csv(ctx.artifact_path("orders.csv"), index=False)
    ctx.report_progress("订单落盘中", 1, 2, f"{len(orders)} 笔")

    cost = estimate_order_cost(orders)
    n_buy = int((orders["side"] == "buy").sum()) if len(orders) else 0
    n_sell = int((orders["side"] == "sell").sum()) if len(orders) else 0
    n_block = int(orders["blocked"].sum()) if len(orders) and "blocked" in orders else 0

    figures = []
    if len(orders):
        notion = (orders["shares"].to_numpy() * orders["limit_price"].to_numpy())
        figures.append(fig_bar(
            "订单金额（前 20）",
            orders.head(20)["symbol"].tolist(), notion[:20],
            yaxis="预计金额 元"))
        # 触发原因分布
        rc = orders["reason"].value_counts()
        figures.append(fig_bar("订单触发原因分布", rc.index.tolist(),
                               rc.to_numpy(), yaxis="笔数"))

    show = orders.copy()
    if len(show):
        show["rank_pct"] = (show["rank_pct"] * 100).round(1)
    kpis = [
        kpi("买入", f"{n_buy}", "只"),
        kpi("卖出", f"{n_sell}", "只"),
        kpi("涨跌停拦截", f"{n_block}", "不可成交"),
        kpi("预计成交额", f"¥{cost['buy_notional']+cost['sell_notional']:,.0f}", "买+卖"),
        kpi("预计成本", f"¥{cost['total']:,.0f}",
            f"佣金 ¥{cost['commission']:,.0f}+印花 ¥{cost['stamp_duty']:,.0f}"),
    ]
    notes = ["提醒：信号基于当日收盘特征，请于【次日】按挂单价执行；ML 仅用于选池排序。"]
    return report("generate_orders", "生成次日订单",
                  summary=f"买 {n_buy} · 卖 {n_sell} · 成本 ¥{cost['total']:,.0f}",
                  kpis=kpis, figures=figures, notes=notes,
                  tables=[table("逐笔交易指令（卖在前买在后）", show, max_rows=80)])
