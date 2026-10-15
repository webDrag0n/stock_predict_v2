"""每晚运行一次：基于截至今天的数据，生成【明日可执行的买卖订单】。

用法
----
  # 收盘后运行（首次空仓 → 全是买单）
  .venv/bin/python predict_tomorrow.py --capital 1000000

  # 确认今日已按订单成交后，把目标持仓写回 holdings.csv
  .venv/bin/python predict_tomorrow.py --commit

  # 自定义参数
  .venv/bin/python predict_tomorrow.py --topk 0.2 --min-notional 5000 \
      --max-positions 20 --use-bb-timing

输出
----
  outputs/orders_YYYYMMDD.csv          明日订单（卖在前、买在后，含 assumed_fill 说明列）
  outputs/holdings.csv                 当前持仓状态（--commit 时更新）
  outputs/account.csv                  账户现金（cash 列，人工维护；提供后启用
                                       vol/回撤两层风控的真实净值入账）
  outputs/live_risk_state.json         风控状态（波动/回撤/IC 历史）
  outputs/live_risk_state_pred_hist.parquet  持久化预测历史（IC 降仓/定权的真 OOS 来源）

重要语义
--------
- 信号用【当日收盘后】才完整的特征算出 → 当晚生成、【次日开盘执行】，无未来信息。
- prediction 仅用于排序选股，不代表预期涨幅（系统点估计弱，优势在横截面排序）。
- 本脚本与回测每日重放共用 src/trading.py::decide_orders 与 src/risk.py 的
  五层风控叠乘链，确保实盘/回测同分布、同数学。
- 默认用原始 close 估值（金钱计算必须用真实市价；adj_close 绝对水平随加载
  窗口漂移，只用于因子）。
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

# pandas 碎片化告警（PerformanceWarning）在因子构建期属噪声。原由各库模块在
# import 期全局屏蔽，已收口为入口脚本显式设置（库不应动全局 warnings 状态）。
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from live import ensemble_predict_latest, make_orders_from_state, live_panel_start_date
from trading import parse_initial_holdings, estimate_order_cost


def _load_holdings_table(path: str) -> pd.DataFrame:
    """读取完整持仓表（symbol/shares/entry_date/ref_close），不存在 → 空表。"""
    cols = ["symbol", "shares", "entry_date", "ref_close"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path, dtype={"symbol": str})
        # 兼容旧版 holdings.csv（中文列）
        if "代码" in df.columns and "股数" in df.columns:
            df = df.rename(columns={"代码": "symbol", "股数": "shares"})
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols]
    except Exception:
        return pd.DataFrame(columns=cols)


def _load_holdings(path: str) -> dict[str, int]:
    """读取持仓 csv → {代码: 股数}。文件不存在视为空仓。"""
    df = _load_holdings_table(path)
    if not len(df):
        return {}
    return dict(zip(df["symbol"], df["shares"].astype(int)))


def _load_cash(path: str) -> float | None:
    """读取 account.csv 的 cash 列（取最后一行非空值）。无文件/无列 → None。"""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "cash" not in df.columns:
            return None
        v = pd.to_numeric(df["cash"], errors="coerce").dropna()
        return float(v.iloc[-1]) if len(v) else None
    except Exception:
        return None


def _save_holdings(path: str, orders: pd.DataFrame, current: dict[str, int],
                   latest_date) -> None:
    """按订单差额把目标持仓写回 holdings.csv（--commit 假设非拦截单全部成交）。

    entry_date 语义：只有【买入成交】（新建/加仓）的持仓刷新为当日；卖出减仓
    与未交易的持仓保留原 entry_date（旧版每次 --commit 全部重置，会丢失真实
    建仓日，已修复）。
    """
    old = _load_holdings_table(path)
    old_entry, old_ref = {}, {}
    for r in old.itertuples(index=False):
        old_entry[r.symbol] = None if pd.isna(r.entry_date) else str(r.entry_date)
        old_ref[r.symbol] = None if pd.isna(r.ref_close) else float(r.ref_close)

    target = dict(current)
    bought: set[str] = set()
    for o in orders.itertuples(index=False):
        if getattr(o, "blocked", False):
            continue
        s = o.symbol
        if o.side == "buy":
            target[s] = target.get(s, 0) + int(o.shares)
            bought.add(s)
        elif o.side == "sell":
            target[s] = max(0, target.get(s, 0) - int(o.shares))

    dstr = str(pd.Timestamp(latest_date).date())
    rows = []
    for s, sh in target.items():
        if sh <= 0:
            continue
        ref = old_ref.get(s)
        sub = orders[orders["symbol"] == s]
        if len(sub) and "ref_close" in orders.columns:
            ref = float(sub.iloc[0]["ref_close"])
        rows.append({
            "symbol": s,
            "shares": int(sh),
            # 未买入的持仓保留原建仓日；查不到（新文件外的来源）才用当日
            "entry_date": dstr if s in bought else (old_entry.get(s) or dstr),
            "ref_close": ref,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _print_gate_result(gate: dict) -> None:
    """门控结论打印（stdout 可见的 PROMOTED/REJECTED + 双方指标对比）。"""
    action = gate["action"]
    if action == "refreshed":
        print(f"[模型门控] 配方与 champion 相同，工件已刷新（{gate['champion_id']}）")
        return
    if action == "bootstrap":
        print(f"[模型门控] 首个 champion 建立：{gate['champion_id']}"
              "（bootstrap，未经对比）")
        return
    ev = gate.get("evaluation") or {}
    cand, champ = ev.get("candidate", {}), ev.get("champion", {})
    win = ev.get("window", ["?", "?"])
    nan = float("nan")
    print(f"[模型门控] 评估窗口 {win[0]} ~ {win[1]}"
          f"（{ev.get('eval_days', '?')} 个交易日，完整每日重放 ×2）")
    print(f"  候选 {gate['candidate_id']}: "
          f"Sharpe {cand.get('sharpe', nan):+.2f} · "
          f"回撤 {cand.get('max_drawdown', nan):+.1f}% · "
          f"α年化 {cand.get('alpha_ann', nan):+.1%} · "
          f"收益 {cand.get('total_return', nan):+.1f}%")
    print(f"  冠军 {ev.get('champion_id', '?')}: "
          f"Sharpe {champ.get('sharpe', nan):+.2f} · "
          f"回撤 {champ.get('max_drawdown', nan):+.1f}% · "
          f"α年化 {champ.get('alpha_ann', nan):+.1%} · "
          f"收益 {champ.get('total_return', nan):+.1f}%")
    verdict = ("PROMOTED → 已晋升为新 champion"
               if action == "promoted" else "REJECTED → 保留现有 champion")
    print(f"[模型门控] {verdict}：{gate['reason']}")
    if action == "rejected":
        print(f"  候选已留档 models/staging/{gate['candidate_id']}/"
              "（含 evaluation.json），可用 --promote 人工晋升")


def _ensemble_params(args) -> dict:
    out = {}
    for key in ("n_seeds", "incr_trees", "rebuild_days", "train_window", "ic_window"):
        v = getattr(args, key, None)
        if v is not None:
            out[key] = int(v)
    return out


def _decision_params(args) -> dict:
    p = {}
    for key in ("topk", "max_positions", "max_weight", "min_notional"):
        v = getattr(args, key, None)
        if v is not None:
            p[key] = float(v) if key != "max_positions" else int(v)
    if args.use_bb_timing is not None:
        p["use_bb_timing"] = bool(args.use_bb_timing)
    if args.order_price_mode:
        p["order_price_mode"] = str(args.order_price_mode)
    return p


def build_parser() -> argparse.ArgumentParser:
    """命令行参数解析器（独立成函数便于单测直接 parse_args）。"""
    ap = argparse.ArgumentParser(description="生成明日交易订单（每晚运行）")
    ap.add_argument("--capital", type=float, default=C.INITIAL_CAPITAL,
                    help=f"组合总资金兜底（默认 {C.INITIAL_CAPITAL:.0f}；"
                         f"提供 account.csv 的 cash 后以真实现金为准）")
    ap.add_argument("--holdings", default=os.path.join(C.OUTPUT_DIR, "holdings.csv"),
                    help="当前持仓 csv 路径")
    ap.add_argument("--account", default=os.path.join(C.OUTPUT_DIR, "account.csv"),
                    help="账户现金 csv 路径（cash 列；缺失则跳过 vol/回撤风控入账）")
    ap.add_argument("--risk-state", default=os.path.join(C.OUTPUT_DIR, "live_risk_state.json"),
                    help="风控状态 JSON 路径")
    # 选股 / 择时
    ap.add_argument("--topk", type=float, default=C.TOPK_RATIO,
                    help=f"候选池比例 TOPK（默认 {C.TOPK_RATIO}）")
    ap.add_argument("--max-positions", type=int, default=C.MAX_POSITIONS,
                    help=f"最大持仓数（默认 {C.MAX_POSITIONS}）")
    ap.add_argument("--max-weight", type=float, default=C.MAX_WEIGHT_PER_NAME,
                    help=f"单股权重上限（默认 {C.MAX_WEIGHT_PER_NAME}）")
    ap.add_argument("--use-bb-timing", action="store_true",
                    default=C.USE_BB_TIMING,
                    help="启用布林/RSI 择时")
    ap.add_argument("--order-price-mode", default=C.ORDER_PRICE_MODE,
                    choices=["limit_bb", "next_open"],
                    help="挂单价模型（默认 next_open）")
    ap.add_argument("--min-notional", type=float, default=C.DAILY_MIN_NOTIONAL,
                    help=f"最小订单金额（默认 {C.DAILY_MIN_NOTIONAL}）")
    ap.add_argument("--price-col", default="close",
                    help="用于估值的价列（默认 close=真实市价，无则退回到 close）")
    # 集成模型参数
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="模型 seed 数（默认 1）")
    ap.add_argument("--train-window", type=int, default=C.TRAIN_WINDOW,
                    help=f"训练窗口（默认 {C.TRAIN_WINDOW}）")
    ap.add_argument("--rebuild-days", type=int, default=C.REBUILD_DAYS,
                    help=f"全量重建周期（默认 {C.REBUILD_DAYS}）")
    ap.add_argument("--incr-trees", type=int, default=C.INCR_TREES,
                    help=f"增量续训树数（默认 {C.INCR_TREES}）")
    ap.add_argument("--ic-window", type=int, default=C.IC_WINDOW,
                    help=f"滚动 IC 定权窗口（默认 {C.IC_WINDOW}）")
    # 模型门控（champion/staging 晋升机制）
    ap.add_argument("--no-gate", action="store_true",
                    help="跳过模型门控（旧行为：今晚训练的模型直接用于订单）")
    ap.add_argument("--eval-days", type=int, default=C.PROMOTE_EVAL_DAYS,
                    help=f"门控评估窗口交易日数（默认 {C.PROMOTE_EVAL_DAYS}）")
    ap.add_argument("--promote", metavar="STAGING_ID", default=None,
                    help="人工晋升 models/staging/<STAGING_ID> 为 champion 后退出")
    # 自适应重训与跟踪
    ap.add_argument("--force-retrain", action="store_true",
                    help="无视触发指标，强制今晚重训并走门控")
    ap.add_argument("--tracking-report", action="store_true",
                    help="只打印回测 vs 实盘偏差跟踪报告后退出（不训练不出单）")
    ap.add_argument("--commit", action="store_true",
                    help="确认成交后把目标持仓写回 holdings.csv")
    return ap


# 模块级解析器：供测试/外部脚本直接 parse_args（历史接口 pt.ap）
ap = build_parser()


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    # 人工晋升：把留档候选直接设为 champion（不训练、不出订单）
    if args.promote:
        from model_store import promote
        m = promote(args.promote)
        print(f"已人工晋升 models/staging/{args.promote} 为 champion "
              f"（model_id={m['model_id']}，promotion={m.get('promotion')}）")
        return

    # dynamic_topk 只在回测实现；实盘显式报错而非静默忽略（回测≠实盘防线）
    if getattr(C, "USE_DYNAMIC_TOPK", False):
        raise SystemExit(
            "USE_DYNAMIC_TOPK=1 目前仅回测支持；实盘 predict_tomorrow 未实现，"
            "请关闭后重跑（避免静默忽略造成回测≠实盘）。")

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    # 1) 加载含【今天】、含 OHLC/布林带 meta 的面板（实盘模式，用最小窗口减少 I/O）
    #    门控启用时 buffer 需额外覆盖评估窗口（双回放要有 eval_days 个决策日）。
    print("加载面板（保留最新无标签日 + 交易列）...")
    gate_on = C.USE_MODEL_GATE and not args.no_gate
    buffer = max(80, args.eval_days + 10) if gate_on else 80
    start_date = live_panel_start_date(train_window=args.train_window,
                                       buffer_trading_days=buffer)
    panel = load_factor_panel(keep_unlabeled=True, keep_trade_cols=True,
                              start_date=start_date)
    factors = list_factor_columns(panel)
    n_label = int(panel[C.TARGET].notna().sum())
    latest = panel["date"].max()
    print(f"面板 {len(panel):,} 行 | {panel['symbol'].nunique()} 只 | "
          f"{len(factors)} 因子 | 有标签 {n_label:,} 行 | "
          f"最新日 {pd.Timestamp(latest).date()} | FORWARD_WINDOW={C.FORWARD_WINDOW}")

    # 2) 跟踪指标 + 重训触发判定（自适应模式：默认不重训，指标触发才重训）
    #    IC 定权用【持久化的真 OOS 预测历史】（pred_hist parquet），而非旧实现的
    #    训练窗内补 observe（样本内偏差 → 权重系统性偏 GBM）。
    from live_state import pred_hist_path_for, load_pred_hist, append_pred_hist
    import model_store
    from live import champion_predict_latest
    from live_tracking import (matured_ic_stats, nav_stats, compute_report,
                               write_report, print_summary,
                               append_daily_snapshot)
    from retrain_trigger import should_retrain, model_age_days

    pred_hist_path = pred_hist_path_for(args.risk_state)
    pred_hist = load_pred_hist(pred_hist_path)

    # --tracking-report：只打印回测 vs 实盘偏差报告后退出（不训练不出单）
    if args.tracking_report:
        import json as _json
        report = compute_report(model_store.load_champion_manifest(),
                                matured_ic_stats(panel, pred_hist),
                                nav_stats(args.risk_state, panel), panel=panel)
        write_report(report)
        print_summary(report)
        print(_json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return

    adaptive = gate_on and C.USE_ADAPTIVE_RETRAIN
    ic_stats = matured_ic_stats(panel, pred_hist)
    fire, reasons, gate = True, [], None
    if adaptive:
        champ_manifest = (lambda c: c[1] if c else None)(
            model_store.load_champion())
        fire, reasons = should_retrain(
            champ_manifest, ic_stats, nav_stats(args.risk_state, panel), panel,
            force=args.force_retrain)
        if fire:
            print("[重训触发] " + "；".join(reasons))
        else:
            age = model_age_days(champ_manifest, panel)
            print(f"[重训触发] 未触发：今晚用冻结 champion 预测"
                  f"（{champ_manifest.get('model_id')}，年龄 {age} 个交易日）")
    elif gate_on:
        reasons = ["自适应重训已关闭（USE_ADAPTIVE_RETRAIN=0），每晚重训"]

    # 3) 训练（触发时）或冻结加载（未触发时）+ 预测最新日
    action = "frozen"
    if not fire:
        today, info = champion_predict_latest(panel, pred_hist=pred_hist)
        n_today = len(today)
        print(f"冻结 champion 已预测 {pd.Timestamp(latest).date()} 的 "
              f"{n_today} 只股票（champion={info.get('champion_id')}）")
    else:
        print("训练并预测（多模型增量集成 + 滚动 IC 定权）...")
        ens_params = _ensemble_params(args)
        ens_params["pred_hist"] = pred_hist
        today, info = ensemble_predict_latest(panel, factors, params=ens_params)
        n_today = len(today)
        print(f"已预测 {pd.Timestamp(latest).date()} 的 {n_today} 只股票")
        action = "trained"

        # 3a) 模型门控：候选配方指纹 vs champion。相同 → 原位刷新工件；不同 →
        #     staging 落盘 + 双方最近 eval_days 天完整重放对比，Sharpe 更高且
        #     回撤不劣才晋升；被拒时今晚信号用冻结 champion（订单永远出自 champion）。
        if gate_on and info.get("ensemble") is not None:
            from model_gate import run_gate
            gate = run_gate(panel, factors, info["ensemble"],
                            _ensemble_params(args),
                            decision_params=_decision_params(args),
                            eval_days=args.eval_days)
            _print_gate_result(gate)
            action = gate["action"]
            if not gate["use_candidate_for_live"]:
                print(f"候选被拒，改用冻结 champion 出信号"
                      f"（champion={gate['champion_id']}）...")
                today, info = champion_predict_latest(panel, pred_hist=pred_hist)
                n_today = len(today)
                action = "rejected"
    wtxt = " · ".join(f"{m}={w:.2f}" for m, w in info["weights"].items())
    print(f"模型权重: {wtxt}")

    # 2b) 落盘当日预测分（供 fw 天后成熟打分入 IC 队列 + OOS IC 定权积累）
    append_pred_hist(pred_hist_path, latest, today)
    print(f"预测历史已累积 -> {pred_hist_path}")

    # 3) 读取当前持仓与账户现金，生成次日订单
    current = parse_initial_holdings(
        "\n".join(f"{s}:{sh}" for s, sh in _load_holdings(args.holdings).items())
    )
    print(f"当前持仓：{len(current)} 只" + ("（空仓）" if not current else ""))
    cash = _load_cash(args.account)
    if cash is not None:
        print(f"账户现金：¥{cash:,.0f}（{args.account}）")

    dec_params = _decision_params(args)
    orders = make_orders_from_state(
        today, today["score"].to_numpy(), current, args.capital,
        params=dec_params,
        price_col=args.price_col,
        risk_state_path=args.risk_state,
        panel=panel,
        cash=cash,
    )

    # 4) 输出订单（assumed_fill 列说明 --commit 的成交假设）
    if len(orders):
        orders = orders.copy()
        blk = (orders["blocked"].astype(bool)
               if "blocked" in orders.columns
               else pd.Series(False, index=orders.index))
        orders["assumed_fill"] = np.where(
            blk, "否：决策层拦截不执行",
            "是：--commit 假设全部成交，请按实际成交手动核对")
    tag = pd.Timestamp(latest).strftime("%Y%m%d")
    out_csv = os.path.join(C.OUTPUT_DIR, f"orders_{tag}.csv")
    orders.to_csv(out_csv, index=False)

    cost = estimate_order_cost(orders)
    n_buy = int((orders["side"] == "buy").sum()) if len(orders) else 0
    n_sell = int((orders["side"] == "sell").sum()) if len(orders) else 0
    n_block = int(orders["blocked"].sum()) if len(orders) and "blocked" in orders else 0

    print("\n" + "=" * 60)
    print(f"明日订单摘要（执行日 = {pd.Timestamp(latest).date()} 的下一交易日）")
    print("=" * 60)
    print(f"  买入 {n_buy} 只 | 卖出 {n_sell} 只"
          + (f" | 涨跌停拦截 {n_block} 只" if n_block else ""))
    print(f"  预计买入额 ¥{cost['buy_notional']:,.0f} | 卖出额 ¥{cost['sell_notional']:,.0f}")
    print(f"  预计成本 ¥{cost['total']:,.0f}"
          f"（佣金 ¥{cost['commission']:,.0f} + 印花税 ¥{cost['stamp_duty']:,.0f}）")
    if len(orders):
        show = orders[orders["side"].isin(["buy", "sell"])].head(15)
        if len(show):
            print("\n  前 15 条交易指令：")
            print(show.to_string(index=False))
    print(f"\n订单明细 -> {out_csv}")

    # 5) 可选：确认成交后更新持仓
    if args.commit:
        _save_holdings(args.holdings, orders, current, latest)
        print(f"已更新持仓 -> {args.holdings}")
        print("注意：--commit 【假设非拦截订单全部成交】，请按实际成交手动核对 "
              f"holdings.csv；并请同步维护 {args.account} 的 cash 列"
              "（真实净值是 vol/回撤两层风控的输入）。")
    else:
        print("（未写持仓。确认按单成交后加 --commit 更新 holdings.csv）")

    # 6) 回测 vs 实盘跟踪：快照落盘 + 偏差报告 + 告警（每晚都记，与是否触发无关）
    nav = nav_stats(args.risk_state, panel)   # 订单步骤已入账今日净值
    report = compute_report(model_store.load_champion_manifest(),
                            ic_stats, nav, panel=panel)
    write_report(report)
    append_daily_snapshot(latest, report["champion"].get("model_id") or "-",
                          action, n_today, ic_stats, nav)
    print_summary(report)

    print("\n提醒：信号基于当日收盘特征，请于【次日】执行；prediction 仅用于排序选股，非涨幅预测。")


if __name__ == "__main__":
    main()
