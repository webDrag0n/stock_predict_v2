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
  outputs/orders_YYYYMMDD.csv   明日订单（卖在前、买在后）
  outputs/holdings.csv          当前持仓状态（--commit 时更新）
  outputs/live_risk_state.json  风控状态（波动/回撤/IC 历史）

重要语义
--------
- 信号用【当日收盘后】才完整的特征算出 → 当晚生成、【次日开盘执行】，无未来信息。
- prediction 仅用于排序选股，不代表预期涨幅（系统点估计弱，优势在横截面排序）。
- 本脚本与回测每日重放共用 src/trading.py::decide_orders，确保实盘/回测同分布。
- 默认用 adj_close 估值；若面板无 adj_close 则退回到 close。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from live import ensemble_predict_latest, make_orders_from_state, live_panel_start_date
from trading import parse_initial_holdings, estimate_order_cost


def _load_holdings(path: str) -> dict[str, int]:
    """读取持仓 csv → {代码: 股数}。文件不存在视为空仓。"""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, dtype={"symbol": str})
        # 兼容旧版 holdings.csv（中文列）
        if "代码" in df.columns and "股数" in df.columns:
            return dict(zip(df["代码"], df["股数"].astype(int)))
        return dict(zip(df["symbol"], df["shares"].astype(int)))
    except Exception:
        return {}


def _save_holdings(path: str, orders: pd.DataFrame, current: dict[str, int],
                   latest_date) -> None:
    """按订单差额把目标持仓写回 holdings.csv。"""
    target = dict(current)
    for o in orders.itertuples(index=False):
        if getattr(o, "blocked", False):
            continue
        s = o.symbol
        if o.side == "buy":
            target[s] = target.get(s, 0) + int(o.shares)
        elif o.side == "sell":
            target[s] = max(0, target.get(s, 0) - int(o.shares))
    rows = []
    for s, sh in target.items():
        if sh <= 0:
            continue
        # 尝试从 orders 找到该 symbol 的 ref_close，否则留空
        ref = None
        sub = orders[orders["symbol"] == s]
        if len(sub):
            ref = float(sub.iloc[0]["ref_close"])
        rows.append({
            "symbol": s,
            "shares": int(sh),
            "entry_date": str(pd.Timestamp(latest_date).date()),
            "ref_close": ref,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


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


def main() -> None:
    ap = argparse.ArgumentParser(description="生成明日交易订单（每晚运行）")
    ap.add_argument("--capital", type=float, default=C.INITIAL_CAPITAL,
                    help=f"组合总资金（默认 {C.INITIAL_CAPITAL:.0f}）")
    ap.add_argument("--holdings", default=os.path.join(C.OUTPUT_DIR, "holdings.csv"),
                    help="当前持仓 csv 路径")
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
    ap.add_argument("--price-col", default="adj_close",
                    help="用于估值的价列（默认 adj_close，无则退回到 close）")
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
    ap.add_argument("--commit", action="store_true",
                    help="确认成交后把目标持仓写回 holdings.csv")
    args = ap.parse_args()

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    # 1) 加载含【今天】、含 OHLC/布林带 meta 的面板（实盘模式，用最小窗口减少 I/O）
    print("加载面板（保留最新无标签日 + 交易列）...")
    start_date = live_panel_start_date(train_window=args.train_window)
    panel = load_factor_panel(keep_unlabeled=True, keep_trade_cols=True,
                              start_date=start_date)
    factors = list_factor_columns(panel)
    n_label = int(panel[C.TARGET].notna().sum())
    latest = panel["date"].max()
    print(f"面板 {len(panel):,} 行 | {panel['symbol'].nunique()} 只 | "
          f"{len(factors)} 因子 | 有标签 {n_label:,} 行 | "
          f"最新日 {pd.Timestamp(latest).date()} | FORWARD_WINDOW={C.FORWARD_WINDOW}")

    # 2) 多模型集成训练 + 预测最新日（与回测同分布）
    print("训练并预测（多模型增量集成 + 滚动 IC 定权）...")
    ens_params = _ensemble_params(args)
    today, info = ensemble_predict_latest(panel, factors, params=ens_params)
    n_today = len(today)
    print(f"已预测 {pd.Timestamp(latest).date()} 的 {n_today} 只股票")
    wtxt = " · ".join(f"{m}={w:.2f}" for m, w in info["weights"].items())
    print(f"模型权重: {wtxt}")

    # 3) 读取当前持仓，生成次日订单
    current = parse_initial_holdings(
        "\n".join(f"{s}:{sh}" for s, sh in _load_holdings(args.holdings).items())
    )
    print(f"当前持仓：{len(current)} 只" + ("（空仓）" if not current else ""))

    dec_params = _decision_params(args)
    orders = make_orders_from_state(
        today, today["score"].to_numpy(), current, args.capital,
        params=dec_params,
        price_col=args.price_col,
        risk_state_path=args.risk_state,
        panel=panel,
    )

    # 4) 输出订单
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
    else:
        print("（未写持仓。确认按单成交后加 --commit 更新 holdings.csv）")

    print("\n提醒：信号基于当日收盘特征，请于【次日】执行；prediction 仅用于排序选股，非涨幅预测。")


if __name__ == "__main__":
    main()
