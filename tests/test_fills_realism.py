"""阶段 3 撮合/估值现实性修复的单元测试。

覆盖：
  · 停牌日净值不跳变（last_price 兜底估值，不再按 0 计）；
  · 除权日（pre_close ≠ 上一原始收盘）无假性盈亏（val_factor 链条）；
  · blocked 单在撮合侧不成交（与实盘 --commit 语义一致）；
  · 跌出横截面的持仓被发出清退单；
  · 现金不足时循环缩手，不留负现金。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading import PortfolioState, decide_orders, simulate_fills, detect_board_limit
from backtest_daily import _price_map


def _next_bar(sym: str, open_: float = 10.0, high: float = 10.1, low: float = 9.9,
              close: float = 10.0, amount: float = 100.0, pre_close: float = 10.0,
              pct_chg: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": [sym], "open": [open_], "high": [high], "low": [low],
        "close": [close], "amount": [amount], "pre_close": [pre_close],
        "pct_chg": [pct_chg],
    })


def _order(sym: str, side: str, shares: int, limit_price: float = 10.0,
           blocked: bool = False, asof: str = "2026-07-09") -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": sym, "name": sym, "side": side, "shares": shares,
        "limit_price": limit_price, "ref_close": limit_price, "reason": "test",
        "rank_pct": 0.0, "bb_pos": 0.5, "rsi": 50.0, "blocked": blocked,
        "asof_date": asof,
    }])


def test_suspension_valuation_no_jump():
    """停牌日：无价格时用 last_price 兜底，净值不跳变（修复前按 0 计）。"""
    state = PortfolioState(cash=5_000.0, holdings={"600000.SH": 1000})
    mv = state.market_value({"600000.SH": 10.0})
    assert mv == 10_000.0
    # 次日停牌（估值价表里没有它）→ 按 last_price 估值
    assert state.market_value({}) == 10_000.0
    assert state.total_value({}) == 15_000.0
    # 完全无历史价的持仓才按 0 计（边缘情形）
    state2 = PortfolioState(cash=0.0, holdings={"000001.SZ": 100})
    assert state2.market_value({}) == 0.0


def test_ex_dividend_no_fake_pnl():
    """除权日（pre_close ≠ 上一原始收盘）：估值链不跳变，之后真涨跌正常反映。"""
    state = PortfolioState(cash=0.0, holdings={"600000.SH": 1000})
    # T 日正常收盘 10 元（pre_close 9.8 是昨日收盘，不对 last_price 触发调整）
    bars_t = pd.DataFrame({"symbol": ["600000.SH"], "close": [10.0], "pre_close": [9.8]})
    v_t = state.market_value(_price_map(bars_t, state))
    assert v_t == 10_000.0
    assert state.val_factor.get("600000.SH", 1.0) == 1.0
    # T+1 除权（10 派 10 类）：原始收盘 5 元、pre_close 5 元——并非真跌 50%
    bars_ex = pd.DataFrame({"symbol": ["600000.SH"], "close": [5.0], "pre_close": [5.0]})
    v_ex = state.market_value(_price_map(bars_ex, state))
    assert abs(v_ex - 10_000.0) < 1e-6            # 无假性腰斩
    assert abs(state.val_factor["600000.SH"] - 2.0) < 1e-9
    # 除权后真涨 2%：close 5.10 / pre_close 5.0 → 估值恰好 +2%
    bars_n = pd.DataFrame({"symbol": ["600000.SH"], "close": [5.10], "pre_close": [5.0]})
    v_n = state.market_value(_price_map(bars_n, state))
    assert abs(v_n - 10_200.0) < 1e-6


def test_blocked_order_not_filled():
    """blocked（决策层涨跌停拦截）订单在撮合侧记 0 成交，state 不动。"""
    state = PortfolioState(cash=1_000_000.0)
    orders = _order("300001.SZ", "buy", 1000, blocked=True)
    res = simulate_fills(state, orders, _next_bar("300001.SZ"))
    assert res["n_fill"] == 0
    assert res["fills"]["fill_shares"].iloc[0] == 0
    assert "拦截" in res["fills"]["status"].iloc[0]
    assert state.cash == 1_000_000.0
    assert not state.holdings


def test_off_universe_holding_exit_order():
    """持仓股跌出当日横截面：有 last_price 则发市价清退单，不再卡死持有。"""
    today = pd.DataFrame({
        "symbol": ["600000.SH", "000001.SZ"],
        "close": [10.0, 20.0],
        "pct_chg": [0.0, 0.0],
        "limit_flag": [0, 0],
    })
    score = np.array([1.0, 0.5])
    for use_bb in (False, True):      # 满仓再平衡分支与布林择时分支都要覆盖
        state = PortfolioState(cash=0.0, holdings={"300001.SZ": 1000})
        state.market_value({"300001.SZ": 20.0})   # 建立 last_price
        orders = decide_orders(
            state, today, score,
            params={"topk": 0.5, "hold_ratio": 0.5, "use_bb_timing": use_bb},
            asof_date=pd.Timestamp("2026-07-09"))
        sell = orders[(orders["symbol"] == "300001.SZ") & (orders["side"] == "sell")]
        assert len(sell) == 1, f"use_bb={use_bb}: 应发出清退单\n{orders}"
        assert sell["shares"].iloc[0] == 1000
        assert not sell["blocked"].iloc[0]
        assert "跌出横截面" in sell["reason"].iloc[0]
    # 完全无历史价 → 不发单，继续持有等待
    state3 = PortfolioState(cash=0.0, holdings={"300001.SZ": 1000})
    orders3 = decide_orders(state3, today, score,
                            params={"topk": 0.5, "hold_ratio": 0.5, "use_bb_timing": False},
                            asof_date=pd.Timestamp("2026-07-09"))
    assert len(orders3[orders3["symbol"] == "300001.SZ"]) == 0


def test_cash_shortfall_shrink_loop():
    """现金不足：冲击成本使总价超现金时循环缩手，直到买得起，不留负现金。"""
    # impact_k 调大使单次缩手不够：1000→900→800 仍超，700 才买得起（手算验证）
    state = PortfolioState(cash=10_000.0)
    orders = _order("600000.SH", "buy", 1000)
    res = simulate_fills(state, orders, _next_bar("600000.SH"),
                         params={"use_impact": True, "impact_k": 1.0})
    f = res["fills"].iloc[0]
    assert f["filled"] and f["fill_shares"] == 700
    assert state.cash >= -1e-6                    # 修复前会扣出负现金
    assert state.holdings["600000.SH"] == 700
    # 现金连一手都买不起 → 记"现金不足"，state 不动
    state2 = PortfolioState(cash=500.0)
    res2 = simulate_fills(state2, _order("600000.SH", "buy", 1000),
                          _next_bar("600000.SH"),
                          params={"use_impact": True, "impact_k": 1.0})
    assert not res2["fills"]["filled"].iloc[0]
    assert res2["fills"]["status"].iloc[0] == "现金不足"
    assert state2.cash == 500.0


def test_one_word_board_tolerance():
    """一字板判定改用容差：high/low 有分笔级差异（≤0.1%×pre_close）仍判一字。"""
    # 精确相等：一字
    assert detect_board_limit("600000.SH", None, 11.0, 11.0, 10.0)[2] is True
    # 容差内（差 0.005 ≤ 0.001×10）：仍一字
    assert detect_board_limit("600000.SH", None, 11.0, 10.995, 10.0)[2] is True
    # 超容差（差 0.02）：非一字
    assert detect_board_limit("600000.SH", None, 11.0, 10.98, 10.0)[2] is False
