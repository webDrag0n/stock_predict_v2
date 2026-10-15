"""成本模型手算断言：佣金双边(每笔最低5元)、印花税卖出单边、平方根冲击成本。

锁定 src/trading.py trade_costs / simulate_fills 的成本数学（config.py 名义值）：
  佣金   = max(notional × COMMISSION_RATE(0.0002), MIN_COMMISSION(5))  —— 买卖双边
  印花税 = notional × STAMP_DUTY_RATE(0.001)          —— 仅卖出
  冲击   = notional × IMPACT_K(0.1) × sqrt(notional / ADV)
  ADV    = 次日 amount(千元) × 1000
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as C
from trading import PortfolioState, simulate_fills, trade_costs

PX = 10.0
SHARES = 1000
AMOUNT = 1000.0          # 千元 → ADV = 1e6 元
ADV = AMOUNT * 1000.0
NOTIONAL = PX * SHARES   # 10_000

EXP_COMM = max(NOTIONAL * 0.0002, C.MIN_COMMISSION)               # 5.0（最低佣金）
EXP_STAMP = NOTIONAL * 0.001                                      # 10.0（仅卖出）
EXP_IMPACT = 0.1 * np.sqrt(NOTIONAL / ADV) * NOTIONAL             # 100.0


def _bar(sym: str = "600000.SH") -> pd.DataFrame:
    """次日行情：open=10，高低留有区间（非一字板），amount 决定 ADV。"""
    return pd.DataFrame({
        "symbol": [sym], "open": [PX], "high": [PX * 1.01], "low": [PX * 0.99],
        "close": [PX], "amount": [AMOUNT], "pre_close": [PX], "pct_chg": [0.0],
    })


def _order(side: str, sym: str = "600000.SH",
           asof: str = "2026-07-09") -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": sym, "name": sym, "side": side, "shares": SHARES,
        "limit_price": PX, "ref_close": PX, "reason": "test",
        "rank_pct": 0.0, "bb_pos": 0.5, "rsi": 50.0, "blocked": False,
        "asof_date": asof,
    }])


def test_buy_costs_commission_plus_impact_no_stamp():
    """买入：佣金 + 冲击，无印花税；现金按 notional+成本 扣减。"""
    state = PortfolioState(cash=1_000_000.0)
    res = simulate_fills(state, _order("buy"), _bar(),
                         params={"order_price_mode": "next_open"}, t1_lock=True)
    assert res["n_fill"] == 1
    assert res["buy_notional"] == pytest.approx(NOTIONAL)
    assert res["commission"] == pytest.approx(EXP_COMM)
    assert res["impact"] == pytest.approx(EXP_IMPACT)
    assert res["stamp"] == 0.0                       # 买入不收印花税
    assert res["total_cost"] == pytest.approx(EXP_COMM + EXP_IMPACT)
    assert state.cash == pytest.approx(1_000_000.0 - NOTIONAL - EXP_COMM - EXP_IMPACT)
    assert state.holdings["600000.SH"] == SHARES


def test_sell_costs_commission_stamp_plus_impact():
    """卖出：佣金 + 印花税 + 冲击；现金按 notional−成本 入账。"""
    state = PortfolioState(cash=0.0, holdings={"600000.SH": SHARES})
    # 建仓日早于订单 asof，避开 T+1 锁
    state.entry_date["600000.SH"] = "2026-07-08"
    res = simulate_fills(state, _order("sell", asof="2026-07-09"), _bar(),
                         params={"order_price_mode": "next_open"}, t1_lock=True)
    assert res["n_fill"] == 1
    assert res["sell_notional"] == pytest.approx(NOTIONAL)
    assert res["commission"] == pytest.approx(EXP_COMM)
    assert res["stamp"] == pytest.approx(EXP_STAMP)   # 印花税仅卖出侧
    assert res["impact"] == pytest.approx(EXP_IMPACT)
    assert res["total_cost"] == pytest.approx(EXP_COMM + EXP_STAMP + EXP_IMPACT)
    assert state.cash == pytest.approx(NOTIONAL - EXP_COMM - EXP_STAMP - EXP_IMPACT)
    assert "600000.SH" not in state.holdings


def test_commission_is_charged_on_both_sides():
    """佣金双边：一买一卖两个方向的 commission 相同（同一 notional）。"""
    buy_state = PortfolioState(cash=1_000_000.0)
    r_buy = simulate_fills(buy_state, _order("buy"), _bar(),
                           params={"order_price_mode": "next_open"})
    sell_state = PortfolioState(cash=0.0, holdings={"600000.SH": SHARES})
    sell_state.entry_date["600000.SH"] = "2026-07-08"
    r_sell = simulate_fills(sell_state, _order("sell"), _bar(),
                            params={"order_price_mode": "next_open"})
    assert r_buy["commission"] == pytest.approx(r_sell["commission"])
    assert r_buy["commission"] == pytest.approx(
        max(NOTIONAL * C.COMMISSION_RATE, C.MIN_COMMISSION))
    assert r_sell["stamp"] == pytest.approx(NOTIONAL * C.STAMP_DUTY_RATE)


def test_impact_scales_with_sqrt_participation():
    """冲击成本 = IMPACT_K·sqrt(N/ADV)·N：成交额 4 倍 → 冲击 8 倍（2²×2）。"""
    big = NOTIONAL * 4
    exp_big = C.IMPACT_K * np.sqrt(big / ADV) * big
    assert exp_big == pytest.approx(EXP_IMPACT * 8)

    state = PortfolioState(cash=10_000_000.0)
    res = simulate_fills(state, _order("buy").assign(shares=SHARES * 4), _bar(),
                         params={"order_price_mode": "next_open"})
    assert res["impact"] == pytest.approx(exp_big)


def test_impact_disabled_or_missing_adv():
    """use_impact=False 或缺 amount 列（ADV 不可得）时冲击成本为 0。"""
    state = PortfolioState(cash=1_000_000.0)
    res = simulate_fills(state, _order("buy"), _bar(),
                         params={"order_price_mode": "next_open",
                                 "use_impact": False})
    assert res["impact"] == 0.0
    assert res["total_cost"] == pytest.approx(EXP_COMM)

    state2 = PortfolioState(cash=1_000_000.0)
    res2 = simulate_fills(state2, _order("buy"), _bar().drop(columns=["amount"]),
                          params={"order_price_mode": "next_open"})
    assert res2["impact"] == 0.0


def test_min_commission_floors_small_orders():
    """最低佣金：小单佣金 = MIN_COMMISSION，大单按比例超过下限。"""
    # 小单：10_000 元 × 0.0002 = 2 元 < 5 元下限
    comm_small, _, _ = trade_costs(10_000.0, "buy")
    assert comm_small == pytest.approx(C.MIN_COMMISSION)
    # 大单：100_000 元 × 0.0002 = 20 元 > 下限 → 按比例
    comm_big, _, _ = trade_costs(100_000.0, "buy")
    assert comm_big == pytest.approx(100_000.0 * C.COMMISSION_RATE)
    # min_commission=0 退回纯比例
    comm_pure, _, _ = trade_costs(10_000.0, "buy", params={"min_commission": 0.0})
    assert comm_pure == pytest.approx(10_000.0 * C.COMMISSION_RATE)


def test_trade_costs_vector_matches_scalar():
    """向量化与逐笔标量结果一致（周期引擎与每日重放共用同一实现的前提）。"""
    notionals = np.array([10_000.0, 0.0, 100_000.0])
    advs = np.array([1e6, np.inf, 1e8])
    cv, sv, iv = trade_costs(notionals, "sell", advs)
    for k in range(len(notionals)):
        c1, s1, i1 = trade_costs(float(notionals[k]), "sell", float(advs[k]))
        assert cv[k] == pytest.approx(float(c1))
        assert sv[k] == pytest.approx(float(s1))
        assert iv[k] == pytest.approx(float(i1))
    # 零成交额不产生最低佣金
    assert cv[1] == 0.0


def test_participation_cap_limits_fill_size():
    """参与率上限：单笔成交 ≤ MAX_PARTICIPATION × 次日成交额，部分成交并标注。"""
    # ADV = 1e6 元，cap 10% → 最多 100_000 元 = 10_000 股 @10元；要 20_000 股
    state = PortfolioState(cash=10_000_000.0)
    order = _order("buy").assign(shares=20_000)
    res = simulate_fills(state, order, _bar(),
                         params={"order_price_mode": "next_open",
                                 "max_participation": 0.10})
    assert res["n_fill"] == 1
    assert res["fills"]["fill_shares"].iloc[0] == 10_000
    assert "限量" in res["fills"]["status"].iloc[0]
    assert state.holdings["600000.SH"] == 10_000

    # 卖出同样受限：持 20_000 股，只能卖 10_000 股
    state2 = PortfolioState(cash=0.0, holdings={"600000.SH": 20_000})
    state2.entry_date["600000.SH"] = "2026-07-08"
    res2 = simulate_fills(state2, _order("sell").assign(shares=20_000), _bar(),
                          params={"order_price_mode": "next_open",
                                  "max_participation": 0.10})
    assert res2["fills"]["fill_shares"].iloc[0] == 10_000
    assert state2.holdings["600000.SH"] == 10_000

    # max_participation=0 → 不限量（退回旧行为）
    state3 = PortfolioState(cash=10_000_000.0)
    res3 = simulate_fills(state3, _order("buy").assign(shares=20_000), _bar(),
                          params={"order_price_mode": "next_open",
                                  "max_participation": 0.0})
    assert res3["fills"]["fill_shares"].iloc[0] == 20_000


def test_participation_cap_rejects_when_below_one_lot():
    """参与率上限不足一手 → 整单拒绝（流动性不足）。"""
    # ADV = 1000 元（amount=1 千元），cap 10% = 100 元 → 不足 1 手(1000元)
    state = PortfolioState(cash=1_000_000.0)
    bar = _bar().assign(amount=1.0)
    res = simulate_fills(state, _order("buy"), bar,
                         params={"order_price_mode": "next_open",
                                 "max_participation": 0.10})
    assert res["n_fill"] == 0
    assert "流动性不足" in res["fills"]["status"].iloc[0]


def test_t1_lock_granularity_only_locks_same_day_buys():
    """T+1 按新增股数粒度：当日新买 200 股锁定，同股老仓 300 股同日可卖。"""
    asof = "2026-07-09"
    state = PortfolioState(cash=0.0, holdings={"600000.SH": 500})
    state.entry_date["600000.SH"] = "2026-07-08"     # 老仓
    state.t1_locked["600000.SH"] = (asof, 200)        # 当日新买 200 股被锁
    res = simulate_fills(state, _order("sell", asof=asof).assign(shares=400),
                         _bar(), params={"order_price_mode": "next_open"},
                         t1_lock=True)
    # 可卖 = 500 - 200 = 300
    assert res["fills"]["fill_shares"].iloc[0] == 300
    assert state.holdings["600000.SH"] == 200

    # 全部仓位都是当日新买 → 整单拒绝
    state2 = PortfolioState(cash=0.0, holdings={"600000.SH": 200})
    state2.entry_date["600000.SH"] = asof
    state2.t1_locked["600000.SH"] = (asof, 200)
    res2 = simulate_fills(state2, _order("sell", asof=asof).assign(shares=200),
                          _bar(), params={"order_price_mode": "next_open"},
                          t1_lock=True)
    assert res2["n_fill"] == 0
    assert "T+1" in res2["fills"]["status"].iloc[0]
    assert state2.holdings["600000.SH"] == 200
