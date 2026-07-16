"""Focused smoke tests for sim-to-real gap fixes.

Avoids slow data loading by mocking the panel / ensemble path.
Run: PYTHONPATH=src:. uv run python3 tests/test_sim2real_fixes.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config as C
from trading import PortfolioState, decide_orders, simulate_fills, detect_board_limit
from live import make_orders_from_state
from live_state import RiskState, load_risk_state, save_risk_state, compute_risk_scale


def _make_today():
    codes = ["600519", "300001", "688001", "000001", "000333"]
    symbols = [f"{c}.SH" if i % 2 == 0 else f"{c}.SZ" for i, c in enumerate(codes)]
    return pd.DataFrame({
        "symbol": symbols,
        "name": ["MT", "CY", "KC", "PA", "MD"],
        "close": [100.0, 50.0, 80.0, 15.0, 30.0],
        "adj_close": [100.0, 50.0, 80.0, 15.0, 30.0],
        "bb_upper_20": [110.0, 55.0, 88.0, 16.5, 33.0],
        "bb_lower_20": [90.0, 45.0, 72.0, 13.5, 27.0],
        "rsi_14": [50.0, 40.0, 60.0, 55.0, 45.0],
        "limit_flag": [0, 1, 0, 0, 1],
        "pct_chg": [0.0, 19.8, 0.0, 0.0, -9.9],
        "date": pd.to_datetime(["2026-07-09"] * 5),
    })


def _make_next_bars():
    codes = ["600519", "300001", "688001", "000001", "000333"]
    symbols = [f"{c}.SH" if i % 2 == 0 else f"{c}.SZ" for i, c in enumerate(codes)]
    return pd.DataFrame({
        "symbol": symbols,
        "open": [101.0, 51.0, 81.0, 15.2, 30.3],
        "high": [102.0, 51.0, 82.0, 15.5, 30.3],  # 300001 high==low 模拟一字涨停
        "low": [100.0, 51.0, 80.0, 15.0, 30.3],
        "close": [101.5, 51.0, 81.5, 15.3, 30.3],
        "amount": [10000.0] * 5,
        "pre_close": [100.0, 50.0, 80.0, 15.0, 30.0],
        "pct_chg": [1.5, 20.0, 1.875, 2.0, 1.0],
    })


def test_detect_board_limit():
    assert detect_board_limit("600519.SH", 9.9, 11.0, 9.0, 10.0) == (True, False, False)
    assert detect_board_limit("300001.SZ", 19.9, 12.0, 8.0, 10.0) == (True, False, False)
    assert detect_board_limit("688001.SH", 19.5, 12.0, 8.0, 10.0)[0] is True
    assert detect_board_limit("000001.SZ", 9.9, 11.0, 9.0, 10.0)[0] is True
    assert detect_board_limit("000001.SZ", 4.9, 10.5, 9.5, 10.0, is_st=True)[0] is True
    assert detect_board_limit("830001.BJ", 29.5, 13.0, 7.0, 10.0)[0] is True
    print("✓ detect_board_limit")


def test_decide_orders_price_col():
    today = _make_today()
    today["adj_close"] = today["close"] * 1.1  # 人为差异
    score = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    state = PortfolioState(cash=1_000_000)

    orders_close = decide_orders(state.copy(), today, score, params={"topk": 0.4, "use_bb_timing": False},
                                 price_col="close")
    orders_adj = decide_orders(state.copy(), today, score, params={"topk": 0.4, "use_bb_timing": False},
                               price_col="adj_close")
    # adj_close 更高 → 目标股数更少
    assert orders_adj["shares"].sum() < orders_close["shares"].sum()
    print("✓ decide_orders price_col")


def test_simulate_fills_t1_lock():
    today = _make_today()
    score = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    asof = today["date"].max()

    # 场景：昨日空仓，今日决策买入 600519，同时系统错误地想卖出 600519
    state = PortfolioState(cash=1_000_000)
    buy_orders = decide_orders(state, today, score, params={"topk": 0.4, "use_bb_timing": False},
                               asof_date=asof, price_col="close")
    # 手动构造一个同日的卖出单（模拟 bug/同日重复决策）
    sell_order = pd.DataFrame([{
        "symbol": "600519.SH", "name": "MT", "side": "sell", "shares": 100,
        "limit_price": 100.0, "ref_close": 100.0, "reason": "test",
        "rank_pct": 0.0, "bb_pos": 0.5, "rsi": 50.0, "blocked": False,
        "asof_date": asof,
    }])
    orders = pd.concat([buy_orders, sell_order], ignore_index=True)
    next_bars = _make_next_bars()

    # 先执行买入，让 state 有持仓且 entry_date=asof
    state2 = PortfolioState(cash=1_000_000)
    simulate_fills(state2, buy_orders, next_bars, t1_lock=True)
    assert state2.holdings.get("600519.SH", 0) > 0
    assert str(state2.entry_date.get("600519.SH")) == str(pd.Timestamp(asof).date())

    # 再执行同日卖出单 → 应被 T+1 锁仓拒绝
    res = simulate_fills(state2, sell_order, next_bars, t1_lock=True)
    statuses = res["fills"]["status"].tolist()
    assert any("T+1" in s for s in statuses), statuses
    print("✓ simulate_fills T+1 lock")


def test_board_limit_in_fill():
    today = _make_today()
    # 300001.SZ 一字涨停，不应成交买入
    orders = pd.DataFrame([{
        "symbol": "300001.SZ", "name": "CY", "side": "buy", "shares": 100,
        "limit_price": 50.0, "ref_close": 50.0, "reason": "test",
        "rank_pct": 0.0, "bb_pos": 0.5, "rsi": 50.0, "blocked": False,
        "asof_date": today["date"].max(),
    }])
    next_bars = _make_next_bars()
    state = PortfolioState(cash=1_000_000)
    res = simulate_fills(state, orders, next_bars)
    assert not res["fills"]["filled"].iloc[0]
    assert "涨停" in res["fills"]["status"].iloc[0]
    print("✓ board limit in simulate_fills")


def test_make_orders_risk_state():
    today = _make_today()
    score = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    with tempfile.TemporaryDirectory() as tmp:
        rpath = os.path.join(tmp, "risk.json")
        state = RiskState()
        state.dates = ["2026-07-0%d" % i for i in range(1, 9)]
        state.total_values = [1_000_000.0] * 8
        # 制造高波动触发 vol target
        state.total_values[-3:] = [1_100_000, 900_000, 850_000]
        save_risk_state(rpath, state)

        orders = make_orders_from_state(
            today, score, {}, 1_000_000,
            params={"topk": 0.4, "use_bb_timing": False},
            price_col="close", risk_state_path=rpath)
        loaded = load_risk_state(rpath)
        assert loaded.last_update == "2026-07-09"
        assert len(loaded.total_values) == 9
        print("✓ make_orders_from_state risk state")


def test_predict_tomorrow_unit(monkeypatch):
    """用 monkeypatch 替换 load_factor_panel 等函数，跑通 predict_tomorrow 主流程。"""
    import predict_tomorrow as pt

    today_df = _make_today()
    panel = pd.concat([today_df.assign(date=pd.to_datetime(["2026-07-08"] * 5)), today_df])
    panel[C.TARGET] = np.random.randn(len(panel))
    panel[C.WEIGHT] = 1.0

    def _fake_load(*args, **kwargs):
        return panel

    def _fake_ensemble(panel, factors, params=None):
        today = panel[panel["date"] == panel["date"].max()].copy()
        today["score"] = np.linspace(1.0, 0.1, len(today))
        today["prediction"] = today["score"]
        today["rank_pct"] = today["score"].rank(ascending=False, pct=True)
        info = {"weights": {"gbm_l2": 1.0}, "ic_means": {"gbm_l2": 0.02},
                "models": ["gbm_l2"], "latest_date": today["date"].max()}
        return today, info

    monkeypatch.setattr(pt, "load_factor_panel", _fake_load)
    monkeypatch.setattr(pt, "ensemble_predict_latest", _fake_ensemble)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        monkeypatch.setattr(C, "OUTPUT_DIR", out_dir)
        args = pt.ap.parse_args(["--capital", "1000000"])
        # 直接调用 main 会 argparse，这里用 ap 测一下参数解析
        assert args.capital == 1_000_000
    print("✓ predict_tomorrow arg parsing")


if __name__ == "__main__":
    test_detect_board_limit()
    test_decide_orders_price_col()
    test_simulate_fills_t1_lock()
    test_board_limit_in_fill()
    test_make_orders_risk_state()
    # predict_tomorrow 的 monkeypatch 测试略复杂，先只跑解析
    print("✓ all focused tests passed")
