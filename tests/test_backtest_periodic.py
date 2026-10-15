"""run_backtest_periodic（周期调仓引擎）测试。

此前该引擎完全无测试。锁定：非重叠调仓、成本与每日重放引擎同公式
（trading.trade_costs，含每笔最低佣金）、年化指标存在且口径正确。

路径由 tests/conftest.py 统一处理；standalone 运行：
PYTHONPATH=src:. uv run python tests/test_backtest_periodic.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as C
from backtest import run_backtest_periodic


def _make_df(n_dates: int = 40, syms=("A", "B")):
    dates = pd.date_range("2026-05-01", periods=n_dates, freq="B")
    rows = []
    rng = np.random.default_rng(7)
    for di, d in enumerate(dates):
        for si, s in enumerate(syms):
            rows.append({
                C.DATE_COL: di,
                C.SYMBOL_COL: s,
                "return_1d": rng.normal(0.001, 0.02),
                "amount": 10000.0,      # 千元 → ADV = 1e7 元
                C.TARGET: rng.normal(0, 0.02),
            })
    return pd.DataFrame(rows)


def test_periodic_runs_and_reports_annualized_metrics():
    df = _make_df()
    pred = df[C.TARGET].to_numpy() + 0.01   # 有微弱相关性的预测
    pos = np.where(df[C.SYMBOL_COL].to_numpy() == "A", 0.5, 0.5)

    out = run_backtest_periodic(df, pred, pos, rebalance_days=10)
    m = out["metrics"]
    curve = out["curve"]

    # 基本结构
    assert len(curve) == df[C.DATE_COL].nunique()
    assert np.allclose(curve["equity"], curve["net"].cumsum())
    assert np.isfinite(curve["net"]).all()
    # 年化指标存在且口径自洽：sharpe ≈ ann_return / ann_vol
    assert "ann_return" in m and "ann_vol" in m
    if m["ann_vol"] > 1e-12:
        assert m["sharpe"] == pytest.approx(m["ann_return"] / m["ann_vol"], rel=1e-6)
    # 成本非负，且只在调仓日发生
    assert (curve["tc"] >= 0).all()
    rebal_idx = np.arange(0, len(curve), 10)
    non_rebal = np.setdiff1d(np.arange(len(curve)), rebal_idx)
    assert (curve["tc"].to_numpy()[non_rebal] == 0.0).all()
    assert curve["tc"].sum() > 0


def test_periodic_min_commission_matches_trade_costs():
    """零费率下成本只剩每笔最低佣金：与 trade_costs 逐笔手算一致。"""
    df = _make_df(n_dates=20)
    pred = np.zeros(len(df))
    pos = np.where(df[C.SYMBOL_COL].to_numpy() == "A", 0.5, 0.5)

    out = run_backtest_periodic(df, pred, pos, rebalance_days=10,
                                commission=0.0, stamp_duty=0.0)
    curve = out["curve"]
    cap = C.INITIAL_CAPITAL
    # 仓位恒定 → 只有第 0 天建仓有换手：2 只股 × (0.5 × cap) 各一笔买入
    # 佣金 = max(0, MIN_COMMISSION) = 5 元/笔（元口径 → 除以 cap 换算成分数）
    exp = 2 * C.MIN_COMMISSION / cap
    assert curve["tc_commission"].iloc[0] == pytest.approx(exp)
    assert curve["tc_commission"].iloc[1:].sum() == pytest.approx(0.0)


def test_periodic_impact_matches_trade_costs():
    """冲击成本与每日重放引擎同公式：0.1·sqrt(N/ADV)·N，按 (日,股) 计后求和。"""
    df = _make_df(n_dates=20)
    pred = np.zeros(len(df))
    pos = np.where(df[C.SYMBOL_COL].to_numpy() == "A", 0.5, 0.5)

    out = run_backtest_periodic(df, pred, pos, rebalance_days=10,
                                commission=0.0, stamp_duty=0.0)
    cap = C.INITIAL_CAPITAL
    notional = 0.5 * cap
    adv = 10000.0 * 1000.0
    exp_impact = 2 * (C.IMPACT_K * np.sqrt(notional / adv) * notional) / cap
    assert out["curve"]["tc_impact"].iloc[0] == pytest.approx(exp_impact)


if __name__ == "__main__":
    test_periodic_runs_and_reports_annualized_metrics()
    test_periodic_min_commission_matches_trade_costs()
    test_periodic_impact_matches_trade_costs()
    print("periodic backtest tests passed")
