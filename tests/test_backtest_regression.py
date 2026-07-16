"""Quick regression test for backtest_daily.py with cached panel."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config as C
from data import list_factor_columns, load_factor_panel
from backtest_daily import run_daily_replay


def main():
    print("Loading cached panel...")
    panel = load_factor_panel(keep_trade_cols=True)
    factors = list_factor_columns(panel)
    latest = panel["date"].max()
    start = latest - pd.Timedelta(days=60)  # ~30 trading days
    print(f"Panel: {len(panel):,} rows, latest={latest.date()}, start={start.date()}")
    print(f"Factors: {len(factors)}")

    res = run_daily_replay(
        panel, factors,
        start_date=start.date(),
        end_date=latest.date(),
        params={"n_seeds": 1, "rebuild_days": 30, "train_window": 120,
                "strict_check": True, "progress": True},
    )
    daily = res["daily"]
    print(f"\nBacktest done: {len(daily)} decision days")
    print(daily[["date", "total_value", "n_positions", "turnover", "cost", "risk_scale"]].tail())
    print(f"Final total value: {daily['total_value'].iloc[-1]:,.0f}")
    print(f"Trades: {len(res['trades'])}")


if __name__ == "__main__":
    main()
