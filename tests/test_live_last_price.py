from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from live import make_orders_from_state


def _today() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-07-09"] * 4)
    close = np.array([10.0, 20.0, 30.0, 40.0])
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": ["600000.SH", "600001.SH", "600002.SH", "600003.SH"],
            "name": ["A", "B", "C", "D"],
            "close": close,
            "adj_close": close * 2.0,
            "bb_upper_20": close * 1.05,
            "bb_lower_20": close * 0.95,
            "rsi_14": 50.0,
            "limit_flag": 0,
            "pct_chg": 0.0,
        }
    )


def _orders(today: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return make_orders_from_state(
        today,
        np.linspace(1.0, 0.0, len(today)),
        {"600999.SH": 1000},
        100_000.0,
        params={"topk": 0.5, "max_positions": 2, "min_notional": 1000},
        cash=50_000.0,
        **kwargs,
    )


def test_missing_symbol_uses_ref_price_for_liquidation():
    orders = _orders(_today(), ref_prices={"600999.SH": 10.0})
    liquidation = orders[
        (orders["symbol"] == "600999.SH")
        & (orders["side"] == "sell")
        & (orders["reason"] == "跌出横截面:清退")
    ]
    assert len(liquidation) == 1
    assert liquidation.iloc[0]["shares"] == 1000
    assert liquidation.iloc[0]["limit_price"] == 10.0


def test_latest_valid_raw_panel_close_has_priority():
    today = _today()
    historical = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-08", "2026-07-09"]),
            "symbol": ["600999.SH"] * 2,
            "close": [12.0, np.nan],
        }
    )
    panel = pd.concat([today, historical], ignore_index=True)
    orders = _orders(today, panel=panel, ref_prices={"600999.SH": 10.0})
    liquidation = orders[orders["symbol"] == "600999.SH"]
    assert liquidation.iloc[0]["limit_price"] == 12.0


def test_missing_all_prices_warns_and_does_not_liquidate(caplog):
    with caplog.at_level(logging.WARNING, logger="live"):
        orders = _orders(_today())
    assert "600999.SH" not in set(orders["symbol"])
    assert "600999.SH" in caplog.text
    assert "last_price" in caplog.text
