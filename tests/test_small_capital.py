from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from capital import capital_profile_params, with_capital_profile
from trading import PortfolioState, decide_orders, simulate_fills


def _today(symbols, closes) -> pd.DataFrame:
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-11-25"] * len(symbols)),
            "symbol": symbols,
            "name": symbols,
            "close": close,
            "limit_flag": 0,
            "pct_chg": 0.0,
        }
    )


def _params(**overrides) -> dict:
    params = {
        "topk": 1.0,
        "max_positions": 4,
        "max_weight": 0.25,
        "min_notional": 4000.0,
        "min_entry_notional": 4000.0,
        "cash_aware": True,
        "cash_buffer": 50.0,
        "skip_unaffordable": True,
        "use_bb_timing": False,
    }
    params.update(overrides)
    return params


def test_small_profile_uses_minimum_commission_to_derive_order_floor(monkeypatch):
    monkeypatch.setattr(C, "MIN_COMMISSION", 5.0)
    params = capital_profile_params("small")
    assert params["max_positions"] == 4
    assert params["max_weight"] == 0.25
    assert params["min_entry_notional"] == 4000.0
    assert params["min_notional"] == 4000.0
    assert params["cash_aware"] is True


def test_new_position_below_entry_floor_is_blocked():
    today = _today(["600000.SH"], [5.0])
    orders = decide_orders(
        PortfolioState(cash=20_000.0),
        today,
        np.array([1.0]),
        params=_params(max_positions=30, max_weight=0.1, min_notional=30_000, min_entry_notional=5000),
        asof_date=today["date"].max(),
    )
    assert orders.empty


def test_full_exit_is_not_blocked_by_entry_floor():
    today = _today(["600000.SH", "600001.SH"], [5.0, 5.0])
    state = PortfolioState(cash=0.0, holdings={"600001.SH": 100}, last_price={"600001.SH": 5.0})
    orders = decide_orders(
        state,
        today,
        np.array([1.0, 0.0]),
        params=_params(topk=0.1, max_positions=1, max_weight=1.0, min_entry_notional=5000),
        asof_date=today["date"].max(),
    )
    sells = orders[orders["side"] == "sell"]
    assert len(sells) == 1
    assert sells.iloc[0]["symbol"] == "600001.SH"
    assert sells.iloc[0]["shares"] == 100


def test_cash_budget_caps_buy_and_preserves_executable_orders():
    symbols = [f"{600000 + i}.SH" for i in range(4)]
    today = _today(symbols, [10.0] * 4)
    orders = decide_orders(
        PortfolioState(cash=20_000.0),
        today,
        np.arange(4, 0, -1),
        params=_params(),
        asof_date=today["date"].max(),
    )
    assert len(orders) == 4
    assert orders.iloc[-1]["shares"] == 400
    report = orders.attrs["cash_report"]
    assert report["capped_buys"] == ["600003.SH"]
    assert report["skipped_buys"] == []
    assert report["estimated_free_cash"] >= 0

    close = np.full(4, 10.0)
    next_bars = pd.DataFrame(
        {
            "symbol": symbols,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "amount": 1e8,
            "pre_close": close,
            "pct_chg": 0.0,
        }
    )
    state = PortfolioState(cash=20_000.0)
    result = simulate_fills(
        state,
        orders,
        next_bars,
        params={**_params(), "order_price_mode": "next_open"},
    )
    assert result["n_reject"] == 0
    assert sum(value > 0 for value in state.holdings.values()) == 4
    assert state.cash >= 0


def test_pipeline_standard_defaults_do_not_mask_small_profile():
    params = with_capital_profile(
        "small",
        {
            "max_positions": C.MAX_POSITIONS,
            "max_weight": C.MAX_WEIGHT_PER_NAME,
            "min_notional": C.DAILY_MIN_NOTIONAL,
        },
        ignore_standard_defaults=True,
    )
    assert params["max_positions"] == 4
    assert params["max_weight"] == 0.25
    assert params["min_notional"] == 4000.0


def test_small_profile_cli_and_explicit_override():
    import predict_tomorrow as pt

    args = pt.build_parser().parse_args(["--capital", "20000", "--capital-profile", "small"])
    params = pt._decision_params(args)
    assert params["max_positions"] == 4
    assert params["min_entry_notional"] == 4000.0
    assert params["cash_aware"] is True

    args = pt.build_parser().parse_args(
        ["--capital-profile", "small", "--max-positions", "5", "--min-entry-notional", "3000"]
    )
    params = pt._decision_params(args)
    assert params["max_positions"] == 5
    assert params["min_entry_notional"] == 3000.0
