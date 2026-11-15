from __future__ import annotations

import numpy as np
import pandas as pd

from trading import PortfolioState, board_of, decide_orders


def _today(symbols, closes) -> pd.DataFrame:
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-09"] * len(symbols)),
            "symbol": symbols,
            "name": [f"S{i}" for i in range(len(symbols))],
            "close": close,
            "bb_upper_20": close * 1.05,
            "bb_lower_20": close * 0.95,
            "rsi_14": 50.0,
            "limit_flag": 0,
            "pct_chg": 0.0,
        }
    )


def _orders(today, current=None, cash=100_000.0, params=None):
    return decide_orders(
        PortfolioState(cash=cash, holdings=dict(current or {})),
        today,
        np.linspace(1.0, 0.0, len(today)),
        params=params,
        asof_date=today["date"].max(),
    )


def test_board_of_prefixes():
    assert board_of("688001.SH") == "star"
    assert board_of("300001.SZ") == "chinext"
    assert board_of("301001.SZ") == "chinext"
    assert board_of("830001.BJ") == "bse"
    assert board_of("430001.BJ") == "bse"
    assert board_of("920001.BJ") == "bse"
    assert board_of("600000.SH") == "main"


def test_allowed_boards_only_restricts_buy_candidates():
    today = _today(
        ["688001.SH", "300001.SZ", "600000.SH", "000001.SZ", "830001.BJ"],
        [10.0] * 5,
    )
    orders = _orders(
        today,
        params={
            "topk": 1.0,
            "max_positions": 5,
            "max_weight": 0.2,
            "allowed_boards": ["main"],
        },
    )
    buys = set(orders.loc[orders["side"] == "buy", "symbol"])
    assert buys == {"600000.SH", "000001.SZ"}


def test_unaffordable_leader_is_skipped_and_next_candidate_fills():
    symbols = [f"{600000 + i}.SH" for i in range(5)]
    today = _today(symbols, [800.0, 10.0, 20.0, 30.0, 40.0])
    base = {"topk": 1.0, "max_positions": 3, "max_weight": 0.2, "min_notional": 0}

    old = _orders(today, cash=30_000.0, params=base)
    disabled = _orders(today, cash=30_000.0, params={**base, "skip_unaffordable": False})
    pd.testing.assert_frame_equal(old, disabled)

    enabled = _orders(today, cash=30_000.0, params={**base, "skip_unaffordable": True})
    buys = set(enabled.loc[enabled["side"] == "buy", "symbol"])
    assert "600000.SH" not in buys
    assert "600003.SH" in buys


def test_disallowed_existing_holding_is_still_sold():
    today = _today(["688001.SH", "600000.SH", "600001.SH"], [10.0] * 3)
    orders = _orders(
        today,
        current={"688001.SH": 100},
        params={
            "topk": 1.0,
            "max_positions": 2,
            "max_weight": 0.5,
            "allowed_boards": ["main"],
        },
    )
    sells = orders[(orders["symbol"] == "688001.SH") & (orders["side"] == "sell")]
    assert len(sells) == 1
    assert sells.iloc[0]["shares"] == 100


def test_predict_cli_passes_tradability_params():
    import predict_tomorrow as pt

    args = pt.build_parser().parse_args(["--allowed-boards", "main,chinext", "--skip-unaffordable"])
    params = pt._decision_params(args)
    assert params["allowed_boards"] == ["main", "chinext"]
    assert params["skip_unaffordable"] is True

    default_params = pt._decision_params(pt.build_parser().parse_args([]))
    assert "allowed_boards" not in default_params
    assert "skip_unaffordable" not in default_params
