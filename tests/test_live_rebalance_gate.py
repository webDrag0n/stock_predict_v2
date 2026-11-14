from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config as C
from live import is_rebalance_day, make_orders_from_state
from live_state import (
    load_rebalance_state,
    load_risk_state,
    rebalance_state_path_for,
    save_rebalance_state,
    trading_days_since,
)
from trading import PortfolioState, decide_orders


def _make_panel(n_days: int = 12, n_sym: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-01", periods=n_days)
    rows = []
    for day_i, date in enumerate(dates):
        for sym_i in range(n_sym):
            close = 10.0 + sym_i + day_i * 0.01
            rows.append(
                {
                    "date": date,
                    "symbol": f"{600000 + sym_i}.SH",
                    "name": f"S{sym_i}",
                    "close": close,
                    "adj_close": close,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "amount": 1_000_000.0,
                    "pre_close": close * 0.99,
                    "pct_chg": 0.5,
                    "limit_flag": 0,
                    "return_1d": 0.005,
                    "bb_upper_20": close * 1.05,
                    "bb_lower_20": close * 0.95,
                    "rsi_14": 50.0,
                    C.TARGET: 0.01 * (sym_i - 2),
                    C.WEIGHT: 1.0,
                    "feature_00": float(sym_i),
                }
            )
    return pd.DataFrame(rows)


def _params() -> dict:
    return {
        "topk": 0.5,
        "max_positions": 2,
        "max_weight": 0.5,
        "min_notional": 1000.0,
        "use_bb_timing": False,
        "use_vol_target": False,
        "use_dd_brake": False,
        "use_ic_scale": False,
        "use_regime": False,
        "use_iv_scale": False,
    }


def test_rebalance_gate_intervals(tmp_path):
    panel = _make_panel()
    dates = np.sort(panel["date"].unique())
    today = dates[-1]
    path = str(tmp_path / "live_risk_state.json")
    rebal_path = rebalance_state_path_for(path)

    assert rebal_path == str(tmp_path / "live_rebalance_state.json")
    assert is_rebalance_day(panel, today, path, every=10)
    assert trading_days_since(panel, None, today) is None

    save_rebalance_state(rebal_path, str(pd.Timestamp(dates[-4]).date()))
    assert trading_days_since(panel, dates[-4], today) == 3
    assert not is_rebalance_day(panel, today, path, every=10)
    assert is_rebalance_day(panel, today, path, every=3)
    assert is_rebalance_day(panel, today, path, every=1)
    assert is_rebalance_day(panel, today, path, every=10, force=True)

    save_rebalance_state(rebal_path, str(pd.Timestamp(dates[-11]).date()))
    assert is_rebalance_day(panel, today, path, every=10)
    assert load_rebalance_state(rebal_path) == {
        "last_rebalance_date": str(pd.Timestamp(dates[-11]).date())
    }


def test_corrupt_rebalance_state_is_backed_up(tmp_path):
    path = str(tmp_path / "live_rebalance_state.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{broken")

    assert load_rebalance_state(path) == {"last_rebalance_date": None}
    assert not os.path.exists(path)
    assert list(tmp_path.glob("live_rebalance_state.json.corrupt-*.bak"))
    assert is_rebalance_day(_make_panel(), _make_panel()["date"].max(), str(tmp_path / "risk.json"))


def test_non_rebalance_day_returns_empty_but_updates_risk(tmp_path, capsys):
    panel = _make_panel()
    dates = np.sort(panel["date"].unique())
    today = panel[panel["date"] == dates[-1]].copy()
    risk_path = str(tmp_path / "live_risk_state.json")
    save_rebalance_state(rebalance_state_path_for(risk_path), str(pd.Timestamp(dates[-4]).date()))

    orders = make_orders_from_state(
        today,
        np.linspace(1.0, 0.0, len(today)),
        {},
        1_000_000.0,
        params=_params(),
        risk_state_path=risk_path,
        panel=panel,
        cash=1_000_000.0,
    )

    expected = decide_orders(
        PortfolioState(cash=1_000_000.0),
        today.iloc[0:0],
        np.array([]),
        params=_params(),
        asof_date=dates[-1],
    )
    assert orders.empty
    assert list(orders.columns) == list(expected.columns)
    assert load_risk_state(risk_path).last_update == str(pd.Timestamp(dates[-1]).date())
    assert load_risk_state(risk_path).total_values == [1_000_000.0]
    assert "今日非调仓日（距上次调仓 3 个交易日，周期 10），未生成订单" in capsys.readouterr().out


def test_force_rebalance_bypasses_interval(tmp_path):
    panel = _make_panel()
    dates = np.sort(panel["date"].unique())
    today = panel[panel["date"] == dates[-1]].copy()
    risk_path = str(tmp_path / "live_risk_state.json")
    save_rebalance_state(rebalance_state_path_for(risk_path), str(pd.Timestamp(dates[-2]).date()))
    params = dict(_params(), force_rebalance=True)

    orders = make_orders_from_state(
        today,
        np.linspace(1.0, 0.0, len(today)),
        {},
        1_000_000.0,
        params=params,
        risk_state_path=risk_path,
        panel=panel,
        cash=1_000_000.0,
    )
    assert not orders.empty


def test_cli_records_only_nonempty_rebalance(monkeypatch, tmp_path):
    import predict_tomorrow as pt

    panel = _make_panel()
    latest = panel["date"].max()
    today = panel[panel["date"] == latest].copy()
    today["score"] = np.linspace(1.0, 0.0, len(today))
    today["prediction"] = today["score"]
    out_dir = tmp_path / "outputs"
    monkeypatch.setattr(C, "OUTPUT_DIR", str(out_dir))
    monkeypatch.setattr(C, "TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setattr(C, "MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(pt, "load_live_panel", lambda **kwargs: (panel, ["feature_00"]))
    monkeypatch.setattr(
        pt,
        "run_gated_prediction",
        lambda *args, **kwargs: {
            "today": today,
            "info": {"weights": {"gbm_l2": 1.0}},
            "gate": None,
            "action": "frozen",
            "ic_stats": None,
        },
    )

    risk_path = str(tmp_path / "live_risk_state.json")
    base_args = [
        "--capital",
        "1000000",
        "--holdings",
        str(tmp_path / "holdings.csv"),
        "--account",
        str(tmp_path / "account.csv"),
        "--risk-state",
        risk_path,
        "--topk",
        "0.5",
        "--max-positions",
        "2",
        "--max-weight",
        "0.5",
        "--min-notional",
        "1000",
        "--no-gate",
    ]
    rebal_path = rebalance_state_path_for(risk_path)
    out_csv = out_dir / f"orders_{pd.Timestamp(latest).strftime('%Y%m%d')}.csv"

    pt.main(base_args + ["--topk", "-1"])
    assert pd.read_csv(out_csv).empty
    assert not os.path.exists(rebal_path)

    pt.main(base_args)
    assert not pd.read_csv(out_csv).empty
    assert load_rebalance_state(rebal_path)["last_rebalance_date"] == str(latest.date())

    pt.main(base_args)
    assert pd.read_csv(out_csv).empty
    assert load_rebalance_state(rebal_path)["last_rebalance_date"] == str(latest.date())

    pt.main(base_args + ["--force-rebalance"])
    assert not pd.read_csv(out_csv).empty
    with open(rebal_path, encoding="utf-8") as f:
        assert json.load(f)["last_rebalance_date"] == str(latest.date())
