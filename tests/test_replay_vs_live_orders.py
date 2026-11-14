from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import model_store
from backtest_daily import run_daily_replay
from live import make_orders_from_state
from live_state import rebalance_state_path_for, save_rebalance_state


def _make_panel(n_days: int = 60, n_sym: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2026-05-01", periods=n_days)
    symbols = [f"{600000 + i}.SH" for i in range(n_sym)]
    previous = np.linspace(8.0, 30.0, n_sym)
    rows = []
    for date in dates:
        close = np.maximum(previous * (1.0 + rng.normal(0.0, 0.005, n_sym)), 2.0)
        open_ = close * (1.0 + rng.normal(0.0, 0.002, n_sym))
        for i, symbol in enumerate(symbols):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "name": symbol[:6],
                    "close": close[i],
                    "adj_close": close[i],
                    "open": open_[i],
                    "high": max(open_[i], close[i]) * 1.02,
                    "low": min(open_[i], close[i]) * 0.98,
                    "amount": 1e9,
                    "pre_close": previous[i],
                    "pct_chg": (close[i] / previous[i] - 1.0) * 100.0,
                    "limit_flag": 0,
                    "return_1d": close[i] / previous[i] - 1.0,
                    "bb_upper_20": close[i] * 1.05,
                    "bb_lower_20": close[i] * 0.95,
                    "rsi_14": 50.0,
                    C.TARGET: float(rng.normal(0.0, 0.02)),
                    C.WEIGHT: 1.0,
                    "feature_00": float(rng.normal()),
                    "feature_01": float(rng.normal()),
                }
            )
        previous = close
    return pd.DataFrame(rows)


def _params() -> dict:
    return {
        "rebalance_every": 10,
        "topk": 0.25,
        "max_positions": 5,
        "max_weight": 0.2,
        "min_notional": 1000,
        "warmup_days": 20,
        "n_seeds": 1,
        "train_window": 20,
        "ic_window": 5,
        "use_bb_timing": False,
        "order_price_mode": "next_open",
        "use_vol_target": False,
        "use_dd_brake": False,
        "use_ic_scale": False,
        "use_regime": False,
        "use_iv_scale": False,
        "strict_check": True,
    }


def _score_cache(panel: pd.DataFrame, feat_cols: list[str], params: dict) -> dict:
    scores = {}
    base = np.linspace(1.0, 0.0, panel["symbol"].nunique())
    symbols = sorted(panel["symbol"].unique())
    for day_i, date in enumerate(np.sort(panel["date"].unique())):
        values = base + (0.01 * day_i * np.arange(len(symbols)) % 7)
        scores[date] = dict(zip(symbols, values, strict=True))
    return {"scores": scores, "fingerprint": model_store.fingerprint_of(feat_cols, params)}


def _decision_dates(panel: pd.DataFrame, params: dict) -> np.ndarray:
    dates = np.sort(panel["date"].unique())
    start = int(params["warmup_days"]) + int(C.EMBARGO_DAYS)
    return dates[start:-1]


def _filled_trades(replay: dict) -> pd.DataFrame:
    trades = replay["trades"]
    if trades.empty:
        return trades
    return trades[trades["filled"] & (trades["fill_shares"] > 0)]


def _state_before(panel: pd.DataFrame, feat_cols, params, decision_dates, index: int):
    if index == 0:
        return None
    return run_daily_replay(
        panel,
        feat_cols,
        params=params,
        end_date=decision_dates[index - 1],
        progress=False,
    )["state"]


def _live_orders(panel, feat_cols, params, decision_dates, index: int, risk_path: str):
    date = decision_dates[index]
    today = panel[panel["date"] == date]
    state = _state_before(panel, feat_cols, params, decision_dates, index)
    current = {} if state is None else {s: sh for s, sh in state.holdings.items() if sh > 0}
    cash = float(C.INITIAL_CAPITAL if state is None else state.cash)
    scores = params["score_cache"]["scores"][date]
    return make_orders_from_state(
        today,
        today["symbol"].map(scores).to_numpy(dtype=float),
        current,
        float(C.INITIAL_CAPITAL),
        params=params,
        risk_state_path=risk_path,
        panel=panel,
        cash=cash,
    )


def test_rebalance_days_live_orders_contain_replay_buys(tmp_path):
    panel = _make_panel()
    feat_cols = ["feature_00", "feature_01"]
    params = _params()
    params["score_cache"] = _score_cache(panel, feat_cols, params)
    decision_dates = _decision_dates(panel, params)
    replay = run_daily_replay(panel, feat_cols, params=params, progress=False)
    fills = _filled_trades(replay)
    all_dates = np.sort(panel["date"].unique())
    risk_path = str(tmp_path / "live_risk_state.json")
    checked_rebalances = 0

    for k, date in enumerate(decision_dates):
        orders = _live_orders(panel, feat_cols, params, decision_dates, k, risk_path)
        next_date = all_dates[np.searchsorted(all_dates, date) + 1]
        replay_day = fills[fills["date"] == next_date]
        if k % params["rebalance_every"] != 0:
            assert orders.empty
            assert replay_day.empty
            continue

        checked_rebalances += 1
        assert orders.empty == replay_day.empty
        live_buys = set(orders.loc[orders["side"] == "buy", "symbol"])
        replay_buys = set(replay_day.loc[replay_day["side"] == "buy", "symbol"])
        assert replay_buys <= live_buys
        if not orders.empty:
            save_rebalance_state(
                rebalance_state_path_for(risk_path), str(pd.Timestamp(date).date())
            )

    assert checked_rebalances >= 2


def test_non_rebalance_day_has_no_orders_or_fills(tmp_path):
    panel = _make_panel()
    feat_cols = ["feature_00", "feature_01"]
    params = _params()
    params["score_cache"] = _score_cache(panel, feat_cols, params)
    decision_dates = _decision_dates(panel, params)
    index = 1
    date = decision_dates[index]
    replay = run_daily_replay(panel, feat_cols, params=params, end_date=date, progress=False)
    fills = _filled_trades(replay)
    all_dates = np.sort(panel["date"].unique())
    next_date = all_dates[np.searchsorted(all_dates, date) + 1]
    assert fills[fills["date"] == next_date].empty

    risk_path = str(tmp_path / "live_risk_state.json")
    save_rebalance_state(
        rebalance_state_path_for(risk_path), str(pd.Timestamp(decision_dates[0]).date())
    )
    orders = _live_orders(panel, feat_cols, params, decision_dates, index, risk_path)
    assert orders.empty
