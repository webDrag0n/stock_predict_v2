"""Synthetic demo of the daily parameter sweep framework.

This does NOT use real market data. It creates a small synthetic panel and runs
optimize_daily_params-style score_cache reuse over a tiny grid, proving the
infrastructure works end-to-end. Use optimize_daily_params.py for real research.
"""
from __future__ import annotations

import os
import sys
import itertools

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from backtest_daily import run_daily_replay
from measure_alpha import _newey_west_alpha_t


def _make_panel(n_dates: int = 80):
    dates = pd.date_range("2026-05-01", periods=n_dates, freq="B")
    rows = []
    np.random.seed(42)
    for d in dates:
        for sym in [f"{c:06d}.SH" if i % 2 == 0 else f"{c:06d}.SZ"
                    for i, c in enumerate(range(600000, 600010))]:
            close = 10.0 + np.random.randn() * 0.5
            rows.append({
                "date": d,
                "symbol": sym,
                "close": close,
                "adj_close": close,
                "open": close * (1 + np.random.randn() * 0.01),
                "high": close * 1.02,
                "low": close * 0.98,
                "amount": 10000.0,
                "pre_close": close * (1 + np.random.randn() * 0.01),
                "pct_chg": np.random.randn() * 2.0,
                "limit_flag": 0,
                "return_1d": np.random.randn() * 0.02,
                "bb_upper_20": close * 1.05,
                "bb_lower_20": close * 0.95,
                "rsi_14": 50.0,
                C.TARGET: np.random.randn() * 0.02,
                C.WEIGHT: 1.0,
                **{f"feature_{i:02d}": np.random.randn() for i in range(8)},
            })
    return pd.DataFrame(rows)


def _metrics(daily, initial_cash):
    tv = daily["total_value"].to_numpy(dtype=np.float64)
    rets = np.zeros(len(tv))
    rets[1:] = tv[1:] / tv[:-1] - 1.0
    rr = rets[1:]
    sharpe = float(rr.mean() / rr.std() * np.sqrt(252)) if rr.std() > 1e-12 else 0.0
    max_dd = float((tv / np.maximum.accumulate(tv) - 1.0).min())
    return {"total_ret": tv[-1] / initial_cash - 1.0, "sharpe": sharpe, "max_dd": max_dd}


def main():
    print("Synthetic demo of daily parameter sweep\n")
    panel = _make_panel(80)
    feat_cols = [c for c in panel.columns if c.startswith("feature_")]
    initial_cash = 1_000_000.0

    base_params = {
        "n_seeds": 1,
        "rebuild_days": 20,
        "train_window": 30,
        "ic_window": 5,
        "warmup_days": 30,
        "collect_scores": True,
        "progress": False,
    }
    print("Training baseline + collecting score_cache...")
    base_res = run_daily_replay(panel, feat_cols, initial_cash=initial_cash,
                                params=base_params, progress=False)
    score_cache = base_res["score_cache"]
    print(f"score_cache size: {len(score_cache)} decision days\n")

    grids = {
        "rebalance_every": [5, 10],
        "topk": [0.10, 0.20],
        "max_positions": [5, 10],
    }
    combos = list(itertools.product(*grids.values()))
    print(f"Sweeping {len(combos)} combos: {list(grids.keys())}\n")

    rows = []
    for combo in combos:
        p = dict(base_params)
        p["score_cache"] = score_cache
        p["collect_scores"] = False
        p.update(dict(zip(grids.keys(), combo)))
        p["use_bb_timing"] = False
        try:
            res = run_daily_replay(panel, feat_cols, initial_cash=initial_cash,
                                   params=p, progress=False)
            m = _metrics(res["daily"], initial_cash)
            row = dict(zip(grids.keys(), combo))
            row.update(m)
            rows.append(row)
            print(f"  {combo} -> ret={m['total_ret']:+.2%} sharpe={m['sharpe']:+.2f} maxdd={m['max_dd']:+.1%}")
        except Exception as e:
            print(f"  {combo} -> FAILED: {e}")

    tbl = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print("\nTop results by Sharpe:")
    print(tbl.to_string(index=False))
    print("\nDemo complete. For real research run: optimize_daily_params.py")


if __name__ == "__main__":
    main()
