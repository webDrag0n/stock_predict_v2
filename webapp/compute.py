"""Fast compute layer for live interactivity.

Wraps the existing periodic backtest (src/backtest.py) so the Backtest Lab can
re-run a multi-seed backtest in seconds when the user moves a slider. Nothing
here trains a model — it operates entirely on the cached per-seed OOS predictions
plus the amount map, so every call is cheap.
"""
from __future__ import annotations

import os
import sys
import functools

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as C
from walkforward import _with_ids
from strategy import generate_positions
from backtest import run_backtest_periodic
from measure_alpha import _newey_west_alpha_t

from . import dataio


@functools.lru_cache(maxsize=64)
def _oos_merged(seed: int) -> pd.DataFrame:
    """Seed OOS joined with the amount column (for the impact cost). Cached."""
    oos = dataio.seed_oos(seed)
    if oos.empty:
        return oos
    amt = dataio.amount_map()
    if not amt.empty:
        oos = oos.merge(amt, on=["date", "symbol"], how="left")
    return oos


def seed_backtest(seed: int, topk: float, rebalance_days: int,
                  impact_k: float, use_impact: bool) -> dict:
    """Run one seed's periodic backtest with the given knobs. Returns curves+metrics.

    Cheap: operates on cached OOS predictions. Overrides the relevant config knobs
    for the duration of the call (process-local; the app is single-user).
    """
    oos = _oos_merged(seed)
    if oos.empty:
        return {}

    # apply knobs to config (read by strategy/backtest)
    C.TOPK_RATIO = topk
    C.HOLD_RATIO = topk                 # buffer off (periodic rebalance)
    C.REBALANCE_DAYS = rebalance_days
    C.USE_IMPACT_COST = bool(use_impact)
    C.IMPACT_K = impact_k

    oos_ids = _with_ids(oos)
    pred = oos["prediction"].to_numpy()
    pos = generate_positions(oos_ids, pred, strategy="long_topk")
    if "amount" in oos.columns:
        oos_ids["amount"] = oos["amount"].to_numpy()
    res = run_backtest_periodic(oos_ids, pred, pos,
                                rebalance_days=rebalance_days,
                                pred_in_return_space=True)
    curve = res["curve"]; m = res["metrics"]
    net = curve["net"].to_numpy()
    bench = oos.groupby("date")["return_1d"].mean().sort_index().to_numpy()
    n = min(len(net), len(bench))
    a_d, beta, t_nw, r2 = _newey_west_alpha_t(net[:n], bench[:n], rebalance_days)
    strat_cum = np.cumprod(1 + net) - 1.0
    bench_cum = np.cumprod(1 + bench[:n]) - 1.0
    return {
        "dates": np.sort(oos["date"].unique()),
        "net": net,
        "strat_cum_pct": strat_cum * 100,
        "bench_cum_pct": bench_cum * 100,
        "alpha_pct": (strat_cum[:n] - bench_cum) * 100,
        "tc_comm": curve["tc_commission"].to_numpy(),
        "tc_impact": curve["tc_impact"].to_numpy(),
        "sharpe": m["sharpe"], "calmar": m["calmar"], "maxdd": m["max_drawdown"],
        "alpha_ann": a_d * 252, "nw_t": t_nw, "beta": beta,
        "total_pnl": m["total_pnl"], "hit": m["hit_rate"],
        "cost_comm": m["total_cost_commission"], "cost_impact": m["total_cost_impact"],
    }


def multiseed_backtest(seeds: tuple[int, ...], topk: float, rebalance_days: int,
                       impact_k: float, use_impact: bool) -> dict:
    """Run the backtest across seeds; return per-seed dicts + aligned mean arrays."""
    runs = {}
    for sd in seeds:
        r = seed_backtest(sd, topk, rebalance_days, impact_k, use_impact)
        if r:
            runs[sd] = r
    if not runs:
        return {"runs": {}}

    def stack(key):
        arrs = [r[key] for r in runs.values()]
        L = min(len(a) for a in arrs)
        return np.vstack([a[:L] for a in arrs])

    net_s = stack("net")
    agg = {
        "runs": runs,
        "seeds": list(runs.keys()),
        "mean_net": net_s.mean(axis=0),
        "alpha_s": stack("alpha_pct"),
        "bench_cum_pct": stack("bench_cum_pct").mean(axis=0),
        "tc_comm_mean": stack("tc_comm").mean(axis=0),
        "tc_impact_mean": stack("tc_impact").mean(axis=0),
        "T": net_s.shape[1],
    }
    # scalar KPI mean±std
    for k in ("sharpe", "alpha_ann", "nw_t", "beta", "calmar", "maxdd", "hit",
              "cost_comm", "cost_impact"):
        v = np.array([runs[s][k] for s in runs])
        agg[f"{k}_mean"] = float(v.mean())
        agg[f"{k}_std"] = float(v.std())
    return agg
