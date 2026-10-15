"""Seed-rank ensemble optimization on the cached whole-market OOS.

The 3 robustness seeds (42/123/7) correlate ~0.977 — near-identical GBM fits.
This tests whether averaging their PER-DATE PERCENTILE RANKS before the
long_topk selection buys any tradeable improvement (variance reduction) over a
single seed, measured through the SAME path-dependent friction backtest used by
realistic_cost.py. Cheap: reuses cached predictions, no retrain.

Compares, at cost level L0 (comm+stamp) and L4 (full real friction):
  - each single seed
  - seed-mean of curves (current "3-seed" reporting)
  - rank-ensemble (mean of per-date pctile ranks → one combined predictor)
  - mean-ensemble (mean of raw predictions)
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts", "research"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realistic_cost import (_load_oos, _positions_long_topk, _backtest,
                            _metrics, SEEDS, TOPK, REBAL)

def _build_ensemble(raw: list[pd.DataFrame], how: str) -> pd.DataFrame:
    """Combine seeds into one OOS frame. `how`='rank' averages per-date pctile
    ranks of prediction; 'mean' averages raw predictions. Grid is identical
    across seeds (verified), so we align by row order after a common sort."""
    base = raw[0].sort_values(["date", "symbol"]).reset_index(drop=True)
    carry = ["date", "symbol", "responder_6", "return_1d",
             "limit_flag", "pct_chg", "amount"]
    out = base[[c for c in carry if c in base.columns]].copy()
    if how == "mean":
        preds = np.mean([df.sort_values(["date", "symbol"])
                         ["prediction"].to_numpy() for df in raw], axis=0)
        out["prediction"] = preds
        return out
    # rank: per-date percentile rank each seed, then average the ranks
    rank_sum = np.zeros(len(base), dtype=np.float64)
    for df in raw:
        d = df.sort_values(["date", "symbol"]).reset_index(drop=True)
        d = d.assign(_p=d["prediction"])
        rk = d.groupby("date")["_p"].rank(pct=True, method="average").to_numpy()
        rank_sum += rk
    out["prediction"] = rank_sum / len(raw)
    return out


def main() -> None:
    print(f"Seed-rank ensemble 优化 | ret_weighted 全市场 | topk={TOPK} "
          f"rebal={REBAL}d | seeds={list(SEEDS)}\n")
    raw = [_load_oos(s) for s in SEEDS]

    variants: dict[str, pd.DataFrame] = {f"seed{s}": r for s, r in zip(SEEDS, raw)}
    variants["ens_rank"] = _build_ensemble(raw, "rank")
    variants["ens_mean"] = _build_ensemble(raw, "mean")

    pos = {name: _positions_long_topk(df, TOPK) for name, df in variants.items()}

    rows = []
    for lvl in (0, 4):
        for name, p in pos.items():
            m = _backtest(p, lvl)
            rows.append({"variant": name, "level": f"L{lvl}", **m})
            print(f"L{lvl} {name:9s} α={m['alpha_ann']*100:+6.2f}% "
                  f"NW-t={m['nw_t']:+.2f} Sharpe={m['sharpe']:+.2f} "
                  f"累计={m['cum']*100:+6.1f}% MaxDD={m['maxdd']*100:.1f}%")
        # seed-mean-of-curves at this level (current reporting convention)
        series = [_backtest(pos[f"seed{s}"], lvl, return_series=True) for s in SEEDS]
        net = np.mean([s["net_series"] for s in series], axis=0)
        bench = series[0]["bench"]
        dates = series[0]["dates"]
        m = _metrics(net, bench, dates, {"comm": 0, "impact": 0, "spread": 0})
        rows.append({"variant": "seed_meancurve", "level": f"L{lvl}", **m})
        print(f"L{lvl} {'meancurve':9s} α={m['alpha_ann']*100:+6.2f}% "
              f"NW-t={m['nw_t']:+.2f} Sharpe={m['sharpe']:+.2f} "
              f"累计={m['cum']*100:+6.1f}% MaxDD={m['maxdd']*100:.1f}%")
        print()

    out = pd.DataFrame(rows)
    out.to_csv("outputs/ensemble_opt.csv", index=False)
    print("-> outputs/ensemble_opt.csv")


if __name__ == "__main__":
    main()
