"""Strategy: convert predicted responder_6 into tradeable positions.

Goal: maximize realized return. The predicted value is a forward-return proxy,
so position should grow with the prediction. Available sizing schemes:

  long_topk        : A-SHARE DEFAULT (tradeable for retail). Each day, go long
                     the top-K%% predicted names, equal weight, gross exposure
                     normalized to 1. No shorting — fits A-share constraints.
  tanh             : smooth directional sizing, position = tanh(scale * pred),
                     clipped to [-1, 1]. Can go short (research / futures only).
  cross_sectional  : per-day dollar-neutral long/short. Demean predictions within
                     each date, size proportional to the demeaned value, normalize
                     so gross exposure == GROSS_LEVERAGE. RESEARCH REFERENCE ONLY —
                     retail A-share investors cannot short single names, so this is
                     a diagnostic of raw signal quality, not a tradeable book.

Positions are returned in the same row order as the input dataframe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _long_topk(df: pd.DataFrame, pred: np.ndarray, ratio: float) -> np.ndarray:
    """Long the top `ratio` fraction each day, weighted by prediction rank.

    Rank-proportional: within the selected stocks, weight ∝ (rank among selected)
    so the highest-predicted stock gets the most weight. Sums to GROSS_LEVERAGE.
    Captures more signal than equal-weight — stocks ranked #1 matter more.

    When C.HOLD_RATIO > ratio, a hysteresis buffer is applied: a held name is
    kept until its rank drops out of the top HOLD_RATIO band (not merely out of
    top `ratio`), and only names entering the top `ratio` band are newly bought.
    The no-trade zone between the two bands suppresses churn from rank noise —
    the dominant transaction-cost driver — while keeping look-ahead-free (today's
    holding set uses only today's ranks and yesterday's holdings).
    """
    hold_ratio = getattr(C, "HOLD_RATIO", ratio)
    tmp = pd.DataFrame({"date": df[C.DATE_COL].to_numpy(), "pred": pred})
    rank_pct = tmp.groupby("date")["pred"].rank(pct=True, method="first")
    tmp["rank_pct"] = rank_pct

    if hold_ratio <= ratio + 1e-12:
        selected = (rank_pct > (1.0 - ratio)).to_numpy()
    else:
        # Chronological hysteresis: carry holdings forward across dates.
        enter = (1.0 - ratio)        # buy threshold (strict top band)
        keep = (1.0 - hold_ratio)    # sell threshold (wider hold band)
        sym = (df[C.SYMBOL_COL].to_numpy() if C.SYMBOL_COL in df.columns
               else np.arange(len(df)))
        selected = np.zeros(len(df), dtype=bool)
        rp = rank_pct.to_numpy()
        held: set = set()
        for d in np.sort(tmp["date"].unique()):
            idx = np.where(tmp["date"].to_numpy() == d)[0]
            new_held = set()
            for i in idx:
                s = sym[i]
                if rp[i] > enter or (s in held and rp[i] > keep):
                    selected[i] = True
                    new_held.add(s)
            held = new_held

    tmp["sel"] = selected
    def _rank_weight(g):
        sel = g["sel"]
        if not sel.any():
            return pd.Series(0.0, index=g.index)
        w = g["rank_pct"].where(sel, 0.0)
        total = w.sum()
        return (C.GROSS_LEVERAGE * w / total) if total > 1e-12 else pd.Series(0.0, index=g.index)

    pos = tmp.groupby("date", group_keys=False).apply(
        _rank_weight, include_groups=False)
    return np.asarray(pos.reindex(tmp.index), dtype=np.float64)


def _tanh(pred: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.tanh(scale * pred), -1.0, 1.0)


def _cross_sectional(df: pd.DataFrame, pred: np.ndarray,
                     gross: float) -> np.ndarray:
    """Dollar-neutral sizing within each (date, time): sum(pos)=0, sum|pos|=gross."""
    tmp = pd.DataFrame({
        C.DATE_COL: df[C.DATE_COL].to_numpy(),
        C.TIME_COL: df[C.TIME_COL].to_numpy(),
        "pred": pred,
    })
    grp = tmp.groupby([C.DATE_COL, C.TIME_COL])["pred"]
    demeaned = tmp["pred"] - grp.transform("mean")
    abs_sum = demeaned.abs().groupby([tmp[C.DATE_COL], tmp[C.TIME_COL]]).transform("sum")
    pos = np.where(abs_sum > 1e-12, gross * demeaned / abs_sum, 0.0)
    return np.asarray(pos, dtype=np.float64)


def _long_short_topk(df: pd.DataFrame, pred: np.ndarray, ratio: float) -> np.ndarray:
    """Dollar-neutral long top-`ratio`, short bottom-`ratio` each date (RESEARCH).

    Goes long the top `ratio` fraction (+0.5*gross spread equal-weight) and short
    the bottom `ratio` fraction (−0.5*gross), so each date is dollar-neutral
    (sum(pos)=0, sum|pos|=gross). Unlike _cross_sectional (which sizes every name
    by its demeaned prediction), this trades only the extreme tails — directly
    mirroring long_topk but adding the short leg, so it isolates "what does adding
    the short side buy us" cleanly.

    NOT retail-tradeable in A-shares (no single-name shorting). It is a research
    diagnostic of whether the bottom-decile edge (which long-only discards) is
    harvestable, e.g. via 融券 / institutional accounts / index-futures overlays.
    Simulated by signed preset positions; the periodic backtester earns
    weight*daily_return correctly for negative weights, and the A-share cost model
    treats opening a short as a stamp-duty 'sell'. Borrow fees are NOT modeled
    (add ~8-10%/yr on the short notional for a fully realistic 融券 estimate).
    """
    tmp = pd.DataFrame({"date": df[C.DATE_COL].to_numpy(), "pred": pred})
    rank_pct = tmp.groupby("date")["pred"].rank(pct=True, method="first")
    long_sel = (rank_pct > (1.0 - ratio)).to_numpy()
    short_sel = (rank_pct <= ratio).to_numpy()
    tmp["long"], tmp["short"] = long_sel, short_sel

    half = 0.5 * C.GROSS_LEVERAGE
    def _weights(g):
        w = pd.Series(0.0, index=g.index)
        nl, ns = int(g["long"].sum()), int(g["short"].sum())
        if nl > 0:
            w[g["long"]] = half / nl          # equal-weight long leg, +0.5 gross
        if ns > 0:
            w[g["short"]] = -half / ns         # equal-weight short leg, -0.5 gross
        return w
    pos = tmp.groupby("date", group_keys=False).apply(_weights, include_groups=False)
    return np.asarray(pos.reindex(tmp.index), dtype=np.float64)


# Strategies that are practically tradeable for A-share retail (long-only).
TRADEABLE = ("long_topk",)
# All strategies the pipeline can evaluate.
ALL_STRATEGIES = ("long_topk", "tanh", "cross_sectional", "long_short_topk")


def generate_positions(df: pd.DataFrame, pred: np.ndarray,
                       strategy: str = None) -> np.ndarray:
    """Map predictions -> positions (long-only weights, or signed for research)."""
    strategy = strategy or C.STRATEGY
    pred = np.asarray(pred, dtype=np.float64)
    if strategy == "long_topk":
        return _long_topk(df, pred, C.TOPK_RATIO)
    if strategy == "tanh":
        return _tanh(pred, C.TANH_SCALE)
    if strategy == "cross_sectional":
        return _cross_sectional(df, pred, C.GROSS_LEVERAGE)
    if strategy == "long_short_topk":
        return _long_short_topk(df, pred, C.TOPK_RATIO)
    raise ValueError(f"Unknown strategy: {strategy}")
