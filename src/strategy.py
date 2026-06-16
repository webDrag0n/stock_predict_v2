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
    """Long the top `ratio` fraction of predictions each day, equal weight.

    Positions are >= 0 (long-only), and sum to 1 within each date (fully
    invested across the selected names). This is the tradeable A-share book.
    """
    tmp = pd.DataFrame({"date": df[C.DATE_COL].to_numpy(), "pred": pred})
    # Per-date percentile rank; select the top `ratio`.
    rank = tmp.groupby("date")["pred"].rank(pct=True, method="first")
    selected = (rank > (1.0 - ratio)).to_numpy()
    pos = selected.astype(np.float64)
    # Normalize so each date's longs sum to GROSS_LEVERAGE (equal weight).
    sums = pd.Series(pos).groupby(tmp["date"]).transform("sum").to_numpy()
    pos = np.where(sums > 0, C.GROSS_LEVERAGE * pos / sums, 0.0)
    return pos


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


# Strategies that are practically tradeable for A-share retail (long-only).
TRADEABLE = ("long_topk",)
# All strategies the pipeline can evaluate.
ALL_STRATEGIES = ("long_topk", "tanh", "cross_sectional")


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
    raise ValueError(f"Unknown strategy: {strategy}")
