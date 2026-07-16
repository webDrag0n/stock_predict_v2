"""Guotai Junan 191 alpha factors — curated OHLCV subset.

GTJA191 is a collection of 191 cross-sectional alpha expressions published by
Guotai Junan Securities, widely used in Chinese equity quant research. The
original DolphinDB implementation is at:
  https://github.com/dolphindb/DolphinDBModules/blob/master/gtja191Alpha/src/gtja191Alpha.dos

This module implements a curated, high-value subset that:
  - uses only OHLCV (plus a VWAP proxy)
  - does not require benchmark/index inputs
  - avoids overlap with existing technical / Alpha158 / orthogonal factors
  - has a documented empirical track record in A-share cross-sectional prediction

All columns are prefixed with ``gtja_`` and exposed through
``compute_gtja191_factors()``.

Note on VWAP: the original formulas use ``vwap``. We proxy it as
``(high + low + close) / 3`` when the ``amount`` column is unavailable; otherwise
we use ``amount / volume``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._base import (
    cs_rank, ts_corr, ts_delta, ts_delay, ts_max, ts_mean, ts_min, ts_std,
    ts_rank, ts_sum, signed_power, safe_div,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# -----------------------------------------------------------------------------
# Default curated subset. Override via config env var GTJA191_ALPHA_SUBSET.
# -----------------------------------------------------------------------------
DEFAULT_SUBSET = [
    "alpha001", "alpha002", "alpha003", "alpha005", "alpha006",
    "alpha007", "alpha008", "alpha010", "alpha012", "alpha014",
    "alpha017", "alpha020", "alpha021", "alpha023", "alpha024",
    "alpha028", "alpha029", "alpha032", "alpha034", "alpha041",
    "alpha045", "alpha054", "alpha060", "alpha101", "alpha102",
]


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Best-effort VWAP proxy aligned to df index."""
    if "amount" in df.columns:
        return (df["amount"] / df["volume"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan)
    return (df["high"] + df["low"] + df["close"]) / 3.0


def _alpha001(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6)
    a = cs_rank(ts_delta(np.log(v), 1))
    b = cs_rank((c - o) / o)
    return -ts_corr(a, b, 6)


def _alpha002(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1)
    return -ts_delta(((c - l) - (h - c)) / (h - l).replace(0, np.nan), 1)


def _alpha003(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # SUM( IF close==delay(close,1) THEN 0
    #      ELSE close - IF close>delay(close,1) THEN min(low,delay(close,1))
    #                                      ELSE max(high,delay(close,1)) , 6)
    delay_c = ts_delay(c, 1)
    cond_up = c > delay_c
    cond_down = c < delay_c
    ref = pd.Series(np.nan, index=c.index)
    ref[cond_up] = np.minimum(l[cond_up], delay_c[cond_up])
    ref[cond_down] = np.maximum(h[cond_down], delay_c[cond_down])
    val = c - ref
    val[~(cond_up | cond_down)] = 0.0
    return ts_sum(val, 6)


def _alpha005(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * TSMAX( CORR( TSRANK(VOLUME,5), TSRANK(HIGH,5), 5 ), 3 )
    return -ts_max(ts_corr(ts_rank(v, 5), ts_rank(h, 5), 5), 3)


def _alpha006(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * DELTA( OPEN*0.85 + HIGH*0.15, 4 )
    return -ts_delta(o * 0.85 + h * 0.15, 4)


def _alpha007(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * ( RANK(MAX(VWAP-CLOSE,3)) ^ TSRANK(CORR(TSRANK(VOLUME,5),TSRANK(LOW,5),5),3) )
    vp = _vwap(df)
    a = cs_rank(ts_max(vp - c, 3))
    b = ts_rank(ts_corr(ts_rank(v, 5), ts_rank(l, 5), 5), 3)
    return -(a ** b)


def _alpha008(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * ( RANK(DELTA(VWAP,1)) ^ TSRANK(CORR(CLOSE,VWAP,4),3) )
    vp = _vwap(df)
    a = cs_rank(ts_delta(vp, 1))
    b = ts_rank(ts_corr(c, vp, 4), 3)
    return -(signed_power(a, b))


def _alpha010(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # RANK( MAX( ((RET < 0) ? STD(RET, 20) : CLOSE)^2 ), 5 )
    ret = c / ts_delay(c, 1) - 1
    cond = ret < 0
    base = c.copy()
    base[cond] = ts_std(ret, 20)[cond]
    return cs_rank(ts_max(base ** 2, 5))


def _alpha012(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * ( RANK(OPEN) - (RANK(VOLUME) * (RANK(HIGH) - RANK(LOW))) )
    return -(cs_rank(o) - cs_rank(v) * (cs_rank(h) - cs_rank(l)))


def _alpha014(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # CLOSE - DELAY(CLOSE, 5)
    return c - ts_delay(c, 5)


def _alpha017(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # RANK(VWAP - MAX(VWAP,15)) ^ DELTA(CLOSE,5)
    vp = _vwap(df)
    a = cs_rank(vp - ts_max(vp, 15))
    b = ts_delta(c, 5)
    return signed_power(a, b)


def _alpha020(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * (OPEN - CLOSE) / OPEN
    return -(o - c) / o.replace(0, np.nan)


def _alpha021(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * ( RANK(CLOSE - MAX(HIGH,8) + MIN(LOW,8)) ^ TSRANK(VOLUME/MEAN(VOLUME,20),1) )
    a = cs_rank(c - ts_max(h, 8) + ts_min(l, 8))
    b = ts_rank(v / ts_mean(v, 20), 1)
    return -(signed_power(a, b))


def _alpha023(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * SMA((TSMAX(HIGH,20)-CLOSE)/(TSMAX(HIGH,20)-TSMIN(LOW,20))*100,13,1)
    num = ts_max(h, 20) - c
    den = ts_max(h, 20) - ts_min(l, 20)
    raw = safe_div(num, den) * 100
    return -ts_mean(raw, 13)   # SMA approximated by rolling mean


def _alpha024(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * (CLOSE - DELAY(CLOSE,5))
    return -(c - ts_delay(c, 5))


def _alpha028(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    #   - 2*SMA(SMA(...),3,1)
    den = ts_max(h, 9) - ts_min(l, 9)
    raw = safe_div(c - ts_min(l, 9), den) * 100
    s1 = ts_mean(raw, 3)
    s2 = ts_mean(s1, 3)
    return 3 * s1 - 2 * s2


def _alpha029(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * (TSMIN(LOW,9)-CLOSE)/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100
    den = ts_max(h, 9) - ts_min(l, 9)
    return -(ts_min(l, 9) - c) / den.replace(0, np.nan) * 100


def _alpha032(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * RANK(SUM(CORR(RANK(VOLUME),RANK(VWAP),5),5))
    vp = _vwap(df)
    corr = ts_corr(cs_rank(v), cs_rank(vp), 5)
    return -cs_rank(ts_sum(corr, 5))


def _alpha034(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * RANK(CORR(CLOSE/OPEN, VOLUME, 3))
    return -cs_rank(ts_corr(c / o.replace(0, np.nan), v, 3))


def _alpha041(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * ( RANK(MAX(DELTA(VWAP,3),5)) ^ TSRANK(CORR(CLOSE*0.35+OPEN*0.65,VOLUME,18),18) )
    vp = _vwap(df)
    a = cs_rank(ts_max(ts_delta(vp, 3), 5))
    b = ts_rank(ts_corr(c * 0.35 + o * 0.65, v, 18), 18)
    return -(signed_power(a, b))


def _alpha045(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * ( RANK(DELTA(CLOSE*0.6+OPEN*0.4,1)) ^ RANK(CORR(VWAP,MEAN(VOLUME,150),15)) )
    vp = _vwap(df)
    a = cs_rank(ts_delta(c * 0.6 + o * 0.4, 1))
    b = cs_rank(ts_corr(vp, ts_mean(v, 150), 15))
    return -(signed_power(a, b))


def _alpha054(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * (LOW - CLOSE) / (LOW - HIGH) ^ VOLUME
    den = (l - h).replace(0, np.nan)
    base = (l - c) / den
    return -(signed_power(base, v))


def _alpha060(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * (2*CLOSE - LOW - HIGH) / (HIGH - LOW) * VOLUME
    den = (h - l).replace(0, np.nan)
    return -(2 * c - l - h) / den * v


def _alpha101(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # -1 * (RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30),37),15))^RANK(CORR(RANK(HIGH),RANK(VOLUME),7)))
    a = cs_rank(ts_corr(c, ts_sum(ts_mean(v, 30), 37), 15))
    b = cs_rank(ts_corr(cs_rank(h), cs_rank(v), 7))
    return -(signed_power(a, b))


def _alpha102(df: pd.DataFrame, o, h, l, c, v) -> pd.Series:
    # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1) / SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1) * 100
    dv = ts_delta(v, 1)
    pos = ts_mean(dv.clip(lower=0), 6)
    ab = ts_mean(dv.abs(), 6)
    return safe_div(pos, ab) * 100


_ALPHAS = {
    "alpha001": _alpha001, "alpha002": _alpha002, "alpha003": _alpha003,
    "alpha005": _alpha005, "alpha006": _alpha006, "alpha007": _alpha007,
    "alpha008": _alpha008, "alpha010": _alpha010, "alpha012": _alpha012,
    "alpha014": _alpha014, "alpha017": _alpha017, "alpha020": _alpha020,
    "alpha021": _alpha021, "alpha023": _alpha023, "alpha024": _alpha024,
    "alpha028": _alpha028, "alpha029": _alpha029, "alpha032": _alpha032,
    "alpha034": _alpha034, "alpha041": _alpha041, "alpha045": _alpha045,
    "alpha054": _alpha054, "alpha060": _alpha060, "alpha101": _alpha101,
    "alpha102": _alpha102,
}


def compute_gtja191_factors(df: pd.DataFrame,
                            subset: list[str] | None = None) -> pd.DataFrame:
    """Compute a curated subset of GTJA191 alpha factors.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV bars in long format, sorted by (symbol, date).
    subset : list[str] | None
        Alpha IDs to compute. If None, uses the default curated subset.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new ``gtja_alphaNNN`` columns appended.
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    idx = df.set_index(["symbol", "date"])
    # Use float64 internally so intermediate rolling/corr operations do not hit
    # LossySetitemError when the input DataFrame has been downcast to float32.
    o = idx["open"].astype(np.float64)
    h = idx["high"].astype(np.float64)
    l = idx["low"].astype(np.float64)
    c = idx["close"].astype(np.float64)
    v = idx["volume"].astype(np.float64)

    ids = subset or DEFAULT_SUBSET
    new_cols = {}
    for aid in ids:
        fn = _ALPHAS.get(aid)
        if fn is None:
            continue
        s = fn(idx, o, h, l, c, v)
        # Clean up numerical artifacts (inf / -inf) that rolling corr can emit
        # when a trailing window has near-zero variance, then store as float32
        # to keep whole-market memory under control.
        s = s.replace([np.inf, -np.inf], np.nan).astype(np.float32)
        new_cols[f"gtja_{aid}"] = s

    if new_cols:
        df = pd.concat([df, pd.DataFrame(
            {k: v.reset_index(drop=True) for k, v in new_cols.items()},
            index=df.index)], axis=1)
    return df
