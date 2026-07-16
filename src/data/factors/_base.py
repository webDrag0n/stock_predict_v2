"""Shared factor-engineering primitives for the project's unified factor library.

The helpers here mirror the DolphinDB primitives used in GTJA191 (delay, tsrank,
correlation, signed power, etc.) and provide a consistent, vectorized API for all
factor families.

Assumption: the input Series has a MultiIndex (symbol, date) or a single-level
index that represents the symbol. Cross-sectional helpers group by the date
level; time-series helpers group by the symbol level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _symbol_level(s: pd.Series) -> int | str:
    """Return the index level name/position to group by symbol."""
    return 0 if isinstance(s.index, pd.MultiIndex) else s.index.name


def _date_level(s: pd.Series) -> int | str:
    """Return the index level name/position to group by date."""
    if isinstance(s.index, pd.MultiIndex):
        return 1
    raise ValueError("Cross-sectional helpers require a (symbol, date) MultiIndex")


def ts_delay(s: pd.Series, n: int) -> pd.Series:
    """Lag a per-symbol series by n trading days."""
    return s.groupby(level=_symbol_level(s), group_keys=False).shift(n)


def ts_delta(s: pd.Series, n: int) -> pd.Series:
    """s - delay(s, n)."""
    return s - ts_delay(s, n)


def ts_sum(s: pd.Series, n: int) -> pd.Series:
    """Rolling sum over n days per symbol."""
    return s.groupby(level=_symbol_level(s), group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).sum())


def ts_mean(s: pd.Series, n: int) -> pd.Series:
    """Rolling mean over n days per symbol."""
    return s.groupby(level=_symbol_level(s), group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).mean())


def ts_std(s: pd.Series, n: int) -> pd.Series:
    """Rolling std over n days per symbol."""
    return s.groupby(level=_symbol_level(s), group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).std())


def ts_max(s: pd.Series, n: int) -> pd.Series:
    """Rolling max over n days per symbol."""
    return s.groupby(level=_symbol_level(s), group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).max())


def ts_min(s: pd.Series, n: int) -> pd.Series:
    """Rolling min over n days per symbol."""
    return s.groupby(level=_symbol_level(s), group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).min())


def ts_skew(s: pd.Series, n: int) -> pd.Series:
    """Rolling skew over n days per symbol."""
    return s.groupby(level=_symbol_level(s), group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).skew())


def ts_argmax(s: pd.Series, n: int) -> pd.Series:
    """Index of the rolling max (0..n-1) per symbol."""
    return s.groupby(level=_symbol_level(s), group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).apply(np.argmax, raw=True))


def ts_argmin(s: pd.Series, n: int) -> pd.Series:
    """Index of the rolling min (0..n-1) per symbol."""
    return s.groupby(level=_symbol_level(s), group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).apply(np.argmin, raw=True))


def ts_corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    """Rolling Pearson correlation of two aligned per-symbol series."""
    sym = _symbol_level(a)
    min_p = max(2, n // 2)

    def _roll_sum(s: pd.Series) -> pd.Series:
        return s.groupby(level=sym, group_keys=False).transform(
            lambda x: x.rolling(n, min_periods=min_p).sum())

    m = a.groupby(level=sym, group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=min_p).count())
    sa = _roll_sum(a)
    sb = _roll_sum(b)
    sab = _roll_sum(a * b)
    sa2 = _roll_sum(a * a)
    sb2 = _roll_sum(b * b)

    num = m * sab - sa * sb
    den = np.sqrt(np.maximum((m * sa2 - sa ** 2) * (m * sb2 - sb ** 2), 0.0))
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def ts_rank(s: pd.Series, n: int) -> pd.Series:
    """Time-series rank of the current value within the trailing n days (1..n)."""
    def _rank(g):
        # pandas rolling.rank() is Cythonized and ~100x faster than scipy rankdata
        # applied per window. The value at each point is its rank within the window.
        return g.rolling(n, min_periods=max(1, n // 2)).rank()
    return s.groupby(level=_symbol_level(s), group_keys=False).transform(_rank)


def cs_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional rank within each date, normalized to (0, 1]."""
    return s.groupby(level=_date_level(s), group_keys=False).rank(pct=True)


def cs_zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score within each date."""
    def _z(g):
        m = g.mean()
        sd = g.std()
        return (g - m) / sd if sd > 1e-12 else pd.Series(np.nan, index=g.index)
    return s.groupby(level=_date_level(s), group_keys=False).transform(_z)


def signed_power(s: pd.Series, e: float) -> pd.Series:
    """sign(s) * |s|^e, preserving the sign of the original value.

    Clips the result to the float32 range and replaces overflowed/infinite
    values with NaN so downstream casts do not raise overflow warnings.
    """
    raw = np.sign(s) * (np.abs(s) ** e)
    max_f32 = np.finfo(np.float32).max
    out = np.where(np.isfinite(raw), np.clip(raw, -max_f32, max_f32), np.nan)
    return pd.Series(out, index=s.index)


def safe_div(a: pd.Series, b: pd.Series, fill: float = np.nan) -> pd.Series:
    """Element-wise a / b with protection against zero/NaN denominators."""
    denom = b.replace(0, np.nan)
    out = a / denom
    out[~np.isfinite(out)] = fill
    return out


def ts_quantile(s: pd.Series, n: int, q: float) -> pd.Series:
    """Rolling q-quantile per symbol."""
    return s.groupby(level=_symbol_level(s), group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).quantile(q))


def ts_reg_resid(s: pd.Series, n: int) -> pd.Series:
    """Residual of the LAST point vs a linear time-trend fit over the window."""
    def _resid(y: np.ndarray) -> float:
        m = len(y)
        if m < 3 or not np.all(np.isfinite(y)):
            return np.nan
        x = np.arange(m, dtype=np.float64)
        b, a = np.polyfit(x, y, 1)
        return float(y[-1] - (b * x[-1] + a))
    return s.groupby(level=_symbol_level(s), group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).apply(_resid, raw=True))


def ts_reg_r2(s: pd.Series, n: int) -> pd.Series:
    """R² of y regressed on a time index 0..n-1 (for trend quality)."""
    def _r2(y: np.ndarray) -> float:
        m = len(y)
        if m < 3 or not np.all(np.isfinite(y)):
            return np.nan
        x = np.arange(m, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        sxx = np.sum((x - xm) ** 2)
        syy = np.sum((y - ym) ** 2)
        if sxx < 1e-12 or syy < 1e-12:
            return np.nan
        sxy = np.sum((x - xm) * (y - ym))
        return (sxy * sxy) / (sxx * syy)
    return s.groupby(level=_symbol_level(s), group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).apply(_r2, raw=True))
