"""Numerical-equivalence tests for the vectorized Alpha158 rolling factors.

`reference_compute_alpha158_factors` below is a verbatim copy of the pre-vectorization
implementation (rolling.apply Python callbacks). The tests freeze that behaviour and
require the vectorized `compute_alpha158_factors` to match it column-by-column
(max abs diff < 1e-5, identical NaN patterns) on synthetic multi-symbol OHLCV data
that includes NaNs, short histories and flat (zero-variance) stretches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.factors.alpha158 import compute_alpha158_factors

# --------------------------------------------------------------------------
# Reference (pre-vectorization) implementation — kept for behavioural parity.
# --------------------------------------------------------------------------


def _ref_rolling_r2(y: np.ndarray) -> float:
    n = len(y)
    if n < 3 or np.any(~np.isfinite(y)):
        return np.nan
    x = np.arange(n, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    syy = np.sum((y - ym) ** 2)
    if sxx < 1e-12 or syy < 1e-12:
        return np.nan
    sxy = np.sum((x - xm) * (y - ym))
    return (sxy * sxy) / (sxx * syy)


def _ref_rolling_resid(y: np.ndarray) -> float:
    n = len(y)
    if n < 3 or np.any(~np.isfinite(y)):
        return np.nan
    x = np.arange(n, dtype=np.float64)
    b, a = np.polyfit(x, y, 1)
    return float(y[-1] - (b * x[-1] + a))


def _ref_rolling_corr(df: pd.DataFrame, col_a, series_b, W: int, series_a=None):
    a = df[col_a] if col_a is not None else series_a
    tmp = pd.DataFrame(
        {
            "symbol": df["symbol"].to_numpy(),
            "a": np.asarray(a, dtype=np.float64),
            "b": np.asarray(series_b, dtype=np.float64),
        }
    )
    return tmp.groupby("symbol", group_keys=False).apply(
        lambda gg: gg["a"].rolling(W).corr(gg["b"]), include_groups=False
    )


def reference_compute_alpha158_factors(df: pd.DataFrame, windows=(5, 20, 60)) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    o, h, low, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"]
    hl = (h - low).replace(0, np.nan)
    g = df.groupby("symbol", group_keys=False)

    df["a158_kmid"] = (c - o) / o
    df["a158_klen"] = (h - low) / o
    df["a158_kmid2"] = (c - o) / hl
    df["a158_kup"] = (h - np.maximum(o, c)) / o
    df["a158_klow"] = (np.minimum(o, c) - low) / o
    df["a158_ksft"] = (2 * c - h - low) / o
    df["a158_ksft2"] = (2 * c - h - low) / hl

    ret1 = c / g["close"].shift(1)
    logv = np.log(v + 1.0)
    dlogv = logv - logv.groupby(df["symbol"]).shift(1)
    up = (c > g["close"].shift(1)).astype(float)
    dn = (c < g["close"].shift(1)).astype(float)
    dpos = (c - g["close"].shift(1)).clip(lower=0)
    dneg = (g["close"].shift(1) - c).clip(lower=0)
    dabs = (c - g["close"].shift(1)).abs()
    rv = (c / g["close"].shift(1) - 1.0).abs() * v

    for W in windows:
        cs = df["close"]

        def _rsqr(x):
            return x.rolling(W).apply(_ref_rolling_r2, raw=True)

        def _resi(x):
            return x.rolling(W).apply(_ref_rolling_resid, raw=True)

        df[f"a158_rsqr{W}"] = g["close"].transform(_rsqr)
        df[f"a158_resi{W}"] = g["close"].transform(_resi) / cs

        df[f"a158_qtlu{W}"] = g["close"].transform(lambda x: x.rolling(W).quantile(0.8)) / cs
        df[f"a158_qtld{W}"] = g["close"].transform(lambda x: x.rolling(W).quantile(0.2)) / cs

        min_l = g["low"].transform(lambda x: x.rolling(W).min())
        max_h = g["high"].transform(lambda x: x.rolling(W).max())
        df[f"a158_rsv{W}"] = (cs - min_l) / (max_h - min_l + 1e-12)

        df[f"a158_imax{W}"] = (
            g["high"].transform(lambda x: x.rolling(W).apply(np.argmax, raw=True)) / W
        )
        df[f"a158_imin{W}"] = (
            g["low"].transform(lambda x: x.rolling(W).apply(np.argmin, raw=True)) / W
        )
        df[f"a158_imxd{W}"] = df[f"a158_imax{W}"] - df[f"a158_imin{W}"]

        df[f"a158_corr{W}"] = _ref_rolling_corr(df, "close", logv, W)
        df[f"a158_cord{W}"] = _ref_rolling_corr(df, None, dlogv, W, series_a=ret1)

        df[f"a158_cntp{W}"] = up.groupby(df["symbol"]).transform(lambda x: x.rolling(W).mean())
        df[f"a158_cntn{W}"] = dn.groupby(df["symbol"]).transform(lambda x: x.rolling(W).mean())
        df[f"a158_cntd{W}"] = df[f"a158_cntp{W}"] - df[f"a158_cntn{W}"]

        sp = dpos.groupby(df["symbol"]).transform(lambda x: x.rolling(W).sum())
        sn = dneg.groupby(df["symbol"]).transform(lambda x: x.rolling(W).sum())
        sa = dabs.groupby(df["symbol"]).transform(lambda x: x.rolling(W).sum()) + 1e-12
        df[f"a158_sump{W}"] = sp / sa
        df[f"a158_sumd{W}"] = (sp - sn) / sa

        rv_std = rv.groupby(df["symbol"]).transform(lambda x: x.rolling(W).std())
        rv_mean = rv.groupby(df["symbol"]).transform(lambda x: x.rolling(W).mean()) + 1e-12
        df[f"a158_wvma{W}"] = rv_std / rv_mean

    return df


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------


def _make_symbol_frame(symbol: str, n: int, rng: np.random.Generator, flat: bool) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    rets = rng.normal(0.0005, 0.02, size=n)
    close = 20.0 * np.exp(np.cumsum(rets))
    if flat and n > 80:
        # zero-variance plateau: exercises the syy<1e-12 / zero-variance paths
        close[10:75] = close[10]
    open_ = close * (1 + rng.normal(0, 0.005, size=n))
    spread = np.abs(rng.normal(0.01, 0.004, size=n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    if flat and n > 80:
        open_[10:75] = close[10:75]
        high[10:75] = close[10:75]
        low[10:75] = close[10:75]
    volume = np.exp(rng.normal(15, 0.5, size=n))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_panel(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lengths = [250, 200, 150, 100, 70, 61, 30, 12]  # some shorter than the 60 window
    frames = [_make_symbol_frame(f"S{i:03d}", n, rng, flat=(i == 2)) for i, n in enumerate(lengths)]
    df = pd.concat(frames, ignore_index=True)
    # inject NaNs (interior + near block starts) in every OHLCV column
    for col, positions in {
        "close": [3, 120, 400],
        "high": [55, 260],
        "low": [90, 610],
        "volume": [1, 330],
        "open": [47],
    }.items():
        df.loc[positions, col] = np.nan
    # shuffle rows: the function must sort/group internally
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _factor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("a158_")]


def _unstable_corr_rows(panel: pd.DataFrame, windows) -> dict[str, np.ndarray]:
    """Rows where the trailing-W corr window is exactly constant.

    The reference implementation feeds these to pandas rolling.corr, whose 0/0
    division emits fp-noise garbage (±inf, 0, ~1e-6) — unmatchable by any
    rewrite. The vectorized version emits NaN there instead, matching the
    project's own `_base.ts_corr` convention (inf -> NaN). Returns per-column
    bool masks aligned to the sorted output frame.
    """
    from src.data.factors.alpha158 import _block_edges, _roll_flat

    d = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    edges = _block_edges(d["symbol"].to_numpy())
    g = d.groupby("symbol", group_keys=False)
    close = d["close"].to_numpy(np.float64)
    logv = np.log(d["volume"].to_numpy(np.float64) + 1.0)
    ret1 = (d["close"] / g["close"].shift(1)).to_numpy(np.float64)
    dlogv = (pd.Series(logv) - pd.Series(logv).groupby(d["symbol"]).shift(1)).to_numpy(np.float64)

    def flat_mask(arr: np.ndarray, W: int) -> np.ndarray:
        m = np.zeros(len(d), dtype=bool)
        for s, e in zip(edges[:-1], edges[1:]):
            m[s:e] = _roll_flat(arr[s:e], W)
        return m

    masks: dict[str, np.ndarray] = {}
    for W in windows:
        masks[f"a158_corr{W}"] = flat_mask(close, W) | flat_mask(logv, W)
        masks[f"a158_cord{W}"] = flat_mask(ret1, W) | flat_mask(dlogv, W)
    return masks


def _assert_equiv(ref: pd.DataFrame, new: pd.DataFrame, panel: pd.DataFrame, windows) -> None:
    assert _factor_cols(new) == _factor_cols(ref)
    assert len(new) == len(ref)
    unstable = _unstable_corr_rows(panel, windows)
    for col in _factor_cols(ref):
        a = ref[col].to_numpy(np.float64)
        b = new[col].to_numpy(np.float64)
        u = unstable.get(col)
        if u is not None:
            assert np.isnan(b[u]).all(), f"{col}: expected NaN at flat-window rows"
            keep = ~u
            a, b = a[keep], b[keep]
        na, nb = np.isnan(a), np.isnan(b)
        assert (na == nb).all(), f"{col}: NaN pattern mismatch ({(na != nb).sum()} rows)"
        both = ~na
        assert np.isfinite(a[both]).all(), f"{col}: non-finite reference values"
        if both.any():
            diff = np.abs(a[both] - b[both]).max()
            assert diff < 1e-5, f"{col}: max abs diff {diff:.3e} >= 1e-5"


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_alpha158_vectorized_matches_reference() -> None:
    panel = _make_panel()
    ref = reference_compute_alpha158_factors(panel.copy())
    new = compute_alpha158_factors(panel.copy())
    _assert_equiv(ref, new, panel, (5, 20, 60))


def test_alpha158_vectorized_matches_reference_custom_windows() -> None:
    panel = _make_panel(seed=11)
    windows = (10, 30)
    ref = reference_compute_alpha158_factors(panel.copy(), windows=windows)
    new = compute_alpha158_factors(panel.copy(), windows=windows)
    _assert_equiv(ref, new, panel, windows)
