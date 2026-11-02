"""Microsoft Qlib Alpha158 factor families — pure OHLCV, A-share-tuned.

Adds the Alpha158 families the project's existing factors do NOT cover, each a
documented, low-correlation alpha dimension (Qlib's empirical staple for Chinese
equities). To avoid redundancy with the existing base factors and keep
dimensionality sane, we use 3 windows {5,20,60} instead of Qlib's 5, and skip
families already present (ROC/MA/STD/MAX/MIN ≈ momentum/volatility/price_to_ma).

Rolling regressions (RSQR/RESI), window-extreme positions (IMAX/IMIN) and
rolling correlations (CORR/CORD) are computed in closed form from rolling sums
/ sliding-window views instead of ``rolling.apply`` Python callbacks, which
dominated the data-import runtime (millions of per-window calls market-wide).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Repeated column insertion in rolling-window loops triggers pandas fragmentation
# warnings; the data size is small enough that the warning is noise here. 但库
# 模块不在 import 期动全局 warnings 状态——由入口脚本显式设置。


def _block_edges(symbol: np.ndarray) -> np.ndarray:
    """Start offsets of each contiguous symbol block (input sorted by symbol)."""
    change = np.empty(len(symbol) + 1, dtype=bool)
    change[0] = True
    change[-1] = True
    change[1:-1] = symbol[1:] != symbol[:-1]
    return np.flatnonzero(change)


def _roll_sum(a: np.ndarray, W: int) -> np.ndarray:
    """Full-window rolling sum; NaN unless the trailing W values are all finite."""
    finite = np.isfinite(a)
    csum = np.concatenate(([0.0], np.cumsum(np.where(finite, a, 0.0))))
    ccnt = np.concatenate(([0.0], np.cumsum(finite.astype(np.float64))))
    out = np.full(a.shape, np.nan)
    full = (ccnt[W:] - ccnt[:-W]) >= W
    out[W - 1 :] = np.where(full, csum[W:] - csum[:-W], np.nan)
    return out


def _rolling_linreg(y: np.ndarray, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form rolling OLS of y on time index 0..W-1 -> (R², last-point resid)."""
    nan = np.full(y.shape, np.nan)
    if W < 3:
        return nan, nan.copy()
    finite = np.isfinite(y)
    if not finite.any():
        return nan, nan.copy()
    # Shift y by a per-block base: R²/slope/resid are shift-invariant, and the
    # smaller magnitude keeps Σy² − (Σy)²/W cancellation error far below the
    # 1e-12 zero-variance gate.
    yc = y - y[np.argmax(finite)]
    x_mean = (W - 1) / 2.0
    sxx = W * (W * W - 1) / 12.0  # Σ(x - x̄)² for x = 0..W-1
    pos = np.arange(len(y), dtype=np.float64)
    sy = _roll_sum(yc, W)
    syy = _roll_sum(yc * yc, W)
    # Σᵢ i·yᵢ over the window ending at t = rollsum(pos·y) − (t−W+1)·Σy
    sxy = _roll_sum(yc * pos, W) - (pos - (W - 1)) * sy
    sxy_c = sxy - x_mean * sy  # Σ(x−x̄)(y−ȳ); Σ(x−x̄) = 0 drops the ȳ terms
    syy_c = syy - sy * sy / W
    slope = sxy_c / sxx
    intercept = sy / W - slope * x_mean
    resid = yc - (slope * (W - 1) + intercept)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = (sxy_c * sxy_c) / (sxx * syy_c)
    flat = _roll_flat(y, W)
    r2 = np.where(~flat & np.isfinite(syy_c) & (syy_c >= 1e-12), r2, np.nan)
    return r2, resid


def _roll_flat(a: np.ndarray, W: int) -> np.ndarray:
    """Bool mask: trailing-W window is fully present and exactly constant."""
    n = len(a)
    flat = np.zeros(n, dtype=bool)
    if n >= W:
        win = np.lib.stride_tricks.sliding_window_view(a, W)
        full = ~np.isnan(win).any(axis=1)
        with np.errstate(invalid="ignore"):
            same = win.max(axis=1) == win.min(axis=1)
        flat[W - 1 :] = full & same
    return flat


def _roll_argextreme(a: np.ndarray, W: int, *, find_max: bool) -> np.ndarray:
    """Index (0..W-1, first occurrence) of the window extreme; NaN if the
    trailing W values are not all present (mirrors rolling.apply min_periods=W).
    """
    n = len(a)
    out = np.full(n, np.nan)
    if n >= W:
        win = np.lib.stride_tricks.sliding_window_view(a, W)
        full = ~np.isnan(win).any(axis=1)
        fn = np.argmax if find_max else np.argmin
        out[W - 1 :] = np.where(full, fn(win, axis=1), np.nan)
    return out


def _rolling_corr(a: np.ndarray, b: np.ndarray, edges: np.ndarray, W: int) -> np.ndarray:
    """Per-symbol rolling Pearson corr over window W (closed form, min_periods=W)."""
    out = np.full(a.shape, np.nan)
    for s, e in zip(edges[:-1], edges[1:]):
        ab, bb = a[s:e], b[s:e]
        valid = np.isfinite(ab) & np.isfinite(bb)
        a0 = np.where(valid, ab, 0.0)
        b0 = np.where(valid, bb, 0.0)
        m = _roll_sum(valid.astype(np.float64), W)
        sa = _roll_sum(a0, W)
        sb = _roll_sum(b0, W)
        sab = _roll_sum(a0 * b0, W)
        sa2 = _roll_sum(a0 * a0, W)
        sb2 = _roll_sum(b0 * b0, W)
        num = m * sab - sa * sb
        var = np.maximum((m * sa2 - sa * sa) * (m * sb2 - sb * sb), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = num / np.sqrt(var)
        flat = _roll_flat(ab, W) | _roll_flat(bb, W)
        out[s:e] = np.where((m >= W) & ~flat & (var > 0.0), corr, np.nan)
    return out


def compute_alpha158_factors(df: pd.DataFrame, windows=(5, 20, 60)) -> pd.DataFrame:
    """Compute Alpha158 factor families.

    Families added (all leak-free: per-symbol rolling on past bars only):
      K-line shape (9, windowless): KMID/KLEN/KMID2/KUP/KUP2/KLOW/KLOW2/KSFT/KSFT2
        — intraday open/high/low/close geometry; candle structure the project lacks.
      RSQR{W} : R² of close on a time trend — trend QUALITY (vs BETA's slope only).
      RESI{W} : residual of that regression / close — deviation from the trend line.
      QTLU/QTLD{W} : 0.8 / 0.2 rolling price quantile / close — distribution position.
      RSV{W}  : stochastic position (close − minLow)/(maxHigh − minLow) over W.
      IMAX/IMIN/IMXD{W} : bar index of the window high/low (and their gap) / W —
        WHERE in the window the extreme occurred (recency of high/low).
      CORR{W} : corr(close, log(volume+1)) — price-volume co-movement (key signal).
      CORD{W} : corr(close-return, log-volume-change) — return/volume-change corr.
      CNTP/CNTN/CNTD{W} : fraction of up / down days, and their diff — persistence.
      SUMP/SUMN/SUMD{W} : up-move / down-move share of total abs move — gain/loss asymmetry.
      WVMA{W} : std/mean of |return|*volume — volume-weighted volatility of activity.
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    o, h, low, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"]
    hl = (h - low).replace(0, np.nan)
    g = df.groupby("symbol", group_keys=False)

    # --- K-line shape (windowless) ---
    df["a158_kmid"] = (c - o) / o
    df["a158_klen"] = (h - low) / o
    df["a158_kmid2"] = (c - o) / hl
    df["a158_kup"] = (h - np.maximum(o, c)) / o
    df["a158_klow"] = (np.minimum(o, c) - low) / o
    df["a158_ksft"] = (2 * c - h - low) / o
    df["a158_ksft2"] = (2 * c - h - low) / hl

    prev_c = g["close"].shift(1)
    ret1 = c / prev_c  # close / prev_close
    logv = np.log(v + 1.0)
    dlogv = logv - logv.groupby(df["symbol"]).shift(1)
    up = (c > prev_c).astype(float)
    dn = (c < prev_c).astype(float)
    dpos = (c - prev_c).clip(lower=0)
    dneg = (prev_c - c).clip(lower=0)
    dabs = (c - prev_c).abs()
    rv = (ret1 - 1.0).abs() * v  # |return|*volume

    sym = df["symbol"]
    edges = _block_edges(sym.to_numpy())
    c_np = c.to_numpy(dtype=np.float64)
    h_np = h.to_numpy(dtype=np.float64)
    low_np = low.to_numpy(dtype=np.float64)
    logv_np = logv.to_numpy(dtype=np.float64)
    ret1_np = ret1.to_numpy(dtype=np.float64)
    dlogv_np = dlogv.to_numpy(dtype=np.float64)

    for W in windows:
        cs = df["close"]

        # RSQR / RESI: regress close on time index over W (per symbol, closed form)
        rsqr = np.empty(len(df))
        resi = np.empty(len(df))
        imax = np.empty(len(df))
        imin = np.empty(len(df))
        for s, e in zip(edges[:-1], edges[1:]):
            rsqr[s:e], resi[s:e] = _rolling_linreg(c_np[s:e], W)
            imax[s:e] = _roll_argextreme(h_np[s:e], W, find_max=True)
            imin[s:e] = _roll_argextreme(low_np[s:e], W, find_max=False)
        df[f"a158_rsqr{W}"] = rsqr
        df[f"a158_resi{W}"] = resi / cs

        df[f"a158_qtlu{W}"] = g["close"].transform(lambda x: x.rolling(W).quantile(0.8)) / cs
        df[f"a158_qtld{W}"] = g["close"].transform(lambda x: x.rolling(W).quantile(0.2)) / cs

        min_l = g["low"].transform(lambda x: x.rolling(W).min())
        max_h = g["high"].transform(lambda x: x.rolling(W).max())
        df[f"a158_rsv{W}"] = (cs - min_l) / (max_h - min_l + 1e-12)

        df[f"a158_imax{W}"] = imax / W
        df[f"a158_imin{W}"] = imin / W
        df[f"a158_imxd{W}"] = df[f"a158_imax{W}"] - df[f"a158_imin{W}"]

        # price-volume correlations (per symbol rolling corr, closed form)
        df[f"a158_corr{W}"] = _rolling_corr(c_np, logv_np, edges, W)
        df[f"a158_cord{W}"] = _rolling_corr(ret1_np, dlogv_np, edges, W)

        df[f"a158_cntp{W}"] = up.groupby(sym).transform(lambda x: x.rolling(W).mean())
        df[f"a158_cntn{W}"] = dn.groupby(sym).transform(lambda x: x.rolling(W).mean())
        df[f"a158_cntd{W}"] = df[f"a158_cntp{W}"] - df[f"a158_cntn{W}"]

        sp = dpos.groupby(sym).transform(lambda x: x.rolling(W).sum())
        sn = dneg.groupby(sym).transform(lambda x: x.rolling(W).sum())
        sa = dabs.groupby(sym).transform(lambda x: x.rolling(W).sum()) + 1e-12
        df[f"a158_sump{W}"] = sp / sa
        df[f"a158_sumd{W}"] = (sp - sn) / sa

        rv_std = rv.groupby(sym).transform(lambda x: x.rolling(W).std())
        rv_mean = rv.groupby(sym).transform(lambda x: x.rolling(W).mean()) + 1e-12
        df[f"a158_wvma{W}"] = rv_std / rv_mean

    return df
