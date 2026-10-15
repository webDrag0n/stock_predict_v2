"""Microsoft Qlib Alpha158 factor families — pure OHLCV, A-share-tuned.

Adds the Alpha158 families the project's existing factors do NOT cover, each a
documented, low-correlation alpha dimension (Qlib's empirical staple for Chinese
equities). To avoid redundancy with the existing base factors and keep
dimensionality sane, we use 3 windows {5,20,60} instead of Qlib's 5, and skip
families already present (ROC/MA/STD/MAX/MIN ≈ momentum/volatility/price_to_ma).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Repeated column insertion in rolling-window loops triggers pandas fragmentation
# warnings; the data size is small enough that the warning is noise here. 但库
# 模块不在 import 期动全局 warnings 状态——由入口脚本显式设置。


def _rolling_r2(y: np.ndarray) -> float:
    """R² of y regressed on a time index 0..n-1 (for RSQR)."""
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


def _rolling_resid(y: np.ndarray) -> float:
    """Residual of the LAST point vs a linear time-trend fit over the window."""
    n = len(y)
    if n < 3 or np.any(~np.isfinite(y)):
        return np.nan
    x = np.arange(n, dtype=np.float64)
    b, a = np.polyfit(x, y, 1)
    return float(y[-1] - (b * x[-1] + a))


def _rolling_corr(df: pd.DataFrame, col_a, series_b, W: int, series_a=None):
    """Per-symbol rolling Pearson corr between two aligned series over window W."""
    a = df[col_a] if col_a is not None else series_a
    tmp = pd.DataFrame({"symbol": df["symbol"].to_numpy(),
                        "a": np.asarray(a, dtype=np.float64),
                        "b": np.asarray(series_b, dtype=np.float64)})
    return tmp.groupby("symbol", group_keys=False).apply(
        lambda gg: gg["a"].rolling(W).corr(gg["b"]), include_groups=False)


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
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"]
    hl = (h - l).replace(0, np.nan)
    g = df.groupby("symbol", group_keys=False)

    # --- K-line shape (windowless) ---
    df["a158_kmid"] = (c - o) / o
    df["a158_klen"] = (h - l) / o
    df["a158_kmid2"] = (c - o) / hl
    df["a158_kup"] = (h - np.maximum(o, c)) / o
    df["a158_klow"] = (np.minimum(o, c) - l) / o
    df["a158_ksft"] = (2 * c - h - l) / o
    df["a158_ksft2"] = (2 * c - h - l) / hl

    ret1 = c / g["close"].shift(1)            # close / prev_close
    logv = np.log(v + 1.0)
    dlogv = logv - logv.groupby(df["symbol"]).shift(1)
    up = (c > g["close"].shift(1)).astype(float)
    dn = (c < g["close"].shift(1)).astype(float)
    dpos = (c - g["close"].shift(1)).clip(lower=0)
    dneg = (g["close"].shift(1) - c).clip(lower=0)
    dabs = (c - g["close"].shift(1)).abs()
    rv = (c / g["close"].shift(1) - 1.0).abs() * v   # |return|*volume

    for W in windows:
        cs = df["close"]
        # RSQR / RESI: regress close on time index over W (per symbol)
        def _rsqr(x):
            return x.rolling(W).apply(_rolling_r2, raw=True)

        def _resi(x):
            return x.rolling(W).apply(_rolling_resid, raw=True)

        df[f"a158_rsqr{W}"] = g["close"].transform(_rsqr)
        df[f"a158_resi{W}"] = g["close"].transform(_resi) / cs

        df[f"a158_qtlu{W}"] = g["close"].transform(
            lambda x: x.rolling(W).quantile(0.8)) / cs
        df[f"a158_qtld{W}"] = g["close"].transform(
            lambda x: x.rolling(W).quantile(0.2)) / cs

        min_l = g["low"].transform(lambda x: x.rolling(W).min())
        max_h = g["high"].transform(lambda x: x.rolling(W).max())
        df[f"a158_rsv{W}"] = (cs - min_l) / (max_h - min_l + 1e-12)

        df[f"a158_imax{W}"] = g["high"].transform(
            lambda x: x.rolling(W).apply(np.argmax, raw=True)) / W
        df[f"a158_imin{W}"] = g["low"].transform(
            lambda x: x.rolling(W).apply(np.argmin, raw=True)) / W
        df[f"a158_imxd{W}"] = df[f"a158_imax{W}"] - df[f"a158_imin{W}"]

        # price-volume correlations (per symbol rolling corr)
        df[f"a158_corr{W}"] = _rolling_corr(df, "close", logv, W)
        df[f"a158_cord{W}"] = _rolling_corr(df, None, dlogv, W, series_a=ret1)

        df[f"a158_cntp{W}"] = up.groupby(df["symbol"]).transform(
            lambda x: x.rolling(W).mean())
        df[f"a158_cntn{W}"] = dn.groupby(df["symbol"]).transform(
            lambda x: x.rolling(W).mean())
        df[f"a158_cntd{W}"] = df[f"a158_cntp{W}"] - df[f"a158_cntn{W}"]

        sp = dpos.groupby(df["symbol"]).transform(lambda x: x.rolling(W).sum())
        sn = dneg.groupby(df["symbol"]).transform(lambda x: x.rolling(W).sum())
        sa = dabs.groupby(df["symbol"]).transform(
            lambda x: x.rolling(W).sum()) + 1e-12
        df[f"a158_sump{W}"] = sp / sa
        df[f"a158_sumd{W}"] = (sp - sn) / sa

        rv_std = rv.groupby(df["symbol"]).transform(lambda x: x.rolling(W).std())
        rv_mean = rv.groupby(df["symbol"]).transform(
            lambda x: x.rolling(W).mean()) + 1e-12
        df[f"a158_wvma{W}"] = rv_std / rv_mean

    return df
