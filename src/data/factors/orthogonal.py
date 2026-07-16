"""Orthogonal anomaly factors derived from OHLCV.

These factors target documented A-share / cross-sectional alpha dimensions that
are not captured by the base technical set: illiquidity, lottery demand,
idiosyncratic volatility, residual momentum, return skewness, and 52-week-high
proximity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_orthogonal_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Anomaly factors orthogonal to the momentum/value/volatility set, all
    derived from OHLCV already on disk. Each targets a documented A-share /
    cross-sectional alpha the existing factors do not capture:

      amihud_20      : Amihud illiquidity = mean(|ret| / turnover_value). Price
                       impact per yuan traded; illiquid names earn a premium.
      max_ret_20     : MAX lottery factor = max daily return over 20d. High-MAX
                       (lottery-like) stocks subsequently UNDER-perform (Bali).
      idio_vol_20    : idiosyncratic vol = std of (ret − cross-sectional mean
                       ret). Residual risk after removing the market move.
      resid_mom_20   : residual momentum = 20d sum of market-excess daily ret.
                       Momentum after stripping market beta (cleaner than raw).
      ret_skew_20    : return skewness over 20d. Negative-skew stocks earn a
                       premium for crash risk.
      high_52w_prox  : close / trailing 1y high. George-Hwang 52-week-high
                       anchor; proximity predicts continuation.
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)

    r1 = df["return_1d"]
    # Amihud: |ret| / turnover value (close*volume proxies 成交额; scale is
    # irrelevant since factors are cross-sectionally z-scored downstream).
    turn_val = (df["close"] * df["volume"]).replace(0, np.nan)
    df["_il"] = (r1.abs() / turn_val)
    df["amihud_20"] = df.groupby("symbol", group_keys=False)["_il"].transform(
        lambda x: x.rolling(20).mean())
    df = df.drop(columns=["_il"])

    df["max_ret_20"] = g["return_1d"].transform(
        lambda x: x.rolling(20).max())

    # market (universe) mean daily return per date → residual series
    mkt = df.groupby("date")["return_1d"].transform("mean")
    df["_excess_r"] = df["return_1d"] - mkt
    ger = df.groupby("symbol", group_keys=False)["_excess_r"]
    df["idio_vol_20"] = ger.transform(lambda x: x.rolling(20).std())
    df["resid_mom_20"] = ger.transform(lambda x: x.rolling(20).sum())
    df = df.drop(columns=["_excess_r"])

    df["ret_skew_20"] = g["return_1d"].transform(
        lambda x: x.rolling(20).skew())

    high_1y = g["high"].transform(
        lambda x: x.rolling(252, min_periods=60).max())
    df["high_52w_prox"] = df["close"] / (high_1y + 1e-9)

    return df
