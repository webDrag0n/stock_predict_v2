"""Base technical + A-share-specific factors.

These are the foundation factors that are always computed (unless explicitly
disabled). They depend only on OHLCV and a few A-share meta columns that the
DataConnector may provide (turnover_rate, volume_ratio, pct_chg).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config as C

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical + A-share-specific factors.

    All per-symbol rolling/shift operations use groupby().transform() so the
    result stays aligned to the original row index (robust on pandas 2.x/3.x).
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)

    # Forward-adjusted close. We fetch raw prices (adj=None) which JUMP on ex-
    # dividend / ex-rights days; close.pct_change() over those days reports a
    # spurious -X% that is corporate action, not a real return. tushare's
    # `pct_chg` IS computed vs the adjusted pre_close, so compounding it forward
    # reconstructs a clean adjusted series. We anchor at the first raw close per
    # symbol and use adj_close for every CROSS-DAY computation (returns,
    # momentum, MA, volatility, and the forward-return target). Raw OHLC stays
    # for intraday-ratio factors (hl_pct, K-line shapes) where the adjustment
    # factor cancels within a day, and raw close is kept for display/sizing.
    if "pct_chg" in df.columns:
        ratio = g["pct_chg"].transform(
            lambda s: (1.0 + s.fillna(0.0) / 100.0).cumprod())
        first_close = g["close"].transform("first")
        df["adj_close"] = first_close * ratio
    else:
        df["adj_close"] = df["close"]
    ga = df.groupby("symbol", group_keys=False)["adj_close"]

    # Returns (various windows) — on the adjusted series
    for window in [1, 2, 5, 10, 20]:
        df[f"return_{window}d"] = ga.transform(
            lambda x, w=window: x.pct_change(w))

    # Log returns
    df["log_return"] = ga.transform(lambda x: np.log(x / x.shift(1)))

    # Volume features
    df["volume_change"] = g["volume"].pct_change()
    df["volume_ma_ratio_5"] = df["volume"] / g["volume"].transform(
        lambda x: x.rolling(5).mean())
    df["volume_ma_ratio_20"] = df["volume"] / g["volume"].transform(
        lambda x: x.rolling(20).mean())

    # Price momentum (on the adjusted series)
    for window in [5, 10, 20]:
        df[f"momentum_{window}"] = ga.transform(
            lambda x, w=window: x / x.shift(w) - 1)

    # Volatility (rolling std of log returns)
    lr = df.groupby("symbol", group_keys=False)["log_return"]
    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = lr.transform(
            lambda x, w=window: x.rolling(w).std())

    # Moving averages and price/MA ratio (on the adjusted series so cross-day
    # comparisons are corporate-action-free; the ratio is unit-free either way)
    for window in [5, 10, 20, 60]:
        ma = ga.transform(lambda x, w=window: x.rolling(w).mean())
        df[f"ma_{window}"] = ma
        df[f"price_to_ma_{window}"] = df["adj_close"] / ma

    # High-low range
    df["hl_pct"] = (df["high"] - df["low"]) / df["close"]
    df["hl_ma_5"] = df.groupby("symbol", group_keys=False)["hl_pct"].transform(
        lambda x: x.rolling(5).mean())

    # RSI (Relative Strength Index)
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = g["close"].transform(compute_rsi)

    # MACD
    exp1 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    exp2 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df.groupby("symbol", group_keys=False)["macd"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean())
    df["macd_diff"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20-day)
    ma20 = g["close"].transform(lambda x: x.rolling(20).mean())
    std20 = g["close"].transform(lambda x: x.rolling(20).std())
    df["bb_upper_20"] = ma20 + 2 * std20
    df["bb_lower_20"] = ma20 - 2 * std20
    df["bb_position_20"] = (df["close"] - df["bb_lower_20"]) / (
        df["bb_upper_20"] - df["bb_lower_20"])

    # Stochastic oscillator
    low_14 = g["low"].transform(lambda x: x.rolling(14).min())
    high_14 = g["high"].transform(lambda x: x.rolling(14).max())
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
    df["stoch_d"] = df.groupby("symbol", group_keys=False)["stoch_k"].transform(
        lambda x: x.rolling(3).mean())

    # --- A-share-specific factors -----------------------------------------
    # Turnover rate (换手率): liquidity/attention factor, key in A-share.
    if "turnover_rate" in df.columns:
        df["turnover_ma_5"] = g["turnover_rate"].transform(
            lambda x: x.rolling(5).mean())
        df["turnover_ma_20"] = g["turnover_rate"].transform(
            lambda x: x.rolling(20).mean())
        df["turnover_ratio"] = df["turnover_rate"] / (df["turnover_ma_20"] + 1e-9)
    # Volume ratio (量比): today's volume vs recent average, from tushare.
    if "volume_ratio" in df.columns:
        df["volume_ratio_ma5"] = g["volume_ratio"].transform(
            lambda x: x.rolling(5).mean())
    # Limit up/down flag (涨跌停): |pct change| at the ±10% band. Positions
    # cannot be opened/closed at the limit, so the model should perceive it.
    if "pct_chg" in df.columns:
        df["limit_flag"] = (df["pct_chg"].abs() >= 9.9).astype(np.float32)
        df["near_limit"] = (df["pct_chg"].abs() >= 7.0).astype(np.float32)

    # Cross-sectional features (rank within each date, percentile)
    rank_cols = ["return_1d", "volume", "volatility_20"]
    if "turnover_rate" in df.columns:
        rank_cols.append("turnover_rate")
    for col in rank_cols:
        if col in df.columns:
            df[f"{col}_rank"] = df.groupby("date")[col].rank(pct=True)

    return df
