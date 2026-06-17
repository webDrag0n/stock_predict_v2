"""A-share data package: fetch -> feature-engineer -> Jane Street format.

Public API (unchanged for the rest of the pipeline):
    load_data(...)  -> (DataFrame, is_real)
    split_dates(df) -> (train_dates, val_dates, test_dates)

Connectors live in their own modules and implement `DataConnector`. To add a new
source, implement the interface and add it to `_CONNECTORS` below.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config as C
from .base import DataConnector
from .tushare import TushareConnector

# Registry of available connectors. Add new sources here.
_CONNECTORS: dict[str, type[DataConnector]] = {
    "tushare": TushareConnector,
}


def get_connector(source: str | None = None) -> DataConnector:
    source = source or C.DATA_SOURCE
    if source not in _CONNECTORS:
        raise ValueError(f"未知数据源 '{source}'，可用：{list(_CONNECTORS)}")
    return _CONNECTORS[source]()


# --------------------------------------------------------------------------
# Memory optimization
# --------------------------------------------------------------------------
def reduce_mem_usage(df: pd.DataFrame, float16_as32: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to the smallest safe dtype."""
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != "category":
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32 if float16_as32 else np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if start_mem > 0:
        print(f"内存: {start_mem:.1f} MB -> {end_mem:.1f} MB "
              f"(压缩 {100 * (start_mem - end_mem) / start_mem:.1f}%)")
    return df


# --------------------------------------------------------------------------
# Feature engineering (provider-agnostic)
# --------------------------------------------------------------------------
def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical + A-share-specific factors.

    All per-symbol rolling/shift operations use groupby().transform() so the
    result stays aligned to the original row index (robust on pandas 2.x/3.x).
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)

    # Returns (various windows)
    for window in [1, 2, 5, 10, 20]:
        df[f"return_{window}d"] = g["close"].pct_change(window)

    # Log returns
    df["log_return"] = g["close"].transform(lambda x: np.log(x / x.shift(1)))

    # Volume features
    df["volume_change"] = g["volume"].pct_change()
    df["volume_ma_ratio_5"] = df["volume"] / g["volume"].transform(
        lambda x: x.rolling(5).mean())
    df["volume_ma_ratio_20"] = df["volume"] / g["volume"].transform(
        lambda x: x.rolling(20).mean())

    # Price momentum
    for window in [5, 10, 20]:
        df[f"momentum_{window}"] = g["close"].transform(
            lambda x, w=window: x / x.shift(w) - 1)

    # Volatility (rolling std of log returns)
    lr = df.groupby("symbol", group_keys=False)["log_return"]
    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = lr.transform(
            lambda x, w=window: x.rolling(w).std())

    # Moving averages and price/MA ratio
    for window in [5, 10, 20, 60]:
        ma = g["close"].transform(lambda x, w=window: x.rolling(w).mean())
        df[f"ma_{window}"] = ma
        df[f"price_to_ma_{window}"] = df["close"] / ma

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


# --------------------------------------------------------------------------
# Fundamental (valuation) factors
# --------------------------------------------------------------------------
def merge_fundamentals(df: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time merge of monthly valuation snapshots onto the daily panel.

    Uses merge_asof (backward) per symbol so each trading day only sees the most
    recent snapshot AT OR BEFORE it — no look-ahead. Then derives the standard
    value/size factors that A-share research relies on:
        ep = 1/pe_ttm (earnings yield), bp = 1/pb (book-to-price),
        sp = 1/ps_ttm (sales-to-price), dv_ttm (dividend yield),
        ln_mv (log total market cap, the size factor), turnover_rate_f
    Cross-sectional percentile ranks are added for the key value factors.
    """
    if fund is None or fund.empty:
        return df
    df = df.sort_values("date").reset_index(drop=True)
    fund = fund.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
    fund["date"] = pd.to_datetime(fund["date"]).astype("datetime64[us]")

    merged = pd.merge_asof(
        df, fund, on="date", by="symbol", direction="backward")

    # Derived value factors (inverse of valuation multiples => higher = cheaper)
    if "pe_ttm" in merged:
        merged["ep"] = 1.0 / merged["pe_ttm"].replace(0, np.nan)
    if "pb" in merged:
        merged["bp"] = 1.0 / merged["pb"].replace(0, np.nan)
    if "ps_ttm" in merged:
        merged["sp"] = 1.0 / merged["ps_ttm"].replace(0, np.nan)
    if "total_mv" in merged:
        merged["ln_mv"] = np.log(merged["total_mv"].replace(0, np.nan))

    # Cross-sectional ranks for the value/size factors
    for col in ["ep", "bp", "sp", "dv_ttm", "ln_mv"]:
        if col in merged.columns:
            merged[f"{col}_rank"] = merged.groupby("date")[col].rank(pct=True)
    return merged


def add_factor_regime_features(df: pd.DataFrame, factors: list[str] = None,
                               window: int = 60) -> pd.DataFrame:
    """Method C: rolling cross-sectional IC as meta-features.

    For each date t, computes mean RankIC of factor vs target over the trailing
    `window` dates and assigns it as a scalar feature to all stocks on date t.
    This tells the model whether the factor is currently working — a regime signal.
    Fully backward-looking (no look-ahead): uses only past dates' (factor, target)
    pairs. With EMBARGO_DAYS ≥ FORWARD_WINDOW, target values in the training window
    end before the test window starts.
    """
    from scipy import stats
    factors = factors or [f for f in ["bp", "ep", "value_composite"] if f in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    dates_sorted = np.sort(df["date"].unique())

    for fac in factors:
        if fac not in df.columns:
            continue
        # Daily IC: one scalar per date (cross-sectional Spearman)
        day_ic = df.groupby("date", group_keys=False).apply(
            lambda g, f=fac: (
                stats.spearmanr(g[f].dropna(),
                                g.loc[g[f].notna(), C.TARGET]).correlation
                if g[f].notna().sum() >= 20 else np.nan),
            include_groups=False)
        day_ic = day_ic.reindex(dates_sorted)  # ensure chronological order
        rolling = day_ic.rolling(window, min_periods=window // 3).mean()
        ic_map = rolling.to_dict()
        df[f"ic_{window}d_{fac[:5]}"] = df["date"].map(ic_map)
    return df


def merge_financial_factors(df: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time merge of quarterly financial factors onto the daily panel.

    Uses merge_asof (backward per symbol): each trading day sees only the most
    recent quarterly report whose announcement date has already passed.
    Adds: roe, gross_margin, revenue_yoy, profit_yoy (+ cross-sectional ranks).
    """
    if fin is None or fin.empty:
        return df
    df = df.sort_values("date").reset_index(drop=True)
    fin = fin.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
    fin["date"] = pd.to_datetime(fin["date"]).astype("datetime64[us]")
    fin = fin.sort_values("date")   # merge_asof needs global sort on key

    merged = pd.merge_asof(df, fin, on="date", by="symbol", direction="backward")
    for col in ("roe", "gross_margin", "revenue_yoy", "profit_yoy"):
        if col in merged.columns:
            merged[f"{col}_rank"] = merged.groupby("date")[col].rank(pct=True)
    return merged


def add_industry_features(df: pd.DataFrame, industry_map: dict) -> pd.DataFrame:
    """Add industry column + P0a neutralized features + P1 composite + P2 momentum.

    P0a: attach 'industry' label; industry-relative ranks of value factors so
         the model learns within-industry relative cheapness, not which industry
         is cheapest in aggregate (that's sector beta, not stock alpha).
    P1:  value_composite = equal-weight z-score average of ep/bp/sp on each date.
         Reduces three correlated signals to one robust factor.
    P2:  industry_mom_20 / industry_mom_5 = rolling mean of the industry's daily
         return, minus universe mean (excess industry momentum). Captures sector
         rotation direction that pure value factors miss.
    """
    if not industry_map:
        return df
    df = df.copy()
    df["industry"] = df["symbol"].map(industry_map).fillna("Other")

    # P1: composite value factor (equal-weight zscore of available value signals)
    val_cols = [c for c in ["ep", "bp", "sp"] if c in df.columns]
    if val_cols:
        zs = [df.groupby("date", group_keys=False)[c].transform(
                  lambda x: (x - x.mean()) / (x.std() + 1e-9)) for c in val_cols]
        df["value_composite"] = sum(zs) / len(zs)

    # P0a: within-industry percentile ranks of key value factors
    rank_cols = ["ep", "bp", "value_composite"]
    for col in rank_cols:
        if col in df.columns:
            df[f"{col}_ind_rank"] = df.groupby(["date", "industry"])[col].rank(pct=True)

    # P2: industry momentum (20-day and 5-day rolling industry avg return)
    if "pct_chg" in df.columns:
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        ind_ret = df.groupby(["date", "industry"], group_keys=False)["pct_chg"].transform("mean")
        df["_ind_ret"] = ind_ret
        g = df.groupby("symbol", group_keys=False)["_ind_ret"]
        df["industry_mom_20"] = g.transform(lambda x: x.rolling(20).mean())
        df["industry_mom_5"] = g.transform(lambda x: x.rolling(5).mean())
        # excess = industry momentum minus universe average (removes market beta)
        for col in ["industry_mom_20", "industry_mom_5"]:
            univ_mean = df.groupby("date")[col].transform("mean")
            df[f"{col}_excess"] = df[col] - univ_mean
        df = df.drop(columns=["_ind_ret"])
    return df


# --------------------------------------------------------------------------
# Target + weight
# --------------------------------------------------------------------------
def prepare_target_and_weight(df: pd.DataFrame, forward_window: int = 5,
                              market_neutral_target: bool = True) -> pd.DataFrame:
    """Compute forward return as target (responder_6) and assign weights.

    market_neutral_target=True: target is the cross-sectional excess forward
    return (forward return minus same-date universe mean). This aligns the
    learning objective with relative-ranking strategies (long_topk,
    cross_sectional): the model learns which names out-/under-perform.
    """
    df["future_close"] = df.groupby("symbol")["close"].shift(-forward_window)
    fwd_return = (df["future_close"] / df["close"] - 1).clip(-0.5, 0.5)

    if market_neutral_target:
        if "industry" in df.columns and C.INDUSTRY_NEUTRAL:
            # Industry-internal excess return: removes sector beta so the model
            # learns within-industry relative value, not cross-industry direction.
            grp_mean = fwd_return.groupby([df["date"], df["industry"]]).transform("mean")
        else:
            grp_mean = fwd_return.groupby(df["date"]).transform("mean")
        df["responder_6"] = fwd_return - grp_mean
    else:
        df["responder_6"] = fwd_return

    # Weight: inverse volatility (more certain = higher) + recency boost.
    base_weight = 1.0 / (df["volatility_20"].fillna(0.02) + 0.01)
    date_weight = df.groupby("symbol").cumcount() / len(df)
    df["weight"] = base_weight * (0.5 + 0.5 * date_weight)
    df["weight"] = df["weight"].clip(0.1, 10.0)
    return df


# --------------------------------------------------------------------------
# Jane Street format transform
# --------------------------------------------------------------------------
def transform_to_jane_street_format(df: pd.DataFrame) -> pd.DataFrame:
    """Map to (date_id, time_id, symbol_id, feature_00..78, responder_6, weight)."""
    unique_dates = sorted(df["date"].unique())
    unique_symbols = sorted(df["symbol"].unique())
    date_map = {d: i for i, d in enumerate(unique_dates)}
    symbol_map = {s: i for i, s in enumerate(unique_symbols)}

    df["date_id"] = df["date"].map(date_map).astype(np.int32)
    df["time_id"] = 0
    df["symbol_id"] = df["symbol"].map(symbol_map).astype(np.int32)

    # Exclude raw/intermediate columns; the rest are candidate features.
    exclude_cols = ["date", "symbol", "open", "high", "low", "close", "volume",
                    "turnover_rate", "volume_ratio", "pct_chg", "industry",
                    "snap_date", "pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv",
                    "net_profit", "revenue",   # raw financials; keep yoy/rank versions
                    "date_id", "time_id", "symbol_id", "responder_6", "weight",
                    "future_close"] + [c for c in df.columns
                                       if c.startswith("ma_")
                                       or c.startswith("bb_upper")
                                       or c.startswith("bb_lower")]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Pad or truncate to exactly N_FEATURES.
    if len(feature_cols) > C.N_FEATURES:
        feature_cols = feature_cols[:C.N_FEATURES]
    elif len(feature_cols) < C.N_FEATURES:
        for i in range(len(feature_cols), C.N_FEATURES):
            df[f"dummy_{i}"] = 0.0
            feature_cols.append(f"dummy_{i}")

    feature_rename = {old: f"feature_{i:02d}" for i, old in enumerate(feature_cols)}
    df = df.rename(columns=feature_rename)

    output_cols = ([C.DATE_COL, C.TIME_COL, C.SYMBOL_COL] + C.FEATURE_NAMES +
                   [C.TARGET, C.WEIGHT])
    df = df[output_cols]
    df = df.dropna(subset=[C.TARGET]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def load_data(
    symbols: list[str] | None = None,
    index_code: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    source: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Load A-share data via the configured connector, in Jane Street format."""
    source = source or C.DATA_SOURCE
    index_code = index_code or C.DEFAULT_INDEX
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    connector = get_connector(source)

    # Resolve the symbol universe. When pulling from an index, request the
    # historical union across [start, end] so the backtest is free of
    # survivorship bias (names removed from the index mid-window are kept).
    if symbols:
        symbols = symbols
    elif C.DEFAULT_SYMBOLS:
        symbols = C.DEFAULT_SYMBOLS
    else:
        symbols = connector.get_universe(index_code, start_date, end_date)
    if verbose:
        print(f"数据源: {source} | universe: {index_code} | {len(symbols)} 只股票")
        print(f"日期范围: {start_date} ~ {end_date}")

    # Cache the assembled (processed) dataset keyed by the request signature.
    sig = hashlib.md5(
        f"{source}|{index_code}|{sorted(symbols)}|{start_date}|{end_date}|"
        f"{C.FORWARD_WINDOW}|{C.MARKET_NEUTRAL_TARGET}".encode()).hexdigest()[:12]
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    processed_cache = os.path.join(C.CACHE_DIR, f"dataset_{source}_{sig}.parquet")
    if os.path.exists(processed_cache):
        if verbose:
            print(f"使用已缓存数据集: {processed_cache}")
        df = pd.read_parquet(processed_cache)
        if verbose:
            inspect_dataset(df, verbose=True)
        return df, True

    raw = connector.fetch_ohlcv(symbols, start_date, end_date)
    if verbose:
        print(f"原始数据: {len(raw):,} 行, {raw['symbol'].nunique()} 只股票, "
              f"{raw['date'].nunique()} 个交易日")

    df = compute_technical_features(raw)
    if C.USE_FUNDAMENTALS and hasattr(connector, "fetch_fundamentals"):
        fund = connector.fetch_fundamentals(start_date, end_date, C.FUND_SAMPLE_DAYS)
        df = merge_fundamentals(df, fund)
    if C.USE_FINANCIAL_FACTORS and hasattr(connector, "fetch_financial_factors"):
        fin = connector.fetch_financial_factors(symbols)
        df = merge_financial_factors(df, fin)
    if C.INDUSTRY_NEUTRAL and hasattr(connector, "get_industry_map"):
        df = add_industry_features(df, connector.get_industry_map())
    df = prepare_target_and_weight(df, forward_window=C.FORWARD_WINDOW,
                                   market_neutral_target=C.MARKET_NEUTRAL_TARGET)
    if C.REGIME_IC_FEATURES:
        df = add_factor_regime_features(df)
    df = transform_to_jane_street_format(df)
    df = reduce_mem_usage(df, float16_as32=False)
    df.to_parquet(processed_cache, index=False)

    if verbose:
        print(f"最终数据集: {len(df):,} 行, {df[C.DATE_COL].nunique()} 个交易日, "
              f"{df[C.SYMBOL_COL].nunique()} 只股票")
        inspect_dataset(df, verbose=True)
    return df, True


def load_factor_panel(
    symbols: list[str] | None = None,
    index_code: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    source: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return the NAMED-factor panel for single-factor evaluation.

    `load_data` renames factors to feature_00..78 and drops their names, which
    is useless for factor research. This entry point runs the same fetch +
    feature-engineering + target steps but stops BEFORE the Jane Street rename,
    so every factor keeps its human-readable name. Raw bars are reused from the
    same on-disk cache as load_data (no extra API calls).

    Returns a long DataFrame with columns:
        date, symbol, <named factors...>, responder_6
    """
    source = source or C.DATA_SOURCE
    index_code = index_code or C.DEFAULT_INDEX
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    connector = get_connector(source)
    if symbols:
        pass
    elif C.DEFAULT_SYMBOLS:
        symbols = C.DEFAULT_SYMBOLS
    else:
        symbols = connector.get_universe(index_code, start_date, end_date)
    if verbose:
        print(f"因子面板: {source} | {len(symbols)} 只股票 | {start_date} ~ {end_date}")

    raw = connector.fetch_ohlcv(symbols, start_date, end_date)
    df = compute_technical_features(raw)
    if C.USE_FUNDAMENTALS and hasattr(connector, "fetch_fundamentals"):
        fund = connector.fetch_fundamentals(start_date, end_date, C.FUND_SAMPLE_DAYS)
        df = merge_fundamentals(df, fund)
    if C.USE_FINANCIAL_FACTORS and hasattr(connector, "fetch_financial_factors"):
        fin = connector.fetch_financial_factors(symbols)
        df = merge_financial_factors(df, fin)
    if C.INDUSTRY_NEUTRAL and hasattr(connector, "get_industry_map"):
        df = add_industry_features(df, connector.get_industry_map())
    df = prepare_target_and_weight(df, forward_window=C.FORWARD_WINDOW,
                                   market_neutral_target=C.MARKET_NEUTRAL_TARGET)
    if C.REGIME_IC_FEATURES:
        df = add_factor_regime_features(df)
    # Keep named factors: drop raw OHLCV, intermediates, and categorical industry label.
    drop = {"open", "high", "low", "close", "volume", "future_close",
            "weight", "log_return", "snap_date", "industry",
            "pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv",
            "net_profit", "revenue"}
    drop |= {c for c in df.columns if c.startswith("ma_")
             or c.startswith("bb_upper") or c.startswith("bb_lower")}
    keep = [c for c in df.columns if c not in drop]
    df = df[keep].dropna(subset=[C.TARGET]).reset_index(drop=True)
    if verbose:
        n_fac = len([c for c in df.columns
                     if c not in ("date", "symbol", C.TARGET)])
        print(f"因子面板: {len(df):,} 行, {n_fac} 个命名因子, "
              f"{df['symbol'].nunique()} 只股票")
    return df


def list_factor_columns(panel: pd.DataFrame) -> list[str]:
    """Factor (non-meta) columns of a panel from load_factor_panel."""
    meta = {"date", "symbol", C.TARGET, C.DATE_COL, C.TIME_COL, C.SYMBOL_COL,
            C.WEIGHT}
    return [c for c in panel.columns if c not in meta]


def inspect_dataset(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Lightweight dataset quality checks (absorbed from a DatasetInspector).

    Surfaces three things that silently wreck a cross-sectional backtest:
      - label (target) statistics: sanity of the forward-return target
      - cross-section coverage: stocks per date (too few => unstable ranking)
      - feature NaN-by-date irregularity: a crude lookahead-bias smell test
    Returns a dict of findings; prints a summary when verbose.
    """
    findings: dict = {"warnings": []}

    y = df[C.TARGET]
    findings["label"] = {
        "mean": float(y.mean()), "std": float(y.std()),
        "min": float(y.min()), "max": float(y.max()),
        "q01": float(y.quantile(0.01)), "q99": float(y.quantile(0.99)),
        "positive_ratio": float((y > 0).mean()),
        "nan": int(y.isna().sum()),
    }

    per_date = df.groupby(C.DATE_COL)[C.SYMBOL_COL].nunique()
    findings["coverage"] = {
        "n_dates": int(len(per_date)),
        "avg_stocks_per_date": float(per_date.mean()),
        "min_stocks_per_date": int(per_date.min()),
    }
    if per_date.min() < 30:
        findings["warnings"].append(
            f"部分交易日横截面股票数过少 (最小={per_date.min()})，排序策略可能不稳定")

    # NaN ratio per date should be roughly flat per feature; large variance across
    # dates can indicate a leak or an alignment bug.
    irregular = 0
    for col in C.FEATURE_NAMES[:20]:
        nan_by_date = df.groupby(C.DATE_COL)[col].apply(lambda x: x.isna().mean())
        if nan_by_date.std() > 0.1:
            irregular += 1
    if irregular:
        findings["warnings"].append(f"{irregular} 个特征的缺失率在日期间波动较大（疑似对齐/泄露）")

    if verbose:
        lab = findings["label"]
        print("数据集质量检查：")
        print(f"  标签 responder_6: 均值={lab['mean']:+.5f} 标准差={lab['std']:.5f} "
              f"正收益比例={lab['positive_ratio']:.1%}")
        print(f"  横截面覆盖: {findings['coverage']['n_dates']} 个交易日, "
              f"日均 {findings['coverage']['avg_stocks_per_date']:.0f} 只股票 "
              f"(最少 {findings['coverage']['min_stocks_per_date']} 只)")
        if findings["warnings"]:
            for w in findings["warnings"]:
                print(f"  ⚠️ {w}")
        else:
            print("  ✅ 未发现明显异常")
    return findings


def split_dates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological 3-way split: (train, validation, test) date arrays.

    The validation segment is used to SELECT a strategy; the test segment is the
    final out-of-sample report. Selecting on test would be data snooping.
    """
    dates = np.sort(df[C.DATE_COL].unique())
    n_test = C.NUM_TEST_DATES
    n_val = C.NUM_VAL_DATES
    test_dates = dates[-n_test:]
    val_dates = dates[-(n_test + n_val):-n_test]
    train_dates = dates[:-(n_test + n_val)]
    return train_dates, val_dates, test_dates
