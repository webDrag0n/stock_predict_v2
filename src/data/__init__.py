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
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config as C
from .base import DataConnector
from .tushare import TushareConnector
from .option_features import get_option_iv_features
from .macro_features import merge_global_macro_features
from .factors import compute_all_factors, get_factor_library_hash

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

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
        if col_type == object or str(col_type) == "category":
            continue
        if pd.api.types.is_datetime64_any_dtype(col_type) \
                or pd.api.types.is_bool_dtype(col_type):
            continue
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


def _drop_intermediate_cols(df: pd.DataFrame, keep_trade_cols: bool = False) -> pd.DataFrame:
    """Drop raw/intermediate columns that are no longer needed after all factor
    families have run and industry features have been added.

    Keeps `date`, `symbol`, `close`, `adj_close`, fundamental/financial columns,
    industry label and all named factor columns.  Drops volume, turnover,
    log_return, moving averages and Bollinger bands (except the 20-day bands
    needed by the daily trading engine when keep_trade_cols=True).

    `pct_chg` is needed by the trade simulator for next-day limit-up/down
    detection, so it is kept when keep_trade_cols=True.
    """
    drop = {"volume", "turnover_rate", "volume_ratio", "log_return"}
    drop |= {c for c in df.columns
             if c.startswith("ma_") or c.startswith("bb_upper") or c.startswith("bb_lower")}
    if keep_trade_cols:
        keep_trade = {"open", "high", "low", "close", "pct_chg",
                      "bb_upper_20", "bb_lower_20"}
        drop -= keep_trade
    else:
        drop |= {"open", "high", "low", "pct_chg"}
    drop = {c for c in drop if c in df.columns}
    if drop:
        df = df.drop(columns=list(drop))
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
# Whole-market quality filters
# --------------------------------------------------------------------------
def _filter_tradable_universe(raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Quality filter for whole-market data — drop untradeable / junk names.

    The whole market (~5500 stocks) includes penny stocks, suspended names, and
    illiquid micro-caps whose noisy prices corrupt the cross-section and aren't
    realistically tradeable. Filters (all leak-free, applied per row on that
    day's own data — no forward info):
      - price floor: close >= C.MIN_PRICE (penny/near-delisting names)
      - liquidity floor: 20-day median amount >= C.MIN_AMOUNT_WAN (万元), removing
        names too thin to trade without large impact
      - exclude STAR/ChiNext/BSE optionally (C.MAIN_BOARD_ONLY) — different price
        limits (20%/30%) and microstructure than the main board
      - drop zero-volume (suspended) rows
    A stock is kept only on dates it passes; a name failing everywhere drops out.
    """
    df = raw.copy()
    n0 = df["symbol"].nunique()

    # Price floor.
    min_price = float(getattr(C, "MIN_PRICE", 2.0))
    df = df[df["close"] >= min_price]

    # Board filter via ts_code prefix (688=STAR, 300/301=ChiNext, 8/4=BSE).
    if getattr(C, "MAIN_BOARD_ONLY", False):
        def _main_board(ts: str) -> bool:
            code = str(ts).split(".")[0]
            is_star = code.startswith("688")
            is_chinext = code.startswith("300") or code.startswith("301")
            is_bse = code.startswith("8") or code.startswith("4")
            return not (is_star or is_chinext or is_bse)
        df = df[df["symbol"].apply(_main_board)]

    # Zero-volume = suspended.
    if "volume" in df.columns:
        df = df[df["volume"] > 0]

    # Liquidity floor: 20-day median amount (万元) per symbol.
    min_amount = float(getattr(C, "MIN_AMOUNT_WAN", 2000))
    if "amount" in df.columns and min_amount > 0:
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df["_amt_med20"] = df.groupby("symbol", group_keys=False)["amount"].transform(
            lambda x: x.rolling(20, min_periods=10).median())
        df = df[df["_amt_med20"] >= min_amount]
        df = df.drop(columns=["_amt_med20"])

    if verbose:
        print(f"  全市场质量过滤: {n0} → {df['symbol'].nunique()} 只可交易股 "
              f"(价格≥{min_price}, 20日成交额≥{getattr(C,'MIN_AMOUNT_WAN',2000)}万)")
    return df


# --------------------------------------------------------------------------
# Option implied-volatility macro features
# --------------------------------------------------------------------------
def merge_option_iv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge ETF option IV macro features onto the stock panel by date.

    The features are lagged by 1 trading day so they are known at decision time.
    Forward-fill up to OPTION_MAX_FILL_DAYS to tolerate missing option quotes.
    """
    if not getattr(C, "USE_OPTION_FEATURES", False):
        return df

    start = df["date"].min().strftime("%Y-%m-%d") if not df.empty else None
    end = df["date"].max().strftime("%Y-%m-%d") if not df.empty else None
    if not start or not end:
        return df

    try:
        iv = get_option_iv_features(
            underlying=C.OPTION_UNDERLYING,
            start_date=start,
            end_date=end,
            lag=1,
        )
    except Exception as e:
        print(f"  ⚠️ option IV 特征获取失败: {e}")
        return df

    if iv is None or iv.empty:
        return df

    iv = iv.copy()
    iv["date"] = pd.to_datetime(iv["date"])
    df = df.merge(iv, on="date", how="left")

    fill_days = getattr(C, "OPTION_MAX_FILL_DAYS", 3)
    if fill_days > 0:
        iv_cols = [c for c in iv.columns if c != "date"]
        df[iv_cols] = df.groupby("symbol")[iv_cols].ffill(limit=fill_days)
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

    Forward return is computed on adj_close (corporate-action-adjusted) — raw
    close jumps on ex-dividend/ex-rights days, which would inject a spurious
    forward return into the target on those days (the bug that inflated the
    backtest). adj_close is built by the technical factor family from pct_chg.

    Also computes additional forward-return targets at MULTI_HORIZON_WINDOWS for
    multi-horizon model objectives. These are stored as responder_fw_{N} and are
    computed with the same market-neutral logic as the primary target.
    """
    price = "adj_close" if "adj_close" in df.columns else "close"

    def _compute_target(fw: int, col: str):
        fut = df.groupby("symbol")[price].shift(-fw)
        ret = (fut / df[price] - 1).clip(-0.5, 0.5)
        if market_neutral_target:
            if "industry" in df.columns and C.INDUSTRY_NEUTRAL:
                grp_mean = ret.groupby([df["date"], df["industry"]]).transform("mean")
            else:
                grp_mean = ret.groupby(df["date"]).transform("mean")
            df[col] = ret - grp_mean
        else:
            df[col] = ret

    _compute_target(forward_window, "responder_6")

    # Multi-horizon targets for fusion objectives.
    for fw in getattr(C, "MULTI_HORIZON_WINDOWS", []):
        if fw != forward_window:
            _compute_target(fw, f"responder_fw_{fw}")

    # --- Residual target: strip known factor premia from the learning target ---
    # When RESIDUAL_TARGET is on, regress the target against known systematic
    # factors (size, momentum, value) per date and keep only the residual.
    # This forces the model to learn stock-specific alpha instead of re-learning
    # well-understood factor premiums. Leak-free: each date uses only its own
    # cross-section for the OLS regression.
    if getattr(C, "RESIDUAL_TARGET", False):
        _residualize_target(df, "responder_6")
        for fw in getattr(C, "MULTI_HORIZON_WINDOWS", []):
            col = f"responder_fw_{fw}"
            if col in df.columns:
                _residualize_target(df, col)

    # Weight: inverse volatility (more certain = higher) + recency boost.
    base_weight = 1.0 / (df["volatility_20"].fillna(0.02) + 0.01)
    date_weight = df.groupby("symbol").cumcount() / len(df)
    df["weight"] = base_weight * (0.5 + 0.5 * date_weight)
    df["weight"] = df["weight"].clip(0.1, 10.0)
    return df


def _residualize_target(df: pd.DataFrame, target_col: str,
                        extra_factors: list[str] | None = None) -> None:
    """Per-date OLS residualization: target ← residual of target ~ control factors.

    Regresses the target column against known systematic factors (size, momentum,
    volatility, value) *within each date's cross-section* and replaces the target
    with the residual. This is leak-free: each date uses only its own data.

    The control factors are drawn from whatever is available in ``df.columns``:
    ln_mv (size), return_20d (momentum), volatility_20 (risk), bp/ep (value).
    Dates with too few valid stocks are left unchanged.
    """
    from sklearn.linear_model import LinearRegression

    candidates = extra_factors or ["ln_mv", "return_20d", "volatility_20", "bp", "ep"]
    control = [c for c in candidates if c in df.columns]
    if not control:
        return

    def _resid_one_date(grp: pd.DataFrame) -> pd.Series:
        y = grp[target_col].to_numpy(dtype=np.float64)
        X = grp[control].to_numpy(dtype=np.float64)
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        min_rows = len(control) + 5
        if valid.sum() <= min_rows:
            return grp[target_col]
        Xv, yv = X[valid], y[valid]
        try:
            model = LinearRegression(fit_intercept=True).fit(Xv, yv)
            residual = y - model.predict(X)
            return pd.Series(residual, index=grp.index)
        except Exception:
            return grp[target_col]

    df[target_col] = df.groupby("date", group_keys=False).apply(
        _resid_one_date, include_groups=False)


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
                    "adj_close", "turnover_rate", "volume_ratio", "pct_chg", "industry",
                    "snap_date", "pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv",
                    "net_profit", "revenue",   # raw financials; keep yoy/rank versions
                    "date_id", "time_id", "symbol_id", "responder_6", "weight",
                    "future_close"] + [c for c in df.columns
                                       if c.startswith("ma_")
                                       or c.startswith("bb_upper")
                                       or c.startswith("bb_lower")
                                       or c.startswith("responder_fw_")]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Pad or truncate to exactly N_FEATURES.
    if len(feature_cols) > C.N_FEATURES:
        dropped = feature_cols[C.N_FEATURES:]
        warnings.warn(
            f"Named factor panel has {len(feature_cols)} candidate features but "
            f"N_FEATURES={C.N_FEATURES}; truncating drops {len(dropped)} columns: "
            f"{dropped[:20]}{'...' if len(dropped) > 20 else ''}. "
            "Raise N_FEATURES or disable factor families to keep them.",
            UserWarning, stacklevel=2)
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
    # The signature MUST include every config knob that changes the engineered
    # feature set; otherwise turning factor families on/off returns a stale cache.
    sig = hashlib.md5(
        f"{source}|{index_code}|{sorted(symbols)}|{start_date}|{end_date}|"
        f"{C.FORWARD_WINDOW}|{C.MARKET_NEUTRAL_TARGET}|"
        f"{C.N_FEATURES}|"
        f"{C.USE_TECHNICAL_FEATURES}|{C.USE_ORTHOGONAL_FEATURES}|"
        f"{C.USE_ALPHA158_FACTORS}|{C.USE_GTJA191_FACTORS}|{C.GTJA191_ALPHA_SUBSET}|"
        f"{C.USE_FUNDAMENTALS}|{C.USE_FINANCIAL_FACTORS}|{C.INDUSTRY_NEUTRAL}|"
        f"{C.USE_OPTION_FEATURES}|{C.USE_GLOBAL_MACRO}|"
        f"{C.USE_MACRO_INTERACTIONS}|{C.RESIDUAL_TARGET}|"
        f"{C.NL_TOPK}|{C.FI_TOPK}|{C.FI_MAX_PAIRS}|{C.MIN_FACTOR_STABILITY}|"
        f"{C.REGIME_IC_FEATURES}|"
        f"{C.USE_FORMULA_FACTORS}|{C.FORMULA_FACTOR_SUBSET}|"
        f"{get_factor_library_hash(C.FACTOR_LIBRARY_DIR)}|"
        f"{C.MULTI_HORIZON_WINDOWS}".encode()).hexdigest()[:12]
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
        print(f"原始数据: {len(raw):,} 行, {raw['symbol'].nunique():,} 只股票, "
              f"{raw['date'].nunique():,} 个交易日")

    # Downcast raw bars before factor engineering to contain whole-market memory.
    if "symbol" in raw.columns:
        raw["symbol"] = raw["symbol"].astype("category")
    raw = reduce_mem_usage(raw, float16_as32=True)
    df = compute_all_factors(raw)
    if C.USE_FUNDAMENTALS and hasattr(connector, "fetch_fundamentals"):
        fund = connector.fetch_fundamentals(start_date, end_date, C.FUND_SAMPLE_DAYS)
        df = merge_fundamentals(df, fund)
    if C.USE_FINANCIAL_FACTORS and hasattr(connector, "fetch_financial_factors"):
        fin = connector.fetch_financial_factors(symbols)
        df = merge_financial_factors(df, fin)
    if C.INDUSTRY_NEUTRAL and hasattr(connector, "get_industry_map"):
        df = add_industry_features(df, connector.get_industry_map())
    # Raw/intermediate columns are no longer needed; drop them before option IV
    # and target computation to keep memory flat on whole-market panels.
    df = _drop_intermediate_cols(df, keep_trade_cols=False)
    df = merge_option_iv_features(df)
    df = merge_global_macro_features(df)
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
    keep_unlabeled: bool = False,
    keep_price: bool = False,
    keep_trade_cols: bool = False,
) -> pd.DataFrame:
    """Return the NAMED-factor panel for single-factor evaluation.

    `load_data` renames factors to feature_00..78 and drops their names, which
    is useless for factor research. This entry point runs the same fetch +
    feature-engineering + target steps but stops BEFORE the Jane Street rename,
    so every factor keeps its human-readable name. Raw bars are reused from the
    same on-disk cache as load_data (no extra API calls).

    Returns a long DataFrame with columns:
        date, symbol, <named factors...>, responder_6

    keep_unlabeled=True (live inference): do NOT drop the latest FORWARD_WINDOW
    days (whose responder_6 is NaN because the forward window extends past the
    last bar) and RETAIN the `close` column. This lets predict_tomorrow.py score
    the most recent trading day and size positions from current prices. The
    default (False) reproduces the leak-free backtest panel exactly.

    keep_trade_cols=True (daily trading engine / per-trade fill simulation):
    RETAIN the raw OHLC bars (open/high/low/close) plus the Bollinger band levels
    (bb_upper_20/bb_lower_20) as META columns. These are needed to (a) decide
    Bollinger/RSI-based entry/exit timing and (b) simulate next-day fills at
    realistic prices against the day's price range / limit bands. They are kept
    OUT of the factor list by list_factor_columns, so the model never sees raw
    price levels as features (which would be non-stationary and leak scale).
    """
    source = source or C.DATA_SOURCE
    index_code = index_code or C.DEFAULT_INDEX
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    connector = get_connector(source)
    whole_market = getattr(C, "UNIVERSE_MODE", "all") == "all" or \
        getattr(C, "UNIVERSE_SIZE", 0) == 0
    if symbols:
        pass
    elif C.DEFAULT_SYMBOLS:
        symbols = C.DEFAULT_SYMBOLS
    elif whole_market:
        symbols = None        # whole-market: fetch every stock by date
    else:
        symbols = connector.get_universe(index_code, start_date, end_date)

    if whole_market and hasattr(connector, "fetch_ohlcv_whole_market"):
        if verbose:
            print(f"因子面板: {source} | 全市场(按日拉取) | {start_date} ~ {end_date}")
        raw = connector.fetch_ohlcv_whole_market(start_date, end_date)
        raw = _filter_tradable_universe(raw, verbose=verbose)
        # Whole-market valuation/financials are infeasible: daily_basic is capped
        # at 5 calls/day and the per-symbol akshare path can't cover 5000+ names.
        # The unified factor library (technical + orthogonal + Alpha158 + GTJA191)
        # from the unlimited daily endpoint carries the signal.
        use_fund = use_fin = False
        if verbose and (C.USE_FUNDAMENTALS or C.USE_FINANCIAL_FACTORS):
            print("  ⚠️ 全市场模式: 基本面/财务因子已自动禁用"
                  "(daily_basic限速5次/天, 全市场不可行)。使用统一因子库。")
    else:
        if verbose:
            print(f"因子面板: {source} | {len(symbols)} 只股票 | {start_date} ~ {end_date}")
        raw = connector.fetch_ohlcv(symbols, start_date, end_date)
        use_fund, use_fin = C.USE_FUNDAMENTALS, C.USE_FINANCIAL_FACTORS
    # Downcast before feature engineering: whole-market panels are huge and
    # factor formulas do not need float64 precision.
    if "symbol" in raw.columns:
        raw["symbol"] = raw["symbol"].astype("category")
    raw = reduce_mem_usage(raw, float16_as32=True)
    df = compute_all_factors(raw)
    if use_fund and hasattr(connector, "fetch_fundamentals"):
        fund = connector.fetch_fundamentals(start_date, end_date, C.FUND_SAMPLE_DAYS)
        df = merge_fundamentals(df, fund)
    if use_fin and hasattr(connector, "fetch_financial_factors"):
        fin = connector.fetch_financial_factors(symbols)
        df = merge_financial_factors(df, fin)
    if C.INDUSTRY_NEUTRAL and hasattr(connector, "get_industry_map"):
        df = add_industry_features(df, connector.get_industry_map())
    # Raw/intermediate columns are no longer needed; drop them before option IV
    # and target computation to keep memory flat on whole-market panels.
    df = _drop_intermediate_cols(df, keep_trade_cols=keep_trade_cols)
    df = merge_option_iv_features(df)
    df = merge_global_macro_features(df)
    df = prepare_target_and_weight(df, forward_window=C.FORWARD_WINDOW,
                                   market_neutral_target=C.MARKET_NEUTRAL_TARGET)
    if C.REGIME_IC_FEATURES:
        df = add_factor_regime_features(df)
    # Keep named factors: drop raw OHLCV, intermediates, and categorical industry label.
    drop = {"open", "high", "low", "close", "volume", "future_close",
            "weight", "log_return", "snap_date", "industry", "adj_close",
            "pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv",
            "net_profit", "revenue"}
    if keep_unlabeled:
        # Live inference needs the current price to size positions, and the
        # limit-flag features already depend on pct_chg being present.
        drop -= {"close"}
    if keep_price:
        # Backtest RMSE in price terms: keep the day-t close so predicted
        # forward returns can be turned into yuan-denominated price errors.
        # close stays a meta column (list_factor_columns excludes it), so it
        # is never fed to the model as a feature.
        drop -= {"close"}
    bb_drop = {c for c in df.columns if c.startswith("ma_")
               or c.startswith("bb_upper") or c.startswith("bb_lower")}
    if keep_trade_cols:
        # Daily trading engine needs the raw OHLC bars (next-day fill simulation
        # against the price range / limit bands) and the Bollinger band levels
        # (entry/exit timing). Keep them as META columns — list_factor_columns
        # excludes open/high/low/bb_* so the model never sees raw price levels.
        drop -= {"open", "high", "low", "close"}
        bb_drop -= {"bb_upper_20", "bb_lower_20"}
    drop |= bb_drop
    keep = [c for c in df.columns if c not in drop]
    df = df[keep].reset_index(drop=True)
    if not keep_unlabeled:
        df = df.dropna(subset=[C.TARGET]).reset_index(drop=True)
    if verbose:
        n_fac = len([c for c in df.columns
                     if c not in ("date", "symbol", C.TARGET, "close")])
        print(f"因子面板: {len(df):,} 行, {n_fac} 个命名因子, "
              f"{df['symbol'].nunique()} 只股票")
    return df


def list_factor_columns(panel: pd.DataFrame) -> list[str]:
    """Factor (non-meta) columns of a panel from load_factor_panel."""
    meta = {"date", "symbol", C.TARGET, C.DATE_COL, C.TIME_COL, C.SYMBOL_COL,
            C.WEIGHT, "close", "adj_close",
            # multi-horizon targets are labels, not features
            "responder_fw_5", "responder_fw_10", "responder_fw_20",
            "responder_fw_30", "responder_fw_60",
            # daily-trading meta (kept by keep_trade_cols, never model features):
            "open", "high", "low", "bb_upper_20", "bb_lower_20"}
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
