"""Global macro regime features for the A-share prediction system.

Fetches US equity indices (SPX/NDX/DJI), VIX, US Treasury yields, DXY,
China bond yields, USD/CNY exchange rate, and North-bound capital flow,
aligns everything to the A-share trading calendar with a 1-day lag, and
returns a daily DataFrame ready to be merged onto the stock panel.

All features are DATE-LEVEL (broadcast to every stock on the same date) —
they are regime indicators, NOT stock-level alpha signals. They are lagged
by 1 A-share trading day so they are known at decision time.

Public entry point:
    get_global_macro_features(start_date, end_date)

which returns a daily DataFrame with columns like:
    us_spx_ret_1d, us_ndx_ret_1d, us_dji_ret_1d,
    us_vix_level, us_vix_chg_5d, us_vix_chg_20d,
    us_10y_yield, us_2y_yield, us_yield_spread_10y2y,
    us_dxy_level, us_dxy_chg_20d,
    cn_10y_yield, cn_us_spread_10y,
    cn_usd_cny, cn_usd_cny_chg_20d,
    north_flow_net, north_flow_5d, north_flow_20d
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

import config as C


# ---------------------------------------------------------------------------
# A-share trading calendar
# ---------------------------------------------------------------------------
def _get_a_share_trade_dates(start_date: str, end_date: str) -> list[str]:
    """Return A-share trading dates in [start, end] as 'YYYYMMDD' strings.

    Uses akshare's tool_trade_date_hist_sina which covers SSE trading
    calendar (same as Shenzhen for our purposes).
    """
    import akshare as ak
    cal = ak.tool_trade_date_hist_sina()
    cal["trade_date"] = pd.to_datetime(cal["trade_date"])
    start = pd.to_datetime(_fmt8(start_date), format="%Y%m%d")
    end = pd.to_datetime(_fmt8(end_date), format="%Y%m%d")
    dates = cal[(cal["trade_date"] >= start) & (cal["trade_date"] <= end)]
    return dates["trade_date"].dt.strftime("%Y%m%d").tolist()


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def _fmt8(d) -> str:
    """Normalize a date-like to 'YYYYMMDD'."""
    if d is None:
        return ""
    if isinstance(d, pd.Timestamp):
        return d.strftime("%Y%m%d")
    if isinstance(d, datetime):
        return d.strftime("%Y%m%d")
    return str(d).replace("-", "")


def _to_dt(d):
    """Convert a date-like to datetime."""
    if isinstance(d, datetime):
        return d
    if isinstance(d, pd.Timestamp):
        return d.to_pydatetime()
    return datetime.strptime(str(d).replace("-", ""), "%Y%m%d")


# ---------------------------------------------------------------------------
# yfinance helpers (MultiIndex-safe)
# ---------------------------------------------------------------------------
def _yf_download(ticker: str, start: str, end: str, column: str = "Close") -> pd.Series:
    """Download a single yfinance series and return a flat (date -> value) Series.

    Handles the pandas MultiIndex columns that yfinance returns in newer
    versions by always pulling the first (and only) ticker's column.
    """
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=np.float64)
    # yfinance returns MultiIndex columns like ('Close', '^GSPC') in newer pandas;
    # or flat 'Close' in older versions.
    if isinstance(df.columns, pd.MultiIndex):
        if (column, ticker) in df.columns:
            s = df[(column, ticker)]
        else:
            # fallback: pick the first column matching the name
            s = df[column].iloc[:, 0] if column in df.columns.get_level_values(0) else df.iloc[:, 0]
    else:
        s = df[column] if column in df.columns else df.iloc[:, 0]
    s.name = ticker
    s.index = pd.to_datetime(s.index)
    return s.dropna()


def _yf_download_multi(tickers: list[str], start: str, end: str,
                       column: str = "Close") -> pd.DataFrame:
    """Download multiple yfinance tickers at once; return date-indexed DataFrame.

    Extracts only the requested ``column`` (default 'Close') for each ticker.
    """
    import yfinance as yf
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex columns: ('Close', '^TNX'), ('High', '^TNX'), ...
        # Extract just the requested column level.
        if column in df.columns.get_level_values(0):
            sub = df[column]  # DataFrame with ticker columns, date index
            result = pd.DataFrame(index=sub.index)
            for tkr in sub.columns:
                result[tkr] = sub[tkr]
            return result
        else:
            # Fallback: use all columns, deduplicate ticker names
            result = pd.DataFrame(index=df.index)
            seen = set()
            for col_level in df.columns.get_level_values(0).unique():
                sub = df[col_level]
                for tkr in sub.columns:
                    if tkr not in seen:
                        result[tkr] = sub[tkr]
                        seen.add(tkr)
            return result
    # Flat columns (older yfinance / pandas)
    if column in df.columns:
        return df[[column]]
    return df


# ---------------------------------------------------------------------------
# Individual data source fetchers
# ---------------------------------------------------------------------------
def _fetch_us_indices(start: str, end: str) -> pd.DataFrame:
    """Fetch SPX, NDX, DJI daily closes via yfinance. Returns date-indexed."""
    # Extend range: yfinance needs YYYY-MM-DD, plus buffer for lag
    s = (pd.to_datetime(start, format="%Y%m%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    e = (pd.to_datetime(end, format="%Y%m%d") + timedelta(days=5)).strftime("%Y-%m-%d")

    df = _yf_download_multi(["^GSPC", "^NDX", "^DJI"], start=s, end=e)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"^GSPC": "spx", "^NDX": "ndx", "^DJI": "dji"})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Daily returns
    for col in ["spx", "ndx", "dji"]:
        if col in df.columns:
            df[f"{col}_ret"] = df[col].pct_change()
    return df


def _fetch_us_vix(start: str, end: str) -> pd.Series:
    """Fetch VIX close via yfinance. Returns date-indexed Series."""
    s = (pd.to_datetime(start, format="%Y%m%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    e = (pd.to_datetime(end, format="%Y%m%d") + timedelta(days=5)).strftime("%Y-%m-%d")
    return _yf_download("^VIX", s, e, "Close")


def _fetch_us_treasury(start: str, end: str) -> pd.DataFrame:
    """Fetch US 10Y and 5Y (proxy for 2Y) Treasury yields via yfinance."""
    s = (pd.to_datetime(start, format="%Y%m%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    e = (pd.to_datetime(end, format="%Y%m%d") + timedelta(days=5)).strftime("%Y-%m-%d")

    df = _yf_download_multi(["^TNX", "^FVX"], start=s, end=e)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"^TNX": "us_10y", "^FVX": "us_5y"})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # ^TNX/^FVX are in percent (e.g. 4.57 = 4.57%), keep as-is.
    if "us_10y" in df.columns and "us_5y" in df.columns:
        df["us_yield_spread_10y2y"] = df["us_10y"] - df["us_5y"]
    return df


def _fetch_us_dollar_index(start: str, end: str) -> pd.Series:
    """Fetch DXY (US Dollar Index) via yfinance."""
    s = (pd.to_datetime(start, format="%Y%m%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    e = (pd.to_datetime(end, format="%Y%m%d") + timedelta(days=5)).strftime("%Y-%m-%d")
    return _yf_download("DX-Y.NYB", s, e, "Close")


def _fetch_cn_10y_yield(start: str, end: str) -> pd.Series:
    """Fetch China 10Y government bond yield from akshare.

    Uses ``bond_china_yield`` which returns the full yield curve (multiple bond
    types) for up to 1 year per call. We filter for the 国债 (government bond)
    curve and extract the "10年" (10-year) maturity column.
    """
    import akshare as ak

    cache_dir = os.path.join(C.CACHE_DIR, "macro")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"cn_10y_{_fmt8(start)}_{_fmt8(end)}.parquet")
    if os.path.exists(cache_file):
        cached = pd.read_parquet(cache_file)
        if isinstance(cached, pd.DataFrame) and cached.shape[1] > 0:
            return cached.iloc[:, 0]
        return pd.Series(dtype=np.float64)

    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")

    records: list[dict] = []
    cursor = start_dt
    while cursor <= end_dt:
        chunk_end = min(cursor + timedelta(days=340), end_dt)
        cs = cursor.strftime("%Y%m%d")
        ce = chunk_end.strftime("%Y%m%d")
        try:
            df = ak.bond_china_yield(start_date=cs, end_date=ce)
        except Exception as e:
            print(f"  ⚠️ 国债收益率获取失败 {cs}-{ce}: {e}")
            cursor = chunk_end + timedelta(days=1)
            continue

        if df is not None and not df.empty:
            # Filter for 国债 curve only
            govt = df[df["曲线名称"] == "中债国债收益率曲线"]
            if govt.empty:
                # Fallback: any row with 国债 in the curve name
                govt = df[df["曲线名称"].str.contains("国债", na=False)]
            if govt.empty:
                cursor = chunk_end + timedelta(days=1)
                continue

            date_col = "日期" if "日期" in govt.columns else govt.columns[0]
            govt[date_col] = pd.to_datetime(govt[date_col])

            # Find the 10-year column
            ten_col = None
            for c in govt.columns:
                if c == "10年":
                    ten_col = c
                    break
            if ten_col is None:
                for c in govt.columns:
                    if "10" in str(c) and ("年" in str(c) or "Y" in str(c) or "y" in str(c)):
                        ten_col = c
                        break
            if ten_col is None:
                cursor = chunk_end + timedelta(days=1)
                continue

            for _, row in govt.iterrows():
                val = row.get(ten_col)
                if pd.notna(val):
                    records.append({
                        "date": row[date_col],
                        "cn_10y": float(val),
                    })

        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    if not records:
        return pd.Series(dtype=np.float64)

    out = pd.DataFrame(records).drop_duplicates(subset=["date"]).set_index("date")
    out = out.sort_index()

    # Bond yields are slow-moving (1-3bp daily changes). Reindex to all
    # calendar days and ffill to make daily.
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq="D")
    out = out.reindex(full_idx).ffill()

    s = out["cn_10y"]
    pd.DataFrame(s).to_parquet(cache_file)
    return s


def _fetch_cn_usd_cny(start: str, end: str) -> pd.Series:
    """Fetch USD/CNY central parity rate from akshare (BOC)."""
    import akshare as ak

    cache_dir = os.path.join(C.CACHE_DIR, "macro")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"usdcny_{_fmt8(start)}_{_fmt8(end)}.parquet")
    if os.path.exists(cache_file):
        cached = pd.read_parquet(cache_file)
        if isinstance(cached, pd.DataFrame) and cached.shape[1] > 0:
            return cached.iloc[:, 0]
        return pd.Series(dtype=np.float64)

    s_date = _to_dt(start).strftime("%Y%m%d")
    e_date = _to_dt(end).strftime("%Y%m%d")
    try:
        df = ak.currency_boc_sina(symbol="美元", start_date=s_date, end_date=e_date)
    except Exception as e:
        print(f"  ⚠️ USD/CNY 获取失败: {e}")
        return pd.Series(dtype=np.float64)

    if df is None or df.empty:
        return pd.Series(dtype=np.float64)

    df = df.copy()
    df["date"] = pd.to_datetime(df["日期"])
    # 央行中间价: 单位是 分/100 = 分 per dollar, so divide by 100 to get yuan/dollar
    df["cn_usd_cny"] = pd.to_numeric(df["央行中间价"], errors="coerce") / 100.0
    df = df.dropna(subset=["cn_usd_cny"])
    out = df.set_index("date").sort_index()
    s = out["cn_usd_cny"]
    pd.DataFrame(s).to_parquet(cache_file)
    return s


def _fetch_north_flow(start: str, end: str) -> pd.Series:
    """Fetch daily north-bound (北向) net flow from akshare (EastMoney).

    Uses ``stock_hsgt_hist_em`` which provides daily net buy amount (亿元)
    since 2014-11-17. After August 2024, China's exchanges stopped publishing
    real-time daily north-bound flow data (regulatory change), so recent values
    will be NaN — only quarterly aggregate is disclosed going forward.

    Returns daily net flow in 亿元 (positive = net inflow).
    """
    import akshare as ak

    cache_dir = os.path.join(C.CACHE_DIR, "macro")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"north_flow_{_fmt8(start)}_{_fmt8(end)}.parquet")
    if os.path.exists(cache_file):
        cached = pd.read_parquet(cache_file)
        if isinstance(cached, pd.DataFrame) and cached.shape[1] > 0:
            return cached.iloc[:, 0]
        return pd.Series(dtype=np.float64)

    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
    except Exception as e:
        print(f"  ⚠️ 沪深港通资金流获取失败: {e}")
        return pd.Series(dtype=np.float64)

    if df is None or df.empty:
        return pd.Series(dtype=np.float64)

    df = df.copy()
    df["date"] = pd.to_datetime(df["日期"])
    df["north_flow_net"] = pd.to_numeric(df["当日成交净买额"], errors="coerce")

    daily = df.set_index("date")["north_flow_net"].dropna()
    daily = daily.sort_index()

    # Filter to requested range
    s_dt = _to_dt(start)
    e_dt = _to_dt(end)
    daily = daily[(daily.index >= s_dt) & (daily.index <= e_dt)]

    pd.DataFrame(daily).to_parquet(cache_file)
    return daily


# ---------------------------------------------------------------------------
# Calendar alignment
# ---------------------------------------------------------------------------
def _align_to_a_share_calendar(
    us_df: pd.DataFrame,
    trade_dates: list[str],
    lag: int = 1,
) -> pd.DataFrame:
    """Map US-calendar data onto A-share trading dates with a `lag`-day lag.

    For each A-share trading date D, the feature value is the most recent US
    close on or before D-`lag` (calendar days). This accounts for the fact that:
      - US close at 4pm ET on calendar date X is 4am Beijing on X+1
      - A-share opens at 9:30am Beijing
    So the latest US close available at A-share open on date D is the US close
    with calendar date D-1 (assuming D-1 is a US trading day).

    Args:
        us_df: DataFrame with DatetimeIndex (US calendar dates).
        trade_dates: list of A-share trading dates as 'YYYYMMDD'.
        lag: number of calendar days to lag (1 = US close on D-1 is used for
            A-share decision on D).

    Returns:
        DataFrame indexed by A-share trading date.
    """
    if us_df.empty:
        return pd.DataFrame()

    # Build a daily lookup series of US data indexed by calendar date.
    us_sorted = us_df.sort_index()
    us_dates = pd.to_datetime(us_sorted.index)

    records = []
    for td_str in trade_dates:
        td = _to_dt(td_str)
        # Most recent available date ≤ td - lag days.
        cutoff = td - timedelta(days=lag)
        # Find the latest US date ≤ cutoff.
        mask = us_dates <= cutoff
        if not mask.any():
            continue
        us_date = us_dates[mask].max()
        row = us_sorted.loc[us_date]
        rec = {"date": td}
        if isinstance(row, pd.Series):
            rec.update(row.to_dict())
        else:
            # Duplicate index entries: take the first.
            row0 = row.iloc[0] if hasattr(row, "iloc") else row
            if isinstance(row0, pd.Series):
                rec.update(row0.to_dict())
        records.append(rec)

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index("date").sort_index()


def _align_series_to_a_share_calendar(
    s: pd.Series,
    trade_dates: list[str],
    lag: int = 1,
    col_name: str = "value",
) -> pd.DataFrame:
    """Like _align_to_a_share_calendar but for a single Series."""
    if s.empty:
        return pd.DataFrame()
    df = pd.DataFrame({col_name: s})
    df.index = pd.to_datetime(df.index)
    return _align_to_a_share_calendar(df, trade_dates, lag=lag)


# ---------------------------------------------------------------------------
# Feature aggregation & derived signals
# ---------------------------------------------------------------------------
def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling changes and derived signals to the daily macro DataFrame."""
    df = df.sort_index()

    # VIX changes
    if "vix" in df.columns:
        df["us_vix_level"] = df["vix"]
        df["us_vix_chg_5d"] = df["vix"].pct_change(5)
        df["us_vix_chg_20d"] = df["vix"].pct_change(20)
        df["us_vix_rank_60d"] = df["vix"].rolling(60, min_periods=20).apply(
            lambda x: x.rank(pct=True).iloc[-1], raw=False)

    # US Treasury
    if "us_10y" in df.columns:
        df["us_10y_chg_20d"] = df["us_10y"].diff(20)

    if "us_yield_spread_10y2y" in df.columns:
        df["us_spread_chg_20d"] = df["us_yield_spread_10y2y"].diff(20)

    # DXY changes
    if "dxy" in df.columns:
        df["us_dxy_level"] = df["dxy"]
        df["us_dxy_ret_1d"] = df["dxy"].pct_change()
        df["us_dxy_chg_20d"] = df["dxy"].pct_change(20)

    # China-US spread
    if "cn_10y" in df.columns and "us_10y" in df.columns:
        df["cn_us_spread_10y"] = df["cn_10y"] - df["us_10y"]
        df["cn_us_spread_10y_chg_20d"] = df["cn_us_spread_10y"].diff(20)

    # RMB changes
    if "cn_usd_cny" in df.columns:
        df["cn_usd_cny_chg_20d"] = df["cn_usd_cny"].pct_change(20)
        df["cn_usd_cny_chg_5d"] = df["cn_usd_cny"].pct_change(5)

    # North flow cumulative
    if "north_flow_net" in df.columns:
        df["north_flow_5d"] = df["north_flow_net"].rolling(5, min_periods=1).sum()
        df["north_flow_20d"] = df["north_flow_net"].rolling(20, min_periods=1).sum()

    return df


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the final named feature columns for the panel merge."""
    wanted = [
        # US equity returns
        "spx_ret", "ndx_ret", "dji_ret",
        # VIX
        "us_vix_level", "us_vix_chg_5d", "us_vix_chg_20d", "us_vix_rank_60d",
        # US yields
        "us_10y", "us_10y_chg_20d",
        "us_5y",
        "us_yield_spread_10y2y", "us_spread_chg_20d",
        # DXY
        "us_dxy_level", "us_dxy_ret_1d", "us_dxy_chg_20d",
        # China bonds
        "cn_10y",
        # China-US spread
        "cn_us_spread_10y", "cn_us_spread_10y_chg_20d",
        # FX
        "cn_usd_cny", "cn_usd_cny_chg_5d", "cn_usd_cny_chg_20d",
        # Capital flow
        "north_flow_net", "north_flow_5d", "north_flow_20d",
    ]
    keep = [c for c in wanted if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------
def get_global_macro_features(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch and compute daily global macro features aligned to A-share calendar.

    Args:
        start_date, end_date: 'YYYY-MM-DD' or 'YYYYMMDD'.  Defaults to the
            panel date range derived from config.

    Returns:
        DataFrame with a 'date' column (A-share trading date) and named macro
        feature columns.  Every value is known as of that date's market open
        (1-day lag on all sources).
    """
    start = start_date or C.START_DATE
    end = end_date or C.END_DATE
    if not start or not end:
        end_dt = datetime.today()
        start_dt = end_dt - timedelta(days=365 * 5)
    else:
        start_dt = _to_dt(start)
        end_dt = _to_dt(end)
    start8 = start_dt.strftime("%Y%m%d")
    end8 = end_dt.strftime("%Y%m%d")

    # Cache key
    cache_dir = os.path.join(C.CACHE_DIR, "macro")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"global_macro_{start8}_{end8}.parquet")
    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        df["date"] = pd.to_datetime(df["date"])
        return df

    print("拉取全球宏观特征 (yfinance + akshare)...")

    # 1. A-share trading calendar
    try:
        trade_dates = _get_a_share_trade_dates(start8, end8)
    except Exception as e:
        print(f"  ⚠️ 获取交易日历失败: {e}")
        return pd.DataFrame()
    if not trade_dates:
        print("  ⚠️ 无交易日数据")
        return pd.DataFrame()
    print(f"  A股交易日: {len(trade_dates)} 天 ({trade_dates[0]} ~ {trade_dates[-1]})")

    # 2. Fetch each source
    all_dfs: list[pd.DataFrame] = []

    # --- US indices (yfinance) ---
    try:
        us_idx = _fetch_us_indices(start8, end8)
        us_idx_aligned = _align_to_a_share_calendar(us_idx, trade_dates, lag=1)
        if not us_idx_aligned.empty:
            all_dfs.append(us_idx_aligned)
            print(f"  美股指数: {len(us_idx_aligned)} 天")
    except Exception as e:
        print(f"  ⚠️ 美股指数获取失败: {e}")

    # --- VIX (yfinance) ---
    try:
        vix = _fetch_us_vix(start8, end8)
        vix_aligned = _align_series_to_a_share_calendar(vix, trade_dates, lag=1, col_name="vix")
        if not vix_aligned.empty:
            all_dfs.append(vix_aligned)
            print(f"  VIX: {len(vix_aligned)} 天")
    except Exception as e:
        print(f"  ⚠️ VIX获取失败: {e}")

    # --- US Treasury (yfinance) ---
    try:
        ust = _fetch_us_treasury(start8, end8)
        ust_aligned = _align_to_a_share_calendar(ust, trade_dates, lag=1)
        if not ust_aligned.empty:
            all_dfs.append(ust_aligned)
            print(f"  美债收益率: {len(ust_aligned)} 天")
    except Exception as e:
        print(f"  ⚠️ 美债收益率获取失败: {e}")

    # --- DXY (yfinance) ---
    try:
        dxy = _fetch_us_dollar_index(start8, end8)
        dxy_aligned = _align_series_to_a_share_calendar(dxy, trade_dates, lag=1, col_name="dxy")
        if not dxy_aligned.empty:
            all_dfs.append(dxy_aligned)
            print(f"  DXY: {len(dxy_aligned)} 天")
    except Exception as e:
        print(f"  ⚠️ DXY获取失败: {e}")

    # --- China 10Y yield (akshare) ---
    try:
        cn10y = _fetch_cn_10y_yield(start8, end8)
        cn10y_aligned = _align_series_to_a_share_calendar(
            cn10y, trade_dates, lag=1, col_name="cn_10y")
        if not cn10y_aligned.empty:
            all_dfs.append(cn10y_aligned)
            print(f"  中国10年国债: {len(cn10y_aligned)} 天")
    except Exception as e:
        print(f"  ⚠️ 中国国债收益率获取失败: {e}")

    # --- USD/CNY (akshare) ---
    try:
        usdcny = _fetch_cn_usd_cny(start8, end8)
        usdcny_aligned = _align_series_to_a_share_calendar(
            usdcny, trade_dates, lag=1, col_name="cn_usd_cny")
        if not usdcny_aligned.empty:
            all_dfs.append(usdcny_aligned)
            print(f"  USD/CNY: {len(usdcny_aligned)} 天")
    except Exception as e:
        print(f"  ⚠️ USD/CNY获取失败: {e}")

    # --- North-bound flow (akshare) ---
    try:
        north = _fetch_north_flow(start8, end8)
        north_aligned = _align_series_to_a_share_calendar(
            north, trade_dates, lag=1, col_name="north_flow_net")
        if not north_aligned.empty:
            all_dfs.append(north_aligned)
            print(f"  北向资金: {len(north_aligned)} 天")
    except Exception as e:
        print(f"  ⚠️ 北向资金获取失败: {e}")

    # 3. Merge all sources on date
    if not all_dfs:
        print("  ⚠️ 所有宏观数据源均获取失败")
        return pd.DataFrame()

    merged = all_dfs[0]
    for other in all_dfs[1:]:
        merged = merged.join(other, how="outer")

    merged = merged.sort_index()

    # 4. Derived features
    merged = _add_derived_features(merged)

    # 5. Select output columns
    merged = _select_output_columns(merged)

    # Reset index to get 'date' column
    merged = merged.reset_index()
    merged = merged.rename(columns={"index": "date"})
    # In case the reset_index did rename it
    if "date" not in merged.columns:
        merged["date"] = merged.index

    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date").reset_index(drop=True)

    # Cache
    merged.to_parquet(cache_file, index=False)

    n_cols = len([c for c in merged.columns if c != "date"])
    print(f"  全球宏观特征: {len(merged)} 天, {n_cols} 个特征")
    return merged


def merge_global_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge global macro regime features onto the stock panel by date.

    All features are date-level (same value for every stock on a given date).
    The merge is a left join on `date` — stocks on dates without macro data
    (e.g. early part of the panel before the macro series starts) keep NaN.

    Forward-fill up to MACRO_MAX_FILL_DAYS to cover A-share trading days that
    fall on US holidays (when yfinance returns no new row).
    """
    if not getattr(C, "USE_GLOBAL_MACRO", False):
        return df

    start = df["date"].min().strftime("%Y-%m-%d") if not df.empty else None
    end = df["date"].max().strftime("%Y-%m-%d") if not df.empty else None
    if not start or not end:
        return df

    try:
        macro = get_global_macro_features(start_date=start, end_date=end)
    except Exception as e:
        print(f"  ⚠️ 全球宏观特征合并失败: {e}")
        return df

    if macro is None or macro.empty:
        return df

    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    df = df.merge(macro, on="date", how="left")

    # Forward-fill for gaps (US holidays etc.)
    fill_days = getattr(C, "MACRO_MAX_FILL_DAYS", 3)
    if fill_days > 0:
        macro_cols = [c for c in macro.columns if c != "date"]
        df[macro_cols] = df.groupby("symbol")[macro_cols].ffill(limit=fill_days)

    # Stock-level macro interactions: make date-level macro features useful
    # for cross-sectional models (ic_linear) by multiplying with stock factors.
    if getattr(C, "USE_MACRO_INTERACTIONS", False):
        df = _compute_macro_interactions(df, macro_cols)

    return df


# ---------------------------------------------------------------------------
# Macro × stock-factor interactions
# ---------------------------------------------------------------------------
# Economically meaningful pairs: macro regime indicator × stock characteristic.
# Each interaction = macro_value × cross_sectional_zscore(stock_factor, date).
# The result is STOCK-LEVEL (has cross-sectional variance), so ic_linear can use it.
_MACRO_INTERACTION_PAIRS: list[tuple[str, str]] = [
    # VIX × risk/momentum (high VIX → different factor behaviour)
    ("us_vix_level", "return_5d"),
    ("us_vix_level", "return_20d"),
    ("us_vix_level", "volatility_20"),
    ("us_vix_level", "ln_mv"),
    ("us_vix_chg_5d", "return_5d"),
    ("us_vix_chg_5d", "volatility_20"),
    ("us_vix_rank_60d", "return_20d"),
    ("us_vix_rank_60d", "ln_mv"),
    # DXY × size/export-exposure (dollar strength affects large/small caps differently)
    ("us_dxy_ret_1d", "pct_chg"),
    ("us_dxy_chg_20d", "return_20d"),
    ("us_dxy_chg_20d", "ln_mv"),
    ("us_dxy_chg_20d", "volatility_20"),
    # US yield changes × value/momentum (rate moves shift factor premiums)
    ("us_10y_chg_20d", "return_20d"),
    ("us_10y_chg_20d", "ln_mv"),
    ("us_yield_spread_10y2y", "return_20d"),
    ("us_yield_spread_10y2y", "volatility_20"),
    # China-US spread × value (foreign capital sensitivity)
    ("cn_us_spread_10y", "return_20d"),
    ("cn_us_spread_10y_chg_20d", "return_5d"),
    ("cn_us_spread_10y_chg_20d", "return_20d"),
    # RMB × size/liquidity (currency moves hit large caps and liquid names more)
    ("cn_usd_cny_chg_5d", "return_5d"),
    ("cn_usd_cny_chg_20d", "return_20d"),
    ("cn_usd_cny_chg_20d", "ln_mv"),
    ("cn_usd_cny_chg_20d", "turnover_rate"),
    # North flow × momentum/liquidity (foreign flow enhances/disrupts factor behaviour)
    ("north_flow_net", "return_5d"),
    ("north_flow_5d", "return_5d"),
    ("north_flow_20d", "return_20d"),
    ("north_flow_20d", "turnover_rate"),
    ("north_flow_20d", "volatility_20"),
]


def _compute_macro_interactions(df: pd.DataFrame, macro_cols: list[str]) -> pd.DataFrame:
    """Create stock-level interaction features: macro × stock factor z-score.

    For each (macro, stock_factor) pair in _MACRO_INTERACTION_PAIRS, compute
    ``macro_value × cross_sectional_zscore(stock_factor)`` per date, producing
    a stock-level feature that the cross-sectional model (ic_linear) can rank on.

    The z-score is leak-free: computed within each date using only that day's
    cross-section. Macro features are already lagged 1 day by the merge step,
    so the interaction inherits that lag.

    Pairs where either column is missing are silently skipped.
    """
    if df.empty:
        return df

    df = df.copy()
    available_macro = {c for c in macro_cols if c in df.columns}

    for macro, stock_fac in _MACRO_INTERACTION_PAIRS:
        if macro not in available_macro or stock_fac not in df.columns:
            continue
        col_name = f"mx_{macro}__{stock_fac}"
        if col_name in df.columns:
            continue  # already computed

        # Per-date z-score of the stock factor (leak-free cross-sectional)
        grp = df.groupby("date", group_keys=False)[stock_fac]
        mu = grp.transform("mean")
        sd = grp.transform("std").replace(0, 1e-9)
        z = (df[stock_fac] - mu) / sd
        z = z.clip(-4.0, 4.0)  # clip outliers

        df[col_name] = df[macro].astype(float) * z

    return df
