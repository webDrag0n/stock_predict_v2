"""ETF option implied-volatility feature engine using Black-Scholes.

Fetches option chain + daily settlement prices from tushare, backs out implied
volatility per contract, and aggregates to daily macro features. These features
are regime indicators, NOT stock-level alpha signals.

Public entry point:
    get_option_iv_features(underlying, start_date, end_date, lag=1)

which returns a daily DataFrame with columns like
    opt_iv_atm_mean, opt_iv_atm_call, opt_iv_atm_put,
    opt_iv_term_1m3m, opt_iv_skew_25d, opt_vix_like,
    opt_iv_chg_5d, opt_iv_chg_20d, opt_iv_rank_60d
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

import config as C


# ---------------------------------------------------------------------------
# Tushare API wrapper with rate-limit handling
# ---------------------------------------------------------------------------
def _tushare_call(func, *args, retries: int = 3, **kwargs):
    """Call a tushare pro function with sleep and retry on rate-limit errors."""
    last_err = None
    for attempt in range(retries):
        try:
            result = func(*args, **kwargs)
            time.sleep(C.OPTION_API_SLEEP)
            return result
        except Exception as e:
            last_err = e
            msg = str(e)
            if "频率超限" in msg or "timeout" in msg.lower() or "timed out" in msg.lower():
                wait = 60 * (attempt + 1)
                print(f"  tushare rate limit / timeout: sleeping {wait}s before retry {attempt + 1}/{retries}")
                time.sleep(wait)
            else:
                raise
    raise last_err


# ---------------------------------------------------------------------------
# Underlying ETF -> option-name matching
# ---------------------------------------------------------------------------
UNDERLYING_NAME_PATTERNS = {
    "510050.SH": "50ETF",   # 华夏上证50ETF
    "510300.SH": "300ETF",  # 华泰柏瑞沪深300ETF
    "510500.SH": "500ETF",  # 南方中证500ETF
}


def _pattern_for(underlying: str) -> str:
    return UNDERLYING_NAME_PATTERNS.get(underlying, underlying.split(".")[0])


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def _fmt(d) -> str:
    """Normalize a date-like to 'YYYYMMDD'."""
    if d is None:
        return ""
    if isinstance(d, pd.Timestamp):
        return d.strftime("%Y%m%d")
    if isinstance(d, datetime):
        return d.strftime("%Y%m%d")
    s = str(d).replace("-", "")
    return s


def _parse(d) -> datetime:
    if isinstance(d, datetime):
        return d
    if isinstance(d, pd.Timestamp):
        return d.to_pydatetime()
    s = str(d).replace("-", "")
    return datetime.strptime(s, "%Y%m%d")


def _trade_dates(pro, start_date: str, end_date: str) -> list[str]:
    """Fetch tushare trade calendar and return open dates in [start, end]."""
    df = _tushare_call(
        pro.trade_cal,
        exchange="SSE",
        start_date=_fmt(start_date),
        end_date=_fmt(end_date),
        fields="cal_date,is_open",
    )
    df = df[df["is_open"] == "1"].sort_values("cal_date")
    return df["cal_date"].astype(str).tolist()


# ---------------------------------------------------------------------------
# Black-Scholes pricing & IV inversion
# ---------------------------------------------------------------------------
def black_scholes_price(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    opt_type: np.ndarray,
) -> np.ndarray:
    """Vectorized Black-Scholes with continuous dividend yield q.

    opt_type is an array of bytes/strings: 'C' for call, 'P' for put.
    All inputs are broadcastable arrays.
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    opt_type = np.asarray(opt_type)

    # Clamp to avoid divide-by-zero / sqrt negative
    T = np.clip(T, 1e-9, None)
    sigma = np.clip(sigma, 1e-9, None)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    Nd1 = stats.norm.cdf(d1)
    Nd2 = stats.norm.cdf(d2)
    Nmd1 = stats.norm.cdf(-d1)
    Nmd2 = stats.norm.cdf(-d2)

    disc_factor = np.exp(-r * T)
    div_factor = np.exp(-q * T)

    price = np.empty_like(S, dtype=np.float64)
    is_call = (opt_type == "C") | (opt_type == "c")
    price[is_call] = (
        S[is_call] * div_factor[is_call] * Nd1[is_call]
        - K[is_call] * disc_factor[is_call] * Nd2[is_call]
    )
    put_mask = ~is_call
    price[put_mask] = (
        K[put_mask] * disc_factor[put_mask] * Nmd2[put_mask]
        - S[put_mask] * div_factor[put_mask] * Nmd1[put_mask]
    )
    return price


def _bs_price_scalar(S, K, T, r, q, sigma, opt_type) -> float:
    """Scalar wrapper for root finding."""
    return float(
        black_scholes_price(
            np.array([S]), np.array([K]), np.array([T]),
            np.array([r]), np.array([q]), np.array([sigma]),
            np.array([opt_type]),
        )[0]
    )


def implied_vol_brentq(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    market_price: float,
    opt_type: str,
    lo: float = 1e-6,
    hi: float = 5.0,
) -> float:
    """Back out implied volatility using Brent's method.

    Returns NaN if no valid root exists (e.g. price below intrinsic).
    """
    if not (S > 0 and K > 0 and T > 1 / 252 and market_price > 0):
        return np.nan

    opt_type = str(opt_type).upper()
    if opt_type not in {"C", "P"}:
        return np.nan

    intrinsic = max(0.0, S - K) if opt_type == "C" else max(0.0, K - S)
    if market_price <= intrinsic:
        return np.nan

    # Upper bound sanity: option price <= S*e^{-qT} for call, <= K*e^{-rT} for put
    if opt_type == "C" and market_price >= S * np.exp(-q * T):
        return np.nan
    if opt_type == "P" and market_price >= K * np.exp(-r * T):
        return np.nan

    def _diff(sigma):
        return _bs_price_scalar(S, K, T, r, q, sigma, opt_type) - market_price

    try:
        if _diff(lo) * _diff(hi) > 0:
            return np.nan
        iv = brentq(_diff, lo, hi, xtol=1e-8, maxiter=100)
        return float(iv)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Data fetching with caching
# ---------------------------------------------------------------------------
def _cache_dir_for(underlying: str) -> str:
    d = os.path.join(C.CACHE_DIR, "option", underlying.replace(".", "_"))
    os.makedirs(d, exist_ok=True)
    return d


def fetch_option_chain(
    pro,
    underlying: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch option contract definitions for one underlying ETF."""
    cache_dir = cache_dir or _cache_dir_for(underlying)
    cache = os.path.join(cache_dir, "chain.parquet")

    if C.OPTION_USE_CACHE and C.OPTION_OFFLINE and os.path.exists(cache):
        return pd.read_parquet(cache)

    if C.OPTION_USE_CACHE and os.path.exists(cache):
        # If offline is not set, still prefer fresh chain if date range expands.
        cached = pd.read_parquet(cache)
        # Simple heuristic: if cached range covers requested range, reuse.
        if "delist_date" in cached.columns:
            cached["delist_date_dt"] = pd.to_datetime(cached["delist_date"], format="%Y%m%d", errors="coerce")
            req_end = pd.to_datetime(_fmt(end_date), format="%Y%m%d")
            if cached["delist_date_dt"].max() >= req_end:
                return cached.drop(columns=["delist_date_dt"])

    # Tushare opt_basic does not expose an underlying field; identify by name.
    pattern = _pattern_for(underlying)
    df = _tushare_call(
        pro.opt_basic,
        exchange="SSE",
        fields="ts_code,name,call_put,exercise_price,list_date,delist_date",
    )
    if df is None or df.empty:
        raise RuntimeError(
            "Tushare option API (opt_basic) returned no data. "
            "The token tier may not have option permissions."
        )

    df = df.dropna(subset=["ts_code", "name", "call_put", "exercise_price"])
    mask = df["name"].astype(str).str.contains(pattern, na=False)
    if underlying == "510050.SH":
        # Avoid matching 500ETF names which contain '50ETF' as substring.
        mask = mask & ~df["name"].astype(str).str.contains("500ETF", na=False)
    df = df[mask].copy()

    if df.empty:
        raise RuntimeError(
            f"No option contracts found for underlying {underlying} (pattern '{pattern}'). "
            "Check tushare opt_basic name format or token permissions."
        )

    for col in ["list_date", "delist_date"]:
        df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
    df["exercise_price"] = pd.to_numeric(df["exercise_price"], errors="coerce")

    if C.OPTION_USE_CACHE:
        df.to_parquet(cache, index=False)
    return df


def fetch_option_daily(
    pro,
    ts_codes: list[str],
    trade_dates: list[str],
    cache_dir: Optional[str] = None,
    sleep: float = 0.35,
) -> pd.DataFrame:
    """Fetch daily option quotes per trade date, cached per date."""
    cache_dir = cache_dir or os.path.join(C.CACHE_DIR, "option", "daily")
    os.makedirs(cache_dir, exist_ok=True)

    ts_codes = sorted(set(ts_codes))
    ts_set = set(ts_codes)

    frames = []
    for td in trade_dates:
        cache = os.path.join(cache_dir, f"{td}.parquet")
        if C.OPTION_USE_CACHE and os.path.exists(cache):
            df = pd.read_parquet(cache)
        else:
            if C.OPTION_OFFLINE:
                continue
            df = _tushare_call(pro.opt_daily, trade_date=td, exchange="SSE")
            if df is None or df.empty:
                continue
            if C.OPTION_USE_CACHE:
                df.to_parquet(cache, index=False)
            time.sleep(C.OPTION_API_SLEEP)

        df = df[df["ts_code"].isin(ts_set)].copy()
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ["settle", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
    return out


def fetch_underlying_prices(
    pro,
    underlying: str,
    trade_dates: list[str],
    cache_dir: Optional[str] = None,
    sleep: float = 0.35,
) -> pd.DataFrame:
    """Fetch ETF closing prices for the underlying."""
    cache_dir = cache_dir or os.path.join(C.CACHE_DIR, "option", "underlying")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{underlying.replace('.', '_')}.parquet")

    if C.OPTION_USE_CACHE and os.path.exists(cache):
        df = pd.read_parquet(cache)
    else:
        if C.OPTION_OFFLINE:
            return pd.DataFrame()
        # fund_daily is the dedicated ETF daily endpoint.
        start = min(trade_dates)
        end = max(trade_dates)
        df = _tushare_call(
            pro.fund_daily,
            ts_code=underlying,
            start_date=start,
            end_date=end,
            fields="trade_date,close",
        )
        if df is None or df.empty:
            # Fallback to stock daily endpoint.
            df = _tushare_call(
                pro.daily,
                ts_code=underlying,
                start_date=start,
                end_date=end,
                fields="trade_date,close",
            )
        if df is None or df.empty:
            raise RuntimeError(f"Could not fetch underlying prices for {underlying}")
        if C.OPTION_USE_CACHE:
            df.to_parquet(cache, index=False)
        time.sleep(C.OPTION_API_SLEEP)

    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna()


# ---------------------------------------------------------------------------
# Feature aggregation
# ---------------------------------------------------------------------------
def compute_option_iv_features(
    chain: pd.DataFrame,
    daily: pd.DataFrame,
    underlying_prices: pd.DataFrame,
    r: float = 0.025,
    q: float = 0.015,
) -> pd.DataFrame:
    """Back out IV per contract and aggregate to daily macro features."""
    if daily.empty or underlying_prices.empty or chain.empty:
        return pd.DataFrame()

    # Merge daily quotes with contract definitions.
    df = daily.merge(chain, on="ts_code", how="inner")
    df = df.merge(underlying_prices.rename(columns={"close": "S"}), on="trade_date", how="inner")
    df = df.dropna(subset=["S", "exercise_price", "settle", "call_put"])
    if df.empty:
        return pd.DataFrame()

    # Time to expiry in years (calendar days).
    df["T"] = (df["delist_date"] - df["trade_date"]).dt.days / 365.0
    df = df[df["T"] > 1 / 252]

    # Back out IV per contract.
    df["iv"] = df.apply(
        lambda row: implied_vol_brentq(
            S=row["S"],
            K=row["exercise_price"],
            T=row["T"],
            r=r,
            q=q,
            market_price=row["settle"],
            opt_type=row["call_put"],
        ),
        axis=1,
    )
    df = df.dropna(subset=["iv"])
    if df.empty:
        return pd.DataFrame()

    # Sort by expiry within each date.
    df["days_to_expiry"] = df["T"] * 365.0

    records = []
    for td, g in df.groupby("trade_date"):
        # Pick near-term and next-term expiry buckets (>=7 days to avoid pin risk).
        expiries = g[g["days_to_expiry"] >= 7].sort_values("days_to_expiry")
        if expiries.empty:
            continue
        near_exp = expiries["days_to_expiry"].min()
        near = g[g["days_to_expiry"] == near_exp]
        next_exps = expiries[expiries["days_to_expiry"] > near_exp + 5]
        next_exp = next_exps["days_to_expiry"].min() if not next_exps.empty else np.nan
        nxt = g[g["days_to_expiry"] == next_exp] if not pd.isna(next_exp) else None

        # ATM = strike closest to spot.
        def _atm_iv(sub):
            sub = sub.copy()
            sub["dist"] = (sub["exercise_price"] - sub["S"].iloc[0]).abs()
            atm_strike = sub.loc[sub["dist"].idxmin(), "exercise_price"]
            atm = sub[sub["exercise_price"] == atm_strike]
            call_iv = atm[atm["call_put"] == "C"]["iv"].mean()
            put_iv = atm[atm["call_put"] == "P"]["iv"].mean()
            return call_iv, put_iv, atm_strike

        call_iv_near, put_iv_near, atmK_near = _atm_iv(near)
        atm_mean_near = np.nanmean([call_iv_near, put_iv_near])

        term_spread = np.nan
        atm_mean_next = np.nan
        if nxt is not None and not nxt.empty:
            call_iv_next, put_iv_next, _ = _atm_iv(nxt)
            atm_mean_next = np.nanmean([call_iv_next, put_iv_next])
            if atm_mean_next > 0:
                term_spread = atm_mean_near / atm_mean_next - 1.0

        # Skew: 25-delta OTM put vs 25-delta OTM call approximation.
        # We proxy by taking the OTM strike whose call/put delta is ~0.25.
        # For simplicity, use the strike whose moneyness is closest to 0.9 (OTM put)
        # and 1.1 (OTM call) of ATM strike.
        def _skew_iv(sub, S):
            put_candidates = sub[(sub["call_put"] == "P") & (sub["exercise_price"] < S)]
            call_candidates = sub[(sub["call_put"] == "C") & (sub["exercise_price"] > S)]
            put_iv = np.nan
            call_iv = np.nan
            if not put_candidates.empty:
                put_candidates = put_candidates.copy()
                put_candidates["moneyness"] = put_candidates["exercise_price"] / S
                best = put_candidates.iloc[(put_candidates["moneyness"] - 0.95).abs().argsort()[:1]]
                put_iv = best["iv"].mean()
            if not call_candidates.empty:
                call_candidates = call_candidates.copy()
                call_candidates["moneyness"] = call_candidates["exercise_price"] / S
                best = call_candidates.iloc[(call_candidates["moneyness"] - 1.05).abs().argsort()[:1]]
                call_iv = best["iv"].mean()
            return put_iv, call_iv

        skew_put, skew_call = _skew_iv(near, near["S"].iloc[0])
        skew = np.nan
        if atm_mean_near > 0:
            skew = (np.nanmean([skew_put]) - np.nanmean([skew_call])) / atm_mean_near

        # VIX-like: variance swap rate synthesized from near and next term ATM IV.
        vix_like = np.nan
        if nxt is not None and not nxt.empty and atm_mean_near > 0 and atm_mean_next > 0:
            T1 = near["T"].iloc[0]
            T2 = nxt["T"].iloc[0]
            # Simple variance interpolation to 30 calendar days (~30/365 years).
            T30 = 30 / 365.0
            if T1 < T30 < T2:
                var1 = atm_mean_near ** 2
                var2 = atm_mean_next ** 2
                vix_like = np.sqrt(
                    (T1 * var1 * (T2 - T30) + T2 * var2 * (T30 - T1)) / (T2 - T1)
                )
            else:
                vix_like = atm_mean_near
        else:
            vix_like = atm_mean_near

        records.append({
            "date": td,
            "opt_iv_atm_mean": atm_mean_near,
            "opt_iv_atm_call": call_iv_near,
            "opt_iv_atm_put": put_iv_near,
            "opt_iv_term_1m3m": term_spread,
            "opt_iv_skew_25d": skew,
            "opt_vix_like": vix_like if not np.isnan(vix_like) else atm_mean_near,
        })

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # Changes and ranks.
    out["opt_iv_chg_5d"] = out["opt_iv_atm_mean"].pct_change(5)
    out["opt_iv_chg_20d"] = out["opt_iv_atm_mean"].pct_change(20)
    out["opt_iv_rank_60d"] = out["opt_iv_atm_mean"].rolling(60, min_periods=20).apply(
        lambda s: s.rank(pct=True).iloc[-1], raw=False
    )
    return out


# ---------------------------------------------------------------------------
# AkShare QVIX fallback (fast, no tushare option API rate limits)
# ---------------------------------------------------------------------------
_QVIX_FUNC_MAP = {
    "510050.SH": "index_option_50etf_qvix",
    "510300.SH": "index_option_300etf_qvix",
    "510500.SH": "index_option_500etf_qvix",
}


def _get_trade_dates_akshare(start_date: str, end_date: str) -> list[str]:
    """Return trading dates in [start, end] using akshare calendar."""
    import akshare as ak
    cal = ak.tool_trade_date_hist_sina()
    cal["trade_date"] = pd.to_datetime(cal["trade_date"])
    start = pd.to_datetime(_fmt(start_date), format="%Y%m%d")
    end = pd.to_datetime(_fmt(end_date), format="%Y%m%d")
    dates = cal[(cal["trade_date"] >= start) & (cal["trade_date"] <= end)]["trade_date"]
    return dates.dt.strftime("%Y%m%d").tolist()


def fetch_option_iv_from_akshare(
    underlying: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch ETF option volatility index (QVIX) from akshare.

    QVIX is already a model-free VIX-style implied-volatility index computed from
    the option chain via the variance-swap methodology (which relies on Black-
    Scholes fair-value interpolation). It is a robust macro regime proxy and
    avoids the tushare option API rate limits entirely.
    """
    import akshare as ak
    if underlying not in _QVIX_FUNC_MAP:
        raise ValueError(
            f"akshare QVIX not available for {underlying}; "
            f"supported: {list(_QVIX_FUNC_MAP)}"
        )

    func = getattr(ak, _QVIX_FUNC_MAP[underlying])
    df = func()
    if df is None or df.empty:
        raise RuntimeError(f"akshare {_QVIX_FUNC_MAP[underlying]} returned no data")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"])
    # QVIX close is in percent points (e.g. 22.75 means 22.75% annualized vol).
    df["opt_iv_atm_mean"] = df["close"] / 100.0
    df["opt_iv_atm_call"] = df["opt_iv_atm_mean"]
    df["opt_iv_atm_put"] = df["opt_iv_atm_mean"]
    df["opt_vix_like"] = df["opt_iv_atm_mean"]

    # Placeholders: QVIX alone does not give term structure or skew.
    df["opt_iv_term_1m3m"] = np.nan
    df["opt_iv_skew_25d"] = np.nan

    # Filter to requested range.
    start = pd.to_datetime(_fmt(start_date), format="%Y%m%d") if start_date else df["date"].min()
    end = pd.to_datetime(_fmt(end_date), format="%Y%m%d") if end_date else df["date"].max()
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    df = df.sort_values("date").reset_index(drop=True)
    df["opt_iv_chg_5d"] = df["opt_iv_atm_mean"].pct_change(5)
    df["opt_iv_chg_20d"] = df["opt_iv_atm_mean"].pct_change(20)
    df["opt_iv_rank_60d"] = df["opt_iv_atm_mean"].rolling(60, min_periods=20).apply(
        lambda s: s.rank(pct=True).iloc[-1], raw=False
    )

    return df[[
        "date", "opt_iv_atm_mean", "opt_iv_atm_call", "opt_iv_atm_put",
        "opt_iv_term_1m3m", "opt_iv_skew_25d", "opt_vix_like",
        "opt_iv_chg_5d", "opt_iv_chg_20d", "opt_iv_rank_60d",
    ]]


def _apply_lag(features: pd.DataFrame, trade_dates: list[str], lag: int) -> pd.DataFrame:
    """Shift option observation date forward by `lag` trading days."""
    if lag <= 0:
        return features
    date_list = sorted(trade_dates)
    date_to_idx = {d: i for i, d in enumerate(date_list)}
    features = features.copy()
    features["date_str"] = features["date"].dt.strftime("%Y%m%d")
    features["new_idx"] = features["date_str"].map(
        lambda d: date_to_idx.get(d, -1) + lag
    )
    features = features[features["new_idx"] < len(date_list)].copy()
    features["date"] = features["new_idx"].map(
        lambda i: pd.to_datetime(date_list[i], format="%Y%m%d")
    )
    features = features.drop(columns=["date_str", "new_idx"])
    return features.reset_index(drop=True)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------
def get_option_iv_features(
    underlying: str = "510300.SH",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lag: int = 1,
) -> pd.DataFrame:
    """Fetch and compute lagged daily ETF option IV features.

    Args:
        underlying: ETF code, e.g. '510300.SH'.
        start_date, end_date: 'YYYY-MM-DD' or 'YYYYMMDD'.
        lag: number of trading days to lag the option IV feature. lag=1 means
            the feature value for stock date t is the option close on t-1.
    """
    if underlying not in UNDERLYING_NAME_PATTERNS:
        raise ValueError(
            f"underlying {underlying} not in supported list: "
            f"{list(UNDERLYING_NAME_PATTERNS)}"
        )

    start = start_date or C.START_DATE
    end = end_date or C.END_DATE
    if not start or not end:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

    cache_dir = _cache_dir_for(underlying)
    source = C.OPTION_SOURCE
    iv_cache = os.path.join(
        cache_dir,
        f"iv_features_{source}_{_fmt(start)}_{_fmt(end)}_lag{lag}.parquet",
    )

    if C.OPTION_USE_CACHE and os.path.exists(iv_cache):
        return pd.read_parquet(iv_cache)

    if source == "akshare":
        features = fetch_option_iv_from_akshare(underlying, start, end)
        trade_dates = _get_trade_dates_akshare(start, end)
    elif source == "tushare":
        import tushare as ts
        token = C.TUSHARE_TOKEN
        if not token:
            raise RuntimeError("TUSHARE_TOKEN not configured")
        ts.set_token(token)
        pro = ts.pro_api()
        trade_dates = _trade_dates(pro, start, end)
        if not trade_dates:
            raise RuntimeError(f"No trading days in range {start} to {end}")

        try:
            chain = fetch_option_chain(pro, underlying, start, end, cache_dir)
        except Exception as e:
            msg = str(e)
            if "频率超限" in msg:
                raise RuntimeError(
                    "Tushare option API (opt_basic) rate limit exceeded on this token tier. "
                    f"Original error: {e}. "
                    "Wait for the rate-limit window to reset, use a higher-tier token, "
                    "or set OPTION_SOURCE=akshare."
                ) from e
            raise RuntimeError(
                "Tushare option API (opt_basic/opt_daily) is not accessible on this token tier. "
                f"Original error: {e}"
            ) from e
        ts_codes = chain["ts_code"].unique().tolist()
        daily = fetch_option_daily(pro, ts_codes, trade_dates, cache_dir)
        if daily.empty:
            raise RuntimeError("No option daily quotes returned; check opt_daily permissions.")

        underlying_prices = fetch_underlying_prices(pro, underlying, trade_dates, cache_dir)
        if underlying_prices.empty:
            raise RuntimeError("No underlying ETF prices returned.")

        features = compute_option_iv_features(
            chain, daily, underlying_prices,
            r=C.OPTION_RISK_FREE, q=C.OPTION_DIV_YIELD,
        )
    else:
        raise ValueError(f"Unknown OPTION_SOURCE: {source}; use 'tushare' or 'akshare'")

    if features.empty:
        raise RuntimeError("Could not compute any option IV features.")

    features = _apply_lag(features, trade_dates, lag)

    if C.OPTION_USE_CACHE:
        features.to_parquet(iv_cache, index=False)
    return features.reset_index(drop=True)
