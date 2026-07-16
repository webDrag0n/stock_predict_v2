"""Cached data-access layer for the web app.

Every loader is memoized so callbacks are fast and the heavy parquet files are
read at most once. Paths resolve against the project root (parent of webapp/).
Nothing here trains or fetches from the network — it only reads on-disk caches
produced by the CLI pipeline.
"""
from __future__ import annotations

import os
import glob
import functools

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data_cache")
TUSHARE_CACHE = os.path.join(CACHE, "tushare")
OUTPUTS = os.path.join(ROOT, "outputs")

SEED_OOS = {42: "wholemkt_oos.parquet", 123: "seed_oos_123.parquet",
            7: "seed_oos_7.parquet"}


def _p(*parts) -> str:
    return os.path.join(*parts)


@functools.lru_cache(maxsize=1)
def available_seeds() -> tuple[int, ...]:
    """Seeds whose OOS prediction cache exists on disk."""
    out = []
    for sd, fn in SEED_OOS.items():
        if os.path.exists(_p(CACHE, fn)):
            out.append(sd)
    return tuple(sorted(out))


@functools.lru_cache(maxsize=8)
def seed_oos(seed: int) -> pd.DataFrame:
    """One seed's continuous OOS predictions: date, symbol, prediction, responder_6, return_1d."""
    fn = SEED_OOS.get(seed)
    if fn is None or not os.path.exists(_p(CACHE, fn)):
        return pd.DataFrame()
    return pd.read_parquet(_p(CACHE, fn))


@functools.lru_cache(maxsize=1)
def amount_map() -> pd.DataFrame:
    """(date, symbol) -> amount (千元) from the whole-market panel, for the impact cost."""
    path = _p(CACHE, "wholemkt_panel.parquet")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "symbol", "amount"])
    return pd.read_parquet(path, columns=["date", "symbol", "amount"])


@functools.lru_cache(maxsize=1)
def oos_with_amount_keys() -> tuple:
    """Cheap merge helper: returns nothing heavy, just confirms amount availability."""
    return ("amount" in amount_map().columns,)


@functools.lru_cache(maxsize=1)
def daily_files() -> tuple[str, ...]:
    """Sorted list of whole-market per-date parquet files (date-keyed cache)."""
    return tuple(sorted(glob.glob(_p(TUSHARE_CACHE, "daily_*.parquet"))))


@functools.lru_cache(maxsize=1)
def trade_dates() -> tuple[str, ...]:
    """YYYYMMDD strings parsed from the daily cache filenames."""
    return tuple(os.path.basename(f).replace("daily_", "").replace(".parquet", "")
                 for f in daily_files())


@functools.lru_cache(maxsize=1)
def universe() -> pd.DataFrame:
    """All symbols seen in the most recent daily file (code list for the explorer)."""
    files = daily_files()
    if not files:
        return pd.DataFrame(columns=["symbol"])
    last = pd.read_parquet(files[-1], columns=["symbol"])
    return pd.DataFrame({"symbol": sorted(last["symbol"].unique())})


@functools.lru_cache(maxsize=256)
def symbol_bars(symbol: str) -> pd.DataFrame:
    """Full OHLCV history for one symbol. Fast path: a consolidated long-format
    parquet (all daily caches concatenated, sorted by symbol) built once; falls
    back to scanning per-date files if the consolidated cache is absent.
    """
    consolidated = consolidated_bars()
    if consolidated is not None and not consolidated.empty:
        out = consolidated[consolidated["symbol"] == symbol]
        return out.sort_values("date").reset_index(drop=True)
    # fallback: scan per-date files (slow, ~5s)
    rows = []
    for f in daily_files():
        df = pd.read_parquet(f)
        sub = df[df["symbol"] == symbol]
        if not sub.empty:
            rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def consolidated_bars() -> pd.DataFrame | None:
    """All whole-market daily bars in one long-format frame (built once, cached
    to disk). ~6M rows; lets per-symbol / per-date slices be instant. Returns None
    if the daily caches are missing.
    """
    path = _p(CACHE, "wholemkt_bars_long.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    files = daily_files()
    if not files:
        return None
    frames = [pd.read_parquet(f) for f in files]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out.to_parquet(path, index=False)
    return out


@functools.lru_cache(maxsize=1)
def market_breadth() -> pd.DataFrame:
    """Per-date whole-market breadth: advancers, decliners, mean/median return, count.

    Computed once across all daily caches. pct_chg is in percent.
    """
    recs = []
    for f in daily_files():
        df = pd.read_parquet(f, columns=["date", "pct_chg"])
        if df.empty:
            continue
        pc = df["pct_chg"].to_numpy(dtype=float)
        d = df["date"].iloc[0]
        recs.append({
            "date": d,
            "advancers": int((pc > 0).sum()),
            "decliners": int((pc < 0).sum()),
            "unchanged": int((pc == 0).sum()),
            "mean_ret": float(np.nanmean(pc)),
            "median_ret": float(np.nanmedian(pc)),
            "limit_up": int((pc >= 9.9).sum()),
            "limit_down": int((pc <= -9.9).sum()),
            "n": len(df),
        })
    return pd.DataFrame(recs).sort_values("date").reset_index(drop=True)


@functools.lru_cache(maxsize=32)
def output_csv(name: str) -> pd.DataFrame:
    """Load an outputs/*.csv by name (e.g. 'multiseed_robust.csv')."""
    path = _p(OUTPUTS, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@functools.lru_cache(maxsize=1)
def panel_factor_columns() -> tuple[str, ...]:
    """Factor (non-meta) column names of the whole-market panel."""
    import pyarrow.parquet as pq
    path = _p(CACHE, "wholemkt_panel.parquet")
    if not os.path.exists(path):
        return ()
    meta = {"date", "symbol", "responder_6", "close", "return_1d", "pct_chg",
            "amount", "weight", "date_id", "time_id", "symbol_id"}
    return tuple(c for c in pq.read_schema(path).names if c not in meta)


def warm_load() -> dict:
    """Pre-load the light caches at startup; return a small status dict."""
    seeds = available_seeds()
    for sd in seeds:
        seed_oos(sd)
    amount_map()
    daily_files()
    universe()
    return {
        "seeds": seeds,
        "n_daily_files": len(daily_files()),
        "n_symbols": len(universe()),
        "amount_available": oos_with_amount_keys()[0],
        "n_factors": len(panel_factor_columns()),
    }
