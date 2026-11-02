"""A-share data package: fetch -> feature-engineer -> Jane Street format.

Public API (unchanged for the rest of the pipeline):
    load_data(...)  -> (DataFrame, is_real)
    split_dates(df) -> (train_dates, val_dates, test_dates)

Connectors live in their own modules and implement `DataConnector`. To add a new
source, implement the interface and add it to `_CONNECTORS` below.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config as C
from .base import (
    DataConnector,
    SupportsFundamentals,
    SupportsFinancialFactors,
    SupportsIndustryMap,
    SupportsNorthbound,
    SupportsWholeMarketFetch,
)
from .tushare import TushareConnector
from .option_features import get_option_iv_features
from .macro_features import merge_global_macro_features
from .factors import MAX_ROLLING_WINDOW, compute_all_factors, get_factor_library_hash

# 注：不在 import 期全局屏蔽 PerformanceWarning——由入口脚本
# （run_session.py / predict_tomorrow.py / scripts/run_walkforward.py）显式设置。

logger = logging.getLogger(__name__)

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
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_object_dtype(col_type) or str(col_type) == "category":
            continue
        if pd.api.types.is_datetime64_any_dtype(col_type) or pd.api.types.is_bool_dtype(col_type):
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
    end_mem = df.memory_usage().sum() / 1024**2
    if start_mem > 0:
        logger.info(
            f"内存: {start_mem:.1f} MB -> {end_mem:.1f} MB "
            f"(压缩 {100 * (start_mem - end_mem) / start_mem:.1f}%)"
        )
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
    drop |= {
        c
        for c in df.columns
        if c.startswith("ma_") or c.startswith("bb_upper") or c.startswith("bb_lower")
    }
    if keep_trade_cols:
        keep_trade = {"open", "high", "low", "close", "pct_chg", "bb_upper_20", "bb_lower_20"}
        drop -= keep_trade
    else:
        drop |= {"open", "high", "low", "pct_chg"}
    drop = {c for c in drop if c in df.columns}
    if drop:
        df = df.drop(columns=list(drop))
    return df


def _align_merge_key(
    left: pd.DataFrame, right: pd.DataFrame, key: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align the dtype of a merge/join key between the big panel (left) and a
    small side table (right).

    The whole-market panel stores `symbol` as category for memory; side tables
    from connectors come back as str/StringDtype. merge_asof requires EXACTLY
    matching key dtypes (a category-vs-string mismatch raises MergeError).
    Only the small side is converted, and categorical categories are taken
    from the panel, so keys absent from the panel become NaN (they could not
    match anyway).
    """
    lt, rt = left[key].dtype, right[key].dtype
    if lt == rt:
        return left, right
    if isinstance(lt, pd.CategoricalDtype) and not isinstance(rt, pd.CategoricalDtype):
        right = right.copy()
        right[key] = pd.Categorical(right[key].astype(str), categories=left[key].cat.categories)
        return left, right
    if isinstance(rt, pd.CategoricalDtype) and not isinstance(lt, pd.CategoricalDtype):
        left = left.copy()
        left[key] = pd.Categorical(left[key].astype(str), categories=right[key].cat.categories)
        return left, right
    left = left.copy()
    right = right.copy()
    left[key] = left[key].astype("string")
    right[key] = right[key].astype("string")
    return left, right


# --------------------------------------------------------------------------
# Fundamental (valuation) factors
# --------------------------------------------------------------------------
def merge_fundamentals(df: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time merge of monthly valuation snapshots onto the daily panel.

    Uses merge_asof (backward) per symbol. Valuation snapshots are published
    after the close, so snapshot dates are shifted +1 day before the merge:
    each trading day only sees snapshots STRICTLY BEFORE it — no look-ahead.
    Then derives the standard
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
    # 防泄露：估值快照（akshare 东财个股估值）是收盘后才发布的数据，
    # 同日 merge_asof 会让当日决策看到当天收盘估值 —— 统一后移 1 天，
    # 使快照只在次日起可见。
    fund["date"] = fund["date"] + pd.Timedelta(days=1)

    df, fund = _align_merge_key(df, fund, "symbol")
    merged = pd.merge_asof(df, fund, on="date", by="symbol", direction="backward")

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


def add_factor_regime_features(
    df: pd.DataFrame, factors: list[str] = None, window: int = 60
) -> pd.DataFrame:
    """Method C: rolling cross-sectional IC as meta-features.

    For each date t, computes mean RankIC of factor vs target over the trailing
    `window` dates and assigns it as a scalar feature to all stocks on date t.
    This tells the model whether the factor is currently working — a regime signal.

    防泄露机制：日 IC 用的是 responder_6（t → t+FORWARD_WINDOW 的远期收益），
    第 d 天的 IC 要到 d+FORWARD_WINDOW 天才“成熟”可知。因此 IC 序列在 rolling
    之前先整体 shift(FORWARD_WINDOW + 1)：第 t 天用到的最新 IC 来自 t-fw-1
    之前，其标签在 t-1 已完全成熟。不加这个 shift，测试窗内的特征会包含
    尚未成熟的 IC（前视）。
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
                stats.spearmanr(g[f].dropna(), g.loc[g[f].notna(), C.TARGET]).correlation
                if g[f].notna().sum() >= 20
                else np.nan
            ),
            include_groups=False,
        )
        day_ic = day_ic.reindex(dates_sorted)  # ensure chronological order
        # 只使用已成熟的 IC：整体后移 fw+1 天再 rolling（见 docstring）
        matured_ic = day_ic.shift(C.FORWARD_WINDOW + 1)
        rolling = matured_ic.rolling(window, min_periods=window // 3).mean()
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
    fin = fin.sort_values("date")  # merge_asof needs global sort on key

    df, fin = _align_merge_key(df, fin, "symbol")
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
        zs = [
            df.groupby("date", group_keys=False)[c].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )
            for c in val_cols
        ]
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
            lambda x: x.rolling(20, min_periods=10).median()
        )
        df = df[df["_amt_med20"] >= min_amount]
        df = df.drop(columns=["_amt_med20"])

    if verbose:
        logger.info(
            f"  全市场质量过滤: {n0} → {df['symbol'].nunique()} 只可交易股 "
            f"(价格≥{min_price}, 20日成交额≥{getattr(C,'MIN_AMOUNT_WAN',2000)}万)"
        )
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
        logger.warning(f"  ⚠️ option IV 特征获取失败: {e}")
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
# Northbound (Stock Connect) holdings factors
# --------------------------------------------------------------------------
def _zero_north_columns(df: pd.DataFrame) -> pd.DataFrame:
    """北向数据不可得时的 schema 兜底：补"无覆盖"编码的北向因子列。

    与 merge_northbound_features 对非陆股通标的的编码一致（数值列 0、
    north_covered 0），保证缺数据期模型输入列集不变。
    """
    for col in ("north_ratio", "north_ratio_chg5", "north_vol_chg5"):
        df[col] = 0.0
    df["north_covered"] = np.int8(0)
    return df


def merge_northbound_features(df: pd.DataFrame, connector: DataConnector) -> pd.DataFrame:
    """Merge stock-level northbound holdings factors onto the panel.

    派生因子（每只票）：
      north_ratio      北向持股占流通股比例（%，滞后一日+ffill）
      north_ratio_chg5 north_ratio 的 5 个交易日变化（百分点）
      north_vol_chg5   北向持股量的 5 个交易日变化率
      north_covered    陆股通覆盖哑变量（当日有可用北向数据=1）
    非陆股通标的（无数据行）数值列填 0，north_covered=0。

    防泄露（关键）：tushare hk_hold 的 T 日持股数据由港交所在 T 日晚间/次晨
    披露，T 日收盘决策时刻不可得。因此按日合并后按 symbol 整体 shift(1)——
    T 日行只能看到 T-1 日（含）之前已披露的持股；随后 ffill 仅用于覆盖
    陆港通休市错开日（上限 NORTHBOUND_MAX_FILL_DAYS，超限视为无数据，
    防止无限沿用陈旧快照）。
    """
    if not getattr(C, "USE_NORTHBOUND", False):
        return df
    if not isinstance(connector, SupportsNorthbound) or df.empty:
        return df

    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")
    try:
        hk = connector.fetch_hk_hold(start, end)
    except Exception as e:
        logger.warning(f"  ⚠️ 北向持股获取失败，按无覆盖编码补零列: {e}")
        return _zero_north_columns(df)
    if hk is None or hk.empty:
        # 数据完全不可得（限速熔断且无缓存）也不能让面板缺列：模型 feat_cols
        # 是在含北向列的面板上训练的，缺列会在预测期 KeyError。按 merge 后
        # "无覆盖"的编码（数值 0 / covered 0）补列，保证因子 schema 稳定。
        logger.warning("  ⚠️ 北向持股为空，按无覆盖编码补零列（schema 兜底）")
        return _zero_north_columns(df)

    hk = hk.copy()
    hk["date"] = pd.to_datetime(hk["date"])
    hk["symbol"] = hk["symbol"].astype(str)
    # 防御性去重：hk_hold 按 ts_code 唯一（exchange 区分沪/深来源）
    hk = hk.drop_duplicates(subset=["symbol", "date"], keep="first")

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df, hk = _align_merge_key(df, hk, "symbol")
    df = df.merge(hk, on=["symbol", "date"], how="left")

    # 披露滞后一日：shift(1) 后 T 日行看到的是 T-1 日已披露的值
    g = df.groupby("symbol", sort=False)
    df["north_ratio"] = g["north_ratio"].shift(1)
    df["north_vol"] = g["north_vol"].shift(1)
    # 陆港通休市错开日（港股通关闭但 A 股开市）前向填充
    fill_limit = int(getattr(C, "NORTHBOUND_MAX_FILL_DAYS", 10))
    if fill_limit > 0:
        g = df.groupby("symbol", sort=False)
        df["north_ratio"] = g["north_ratio"].ffill(limit=fill_limit)
        df["north_vol"] = g["north_vol"].ffill(limit=fill_limit)

    df["north_covered"] = df["north_ratio"].notna().astype(np.int8)
    g = df.groupby("symbol", sort=False)
    df["north_ratio_chg5"] = df["north_ratio"] - g["north_ratio"].shift(5)
    prev_vol = g["north_vol"].shift(5)
    df["north_vol_chg5"] = df["north_vol"] / prev_vol.replace(0, np.nan) - 1.0
    for col in ["north_ratio", "north_ratio_chg5", "north_vol_chg5"]:
        df[col] = df[col].fillna(0.0)
    # north_vol 原始持股量（股数）截面量纲差异大、非平稳，不作为特征保留
    df = df.drop(columns=["north_vol"])
    return df


# --------------------------------------------------------------------------
# Target + weight
# --------------------------------------------------------------------------
def prepare_target_and_weight(
    df: pd.DataFrame, forward_window: int = 5, market_neutral_target: bool = True
) -> pd.DataFrame:
    """Compute forward return as target (responder_6) and assign weights.

    market_neutral_target=True: target is the cross-sectional excess forward
    return (forward return minus same-date universe mean). This aligns the
    learning objective with relative-ranking strategies (long_topk,
    cross_sectional): the model learns which names out-/under-perform.

    C.INDUSTRY_NEUTRAL=True（且面板带 industry 标签）时改为减当日**同行业**
    均值（行业中性，剔除板块轮动成分）；当日有效样本数 < INDUSTRY_MIN_COUNT
    的行业回退到全市场均值（小样本行业均值噪声大）。INDUSTRY_NEUTRAL=1
    隐含截面中性——即使 market_neutral_target=False 也照常去均值。

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
        # 截面去均值只用当日截面（行业均值/全市场均值都是当日横截面统计量，
        # 决策时刻未知但标签本身允许——它是训练目标而非特征）。
        mkt_mean = ret.groupby(df["date"]).transform("mean")
        if C.INDUSTRY_NEUTRAL:
            # INDUSTRY_NEUTRAL=1 隐含截面中性：即使 market_neutral_target=False
            # 也去均值（行业均值是截面均值的一种）。
            if "industry" in df.columns:
                grp = [df["date"], df["industry"]]
                ind_mean = ret.groupby(grp).transform("mean")
                ind_cnt = ret.groupby(grp).transform("count")
                # 行业当日有效样本过少 → 回退全市场均值，防小行业均值噪声
                grp_mean = ind_mean.where(ind_cnt >= C.INDUSTRY_MIN_COUNT, mkt_mean)
            else:
                # 无行业标签（连接器不支持）→ 回退全市场中性
                grp_mean = mkt_mean
            df[col] = ret - grp_mean
        elif market_neutral_target:
            df[col] = ret - mkt_mean
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
    # 近因权重：旧实现 groupby("symbol").cumcount()/len(df) 的分母是全面板行数
    # （含全部股票），商恒≈0，时间衰减实际失效。改为按 date 的全局百分位秩
    # （最早→0，最新→1），恢复 0.5→1.0 的时间衰减意图。
    date_weight = df["date"].rank(pct=True)
    df["weight"] = base_weight * (0.5 + 0.5 * date_weight)
    df["weight"] = df["weight"].clip(0.1, 10.0)
    return df


def _residualize_target(
    df: pd.DataFrame, target_col: str, extra_factors: list[str] | None = None
) -> None:
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
        _resid_one_date, include_groups=False
    )


# --------------------------------------------------------------------------
# Jane Street format transform
# --------------------------------------------------------------------------
def transform_to_jane_street_format(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map to (date_id, time_id, symbol_id, feature_NN..., responder_6, weight).

    返回 (df, feature_map)：feature_map = {因子名: 列号}（含 dummy 填充列），
    由调用方随缓存落盘 sidecar JSON——feature_NN 与因子名的对应关系随缓存
    签名变化，没有映射就无法事后对齐历史缓存的列含义（静默错位风险）。
    """
    unique_dates = sorted(df["date"].unique())
    unique_symbols = sorted(df["symbol"].unique())
    date_map = {d: i for i, d in enumerate(unique_dates)}
    symbol_map = {s: i for i, s in enumerate(unique_symbols)}

    df["date_id"] = df["date"].map(date_map).astype(np.int32)
    df["time_id"] = 0
    df["symbol_id"] = df["symbol"].map(symbol_map).astype(np.int32)

    # Exclude raw/intermediate columns; the rest are candidate features.
    exclude_cols = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
        "turnover_rate",
        "volume_ratio",
        "pct_chg",
        "industry",
        "snap_date",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "total_mv",
        "circ_mv",
        "net_profit",
        "revenue",  # raw financials; keep yoy/rank versions
        "date_id",
        "time_id",
        "symbol_id",
        "responder_6",
        "weight",
        "future_close",
    ] + [
        c
        for c in df.columns
        if c.startswith("ma_")
        or c.startswith("bb_upper")
        or c.startswith("bb_lower")
        or c.startswith("responder_fw_")
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Pad or truncate to exactly N_FEATURES.
    if len(feature_cols) > C.N_FEATURES:
        dropped = feature_cols[C.N_FEATURES :]
        warnings.warn(
            f"Named factor panel has {len(feature_cols)} candidate features but "
            f"N_FEATURES={C.N_FEATURES}; truncating drops {len(dropped)} columns: "
            f"{dropped[:20]}{'...' if len(dropped) > 20 else ''}. "
            "Raise N_FEATURES or disable factor families to keep them.",
            UserWarning,
            stacklevel=2,
        )
        feature_cols = feature_cols[: C.N_FEATURES]
    elif len(feature_cols) < C.N_FEATURES:
        for i in range(len(feature_cols), C.N_FEATURES):
            df[f"dummy_{i}"] = 0.0
            feature_cols.append(f"dummy_{i}")

    feature_rename = {old: f"feature_{i:02d}" for i, old in enumerate(feature_cols)}
    feature_map = {old: i for i, old in enumerate(feature_cols)}  # 因子名 → 列号
    df = df.rename(columns=feature_rename)

    output_cols = [C.DATE_COL, C.TIME_COL, C.SYMBOL_COL] + C.FEATURE_NAMES + [C.TARGET, C.WEIGHT]
    df = df[output_cols]
    df = df.dropna(subset=[C.TARGET]).reset_index(drop=True)
    return df, feature_map


def _factor_source_hash() -> str:
    """原生因子源码的内容哈希，并入 processed 缓存签名。

    因子代码（src/data/factors/*.py、option_features.py、macro_features.py）
    一改，产出的数据集内容就变，但旧签名不含代码版本，会静默命中陈旧缓存
    （缓存污染）。把源码内容哈希进签名后，任何因子代码改动都自动使缓存失效。
    """
    import glob

    base = os.path.dirname(os.path.abspath(__file__))
    paths = sorted(glob.glob(os.path.join(base, "factors", "*.py")))
    paths += [os.path.join(base, "option_features.py"), os.path.join(base, "macro_features.py")]
    h = hashlib.md5()
    for p in paths:
        if not os.path.exists(p):
            continue
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


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
        logger.info(f"数据源: {source} | universe: {index_code} | {len(symbols)} 只股票")
        logger.info(f"日期范围: {start_date} ~ {end_date}")

    # Cache the assembled (processed) dataset keyed by the request signature.
    # The signature MUST include every config knob that changes the engineered
    # feature set; otherwise turning factor families on/off returns a stale cache.
    sig = hashlib.md5(
        f"{source}|{index_code}|{sorted(symbols)}|{start_date}|{end_date}|"
        f"{C.FORWARD_WINDOW}|{C.MARKET_NEUTRAL_TARGET}|"
        f"{C.N_FEATURES}|"
        f"{C.USE_TECHNICAL_FEATURES}|{C.USE_ORTHOGONAL_FACTORS}|"
        f"{C.USE_ALPHA158_FACTORS}|{C.USE_GTJA191_FACTORS}|{C.GTJA191_ALPHA_SUBSET}|"
        f"{C.USE_FUNDAMENTALS}|{C.USE_AK_VALUE}|{C.USE_FINANCIAL_FACTORS}|{C.INDUSTRY_NEUTRAL}|"
        f"{C.INDUSTRY_MIN_COUNT}|{C.USE_NORTHBOUND}|{C.NORTHBOUND_MAX_FILL_DAYS}|"
        f"{C.USE_OPTION_FEATURES}|{C.OPTION_UNDERLYING}|{C.OPTION_SOURCE}|"
        f"{C.OPTION_MAX_FILL_DAYS}|"
        f"{C.USE_GLOBAL_MACRO}|{C.MACRO_MAX_FILL_DAYS}|{C.FUND_SAMPLE_DAYS}|"
        f"{C.USE_MACRO_INTERACTIONS}|{C.RESIDUAL_TARGET}|"
        f"{C.REGIME_IC_FEATURES}|"
        f"{C.USE_FORMULA_FACTORS}|{C.FORMULA_FACTOR_SUBSET}|"
        f"{get_factor_library_hash(C.FACTOR_LIBRARY_DIR)}|"
        f"{_factor_source_hash()}|"
        f"{C.MULTI_HORIZON_WINDOWS}".encode()
    ).hexdigest()[:12]
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    processed_cache = os.path.join(C.CACHE_DIR, f"dataset_{source}_{sig}.parquet")
    if os.path.exists(processed_cache):
        if verbose:
            logger.info(f"使用已缓存数据集: {processed_cache}")
        df = pd.read_parquet(processed_cache)
        if verbose:
            inspect_dataset(df, verbose=True)
        return df, True

    raw = connector.fetch_ohlcv(symbols, start_date, end_date)
    if verbose:
        logger.info(
            f"原始数据: {len(raw):,} 行, {raw['symbol'].nunique():,} 只股票, "
            f"{raw['date'].nunique():,} 个交易日"
        )

    # Downcast raw bars before factor engineering to contain whole-market memory.
    if "symbol" in raw.columns:
        raw["symbol"] = raw["symbol"].astype("category")
    raw = reduce_mem_usage(raw, float16_as32=True)
    df = compute_all_factors(raw)
    if C.USE_FUNDAMENTALS and isinstance(connector, SupportsFundamentals):
        fund = connector.fetch_fundamentals(
            start_date, end_date, C.FUND_SAMPLE_DAYS, symbols=symbols
        )
        df = merge_fundamentals(df, fund)
    if C.USE_FINANCIAL_FACTORS and isinstance(connector, SupportsFinancialFactors):
        fin = connector.fetch_financial_factors(symbols)
        df = merge_financial_factors(df, fin)
    # 行业特征族与 INDUSTRY_NEUTRAL（目标开关）解耦：连接器支持行业映射就
    # 始终附加 industry 标签与行业内特征（行业内 rank、行业动量）；
    # INDUSTRY_NEUTRAL 只控制目标是否按行业去均值（prepare_target_and_weight）。
    if isinstance(connector, SupportsIndustryMap):
        try:
            df = add_industry_features(df, connector.get_industry_map())
        except Exception as e:
            logger.warning(f"  ⚠️ 行业特征获取失败，跳过: {e}")
    df = merge_northbound_features(df, connector)
    # Raw/intermediate columns are no longer needed; drop them before option IV
    # and target computation to keep memory flat on whole-market panels.
    df = _drop_intermediate_cols(df, keep_trade_cols=False)
    df = merge_option_iv_features(df)
    df = merge_global_macro_features(df)
    df = prepare_target_and_weight(
        df, forward_window=C.FORWARD_WINDOW, market_neutral_target=C.MARKET_NEUTRAL_TARGET
    )
    if C.REGIME_IC_FEATURES:
        df = add_factor_regime_features(df)
    df, feature_map = transform_to_jane_street_format(df)
    df = reduce_mem_usage(df, float16_as32=False)
    df.to_parquet(processed_cache, index=False)
    # sidecar：因子名 → feature_NN 列号映射随缓存落盘，防静默错位
    sidecar = processed_cache.replace(".parquet", ".features.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(feature_map, f, ensure_ascii=False, indent=1)

    if verbose:
        logger.info(
            f"最终数据集: {len(df):,} 行, {df[C.DATE_COL].nunique()} 个交易日, "
            f"{df[C.SYMBOL_COL].nunique()} 只股票"
        )
        inspect_dataset(df, verbose=True)
    return df, True


# --------------------------------------------------------------------------
# Factor-panel disk cache (signature hit / tail-incremental rebuild)
# --------------------------------------------------------------------------
# load_factor_panel 的调用方（次日交易流水线、walkforward、因子评估）在同一配置
# 下反复构建全市场全历史面板（~330 列）。因子全部是"逐股时间序列滚动 + 同日
# 截面"操作：某交易日的因子值只依赖该股过去 MAX_ROLLING_WINDOW 天历史（ewm 类
# 无限记忆因子的截断误差见 factors/__init__.py 注释，实测远低于 float32 精度），
# 标签只依赖未来 max(FORWARD_WINDOW, MULTI_HORIZON_WINDOWS) 天。因此面板可以
# 整体落盘：每晚只重算尾部 N 天窗口并与缓存旧部分拼接。
#
# 缓存内容 = 各 keep_* 标志的"超集面板"（全部交易列保留 + 隐藏列 adj_close），
# 各调用方的视图差异由 _factor_panel_view 在读取后廉价复现（列裁剪 + 无标签行
# 丢弃）。adj_close 必须进缓存：否则尾部重算时 technical 族会把复权锚点重置到
# 窗口起点，MACD 等非尺度不变因子整体错一个乘性因子——缓存面板旧部分逐股
# 携带该列，增量路径用它把复权序列前向链接到新交易日（_extend_adj_close）。
_PANEL_CACHE_PARQUET = "factor_panel_cache.parquet"
_PANEL_CACHE_META = "factor_panel_cache.sig.json"
# 拼接保留段（交易日数）= 最长标签 horizon + 该余量。两次运行间积累的新交易日
# 数（staleness）不超过余量时才允许增量，否则缓存里仍有未成熟标签行拼不到。
_PANEL_CACHE_STALE_SLACK = 40
# 任何 keep_* 组合都不返回、且尾部重算也不需要旧值的列——不进缓存文件。
_PANEL_CACHE_DROP = {
    "volume",
    "future_close",
    "weight",
    "log_return",
    "snap_date",
    "industry",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "total_mv",
    "circ_mv",
    "net_profit",
    "revenue",
}


def _panel_cache_windows() -> tuple[int, int]:
    """(raw lookback N, keep span) in trading days for the tail-incremental rebuild.

    keep span = 最长标签 horizon + 余量：保证缓存写出时未成熟的标签行全部落在
    重算拼接段内。N 再加最长滚动窗口，保证拼接段首行的因子看到完整历史。
    """
    max_fw = max([C.FORWARD_WINDOW, *getattr(C, "MULTI_HORIZON_WINDOWS", [])])
    keep_span = max_fw + _PANEL_CACHE_STALE_SLACK
    return MAX_ROLLING_WINDOW + keep_span, keep_span


def _panel_cache_paths() -> tuple[str, str]:
    return (
        os.path.join(C.CACHE_DIR, _PANEL_CACHE_PARQUET),
        os.path.join(C.CACHE_DIR, _PANEL_CACHE_META),
    )


def _read_panel_cache_meta(meta_path: str) -> dict | None:
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_panel_cache(
    df: pd.DataFrame,
    sig: str,
    parquet_path: str,
    meta_path: str,
    req_start: str,
    req_end: str,
) -> None:
    """Atomic cache write: temp file + os.replace, readers never see a torn file.

    parquet 先落盘、meta 后落盘：并发读者最坏看到"旧 meta + 新 parquet"→ 签名
    不一致 → 全量重算（安全方向），不会拿到错配内容。meta 同时记录构建时的
    请求区间（req_*）与面板实际数据区间（data_*，暖机/质量过滤会裁掉头部若干
    交易日）——命中判定用 req_start（请求起点不早于构建请求即可由缓存服务），
    增量判定用 data_end。
    """
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    out = df[[c for c in df.columns if c not in _PANEL_CACHE_DROP]]
    tmp = f"{parquet_path}.{os.getpid()}.tmp"
    out.to_parquet(tmp, index=False, compression="snappy")
    os.replace(tmp, parquet_path)
    meta = {
        "sig": sig,
        "req_start": req_start,
        "req_end": req_end,
        "data_start": str(pd.Timestamp(out["date"].min()).date()),
        "data_end": str(pd.Timestamp(out["date"].max()).date()),
        "n_rows": int(len(out)),
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    tmp_m = f"{meta_path}.{os.getpid()}.tmp"
    with open(tmp_m, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    os.replace(tmp_m, meta_path)
    logger.info(f"因子面板缓存已写入: {parquet_path} ({len(out):,} 行)")


def _factor_panel_signature(source: str, index_code: str, whole_market: bool) -> str:
    """Content signature of every knob that changes panel VALUES.

    日期范围刻意排除：覆盖范围单独判定，start_date 随每日滚动也能命中。因子
    代码用内容哈希（factors/*.py + option/macro 特征模块 + 本模块 + 连接器），
    改任何相关文件即失效回退全量。
    """
    base = os.path.dirname(os.path.abspath(__file__))
    code = hashlib.md5()
    for name in ("__init__.py", "base.py", "tushare.py"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                code.update(f.read())
    n_win, keep_span = _panel_cache_windows()
    return hashlib.md5(
        (
            f"panel-v1|{source}|{index_code}|{whole_market}|"
            f"{C.FORWARD_WINDOW}|{C.MARKET_NEUTRAL_TARGET}|"
            f"{C.USE_TECHNICAL_FEATURES}|{C.USE_ORTHOGONAL_FACTORS}|"
            f"{C.USE_ALPHA158_FACTORS}|{C.USE_GTJA191_FACTORS}|{C.GTJA191_ALPHA_SUBSET}|"
            f"{C.USE_FUNDAMENTALS}|{C.USE_AK_VALUE}|{C.USE_FINANCIAL_FACTORS}|"
            f"{C.INDUSTRY_NEUTRAL}|{C.INDUSTRY_MIN_COUNT}|"
            f"{C.USE_NORTHBOUND}|{C.NORTHBOUND_MAX_FILL_DAYS}|"
            f"{C.USE_OPTION_FEATURES}|{C.OPTION_UNDERLYING}|{C.OPTION_SOURCE}|"
            f"{C.OPTION_MAX_FILL_DAYS}|{C.USE_GLOBAL_MACRO}|{C.MACRO_MAX_FILL_DAYS}|"
            f"{C.FUND_SAMPLE_DAYS}|{C.USE_MACRO_INTERACTIONS}|{C.RESIDUAL_TARGET}|"
            f"{C.REGIME_IC_FEATURES}|"
            f"{C.USE_FORMULA_FACTORS}|{C.FORMULA_FACTOR_SUBSET}|"
            f"{get_factor_library_hash(C.FACTOR_LIBRARY_DIR)}|{_factor_source_hash()}|"
            f"{code.hexdigest()[:12]}|"
            f"{C.MULTI_HORIZON_WINDOWS}|{C.UNIVERSE_MODE}|{C.UNIVERSE_SIZE}|"
            f"{getattr(C, 'MIN_PRICE', 2.0)}|{getattr(C, 'MIN_AMOUNT_WAN', 2000)}|"
            f"{getattr(C, 'MAIN_BOARD_ONLY', False)}|{n_win}|{keep_span}"
        ).encode()
    ).hexdigest()[:12]


def _extend_adj_close(raw: pd.DataFrame, hint: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the full-history-anchored adj_close/_adj_cp for a truncated raw window.

    technical.py anchors adj_close at each symbol's FIRST loaded close and
    accumulates the pct_chg cumprod (_adj_cp) in float32; a tail window would
    re-anchor at the window start and silently rescale MACD（唯一非尺度不变的
    adj_close 消费者），而任何近似接续的 1ulp 偏差都会被 stoch_k 的比率形态
    放大 ~100 倍。因此这里逐股逐日原样复现 f32 运算序列：hint 带缓存面板的
    adj_close/_adj_cp/首行 close，重叠行取缓存值，锚点之后从缓存的 f32 累加器
    原值继续连乘——与全量构建逐位一致。缓存外的新上市股票按 technical.py 原
    逻辑以窗口首收价为锚。
    """
    df = raw[["symbol", "date", "close", "pct_chg"]].copy()
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = df["date"].astype("datetime64[ns]")
    h = hint.rename(columns={"adj_close": "_hist", "_adj_cp": "_hist_cp", "close": "_hist_close"})
    h["symbol"] = h["symbol"].astype(str)
    h["date"] = h["date"].astype("datetime64[ns]")
    df = df.merge(
        h[["symbol", "date", "_hist", "_hist_cp"]], on=["symbol", "date"], how="left", sort=False
    )
    df = df.sort_values(["symbol", "date"], kind="mergesort")
    first_close_map = h.sort_values("date").groupby("symbol")["_hist_close"].first()
    hist = df["_hist"].to_numpy(np.float64)
    hist_cp = df["_hist_cp"].to_numpy(np.float64)
    ratio = (1.0 + df["pct_chg"].fillna(0.0) / 100.0).to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    sym = df["symbol"].to_numpy()
    adj = hist.copy()
    cp_out = hist_cp.copy()
    _, starts = np.unique(sym, return_index=True)
    starts = np.r_[starts, len(df)]
    for s, e in zip(starts[:-1], starts[1:]):
        known = ~np.isnan(hist_cp[s:e])
        if known.any():
            ai = s + int(np.nonzero(known)[0][-1])  # 该股窗口内最后缓存行
            fc = np.float32(first_close_map.get(sym[s], np.nan))
            if not np.isfinite(fc) or fc == 0.0:
                fc = np.float32(close[s])
            cp = np.float32(hist_cp[ai])  # f32 累加器原值，接续无损
            for t in range(ai + 1, e):
                cp = np.float32(cp * np.float32(ratio[t]))
                cp_out[t] = cp
                adj[t] = np.float32(fc * cp)
        else:
            # 新上市（缓存外）：与全量构建一致——窗口首收价 × f32 累乘
            fc32 = np.float32(close[s])
            cp = np.float32(1.0)
            for t in range(s, e):
                cp = np.float32(cp * np.float32(ratio[t]))
                cp_out[t] = cp
                adj[t] = np.float32(fc32 * cp)
    out = pd.DataFrame(
        {"adj_close": np.float32(adj), "_adj_cp": np.float32(cp_out)}, index=df.index
    )
    return out.sort_index()


def _build_factor_panel(
    connector: DataConnector,
    symbols: list[str] | None,
    start_date: str,
    end_date: str,
    verbose: bool = True,
    raw_start_date: str | None = None,
    adj_close_hint: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the canonical superset panel (fetch -> factor families -> merges -> target).

    "Superset": built as if keep_unlabeled/keep_price/keep_trade_cols were all
    True, PLUS the hidden adj_close column, so a single on-disk cache serves
    every keep_* combination via _factor_panel_view. raw_start_date /
    adj_close_hint are used only by the incremental cache path to rebuild a
    trailing raw window with the full-history adjustment anchor intact.
    """
    whole_market = (
        getattr(C, "UNIVERSE_MODE", "all") == "all" or getattr(C, "UNIVERSE_SIZE", 0) == 0
    )
    raw_start = raw_start_date or start_date
    fund_symbols = symbols
    if whole_market and isinstance(connector, SupportsWholeMarketFetch):
        if verbose:
            logger.info(
                f"因子面板: {connector.name} | 全市场(按日拉取) | {start_date} ~ {end_date}"
            )
        raw = connector.fetch_ohlcv_whole_market(raw_start, end_date)
        raw = _filter_tradable_universe(raw, verbose=verbose)
        # 基本面按数据源区分：akshare 东财个股估值（stock_value_em）免费无限速，
        # 逐股全历史 + 增量缓存，全市场 ~5700 只可行——USE_AK_VALUE 独立开关；
        # tushare daily_basic/fina_indicator 限速过严（全市场不可行），财务因子
        # 在全市场模式下保持禁用。
        fund_symbols = sorted(raw["symbol"].astype(str).unique().tolist())
        use_fund = C.USE_FUNDAMENTALS and getattr(C, "USE_AK_VALUE", True)
        use_fin = False
        if verbose and C.USE_FINANCIAL_FACTORS:
            logger.warning(
                "  ⚠️ 全市场模式: 财务因子已自动禁用"
                "(tushare daily_basic/fina_indicator 限速, 全市场不可行)。"
            )
        if verbose and C.USE_FUNDAMENTALS and not use_fund:
            logger.warning("  ⚠️ 全市场模式: akshare 估值因子被 USE_AK_VALUE=0 禁用。")
        if verbose and use_fund:
            logger.info(
                f"  全市场模式: akshare 估值因子启用，覆盖 {len(fund_symbols)} 只"
                "(缺缓存的股票首次回填约需几十分钟，之后每日增量)"
            )
    else:
        if verbose:
            logger.info(
                f"因子面板: {connector.name} | {len(symbols)} 只股票 | {start_date} ~ {end_date}"
            )
        raw = connector.fetch_ohlcv(symbols, raw_start, end_date)
        use_fund, use_fin = C.USE_FUNDAMENTALS, C.USE_FINANCIAL_FACTORS
    # Downcast before feature engineering: whole-market panels are huge and
    # factor formulas do not need float64 precision.
    if "symbol" in raw.columns:
        raw["symbol"] = raw["symbol"].astype("category")
    raw = reduce_mem_usage(raw, float16_as32=True)
    if adj_close_hint is not None and "pct_chg" in raw.columns:
        # 增量路径：用缓存面板的复权序列做锚点，保持与全历史构建一致的量纲。
        # 注入列与全量构建同 dtype（f32）——否则 technical 族内 adj_ratio/
        # stoch/MACD 的中间运算精度不同，值无法对齐。
        raw = raw.reset_index(drop=True)
        ext = _extend_adj_close(raw, adj_close_hint)
        raw["adj_close"] = ext["adj_close"]
        raw["_adj_cp"] = ext["_adj_cp"]
    df = compute_all_factors(raw)
    if use_fund and isinstance(connector, SupportsFundamentals):
        # 估值只经 merge_asof 并入 raw 窗口（输出以 raw 行为准），raw_start 之前
        # 的全历史估值对结果无影响：增量路径里 raw_start ~ 窗口起点，起点附近
        # 的暖机行（无匹配估值 → NaN）在拼接时会被丢弃；全量路径 raw_start ==
        # start_date，行为不变。收窄拉取范围可大幅降低估值拼装内存峰值。
        fund = connector.fetch_fundamentals(
            raw_start, end_date, C.FUND_SAMPLE_DAYS, symbols=fund_symbols
        )
        df = merge_fundamentals(df, fund)
    if use_fin and isinstance(connector, SupportsFinancialFactors):
        fin = connector.fetch_financial_factors(symbols)
        df = merge_financial_factors(df, fin)
    # 行业特征族与 INDUSTRY_NEUTRAL（目标开关）解耦：连接器支持行业映射就
    # 始终附加 industry 标签与行业内特征（行业内 rank、行业动量）；
    # INDUSTRY_NEUTRAL 只控制目标是否按行业去均值（prepare_target_and_weight）。
    if isinstance(connector, SupportsIndustryMap):
        try:
            df = add_industry_features(df, connector.get_industry_map())
        except Exception as e:
            logger.warning(f"  ⚠️ 行业特征获取失败，跳过: {e}")
    df = merge_northbound_features(df, connector)
    # Raw/intermediate columns are no longer needed; drop them before option IV
    # and target computation to keep memory flat on whole-market panels. The
    # canonical superset panel always retains the trade columns; per-caller
    # visibility is decided by _factor_panel_view.
    df = _drop_intermediate_cols(df, keep_trade_cols=True)
    df = merge_option_iv_features(df)
    df = merge_global_macro_features(df)
    df = prepare_target_and_weight(
        df, forward_window=C.FORWARD_WINDOW, market_neutral_target=C.MARKET_NEUTRAL_TARGET
    )
    if C.REGIME_IC_FEATURES:
        df = add_factor_regime_features(df)
    return df


def _factor_panel_view(
    df: pd.DataFrame,
    keep_unlabeled: bool,
    keep_price: bool,
    keep_trade_cols: bool,
    verbose: bool,
) -> pd.DataFrame:
    """Per-caller column/row view over the canonical superset panel.

    Reproduces exactly what the pre-cache pipeline returned for each keep_*
    combination: the superset panel is built with all keeps on, so the view
    re-applies the flag-dependent drops (cheap column selection + dropna).
    """
    # Keep named factors: drop raw OHLCV, intermediates, and categorical industry label.
    drop = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "future_close",
        "weight",
        "log_return",
        "snap_date",
        "industry",
        "adj_close",
        "_adj_cp",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "total_mv",
        "circ_mv",
        "net_profit",
        "revenue",
    }
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
    bb_drop = {
        c
        for c in df.columns
        if c.startswith("ma_") or c.startswith("bb_upper") or c.startswith("bb_lower")
    }
    if keep_trade_cols:
        # Daily trading engine needs the raw OHLC bars (next-day fill simulation
        # against the price range / limit bands) and the Bollinger band levels
        # (entry/exit timing). Keep them as META columns — list_factor_columns
        # excludes open/high/low/bb_* so the model never sees raw price levels.
        drop -= {"open", "high", "low", "close"}
        bb_drop -= {"bb_upper_20", "bb_lower_20"}
    else:
        # 旧流程在 _drop_intermediate_cols(keep_trade_cols=False) 阶段丢弃
        # pct_chg；超集面板保留了它，视图中补齐同样的丢弃。
        drop |= {"pct_chg"}
    drop |= bb_drop
    # 列裁剪用 inplace drop（整块释放，内存平），不用 df[keep]（全帧拷贝，
    # 全市场面板一次拷贝 ~10GB，与读缓存峰值叠加会 OOM）。调用方独占该帧，
    # 原地修改安全。
    df.drop(columns=[c for c in df.columns if c in drop], inplace=True)
    if not keep_unlabeled:
        tgt_na = df[C.TARGET].isna().to_numpy()
        if tgt_na.any():
            # 快速路径：未打标行是排序面板的最末 FORWARD_WINDOW 个交易日
            # （连续后缀），iloc 行切片返回视图、零拷贝；存在散点 NaN 时才
            # 回退 dropna 全帧拷贝。
            first_na = int(tgt_na.argmax())
            if tgt_na[first_na:].all():
                df = df.iloc[:first_na].reset_index(drop=True)
            else:
                df = df.dropna(subset=[C.TARGET]).reset_index(drop=True)
    if verbose:
        n_fac = len([c for c in df.columns if c not in ("date", "symbol", C.TARGET, "close")])
        logger.info(
            f"因子面板: {len(df):,} 行, {n_fac} 个命名因子, " f"{df['symbol'].nunique()} 只股票"
        )
    return df


def _try_incremental_factor_panel(
    connector: DataConnector,
    symbols: list[str] | None,
    start_date: str,
    end_date: str,
    cached: pd.DataFrame,
    meta: dict,
    sig: str,
    parquet_path: str,
    meta_path: str,
    verbose: bool,
) -> pd.DataFrame | None:
    """Rebuild only the trailing factor window and splice it onto the cache.

    返回 None（调用方回退全量）当：尾部增长超出增量包络（staleness 超过
    _PANEL_CACHE_STALE_SLACK，缓存里会残留未成熟标签行）、缓存不含 adj_close
    锚点列、或缓存 schema 与新构建不一致。
    """
    req_start, req_end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    c_end = pd.Timestamp(meta["data_end"])
    built_req_start = pd.Timestamp(meta.get("req_start") or meta["data_start"])
    if not (req_end > c_end and built_req_start <= req_start <= c_end):
        return None
    if "adj_close" not in cached.columns or "_adj_cp" not in cached.columns:
        return None
    # 便宜的日历/行情预判：请求尾部没有新交易日（周末/节假日重跑），或虽有
    # 交易日但行情尚未入库（盘中/数据延迟）时，缓存即最新，直接返回，不必
    # 构建尾部窗口。
    try:
        days = connector.trade_dates(
            (c_end + timedelta(days=1)).strftime("%Y-%m-%d"), req_end.strftime("%Y-%m-%d")
        )
    except Exception:
        days = None
    if days is not None:
        if not days:
            if verbose:
                logger.info("因子面板缓存命中: 请求尾部无新交易日（缓存即最新）")
            return cached
        avail = []
        for d in days:
            try:
                avail.append(connector.bar_available(d))
            except Exception:
                avail.append(None)
        if all(a is False for a in avail):
            if verbose:
                logger.info("因子面板缓存命中: 新交易日行情尚未入库（缓存即最新）")
            return cached
    n_win, keep_span = _panel_cache_windows()
    max_fw = keep_span - _PANEL_CACHE_STALE_SLACK
    # 便宜的日历日上界预判：staleness（交易日）超包络时不必先构建尾部窗口
    if (req_end - c_end).days > int((keep_span - max_fw) * 1.5):
        return None
    # _filter_tradable_universe 用 rolling(20, min_periods=10) 的成交额中位数：
    # 窗口最前的部分窗口过滤结果与全历史不一致，多取 20 个交易日暖机，保证
    # 拼接段 252 天回看内的过滤行集与全量构建逐行一致。
    n_fetch = n_win + 20
    # 交易日 ≈ 0.7×自然日，多留余量保证取够 n_fetch 个交易日
    raw_start = (req_end - timedelta(days=int(n_fetch * 1.6) + 15)).strftime("%Y-%m-%d")
    tail = _build_factor_panel(
        connector,
        symbols,
        start_date,
        end_date,
        verbose,
        raw_start_date=raw_start,
        adj_close_hint=cached[["symbol", "date", "close", "adj_close", "_adj_cp"]],
    )
    dates = np.sort(tail["date"].unique())
    if len(dates) < n_fetch:
        return None  # 尾部窗口不足（缓存过旧或数据缺口）
    stale = int((dates > c_end).sum())
    if stale == 0:
        # 尾部构建未发现新交易日（当日数据尚未入库），缓存即最新
        return cached
    if stale > keep_span - max_fw:
        # stale 超上限时缓存仍含未成熟标签行
        return None
    splice = dates[-keep_span]
    if any(c not in tail.columns for c in cached.columns):
        return None  # schema 漂移（连接器列变化等）
    new_part = tail[tail["date"] >= splice][list(cached.columns)].copy()
    old_part = cached[cached["date"] < splice].copy()
    # dtype 对齐：新尾部按缓存 schema 逐列转换（f32/f64 的 downcast 决策可能
    # 因窗口取值范围不同而不一致）；symbol category 取并集——窗口内可能有
    # 缓存写出后新上市的股票。
    for col in cached.columns:
        if col == "symbol":
            continue
        target_dt = cached[col].dtype
        if new_part[col].dtype != target_dt:
            try:
                new_part[col] = new_part[col].astype(target_dt)
            except (ValueError, TypeError):
                return None
    if isinstance(cached["symbol"].dtype, pd.CategoricalDtype) and isinstance(
        new_part["symbol"].dtype, pd.CategoricalDtype
    ):
        cats = cached["symbol"].cat.categories.union(new_part["symbol"].cat.categories)
        old_part["symbol"] = old_part["symbol"].cat.set_categories(cats)
        new_part["symbol"] = new_part["symbol"].cat.set_categories(cats)
    else:
        cats = sorted(set(old_part["symbol"].astype(str)) | set(new_part["symbol"].astype(str)))
        old_part["symbol"] = pd.Categorical(old_part["symbol"].astype(str), categories=cats)
        new_part["symbol"] = pd.Categorical(new_part["symbol"].astype(str), categories=cats)
    combined = pd.concat([old_part, new_part], ignore_index=True)
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
    if verbose:
        logger.info(
            f"因子面板增量重算: 尾部 {stale} 个新交易日，重算窗口 {len(dates)} 天，"
            f"拼接点 {pd.Timestamp(splice):%Y-%m-%d}"
        )
    # 覆盖起点沿用缓存构建时的请求起点（旧部分未变）
    _write_panel_cache(combined, sig, parquet_path, meta_path, meta["req_start"], end_date)
    return combined


def _cached_factor_panel(
    connector: DataConnector,
    source: str,
    index_code: str,
    whole_market: bool,
    symbols: list[str] | None,
    start_date: str,
    end_date: str,
    verbose: bool,
) -> pd.DataFrame:
    """Serve the canonical superset panel from the disk cache when possible.

    三条路径：
      命中 —— 请求 [start, end] ⊆ 缓存覆盖范围 → 过滤后直接返回
      增量 —— 仅尾部新增交易日 → 只重算尾部 N 天原始窗口并与缓存拼接
      全量 —— 无缓存/签名不一致/缓存损坏/超出增量包络 → 重建并写缓存
    """

    def _range_view(df: pd.DataFrame) -> pd.DataFrame:
        """按请求范围过滤；全覆盖时直接返回原帧，避免 12GB+ 的全帧布尔拷贝
        （32GB 机器上读缓存 + 过滤拷贝的叠加峰值足以触发 OOM）。"""
        mask = (df["date"] >= req_start) & (df["date"] <= req_end)
        return df if bool(mask.all()) else df[mask]

    sig = _factor_panel_signature(source, index_code, whole_market)
    parquet_path, meta_path = _panel_cache_paths()
    req_start, req_end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    meta = _read_panel_cache_meta(meta_path)
    if meta is not None and meta.get("sig") == sig and os.path.exists(parquet_path):
        try:
            cached = _read_panel_parquet(parquet_path)
        except Exception as e:
            logger.warning(f"  ⚠️ 因子面板缓存读取失败，回退全量重算: {e}")
            cached = None
        if cached is not None:
            c_end = pd.Timestamp(meta["data_end"])
            built_req_start = pd.Timestamp(meta.get("req_start") or meta["data_start"])
            if built_req_start <= req_start and req_end <= c_end:
                if verbose:
                    logger.info(
                        f"因子面板缓存命中: {parquet_path} "
                        f"(数据 {meta['data_start']} ~ {meta['data_end']}, {meta['n_rows']:,} 行)"
                    )
                return _range_view(cached)
            combined = _try_incremental_factor_panel(
                connector,
                symbols,
                start_date,
                end_date,
                cached,
                meta,
                sig,
                parquet_path,
                meta_path,
                verbose,
            )
            if combined is not None:
                return _range_view(combined)
            if verbose:
                logger.info("因子面板缓存不满足增量条件，回退全量重算")
    df = _build_factor_panel(connector, symbols, start_date, end_date, verbose)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    _write_panel_cache(df, sig, parquet_path, meta_path, start_date, end_date)
    return df


def _read_panel_parquet(parquet_path: str) -> pd.DataFrame:
    """低峰值读取全市场面板缓存。

    pd.read_parquet 的默认转换路径（to_pandas 不带 self_destruct）会让
    Arrow 解压缓冲与 pandas 帧同时驻留，2.5GB snappy parquet 峰值实测
    ~18GB（32GB 机器上直接触发 OOM）。self_destruct=True 在转换过程中
    逐列释放 Arrow 内存，峰值压到单边 ~10GB；split_blocks 只影响 block
    布局，dtype/值与默认路径逐项一致。
    """
    import pyarrow.parquet as pq

    t = pq.read_table(parquet_path)
    try:
        return t.to_pandas(split_blocks=True, self_destruct=True)
    finally:
        del t


def _panel_row_order_ok(df: pd.DataFrame) -> bool:
    """逐行检查面板是否已按 (date, symbol) 非降序排列（缓存写出时即此序）。

    只读 date/symbol 两列（~100MB 临时量），避免对 12GB 全帧做无谓的
    sort_values 拷贝。symbol 为 category 时按 codes 比较——与 sort_values
    对 category 的排序键一致（同语义，不会误判）。
    """
    n = len(df)
    if n < 2:
        return True
    d = df["date"].to_numpy()
    s = df["symbol"]
    codes = s.cat.codes.to_numpy() if isinstance(s.dtype, pd.CategoricalDtype) else s.to_numpy()
    d_prev, d_next = d[:-1], d[1:]
    if bool((d_next < d_prev).any()):
        return False
    eq = d_next == d_prev
    return not bool((eq & (codes[1:] < codes[:-1])).any())


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

    磁盘缓存（FACTOR_PANEL_CACHE=1 默认开，仅服务默认宇宙的调用——显式
    symbols 的研究型调用直接全量构建）：签名（因子配置 + 因子代码内容哈希 +
    全市场过滤参数，不含日期范围）一致时，请求区间 ⊆ 缓存覆盖 → 直接过滤返回；
    仅尾部新增交易日 → 尾部 N 天窗口增量重算拼接（N 见 _panel_cache_windows）；
    否则全量重建并覆写缓存。
    """
    source = source or C.DATA_SOURCE
    index_code = index_code or C.DEFAULT_INDEX
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    connector = get_connector(source)
    whole_market = (
        getattr(C, "UNIVERSE_MODE", "all") == "all" or getattr(C, "UNIVERSE_SIZE", 0) == 0
    )
    caller_symbols = symbols
    if symbols:
        pass
    elif C.DEFAULT_SYMBOLS:
        symbols = C.DEFAULT_SYMBOLS
    elif whole_market:
        symbols = None  # whole-market: fetch every stock by date
    else:
        symbols = connector.get_universe(index_code, start_date, end_date)

    cacheable = (
        getattr(C, "FACTOR_PANEL_CACHE", True) and caller_symbols is None and not C.DEFAULT_SYMBOLS
    )
    if cacheable:
        df = _cached_factor_panel(
            connector, source, index_code, whole_market, symbols, start_date, end_date, verbose
        )
    else:
        df = _build_factor_panel(connector, symbols, start_date, end_date, verbose)
    # 统一行序 (date, symbol)：缓存命中/增量/全量/无缓存四条路径输出一致。
    # 缓存写出前已按同键排序（_write_panel_cache），命中路径逐行单调性检查
    # 通过即零拷贝跳过——全帧 sort_values 的拷贝在 32GB 机器上足以 OOM。
    if not _panel_row_order_ok(df):
        df = df.sort_values(["date", "symbol"], kind="mergesort")
    df = df.reset_index(drop=True)
    out = _factor_panel_view(
        df,
        keep_unlabeled=keep_unlabeled,
        keep_price=keep_price,
        keep_trade_cols=keep_trade_cols,
        verbose=verbose,
    )
    # pyarrow 的 mimalloc 内存池在 read_parquet 后保留数 GB 空闲页不还 OS
    # （实测全市场缓存读入后 RSS 8.1GB vs 活对象 3.9GB），显式归还；
    # 只释放未引用页，不影响返回面板的数据。
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    return out


def list_factor_columns(panel: pd.DataFrame) -> list[str]:
    """Factor (non-meta) columns of a panel from load_factor_panel."""
    meta = {
        "date",
        "symbol",
        C.TARGET,
        C.DATE_COL,
        C.TIME_COL,
        C.SYMBOL_COL,
        C.WEIGHT,
        "close",
        "adj_close",
        # daily-trading meta (kept by keep_trade_cols, never model features):
        "open",
        "high",
        "low",
        "bb_upper_20",
        "bb_lower_20",
    }
    # multi-horizon targets (responder_fw_*) are labels, not features — 按前缀
    # 排除而非硬编码名单，改 MULTI_HORIZON_WINDOWS 后新标签也不会混进特征
    # （标签进特征 = 直接前视泄露）。
    return [c for c in panel.columns if c not in meta and not c.startswith("responder_fw_")]


def inspect_dataset(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Lightweight dataset quality checks (absorbed from a DatasetInspector).

    Surfaces three things that silently wreck a cross-sectional backtest:
      - label (target) statistics: sanity of the forward-return target
      - cross-section coverage: stocks per date (too few => unstable ranking)
      - feature NaN-by-date irregularity: a crude lookahead-bias smell test
    Returns a dict of findings; logs a summary when verbose.
    """
    findings: dict = {"warnings": []}

    y = df[C.TARGET]
    findings["label"] = {
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
        "q01": float(y.quantile(0.01)),
        "q99": float(y.quantile(0.99)),
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
            f"部分交易日横截面股票数过少 (最小={per_date.min()})，排序策略可能不稳定"
        )

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
        logger.info("数据集质量检查：")
        logger.info(
            f"  标签 responder_6: 均值={lab['mean']:+.5f} 标准差={lab['std']:.5f} "
            f"正收益比例={lab['positive_ratio']:.1%}"
        )
        logger.info(
            f"  横截面覆盖: {findings['coverage']['n_dates']} 个交易日, "
            f"日均 {findings['coverage']['avg_stocks_per_date']:.0f} 只股票 "
            f"(最少 {findings['coverage']['min_stocks_per_date']} 只)"
        )
        if findings["warnings"]:
            for w in findings["warnings"]:
                logger.warning(f"  ⚠️ {w}")
        else:
            logger.info("  ✅ 未发现明显异常")
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
    val_dates = dates[-(n_test + n_val) : -n_test]
    train_dates = dates[: -(n_test + n_val)]
    return train_dates, val_dates, test_dates
