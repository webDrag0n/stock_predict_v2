"""Unified factor registry combining native (Python) and formula (YAML) factors.

Provides a queryable catalog of all factors in the system, regardless of
source.  Also serves as the CLI entry point for factor discovery:

    python -m src.data.factors.registry list
    python -m src.data.factors.registry list --category momentum
    python -m src.data.factors.registry search volume
    python -m src.data.factors.registry info mom_5d_raw
    python -m src.data.factors.registry validate
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field

from .schema import FactorDefinition, load_factor_definitions
from .parser import validate_formula

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified metadata
# ---------------------------------------------------------------------------


@dataclass
class FactorInfo:
    """Lightweight unified factor record for discovery and reporting."""

    id: str
    name: str
    category: str
    source: str  # "native:technical", "native:gtja191", "formula", etc.
    description: str
    formula: str  # empty string for native factors
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    produces: str = ""  # = id
    enabled: bool = True


# ---------------------------------------------------------------------------
# Native factor catalog (hardcoded — the "source of truth" for Python factors)
# ---------------------------------------------------------------------------

NATIVE_FACTOR_METADATA: dict[str, list[dict]] = {
    "technical": [
        # Returns & momentum
        {"id": "return_1d", "name": "1日收益率", "category": "returns",
         "description": "adj_close / shift(1) - 1", "tags": ["return", "short-term"]},
        {"id": "return_5d", "name": "5日收益率", "category": "returns",
         "description": "adj_close / shift(5) - 1", "tags": ["return", "medium-term"]},
        {"id": "return_10d", "name": "10日收益率", "category": "returns",
         "description": "adj_close / shift(10) - 1", "tags": ["return", "medium-term"]},
        {"id": "return_20d", "name": "20日收益率", "category": "returns",
         "description": "adj_close / shift(20) - 1", "tags": ["return", "long-term"]},
        {"id": "momentum_5", "name": "5日动量", "category": "momentum",
         "description": "adj_close / shift(5) - 1 (same as return_5d)", "tags": ["momentum"]},
        {"id": "momentum_10", "name": "10日动量", "category": "momentum",
         "description": "adj_close / shift(10) - 1", "tags": ["momentum"]},
        {"id": "momentum_20", "name": "20日动量", "category": "momentum",
         "description": "adj_close / shift(20) - 1", "tags": ["momentum"]},
        # Volatility
        {"id": "volatility_5", "name": "5日波动率", "category": "volatility",
         "description": "5d rolling std of log_return", "tags": ["volatility"]},
        {"id": "volatility_10", "name": "10日波动率", "category": "volatility",
         "description": "10d rolling std of log_return", "tags": ["volatility"]},
        {"id": "volatility_20", "name": "20日波动率", "category": "volatility",
         "description": "20d rolling std of log_return", "tags": ["volatility"]},
        # Volume
        {"id": "volume_ma_ratio_5", "name": "5日量比", "category": "volume",
         "description": "volume / 5d MA of volume", "tags": ["volume"]},
        {"id": "volume_ma_ratio_20", "name": "20日量比", "category": "volume",
         "description": "volume / 20d MA of volume", "tags": ["volume"]},
        # Moving averages
        {"id": "ma_5", "name": "5日均价", "category": "price_level",
         "description": "5d rolling mean of adj_close", "tags": ["ma"]},
        {"id": "ma_20", "name": "20日均价", "category": "price_level",
         "description": "20d rolling mean of adj_close", "tags": ["ma"]},
        {"id": "ma_60", "name": "60日均价", "category": "price_level",
         "description": "60d rolling mean of adj_close", "tags": ["ma"]},
        {"id": "price_to_ma_5", "name": "价格/5日均线", "category": "price_level",
         "description": "adj_close / ma_5", "tags": ["ma", "deviation"]},
        {"id": "price_to_ma_20", "name": "价格/20日均线", "category": "price_level",
         "description": "adj_close / ma_20", "tags": ["ma", "deviation"]},
        {"id": "price_to_ma_60", "name": "价格/60日均线", "category": "price_level",
         "description": "adj_close / ma_60", "tags": ["ma", "deviation"]},
        # Oscillators
        {"id": "rsi_14", "name": "14日RSI", "category": "oscillator",
         "description": "Wilder 14d RSI on close", "tags": ["rsi", "oscillator"]},
        {"id": "macd", "name": "MACD", "category": "oscillator",
         "description": "EMA(12) - EMA(26) of close", "tags": ["macd", "oscillator"]},
        {"id": "macd_signal", "name": "MACD信号线", "category": "oscillator",
         "description": "9d EMA of MACD", "tags": ["macd"]},
        {"id": "macd_diff", "name": "MACD柱", "category": "oscillator",
         "description": "MACD - MACD signal", "tags": ["macd"]},
        {"id": "stoch_k", "name": "随机指标K", "category": "oscillator",
         "description": "Stochastic %K (14,3)", "tags": ["stochastic"]},
        {"id": "stoch_d", "name": "随机指标D", "category": "oscillator",
         "description": "Stochastic %D (14,3)", "tags": ["stochastic"]},
        # Bollinger
        {"id": "bb_position_20", "name": "布林带位置", "category": "oscillator",
         "description": "(close - bb_lower) / (bb_upper - bb_lower)", "tags": ["bollinger"]},
        # Cross-sectional ranks
        {"id": "return_1d_rank", "name": "1日收益排名", "category": "cs_rank",
         "description": "Cross-sectional rank of return_1d", "tags": ["rank"]},
        {"id": "volume_rank", "name": "成交量排名", "category": "cs_rank",
         "description": "Cross-sectional rank of volume", "tags": ["rank", "volume"]},
        {"id": "volatility_20_rank", "name": "波动率排名", "category": "cs_rank",
         "description": "Cross-sectional rank of volatility_20", "tags": ["rank", "volatility"]},
        # A-share specific
        {"id": "turnover_ma_5", "name": "5日换手率均线", "category": "turnover",
         "description": "5d MA of turnover_rate", "tags": ["turnover"]},
        {"id": "turnover_ma_20", "name": "20日换手率均线", "category": "turnover",
         "description": "20d MA of turnover_rate", "tags": ["turnover"]},
        {"id": "turnover_ratio", "name": "换手率发散", "category": "turnover",
         "description": "turnover_ma_5 / turnover_ma_20", "tags": ["turnover"]},
        {"id": "limit_flag", "name": "涨跌停标志", "category": "ashare",
         "description": "是否涨跌停", "tags": ["limit", "ashare"]},
        {"id": "near_limit", "name": "近涨跌停", "category": "ashare",
         "description": "是否接近涨跌停", "tags": ["limit", "ashare"]},
        # Misc
        {"id": "hl_pct", "name": "日内振幅", "category": "price_level",
         "description": "(high - low) / close", "tags": ["volatility", "intraday"]},
        {"id": "hl_ma_5", "name": "5日平均振幅", "category": "price_level",
         "description": "5d MA of hl_pct", "tags": ["volatility"]},
    ],
    "orthogonal": [
        {"id": "amihud_20", "name": "Amihud非流动性", "category": "liquidity",
         "description": "mean(|return| / turnover_value) over 20d", "tags": ["illiquidity", "anomaly"],
         "formula": "ts_mean(abs(return_1d) / safe_div(volume * close, 1), 20)"},
        {"id": "max_ret_20", "name": "20日最大收益(MAX)", "category": "reversal",
         "description": "max daily return over 20d — lottery factor", "tags": ["lottery", "anomaly"]},
        {"id": "idio_vol_20", "name": "20日特质波动", "category": "volatility",
         "description": "std of (ret - cross-sectional mean ret)", "tags": ["idiosyncratic", "anomaly"]},
        {"id": "resid_mom_20", "name": "20日残差动量", "category": "momentum",
         "description": "20d sum of market-excess daily ret", "tags": ["momentum", "anomaly"]},
        {"id": "ret_skew_20", "name": "20日收益偏度", "category": "reversal",
         "description": "rolling 20d return skewness", "tags": ["skew", "crash_risk", "anomaly"]},
        {"id": "high_52w_prox", "name": "52周高点接近度", "category": "price_level",
         "description": "close / trailing 252d high", "tags": ["anchoring", "anomaly"]},
    ],
    "alpha158": [
        # K-line shape (windowless)
        {"id": "a158_kmid", "name": "K线中点", "category": "alpha158",
         "description": "(close - open) / (high - low + 1e-12)", "tags": ["kline", "alpha158"]},
        {"id": "a158_klen", "name": "K线长度", "category": "alpha158",
         "description": "(high - low) / open", "tags": ["kline", "alpha158"]},
        {"id": "a158_kmid2", "name": "K线中点2", "category": "alpha158",
         "description": "(close - open) / (high - low)", "tags": ["kline", "alpha158"]},
        {"id": "a158_kup", "name": "上影线", "category": "alpha158",
         "description": "(high - max(open, close)) / open", "tags": ["kline", "alpha158"]},
        {"id": "a158_klow", "name": "下影线", "category": "alpha158",
         "description": "(min(open, close) - low) / open", "tags": ["kline", "alpha158"]},
        {"id": "a158_ksft", "name": "K线位移", "category": "alpha158",
         "description": "2*close - high - low", "tags": ["kline", "alpha158"]},
        {"id": "a158_ksft2", "name": "K线位移2", "category": "alpha158",
         "description": "close - open", "tags": ["kline", "alpha158"]},
        # Per-window patterns — documented with W=20 as representative
        {"id": "a158_rsqr20", "name": "趋势拟合度R²(20) ", "category": "alpha158",
         "description": "R-squared of close on time trend", "tags": ["trend", "alpha158"]},
        {"id": "a158_resi20", "name": "趋势残差(20)", "category": "alpha158",
         "description": "residual of close vs time trend / close", "tags": ["trend", "alpha158"]},
        {"id": "a158_qtlu20", "name": "上分位数(20)", "category": "alpha158",
         "description": "80% rolling quantile / close", "tags": ["quantile", "alpha158"]},
        {"id": "a158_qtld20", "name": "下分位数(20)", "category": "alpha158",
         "description": "20% rolling quantile / close", "tags": ["quantile", "alpha158"]},
        {"id": "a158_rsv20", "name": "随机位置(20)", "category": "alpha158",
         "description": "(close - min_low) / (max_high - min_low)", "tags": ["stochastic", "alpha158"]},
        {"id": "a158_imax20", "name": "最高价位置(20)", "category": "alpha158",
         "description": "bar index of window high / W", "tags": ["pattern", "alpha158"]},
        {"id": "a158_imin20", "name": "最低价位置(20)", "category": "alpha158",
         "description": "bar index of window low / W", "tags": ["pattern", "alpha158"]},
        {"id": "a158_imxd20", "name": "高低位置差(20)", "category": "alpha158",
         "description": "(imax - imin) / W", "tags": ["pattern", "alpha158"]},
        {"id": "a158_corr20", "name": "量价相关(20)", "category": "alpha158",
         "description": "corr(close, log(volume+1)) over 20d", "tags": ["correlation", "alpha158"]},
        {"id": "a158_cord20", "name": "量价变化相关(20)", "category": "alpha158",
         "description": "corr(ret, log-volume-change) over 20d", "tags": ["correlation", "alpha158"]},
        {"id": "a158_cntp20", "name": "上涨天数占比(20)", "category": "alpha158",
         "description": "fraction of up days", "tags": ["persistence", "alpha158"]},
        {"id": "a158_cntn20", "name": "下跌天数占比(20)", "category": "alpha158",
         "description": "fraction of down days", "tags": ["persistence", "alpha158"]},
        {"id": "a158_cntd20", "name": "涨跌天数差(20)", "category": "alpha158",
         "description": "(up - down) / W", "tags": ["persistence", "alpha158"]},
        {"id": "a158_sump20", "name": "上涨幅度占比(20)", "category": "alpha158",
         "description": "sum of positive ret / sum of abs ret", "tags": ["asymmetry", "alpha158"]},
        {"id": "a158_sumd20", "name": "涨跌幅差(20)", "category": "alpha158",
         "description": "(sum_pos - sum_neg) / sum_abs", "tags": ["asymmetry", "alpha158"]},
        {"id": "a158_wvma20", "name": "量加权波动(20)", "category": "alpha158",
         "description": "std / mean of |ret|*volume", "tags": ["volatility", "volume", "alpha158"]},
    ],
    "gtja191": [
        {"id": "gtja_alpha001", "name": "GTJA Alpha001 量价反转相关", "category": "gtja191",
         "description": "-corr(rank(log-vol change), rank(intraday ret), 6)",
         "formula": "-ts_corr(cs_rank(ts_delta(log(volume), 1)), cs_rank((close - open) / open), 6)",
         "tags": ["volume", "reversal", "correlation"]},
        {"id": "gtja_alpha008", "name": "GTJA Alpha008 量价背离", "category": "gtja191",
         "description": "-rank(delta(ret, 2) * ts_max(volume-turnover-MA-diff, 5))",
         "tags": ["volume", "reversal"]},
        {"id": "gtja_alpha009", "name": "GTJA Alpha009 日内强度动量", "category": "gtja191",
         "description": "corr(high-close, VWAP-close, 4) * rank(h/l, 10)",
         "tags": ["momentum", "volume"]},
        {"id": "gtja_alpha012", "name": "GTJA Alpha012 成交量反转", "category": "gtja191",
         "description": "sign(delta(volume, 1)) * (-delta(close, 1))",
         "tags": ["volume", "reversal"]},
        {"id": "gtja_alpha017", "name": "GTJA Alpha017 振幅反转", "category": "gtja191",
         "description": "-rank(cs_rank(hl-range) * rank(corr(adj_close, ts_delay(adj_close, 1), 5)))",
         "tags": ["reversal", "range"]},
        {"id": "gtja_alpha020", "name": "GTJA Alpha020 量价背离1", "category": "gtja191",
         "description": "-rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(volume change)",
         "tags": ["volume", "reversal"]},
        {"id": "gtja_alpha024", "name": "GTJA Alpha024 趋势加速", "category": "gtja191",
         "description": "rank(corr(close change, ts_delay(volume, 5), 5))",
         "tags": ["trend", "volume"]},
        {"id": "gtja_alpha028", "name": "GTJA Alpha028 量价背离2", "category": "gtja191",
         "description": "scale(corr(ts_rank(volume, 20), ts_rank(high, 20), 5))",
         "tags": ["correlation", "volume"]},
        {"id": "gtja_alpha032", "name": "GTJA Alpha032 趋势强度", "category": "gtja191",
         "description": "scale(corr(open, VWAP, 5)) * rank(log(volume) - delay(log(volume), 2))",
         "tags": ["correlation", "volume"]},
        {"id": "gtja_alpha033", "name": "GTJA Alpha033 均值回复", "category": "gtja191",
         "description": "-rank(signed_power(ret-1, 3))",
         "tags": ["reversal", "power"]},
        {"id": "gtja_alpha038", "name": "GTJA Alpha038 波动率反转", "category": "gtja191",
         "description": "mean(rank(corr(high, volume, 5)), 20) * rank(deviation from MA)",
         "tags": ["volatility", "reversal"]},
        {"id": "gtja_alpha041", "name": "GTJA Alpha041 价格极值", "category": "gtja191",
         "description": "rank(signed_power(close - VWAP, 3))",
         "tags": ["price", "power"]},
        {"id": "gtja_alpha049", "name": "GTJA Alpha049 日内反转", "category": "gtja191",
         "description": "rank(delta(daily-change, 1) * cs_rank(corr(close, volume, 6)))",
         "tags": ["intraday", "reversal"]},
        {"id": "gtja_alpha051", "name": "GTJA Alpha051 隔夜反转", "category": "gtja191",
         "description": "rank(daily-change * cs_rank(corr(close, volume, 2)))",
         "tags": ["reversal", "correlation"]},
        {"id": "gtja_alpha053", "name": "GTJA Alpha053 日内强度", "category": "gtja191",
         "description": "-rank(cs_rank((close - open) / open) + (1 - close/low))",
         "tags": ["intraday", "strength"]},
        {"id": "gtja_alpha057", "name": "GTJA Alpha057 量价背离3", "category": "gtja191",
         "description": "rank(ts_corr(close, ts_delay(volume, 5), 5)) * rank(volume turnover divergence)",
         "tags": ["volume", "correlation"]},
        {"id": "gtja_alpha064", "name": "GTJA Alpha064 波动率反转2", "category": "gtja191",
         "description": "-rank(ts_max(corr(cs_rank(volume), cs_rank(close), 5), 3))",
         "tags": ["volatility", "reversal"]},
        {"id": "gtja_alpha065", "name": "GTJA Alpha065 均值回复2", "category": "gtja191",
         "description": "mean(ts_rank(close, 20), 60) - rank(close)",
         "tags": ["reversal", "mean"]},
        {"id": "gtja_alpha071", "name": "GTJA Alpha071 量价趋势背离", "category": "gtja191",
         "description": "cs_rank(ts_mean(ts_corr(high, volume, 5), 20)) * rank(price trend)",
         "tags": ["trend", "volume"]},
        {"id": "gtja_alpha075", "name": "GTJA Alpha075 极端价量关系", "category": "gtja191",
         "description": "rank(corr(close, ts_delay(close, 10), 10)) * rank(signed_power(ret, 2))",
         "tags": ["correlation", "power"]},
        {"id": "gtja_alpha085", "name": "GTJA Alpha085 流动性趋势", "category": "gtja191",
         "description": "rank(cs_rank(log(volume)) * rank(volume zone)) * rank(close return trend)",
         "tags": ["liquidity", "trend"]},
        {"id": "gtja_alpha092", "name": "GTJA Alpha092 价格突破", "category": "gtja191",
         "description": "rank(cs_rank(close return) * rank(HL range return) * rank(log profit range))",
         "tags": ["breakout", "range"]},
        {"id": "gtja_alpha095", "name": "GTJA Alpha095 信息增强动量", "category": "gtja191",
         "description": "rank(hl range change adjusted by volume decile)",
         "tags": ["momentum", "volume"]},
        {"id": "gtja_alpha098", "name": "GTJA Alpha098 量价背离4", "category": "gtja191",
         "description": "rank(cs_rank(ts_corr(close, ts_delay(close, 10), 10)) * rank(volume ratio deviation))",
         "tags": ["correlation", "volume"]},
    ],
    "formula": [],  # populated at runtime from factor_library/
    "fundamental": [
        {"id": "ep", "name": "盈利市值比", "category": "valuation",
         "description": "1/PE = earnings / price", "tags": ["value"]},
        {"id": "bp", "name": "账面市值比", "category": "valuation",
         "description": "1/PB = book / price", "tags": ["value"]},
        {"id": "sp", "name": "营收市值比", "category": "valuation",
         "description": "1/PS = sales / price", "tags": ["value"]},
        {"id": "ln_mv", "name": "对数市值", "category": "size",
         "description": "ln(market_cap)", "tags": ["size"]},
    ],
    "financial": [
        {"id": "roe", "name": "ROE", "category": "financial",
         "description": "Return on equity (quarterly)", "tags": ["profitability"]},
        {"id": "gross_margin", "name": "毛利率", "category": "financial",
         "description": "Gross margin (quarterly)", "tags": ["profitability"]},
        {"id": "revenue_yoy", "name": "营收同比", "category": "financial",
         "description": "Revenue YoY growth", "tags": ["growth"]},
        {"id": "profit_yoy", "name": "利润同比", "category": "financial",
         "description": "Net profit YoY growth", "tags": ["growth"]},
    ],
}


def _make_native_infos() -> list[FactorInfo]:
    """Convert the hardcoded native metadata dict into a list of FactorInfo."""
    result: list[FactorInfo] = []
    for family, entries in NATIVE_FACTOR_METADATA.items():
        # Skip "formula" — it's populated dynamically
        if family == "formula":
            continue
        for entry in entries:
            result.append(FactorInfo(
                id=entry["id"],
                name=entry.get("name", entry["id"]),
                category=entry.get("category", ""),
                source=f"native:{family}",
                description=entry.get("description", ""),
                formula=entry.get("formula", ""),
                tags=entry.get("tags", []),
                produces=entry["id"],
            ))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_all_factors(
    library_dir: str | None = None,
    category: str | None = None,
    enabled_only: bool = False,
) -> list[FactorInfo]:
    """List all factors (native + formula).

    Parameters
    ----------
    library_dir: path to factor_library/
    category: filter by category (e.g. "momentum", "volatility", "gtja191")
    enabled_only: if True, only return enabled factors
    """
    all_factors = _make_native_infos()

    # Add formula factors
    try:
        formula_defs = load_factor_definitions(library_dir)
        for fd in formula_defs:
            all_factors.append(FactorInfo(
                id=fd.id,
                name=fd.name,
                category=fd.category,
                source="formula",
                description=fd.description,
                formula=fd.formula,
                tags=fd.tags,
                produces=fd.id,
                enabled=fd.enabled,
            ))
    except Exception:
        logger.warning("加载公式因子失败", exc_info=True)

    # Filter
    if category:
        all_factors = [f for f in all_factors if f.category == category]
    if enabled_only:
        all_factors = [f for f in all_factors if f.enabled]

    return sorted(all_factors, key=lambda f: f.id)


def search_factors(
    query: str,
    library_dir: str | None = None,
) -> list[FactorInfo]:
    """Search factors by id, name, description, or tags (case-insensitive)."""
    q = query.lower()
    results: list[FactorInfo] = []
    for f in list_all_factors(library_dir):
        text = f"{f.id} {f.name} {f.description} {' '.join(f.tags)}".lower()
        if q in text:
            results.append(f)
    return results


def get_factor_info(
    factor_id: str,
    library_dir: str | None = None,
) -> FactorInfo | None:
    """Get detailed info about a single factor by id."""
    for f in list_all_factors(library_dir):
        if f.id == factor_id:
            return f
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_table(infos: list[FactorInfo], columns: list[str], limit: int = 80) -> None:
    """Print a list of FactorInfo as a formatted table."""
    if not infos:
        print("(empty)")
        return

    # Determine column widths
    widths = {c: len(c) for c in columns}
    for fi in infos:
        for c in columns:
            val = str(getattr(fi, c, ""))
            widths[c] = min(max(widths[c], len(val)), 60)

    # Header
    header = "  ".join(f"{c:{widths[c]}}" for c in columns)
    print(header)
    print("-" * len(header))

    for fi in infos:
        row = "  ".join(
            str(getattr(fi, c, ""))[:widths[c]].ljust(widths[c])
            for c in columns
        )
        print(row)


def _cli_list(args: list[str]) -> None:
    """Handle 'list' subcommand."""
    category = None
    if "--category" in args:
        idx = args.index("--category")
        if idx + 1 < len(args):
            category = args[idx + 1]

    infos = list_all_factors(category=category)

    # Group by category for display
    if not category:
        cats: dict[str, list[FactorInfo]] = {}
        for fi in infos:
            cats.setdefault(fi.category or "uncategorized", []).append(fi)
        for cat in sorted(cats):
            print(f"\n--- {cat} ({len(cats[cat])} factors) ---")
            _print_table(cats[cat], ["id", "source", "description"])
    else:
        _print_table(infos, ["id", "source", "description"])

    print(f"\n共 {len(infos)} 个因子")


def _cli_search(args: list[str]) -> None:
    """Handle 'search' subcommand."""
    query = args[0] if args else ""
    if not query:
        print("Usage: registry search <query>")
        return
    results = search_factors(query)
    print(f"搜索 '{query}': {len(results)} 个结果\n")
    _print_table(results, ["id", "category", "source", "description"])


def _cli_info(args: list[str]) -> None:
    """Handle 'info' subcommand."""
    factor_id = args[0] if args else ""
    if not factor_id:
        print("Usage: registry info <factor_id>")
        return
    fi = get_factor_info(factor_id)
    if fi is None:
        print(f"因子 '{factor_id}' 未找到")
        return
    print(f"ID:          {fi.id}")
    print(f"Name:        {fi.name}")
    print(f"Category:    {fi.category}")
    print(f"Source:      {fi.source}")
    print(f"Description: {fi.description}")
    print(f"Tags:        {', '.join(fi.tags)}")
    if fi.formula:
        print(f"Formula:     {fi.formula}")
    print(f"Produces:    {fi.produces}")
    print(f"Enabled:     {fi.enabled}")


def _cli_validate(args: list[str]) -> None:
    """Handle 'validate' subcommand."""
    try:
        from .schema import load_factor_definitions
    except ImportError:
        definitions = load_factor_definitions()
    else:
        definitions = load_factor_definitions()

    if not definitions:
        print("没有找到公式因子定义 (factor_library/ 为空或不存在)")
        return

    from .parser import extract_dependencies as _extract_deps

    print(f"加载了 {len(definitions)} 个因子定义\n")
    ok = 0
    failed = 0
    for fd in definitions:
        try:
            validate_formula(fd.formula)
            deps = _extract_deps(fd.formula)
            print(f"  ✓ {fd.id}  (依赖: {sorted(deps)})")
            ok += 1
        except Exception as exc:
            print(f"  ✗ {fd.id}: {exc}")
            failed += 1

    print(f"\n通过: {ok}  失败: {failed}")


def main() -> None:
    """CLI entry point: python -m src.data.factors.registry <command>"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="factor-registry",
        description="统一因子注册表 — 浏览、搜索、校验因子定义",
    )
    parser.add_argument("--library", default=None,
                        help="因子库目录路径 (默认从 config / 环境变量读取)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="列出所有因子").add_argument(
        "--category", default=None, help="按分类过滤")
    sp_search = sub.add_parser("search", help="搜索因子")
    sp_search.add_argument("query", nargs="?", default="", help="搜索关键词")
    sp_info = sub.add_parser("info", help="查看单个因子详情")
    sp_info.add_argument("factor_id", nargs="?", default="", help="因子ID")
    sub.add_parser("validate", help="校验所有公式因子")

    # If no args were passed, try to use sys.argv
    parsed = parser.parse_args()

    if parsed.command == "list":
        _cli_list(sys.argv[2:])
    elif parsed.command == "search":
        _cli_search([parsed.query] if hasattr(parsed, "query") and parsed.query else sys.argv[2:])
    elif parsed.command == "info":
        _cli_info([parsed.factor_id] if hasattr(parsed, "factor_id") and parsed.factor_id else sys.argv[2:])
    elif parsed.command == "validate":
        _cli_validate([])
    else:
        parser.print_help()

    # Need import for validate
    from .parser import extract_dependencies


# Allow running as: python -m src.data.factors.registry
if __name__ == "__main__":
    main()
