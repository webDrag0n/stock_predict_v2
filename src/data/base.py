"""Data connector abstraction.

A connector knows how to talk to one market-data provider. Adding a new source
(akshare, baostock, a local CSV dump, ...) means implementing this interface and
registering it in `src/data/__init__.py` — nothing else in the pipeline changes.

A connector deals only in raw OHLCV(+factor) bars. Feature engineering, target
construction, and the Jane Street format transform live in the package's
`__init__.py` and are provider-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import pandas as pd


class DataConnector(ABC):
    """Minimal interface every market-data source must implement."""

    #: short identifier, e.g. "tushare"
    name: str = "base"

    @abstractmethod
    def get_universe(self, index_code: str, start_date: str | None = None,
                     end_date: str | None = None) -> list[str]:
        """Return the symbol codes making up an index/universe.

        If start_date/end_date (YYYY-MM-DD) are provided, implementations should
        return the historical *union* of constituents over that window to avoid
        survivorship bias; otherwise the latest snapshot.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch daily bars for `symbols` between start/end (YYYY-MM-DD).

        Must return a long-format DataFrame with at least these columns:
            date (datetime64), symbol (str), open, high, low, close, volume
        Extra provider-specific columns (e.g. turnover_rate, pct_chg) are kept
        and become candidate features downstream.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# 可选能力契约（显式化原先的 hasattr 鸭子类型检查）
# --------------------------------------------------------------------------
# 连接器的可选能力用 runtime_checkable Protocol 声明；src/data/__init__.py 的
# 调用点用 isinstance 检查（结构性判定 = 方法存在即可，连接器无需显式继承）。
# 新增可选能力时在此加 Protocol，不要再用 hasattr 隐式探测。

@runtime_checkable
class SupportsFundamentals(Protocol):
    """可提供估值因子（PE/PB/PS/市值）快照的连接器。"""

    def fetch_fundamentals(self, start_date: str, end_date: str,
                           freq_days: int = 21) -> pd.DataFrame: ...


@runtime_checkable
class SupportsFinancialFactors(Protocol):
    """可提供财务因子（营收/净利 YoY 等）的连接器。"""

    def fetch_financial_factors(self, symbols: list[str]) -> pd.DataFrame: ...


@runtime_checkable
class SupportsIndustryMap(Protocol):
    """可提供 symbol → industry 映射的连接器。"""

    def get_industry_map(self) -> dict[str, str]: ...


@runtime_checkable
class SupportsWholeMarketFetch(Protocol):
    """可按交易日整市场拉取日线（1 次 API 调用/交易日）的连接器。"""

    def fetch_ohlcv_whole_market(self, start_date: str, end_date: str,
                                 symbols: list[str] | None = None) -> pd.DataFrame: ...
