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
    def get_universe(
        self, index_code: str, start_date: str | None = None, end_date: str | None = None
    ) -> list[str]:
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

    def trade_dates(self, start_date: str, end_date: str) -> list[str] | None:
        """[start_date, end_date]（YYYY-MM-DD）内的交易日列表（YYYYMMDD），未知返回 None。

        尽力而为的廉价查询（缓存优先），供面板缓存做「尾部是否真有新交易日」
        预判；实现侧不应为了回答它发起昂贵的网络抓取。
        """
        return None

    def bar_available(self, trade_date: str) -> bool | None:
        """指定交易日（YYYYMMDD）的日线行情是否已可得（本地缓存或一次廉价探测）。

        True/False 为确定答案，None 未知（探测失败等）。供面板缓存区分
        「日历上的交易日但数据尚未入库」与「真有新数据」。
        """
        return None


# --------------------------------------------------------------------------
# 可选能力契约（显式化原先的 hasattr 鸭子类型检查）
# --------------------------------------------------------------------------
# 连接器的可选能力用 runtime_checkable Protocol 声明；src/data/__init__.py 的
# 调用点用 isinstance 检查（结构性判定 = 方法存在即可，连接器无需显式继承）。
# 新增可选能力时在此加 Protocol，不要再用 hasattr 隐式探测。


@runtime_checkable
class SupportsFundamentals(Protocol):
    """可提供估值因子（PE/PB/PS/市值）快照的连接器。"""

    def fetch_fundamentals(
        self,
        start_date: str,
        end_date: str,
        freq_days: int = 21,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame: ...


@runtime_checkable
class SupportsFinancialFactors(Protocol):
    """可提供财务因子（营收/净利 YoY 等）的连接器。"""

    def fetch_financial_factors(self, symbols: list[str]) -> pd.DataFrame: ...


@runtime_checkable
class SupportsIndustryMap(Protocol):
    """可提供 symbol → industry 映射的连接器。"""

    def get_industry_map(self) -> dict[str, str]: ...


@runtime_checkable
class SupportsNorthbound(Protocol):
    """可提供北向持股（陆股通个股持股）日频快照的连接器。"""

    def fetch_hk_hold(self, start_date: str, end_date: str) -> pd.DataFrame:
        """返回长表：date, symbol, north_vol（持股量）, north_ratio（持股比例 %）。"""
        ...


@runtime_checkable
class SupportsWholeMarketFetch(Protocol):
    """可按交易日整市场拉取日线（1 次 API 调用/交易日）的连接器。"""

    def fetch_ohlcv_whole_market(
        self, start_date: str, end_date: str, symbols: list[str] | None = None
    ) -> pd.DataFrame: ...
