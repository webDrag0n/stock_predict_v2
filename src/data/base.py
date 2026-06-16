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
