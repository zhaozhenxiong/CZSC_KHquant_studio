"""Online data downloaders. This module never writes cache.

The legacy ``DataDownloader`` API is preserved for backwards compatibility.  New
code should prefer ``UnifiedDataProvider`` from ``market_data_provider``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from .market_data_provider import (
    AkShareProvider,
    BaostockProvider,
    LocalCSVProvider,
    UnifiedDataProvider,
    get_unified_provider,
)
from .validator import DataValidator


@dataclass
class DownloadResult:
    ok: bool
    data: pd.DataFrame | None = None
    source: str = ""
    error: str = ""


class DataDownloader:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._provider: UnifiedDataProvider | None = None

    def _provider_instance(self) -> UnifiedDataProvider:
        if self._provider is None:
            self._provider = get_unified_provider(self.config)
        return self._provider

    def download_stock_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        sources: Iterable[str] | None = None,
    ) -> DownloadResult:
        """Download daily OHLCV while preserving the old DownloadResult shape."""
        result = self._provider_instance().fetch_daily(
            code=code,
            start_date=start_date,
            end_date=end_date,
            sources=list(sources) if sources is not None else None,
        )
        if result.ok and result.data is not None and not result.data.empty:
            clean = DataValidator.clean_stock_daily(result.data, code)
            valid = DataValidator.validate_stock_daily(clean, min_rows=1, require_min_rows=False)
            if not valid.ok:
                return DownloadResult(False, None, result.source, valid.error)
            return DownloadResult(True, clean, result.source, "")
        return DownloadResult(False, None, result.source or "", result.error)

    def download_stocks_daily(
        self,
        stocks: Iterable[str],
        start_date: str,
        end_date: str,
        sources: Iterable[str] | None = None,
    ) -> DownloadResult:
        """Batch download daily OHLCV for many codes in one upstream call.

        The returned DataFrame contains rows for all successfully fetched codes
        with a ``code`` column; per-code validation and cache merging are the
        caller's responsibility.
        """
        result = self._provider_instance().fetch_daily_batch(
            codes=list(stocks),
            start_date=start_date,
            end_date=end_date,
            sources=list(sources) if sources is not None else None,
        )
        if result.ok and result.data is not None and not result.data.empty:
            clean = DataValidator.clean_stock_daily(result.data)
            valid = DataValidator.validate_stock_daily(clean, min_rows=1, require_min_rows=False)
            if not valid.ok:
                return DownloadResult(False, None, result.source, valid.error)
            return DownloadResult(True, clean, result.source, "")
        return DownloadResult(False, None, result.source or "", result.error)

    def download_stock_basic(self, min_count: int = 3000) -> DownloadResult:
        result = self._provider_instance().fetch_basic(min_count=min_count)
        if result.ok and result.data is not None:
            out = result.data.copy()
            if "total_market_cap" in out.columns:
                out["total_market_cap"] = pd.to_numeric(out["total_market_cap"], errors="coerce")
            if "turnover" in out.columns:
                out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce")
            if len(out) < min_count:
                return DownloadResult(False, None, result.source, f"too few all-A rows: {len(out)} < {min_count}")
            return DownloadResult(True, out.reset_index(drop=True), result.source, "")
        return DownloadResult(False, None, result.source or "", result.error)

    def download_index_daily(self, index_code: str, start_date: str, end_date: str) -> DownloadResult:
        try:
            import akshare as ak

            symbol = index_code.split(".")[0]
            df = ak.index_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"})
            clean = DataValidator.clean_stock_daily(df, index_code)
            return DownloadResult(not clean.empty, clean, "akshare", "" if not clean.empty else "empty data")
        except Exception as exc:
            return DownloadResult(False, None, "akshare", str(exc))

    def _download_stock_daily_akshare(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        provider = AkShareProvider()
        result = provider.fetch_daily(code, start_date, end_date)
        if not result.ok or result.data is None:
            raise RuntimeError(result.error or "akshare daily fetch failed")
        return result.data

    def _download_stock_daily_baostock(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        provider = BaostockProvider()
        result = provider.fetch_daily(code, start_date, end_date)
        if not result.ok or result.data is None:
            raise RuntimeError(result.error or "baostock daily fetch failed")
        return result.data

    def _download_stock_daily_local_csv(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        provider = LocalCSVProvider(self.config)
        result = provider.fetch_daily(code, start_date, end_date)
        if not result.ok or result.data is None:
            return pd.DataFrame()
        return result.data
