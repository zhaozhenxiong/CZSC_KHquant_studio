"""Public data-center API used by backtests and notebooks."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from my_strategy.core.tz import local_now

from .config import load_data_config
from .local_cache import LocalCache
from .stock_pool import normalize_stock_code
from .updater import DailyUpdater


class DataCenter:
    def __init__(self, config: dict | None = None):
        self.config = config or load_data_config()
        self.cache = LocalCache(self.config)
        self.updater = DailyUpdater(self.config)

    def get_stock_daily(
        self,
        code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        auto_update: bool = True,
        allow_old_cache: bool | None = None,
    ) -> pd.DataFrame:
        code = normalize_stock_code(code)
        start_date = start_date or self.config.get("update", {}).get("start_date", "2015-01-01")
        end_date = end_date or local_now().strftime("%Y-%m-%d")
        cache_cfg = self.config.get("cache", {})
        allow_old_cache = cache_cfg.get("allow_use_old_cache", True) if allow_old_cache is None else allow_old_cache

        cached = self.cache.slice_stock_daily(code, start_date, end_date)
        if self.cache.has_complete_range(code, start_date, end_date) and _has_trade_prices(cached):
            _annotate_source(cached, self.cache)
            return cached

        if auto_update:
            self.updater.update_one(code, start_date=start_date, end_date=end_date)
            updated = self.cache.slice_stock_daily(code, start_date, end_date)
            if self.cache.has_complete_range(code, start_date, end_date) and _has_trade_prices(updated):
                _annotate_source(updated, self.cache)
                return updated

        old = self.cache.slice_stock_daily(code, start_date, end_date)
        if allow_old_cache and not old.empty:
            _annotate_source(old, self.cache)
            return old
        raise RuntimeError(f"No usable local data for {code} ({start_date} ~ {end_date})")


def get_stock_daily(code: str, start_date: str, end_date: str, auto_update: bool = True) -> pd.DataFrame:
    return DataCenter().get_stock_daily(code, start_date, end_date, auto_update=auto_update)


def _has_trade_prices(df: pd.DataFrame) -> bool:
    return "trade_close" in df.columns and pd.to_numeric(df["trade_close"], errors="coerce").notna().any()


def _annotate_source(df: pd.DataFrame, cache: LocalCache) -> None:
    df.attrs.setdefault("khquant_storage_mode", cache.storage.settings.mode)
    if cache.storage.settings.read_db_first:
        df.attrs.setdefault("khquant_data_source", "raw_sqlite")
        df.attrs.setdefault("khquant_raw_db", str(cache.storage.raw.db_path))
