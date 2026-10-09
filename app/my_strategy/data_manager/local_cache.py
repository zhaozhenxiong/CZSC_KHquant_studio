"""Local parquet/cache read-write layer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import threading

import pandas as pd

from .config import load_data_config, project_path
from .validator import DataValidator
from my_strategy.storage.access_layer import get_data_access
from my_strategy.storage.db_paths import read_db_first, write_db, write_files


class LocalCache:
    """Own all local cache IO. Network code must not write cache directly."""

    STATUS_COLUMNS = ["code", "last_date", "last_update_time", "rows", "source", "status", "error_msg"]
    LOG_COLUMNS = ["time", "code", "action", "start_date", "end_date", "rows", "source", "status", "error_msg"]

    def __init__(self, config: dict | None = None):
        self.config = config or load_data_config()
        self.data_root = project_path(self.config.get("data_root", "my_strategy/data"))
        self.stock_daily_dir = self.data_root / "raw" / "stock_daily"
        self.index_daily_dir = self.data_root / "raw" / "index_daily"
        self.stock_basic_dir = self.data_root / "raw" / "stock_basic"
        self.custom_pool_dir = self.data_root / "raw" / "custom_pool"
        self.metadata_dir = self.data_root / "metadata"
        self.storage = get_data_access()
        self.raw_repository = self.storage.raw
        self._lock = threading.Lock()
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        for path in [
            self.stock_daily_dir,
            self.index_daily_dir,
            self.stock_basic_dir,
            self.custom_pool_dir,
            self.data_root / "processed" / "stock_daily_adj",
            self.data_root / "processed" / "factors",
            self.data_root / "processed" / "signals",
            self.data_root / "qlib_data",
            self.metadata_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def stock_daily_path(self, code: str) -> Path:
        return self.stock_daily_dir / f"{code}.parquet"

    def stock_basic_path(self) -> Path:
        return self.stock_basic_dir / "all_a_stock_list.parquet"

    def cache_exists(self, code: str) -> bool:
        if read_db_first() and self.storage.has_stock(code):
            return True
        return write_files() and self.stock_daily_path(code).exists()

    def read_stock_daily(self, code: str) -> pd.DataFrame:
        if read_db_first():
            df = self.storage.read_stock_daily(code)
            if not df.empty:
                return df.sort_values("date").reset_index(drop=True)
            if not write_files():
                return pd.DataFrame()
        path = self.stock_daily_path(code)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df.attrs["khquant_data_source"] = "parquet"
        df.attrs["khquant_storage_mode"] = self.storage.settings.mode
        return df.sort_values("date").reset_index(drop=True)

    def save_stock_daily(self, code: str, df: pd.DataFrame, source: str = "unknown", *, incremental: bool = False) -> None:
        if df is None or df.empty:
            raise ValueError("refuse to save empty stock daily cache")
        out = DataValidator.clean_stock_daily(df, code)
        out["source"] = source
        out["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = DataValidator.validate_stock_daily(
            out,
            min_rows=1 if incremental else int(self.config.get("cache", {}).get("min_rows_for_valid_stock", 100)),
            require_min_rows=not incremental,
        )
        if not result.ok:
            raise ValueError(f"invalid stock daily cache for {code}: {result.error}")
        with self._lock:
            if write_db():
                self.storage.save_stock_daily(code, out, source=source)
            if write_files():
                if incremental and self.stock_daily_path(code).exists():
                    existing = pd.read_parquet(self.stock_daily_path(code))
                    out = DataValidator.clean_stock_daily(pd.concat([existing, out], ignore_index=True), code)
                self._atomic_write_parquet(out, self.stock_daily_path(code))

    def save_stocks_daily(self, code_to_df: dict[str, pd.DataFrame], source: str | dict[str, str] = "unknown", *, incremental: bool = False) -> None:
        """Bulk-save daily bars for many codes in one DB transaction.

        File-mode writes remain per-code because parquet paths are separate, but
        the SQLite path no longer pays per-stock transaction overhead. The
        ``source`` argument may be a single string or a per-code mapping.
        """
        code_to_df = {code: df for code, df in code_to_df.items() if df is not None and not df.empty}
        if not code_to_df:
            return
        min_rows_for_valid = int(self.config.get("cache", {}).get("min_rows_for_valid_stock", 100))
        validated: dict[str, pd.DataFrame] = {}
        source_for_code: dict[str, str] = {}
        for code, df in code_to_df.items():
            src = source.get(code, "unknown") if isinstance(source, dict) else source
            out = DataValidator.clean_stock_daily(df, code)
            out["source"] = src
            out["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = DataValidator.validate_stock_daily(
                out,
                min_rows=1 if incremental else min_rows_for_valid,
                require_min_rows=not incremental,
            )
            if not result.ok:
                raise ValueError(f"invalid stock daily cache for {code}: {result.error}")
            validated[code] = out
            source_for_code[code] = src
        with self._lock:
            if write_db():
                self.storage.save_stocks_daily(validated, source=source_for_code)
            if write_files():
                for code, out in validated.items():
                    if incremental and self.stock_daily_path(code).exists():
                        existing = pd.read_parquet(self.stock_daily_path(code))
                        out = DataValidator.clean_stock_daily(pd.concat([existing, out], ignore_index=True), code)
                    self._atomic_write_parquet(out, self.stock_daily_path(code))

    def get_last_date(self, code: str) -> pd.Timestamp | None:
        df = self.read_stock_daily(code)
        if df.empty or "date" not in df.columns:
            return None
        return pd.to_datetime(df["date"]).max()

    def has_complete_range(self, code: str, start_date: str, end_date: str) -> bool:
        df = self.read_stock_daily(code)
        if df.empty:
            return False
        dates = pd.to_datetime(df["date"])
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return dates.min() <= start and dates.max() >= end

    def slice_stock_daily(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.read_stock_daily(code)
        if df.empty:
            return df
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    def merge_stock_daily(self, code: str, new_df: pd.DataFrame, source: str) -> pd.DataFrame:
        old_df = self.read_stock_daily(code)
        frames = [df for df in [old_df, new_df] if df is not None and not df.empty]
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        merged = DataValidator.clean_stock_daily(merged, code)
        self.save_stock_daily(code, merged, source=source)
        return merged

    def read_stock_basic(self) -> pd.DataFrame:
        if read_db_first():
            df = self.storage.read_stock_basic()
            if not df.empty:
                return df
            if not write_files():
                return pd.DataFrame()
        path = self.stock_basic_path()
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def save_stock_basic(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            raise ValueError("refuse to save empty stock basic cache")
        if write_db():
            self.storage.save_stock_basic(df)
        if write_files():
            self._atomic_write_parquet(df, self.stock_basic_path())

    def update_status(self, row: dict) -> None:
        with self._lock:
            if write_db():
                self.storage.update_source_status(row)
            if not write_files():
                return
            path = self.metadata_dir / "update_status.csv"
            if path.exists():
                status = pd.read_csv(path)
            else:
                status = pd.DataFrame(columns=self.STATUS_COLUMNS)
            row = {col: row.get(col, "") for col in self.STATUS_COLUMNS}
            status = status[status["code"].astype(str) != str(row["code"])]
            status = pd.concat([status, pd.DataFrame([row])], ignore_index=True)
            status.to_csv(path, index=False, encoding="utf-8-sig")

    def update_status_many(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self._lock:
            if write_db():
                self.storage.update_source_status_many(rows)
            if not write_files():
                return
            path = self.metadata_dir / "update_status.csv"
            if path.exists():
                status = pd.read_csv(path)
            else:
                status = pd.DataFrame(columns=self.STATUS_COLUMNS)
            incoming = pd.DataFrame([{col: row.get(col, "") for col in self.STATUS_COLUMNS} for row in rows])
            codes = incoming["code"].astype(str).unique()
            status = status[~status["code"].astype(str).isin(codes)]
            status = pd.concat([status, incoming], ignore_index=True)
            status.to_csv(path, index=False, encoding="utf-8-sig")

    def append_log(self, row: dict) -> None:
        with self._lock:
            if write_db():
                self.storage.append_update_log(row)
            if not write_files():
                return
            path = self.metadata_dir / "update_log.csv"
            log_row = {col: row.get(col, "") for col in self.LOG_COLUMNS}
            exists = path.exists()
            pd.DataFrame([log_row]).to_csv(path, mode="a", header=not exists, index=False, encoding="utf-8-sig")

    def append_log_many(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self._lock:
            if write_db():
                self.storage.append_update_log_many(rows)
            if not write_files():
                return
            path = self.metadata_dir / "update_log.csv"
            log_rows = pd.DataFrame([{col: row.get(col, "") for col in self.LOG_COLUMNS} for row in rows])
            exists = path.exists()
            log_rows.to_csv(path, mode="a", header=not exists, index=False, encoding="utf-8-sig")

    @staticmethod
    def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.parquet")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
