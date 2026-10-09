"""Raw market-data access used by ingestion; CZSC results use their own store."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from my_strategy.storage.db_paths import read_db_first, storage_mode, write_db, write_files
from my_strategy.storage.raw_repository import RawMarketRepository


@dataclass(frozen=True)
class StorageSettings:
    mode: str
    raw_db: Path
    read_db_first: bool
    write_db: bool
    write_files: bool


class DataAccessLayer:
    """Read and update raw bars and source metadata without opening a result DB."""

    def __init__(self, raw_db: str | Path | None = None):
        self.raw = RawMarketRepository(raw_db)

    @property
    def settings(self) -> StorageSettings:
        return StorageSettings(
            mode=storage_mode(), raw_db=self.raw.db_path,
            read_db_first=read_db_first(), write_db=write_db(), write_files=write_files(),
        )

    def list_stocks(self) -> list[str]:
        return self.raw.list_stocks()

    def has_stock(self, code: str) -> bool:
        return self.raw.has_stock(code)

    def read_stock_daily(self, code: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        return self.raw.read_stock_daily(code, start_date, end_date)

    def raw_daily_coverage(
        self,
        stocks: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        return self.raw.stock_daily_coverage(stocks, start_date, end_date)

    def save_stock_daily(self, code: str, df: pd.DataFrame, *, source: str = "unknown") -> int:
        return self.raw.upsert_stock_daily(code, df, source=source)

    def save_stocks_daily(self, code_to_df: dict[str, pd.DataFrame], *, source: str | dict[str, str] = "unknown") -> int:
        return self.raw.upsert_stocks_daily(code_to_df, source=source)

    def read_stock_basic(self) -> pd.DataFrame:
        return self.raw.read_stock_basic()

    def save_stock_basic(self, df: pd.DataFrame) -> int:
        return self.raw.save_stock_basic(df)

    def read_industry_metadata(self, *, eligible_only: bool = False) -> pd.DataFrame:
        return self.raw.read_industry_metadata(eligible_only=eligible_only)

    def read_industry_exclusions(self) -> pd.DataFrame:
        return self.raw.read_industry_exclusions()

    def read_latest_industry_run(self) -> dict[str, Any]:
        return self.raw.read_latest_industry_run()

    def replace_industry_metadata(
        self,
        frame: pd.DataFrame,
        exclusions: pd.DataFrame,
        *,
        run_id: str,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.raw.replace_industry_metadata(
            frame,
            exclusions,
            run_id=run_id,
            summary=summary,
        )

    def update_source_status(self, row: dict[str, Any]) -> None:
        self.raw.update_status(row)

    def update_source_status_many(self, rows: list[dict[str, Any]]) -> None:
        self.raw.update_status_many(rows)

    def append_update_log(self, row: dict[str, Any]) -> None:
        self.raw.append_log(row)

    def append_update_log_many(self, rows: list[dict[str, Any]]) -> None:
        self.raw.append_log_many(rows)

    def record_data_update(
        self,
        *,
        update_id: str,
        started_at: str,
        finished_at: str | None,
        mode: str,
        stocks_requested: str,
        end_date: str,
        report: pd.DataFrame,
    ) -> dict[str, Any]:
        finished = finished_at or datetime.now().astimezone().isoformat(timespec="seconds")
        status = report.get("status", pd.Series(dtype=str)).astype(str).str.lower() if not report.empty else pd.Series(dtype=str)
        success = int((status == "success").sum())
        failed = int((status == "failed").sum())
        skipped = int((status == "skipped").sum())
        settings = self.settings
        payload = {
            "update_id": update_id,
            "started_at": started_at,
            "finished_at": finished,
            "mode": mode,
            "stocks_requested": stocks_requested,
            "end_date": end_date,
            "storage_mode": settings.mode,
            "raw_db_path": str(settings.raw_db),
            "total": int(len(report)),
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "report_json": json.dumps(report.to_dict("records"), ensure_ascii=False, sort_keys=True, default=str),
        }
        with self.raw.transaction() as conn:
            conn.execute(
                """
                INSERT INTO data_update_runs (
                    update_id, started_at, finished_at, mode, stocks_requested, end_date,
                    storage_mode, raw_db_path, total, success, failed, skipped, report_json
                ) VALUES (
                    :update_id, :started_at, :finished_at, :mode, :stocks_requested, :end_date,
                    :storage_mode, :raw_db_path, :total, :success, :failed, :skipped, :report_json
                )
                ON CONFLICT(update_id) DO UPDATE SET
                    finished_at=excluded.finished_at,
                    total=excluded.total,
                    success=excluded.success,
                    failed=excluded.failed,
                    skipped=excluded.skipped,
                    report_json=excluded.report_json
                """,
                payload,
            )
        return {key: payload[key] for key in ["update_id", "total", "success", "failed", "skipped", "storage_mode", "raw_db_path"]}


def get_data_access(raw_db: str | Path | None = None) -> DataAccessLayer:
    return DataAccessLayer(raw_db)
