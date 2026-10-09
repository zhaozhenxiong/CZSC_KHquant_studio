from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from my_strategy.data_manager.updater import DailyUpdater
from my_strategy.storage.raw_repository import RawMarketRepository


class RawMarketRepositoryTests(unittest.TestCase):
    @staticmethod
    def _daily_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": ["2026-07-28"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.0],
                "close": [10.5],
                "volume": [100.0],
            }
        )

    def test_read_connection_is_query_only_and_does_not_write_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "raw.db"
            repository = RawMarketRepository(db_path, busy_timeout_ms=50)
            repository.upsert_stock_daily("000001.SZ", self._daily_frame(), source="unit")

            with closing(sqlite3.connect(db_path)) as conn, conn:
                before = conn.execute(
                    "SELECT updated_at FROM warehouse_meta WHERE key='schema_version'"
                ).fetchone()[0]
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "UPDATE warehouse_meta SET value=value WHERE key='schema_version'"
                )

                # WAL permits this read while another connection owns the write
                # lock. It would fail if connect() still ran init_warehouse().
                frame = repository.read_stock_daily("000001.SZ")
                self.assertEqual(frame["close"].tolist(), [10.5])
                batch = repository.read_stock_daily_batch(["000001.SZ"])
                self.assertEqual(batch["close"].tolist(), [10.5])
                with repository.read_connection() as read_conn:
                    self.assertEqual(read_conn.execute("PRAGMA query_only").fetchone()[0], 1)
                    self.assertEqual(read_conn.execute("PRAGMA busy_timeout").fetchone()[0], 50)
                    with self.assertRaises(sqlite3.OperationalError):
                        read_conn.execute(
                            "UPDATE warehouse_meta SET value='invalid' WHERE key='schema_version'"
                        )
                conn.rollback()

            with closing(sqlite3.connect(db_path)) as conn:
                after = conn.execute(
                    "SELECT updated_at FROM warehouse_meta WHERE key='schema_version'"
                ).fetchone()[0]
            self.assertEqual(after, before)

    def test_write_transaction_retries_until_short_lock_is_released(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "raw.db"
            repository = RawMarketRepository(
                db_path,
                busy_timeout_ms=20,
                write_retries=8,
                write_retry_backoff_seconds=0.01,
            )
            repository.upsert_stock_daily("000001.SZ", self._daily_frame(), source="unit")

            blocker = sqlite3.connect(db_path)
            blocker.execute("BEGIN IMMEDIATE")
            outcome: dict[str, Exception | bool] = {}

            def write_status() -> None:
                try:
                    repository.update_status(
                        {
                            "code": "000001.SZ",
                            "status": "success",
                            "rows": 1,
                        }
                    )
                    outcome["success"] = True
                except Exception as exc:  # pragma: no cover - assertion reports it
                    outcome["error"] = exc

            worker = threading.Thread(target=write_status)
            worker.start()
            time.sleep(0.12)
            blocker.commit()
            blocker.close()
            worker.join(timeout=3)

            self.assertFalse(worker.is_alive())
            self.assertNotIn("error", outcome)
            self.assertTrue(outcome.get("success"))
            with repository.read_connection() as conn:
                row = conn.execute(
                    "SELECT status, rows FROM data_source_status WHERE code='000001.SZ'"
                ).fetchone()
            self.assertEqual(tuple(row), ("success", 1))

    def test_wal_writer_succeeds_while_training_style_reader_is_open(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "raw.db"
            repository = RawMarketRepository(db_path, busy_timeout_ms=100)
            repository.upsert_stock_daily("000001.SZ", self._daily_frame(), source="unit")

            reader = repository.connect(read_only=True)
            try:
                reader.execute("BEGIN")
                self.assertEqual(
                    reader.execute(
                        "SELECT COUNT(*) FROM stock_daily_normalized"
                    ).fetchone()[0],
                    1,
                )
                repository.update_status(
                    {"code": "000001.SZ", "status": "success", "rows": 1}
                )
            finally:
                reader.rollback()
                reader.close()

            with repository.read_connection() as conn:
                status = conn.execute(
                    "SELECT status FROM data_source_status WHERE code='000001.SZ'"
                ).fetchone()[0]
            self.assertEqual(status, "success")

    def test_incremental_upsert_updates_only_supplied_dates_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "raw.db"
            repository = RawMarketRepository(db_path)
            first = pd.DataFrame(
                {
                    "date": ["2026-07-28", "2026-07-29"],
                    "open": [10.0, 20.0],
                    "high": [11.0, 21.0],
                    "low": [9.0, 19.0],
                    "close": [10.5, 20.5],
                    "volume": [100.0, 200.0],
                }
            )
            update = pd.DataFrame(
                {
                    "date": ["2026-07-29", "2026-07-30"],
                    "open": [22.0, 30.0],
                    "high": [23.0, 31.0],
                    "low": [21.0, 29.0],
                    "close": [22.5, 30.5],
                    "volume": [220.0, 300.0],
                }
            )

            self.assertEqual(repository.upsert_stock_daily("000001.SZ", first, source="unit"), 2)
            self.assertEqual(repository.upsert_stock_daily("000001.SZ", update, source="unit"), 2)
            self.assertEqual(repository.upsert_stock_daily("000001.SZ", update, source="unit"), 2)

            conn = sqlite3.connect(db_path)
            try:
                normalized = conn.execute(
                    "SELECT date, close FROM stock_daily_normalized WHERE stock = ? ORDER BY date",
                    ("000001.SZ",),
                ).fetchall()
                raw_rows = conn.execute(
                    "SELECT COUNT(*) FROM stock_daily_raw WHERE stock = ?", ("000001.SZ",)
                ).fetchone()[0]
                coverage = conn.execute(
                    "SELECT first_date, last_date, raw_rows FROM securities WHERE stock = ?", ("000001.SZ",)
                ).fetchone()
            finally:
                conn.close()

            self.assertEqual(normalized, [("2026-07-28", 10.5), ("2026-07-29", 22.5), ("2026-07-30", 30.5)])
            self.assertEqual(raw_rows, 3)
            self.assertEqual(coverage, ("2026-07-28", "2026-07-30", 3))

    def test_daily_updater_persists_only_downloaded_dates(self) -> None:
        updater = DailyUpdater.__new__(DailyUpdater)
        updater.config = {
            "update": {"start_date": "2026-07-28", "auto_incremental": True},
            "cache": {"min_rows_for_valid_stock": 1},
        }
        updater.cache = Mock()
        updater.cache.read_stock_daily.return_value = pd.DataFrame(
            {
                "date": ["2026-07-28"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.0],
                "close": [10.5],
                "trade_open": [10.0],
                "trade_high": [11.0],
                "trade_low": [9.0],
                "trade_close": [10.5],
                "volume": [100.0],
            }
        )
        new_daily = pd.DataFrame(
            {
                "date": ["2026-07-29"],
                "open": [11.0],
                "high": [12.0],
                "low": [10.0],
                "close": [11.5],
                "trade_open": [11.0],
                "trade_high": [12.0],
                "trade_low": [10.0],
                "trade_close": [11.5],
                "volume": [120.0],
            }
        )
        updater.downloader = Mock()
        updater.downloader.download_stock_daily.return_value = SimpleNamespace(
            ok=True, data=new_daily, source="unit", error=""
        )

        result = updater.update_one("000001.SZ", end_date="2026-07-29")

        self.assertEqual(result["status"], "success")
        saved = updater.cache.save_stock_daily.call_args
        self.assertEqual(saved.kwargs["incremental"], True)
        self.assertEqual(saved.kwargs["source"], "unit")
        self.assertEqual(saved.args[0], "000001.SZ")
        self.assertEqual(saved.args[1]["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-07-29"])

    def test_daily_updater_accepts_structurally_valid_short_listing_history(self) -> None:
        updater = DailyUpdater.__new__(DailyUpdater)
        updater.config = {
            "update": {"start_date": "2015-01-01", "auto_incremental": True},
            "cache": {"min_rows_for_valid_stock": 100},
        }
        updater.cache = Mock()
        old_dates = pd.bdate_range("2026-07-27", periods=5)
        updater.cache.read_stock_daily.return_value = pd.DataFrame(
            {
                "date": old_dates,
                "open": [10.0] * 5,
                "high": [11.0] * 5,
                "low": [9.0] * 5,
                "close": [10.5] * 5,
                "trade_close": [10.5] * 5,
                "volume": [100.0] * 5,
            }
        )
        new_dates = pd.bdate_range("2026-08-03", periods=14)
        new_daily = pd.DataFrame(
            {
                "date": new_dates,
                "open": [10.1] * 14,
                "high": [11.1] * 14,
                "low": [9.1] * 14,
                "close": [10.6] * 14,
                "volume": [120.0] * 14,
            }
        )
        updater.downloader = Mock()
        updater.downloader.download_stock_daily.return_value = SimpleNamespace(
            ok=True, data=new_daily, source="unit", error=""
        )

        result = updater.update_one("688825.SH", end_date="2026-08-20")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["rows"], 19)
        self.assertIn("stored but not warm-up eligible", result["error_msg"])
        updater.cache.save_stock_daily.assert_called_once()
        self.assertTrue(updater.cache.save_stock_daily.call_args.kwargs["incremental"])

    def test_batch_updater_accepts_structurally_valid_short_listing_history(self) -> None:
        updater = DailyUpdater.__new__(DailyUpdater)
        updater.config = {
            "update": {"start_date": "2015-01-01", "auto_incremental": True},
            "cache": {"min_rows_for_valid_stock": 100},
            "tushare": {"batch_size": 2, "bulk_batch_size": 200},
        }
        updater.workers = 1
        updater.cache = Mock()
        old_dates = pd.bdate_range("2026-07-27", periods=5)
        updater.cache.read_stock_daily.return_value = pd.DataFrame(
            {
                "date": old_dates,
                "open": [10.0] * 5,
                "high": [11.0] * 5,
                "low": [9.0] * 5,
                "close": [10.5] * 5,
                "trade_close": [10.5] * 5,
                "volume": [100.0] * 5,
            }
        )
        new_dates = pd.bdate_range("2026-08-03", periods=14)
        new_daily = pd.DataFrame(
            {
                "date": new_dates,
                "code": ["688825.SH"] * 14,
                "open": [10.1] * 14,
                "high": [11.1] * 14,
                "low": [9.1] * 14,
                "close": [10.6] * 14,
                "volume": [120.0] * 14,
            }
        )
        updater.downloader = Mock()
        updater.downloader.download_stocks_daily.return_value = SimpleNamespace(
            ok=True, data=new_daily, source="unit", error=""
        )

        report = updater.update_batch(["688825.SH"], end_date="2026-08-20")

        row = report.iloc[0]
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["rows"], 19)
        self.assertIn("stored but not warm-up eligible", row["error_msg"])
        updater.cache.save_stocks_daily.assert_called_once()
        self.assertTrue(updater.cache.save_stocks_daily.call_args.kwargs["incremental"])


if __name__ == "__main__":
    unittest.main()
