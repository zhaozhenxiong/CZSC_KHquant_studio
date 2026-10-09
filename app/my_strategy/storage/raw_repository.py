#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DB-first repository for raw daily market data."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from my_strategy.storage.db_paths import raw_db_path
from my_strategy.storage.warehouse_schema import (
    SCHEMA_VERSION,
    create_views,
    init_warehouse,
    validate_warehouse_write,
)

DEFAULT_BUSY_TIMEOUT_MS = 30_000
DEFAULT_WRITE_RETRIES = 3
DEFAULT_WRITE_RETRY_BACKOFF_SECONDS = 0.25

logger = logging.getLogger(__name__)


class RawMarketRepository:
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        busy_timeout_ms: int | None = None,
        write_retries: int | None = None,
        write_retry_backoff_seconds: float | None = None,
    ):
        self.db_path = Path(db_path) if db_path else raw_db_path()
        if not self.db_path.is_absolute():
            self.db_path = Path.cwd() / self.db_path
        self.busy_timeout_ms = max(
            1,
            int(
                busy_timeout_ms
                if busy_timeout_ms is not None
                else os.environ.get("KHQUANT_SQLITE_BUSY_TIMEOUT_MS", DEFAULT_BUSY_TIMEOUT_MS)
            ),
        )
        self.write_retries = max(
            0,
            int(
                write_retries
                if write_retries is not None
                else os.environ.get("KHQUANT_SQLITE_WRITE_RETRIES", DEFAULT_WRITE_RETRIES)
            ),
        )
        self.write_retry_backoff_seconds = max(
            0.0,
            float(
                write_retry_backoff_seconds
                if write_retry_backoff_seconds is not None
                else os.environ.get(
                    "KHQUANT_SQLITE_WRITE_RETRY_BACKOFF_SECONDS",
                    DEFAULT_WRITE_RETRY_BACKOFF_SECONDS,
                )
            ),
        )
        self._schema_initialized = False
        self._schema_lock = threading.Lock()

    def exists(self) -> bool:
        return self.db_path.exists()

    def _open_connection(self, *, read_only: bool) -> sqlite3.Connection:
        timeout_seconds = self.busy_timeout_ms / 1000.0
        if read_only:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=timeout_seconds)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        if read_only:
            conn.execute("PRAGMA query_only=ON")
        else:
            conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _schema_is_current(self) -> bool:
        if not self.db_path.exists() or self.db_path.stat().st_size == 0:
            return False
        conn = self._open_connection(read_only=True)
        try:
            validate_warehouse_write(conn, role="raw")
            row = conn.execute(
                "SELECT value FROM warehouse_meta WHERE key = 'schema_version'"
            ).fetchone()
            role = conn.execute("SELECT value FROM warehouse_meta WHERE key='database_role'").fetchone()
            return row is not None and str(row["value"]) == str(SCHEMA_VERSION) and role is not None and role["value"] == "raw"
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return False
            raise
        finally:
            conn.close()

    def _is_locked_error(self, exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return "locked" in message or "busy" in message

    def _retry_delay(self, attempt: int, *, operation: str) -> None:
        delay = min(self.write_retry_backoff_seconds * (2**attempt), 5.0)
        logger.warning(
            "SQLite busy during %s; retry %s/%s in %.2fs",
            operation,
            attempt + 1,
            self.write_retries,
            delay,
        )
        if delay > 0:
            time.sleep(delay)

    def _ensure_schema(self) -> None:
        if self._schema_initialized:
            return
        with self._schema_lock:
            if self._schema_initialized:
                return
            if self._schema_is_current():
                self._schema_initialized = True
                return

            attempts = self.write_retries + 1
            for attempt in range(attempts):
                conn = self._open_connection(read_only=False)
                try:
                    init_warehouse(conn, role="raw")
                    conn.commit()
                    self._schema_initialized = True
                    return
                except sqlite3.OperationalError as exc:
                    conn.rollback()
                    if not self._is_locked_error(exc) or attempt + 1 >= attempts:
                        raise
                    self._retry_delay(attempt, operation="raw schema initialization")
                finally:
                    conn.close()

    def connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        """Open a configured connection without making reads acquire a write lock."""

        if read_only:
            return self._open_connection(read_only=True)
        self._ensure_schema()
        conn = self._open_connection(read_only=False)
        try:
            validate_warehouse_write(conn, role="raw")
            return conn
        except Exception:
            conn.close()
            raise

    def _begin_immediate_with_retry(self, conn: sqlite3.Connection) -> None:
        attempts = self.write_retries + 1
        for attempt in range(attempts):
            try:
                conn.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as exc:
                if not self._is_locked_error(exc) or attempt + 1 >= attempts:
                    raise
                self._retry_delay(attempt, operation="raw write lock acquisition")

    def _commit_with_retry(self, conn: sqlite3.Connection) -> None:
        attempts = self.write_retries + 1
        for attempt in range(attempts):
            try:
                conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if not self._is_locked_error(exc) or attempt + 1 >= attempts:
                    raise
                self._retry_delay(attempt, operation="raw transaction commit")

    @contextmanager
    def read_connection(self):
        """Yield a query-only connection that never initializes or mutates schema."""

        conn = self.connect(read_only=True)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        """Acquire the write lock with bounded retries, then commit atomically."""

        conn = self.connect()
        try:
            self._begin_immediate_with_retry(conn)
            yield conn
            self._commit_with_retry(conn)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def list_stocks(self) -> list[str]:
        if not self.db_path.exists():
            return []
        with self.read_connection() as conn:
            rows = conn.execute("SELECT stock FROM securities ORDER BY stock").fetchall()
        return [str(row["stock"]) for row in rows]

    def has_stock(self, code: str) -> bool:
        if not self.db_path.exists():
            return False
        with self.read_connection() as conn:
            row = conn.execute("SELECT 1 FROM stock_daily_normalized WHERE stock = ? LIMIT 1", (code,)).fetchone()
        return row is not None

    def read_stock_daily(self, code: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()
        clauses = ["stock = ?"]
        params: list[Any] = [code]
        if start_date:
            clauses.append("date >= ?")
            params.append(str(start_date))
        if end_date:
            clauses.append("date <= ?")
            params.append(str(end_date))
        sql = f"""
            SELECT stock AS code, date, open, high, low, close,
                   factor_open, factor_high, factor_low, factor_close,
                   volume, amount, turnover, pct_chg,
                   open AS trade_open, high AS trade_high, low AS trade_low, close AS trade_close,
                   source, source_file, imported_at AS updated_at
            FROM stock_daily_normalized
            WHERE {' AND '.join(clauses)}
            ORDER BY date
        """
        with self.read_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def read_stock_daily_batch(
        self,
        stocks: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        chunk_size: int = 500,
    ) -> pd.DataFrame:
        """Read daily bars for many stocks with chunked SQL IN queries."""
        if not self.db_path.exists() or not stocks:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        ordered_stocks = [str(stock) for stock in stocks if str(stock).strip()]
        with self.read_connection() as conn:
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-200000")
            conn.execute("PRAGMA mmap_size=268435456")
            for start in range(0, len(ordered_stocks), chunk_size):
                chunk = ordered_stocks[start : start + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                clauses = [f"stock IN ({placeholders})"]
                params: list[Any] = list(chunk)
                if start_date:
                    clauses.append("date >= ?")
                    params.append(str(start_date))
                if end_date:
                    clauses.append("date <= ?")
                    params.append(str(end_date))
                sql = f"""
                    SELECT stock, stock AS code, date, open, high, low, close,
                           factor_open, factor_high, factor_low, factor_close,
                           volume, amount, turnover, pct_chg,
                           open AS trade_open, high AS trade_high, low AS trade_low, close AS trade_close,
                           source, source_file, imported_at AS updated_at
                    FROM stock_daily_normalized
                    WHERE {' AND '.join(clauses)}
                    ORDER BY stock, date
                """
                frame = pd.read_sql_query(sql, conn, params=params)
                if not frame.empty:
                    frames.append(frame)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["stock", "date"]).reset_index(drop=True)

    def stock_daily_coverage(
        self,
        stocks: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not self.db_path.exists() or not stocks:
            return {}
        out: dict[str, dict[str, Any]] = {}
        ordered_stocks = [str(stock) for stock in stocks if str(stock).strip()]
        with self.read_connection() as conn:
            for start in range(0, len(ordered_stocks), 500):
                chunk = ordered_stocks[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                clauses = [f"stock IN ({placeholders})"]
                params: list[Any] = list(chunk)
                if start_date:
                    clauses.append("date >= ?")
                    params.append(str(start_date))
                if end_date:
                    clauses.append("date <= ?")
                    params.append(str(end_date))
                rows = conn.execute(
                    f"""
                    SELECT stock, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS rows
                    FROM stock_daily_normalized
                    WHERE {' AND '.join(clauses)}
                    GROUP BY stock
                    """,
                    params,
                ).fetchall()
                for row in rows:
                    out[str(row["stock"])] = {
                        "first_date": row["first_date"],
                        "last_date": row["last_date"],
                        "rows": int(row["rows"] or 0),
                    }
        return out

    def upsert_stock_daily(self, code: str, df: pd.DataFrame, *, source: str = "unknown") -> int:
        """Upsert only the supplied daily bars into the canonical raw database."""
        if df is None or df.empty:
            return 0
        imported_at = datetime.now().astimezone().isoformat(timespec="seconds")
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out = out.dropna(subset=["date"])
        out["stock"] = code
        out["source"] = source
        out["updated_at"] = imported_at
        out["source_file"] = f"db://stock_daily/{code}"
        out["imported_at"] = imported_at
        out["row_number"] = range(len(out))

        raw_cols = [
            "stock",
            "date",
            "row_number",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover",
            "pct_chg",
            "trade_open",
            "trade_high",
            "trade_low",
            "trade_close",
            "source",
            "updated_at",
            "source_file",
            "imported_at",
        ]
        raw = _select_columns(out, raw_cols)
        # The source-row position is not a stable database identity.  Keep one
        # canonical audit row per source/date so repeated incremental updates
        # cannot accumulate duplicates.
        raw = raw.drop_duplicates(subset=["stock", "date", "source_file"], keep="last").copy()
        raw["row_number"] = 0
        normalized = _normalize_daily(out, code=code, imported_at=imported_at)
        with self.transaction() as conn:
            _delete_raw_dates(conn, code, raw["date"].tolist())
            _append_dataframe(conn, "stock_daily_raw", raw)
            _upsert_dataframe(conn, "stock_daily_normalized", normalized, key_columns=["stock", "date"])
            coverage = conn.execute(
                """
                SELECT MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS rows
                FROM stock_daily_normalized
                WHERE stock = ?
                """,
                (code,),
            ).fetchone()
            raw_rows = conn.execute(
                "SELECT COUNT(*) AS rows FROM stock_daily_raw WHERE stock = ?", (code,)
            ).fetchone()
            conn.execute(
                """
                INSERT INTO securities (stock, exchange, name, first_date, last_date, raw_rows, source_file, updated_at, imported_at)
                VALUES (?, ?, '', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stock) DO UPDATE SET
                    first_date=excluded.first_date,
                    last_date=excluded.last_date,
                    raw_rows=excluded.raw_rows,
                    source_file=excluded.source_file,
                    updated_at=excluded.updated_at,
                    imported_at=excluded.imported_at
                """,
                (
                    code,
                    code.split(".")[-1] if "." in code else "",
                    coverage["first_date"],
                    coverage["last_date"],
                    int(raw_rows["rows"] or 0),
                    f"db://stock_daily/{code}",
                    imported_at,
                    imported_at,
                ),
            )
            create_views(conn)
        return int(len(normalized))

    def upsert_stocks_daily(
        self,
        code_to_df: dict[str, pd.DataFrame],
        *,
        source: str | dict[str, str] = "unknown",
    ) -> int:
        """Bulk upsert many stocks' daily bars in a single SQLite transaction.

        This is the write-side counterpart of the batch update flow: it avoids
        the per-stock transaction overhead that dominates large daily updates.
        The ``source`` argument may be a single string or a per-code mapping.
        """
        code_to_df = {code: df for code, df in code_to_df.items() if df is not None and not df.empty}
        if not code_to_df:
            return 0
        imported_at = datetime.now().astimezone().isoformat(timespec="seconds")

        raw_frames: list[pd.DataFrame] = []
        normalized_frames: list[pd.DataFrame] = []
        raw_cols = [
            "stock",
            "date",
            "row_number",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover",
            "pct_chg",
            "trade_open",
            "trade_high",
            "trade_low",
            "trade_close",
            "source",
            "updated_at",
            "source_file",
            "imported_at",
        ]

        for code, df in code_to_df.items():
            source_for_code = source.get(code, "unknown") if isinstance(source, dict) else source
            out = df.copy()
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            out = out.dropna(subset=["date"])
            if out.empty:
                continue
            out["stock"] = code
            out["source"] = source_for_code
            out["updated_at"] = imported_at
            out["source_file"] = f"db://stock_daily/{code}"
            out["imported_at"] = imported_at
            out["row_number"] = range(len(out))

            raw = _select_columns(out, raw_cols)
            raw = raw.drop_duplicates(subset=["stock", "date", "source_file"], keep="last").copy()
            raw["row_number"] = 0
            raw_frames.append(raw)
            normalized_frames.append(_normalize_daily(out, code=code, imported_at=imported_at))

        if not normalized_frames:
            return 0
        combined_raw = pd.concat(raw_frames, ignore_index=True)
        combined_normalized = pd.concat(normalized_frames, ignore_index=True)

        with self.transaction() as conn:
            for code, df in code_to_df.items():
                dates = (
                    pd.to_datetime(df["date"], errors="coerce")
                    .dt.strftime("%Y-%m-%d")
                    .dropna()
                    .tolist()
                )
                _delete_raw_dates(conn, code, dates)
            _append_dataframe(conn, "stock_daily_raw", combined_raw)
            _upsert_dataframe(conn, "stock_daily_normalized", combined_normalized, key_columns=["stock", "date"])
            for code in code_to_df:
                coverage = conn.execute(
                    """
                    SELECT MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS rows
                    FROM stock_daily_normalized
                    WHERE stock = ?
                    """,
                    (code,),
                ).fetchone()
                raw_rows = conn.execute(
                    "SELECT COUNT(*) AS rows FROM stock_daily_raw WHERE stock = ?", (code,)
                ).fetchone()
                conn.execute(
                    """
                    INSERT INTO securities (stock, exchange, name, first_date, last_date, raw_rows, source_file, updated_at, imported_at)
                    VALUES (?, ?, '', ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(stock) DO UPDATE SET
                        first_date=excluded.first_date,
                        last_date=excluded.last_date,
                        raw_rows=excluded.raw_rows,
                        source_file=excluded.source_file,
                        updated_at=excluded.updated_at,
                        imported_at=excluded.imported_at
                    """,
                    (
                        code,
                        code.split(".")[-1] if "." in code else "",
                        coverage["first_date"],
                        coverage["last_date"],
                        int(raw_rows["rows"] or 0),
                        f"db://stock_daily/{code}",
                        imported_at,
                        imported_at,
                    ),
                )
            create_views(conn)
        return int(len(combined_normalized))

    def update_status(self, row: dict[str, Any]) -> None:
        payload = {
            "code": row.get("code", ""),
            "last_date": row.get("last_date", ""),
            "last_update_time": row.get("last_update_time", ""),
            "rows": int(row.get("rows") or 0),
            "source": row.get("source", ""),
            "status": row.get("status", ""),
            "error_msg": row.get("error_msg", ""),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO data_source_status (code, last_date, last_update_time, rows, source, status, error_msg, updated_at)
                VALUES (:code, :last_date, :last_update_time, :rows, :source, :status, :error_msg, :updated_at)
                ON CONFLICT(code) DO UPDATE SET
                    last_date=excluded.last_date,
                    last_update_time=excluded.last_update_time,
                    rows=excluded.rows,
                    source=excluded.source,
                    status=excluded.status,
                    error_msg=excluded.error_msg,
                    updated_at=excluded.updated_at
                """,
                payload,
            )

    def update_status_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        payloads = []
        for row in rows:
            payloads.append(
                {
                    "code": row.get("code", ""),
                    "last_date": row.get("last_date", ""),
                    "last_update_time": row.get("last_update_time", ""),
                    "rows": int(row.get("rows") or 0),
                    "source": row.get("source", ""),
                    "status": row.get("status", ""),
                    "error_msg": row.get("error_msg", ""),
                    "updated_at": now,
                }
            )
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO data_source_status (code, last_date, last_update_time, rows, source, status, error_msg, updated_at)
                VALUES (:code, :last_date, :last_update_time, :rows, :source, :status, :error_msg, :updated_at)
                ON CONFLICT(code) DO UPDATE SET
                    last_date=excluded.last_date,
                    last_update_time=excluded.last_update_time,
                    rows=excluded.rows,
                    source=excluded.source,
                    status=excluded.status,
                    error_msg=excluded.error_msg,
                    updated_at=excluded.updated_at
                """,
                payloads,
            )

    def append_log(self, row: dict[str, Any]) -> None:
        payload = {key: row.get(key, "") for key in ["time", "code", "action", "start_date", "end_date", "rows", "source", "status", "error_msg"]}
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO update_log (time, code, action, start_date, end_date, rows, source, status, error_msg)
                VALUES (:time, :code, :action, :start_date, :end_date, :rows, :source, :status, :error_msg)
                """,
                payload,
            )

    def append_log_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        payloads = []
        for row in rows:
            payloads.append(
                {key: row.get(key, "") for key in ["time", "code", "action", "start_date", "end_date", "rows", "source", "status", "error_msg"]}
            )
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO update_log (time, code, action, start_date, end_date, rows, source, status, error_msg)
                VALUES (:time, :code, :action, :start_date, :end_date, :rows, :source, :status, :error_msg)
                """,
                payloads,
            )

    def read_stock_basic(self) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()
        with self.read_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM stock_basic ORDER BY code", conn)
        if df.empty:
            return df
        payloads = []
        for _, row in df.iterrows():
            payload = json.loads(row.get("payload_json") or "{}")
            payload.update({col: row[col] for col in df.columns if col not in {"payload_json", "updated_at"} and pd.notna(row[col])})
            payloads.append(payload)
        return pd.DataFrame(payloads)

    def save_stock_basic(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        rows = []
        for _, item in df.iterrows():
            payload = item.to_dict()
            code = str(payload.get("code") or payload.get("stock") or "")
            if not code:
                continue
            rows.append(
                {
                    "code": code,
                    "name": payload.get("name"),
                    "stock_name": payload.get("stock_name"),
                    "market": payload.get("market"),
                    "latest_price": payload.get("latest_price"),
                    "total_market_cap": payload.get("total_market_cap"),
                    "turnover": payload.get("turnover"),
                    "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                    "updated_at": now,
                }
            )
        with self.transaction() as conn:
            conn.execute("DELETE FROM stock_basic")
            if rows:
                pd.DataFrame(rows).to_sql("stock_basic", conn, if_exists="append", index=False)
        return len(rows)

    def read_industry_metadata(self, *, eligible_only: bool = False) -> pd.DataFrame:
        """Read the authoritative current industry metadata snapshot."""
        if not self.db_path.exists():
            return pd.DataFrame()
        with self.read_connection() as conn:
            present = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_industry_metadata'"
            ).fetchone()
            if present is None:
                return pd.DataFrame()
            clause = " WHERE classification_eligible = 1" if eligible_only else ""
            df = pd.read_sql_query(
                f"SELECT * FROM stock_industry_metadata{clause} ORDER BY stock",
                conn,
            )
        if df.empty:
            return df
        if "classification_eligible" in df.columns:
            df["classification_eligible"] = df["classification_eligible"].astype(bool)
        else:
            df["classification_eligible"] = True
        df.attrs["khquant_data_source"] = "raw_sqlite:stock_industry_metadata"
        df.attrs["khquant_raw_db"] = str(self.db_path)
        return df

    def read_industry_exclusions(self) -> pd.DataFrame:
        """Read the authoritative current industry-exclusion snapshot."""
        if not self.db_path.exists():
            return pd.DataFrame(columns=["stock", "reason", "source_run_id", "detected_at"])
        with self.read_connection() as conn:
            present = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_industry_exclusions'"
            ).fetchone()
            if present is None:
                return pd.DataFrame(columns=["stock", "reason", "source_run_id", "detected_at"])
            df = pd.read_sql_query(
                "SELECT stock, reason, source_run_id, detected_at FROM stock_industry_exclusions ORDER BY stock",
                conn,
            )
        df.attrs["khquant_data_source"] = "raw_sqlite:stock_industry_exclusions"
        df.attrs["khquant_raw_db"] = str(self.db_path)
        return df

    def read_latest_industry_run(self) -> dict[str, Any]:
        """Read the latest industry refresh provenance without mutating SQLite."""
        if not self.db_path.exists():
            return {}
        with self.read_connection() as conn:
            present = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='industry_classification_runs'"
            ).fetchone()
            if present is None:
                return {}
            row = conn.execute(
                """
                SELECT *
                FROM industry_classification_runs
                ORDER BY finished_at DESC, run_id DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return {}
        result = dict(row)
        try:
            result["summary"] = json.loads(str(result.get("summary_json") or "{}"))
        except (TypeError, ValueError):
            result["summary"] = {}
        return result

    def replace_industry_metadata(
        self,
        frame: pd.DataFrame,
        exclusions: pd.DataFrame,
        *,
        run_id: str,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically replace current industry metadata and exclusions.

        The raw warehouse is the source of record. CSV/JSON files are generated
        by callers only after this transaction is committed and verified.
        """
        if not run_id:
            raise ValueError("run_id is required for industry metadata replacement")
        if frame is None or frame.empty or "stock" not in frame.columns:
            raise ValueError("industry metadata frame is empty or missing stock")
        if frame["stock"].fillna("").astype(str).str.strip().duplicated().any():
            raise ValueError("industry metadata contains duplicate stock codes")

        now = datetime.now().astimezone().isoformat(timespec="seconds")
        rows = _prepare_industry_metadata_rows(frame, run_id=run_id, now=now)
        exclusion_rows = _prepare_industry_exclusion_rows(exclusions, run_id=run_id, now=now)
        run_row = _prepare_industry_run_row(
            run_id=run_id,
            rows=rows,
            exclusions=exclusion_rows,
            summary=summary or {},
            now=now,
        )
        expected_eligible = int(rows["classification_eligible"].sum())

        with self.transaction() as conn:
            existing_created = {
                str(row["stock"]): str(row["created_at"])
                for row in conn.execute("SELECT stock, created_at FROM stock_industry_metadata")
            }
            rows["created_at"] = rows["stock"].map(existing_created).fillna(now)
            conn.execute(
                """
                INSERT INTO industry_classification_runs (
                    run_id, started_at, finished_at, status, taxonomy,
                    source_provider, source_endpoint, source_updated_at,
                    source_fetched_at, source_rows, metadata_rows, eligible_rows,
                    excluded_rows, coverage, summary_json
                ) VALUES (
                    :run_id, :started_at, :finished_at, :status, :taxonomy,
                    :source_provider, :source_endpoint, :source_updated_at,
                    :source_fetched_at, :source_rows, :metadata_rows, :eligible_rows,
                    :excluded_rows, :coverage, :summary_json
                )
                ON CONFLICT(run_id) DO UPDATE SET
                    finished_at=excluded.finished_at,
                    status=excluded.status,
                    taxonomy=excluded.taxonomy,
                    source_provider=excluded.source_provider,
                    source_endpoint=excluded.source_endpoint,
                    source_updated_at=excluded.source_updated_at,
                    source_fetched_at=excluded.source_fetched_at,
                    source_rows=excluded.source_rows,
                    metadata_rows=excluded.metadata_rows,
                    eligible_rows=excluded.eligible_rows,
                    excluded_rows=excluded.excluded_rows,
                    coverage=excluded.coverage,
                    summary_json=excluded.summary_json
                """,
                run_row,
            )
            conn.execute("DELETE FROM stock_industry_exclusions")
            conn.execute("DELETE FROM stock_industry_metadata")
            _append_dataframe(conn, "stock_industry_metadata", rows)
            _append_dataframe(conn, "stock_industry_exclusions", exclusion_rows)
            create_views(conn)
            stored_metadata = int(conn.execute("SELECT COUNT(*) FROM stock_industry_metadata").fetchone()[0])
            stored_eligible = int(
                conn.execute(
                    "SELECT COUNT(*) FROM stock_industry_metadata WHERE classification_eligible = 1"
                ).fetchone()[0]
            )
            stored_exclusions = int(conn.execute("SELECT COUNT(*) FROM stock_industry_exclusions").fetchone()[0])
            quick_check = str(conn.execute("PRAGMA quick_check").fetchone()[0])
            if (stored_metadata, stored_eligible, stored_exclusions, quick_check) != (
                len(rows),
                expected_eligible,
                len(exclusion_rows),
                "ok",
            ):
                raise RuntimeError(
                    "industry SQLite verification failed: "
                    f"metadata={stored_metadata}/{len(rows)} eligible={stored_eligible}/{expected_eligible} "
                    f"exclusions={stored_exclusions}/{len(exclusion_rows)} quick_check={quick_check}"
                )
        return {
            "run_id": run_id,
            "database": str(self.db_path),
            "metadata_table": "stock_industry_metadata",
            "exclusions_table": "stock_industry_exclusions",
            "runs_table": "industry_classification_runs",
            "metadata_rows": stored_metadata,
            "eligible_rows": stored_eligible,
            "excluded_rows": stored_exclusions,
            "quick_check": quick_check,
        }


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame()
    for col in columns:
        out[col] = df[col] if col in df.columns else None
    return out


def _truthy_value(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _prepare_industry_metadata_rows(frame: pd.DataFrame, *, run_id: str, now: str) -> pd.DataFrame:
    columns = [
        "stock",
        "stock_name",
        "board",
        "industry",
        "neutral_group",
        "industry_l1_code",
        "industry_l1_name",
        "industry_l2_code",
        "industry_l2_name",
        "industry_taxonomy",
        "industry_source",
        "industry_source_provider",
        "industry_source_endpoint",
        "industry_confidence",
        "industry_confidence_score",
        "industry_confidence_method",
        "industry_source_updated_at",
        "industry_source_fetched_at",
        "metadata_updated_at",
        "classification_eligible",
        "source_run_id",
        "source_payload_json",
        "created_at",
        "updated_at",
    ]
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = ""
    text_columns = [
        column
        for column in columns
        if column not in {"industry_confidence_score", "classification_eligible"}
    ]
    for column in text_columns:
        out[column] = out[column].fillna("").astype(str).str.strip()
    source_parts = out["industry_source"].str.split(":", n=1, expand=True)
    out["industry_source_provider"] = out["industry_source_provider"].mask(
        out["industry_source_provider"].eq(""), source_parts[0].fillna("")
    )
    if source_parts.shape[1] > 1:
        out["industry_source_endpoint"] = out["industry_source_endpoint"].mask(
            out["industry_source_endpoint"].eq(""), source_parts[1].fillna("")
        )
    out["classification_eligible"] = out["classification_eligible"].map(_truthy_value).astype(int)
    inferred_score = out["industry_confidence"].str.lower().map({"high": 1.0, "medium": 0.7, "low": 0.4, "none": 0.0})
    out["industry_confidence_score"] = pd.to_numeric(
        out["industry_confidence_score"], errors="coerce"
    ).fillna(inferred_score).fillna(0.0).clip(0.0, 1.0)
    out["industry_confidence"] = out["industry_confidence"].mask(
        out["industry_confidence"].eq(""), "none"
    )
    out["industry_source_fetched_at"] = out["industry_source_fetched_at"].mask(
        out["industry_source_fetched_at"].eq(""), now
    )
    out["metadata_updated_at"] = out["metadata_updated_at"].mask(
        out["metadata_updated_at"].eq(""), now
    )
    out["source_run_id"] = run_id
    out["source_payload_json"] = out["source_payload_json"].mask(
        out["source_payload_json"].eq(""), "{}"
    )
    out["created_at"] = now
    out["updated_at"] = now
    return out[columns].sort_values("stock").reset_index(drop=True)


def _prepare_industry_exclusion_rows(
    exclusions: pd.DataFrame,
    *,
    run_id: str,
    now: str,
) -> pd.DataFrame:
    if exclusions is None or exclusions.empty:
        return pd.DataFrame(columns=["stock", "reason", "source_run_id", "detected_at"])
    if {"stock", "reason"} - set(exclusions.columns):
        raise ValueError("industry exclusions must contain stock and reason")
    out = exclusions[["stock", "reason"]].copy()
    out["stock"] = out["stock"].fillna("").astype(str).str.strip()
    out["reason"] = out["reason"].fillna("").astype(str).str.strip()
    out = out[(out["stock"] != "") & (out["reason"] != "")]
    if out["stock"].duplicated().any():
        raise ValueError("industry exclusions contain duplicate stock codes")
    out["source_run_id"] = run_id
    out["detected_at"] = now
    return out.sort_values("stock").reset_index(drop=True)


def _prepare_industry_run_row(
    *,
    run_id: str,
    rows: pd.DataFrame,
    exclusions: pd.DataFrame,
    summary: dict[str, Any],
    now: str,
) -> dict[str, Any]:
    classification = summary.get("classification") if isinstance(summary.get("classification"), dict) else summary
    provider = str(classification.get("source_provider") or "")
    endpoint = str(classification.get("source_endpoint") or "")
    if not provider or not endpoint:
        source = str(classification.get("source") or "")
        provider_part, _, endpoint_part = source.partition(":")
        provider = provider or provider_part
        endpoint = endpoint or endpoint_part
    fetched = rows["industry_source_fetched_at"].replace("", pd.NA).dropna()
    updated = rows["industry_source_updated_at"].replace("", pd.NA).dropna()
    eligible = int(rows["classification_eligible"].sum())
    metadata_rows = int(len(rows))
    return {
        "run_id": run_id,
        "started_at": str(summary.get("created_at") or now),
        "finished_at": now,
        "status": "success",
        "taxonomy": str(classification.get("taxonomy") or ""),
        "source_provider": provider,
        "source_endpoint": endpoint,
        "source_updated_at": str(classification.get("source_updated_at") or (updated.max() if not updated.empty else "")),
        "source_fetched_at": str(classification.get("source_fetched_at") or (fetched.max() if not fetched.empty else now)),
        "source_rows": int(classification.get("source_rows") or 0),
        "metadata_rows": metadata_rows,
        "eligible_rows": eligible,
        "excluded_rows": int(len(exclusions)),
        "coverage": float(classification.get("coverage") or (eligible / metadata_rows if metadata_rows else 0.0)),
        "summary_json": json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str),
    }


def _normalize_daily(df: pd.DataFrame, *, code: str, imported_at: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["open", "high", "low", "close", "trade_open", "trade_high", "trade_low", "trade_close", "volume", "amount", "turnover", "pct_chg"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for price_col, trade_col in [("open", "trade_open"), ("high", "trade_high"), ("low", "trade_low"), ("close", "trade_close")]:
        if trade_col in out.columns:
            trade = pd.to_numeric(out[trade_col], errors="coerce")
            base = pd.to_numeric(out.get(price_col), errors="coerce")
            out[price_col] = trade.where(trade > 0, base)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["date"])
    out["_has_trade_price"] = (pd.to_numeric(out.get("trade_close"), errors="coerce") > 0).fillna(False).astype(int) if "trade_close" in out.columns else 0
    out["_order"] = range(len(out))
    out = out.sort_values(["date", "_has_trade_price", "_order"], ascending=[True, False, True])
    duplicate_counts = out.groupby("date")["date"].transform("size")
    out = out.drop_duplicates(subset=["date"], keep="first")
    normalized = pd.DataFrame(
        {
            "stock": code,
            "date": out["date"],
            "open": out.get("open"),
            "high": out.get("high"),
            "low": out.get("low"),
            "close": out.get("close"),
            "factor_open": out.get("open"),
            "factor_high": out.get("high"),
            "factor_low": out.get("low"),
            "factor_close": out.get("close"),
            "volume": out.get("volume"),
            "amount": out.get("amount"),
            "turnover": out.get("turnover"),
            "pct_chg": out.get("pct_chg"),
            "source": out.get("source"),
            "source_file": out.get("source_file"),
            "duplicate_source_rows": duplicate_counts.loc[out.index].astype(int).values,
            "has_trade_price": out["_has_trade_price"].astype(int).values,
            "imported_at": imported_at,
        }
    )
    return normalized


def _append_dataframe(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    clean = df.where(pd.notna(df), None)
    clean.to_sql(table, conn, if_exists="append", index=False, chunksize=2000)
    return int(len(clean))


def _delete_raw_dates(conn: sqlite3.Connection, code: str, dates: list[object]) -> None:
    """Replace only the raw audit rows for dates present in an update payload."""
    values = sorted({str(value) for value in dates if value})
    for start in range(0, len(values), 500):
        batch = values[start : start + 500]
        placeholders = ", ".join("?" for _ in batch)
        conn.execute(
            f"DELETE FROM stock_daily_raw WHERE stock = ? AND date IN ({placeholders})",
            [code, *batch],
        )


def _upsert_dataframe(
    conn: sqlite3.Connection,
    table: str,
    df: pd.DataFrame,
    *,
    key_columns: list[str],
) -> int:
    """Insert or update a dataframe using the table's stable logical key."""
    if df.empty:
        return 0
    columns = list(df.columns)
    update_columns = [column for column in columns if column not in key_columns]
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    conflict_columns = ", ".join(f'"{column}"' for column in key_columns)
    updates = ", ".join(f'"{column}" = excluded."{column}"' for column in update_columns)
    statement = (
        f'INSERT INTO "{table}" ({quoted_columns}) VALUES ({placeholders}) '
        f"ON CONFLICT ({conflict_columns}) DO UPDATE SET {updates}"
    )
    clean = df.where(pd.notna(df), None)
    conn.executemany(statement, clean[columns].itertuples(index=False, name=None))
    return int(len(clean))
