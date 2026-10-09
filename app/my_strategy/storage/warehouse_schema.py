"""SQLite schema for raw market data; CZSC outputs live in a separate database."""
from __future__ import annotations

import os
import sqlite3

# Keep the raw-data format number: retiring strategy implementations does not
# invalidate the existing OHLCV source format.
SCHEMA_VERSION = 11


def validate_warehouse_write(conn: sqlite3.Connection, *, role: str | None = "raw") -> str:
    """Reject a result DB or unsupported raw format before changing any schema."""
    if role not in {None, "raw"}:
        raise ValueError(f"Unknown warehouse role: {role}")
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='warehouse_meta'").fetchone():
        metadata = dict(conn.execute("SELECT key, value FROM warehouse_meta"))
        if int(metadata.get("schema_version", 0)) > SCHEMA_VERSION:
            raise ValueError(f"Refusing to write newer schema_version={metadata['schema_version']} (supported {SCHEMA_VERSION})")
        existing_role = metadata.get("database_role")
        if existing_role and existing_role != "raw":
            raise ValueError(f"Warehouse role mismatch: expected raw, found {existing_role}")
    return "raw"


def init_warehouse(conn: sqlite3.Connection, *, role: str | None = "raw") -> None:
    """Create raw-data tables only; result and model schemas are retired."""
    role = validate_warehouse_write(conn, role=role)
    # WAL is unsafe on some host filesystems exposed through Docker Desktop
    # bind mounts (exFAT/APFS via virtiofs), where file rename/extend
    # operations can corrupt the main DB or leave stale -wal/-shm files.
    journal_mode = os.environ.get("KHQUANT_SQLITE_JOURNAL_MODE", "WAL").upper()
    if journal_mode not in {"WAL", "DELETE", "TRUNCATE", "PERSIST", "MEMORY", "OFF"}:
        journal_mode = "WAL"
    conn.execute(f"PRAGMA journal_mode={journal_mode}")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Wait (rather than raise "database is locked") when another connection holds
    # the write lock — e.g. a running backtest writing run_registry while the
    # dashboard polls /api/runs.  Writes are short (per-stock commits), so 30s is ample.
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS warehouse_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS import_runs (
            import_id TEXT PRIMARY KEY,
            imported_at TEXT NOT NULL,
            mode TEXT NOT NULL,
            source_root TEXT NOT NULL,
            db_path TEXT NOT NULL,
            dry_run INTEGER NOT NULL DEFAULT 0,
            summary_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS securities (
            stock TEXT PRIMARY KEY,
            exchange TEXT,
            name TEXT,
            first_date TEXT,
            last_date TEXT,
            raw_rows INTEGER,
            source_file TEXT,
            updated_at TEXT,
            imported_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_daily_raw (
            stock TEXT NOT NULL,
            date TEXT NOT NULL,
            row_number INTEGER NOT NULL,
            code TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            turnover REAL,
            pct_chg REAL,
            trade_open REAL,
            trade_high REAL,
            trade_low REAL,
            trade_close REAL,
            source TEXT,
            updated_at TEXT,
            source_file TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            PRIMARY KEY (stock, date, source_file, row_number)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_daily_normalized (
            stock TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            factor_open REAL,
            factor_high REAL,
            factor_low REAL,
            factor_close REAL,
            volume REAL,
            amount REAL,
            turnover REAL,
            pct_chg REAL,
            source TEXT,
            source_file TEXT NOT NULL,
            duplicate_source_rows INTEGER NOT NULL DEFAULT 1,
            has_trade_price INTEGER NOT NULL DEFAULT 0,
            imported_at TEXT NOT NULL,
            PRIMARY KEY (stock, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_quality_issues (
            issue_id TEXT PRIMARY KEY,
            imported_at TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            source_file TEXT,
            message TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_source_status (
            code TEXT PRIMARY KEY,
            last_date TEXT,
            last_update_time TEXT,
            rows INTEGER,
            source TEXT,
            status TEXT,
            error_msg TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS update_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            code TEXT,
            action TEXT,
            start_date TEXT,
            end_date TEXT,
            rows INTEGER,
            source TEXT,
            status TEXT,
            error_msg TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_update_runs (
            update_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            mode TEXT,
            stocks_requested TEXT,
            end_date TEXT,
            storage_mode TEXT NOT NULL,
            raw_db_path TEXT NOT NULL,
            total INTEGER NOT NULL,
            success INTEGER NOT NULL,
            failed INTEGER NOT NULL,
            skipped INTEGER NOT NULL,
            report_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_basic (
            code TEXT PRIMARY KEY,
            name TEXT,
            stock_name TEXT,
            market TEXT,
            latest_price REAL,
            total_market_cap REAL,
            turnover REAL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS industry_classification_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            status TEXT NOT NULL,
            taxonomy TEXT NOT NULL,
            source_provider TEXT NOT NULL,
            source_endpoint TEXT NOT NULL,
            source_updated_at TEXT,
            source_fetched_at TEXT NOT NULL,
            source_rows INTEGER NOT NULL,
            metadata_rows INTEGER NOT NULL,
            eligible_rows INTEGER NOT NULL,
            excluded_rows INTEGER NOT NULL,
            coverage REAL NOT NULL,
            summary_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_industry_metadata (
            stock TEXT PRIMARY KEY,
            stock_name TEXT NOT NULL DEFAULT '',
            board TEXT NOT NULL DEFAULT '',
            industry TEXT NOT NULL DEFAULT '',
            neutral_group TEXT NOT NULL DEFAULT '',
            industry_l1_code TEXT NOT NULL DEFAULT '',
            industry_l1_name TEXT NOT NULL DEFAULT '',
            industry_l2_code TEXT NOT NULL DEFAULT '',
            industry_l2_name TEXT NOT NULL DEFAULT '',
            industry_taxonomy TEXT NOT NULL DEFAULT '',
            industry_source TEXT NOT NULL DEFAULT '',
            industry_source_provider TEXT NOT NULL DEFAULT '',
            industry_source_endpoint TEXT NOT NULL DEFAULT '',
            industry_confidence TEXT NOT NULL DEFAULT 'none',
            industry_confidence_score REAL NOT NULL DEFAULT 0.0
                CHECK (industry_confidence_score >= 0.0 AND industry_confidence_score <= 1.0),
            industry_confidence_method TEXT NOT NULL DEFAULT '',
            industry_source_updated_at TEXT,
            industry_source_fetched_at TEXT NOT NULL,
            metadata_updated_at TEXT NOT NULL,
            classification_eligible INTEGER NOT NULL DEFAULT 0
                CHECK (classification_eligible IN (0, 1)),
            source_run_id TEXT NOT NULL,
            source_payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (source_run_id) REFERENCES industry_classification_runs(run_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_industry_exclusions (
            stock TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            source_run_id TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            FOREIGN KEY (source_run_id) REFERENCES industry_classification_runs(run_id)
        )
        """
    )
    if role == "raw":
        create_indexes(conn)
        create_views(conn)
        _ensure_meta_version(conn, "schema_version", str(SCHEMA_VERSION))
        _ensure_meta_version(conn, "database_role", role)
        return

def create_indexes(conn: sqlite3.Connection) -> None:
    index_specs = [('idx_raw_stock_date', 'stock_daily_raw', 'stock', 'date'), ('idx_norm_date_stock', 'stock_daily_normalized', 'date', 'stock'), ('idx_quality_entity', 'data_quality_issues', 'entity_type', 'entity_id'), ('idx_source_status_status', 'data_source_status', 'status', 'last_update_time'), ('idx_data_update_runs_finished', 'data_update_runs', 'finished_at', 'mode'), ('idx_industry_metadata_eligible', 'stock_industry_metadata', 'classification_eligible', 'stock'), ('idx_industry_metadata_l1', 'stock_industry_metadata', 'industry_taxonomy', 'industry_l1_code'), ('idx_industry_runs_finished', 'industry_classification_runs', 'finished_at', 'status'), ('idx_industry_exclusions_reason', 'stock_industry_exclusions', 'reason', 'stock')]
    for spec in index_specs:
        name, table, *cols = spec
        existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        if all(col in existing for col in cols):
            quoted = ", ".join(f'"{col}"' for col in cols)
            conn.execute(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}" ({quoted})')

def create_views(conn: sqlite3.Connection) -> None:
    """Expose raw daily bars and eligible industry metadata."""
    _create_view_if_columns(
        conn,
        "v_stock_industry_eligible",
        "stock_industry_metadata",
        ["stock", "industry_l1_name", "classification_eligible"],
        """
        CREATE VIEW v_stock_industry_eligible AS
        SELECT stock, stock_name, board, industry, neutral_group,
               industry_l1_code, industry_l1_name, industry_l2_code, industry_l2_name,
               industry_taxonomy, industry_source, industry_source_provider,
               industry_source_endpoint, industry_confidence, industry_confidence_score,
               industry_confidence_method, industry_source_updated_at,
               industry_source_fetched_at, metadata_updated_at, source_run_id
        FROM stock_industry_metadata
        WHERE classification_eligible = 1
        """,
    )
    _create_view_if_columns(
        conn,
        "v_latest_stock_daily",
        "stock_daily_normalized",
        ["stock", "date", "open", "high", "low", "close", "volume", "amount", "source"],
        """
        CREATE VIEW v_latest_stock_daily AS
        SELECT stock, date, open, high, low, close, volume, amount, source, source_file, has_trade_price
        FROM stock_daily_normalized
        """,
    )

def _create_view_if_columns(conn: sqlite3.Connection, view_name: str, table: str, cols: list[str], sql: str) -> None:
    if not _has_columns(conn, table, cols):
        return
    safe_sql = sql.replace("CREATE VIEW ", "CREATE VIEW IF NOT EXISTS ", 1)
    try:
        conn.execute(safe_sql)
    except sqlite3.OperationalError as exc:
        if "already exists" not in str(exc).lower():
            raise

def _has_columns(conn: sqlite3.Connection, table: str, cols: list[str]) -> bool:
    existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
    return all(col in existing for col in cols)

def _ensure_meta_version(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Write a ``warehouse_meta`` row only when its value has actually changed.

    ``init_warehouse`` runs on every connection open — including read-only paths
    such as the dashboard's ``list_runs``.  Rewriting the meta rows unconditionally
    turns those reads into writes and collides with a concurrently running
    backtest's write lock (``database is locked``).  Read-first makes the common
    case a no-op write.
    """
    row = conn.execute("SELECT value FROM warehouse_meta WHERE key = ?", (key,)).fetchone()
    if row is not None and row[0] == value:
        return
    upsert_meta(conn, key, value)

def upsert_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO warehouse_meta (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (key, value),
    )
