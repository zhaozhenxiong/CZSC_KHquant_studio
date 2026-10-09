"""Raw ingestion remains independent of retired model and result storage."""
from __future__ import annotations

import sqlite3
from contextlib import closing

import pandas as pd
import pytest

from my_strategy.storage.access_layer import DataAccessLayer
from my_strategy.storage.warehouse_schema import init_warehouse


def test_ingestion_opens_only_explicit_raw_database(tmp_path) -> None:
    raw = tmp_path / "raw.sqlite"
    access = DataAccessLayer(raw)
    assert not raw.exists()
    assert not hasattr(access, "processed")
    assert not hasattr(access, "artifacts")
    assert access.settings.raw_db == raw

    bars = pd.DataFrame({
        "date": ["2026-09-24"], "open": [10.0], "high": [11.0],
        "low": [9.0], "close": [10.5], "volume": [100.0], "amount": [1050.0],
    })
    access.save_stock_daily("000001.SZ", bars, source="unit")
    assert access.read_stock_daily("000001.SZ")["close"].tolist() == [10.5]
    with closing(sqlite3.connect(raw)) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"stock_daily_raw", "stock_daily_normalized", "data_update_runs"} <= tables
    assert not tables.intersection({
        "chip_state_checkpoints", "factor_snapshots", "model_feature_snapshots",
        "model_feature_labels", "artifact_blobs", "runs", "backtest_trades",
    })
    assert list(tmp_path.iterdir()) == [raw]


def test_raw_schema_rejects_processed_database_before_mutation(tmp_path) -> None:
    with closing(sqlite3.connect(tmp_path / "legacy.sqlite")) as connection:
        connection.execute("CREATE TABLE warehouse_meta(key TEXT PRIMARY KEY,value TEXT,updated_at TEXT)")
        connection.execute("INSERT INTO warehouse_meta VALUES ('database_role','processed','frozen')")
        connection.commit()
        with pytest.raises(ValueError, match="role mismatch"):
            init_warehouse(connection, role="raw")
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='stock_daily_normalized'").fetchone() is None
