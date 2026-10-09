from __future__ import annotations

import sqlite3

from my_strategy.validation.data_quality import DataQualityGate, assess_stock_data_quality


def _insert(conn: sqlite3.Connection, stock: str, date: str, *, close: float | None = 10.0) -> None:
    conn.execute(
        "INSERT INTO stock_daily_normalized VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (stock, date, 10.0, 11.0, 9.0, close, 1000.0, 10000.0),
    )


def test_quality_gate_excludes_sparse_and_invalid_stocks(tmp_path) -> None:
    raw_db = tmp_path / "raw.db"
    with sqlite3.connect(raw_db) as conn:
        conn.execute(
            """
            CREATE TABLE stock_daily_normalized (
                stock TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, amount REAL
            )
            """
        )
        dates = [f"2026-01-{day:02d}" for day in range(1, 11)]
        for date in dates:
            _insert(conn, "GOOD.SZ", date)
        for date in dates[:3] + dates[-2:]:
            _insert(conn, "SPARSE.SZ", date)
        for date in dates:
            _insert(conn, "INVALID.SZ", date, close=None)

    assessment = assess_stock_data_quality(
        ["GOOD.SZ", "SPARSE.SZ", "INVALID.SZ", "MISSING.SZ"],
        "2026-01-01",
        "2026-01-10",
        raw_db=raw_db,
        gate=DataQualityGate(
            min_date_coverage=0.9,
            max_core_missing_ratio=0.05,
            max_consecutive_missing_days=2,
        ),
    )

    assert assessment.eligible_stocks == ["GOOD.SZ"]
    rows = {row["stock"]: row for row in assessment.rows}
    assert "low_date_coverage" in rows["SPARSE.SZ"]["reasons"]
    assert "long_consecutive_gap" in rows["SPARSE.SZ"]["reasons"]
    assert rows["INVALID.SZ"]["core_missing_ratio"] == 1.0
    assert "high_core_missingness" in rows["INVALID.SZ"]["reasons"]
    assert "no_raw_rows" in rows["MISSING.SZ"]["reasons"]
