"""Tests for the data quality checker."""

from __future__ import annotations

import os
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from my_strategy.validation.data_quality import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    DataQualityChecker,
    DataQualityIssue,
    DataQualityReport,
    validate_ohlcv,
)


class FakeStorage:
    """Minimal in-memory storage for unit tests."""

    def __init__(self, daily: dict[str, pd.DataFrame] | None = None, basic: pd.DataFrame | None = None):
        self._daily = daily if daily is not None else {}
        self._basic = basic if basic is not None else pd.DataFrame()
        self.settings = type("Settings", (), {"write_db": False, "raw_db": Path("/dev/null")})()

    def read_stock_daily(self, code: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        df = self._daily.get(code, pd.DataFrame()).copy()
        if df.empty or "date" not in df.columns:
            return df
        df["date"] = pd.to_datetime(df["date"])
        mask = pd.Series(True, index=df.index)
        if start_date:
            mask &= df["date"] >= pd.to_datetime(start_date)
        if end_date:
            mask &= df["date"] <= pd.to_datetime(end_date)
        return df.loc[mask].reset_index(drop=True)

    def read_stock_basic(self) -> pd.DataFrame:
        return self._basic.copy()


def _good_bars(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10.0 + i * 0.01 for i in range(len(dates))],
            "high": [10.1 + i * 0.01 for i in range(len(dates))],
            "low": [9.9 + i * 0.01 for i in range(len(dates))],
            "close": [10.0 + i * 0.01 for i in range(len(dates))],
            "volume": [1000 + i for i in range(len(dates))],
        }
    )


def test_validate_ohlcv_ok():
    df = _good_bars("2024-01-01", 5)
    result = validate_ohlcv(df)
    assert result["ok"] is True
    assert result["issues"] == []


def test_validate_ohlcv_detects_bad_ohlc():
    df = _good_bars("2024-01-01", 3)
    df.loc[1, "high"] = df.loc[1, "low"] - 0.5
    result = validate_ohlcv(df)
    assert result["ok"] is False
    assert any("high" in issue for issue in result["issues"])


def test_validate_ohlcv_detects_negative_volume():
    df = _good_bars("2024-01-01", 3)
    df.loc[0, "volume"] = -100
    result = validate_ohlcv(df)
    assert result["ok"] is False
    assert "negative_volume" in result["issues"]


def test_checker_reports_missing_data_for_empty_stock():
    storage = FakeStorage()
    checker = DataQualityChecker(config={}, storage=storage)
    report = checker.check(["000001.SZ"], "2024-01-01", "2024-01-10")
    missing_issues = [i for i in report.issues if i.category == "missing_data"]
    assert len(missing_issues) == 1
    assert missing_issues[0].severity == SEVERITY_ERROR


def test_checker_detects_low_volume_warning():
    df = _good_bars("2024-01-01", 5)
    df.loc[2, "volume"] = 10
    storage = FakeStorage(daily={"000001.SZ": df})
    checker = DataQualityChecker(config={}, storage=storage, min_volume_warn=100)
    report = checker.check(["000001.SZ"], "2024-01-01", "2024-01-10")
    warnings = [i for i in report.issues if i.category == "volume_invalid" and i.severity == SEVERITY_WARNING]
    assert len(warnings) == 1


def test_checker_detects_missing_trading_day():
    # Two stocks share the same calendar except one misses a date.
    df_a = _good_bars("2024-01-01", 5)
    df_b = df_a.drop(df_a.index[2]).reset_index(drop=True)
    storage = FakeStorage(daily={"000001.SZ": df_a, "000002.SZ": df_b})
    checker = DataQualityChecker(config={}, storage=storage)
    report = checker.check(["000001.SZ", "000002.SZ"], "2024-01-01", "2024-01-10")
    missing = [i for i in report.issues if i.category == "missing_trading_day"]
    assert len(missing) == 1
    assert missing[0].entity_id == "000002.SZ"


def test_checker_detects_adjustment_gap():
    df = _good_bars("2024-01-01", 5)
    # Insert a 30% overnight gap.
    df.loc[3, "close"] = df.loc[2, "close"] * 1.35
    df.loc[3, ["open", "high", "low"]] = df.loc[3, "close"]
    storage = FakeStorage(daily={"000001.SZ": df})
    checker = DataQualityChecker(config={}, storage=storage, max_adjust_gap_pct=0.15)
    report = checker.check(["000001.SZ"], "2024-01-01", "2024-01-10")
    gaps = [i for i in report.issues if i.category == "adjustment_gap"]
    assert len(gaps) == 1
    assert gaps[0].severity == SEVERITY_WARNING


def test_checker_flags_st_and_delisted():
    df = _good_bars("2024-01-01", 5)
    basic = pd.DataFrame({"code": ["000001.SZ"], "name": ["*ST Test"]})
    storage = FakeStorage(daily={"000001.SZ": df}, basic=basic)
    checker = DataQualityChecker(config={}, storage=storage)
    report = checker.check(["000001.SZ"], "2024-01-01", "2024-01-31")
    st_issues = [i for i in report.issues if i.category == "st_flag"]
    delisted = [i for i in report.issues if i.category == "delisted"]
    assert len(st_issues) == 1
    assert st_issues[0].severity == SEVERITY_INFO
    assert len(delisted) == 1


def test_checker_save_report_produces_artifact():
    df = _good_bars("2024-01-01", 5)
    storage = FakeStorage(daily={"000001.SZ": df})
    checker = DataQualityChecker(config={}, storage=storage)
    report = checker.check(["000001.SZ"], "2024-01-01", "2024-01-10")
    result = checker.save_report(report)
    assert result["run_id"] == report.run_id
    assert result["artifact_id"].startswith("data_quality:csv:")
    assert Path(result["artifact_path"]).is_file()
    assert json.loads(Path(result["report_path"]).read_text())["run_id"] == report.run_id


def test_save_report_persists_actual_warnings_and_errors_in_raw_warehouse(tmp_path, monkeypatch):
    from my_strategy.storage.access_layer import DataAccessLayer

    monkeypatch.setenv("KHQUANT_STORAGE_MODE", "db")
    storage = DataAccessLayer(raw_db=tmp_path / "raw.db")
    bars = _good_bars("2024-01-01", 5)
    bars.loc[2, "volume"] = 10
    storage.save_stock_daily("000001.SZ", bars, source="fixture")
    checker = DataQualityChecker(config={}, storage=storage)
    report = checker.check(["000001.SZ", "000002.SZ"], "2024-01-01", "2024-01-05", run_id="quality-persistence")
    assert {SEVERITY_WARNING, SEVERITY_ERROR} <= {issue.severity for issue in report.issues}
    first = checker.save_report(report)
    second = checker.save_report(report)
    with storage.raw.read_connection() as connection:
        rows = connection.execute("SELECT severity,entity_id,source_file,details_json FROM data_quality_issues").fetchall()
        role = connection.execute("SELECT value FROM warehouse_meta WHERE key='database_role'").fetchone()[0]
    assert role == "raw"
    assert len(rows) == len(report.issues) == first["inserted_issues"] == second["inserted_issues"]
    assert {row["severity"] for row in rows} >= {"warning", "error"}
    assert {row["source_file"] for row in rows} == {"quality-persistence"}
    assert all(isinstance(json.loads(row["details_json"]), dict) for row in rows)
    csv = pd.read_csv(first["artifact_path"])
    saved = json.loads(Path(first["report_path"]).read_text())
    assert len(csv) == len(saved["issues"]) == len(report.issues)
    assert saved["summary"] == report.summary and saved["stock_stats"] == report.stock_stats
    assert first["raw_db"] == str(tmp_path / "raw.db") and "processed_db" not in first


def test_save_report_rolls_back_all_issues_on_invalid_record(tmp_path, monkeypatch):
    from my_strategy.storage.access_layer import DataAccessLayer

    monkeypatch.setenv("KHQUANT_STORAGE_MODE", "db")
    storage = DataAccessLayer(raw_db=tmp_path / "raw.db")
    report = DataQualityReport("quality-rollback", "2026-10-08T12:00:00", "2026-10-08", "2026-10-08", [], [
        DataQualityIssue("volume_invalid", "warning", "stock", "000001.SZ", "valid first record"),
        DataQualityIssue(None, "error", "stock", "000002.SZ", "invalid category"),
    ], {})
    with pytest.raises(sqlite3.IntegrityError):
        DataQualityChecker(config={}, storage=storage).save_report(report)
    with storage.raw.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM data_quality_issues").fetchone()[0] == 0


def test_clean_report_in_file_mode_keeps_readable_artifacts_without_creating_database(tmp_path, monkeypatch):
    from my_strategy.storage.access_layer import DataAccessLayer

    monkeypatch.setenv("KHQUANT_STORAGE_MODE", "file")
    database = tmp_path / "unused-raw.db"
    storage = DataAccessLayer(raw_db=database)
    report = DataQualityReport("quality-clean-file", "2026-10-08T12:00:00", "2026-10-08", "2026-10-08", [], [], {}, {"total_issues": 0})
    result = DataQualityChecker(config={}, storage=storage).save_report(report)
    assert not database.exists() and result["inserted_issues"] == 0
    assert pd.read_csv(result["artifact_path"]).empty
    assert json.loads(Path(result["report_path"]).read_text())["issues"] == []


def test_checker_summary_counts_are_consistent():
    df = _good_bars("2024-01-01", 5)
    df.loc[0, "volume"] = 0
    storage = FakeStorage(daily={"000001.SZ": df, "000002.SZ": pd.DataFrame()})
    checker = DataQualityChecker(config={}, storage=storage, min_volume_warn=50)
    report = checker.check(["000001.SZ", "000002.SZ"], "2024-01-01", "2024-01-10")
    summary = report.summary
    assert summary["stocks_checked"] == 2
    assert summary["stocks_with_data"] == 1
    assert summary["stocks_missing_data"] == 1
    assert summary["total_issues"] == len(report.issues)
    assert summary["total_issues"] > 0
