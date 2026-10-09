"""Data quality checks for OHLCV data.

This module contains low-level OHLCV validation and a higher-level
``DataQualityChecker`` that produces a structured ``DataQualityReport`` for a
universe of stocks over a date range.  Reports are persisted into the processed
SQLite warehouse.

No future information is used: all checks operate on the supplied historical bars
and, when available, a point-in-time basic stock table.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from my_strategy.data_manager.config import load_data_config
from my_strategy.data_manager.stock_pool import normalize_stock_code
from my_strategy.storage.access_layer import get_data_access

logger = logging.getLogger(__name__)

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

ISSUE_CATEGORIES = {
    "missing_trading_day",
    "ohlc_invalid",
    "volume_invalid",
    "adjustment_gap",
    "suspension",
    "delisted",
    "st_flag",
    "duplicate_date",
    "missing_data",
    "future_data",
}


@dataclass
class DataQualityIssue:
    category: str
    severity: str
    entity_type: str
    entity_id: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "message": self.message,
            "details_json": json.dumps(self.details, ensure_ascii=False, sort_keys=True, default=str),
        }


@dataclass
class DataQualityReport:
    run_id: str
    checked_at: str
    start_date: str
    end_date: str
    stocks: list[str]
    issues: list[DataQualityIssue]
    stock_stats: dict[str, dict[str, Any]]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_issues_dataframe(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(
                columns=["category", "severity", "entity_type", "entity_id", "message", "details_json"]
            )
        return pd.DataFrame([issue.to_record() for issue in self.issues])

    def has_errors(self) -> bool:
        return any(issue.severity == SEVERITY_ERROR for issue in self.issues)


def validate_ohlcv(df: pd.DataFrame, *, date_col: str = "date") -> dict[str, Any]:
    issues: list[str] = []
    if df.empty:
        issues.append("empty_dataframe")
        return {"ok": False, "issues": issues, "rows": 0}
    if date_col not in df.columns:
        issues.append(f"missing_{date_col}")
    else:
        duplicated = int(pd.to_datetime(df[date_col]).duplicated().sum())
        if duplicated:
            issues.append(f"duplicate_dates:{duplicated}")
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            issues.append(f"missing_{col}")
    if {"open", "high", "low", "close"}.issubset(df.columns):
        numeric = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            issues.append("ohlc_nan")
        if (numeric <= 0).any().any():
            issues.append("ohlc_non_positive")
        if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
            issues.append("high_below_ohlc")
        if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
            issues.append("low_above_ohlc")
    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce")
        if (volume < 0).any():
            issues.append("negative_volume")
    return {"ok": not issues, "issues": issues, "rows": len(df)}


class DataQualityChecker:
    """Check market data quality for a universe and persist the report.

    The checker is intentionally conservative.  Missing-trading-day detection
    uses the union of all trading dates observed across the supplied universe,
    which avoids a hard dependency on an external calendar file while still
    catching obvious calendar gaps.
    """

    def __init__(
        self,
        config: dict | None = None,
        storage: Any | None = None,
        *,
        min_volume_warn: int = 100,
        max_adjust_gap_pct: float = 0.15,
        min_rows_for_trading_calendar: int = 3,
    ):
        self.config = config or load_data_config()
        self.storage = storage or get_data_access()
        self.min_volume_warn = int(min_volume_warn)
        self.max_adjust_gap_pct = float(max_adjust_gap_pct)
        self.min_rows_for_trading_calendar = int(min_rows_for_trading_calendar)

    def check(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        *,
        run_id: str = "",
        basic_df: pd.DataFrame | None = None,
    ) -> DataQualityReport:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if start > end:
            raise ValueError(f"start_date {start_date} is after end_date {end_date}")
        if basic_df is None:
            basic_df = self._load_basic(stocks)
        stocks = [normalize_stock_code(stock) for stock in stocks]
        issues: list[DataQualityIssue] = []
        stock_stats: dict[str, dict[str, Any]] = {}

        all_bars: dict[str, pd.DataFrame] = {}
        for code in stocks:
            df = self._load_bars(code, start_date, end_date)
            all_bars[code] = df
            stats = self._per_stock_stats(df, start, end)
            stock_stats[code] = stats
            issues.extend(self._check_ohlc(code, df))
            issues.extend(self._check_volume(code, df))
            issues.extend(self._check_duplicates(code, df))
            issues.extend(self._check_lifecycle(code, df, basic_df, start, end))

        calendar_dates = self._build_trading_calendar(all_bars)
        for code, df in all_bars.items():
            issues.extend(self._check_missing_trading_days(code, df, calendar_dates, start, end))
            issues.extend(self._check_adjustment_gaps(code, df))

        summary = self._summarize(issues, stocks, stock_stats)
        report = DataQualityReport(
            run_id=run_id or self._make_run_id(),
            checked_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            stocks=stocks,
            issues=issues,
            stock_stats=stock_stats,
            summary=summary,
        )
        return report

    def save_report(self, report: DataQualityReport) -> dict[str, Any]:
        """Persist the report into the processed SQLite warehouse."""
        issues_df = report.to_issues_dataframe()
        if issues_df.empty:
            logger.info("No quality issues to persist for run_id=%s", report.run_id)
        inserted_issues = 0
        if self.storage.settings.write_db:
            with self.storage.processed.connect() as conn:
                for _, row in issues_df.iterrows():
                    payload = {
                        "issue_id": f"{report.run_id}:{row['entity_id']}:{row['category']}:{_stable_hash(row['message'])}",
                        "imported_at": report.checked_at,
                        "category": row["category"],
                        "severity": row["severity"],
                        "entity_type": row["entity_type"],
                        "entity_id": row["entity_id"],
                        "source_file": report.run_id,
                        "message": row["message"],
                        "details_json": row["details_json"],
                    }
                    conn.execute(
                        """
                        INSERT INTO data_quality_issues (
                            issue_id, imported_at, category, severity, entity_type, entity_id, source_file, message, details_json
                        ) VALUES (
                            :issue_id, :imported_at, :category, :severity, :entity_type, :entity_id, :source_file, :message, :details_json
                        )
                        ON CONFLICT(issue_id) DO UPDATE SET
                            imported_at=excluded.imported_at,
                            severity=excluded.severity,
                            message=excluded.message,
                            details_json=excluded.details_json
                        """,
                        payload,
                    )
                    inserted_issues += 1
        artifact_id = None
        if issues_df.empty:
            issues_bytes = b""
        else:
            issues_bytes = issues_df.to_csv(index=False).encode("utf-8")
        artifact_id = self.storage.put_artifact_bytes(
            namespace="data_quality",
            kind="csv",
            name=report.run_id,
            payload=issues_bytes,
            metadata={
                "run_id": report.run_id,
                "checked_at": report.checked_at,
                "start_date": report.start_date,
                "end_date": report.end_date,
                "stocks": len(report.stocks),
                "issues": len(report.issues),
                "summary": json.dumps(report.summary, ensure_ascii=False, sort_keys=True, default=str),
            },
        )
        return {
            "run_id": report.run_id,
            "inserted_issues": inserted_issues,
            "artifact_id": artifact_id,
            "processed_db": str(self.storage.settings.processed_db),
        }

    def _load_basic(self, stocks: list[str]) -> pd.DataFrame:
        try:
            df = self.storage.read_stock_basic()
            if not df.empty:
                return df
        except Exception as exc:
            logger.debug("Could not load stock basic table: %s", exc)
        return pd.DataFrame()

    def _load_bars(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            df = self.storage.read_stock_daily(code, start_date=start_date, end_date=end_date)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as exc:
            logger.warning("Could not load bars for %s: %s", code, exc)
            return pd.DataFrame()

    def _per_stock_stats(self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
        if df.empty or "date" not in df.columns:
            return {
                "rows": 0,
                "first_date": "",
                "last_date": "",
                "expected_days": int((end - start).days + 1),
                "coverage": 0.0,
            }
        dates = pd.to_datetime(df["date"])
        return {
            "rows": len(df),
            "first_date": dates.min().strftime("%Y-%m-%d"),
            "last_date": dates.max().strftime("%Y-%m-%d"),
            "expected_days": int((end - start).days + 1),
            "coverage": round(len(df) / max((end - start).days + 1, 1), 4),
        }

    def _check_ohlc(self, code: str, df: pd.DataFrame) -> list[DataQualityIssue]:
        if df.empty:
            return []
        result = validate_ohlcv(df)
        if result["ok"]:
            return []
        errors = []
        for issue in result["issues"]:
            if issue.startswith("duplicate_dates"):
                errors.append(
                    DataQualityIssue(
                        category="duplicate_date",
                        severity=SEVERITY_ERROR,
                        entity_type="stock",
                        entity_id=code,
                        message=f"Duplicate trading dates: {issue}",
                        details={"issue": issue, "rows": result["rows"]},
                    )
                )
            elif issue in ("ohlc_nan", "ohlc_non_positive", "high_below_ohlc", "low_above_ohlc"):
                errors.append(
                    DataQualityIssue(
                        category="ohlc_invalid",
                        severity=SEVERITY_ERROR,
                        entity_type="stock",
                        entity_id=code,
                        message=f"OHLC validation failure: {issue}",
                        details={"issue": issue, "rows": result["rows"]},
                    )
                )
            elif issue == "negative_volume":
                errors.append(
                    DataQualityIssue(
                        category="volume_invalid",
                        severity=SEVERITY_ERROR,
                        entity_type="stock",
                        entity_id=code,
                        message="Negative volume detected",
                        details={"rows": result["rows"]},
                    )
                )
            elif issue.startswith("missing_"):
                errors.append(
                    DataQualityIssue(
                        category="missing_data",
                        severity=SEVERITY_ERROR,
                        entity_type="stock",
                        entity_id=code,
                        message=f"Missing required column: {issue}",
                        details={"issue": issue, "rows": result["rows"]},
                    )
                )
        return errors

    def _check_volume(self, code: str, df: pd.DataFrame) -> list[DataQualityIssue]:
        if df.empty or "volume" not in df.columns:
            return []
        volume = pd.to_numeric(df["volume"], errors="coerce")
        low_volume = volume[(volume >= 0) & (volume < self.min_volume_warn)].dropna()
        if low_volume.empty:
            return []
        return [
            DataQualityIssue(
                category="volume_invalid",
                severity=SEVERITY_WARNING,
                entity_type="stock",
                entity_id=code,
                message=f"Found {len(low_volume)} rows with volume below {self.min_volume_warn}",
                details={"threshold": self.min_volume_warn, "rows": len(low_volume)},
            )
        ]

    def _check_duplicates(self, code: str, df: pd.DataFrame) -> list[DataQualityIssue]:
        if df.empty or "date" not in df.columns:
            return []
        dupes = int(pd.to_datetime(df["date"]).duplicated().sum())
        if dupes == 0:
            return []
        return [
            DataQualityIssue(
                category="duplicate_date",
                severity=SEVERITY_ERROR,
                entity_type="stock",
                entity_id=code,
                message=f"Duplicate daily dates: {dupes}",
                details={"duplicate_count": dupes},
            )
        ]

    def _check_lifecycle(
        self,
        code: str,
        df: pd.DataFrame,
        basic_df: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        if not basic_df.empty and "code" in basic_df.columns:
            normalized_code = normalize_stock_code(code)
            row = basic_df[basic_df["code"] == normalized_code]
            if not row.empty:
                name = str(row.iloc[0].get("name", ""))
                if "ST" in name or "*ST" in name:
                    issues.append(
                        DataQualityIssue(
                            category="st_flag",
                            severity=SEVERITY_INFO,
                            entity_type="stock",
                            entity_id=code,
                            message=f"Stock name indicates ST/*ST: {name}",
                            details={"name": name},
                        )
                    )
        if df.empty:
            issues.append(
                DataQualityIssue(
                    category="missing_data",
                    severity=SEVERITY_ERROR,
                    entity_type="stock",
                    entity_id=code,
                    message=f"No data found for {code} between {start.strftime('%Y-%m-%d')} and {end.strftime('%Y-%m-%d')}",
                    details={"start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d")},
                )
            )
            return issues
        if "date" in df.columns:
            last_date = pd.to_datetime(df["date"]).max()
            if last_date < end:
                gap_days = int((end - last_date).days)
                issues.append(
                    DataQualityIssue(
                        category="delisted",
                        severity=SEVERITY_WARNING if gap_days > 5 else SEVERITY_INFO,
                        entity_type="stock",
                        entity_id=code,
                        message=f"Last observed date {last_date.strftime('%Y-%m-%d')} precedes check end {end.strftime('%Y-%m-%d')} by {gap_days} days",
                        details={"last_date": last_date.strftime("%Y-%m-%d"), "gap_days": gap_days},
                    )
                )
        return issues

    def _build_trading_calendar(self, all_bars: dict[str, pd.DataFrame]) -> set[pd.Timestamp]:
        dates: set[pd.Timestamp] = set()
        for df in all_bars.values():
            if df.empty or "date" not in df.columns:
                continue
            if len(df) < self.min_rows_for_trading_calendar:
                continue
            dates.update(pd.to_datetime(df["date"]).dt.normalize().unique())
        return dates

    def _check_missing_trading_days(
        self,
        code: str,
        df: pd.DataFrame,
        calendar_dates: set[pd.Timestamp],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[DataQualityIssue]:
        if df.empty or "date" not in df.columns or len(calendar_dates) < 2:
            return []
        stock_dates = set(pd.to_datetime(df["date"]).dt.normalize().unique())
        expected = {d for d in calendar_dates if start <= d <= end}
        missing = sorted(expected - stock_dates)
        if not missing:
            return []
        return [
            DataQualityIssue(
                category="missing_trading_day",
                severity=SEVERITY_WARNING,
                entity_type="stock",
                entity_id=code,
                message=f"Missing {len(missing)} trading days compared to the inferred market calendar",
                details={
                    "missing_count": len(missing),
                    "missing_sample": [d.strftime("%Y-%m-%d") for d in missing[:10]],
                },
            )
        ]

    def _check_adjustment_gaps(self, code: str, df: pd.DataFrame) -> list[DataQualityIssue]:
        if df.empty or len(df) < 2 or "close" not in df.columns:
            return []
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if close.empty or len(close) < 2:
            return []
        prev_close = close.shift(1)
        denom = prev_close.replace(0, pd.NA)
        pct_change = ((close - prev_close) / denom).abs()
        big_gaps = pct_change[pct_change > self.max_adjust_gap_pct].dropna()
        if big_gaps.empty:
            return []
        gap_dates = []
        for idx in big_gaps.index:
            if idx < len(df):
                date_val = df["date"].iloc[idx]
                gap_dates.append(str(date_val)[:10] if date_val is not None else str(idx))
        return [
            DataQualityIssue(
                category="adjustment_gap",
                severity=SEVERITY_WARNING,
                entity_type="stock",
                entity_id=code,
                message=f"Detected {len(big_gaps)} daily close gaps exceeding {self.max_adjust_gap_pct:.1%}",
                details={"gap_count": len(big_gaps), "gap_dates": gap_dates[:10]},
            )
        ]

    def _summarize(self, issues: list[DataQualityIssue], stocks: list[str], stock_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for issue in issues:
            by_category[issue.category] = by_category.get(issue.category, 0) + 1
            by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1
        missing = [code for code in stocks if stock_stats.get(code, {}).get("rows", 0) == 0]
        return {
            "stocks_checked": len(stocks),
            "stocks_with_data": len([c for c in stocks if stock_stats.get(c, {}).get("rows", 0) > 0]),
            "stocks_missing_data": len(missing),
            "total_issues": len(issues),
            "by_category": by_category,
            "by_severity": by_severity,
            "missing_data_stocks": missing,
        }

    def _make_run_id(self) -> str:
        return f"dq_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _stable_hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


# Strict execution gate.  This is intentionally separate from the diagnostic
# checker above: it reads the normalized raw SQLite table in one pass and never
# mutates it or replaces missing values.
import argparse
from math import isfinite
from pathlib import Path
import sqlite3
from typing import Iterable

from my_strategy.core.run_context import stable_hash

CORE_PRICE_FIELDS = ("open", "high", "low", "close")
CORE_ACTIVITY_FIELDS = ("volume", "amount")
CORE_FIELDS = CORE_PRICE_FIELDS + CORE_ACTIVITY_FIELDS


@dataclass(frozen=True)
class DataQualityGate:
    min_date_coverage: float = 0.95
    max_core_missing_ratio: float = 0.05
    max_consecutive_missing_days: int = 5

    def validate(self) -> None:
        if not 0 < self.min_date_coverage <= 1:
            raise ValueError("quality min date coverage must be in (0, 1]")
        if not 0 <= self.max_core_missing_ratio < 1:
            raise ValueError("quality max core missing ratio must be in [0, 1)")
        if self.max_consecutive_missing_days < 0:
            raise ValueError("quality max consecutive missing days must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_date_coverage": self.min_date_coverage,
            "max_core_missing_ratio": self.max_core_missing_ratio,
            "max_consecutive_missing_days": self.max_consecutive_missing_days,
            "core_fields": list(CORE_FIELDS),
        }


@dataclass(frozen=True)
class DataQualityAssessment:
    eligible_stocks: list[str]
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def add_data_quality_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("raw data quality gate")
    group.add_argument("--quality-min-date-coverage", type=float, default=0.95)
    group.add_argument("--quality-max-core-missing-ratio", type=float, default=0.05)
    group.add_argument("--quality-max-consecutive-missing-days", type=int, default=5)


def quality_gate_from_args(args: argparse.Namespace) -> DataQualityGate:
    gate = DataQualityGate(
        min_date_coverage=float(args.quality_min_date_coverage),
        max_core_missing_ratio=float(args.quality_max_core_missing_ratio),
        max_consecutive_missing_days=int(args.quality_max_consecutive_missing_days),
    )
    gate.validate()
    return gate


def assess_stock_data_quality(
    stocks: Iterable[str],
    start: str,
    end: str,
    *,
    gate: DataQualityGate,
    raw_db: str | Path | None = None,
) -> DataQualityAssessment:
    """Return a per-stock, read-only eligibility decision for the requested period."""
    gate.validate()
    requested = list(dict.fromkeys(str(stock).strip() for stock in stocks if str(stock).strip()))
    db_path = Path(raw_db) if raw_db is not None else get_data_access().settings.raw_db
    if not db_path.exists():
        raise FileNotFoundError(f"raw data quality gate cannot find SQLite database: {db_path}")
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(stock_daily_normalized)")}
        missing_columns = {"stock", "date", *CORE_FIELDS} - columns
        if missing_columns:
            raise RuntimeError(f"raw data quality gate requires stock_daily_normalized columns: {sorted(missing_columns)}")
        calendar = [str(row[0]) for row in conn.execute(
            "SELECT DISTINCT date FROM stock_daily_normalized WHERE date >= ? AND date <= ? AND date IS NOT NULL AND TRIM(date) <> '' ORDER BY date",
            (start, end),
        )]
        stats = {stock: {"rows": 0, "observed_dates": 0, "invalid_core_rows": 0, "last_index": -1, "max_gap": 0} for stock in requested}
        date_index = {value: index for index, value in enumerate(calendar)}
        for chunk in _stock_chunks(requested):
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT stock, date, open, high, low, close, volume, amount "
                f"FROM stock_daily_normalized WHERE stock IN ({placeholders}) AND date >= ? AND date <= ? ORDER BY stock, date"
            )
            for stock, date_value, *values in conn.execute(query, [*chunk, start, end]):
                stat = stats[str(stock)]
                stat["rows"] += 1
                index = date_index.get(str(date_value))
                if index is None:
                    continue
                stat["observed_dates"] += 1
                stat["max_gap"] = max(stat["max_gap"], index - stat["last_index"] - 1)
                stat["last_index"] = index
                if _invalid_core(values):
                    stat["invalid_core_rows"] += 1
    rows: list[dict[str, Any]] = []
    eligible: list[str] = []
    reason_counts: dict[str, int] = {}
    for stock in requested:
        stat = stats[stock]
        if stat["last_index"] >= 0:
            stat["max_gap"] = max(stat["max_gap"], len(calendar) - stat["last_index"] - 1)
        coverage = stat["observed_dates"] / len(calendar) if calendar else 0.0
        missing_ratio = stat["invalid_core_rows"] / stat["rows"] if stat["rows"] else 1.0
        reasons = []
        if not stat["rows"]:
            reasons.append("no_raw_rows")
        if coverage < gate.min_date_coverage:
            reasons.append("low_date_coverage")
        if missing_ratio > gate.max_core_missing_ratio:
            reasons.append("high_core_missingness")
        if stat["max_gap"] > gate.max_consecutive_missing_days:
            reasons.append("long_consecutive_gap")
        accepted = not reasons
        row = {
            "stock": stock, "accepted": accepted, "reasons": ";".join(reasons),
            "rows": stat["rows"], "market_sessions": len(calendar), "observed_dates": stat["observed_dates"],
            "date_coverage": coverage, "invalid_core_rows": stat["invalid_core_rows"],
            "core_missing_ratio": missing_ratio, "max_consecutive_missing_days": stat["max_gap"],
        }
        rows.append(row)
        if accepted:
            eligible.append(stock)
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        "status": "passed" if eligible else "failed", "raw_db": str(db_path), "start": str(start), "end": str(end),
        "market_sessions": len(calendar), "requested_stocks": len(requested), "eligible_stocks": len(eligible),
        "excluded_stocks": len(requested) - len(eligible), "excluded_by_reason": reason_counts,
        "gate": gate.to_dict(), "eligible_stocks_sha256": stable_hash(eligible),
    }
    return DataQualityAssessment(eligible, rows, summary)


def write_data_quality_artifacts(run_context: Any, assessment: DataQualityAssessment) -> dict[str, str]:
    validation_dir = run_context.subdir("validation")
    json_path, csv_path = validation_dir / "data_quality.json", validation_dir / "data_quality.csv"
    _write_json_atomic(json_path, {"summary": assessment.summary, "stocks": assessment.rows})
    pd.DataFrame(assessment.rows).to_csv(csv_path, index=False, encoding="utf-8")
    metadata_path = run_context.run_dir() / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else run_context.to_dict()
    metadata["data_quality"] = {**assessment.summary, "report_json": str(json_path), "report_csv": str(csv_path)}
    _write_json_atomic(metadata_path, metadata)
    return {"json": str(json_path), "csv": str(csv_path)}


def _stock_chunks(values: list[str], size: int = 800) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _invalid_core(values: list[Any]) -> bool:
    for field, value in zip(CORE_FIELDS, values, strict=True):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return True
        if not isfinite(number) or (field in CORE_PRICE_FIELDS and number <= 0) or (field in CORE_ACTIVITY_FIELDS and number < 0):
            return True
    return False


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{__import__('os').getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
