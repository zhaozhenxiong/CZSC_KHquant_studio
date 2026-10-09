#!/usr/bin/env python
"""Bulk-refresh the approved stock-metadata universe with CSRC L1 industries.

The command is a dry-run by default.  ``--execute`` writes an atomic metadata
replacement, a source snapshot, an exclusion audit, and a RunContext report.
It never deletes or edits raw OHLCV data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from my_strategy.core.paths import (
    ARTIFACT_RUNS_ROOT,
    RAW_DATA_ROOT,
    STOCK_METADATA_CSV_PATH,
)
from my_strategy.core.run_context import create_run_context
from my_strategy.data_manager.config import load_data_config
from my_strategy.data_manager.industry_enricher import (
    apply_csrc_industry_snapshot,
    build_industry_exclusions,
    fetch_baostock_industry_snapshot,
    industry_quality_settings,
)
from my_strategy.storage.access_layer import DataAccessLayer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or execute a unified CSRC level-one industry backfill.",
    )
    parser.add_argument("--metadata-path", default=str(STOCK_METADATA_CSV_PATH))
    parser.add_argument("--raw-db", default=str(RAW_DATA_ROOT / "khquant_raw.db"))
    parser.add_argument("--source-csv", default="", help="Replay a previously saved Baostock source snapshot.")
    parser.add_argument("--min-coverage", type=float, default=None)
    parser.add_argument("--max-coverage-drop", type=float, default=None)
    parser.add_argument("--max-source-conflicts", type=int, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--execute", action="store_true", help="Atomically replace metadata after validation.")
    return parser.parse_args(argv)


def _truthy(series: pd.Series) -> pd.Series:
    return series.fillna(False).map(
        lambda value: value is True or str(value).strip().lower() in {"1", "true", "yes"}
    )


def _read_raw_universe(raw_db: Path) -> set[str]:
    if not raw_db.exists():
        return set()
    with sqlite3.connect(raw_db) as conn:
        rows = conn.execute("select distinct stock from securities").fetchall()
    return {str(row[0]).strip() for row in rows if str(row[0]).strip()}


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".tmp",
            prefix=f"{path.name}.",
            dir=path.parent,
            delete=False,
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            temp_path = Path(handle.name)
            frame.to_csv(handle, index=False)
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temp_path, path)


def _load_source(path: str) -> pd.DataFrame:
    if path:
        source_path = Path(path)
        frame = pd.read_csv(source_path, dtype={"stock": str})
        if "industry_source_fetched_at" not in frame.columns:
            fetched_at = datetime.fromtimestamp(source_path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
            frame["industry_source_fetched_at"] = fetched_at
        return frame
    return fetch_baostock_industry_snapshot()


def _read_sqlite_metadata_if_present(raw_db: Path) -> pd.DataFrame:
    """Read without schema initialization so dry-run remains side-effect free."""
    if not raw_db.exists():
        return pd.DataFrame()
    with sqlite3.connect(raw_db) as conn:
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_industry_metadata'"
        ).fetchone()
        if present is None:
            return pd.DataFrame()
        return pd.read_sql_query("SELECT * FROM stock_industry_metadata ORDER BY stock", conn)


def _provenance_quality(frame: pd.DataFrame) -> dict[str, int]:
    def filled(column: str) -> pd.Series:
        return frame.get(column, pd.Series("", index=frame.index)).fillna("").astype(str).str.strip().ne("")

    scores = pd.to_numeric(
        frame.get("industry_confidence_score", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    timezone_pattern = r"(?:Z|[+-]\d{2}:\d{2})$"
    fetched = frame.get("industry_source_fetched_at", pd.Series("", index=frame.index)).fillna("").astype(str)
    metadata_time = frame.get("metadata_updated_at", pd.Series("", index=frame.index)).fillna("").astype(str)
    return {
        "rows": len(frame),
        "source_identity_complete": int(
            (filled("industry_source") & filled("industry_source_provider") & filled("industry_source_endpoint")).sum()
        ),
        "confidence_complete": int(
            (filled("industry_confidence") & filled("industry_confidence_method") & scores.between(0.0, 1.0)).sum()
        ),
        "source_effective_date_filled": int(filled("industry_source_updated_at").sum()),
        "source_fetched_at_timezone_aware": int(fetched.str.contains(timezone_pattern, regex=True).sum()),
        "metadata_updated_at_timezone_aware": int(metadata_time.str.contains(timezone_pattern, regex=True).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configured = industry_quality_settings(load_data_config())
    min_coverage = float(
        configured["min_coverage"] if args.min_coverage is None else args.min_coverage
    )
    max_coverage_drop = float(
        configured["max_coverage_drop"]
        if args.max_coverage_drop is None
        else args.max_coverage_drop
    )
    max_source_conflicts = int(
        configured["max_source_conflicts"]
        if args.max_source_conflicts is None
        else args.max_source_conflicts
    )
    if not 0 < min_coverage <= 1:
        raise ValueError("--min-coverage must be in (0, 1]")
    if not 0 <= max_coverage_drop <= 1:
        raise ValueError("--max-coverage-drop must be in [0, 1]")
    if max_source_conflicts < 0:
        raise ValueError("--max-source-conflicts must be >= 0")

    metadata_path = Path(args.metadata_path).resolve()
    raw_db = Path(args.raw_db).resolve()
    sqlite_existing = _read_sqlite_metadata_if_present(raw_db)
    if sqlite_existing.empty and not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    existing = (
        sqlite_existing
        if not sqlite_existing.empty
        else pd.read_csv(metadata_path, dtype={"stock": str})
    )
    metadata_read_source = "raw_sqlite:stock_industry_metadata" if not sqlite_existing.empty else "compatibility_csv"
    source = _load_source(args.source_csv)
    refreshed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    updated, classification = apply_csrc_industry_snapshot(
        existing,
        source,
        refreshed_at=refreshed_at,
        min_coverage=min_coverage,
        max_coverage_drop=max_coverage_drop,
        max_source_conflicts=max_source_conflicts,
    )
    raw_universe = _read_raw_universe(raw_db)
    exclusions = build_industry_exclusions(raw_universe, updated)
    reason_counts = exclusions["reason"].value_counts().to_dict() if not exclusions.empty else {}
    eligible_count = int(_truthy(updated["classification_eligible"]).sum())
    raw_excluded_count = int(exclusions["stock"].isin(raw_universe).sum())

    report: dict[str, Any] = {
        "run_id": args.run_id,
        "mode": "execute" if args.execute else "dry_run",
        "created_at": refreshed_at,
        "metadata_path": str(metadata_path),
        "metadata_read_source": metadata_read_source,
        "raw_db": str(raw_db),
        "raw_universe_stocks": len(raw_universe),
        "metadata_rows_before": len(existing),
        "metadata_rows_after": len(updated),
        "eligible_industry_stocks": eligible_count,
        "excluded_from_industry_count": len(exclusions),
        "raw_universe_excluded_count": raw_excluded_count,
        "metadata_excluded_count": int(len(exclusions) - reason_counts.get("missing_metadata", 0)),
        "exclusion_reason_counts": reason_counts,
        "exclusion_sample": exclusions.head(20).to_dict("records"),
        "classification": classification,
        "provenance_quality": _provenance_quality(updated),
        "raw_market_data_modified": False,
        "raw_ohlcv_modified": False,
        "raw_database_metadata_modified": bool(args.execute),
        "artifact_runs_root": str(ARTIFACT_RUNS_ROOT),
    }

    if args.execute:
        context = create_run_context(
            task="industry-classification-refresh",
            as_of_date=refreshed_at[:10],
            data_version=classification.get("source_updated_at", ""),
            run_id=args.run_id or None,
            scope="daily_update",
            source="cli",
        )
        report["run_id"] = context.run_id
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        snapshot_dir = metadata_path.parent / "source_snapshots"
        source_path = snapshot_dir / f"baostock_csrc_industry_{stamp}.csv"
        backup_path = snapshot_dir / f"stock_metadata_before_csrc_{stamp}.csv"
        exclusions_path = metadata_path.parent / "industry_exclusions.csv"
        report_path = metadata_path.parent / "industry_classification_report.json"

        snapshot_dir.mkdir(parents=True, exist_ok=True)
        if metadata_path.exists():
            shutil.copy2(metadata_path, backup_path)
        _atomic_write_csv(source, source_path)
        storage = DataAccessLayer(raw_db=raw_db)
        sqlite_result = storage.replace_industry_metadata(
            updated,
            exclusions,
            run_id=context.run_id,
            summary=report,
        )
        authoritative = storage.read_industry_metadata()
        authoritative_exclusions = storage.read_industry_exclusions()
        if len(authoritative) != len(updated) or len(authoritative_exclusions) != len(exclusions):
            raise RuntimeError(
                "SQLite/compatibility parity check failed before export: "
                f"metadata={len(authoritative)}/{len(updated)} "
                f"exclusions={len(authoritative_exclusions)}/{len(exclusions)}"
            )
        _atomic_write_csv(authoritative, metadata_path)
        _atomic_write_csv(authoritative_exclusions[["stock", "reason"]], exclusions_path)
        report.update(
            {
                "authority": "raw_sqlite:stock_industry_metadata",
                "sqlite": sqlite_result,
                "source_snapshot": str(source_path),
                "metadata_backup": str(backup_path) if backup_path.exists() else "",
                "exclusions_path": str(exclusions_path),
                "report_path": str(report_path),
            }
        )
        _atomic_write_json(report, report_path)
        artifact_report = context.subdir("reports") / "industry_classification_report.json"
        _atomic_write_json(report, artifact_report)
        context.write_metadata({"report_path": str(artifact_report), "summary": report})

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
