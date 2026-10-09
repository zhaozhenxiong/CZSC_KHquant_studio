#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initialize a KHQuant data directory.

Creates the canonical ``/khquant-data/{raw,processed,metadata}`` tree, sets up the
raw market-data SQLite schema, and optionally
seeds ``metadata/stock_metadata.csv`` from ``IndustryEnricher`` when the file is
empty or missing.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.storage.warehouse_schema import init_warehouse  # noqa: E402


DEFAULT_DATA_ROOT = Path("/khquant-data")
DEFAULT_SCHEMA = "khquant-data-dir-v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a KHQuant data directory.")
    parser.add_argument(
        "data_root",
        nargs="?",
        default=os.environ.get("KHQUANT_DATA_ROOT", str(DEFAULT_DATA_ROOT)),
        help="Target data directory (default: /khquant-data or KHQUANT_DATA_ROOT).",
    )
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, help="Data directory schema version label.")
    parser.add_argument(
        "--seed-metadata",
        action="store_true",
        help="Seed metadata/stock_metadata.csv from IndustryEnricher if it is empty or missing.",
    )
    parser.add_argument("--force", action="store_true", help="Re-seed metadata even if it already exists.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary.")
    return parser.parse_args(argv)


def _mkdirs(data_root: Path) -> dict[str, Path]:
    dirs = {
        "raw": data_root / "raw",
        "processed": data_root / "processed",
        "metadata": data_root / "metadata",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    (data_root / "processed" / "czsc").mkdir(parents=True, exist_ok=True)
    (data_root / "logs").mkdir(parents=True, exist_ok=True)
    return dirs


def _init_raw_database(data_root: Path) -> Path:
    db_path = data_root / "raw" / "khquant_raw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        init_warehouse(conn, role="raw")
        conn.commit()
    finally:
        conn.close()
    return db_path


def _discover_seed_stocks(data_root: Path) -> list[str] | None:
    parquet = data_root / "raw" / "stock_basic" / "all_a_stock_list.parquet"
    if not parquet.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_parquet(parquet, columns=["stock"])
        stocks = df["stock"].dropna().astype(str).unique().tolist()
        return stocks if stocks else []
    except Exception:
        return []


def _seed_stock_metadata(data_root: Path, force: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"seeded": False, "path": "", "rows": 0, "source": "industry_enricher"}
    try:
        from my_strategy.data_manager.industry_enricher import IndustryEnricher
    except Exception as exc:  # pragma: no cover - dependency issues
        result["error"] = f"could not import IndustryEnricher: {exc}"
        return result

    metadata_path = data_root / "metadata" / "stock_metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    result["path"] = str(metadata_path)

    if metadata_path.exists() and not force:
        try:
            import pandas as pd

            existing = pd.read_csv(metadata_path)
            if not existing.empty:
                result["rows"] = len(existing)
                result["skipped"] = "metadata already exists and is not empty"
                return result
        except Exception as exc:
            result["warning"] = f"could not read existing metadata: {exc}"

    stocks = _discover_seed_stocks(data_root)
    if stocks is None:
        result["skipped"] = "no raw/stock_basic/all_a_stock_list.parquet; nothing to enrich"
        return result
    if not stocks:
        result["skipped"] = "all_a_stock_list.parquet is empty; nothing to enrich"
        return result

    try:
        enricher = IndustryEnricher(metadata_path=metadata_path)
        df = enricher.refresh_metadata(stocks=stocks, force=force)
        result["seeded"] = True
        result["rows"] = len(df)
    except Exception as exc:
        result["error"] = f"IndustryEnricher refresh failed: {exc}"
    return result


def _write_manifest(data_root: Path, schema: str) -> Path:
    manifest = data_root / "khquant-data.json"
    payload = {
        "schema": schema,
        "project": "KHQuant",
        "raw_root": "raw",
        "processed_root": "processed",
        "metadata_root": "metadata",
        "raw_db": "raw/khquant_raw.db",
        "results_db": "processed/czsc/results.db",
        "tasks_db": "metadata/czsc_tasks.db",
        "artifact_root": "artifacts",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            existing.update({key: value for key, value in payload.items() if key not in existing})
            payload = existing
        except json.JSONDecodeError:
            pass
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_root = Path(args.data_root).resolve()

    dirs = _mkdirs(data_root)
    raw_db = _init_raw_database(data_root)
    manifest_path = _write_manifest(data_root, args.schema)

    seed_result: dict[str, Any] = {"seeded": False}
    if args.seed_metadata:
        seed_result = _seed_stock_metadata(data_root, args.force)

    summary = {
        "data_root": str(data_root),
        "schema": args.schema,
        "directories": {name: str(path) for name, path in dirs.items()},
        "raw_db": str(raw_db),
        "manifest": str(manifest_path),
        "metadata_seed": seed_result,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Initialized KHQuant data directory: {data_root}")
        print(f"Raw DB: {raw_db}")
        print(f"Manifest: {manifest_path}")
        if seed_result.get("seeded"):
            print(f"Seeded stock metadata: {seed_result['path']} ({seed_result['rows']} rows)")
        elif seed_result.get("skipped"):
            print(f"Metadata seed skipped: {seed_result['skipped']}")
        elif seed_result.get("error"):
            print(f"Metadata seed error: {seed_result['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
