"""Read-only all-source quality and point-in-time identity availability audit."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_strategy.core.paths import artifact_run_dir
from my_strategy.adapters.czsc_adapter import raw_path
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_wyckoff_features import _raw_quality, get_wyckoff_profile


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _record(payload):
    record, root, calendar, data_end = payload
    path = Path(record["path"]).resolve(); source_hash = _sha(path)
    if not path.is_relative_to(Path(root).resolve()) or source_hash != record["sha256"]:
        raise ValueError("immutable source path/hash mismatch: " + record["symbol"])
    frame = pd.read_parquet(path, columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount",
                                         "source", "has_trade_price", "model_reason_codes"])
    dates = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d").tolist()
    if len(dates) != record["bars"] or len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError("source row/date contract mismatch: " + record["symbol"])
    if not frame.symbol.eq(record["symbol"]).all() or dates[-1] != record["data_end"]:
        raise ValueError("source identity/end mismatch: " + record["symbol"])
    eligibility, reasons, quality = _raw_quality(frame, frame, {"market_dates": calendar})
    counts = Counter(reason for row in reasons for reason in row)
    known = np.asarray(calendar)
    expected = known[(known >= dates[0]) & (known <= dates[-1])]
    missing = sorted(set(expected) - set(dates))
    source_counts = Counter(frame.source.fillna("").astype(str))
    transitions = int(frame.source.fillna("").ne(frame.source.fillna("").shift(1)).iloc[1:].sum())
    actions = np.flatnonzero(quality["corporate_action_suspected"])
    unit_bad = np.flatnonzero(quality["volume_amount_unit_disagreement"])
    result = {"symbol": record["symbol"], "rows": len(frame), "data_start": dates[0], "data_end": dates[-1],
              "source_sha256": source_hash, "source_hash_verified": True,
              "volume_unit": frame.attrs.get("volume_unit"), "amount_unit": frame.attrs.get("amount_unit"),
              "price_basis": frame.attrs.get("price_basis"), "source_counts": dict(source_counts),
              "source_transitions": transitions, "missing_calendar_dates": missing,
              "missing_calendar_count": len(missing), "tail_gap": dates[-1] != data_end,
              "tail_missing_dates": [day for day in calendar if dates[-1] < day <= data_end],
              "invalid_price_rows": int(quality["invalid_price"].sum()),
              "invalid_quantity_rows": int(quality["invalid_quantity"].sum()),
              "suspended_rows": int(quality["suspended"].sum()),
              "unit_disagreement_rows": len(unit_bad), "unit_disagreement_dates": [dates[i] for i in unit_bad],
              "suspected_action_rows": len(actions), "suspected_action_dates": [dates[i] for i in actions],
              "quality_eligible_rows_before_numeric_context": int(eligibility.sum()),
              "quality_reason_counts": dict(counts)}
    if _sha(path) != source_hash:
        raise ValueError("source changed while auditing: " + record["symbol"])
    return result


def audit(source_run_id, output, workers=4, raw_db=None):
    root = artifact_run_dir(source_run_id, create=False).resolve()
    source_path = root / "reports/research.json"
    source_hash = _sha(source_path); source = json.loads(source_path.read_text())
    records = source["dataset_records"]
    if source["coverage"]["failed"] or len(records) != source["coverage"]["requested"]:
        raise ValueError("source coverage incomplete")
    calendar_path = Path(source["calendar"]["path"]).resolve()
    calendar_hash = _sha(calendar_path); calendar = json.loads(calendar_path.read_text())
    if not calendar.get("verified") or stable_hash(calendar) != source["calendar"]["hash"]:
        raise ValueError("source calendar not verified/bound")
    date_list = calendar["dates"]
    if date_list != sorted(set(date_list)):
        raise ValueError("source calendar duplicate/unordered")
    results = []
    payloads = [(record, str(root), date_list, source["data_end"]) for record in records]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_record, payloads, chunksize=8):
            results.append(result)
            if len(results) % 128 == 0 or len(results) == len(records):
                print(f"source_quality {len(results)}/{len(records)}", flush=True)
    db_path = raw_path(raw_db)
    connection = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    try:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        historical_identity = [name for name in tables if any(key in name.lower() for key in ("st_history", "delist", "corporate_action", "dividend", "split_history"))]
        raw_rows, first, last = connection.execute("SELECT COUNT(*),MIN(date),MAX(date) FROM stock_daily_normalized").fetchone()
        raw_duplicates = connection.execute("SELECT COUNT(*) FROM stock_daily_normalized WHERE duplicate_source_rows>1").fetchone()[0]
        current_names = connection.execute("SELECT COUNT(*) FROM securities WHERE UPPER(name) LIKE '%ST%'").fetchone()[0]
        current_metadata = {"current_ST_named_securities": current_names,
                            "historical_identity_tables": historical_identity,
                            "historical_identity_verified": False,
                            "corporate_action_effective_date_verified": False,
                            "current_name_cannot_reconstruct_historical_ST": True,
                            "factor_columns_are_not_point_in_time_corporate_action_records": True}
    finally:
        connection.close()
    source_counts, reasons = Counter(), Counter()
    for record in results:
        source_counts.update(record["source_counts"]); reasons.update(record["quality_reason_counts"])
    totals = {key: sum(record[key] for record in results) for key in
              ("rows", "source_transitions", "missing_calendar_count", "invalid_price_rows", "invalid_quantity_rows",
               "suspended_rows", "unit_disagreement_rows", "suspected_action_rows", "quality_eligible_rows_before_numeric_context")}
    report = {"audit_version": "wyckoff_raw_quality_v1", "source_run_id": source_run_id,
              "source_report_path": str(source_path), "source_report_sha256": source_hash,
              "data_end": source["data_end"], "coverage": {"requested": len(records), "success": len(results), "failed": 0},
              "source_hashes_all_verified_and_unchanged": True, "calendar": {"path": str(calendar_path), "file_sha256": calendar_hash,
                  "contract_hash": source["calendar"]["hash"], "sessions": len(date_list)},
              "profile": get_wyckoff_profile(), "totals": totals,
              "source_counts": dict(source_counts), "quality_reason_counts": dict(reasons),
              "tail_gap_symbols": [record["symbol"] for record in results if record["tail_gap"]],
              "raw_database": {"path": str(db_path), "mode": "ro", "normalized_rows": raw_rows,
                  "data_start": first, "data_end": last, "normalized_rows_with_duplicate_source_inputs": raw_duplicates,
                  "primary_key": "stock,date; duplicate sources retained as metadata; frozen unique snapshots audited"},
              "historical_identity": current_metadata,
              "formal_qualification": {"qualified": False, "blockers": ["no_verified_corporate_action_effective_date_timeline",
                  "no_historical_ST_and_delisting_timeline", "development_history_already_observed", "independent_future_evidence_not_mature"]},
              "policy": {"prices": "unchanged_unadjusted", "units": "shares,CNY; per-row implied price checked",
                  "missing_rows": "no insertion/forward-fill; trailing windows unsafe",
                  "suspected_actions": "causal current gap mask and 120-bar context veto; not a confirmed action label",
                  "current_metadata": "scope diagnostics only; never backfilled to historical identity",
                  "shadow": "safe raw-quality rows may be scored; formal eligibility remains blocked"},
              "records": results}
    if _sha(source_path) != source_hash or _sha(calendar_path) != calendar_hash:
        raise ValueError("source report/calendar changed while auditing")
    destination = Path(output).resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(destination), "sha256": _sha(destination), "coverage": report["coverage"],
                      "totals": totals, "tail_gap_symbols": report["tail_gap_symbols"], "qualification": report["formal_qualification"]}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--raw-db")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    audit(args.source_run_id, args.output, args.workers, args.raw_db)


if __name__ == "__main__":
    main()
