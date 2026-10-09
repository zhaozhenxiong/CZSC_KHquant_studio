"""Read-only, source-aware eligibility audit for CZSC research inputs."""
from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

from my_strategy.adapters.czsc_adapter import raw_path


# These names refer to actual unadjusted retrieval contracts, not renamed rows.
VERIFIED_SOURCES = frozenset({"tushare", "tushare:daily", "fuyao_unadjusted_v1", "baostock"})


def _main_board(symbol: str) -> bool:
    return ((symbol.endswith(".SH") and symbol.startswith(("600", "601", "603", "605")))
            or (symbol.endswith(".SZ") and symbol.startswith(("000", "001", "002", "003"))))


def audit_data(end: str, min_history: int = 120, db_path: str | Path | None = None) -> dict[str, Any]:
    """Audit every local symbol as of ``end`` without initializing/writing SQLite.

    ``market_dates`` are observed raw dates, not a verified trading calendar.
    Missing dates and zero activity remain visible; a zero/zero row alone does
    not establish official suspension status. Eligibility concerns the latest
    ``min_history`` stored bars. Earlier training rows need their own gates.
    """
    target = date.fromisoformat(str(end)).isoformat()
    if min_history < 1:
        raise ValueError("min_history must be positive")
    path = raw_path(db_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("BEGIN")  # A single read snapshot across all symbols.
    records: list[dict[str, Any]] = []
    observed_dates: set[str] = set()
    identity = hashlib.sha256()
    try:
        columns = {r[1] for r in connection.execute("PRAGMA table_info(stock_daily_normalized)")}
        required = {"stock", "date", "open", "high", "low", "close", "volume", "amount"}
        if not required <= columns:
            raise ValueError("Missing raw fields: " + ",".join(sorted(required - columns)))
        extras = [c for c in ("source", "has_trade_price") if c in columns]
        selected = "date,open,high,low,close,volume,amount" + ("," + ",".join(extras) if extras else "")
        symbols = [r[0] for r in connection.execute(
            "SELECT stock FROM securities UNION SELECT DISTINCT stock FROM stock_daily_normalized ORDER BY stock")]
        # One indexed query per stock avoids ranking all historical rows.
        for symbol in symbols:
            rows = connection.execute(
                f"SELECT {selected} FROM stock_daily_normalized WHERE stock=? AND date<=? ORDER BY date DESC LIMIT ?",
                (symbol, target, int(min_history))).fetchall()
            issues: Counter[str] = Counter()
            source_issues: Counter[str] = Counter()
            source_counts: Counter[str] = Counter()
            zero_activity = 0
            invalid_target = False
            for row in rows:
                item = dict(row)
                observed_dates.add(str(row["date"]))
                identity.update(json.dumps([symbol, item], sort_keys=True, separators=(",", ":")).encode())
                source = str(item.get("source") or "unknown")
                source_counts[source] += 1
                if source not in VERIFIED_SOURCES:
                    source_issues[source] += 1
                    issues["unverified_source"] += 1
                if item.get("has_trade_price") != 1:
                    issues["unverified_trade_price"] += 1
                values = [item[c] for c in ("open", "high", "low", "close", "volume", "amount")]
                finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
                row_invalid = not finite
                if not finite:
                    issues["missing_or_nonfinite_ohlcv"] += 1
                else:
                    op, hi, lo, cl, volume, amount = map(float, values)
                    if min(op, hi, lo, cl) <= 0 or hi < max(op, lo, cl) or lo > min(op, hi, cl):
                        issues["invalid_ohlc"] += 1
                        row_invalid = True
                    if volume < 0 or amount < 0:
                        issues["negative_volume_amount"] += 1
                        row_invalid = True
                    elif volume == 0 and amount == 0:
                        zero_activity += 1
                    elif volume == 0 or amount == 0:
                        issues["inconsistent_volume_amount"] += 1
                        row_invalid = True
                if row["date"] == target:
                    invalid_target = row_invalid or item.get("has_trade_price") != 1 or source not in VERIFIED_SOURCES
            covered = bool(rows and rows[0]["date"] == target)
            history = len(rows) >= min_history
            if not covered:
                issues["target_date_missing"] += 1
            if not history:
                issues["insufficient_history"] += 1
            eligible = not issues
            target_active = bool(covered and not invalid_target and rows[0]["volume"] > 0 and rows[0]["amount"] > 0)
            record = {"symbol": symbol, "target_covered": covered, "history_available": history,
                      "history_rows": len(rows), "first_window_date": rows[-1]["date"] if rows else None,
                      "actual_end": rows[0]["date"] if rows else None, "source_counts": dict(source_counts),
                      "source_issues": dict(source_issues), "issues": dict(issues),
                      "zero_activity_count": zero_activity, "target_active": target_active,
                      "window_dates": sorted(str(row["date"]) for row in rows),
                      "main_board": _main_board(symbol), "eligible": eligible,
                      "tradable": eligible and target_active and _main_board(symbol)}
            records.append(record)
    finally:
        connection.close()
    market_dates = sorted(observed_dates)
    for record in records:
        first = record["first_window_date"]
        present = set(record["window_dates"])
        missing = [day for day in market_dates if first and first <= day <= target and day not in present]
        record["missing_observed_dates"] = missing
        record["missing_observed_dates_count"] = len(missing)
        # Missing raw days may be suspensions or gaps; neither is fabricated.
        # Labeling/execution must use a separately verified market calendar.
    counts = {"symbols": len(records), "target_covered": sum(r["target_covered"] for r in records),
              "history_available": sum(r["history_available"] for r in records),
              "eligible": sum(r["eligible"] for r in records), "tradable": sum(r["tradable"] for r in records),
              "zero_activity_symbols": sum(r["zero_activity_count"] > 0 for r in records)}
    return {"end": target, "min_history": int(min_history), "raw_db": str(path), "records": records,
            "symbols": [r["symbol"] for r in records], "counts": counts,
            "eligible_symbols": [r["symbol"] for r in records if r["eligible"]],
            "tradable_symbols": [r["symbol"] for r in records if r["tradable"]],
            "market_dates": market_dates, "market_dates_basis": "observed_raw_union_not_verified_calendar",
            "data_version": identity.hexdigest(), "verified_sources": sorted(VERIFIED_SOURCES),
            "price_basis": "unadjusted; corporate actions and historical ST remain unverified",
            "zero_activity_basis": "observed volume=amount=0; official suspension not established"}
