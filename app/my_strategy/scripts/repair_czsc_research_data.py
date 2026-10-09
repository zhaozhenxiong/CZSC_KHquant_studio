"""Retrieve true Tushare historical daily bars and preserve revision evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
env_file = PROJECT_ROOT.parent / ".env"
if env_file.is_file():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
from my_strategy.runtime_env import normalize_native_runtime_paths
normalize_native_runtime_paths(PROJECT_ROOT.parent)

import numpy as np
import pandas as pd

from my_strategy.core.run_context import create_run_context
from my_strategy.data_manager.market_data_provider import RateLimiter, TushareProvider
from my_strategy.services.czsc_research_data import audit_data
from my_strategy.storage.access_layer import get_data_access
from my_strategy.storage.raw_repository import RawMarketRepository

DAILY_DOCUMENTATION = "https://tushare.pro/document/1?doc_id=27"
CALENDAR_DOCUMENTATION = "https://tushare.pro/document/1?doc_id=26"


def normalize_tushare_daily(frame: pd.DataFrame, expected_date: str) -> pd.DataFrame:
    """The daily API is unadjusted; convert lots to shares and thousands to CNY."""
    required = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg"}
    if not required <= set(frame.columns):
        raise ValueError("Tushare missing fields: " + ",".join(sorted(required - set(frame.columns))))
    out = frame.rename(columns={"ts_code": "code", "trade_date": "date", "vol": "volume"}).copy()
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d", errors="raise")
    if not out["date"].dt.strftime("%Y-%m-%d").eq(expected_date).all():
        raise ValueError("Unexpected historical response dates")
    if out.duplicated(["code", "date"]).any():
        raise ValueError("Duplicate historical response keys")
    for col in ("open", "high", "low", "close", "volume", "amount", "pct_chg"):
        out[col] = pd.to_numeric(out[col], errors="raise")
    out["volume"] *= 100
    out["amount"] *= 1000
    numeric = out[["open", "high", "low", "close", "volume", "amount"]]
    if not np.isfinite(numeric.to_numpy()).all() or (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Nonfinite or nonpositive historical OHLC")
    if ((out["volume"] < 0) | (out["amount"] < 0)).any():
        raise ValueError("Negative volume or amount")
    if ((out["volume"] == 0) ^ (out["amount"] == 0)).any():
        raise ValueError("Inconsistent volume/amount")
    if ((out.high < out[["open", "close", "low"]].max(axis=1))
            | (out.low > out[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC range")
    for col in ("open", "high", "low", "close"):
        out["trade_" + col] = out[col]
    out["turnover"] = 0.0
    out["source"] = "tushare:daily"
    return out.sort_values("code").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", required=True)
    parser.add_argument("--repair-start", default="2026-04-01")
    parser.add_argument("--dates", default="", help="Explicit comma-separated historical dates")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db-path")
    parser.add_argument("--calendar-only", action="store_true")
    parser.add_argument("--skip-calendar", action="store_true", help="Repair explicit historical dates without calling calendar")
    args = parser.parse_args()
    repo = RawMarketRepository(args.db_path)
    symbols = repo.list_stocks()
    config = {"end": args.end, "repair_start": args.repair_start, "provider": "tushare:daily",
              "volume_unit": "shares", "amount_unit": "CNY", "price_basis": "unadjusted",
              "universe_basis": "frozen local securities", "documentation": DAILY_DOCUMENTATION}
    context = create_run_context(task="czsc-research-data-update", as_of_date=args.end,
                                 run_id=args.run_id, config=config, stocks=symbols,
                                 start_date=args.repair_start, end_date=args.end, scope="daily_update")
    root = context.run_dir()
    snapshots = context.subdir("data_snapshots")
    reports = context.subdir("reports")
    limiter = RateLimiter(calls=45, per_seconds=60)
    api = TushareProvider()._pro_api()
    statuses: list[dict] = []
    artifacts: list[dict] = []

    def save_frame(frame: pd.DataFrame, filename: str) -> Path:
        path = snapshots / filename
        frame.to_parquet(path, index=False)
        artifacts.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(frame)})
        return path

    def query(name: str, **params):
        for attempt in range(3):
            limiter.acquire()
            try:
                return api.query(name, **params)
            except Exception as exc:
                statuses.append({"operation": name, "parameters": params, "attempt": attempt + 1,
                                 "status": "failed", "error": str(exc)})
                if attempt == 2:
                    raise
                # Calendar permissions can be only one request per minute,
                # independent of the daily endpoint's separate 45/minute cap.
                if name == "trade_cal" and "频率" in str(exc):
                    print(json.dumps({"stage": "calendar_rate_limit_wait", "seconds": 65}), flush=True)
                    time.sleep(35)
                    time.sleep(30)
                else:
                    time.sleep(2 * (attempt + 1))

    try:
        if args.skip_calendar:
            if not args.dates or args.calendar_only:
                raise ValueError("skip-calendar requires explicit dates and cannot be calendar-only")
            market_dates = []
            cal_info = {"verified": False, "dates": [], "source": "not_requested",
                        "reason": "Calendar endpoint rate limited; explicit daily retrieval dates checked separately"}
        else:
            calendar = query("trade_cal", exchange="SSE", start_date="20100101", end_date="20261008")
            calendar = calendar.sort_values("cal_date").reset_index(drop=True)
            if calendar.empty or calendar["cal_date"].duplicated().any():
                raise ValueError("Empty or duplicate trading calendar")
            expected = pd.date_range("2010-01-01", "2026-10-08").strftime("%Y%m%d").tolist()
            if calendar["cal_date"].astype(str).tolist() != expected:
                raise ValueError("Incomplete trading calendar")
            calendar_path = save_frame(calendar, "tushare_trade_cal_sse.parquet")
            market_dates = pd.to_datetime(calendar.loc[calendar.is_open.astype(int) == 1, "cal_date"]).dt.strftime("%Y-%m-%d").tolist()
            cal_info = {"verified": True, "source": "tushare:trade_cal:SSE", "documentation": CALENDAR_DOCUMENTATION,
                        "start": "2010-01-01", "end": "2026-10-08", "dates": market_dates,
                        "artifact": str(calendar_path), "sha256": artifacts[-1]["sha256"]}
        (reports / "calendar.json").write_text(json.dumps(cal_info, indent=2), encoding="utf-8")
        print(json.dumps({"stage": "calendar", "run_id": context.run_id, "path": str(reports / "calendar.json"),
                          "trading_dates": len(market_dates), "verified": cal_info["verified"]}), flush=True)
        if args.calendar_only:
            context.write_metadata({"status": "completed", "calendar": cal_info, "artifacts": artifacts})
            return 0
        before_audit = audit_data(args.end, db_path=repo.db_path)
        (reports / "audit_before.json").write_text(json.dumps(before_audit, ensure_ascii=False, indent=2), encoding="utf-8")
        with repo.read_connection() as connection:
            suspicious = [r[0] for r in connection.execute(
                "SELECT DISTINCT date FROM stock_daily_normalized WHERE date>=? AND date<=? AND "
                "(source='fuyao' OR has_trade_price<>1 OR volume IS NULL OR amount IS NULL OR "
                "(volume>0 AND amount=0) OR (amount>0 AND volume=0))", (args.repair_start, args.end))]
            latest = connection.execute("SELECT MAX(date) FROM stock_daily_normalized WHERE date<=?", (args.end,)).fetchone()[0]
        dates = sorted(set(args.dates.split(",")) - {""}) if args.dates else sorted(
            set(suspicious) | {d for d in market_dates if (latest or args.repair_start) < d <= args.end})
        if any(d > args.end or (not args.skip_calendar and d not in market_dates) for d in dates):
            raise ValueError("Requested repair day is not a verified trading date before end")
        (reports / "frozen_universe.json").write_text(json.dumps(symbols), encoding="utf-8")
        total_written = 0
        for index, day in enumerate(dates, 1):
            started = time.monotonic()
            status = {"date": day, "index": index, "total_dates": len(dates)}
            try:
                incoming = query("daily", trade_date=day.replace("-", ""), limit=6000)
                if len(incoming) >= 6000:
                    # A full cap is not evidence of completeness: paginate until exhausted.
                    frames = [incoming]
                    offset = 6000
                    while True:
                        part = query("daily", trade_date=day.replace("-", ""), limit=6000, offset=offset)
                        if part.empty:
                            break
                        frames.append(part)
                        if len(part) < 6000:
                            break
                        offset += 6000
                    incoming = pd.concat(frames, ignore_index=True)
                if incoming.empty:
                    raise ValueError("Empty daily historical response")
                save_frame(incoming, day + "_upstream.parquet")
                frame = normalize_tushare_daily(incoming, day)
                frame = frame[frame.code.isin(symbols)].copy()
                if frame.empty:
                    raise ValueError("Historical response does not match frozen universe")
                with repo.read_connection() as connection:
                    before = pd.read_sql_query("SELECT * FROM stock_daily_normalized WHERE date=?", connection, params=[day])
                    raw_before = pd.read_sql_query("SELECT * FROM stock_daily_raw WHERE date=?", connection, params=[day])
                save_frame(before, day + "_normalized_before.parquet")
                save_frame(raw_before, day + "_raw_before.parquet")
                # Durable before snapshots are written before opening the mutation transaction.
                written = repo.upsert_stocks_daily({code: group for code, group in frame.groupby("code")}, source="tushare:daily")
                with repo.read_connection() as connection:
                    after = pd.read_sql_query("SELECT * FROM stock_daily_normalized WHERE date=?", connection, params=[day])
                save_frame(after, day + "_normalized_after.parquet")
                diff_columns = ["open", "high", "low", "close", "volume", "amount", "source", "has_trade_price"]
                diff = before[["stock", "date", *diff_columns]].merge(
                    after[["stock", "date", *diff_columns]], on=["stock", "date"], how="outer", suffixes=("_before", "_after"), indicator=True)
                save_frame(diff, day + "_revision.parquet")
                missing = sorted(set(symbols) - set(after.stock))
                total_written += written
                status.update({"status": "success", "upstream_rows": len(incoming), "written": written,
                               "coverage": len(after), "frozen_universe_missing": missing,
                               "unmatched_upstream": len(incoming) - len(frame)})
            except Exception as exc:
                status.update({"status": "failed", "error": str(exc)})
            status["seconds"] = round(time.monotonic() - started, 3)
            statuses.append(status)
            (reports / "progress.json").write_text(json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8")
            (reports / "revision_artifacts.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(status, ensure_ascii=False), flush=True)
        audit = audit_data(args.end, db_path=repo.db_path)
        (reports / "audit_after.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        target_report = pd.DataFrame([{"code": row["symbol"], "last_date": row["actual_end"],
                                      "status": "success" if row["target_covered"] else "failed",
                                      "error_msg": "" if row["target_covered"] else "provider returned no target-date bar"}
                                     for row in audit["records"]])
        db_update_run = get_data_access(repo.db_path).record_data_update(
            update_id=context.run_id, started_at=context.created_at, finished_at=None,
            mode="czsc_research_historical_repair", stocks_requested="frozen_local_universe",
            end_date=args.end, report=target_report)
        summary = {"run_id": context.run_id, "target_date": args.end, "total_written": total_written,
                   "counts": audit["counts"], "data_version": audit["data_version"], "dates": dates,
                   "failures": [s for s in statuses if s["status"] == "failed" and "date" in s],
                   "source_attempts": [s for s in statuses if "operation" in s],
                   "calendar": cal_info, "artifacts": artifacts, "config": config, "db_update_run": db_update_run}
        (reports / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        context.write_metadata({**summary, "status": "partial" if summary["failures"] else "completed"})
        print(json.dumps({"stage": "complete", "run_id": context.run_id, "counts": audit["counts"],
                          "total_written": total_written, "failures": len(summary["failures"]), "report": str(reports / "summary.json")}), flush=True)
        return 1 if summary["failures"] else 0
    except Exception as exc:
        context.write_metadata({"status": "failed", "error": str(exc), "operations": statuses, "artifacts": artifacts})
        print(json.dumps({"stage": "failed", "run_id": context.run_id, "error": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
