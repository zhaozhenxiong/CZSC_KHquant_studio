#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bulk update one trading day's OHLCV for the local A-share universe via Baostock.

A single Baostock login is reused for all stocks, which is much faster than the
per-stock login/logout cycle used by the generic DailyUpdater.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.data_manager.downloader import DataDownloader
from my_strategy.data_manager.local_cache import LocalCache
from my_strategy.data_manager.stock_pool import StockPoolManager
from my_strategy.data_manager.validator import DataValidator
from my_strategy.storage.access_layer import get_data_access


def _normalize_code(code: str) -> str:
    raw = str(code).strip().split(".")[0].zfill(6)
    return f"{raw}.{'SH' if raw.startswith('6') else 'SZ'}"


def _bs_code(code: str) -> str:
    market = "sh" if code.endswith(".SH") else "sz"
    return f"{market}.{code.split('.')[0]}"


def _query_one(code: str, target_date: str, retries: int = 2, sleep: float = 0.1) -> pd.DataFrame | None:
    import baostock as bs

    bs_code = _bs_code(code)
    fields = "date,open,high,low,close,volume,amount"
    for attempt in range(retries + 1):
        try:
            rs = bs.query_history_k_data_plus(bs_code, fields, start_date=target_date, end_date=target_date, frequency="d", adjustflag="2")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if getattr(rs, "error_code", "0") != "0" or not rows:
                if attempt < retries:
                    time.sleep(sleep * (attempt + 1))
                    continue
                return None
            adjusted = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])

            raw_rs = bs.query_history_k_data_plus(bs_code, "date,open,high,low,close", start_date=target_date, end_date=target_date, frequency="d", adjustflag="3")
            raw_rows = []
            while raw_rs.next():
                raw_rows.append(raw_rs.get_row_data())
            if getattr(raw_rs, "error_code", "0") != "0" or not raw_rows:
                if attempt < retries:
                    time.sleep(sleep * (attempt + 1))
                    continue
                return None
            raw = pd.DataFrame(raw_rows, columns=["date", "trade_open", "trade_high", "trade_low", "trade_close"])

            merged = adjusted.merge(raw, on="date", how="left")
            merged["code"] = code
            merged = DataValidator.clean_stock_daily(merged, code)
            return merged
        except Exception:
            if attempt < retries:
                time.sleep(sleep * (attempt + 1))
            else:
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk update one trading day for the local universe.")
    parser.add_argument("--date", default="2026-08-07", help="Trading date to insert/update.")
    parser.add_argument("--offset", type=int, default=0, help="Start index in the universe.")
    parser.add_argument("--limit", type=int, default=0, help="Max stocks to process; 0 means all from offset.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    args = parser.parse_args()

    import baostock as bs

    pool = StockPoolManager().build_pool(mode="all_a")
    codes = pool["stock"].astype(str).map(_normalize_code).tolist()
    if args.offset:
        codes = codes[args.offset:]
    if args.limit:
        codes = codes[:args.limit]

    storage = get_data_access()
    login = bs.login()
    if getattr(login, "error_code", "0") != "0":
        raise RuntimeError(getattr(login, "error_msg", "baostock login failed"))

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    rows: list[dict] = []
    try:
        for idx, code in enumerate(codes, start=1):
            try:
                df = _query_one(code, args.date)
                if df is None or df.empty:
                    rows.append({"code": code, "status": "failed", "error_msg": "no data", "last_date": ""})
                    continue
                storage.save_stock_daily(code, df, source="baostock")
                rows.append({"code": code, "status": "success", "error_msg": "", "last_date": args.date})
            except Exception as exc:
                rows.append({"code": code, "status": "failed", "error_msg": str(exc)[:200], "last_date": ""})
            if idx % 100 == 0 or idx == len(codes):
                elapsed = max(time.time() - pd.Timestamp(started).timestamp(), 0.001)
                rate = idx / elapsed
                print(f"[{idx}/{len(codes)}] {code} rate={rate:.2f}/s", flush=True)
            time.sleep(0.005)
    finally:
        bs.logout()

    report = pd.DataFrame(rows)
    success = int((report["status"] == "success").sum())
    failed = int((report["status"] == "failed").sum())
    finished = datetime.now().astimezone().isoformat(timespec="seconds")
    summary = {
        "update_id": f"bulk_update_{args.date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "started_at": started,
        "finished_at": finished,
        "target_date": args.date,
        "total": len(report),
        "success": success,
        "failed": failed,
    }
    if args.json:
        import json
        print(json.dumps({"summary": summary, "rows": report.to_dict("records")}, ensure_ascii=False, indent=2, default=str))
    else:
        print(report.to_string(index=False))
        print("\nSummary:", summary)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
