#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batched baostock retry for a handful of failed stocks."""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.data_manager.validator import DataValidator
from my_strategy.storage.db_paths import raw_db_path
from my_strategy.storage.raw_repository import RawMarketRepository


def _merge_adjusted_with_trade_price(adjusted: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    adjusted = adjusted.copy()
    raw = raw.copy()
    adjusted["date"] = pd.to_datetime(adjusted["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = adjusted.merge(raw[["date", "trade_open", "trade_high", "trade_low", "trade_close"]], on="date", how="left")
    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "trade_open", "trade_high", "trade_low", "trade_close"]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def fetch_baostock(bs, code: str, start: str, end: str) -> pd.DataFrame:
    market = "sh" if code.endswith(".SH") else "sz"
    bs_code = f"{market}.{code.split('.')[0]}"
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close,volume,amount",
        start_date=start, end_date=end, frequency="d", adjustflag="2",
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if getattr(rs, "error_code", "0") != "0":
        raise RuntimeError(getattr(rs, "error_msg", "baostock query failed"))
    adjusted = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])

    raw_rs = bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close",
        start_date=start, end_date=end, frequency="d", adjustflag="3",
    )
    raw_rows = []
    while raw_rs.next():
        raw_rows.append(raw_rs.get_row_data())
    if getattr(raw_rs, "error_code", "0") != "0":
        raise RuntimeError(getattr(raw_rs, "error_msg", "baostock raw query failed"))
    raw = pd.DataFrame(raw_rows, columns=["date", "trade_open", "trade_high", "trade_low", "trade_close"])
    return _merge_adjusted_with_trade_price(adjusted, raw)


def main() -> int:
    import baostock as bs

    raw = Path(raw_db_path())
    conn = sqlite3.connect(raw)
    rows = conn.execute(
        "SELECT code, error_msg FROM data_source_status WHERE status='failed' ORDER BY code"
    ).fetchall()
    conn.close()

    targets = [r[0] for r in rows]
    if not targets:
        print("No failed stocks to retry.")
        return 0

    print(f"Retrying {len(targets)} failed stocks through 2026-07-31 (baostock batch)...")
    lg = bs.login()
    if getattr(lg, "error_code", "0") != "0":
        print(f"baostock login failed: {getattr(lg, 'error_msg', '')}")
        return 1

    repo = RawMarketRepository(str(raw))
    success = failed = 0
    t0 = time.time()
    try:
        for i, code in enumerate(targets, 1):
            try:
                df = fetch_baostock(bs, code, "2020-01-01", "2026-07-31")
                clean = DataValidator.clean_stock_daily(df, code)
                valid = DataValidator.validate_stock_daily(clean, min_rows=100, require_min_rows=True)
                if not valid.ok:
                    raise RuntimeError(valid.error)
                repo.upsert_stock_daily(code, clean)
                success += 1
            except Exception as exc:
                failed += 1
                print(f"  still failed {code}: {exc}")
            if i % 50 == 0 or i == len(targets):
                elapsed = time.time() - t0
                print(f"  {i}/{len(targets)} done, success={success}, failed={failed}, elapsed={elapsed:.1f}s")
    finally:
        bs.logout()

    print(f"Finished: success={success}, still_failed={failed}, total={len(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
