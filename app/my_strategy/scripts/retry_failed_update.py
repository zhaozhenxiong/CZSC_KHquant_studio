#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Retry failed full-market daily updates using baostock (avoid unreliable akshare)."""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.data_manager.config import load_data_config
from my_strategy.data_manager.updater import DailyUpdater
from my_strategy.storage.db_paths import raw_db_path


def main() -> int:
    config = load_data_config()
    # Force baostock-only for this recovery run; akshare is returning connection resets.
    config["data_sources"] = {"primary": "baostock", "fallback": []}
    config.setdefault("update", {})
    config["update"]["sleep_seconds"] = 0.05

    updater = DailyUpdater(config)
    raw = Path(raw_db_path())
    conn = sqlite3.connect(raw)
    rows = conn.execute(
        "SELECT code, error_msg FROM data_source_status WHERE status='failed' ORDER BY code"
    ).fetchall()
    conn.close()

    targets = [r[0] for r in rows]
    print(f"Retrying {len(targets)} failed stocks through 2026-07-31 (baostock-only)...")

    success = failed = 0
    t0 = time.time()
    for i, code in enumerate(targets, 1):
        row = updater.update_one(code, end_date="2026-07-31")
        if row["status"] == "success":
            success += 1
        else:
            failed += 1
            print(f"  still failed {code}: {row['error_msg']}")
        if i % 50 == 0 or i == len(targets):
            elapsed = time.time() - t0
            print(f"  {i}/{len(targets)} done, success={success}, failed={failed}, elapsed={elapsed:.1f}s")

    print(f"Finished: success={success}, still_failed={failed}, total={len(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
