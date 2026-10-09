"""Restore full-market daily bars via tushare batch-of-2 queries with resume.

Why batch-of-2: tushare's ``daily`` query returns at most 6000 rows per call, so
a 300-code batch silently truncates to 20 rows/stock.  Two codes per call keep
each stock under the 6000-row cap while returning the full 2015->today history,
halving the number of upstream calls vs per-stock (2772 vs 5544).

Why strict throttling: tushare enforces 50 calls/minute on the ``daily``
interface.  Exceeding it triggers an error that also flips the provider's
process-wide global-failed flag, cascading every later call to akshare.
``_Throttle(60/45)`` spaces call starts >=1.33s apart, bounding the sustained
rate at 45/min deterministically.

Resilience:
- Each stock is persisted immediately (``LocalCache.save_stock_daily``), so a
  hang or crash keeps all completed work; on restart, codes with >=1000 bars in
  the raw DB are skipped (the DB is the checkpoint).
- Rate-limit errors reset the provider's global-failed flag and back off 60s.
- A handful of workers keep several calls in flight; ``_Throttle`` spaces their
  starts so the sustained rate can never exceed 45/min.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime

from my_strategy.data_manager.config import load_data_config
from my_strategy.data_manager.local_cache import LocalCache
from my_strategy.data_manager.market_data_provider import TushareProvider
from my_strategy.data_manager.stock_pool import StockPoolManager

BATCH = 2
MIN_FULL_ROWS = 1000
RATE_ERRORS = ("超限", "频率", "freq", "rate", "limit", "太快", "抱歉", "frequency", "limit")


class _Throttle:
    """Deterministic serialized rate limiter: call starts spaced >= interval apart.

    A token-bucket style limiter is deliberately avoided: the one in
    ``market_data_provider`` collapses (and its semantics don't map cleanly to
    tushare's 50-calls/min ``daily`` cap).  This holds the lock during sleep so
    call starts are exactly ``interval`` apart, bounding the sustained rate at
    1/interval regardless of worker concurrency.
    """

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
                now = time.monotonic()
            self._next_at = now + self._interval


def _is_rate_error(err: str) -> bool:
    lowered = str(err).lower()
    return any(kw in lowered for kw in RATE_ERRORS)


def _tushare_batch(provider: TushareProvider, codes: list[str], start: str, end: str):
    """fetch_daily_batch, resilient to the process-wide rate-limit flag."""
    for attempt in range(3):
        if TushareProvider._is_global_failed():
            # A prior rate-limit error flagged the whole process; clear and back off.
            TushareProvider._global_failed = False
            time.sleep(30)
        result = provider.fetch_daily_batch(codes, start, end)
        if result.ok or not _is_rate_error(result.error or ""):
            return result
        time.sleep(60 * (attempt + 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6, help="Concurrent batch downloaders.")
    parser.add_argument("--start", default="2015-01-01", help="Inclusive start date.")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="Inclusive end date.")
    parser.add_argument("--limit", type=int, default=0, help="Restrict to the first N pool codes (testing).")
    parser.add_argument("--force", action="store_true", help="Re-download even if a code already has bars.")
    args = parser.parse_args()

    config = load_data_config()
    pool = StockPoolManager(config).build_pool(mode="all_a")
    codes = [str(c) for c in pool["stock"].tolist()]
    if args.limit:
        codes = codes[: args.limit]
    total = len(codes)
    print(f"pool={total} codes, workers={args.workers}, range={args.start}..{args.end}", flush=True)

    cache = LocalCache(config)
    provider = TushareProvider(config)
    # tushare's daily interface allows 50 calls/min; leave a small margin at 45.
    throttle = _Throttle(60.0 / 45.0)

    existing_counts: dict[str, int] = {}
    if not args.force:
        try:
            conn = cache.storage.raw.connect(read_only=True)
            try:
                for row in conn.execute("SELECT stock, COUNT(*) AS n FROM stock_daily_normalized GROUP BY stock"):
                    existing_counts[str(row["stock"])] = int(row["n"])
            finally:
                conn.close()
        except Exception as exc:
            print(f"WARN could not read existing counts: {exc}", flush=True)
    already = {c for c, n in existing_counts.items() if n >= MIN_FULL_ROWS}
    todo = [c for c in codes if c not in already]
    print(f"already_full={len(already)} todo={len(todo)}", flush=True)
    if not todo:
        print("DONE nothing to download", flush=True)
        return

    pairs = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
    stats = {"ok": 0, "fail": 0}
    failures: list[tuple[str, str]] = []
    lock = threading.Lock()
    done = [0]

    def _one(pair: list[str]) -> None:
        throttle.acquire()
        result = _tushare_batch(provider, pair, args.start, args.end)
        if result.ok and result.data is not None and not result.data.empty:
            for code in pair:
                group = result.data[result.data["code"] == code]
                if group.empty:
                    with lock:
                        stats["fail"] += 1
                        failures.append((code, "no rows returned"))
                    continue
                try:
                    cache.save_stock_daily(code, group, source=result.source, incremental=True)
                    with lock:
                        stats["ok"] += 1
                        done[0] += 1
                        if done[0] % 50 == 0:
                            print(
                                json.dumps(
                                    {"event": "progress", "done": done[0], "total": len(todo), "ok": stats["ok"], "fail": stats["fail"]},
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                except Exception as exc:
                    with lock:
                        stats["fail"] += 1
                        failures.append((code, f"save: {exc}"))
        else:
            msg = result.error or "empty batch"
            for code in pair:
                with lock:
                    stats["fail"] += 1
                    failures.append((code, msg))

    t0 = time.monotonic()
    if args.workers <= 1:
        for pair in pairs:
            _one(pair)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futs = {executor.submit(_one, pair): pair for pair in pairs}
            for fut in as_completed(futs):
                fut.result()

    elapsed = time.monotonic() - t0
    print(f"RESULT ok={stats['ok']} fail={stats['fail']} elapsed={elapsed:.0f}s", flush=True)
    if failures:
        print("FAILURES (first 20):", flush=True)
        for code, msg in failures[:20]:
            print(f"  {code}: {msg}", flush=True)

    with_cache = sum(1 for c in codes if existing_counts.get(c, 0) >= MIN_FULL_ROWS or cache.storage.has_stock(c))
    print(f"VERIFY stocks_full={with_cache}/{total}", flush=True)


if __name__ == "__main__":
    main()
