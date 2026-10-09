"""Bounded CPU preparation followed by one parent-only tensor replay per batch."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterator

import pandas as pd

from my_strategy.adapters.czsc_adapter import load_bars, normalize_symbol
from my_strategy.core.run_context import stable_hash


def batch_settings(cpu_workers: int | None = None, batch_size: int | None = None) -> tuple[int, int]:
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, int)) for value in (cpu_workers, batch_size)):
        raise ValueError("进程数与微批大小必须为整数")
    workers = int(cpu_workers if cpu_workers is not None else os.environ.get("KHQUANT_CPU_WORKERS", "4"))
    size = int(batch_size if batch_size is not None else os.environ.get("KHQUANT_GPU_BATCH_SIZE", "32"))
    if not 1 <= workers <= 16:
        raise ValueError("CPU 准备进程数应为 1 到 16")
    if not 1 <= size <= 128:
        raise ValueError("GPU 微批大小应为 1 到 128")
    return workers, size


def _prepare_stock(symbol: str, end: str, settings: dict[str, Any], raw_db: str | None,
                   require_latest: bool):
    # Spawned workers import no Torch and never receive artifact/output roots.
    from my_strategy.services.czsc_compute import prepare_replay
    frame = load_bars(symbol, end=end, db_path=raw_db)
    last = pd.Timestamp(frame.iloc[-1]["date"]).date().isoformat()
    if require_latest and last != end:
        raise ValueError(f"行情缺少扫描日 {end}，标的最后日期为 {last}")
    return prepare_replay(frame, settings)


def iter_replay_batches(symbols: list[str], *, end: str, config: dict[str, Any],
                        device: str | None = None, cpu_workers: int | None = None,
                        batch_size: int | None = None, db_path: str | Path | None = None,
                        require_latest: bool = False, check_cancel: Callable[[], None] | None = None,
                        progress: Callable[[int, int, int], None] | None = None) -> Iterator[dict[str, Any]]:
    """Keep at most one microbatch of frames in memory, preserving input order.

    Preparation errors are explicit per-symbol failures. Tensor/device failures
    abort the job. On cancellation queued CPU work is cancelled; already-running
    read-only CPU preparations finish without scheduling further GPU replay.
    """
    from my_strategy.services.czsc_compute import compute_status, replay_prepared_batch
    workers, size = batch_settings(cpu_workers, batch_size)
    symbols = list(dict.fromkeys(normalize_symbol(symbol) for symbol in symbols))
    if not symbols:
        raise ValueError("批处理至少需要一个股票代码")
    end = pd.Timestamp(end).date().isoformat()
    raw_db = str(Path(db_path).resolve()) if db_path is not None else None
    check = check_cancel or (lambda: None)
    check()
    status = compute_status(device)
    if not status["available"]:
        raise RuntimeError(status.get("fallback_reason") or "请求的计算设备不可用")
    check()
    completed, failed = 0, 0
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) if workers > 1 else None
    finished = False
    try:
        for offset in range(0, len(symbols), size):
            check()
            batch_symbols = symbols[offset:offset + size]
            prepared: dict[int, Any] = {}
            failures: dict[int, dict[str, str]] = {}
            started = time.perf_counter()
            if executor is None:
                for index, symbol in enumerate(batch_symbols):
                    check()
                    try:
                        prepared[index] = _prepare_stock(symbol, end, config, raw_db, require_latest)
                    except (ValueError, FileNotFoundError) as exc:
                        failures[index] = {"symbol": symbol, "error": str(exc)}
                        failed += 1
                    completed += 1
                    if progress:
                        progress(completed, len(symbols), failed)
            else:
                pending = {executor.submit(_prepare_stock, symbol, end, config, raw_db, require_latest): index
                           for index, symbol in enumerate(batch_symbols)}
                while pending:
                    check()
                    done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        check()
                        index = pending.pop(future)
                        try:
                            prepared[index] = future.result()
                        except (ValueError, FileNotFoundError) as exc:
                            failures[index] = {"symbol": batch_symbols[index], "error": str(exc)}
                            failed += 1
                        completed += 1
                        if progress:
                            progress(completed, len(symbols), failed)
            preparation_seconds = time.perf_counter() - started
            check()
            items = [prepared[index] for index in sorted(prepared)]
            sessions, compute = replay_prepared_batch(items, config, device=device)
            check()
            info = {**compute, "pipeline": "cpu_prepare_parent_tensor_replay",
                    "cpu_preparation_executor": "spawn_process_pool" if executor is not None else "parent_serial",
                    "cpu_workers": workers, "configured_batch_size": size,
                    "requested_in_batch": len(batch_symbols), "prepared_in_batch": len(items),
                    "preparation_wall_seconds": preparation_seconds, "config_hash": stable_hash(config)}
            yield {"prepared": items, "sessions": sessions,
                   "failures": [failures[index] for index in sorted(failures)], "compute_info": info}
        finished = True
    finally:
        if executor is not None:
            executor.shutdown(wait=finished, cancel_futures=not finished)
