#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small parallel execution helpers for KHQuant compute workloads."""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

ExecutorKind = Literal["serial", "thread", "process"]
ExecutorOption = Literal["auto", "serial", "thread", "process"]


@dataclass(frozen=True)
class ParallelTaskResult:
    index: int
    key: str
    ok: bool
    value: Any = None
    error: str = ""
    elapsed_seconds: float = 0.0


def resolve_worker_count(workers: str | int, total_tasks: int, *, cap: int = 8) -> int:
    """Resolve auto/N worker configuration with conservative defaults."""
    if total_tasks <= 1:
        return 1
    if isinstance(workers, int):
        requested = workers
    else:
        text = str(workers or "1").strip().lower()
        if text == "auto":
            cpu_count = os.cpu_count() or 2
            requested = max(1, min(cpu_count - 1, cap))
        else:
            requested = int(text)
    return max(1, min(int(requested), int(total_tasks), int(cap)))


def resolve_executor_kind(executor: ExecutorOption | str, total_tasks: int) -> ExecutorKind:
    """Choose the fastest safe executor for the current runtime when set to auto."""
    selected = str(executor or "auto").strip().lower()
    if selected in {"serial", "thread", "process"}:
        return selected  # type: ignore[return-value]
    if total_tasks <= 1:
        return "serial"
    if Path("/.dockerenv").exists():
        return "thread"
    if os.name == "nt":
        return "process"
    return "thread"


def run_parallel_tasks(
    tasks: Iterable[tuple[str, Any]],
    worker: Callable[[Any], Any],
    *,
    workers: str | int = 1,
    executor: ExecutorOption | str = "auto",
    worker_cap: int = 8,
    progress_callback: Callable[[ParallelTaskResult], None] | None = None,
    initializer: Callable[..., None] | None = None,
    initargs: Sequence[Any] = (),
) -> list[ParallelTaskResult]:
    """Run keyed tasks and return results in input order."""
    task_list = list(tasks)
    with ParallelTaskPool(
        workers=workers,
        total_tasks=len(task_list),
        executor=executor,
        worker_cap=worker_cap,
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        return pool.run(task_list, worker, progress_callback=progress_callback)


class ParallelTaskPool:
    """Reusable bounded worker pool for pipelines that feed tasks in batches."""

    def __init__(
        self,
        *,
        workers: str | int,
        total_tasks: int,
        executor: ExecutorOption | str = "auto",
        worker_cap: int = 8,
        initializer: Callable[..., None] | None = None,
        initargs: Sequence[Any] = (),
    ) -> None:
        self.worker_count = resolve_worker_count(workers, total_tasks, cap=worker_cap)
        self.executor_kind = resolve_executor_kind(executor, total_tasks)
        self.initializer = initializer
        self.initargs = tuple(initargs)
        self._pool: Executor | None = None

    def __enter__(self) -> ParallelTaskPool:
        if self.worker_count <= 1 or self.executor_kind == "serial":
            return self
        pool_kwargs: dict[str, Any] = {"max_workers": self.worker_count}
        if self.executor_kind == "process":
            # Spawn keeps accelerator state in the parent instead of inheriting
            # a loaded MPS/CUDA runtime into CPU workers.
            pool_kwargs["mp_context"] = mp.get_context("spawn")
            pool_cls = ProcessPoolExecutor
        else:
            pool_cls = ThreadPoolExecutor
        if self.initializer is not None:
            pool_kwargs["initializer"] = self.initializer
            pool_kwargs["initargs"] = self.initargs
        self._pool = pool_cls(**pool_kwargs)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=exc_type is not None)
            self._pool = None

    def run(
        self,
        tasks: Iterable[tuple[str, Any]],
        worker: Callable[[Any], Any],
        *,
        progress_callback: Callable[[ParallelTaskResult], None] | None = None,
    ) -> list[ParallelTaskResult]:
        task_list = list(tasks)
        if self._pool is None:
            results = [_run_one(index, key, payload, worker) for index, (key, payload) in enumerate(task_list)]
            for result in results:
                _report_progress(result, progress_callback)
            return results

        results: list[ParallelTaskResult] = []
        futures = {
            self._pool.submit(_run_one, index, key, payload, worker): index
            for index, (key, payload) in enumerate(task_list)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _report_progress(result, progress_callback)
        results.sort(key=lambda item: item.index)
        return results


def _report_progress(
    result: ParallelTaskResult,
    callback: Callable[[ParallelTaskResult], None] | None,
) -> None:
    if callback is None:
        return
    try:
        callback(result)
    except Exception:
        pass


def _run_one(index: int, key: str, payload: Any, worker: Callable[[Any], Any]) -> ParallelTaskResult:
    started = time.perf_counter()
    try:
        value = worker(payload)
        return ParallelTaskResult(
            index=index,
            key=key,
            ok=True,
            value=value,
            elapsed_seconds=round(time.perf_counter() - started, 4),
        )
    except Exception as exc:  # pragma: no cover - exercised by integration paths
        return ParallelTaskResult(
            index=index,
            key=key,
            ok=False,
            error=str(exc),
            elapsed_seconds=round(time.perf_counter() - started, 4),
        )
