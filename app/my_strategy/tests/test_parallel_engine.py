from __future__ import annotations

from my_strategy.execution.parallel_engine import ParallelTaskPool


def _double(value: int) -> int:
    return value * 2


def test_reusable_thread_pool_processes_multiple_bounded_batches():
    with ParallelTaskPool(workers=2, total_tasks=4, executor="thread") as pool:
        executor_id = id(pool._pool)
        first = pool.run([("a", 1), ("b", 2)], _double)
        second = pool.run([("c", 3), ("d", 4)], _double)

        assert id(pool._pool) == executor_id

    assert [item.value for item in first] == [2, 4]
    assert [item.value for item in second] == [6, 8]
