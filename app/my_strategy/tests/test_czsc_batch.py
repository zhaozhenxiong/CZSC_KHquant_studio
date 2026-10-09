"""Spawned read-only preparation, ordered microbatches and unchanged execution."""
from __future__ import annotations

import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from my_strategy.adapters.czsc_adapter import load_bars
from my_strategy.cli import build_parser
from my_strategy.services.czsc_analysis import run_session, strategy_config
from my_strategy.services.czsc_backtest import backtest_stock, execute_decisions
from my_strategy.services.czsc_batch import batch_settings, iter_replay_batches
from my_strategy.services.czsc_compute import prepare_replay, replay_prepared_batch


@pytest.fixture
def raw_db(tmp_path):
    path = tmp_path / "raw.db"
    dates = pd.bdate_range("2020-01-02", periods=420)
    prices = 12 + np.arange(len(dates)) * .006 + 1.5 * np.sin(np.arange(len(dates)) / 13)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE stock_daily_normalized (stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,has_trade_price INTEGER,source TEXT,PRIMARY KEY(stock,date))")
        for symbol, count in (("000001.SZ", 420), ("000002.SZ", 417)):
            conn.executemany("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)", [
                (symbol, day.date().isoformat(), float(price), float(price + .15), float(price - .15),
                 float(price + .04), 1e6, float(price * 1e6), 1, "test")
                for day, price in zip(dates[:count], prices[:count])])
    return path


def test_spawned_batches_preserve_stock_order_and_report_missing(raw_db):
    progress = []
    result = list(iter_replay_batches(["000002.SZ", "000001.SZ", "999999.SZ"], end="2021-08-11",
                                     config=strategy_config(), device="cpu", cpu_workers=2, batch_size=2,
                                     db_path=raw_db, progress=lambda *values: progress.append(values)))
    assert [[item.symbol for item in batch["prepared"]] for batch in result] == [["000002.SZ", "000001.SZ"], []]
    assert [failure["symbol"] for batch in result for failure in batch["failures"]] == ["999999.SZ"]
    assert progress[-1] == (3, 3, 1)
    assert result[0]["compute_info"]["cpu_preparation_executor"] == "spawn_process_pool"
    assert result[0]["compute_info"]["selected_device"] == "cpu"
    assert result[0]["compute_info"]["bars"] == 837
    assert len(result[0]["sessions"][0].decisions) == 417


def test_scan_preparation_does_not_accept_stale_event_date(raw_db):
    result = list(iter_replay_batches(["000002.SZ", "000001.SZ"], end="2021-08-11",
                                     config=strategy_config(), device="cpu", cpu_workers=1, batch_size=2,
                                     db_path=raw_db, require_latest=True))
    assert [item.symbol for item in result[0]["prepared"]] == ["000001.SZ"]
    assert result[0]["failures"][0]["symbol"] == "000002.SZ"
    assert "行情缺少扫描日" in result[0]["failures"][0]["error"]


def test_cancellation_stops_before_tensor_replay(raw_db, monkeypatch):
    from my_strategy.services import czsc_compute
    completed = []
    called = []
    class UserCancelled(RuntimeError):
        pass
    def check():
        if completed:
            raise UserCancelled("cancel")
    monkeypatch.setattr(czsc_compute, "replay_prepared_batch", lambda *args, **kwargs: called.append(True))
    with pytest.raises(UserCancelled):
        list(iter_replay_batches(["000001.SZ", "000002.SZ"], end="2021-08-11",
                                 config=strategy_config(), device="cpu", cpu_workers=1, batch_size=2,
                                 db_path=raw_db, check_cancel=check,
                                 progress=lambda *values: completed.append(values)))
    assert completed == [(1, 2, 0)]
    assert not called


def test_requested_unavailable_gpu_fails_before_cpu_work(raw_db, monkeypatch):
    from my_strategy.services import czsc_compute
    monkeypatch.setattr(czsc_compute, "compute_status", lambda device: {"available": False, "fallback_reason": "requested GPU unavailable"})
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        list(iter_replay_batches(["000001.SZ"], end="2021-08-11", config=strategy_config(),
                                 device="cuda:99", cpu_workers=1, db_path=raw_db))


def test_prepared_cpu_intents_preserve_actual_ledger_and_daily_equity(raw_db, tmp_path, monkeypatch):
    from my_strategy.core import paths
    monkeypatch.setattr(paths, "ARTIFACT_RUNS_ROOT", tmp_path / "runs")
    config = strategy_config()
    frame = load_bars("000001.SZ", end="2021-08-11", db_path=raw_db)
    full = run_session(frame, config)
    baseline = execute_decisions("000001.SZ", frame, full.decisions, "2020-04-01", 100000, config, "batch-equivalence")
    prepared = prepare_replay(frame, config)
    sessions, compute = replay_prepared_batch([prepared], config, device="cpu")
    result = backtest_stock("000001.SZ", "2020-04-01", "2021-08-11", config=config,
                            run_id="batch-equivalence", prepared=prepared, session=sessions[0], compute_info=compute)
    assert result["ledger"] == baseline["ledger"]
    assert result["daily"] == baseline["daily"]
    assert result["metrics"] == baseline["metrics"]
    assert result["compute_info"]["signal_elements"] == len(frame) * 2
    assert all(row["signal_date"] < row["date"] for row in result["ledger"])
    assert pd.read_csv(result["artifacts"]["ledger"]).shape[0] == len(result["ledger"])


def test_reused_preparation_rejects_wrong_cutoff_config_or_decisions(raw_db):
    frame = load_bars("000001.SZ", db_path=raw_db)
    config = strategy_config()
    prepared = prepare_replay(frame, config)
    sessions, compute = replay_prepared_batch([prepared], config, device="cpu")
    with pytest.raises(ValueError, match="截止日之后"):
        backtest_stock("000001.SZ", "2020-04-01", "2020-05-01", config=config, prepared=prepared, session=sessions[0])
    changed = copy.deepcopy(config)
    changed["execution"]["commission"] = .001
    with pytest.raises(ValueError, match="配置不匹配"):
        backtest_stock("000001.SZ", "2020-04-01", "2021-08-11", config=changed, prepared=prepared, session=sessions[0])
    sessions[0].symbol = "000002.SZ"
    with pytest.raises(ValueError, match="会话股票"):
        backtest_stock("000001.SZ", "2020-04-01", "2021-08-11", config=config, prepared=prepared, session=sessions[0])
    sessions[0].symbol = "000001.SZ"
    sessions[0].decisions[0]["date"] = "1999-01-01"
    with pytest.raises(ValueError, match="日期不一致"):
        backtest_stock("000001.SZ", "2020-04-01", "2021-08-11", config=config, prepared=prepared, session=sessions[0])


def test_batch_cli_options_and_bounded_settings(monkeypatch):
    args = build_parser().parse_args(["backtest-batch", "--stocks", "000001.SZ", "--start", "2026-01-01",
                                      "--device", "cuda:1", "--cpu-workers", "4", "--batch-size", "32"])
    assert (args.device, args.cpu_workers, args.batch_size) == ("cuda:1", 4, 32)
    for command in ("analyze", "scan", "backtest", "backtest-batch"):
        defaults = build_parser().parse_args([command])
        assert defaults.start == "2026-01-01"
        assert defaults.end is None
    monkeypatch.setenv("KHQUANT_CPU_WORKERS", "3")
    monkeypatch.setenv("KHQUANT_GPU_BATCH_SIZE", "16")
    assert batch_settings() == (3, 16)
    for workers, size in ((0, 16), (17, 16), (4, 0), (4, 129), (1.5, 16), (True, 16)):
        with pytest.raises(ValueError):
            batch_settings(workers, size)
