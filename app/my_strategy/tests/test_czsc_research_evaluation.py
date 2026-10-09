"""CPU worker extraction preserves alignment, executions and isolated outputs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from my_strategy.adapters.czsc_adapter import load_bars
from my_strategy.services.czsc_research_evaluation import EVALUATION_MODES, evaluate_record


@dataclass
class EvaluationContext:
    directory: Path
    run_id: str = "isolated-evaluation-fixture"

    def subdir(self, *parts):
        path = self.directory.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path


@pytest.fixture
def evaluation_inputs(tmp_path, monkeypatch):
    path = tmp_path / "raw.db"
    dates = pd.bdate_range("2024-01-02", periods=150)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE stock_daily_normalized (stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,source TEXT,has_trade_price REAL)")
        for symbol in ("000001.SZ", "600519.SH"):
            connection.executemany("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)",
                                   [(symbol, date.date().isoformat(), 10., 10.1, 9.9, 10., 1_000_000., 10_000_000., "tushare", 1.) for date in dates])
    monkeypatch.setenv("KHQUANT_RAW_DB", str(path))
    records = []
    for symbol in ("000001.SZ", "600519.SH"):
        raw = load_bars(symbol, db_path=path)
        features = raw.copy()
        features["date"] = features["date"].dt.strftime("%Y-%m-%d")
        features["input_eligible"] = features["id"] >= 60
        features["rule_buy"] = features["id"].isin([65, 111])
        features["rule_sell"] = features["id"].isin([95, 130])
        features["rule_price_volume_buy"] = features["rule_buy"]
        features["rule_price_volume_sell"] = features["rule_sell"]
        destination = tmp_path / (symbol.replace(".", "_") + ".parquet")
        features.to_parquet(destination, index=False)
        records.append({"symbol": symbol, "path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    fold = {"name": "fixture", "test_start": dates[60].date().isoformat(), "test_end": dates[-1].date().isoformat()}
    config = {"initial_cash": 100000., "probability_threshold": .55}
    calendar = dates.strftime("%Y-%m-%d").tolist()
    return records, fold, config, calendar, path


def test_worker_runs_real_native_position_broker_and_distinct_probability_gate(evaluation_inputs, tmp_path):
    records, fold, config, calendar, _ = evaluation_inputs
    context = EvaluationContext(tmp_path / "results")
    result = evaluate_record(records[0], fold, context, config, np.full(150, .2), calendar)
    assert result["missing"] is None and set(result["accounts"]) == set(EVALUATION_MODES)
    assert result["accounts"]["ml"]["ledger"] == []
    for mode in ("price_volume", "czsc_price_volume"):
        account = result["accounts"][mode]
        assert [row["action"] for row in account["ledger"]] == ["BUY", "SELL", "BUY", "SELL"]
        assert all(row["signal_date"] < row["date"] and row["fee"] > 0 for row in account["ledger"])
        assert account["metrics"]["open_shares"] == 0
    for mode, account in result["accounts"].items():
        assert all(row["cash"] + row["shares"] * row["close"] == pytest.approx(row["equity"]) for row in account["daily"])
        destination = tmp_path / "results" / "evaluation" / "fixture" / mode / "000001_SZ"
        assert {path.name for path in destination.iterdir()} == {"daily.csv", "ledger.csv", "rejections.csv"}
    assert not list((tmp_path / "results").rglob("metadata.json"))


def test_spawn_workers_match_serial_accounts_and_keep_stock_outputs_separate(evaluation_inputs, tmp_path):
    records, fold, config, calendar, _ = evaluation_inputs
    probabilities = np.full(150, .9)
    serial_context = EvaluationContext(tmp_path / "serial")
    parallel_context = EvaluationContext(tmp_path / "parallel")
    serial = [evaluate_record(record, fold, serial_context, config, probabilities, calendar) for record in records]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(evaluate_record, record, fold, parallel_context, config, probabilities, calendar) for record in records]
        parallel = [future.result(timeout=30) for future in futures]
    assert parallel == serial
    assert len(list((tmp_path / "parallel").rglob("daily.csv"))) == 8
    assert {path.parent.name for path in (tmp_path / "parallel").rglob("daily.csv")} == {"000001_SZ", "600519_SH"}


@pytest.mark.parametrize("change,message", [("hash", "哈希"), ("date", "日期"), ("symbol", "股票"), ("close", "修订"), ("probabilities", "概率")])
def test_worker_fails_before_writing_on_snapshot_or_alignment_error(evaluation_inputs, tmp_path, change, message):
    records, fold, config, calendar, raw_path = evaluation_inputs
    record = records[0].copy()
    probabilities = np.full(150, .9)
    if change == "hash":
        record["sha256"] = "not-the-file-hash"
    elif change in {"date", "symbol"}:
        frame = pd.read_parquet(record["path"])
        frame.loc[100, change] = "2024-05-25" if change == "date" else "600519.SH"
        frame.to_parquet(record["path"], index=False)
        record["sha256"] = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    elif change == "close":
        with sqlite3.connect(raw_path) as connection:
            connection.execute("UPDATE stock_daily_normalized SET close=10.01 WHERE stock='000001.SZ' AND date=?", (calendar[100],))
    else:
        probabilities = probabilities[:-1]
    with pytest.raises(ValueError, match=message):
        evaluate_record(record, fold, EvaluationContext(tmp_path / "results"), config, probabilities, calendar)
    assert not (tmp_path / "results").exists()


def test_missing_test_interval_is_explicit_cash_retained(evaluation_inputs, tmp_path):
    records, fold, config, calendar, _ = evaluation_inputs
    fold = {**fold, "test_start": "2026-01-01", "test_end": "2026-03-31"}
    result = evaluate_record(records[0], fold, EvaluationContext(tmp_path / "results"), config, np.array([]), calendar)
    assert result == {"symbol": "000001.SZ", "accounts": {},
                      "missing": {"symbol": "000001.SZ", "reason": "no_test_bars_cash_retained"}}
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("guard,reason", [("source", "unverified_research_source"), ("calendar_gap", "missing_next_market_bar")])
def test_worker_passes_verified_source_and_calendar_guards_to_actual_broker(evaluation_inputs, tmp_path, guard, reason):
    records, fold, config, calendar, raw_path = evaluation_inputs
    record = records[0].copy()
    frame = pd.read_parquet(record["path"])
    with sqlite3.connect(raw_path) as connection:
        if guard == "source":
            connection.execute("UPDATE stock_daily_normalized SET source='unknown_vendor' WHERE stock='000001.SZ' AND date=?", (calendar[66],))
            frame.loc[66, "source"] = "unknown_vendor"
        else:
            connection.execute("DELETE FROM stock_daily_normalized WHERE stock='000001.SZ' AND date=?", (calendar[66],))
            frame = frame.drop(index=66)
    frame.to_parquet(record["path"], index=False)
    record["sha256"] = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    result = evaluate_record(record, fold, EvaluationContext(tmp_path / "guarded"), config, np.full(len(frame), .9), calendar)
    for mode in ("price_volume", "czsc_price_volume", "ml"):
        rejections = result["accounts"][mode]["rejections"]
        assert rejections[0]["reason"] == reason
        assert rejections[0]["action"] == "BUY"
        assert rejections[0]["signal_date"] == calendar[65]
        assert not any(row["date"] == calendar[66] for row in result["accounts"][mode]["ledger"])


def test_same_missing_raw_metadata_is_not_misclassified_as_revision(evaluation_inputs, tmp_path):
    records, fold, config, calendar, raw_path = evaluation_inputs
    record = records[0].copy()
    with sqlite3.connect(raw_path) as connection:
        connection.execute("UPDATE stock_daily_normalized SET has_trade_price=NULL WHERE stock='000001.SZ' AND date=?", (calendar[40],))
    frame = pd.read_parquet(record["path"])
    frame.loc[40, "has_trade_price"] = np.nan
    frame.to_parquet(record["path"], index=False)
    record["sha256"] = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    result = evaluate_record(record, fold, EvaluationContext(tmp_path / "nullable"), config, np.full(len(frame), .9), calendar)
    assert result["missing"] is None and len(result["accounts"]) == 4


def test_cpu_worker_does_not_import_torch_even_during_replay(evaluation_inputs, tmp_path):
    records, fold, config, calendar, raw_path = evaluation_inputs
    program = """import sys, json, pathlib, numpy as np
from my_strategy.tests.test_czsc_research_evaluation import EvaluationContext
from my_strategy.services.czsc_research_evaluation import evaluate_record
record, fold, config, calendar, destination = json.loads(sys.argv[1])
assert 'torch' not in sys.modules
result = evaluate_record(record, fold, EvaluationContext(pathlib.Path(destination)), config, np.full(150, .9), calendar)
assert 'torch' not in sys.modules, 'CPU replay imported torch'
print(len(result['accounts']))
"""
    environment = {**os.environ, "KHQUANT_RAW_DB": str(raw_path)}
    result = subprocess.run([sys.executable, "-c", program, json.dumps([records[0], fold, config, calendar, str(tmp_path / "subprocess")])],
                            capture_output=True, text=True, env=environment, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "4"
