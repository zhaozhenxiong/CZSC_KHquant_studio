from __future__ import annotations

import copy
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from my_strategy.services import czsc_compute as compute
from my_strategy.services.czsc_analysis import run_session, strategy_config
from my_strategy.services.czsc_backtest import execute_decisions


@pytest.fixture
def frames():
    results = []
    for symbol, count, scale in (("000001.SZ", 420, 1.0), ("600036.SH", 147, 2.0), ("000002.SZ", 35, 1.0)):
        dates = pd.bdate_range("2020-01-02", periods=count)
        prices = scale * (12 + np.arange(count) * .006 + 1.5 * np.sin(np.arange(count) / 13))
        results.append(pd.DataFrame({"symbol": symbol, "date": dates, "dt": dates + pd.Timedelta(hours=15),
                                     "id": np.arange(count), "open": prices, "high": prices + .15,
                                     "low": prices - .15, "close": prices + .04, "volume": 1000000.,
                                     "amount": prices * 1000000, "has_trade_price": 1, "source": "test"}))
    return results


def assert_parity(frames, sessions, settings):
    for frame, candidate in zip(frames, sessions):
        baseline = run_session(frame, settings)
        assert candidate.decisions == baseline.decisions
        assert candidate.events == baseline.events
        assert candidate.signals == baseline.signals
        start = frame.iloc[0]["date"].date().isoformat()
        reference = execute_decisions(candidate.symbol, frame, baseline.decisions, start, 100000, settings, "compute-test")
        actual = execute_decisions(candidate.symbol, frame, candidate.decisions, start, 100000, settings, "compute-test")
        assert actual == reference
        with pytest.raises(RuntimeError, match="不能用于图表"):
            candidate.payload()


def test_cpu_batch_is_chronological_native_position_and_ledger(frames, monkeypatch):
    monkeypatch.setattr(compute, "_torch_runtime", lambda: (None, "未安装 Torch"))
    settings = strategy_config()
    prepared = [pickle.loads(pickle.dumps(compute.prepare_replay(frame, settings))) for frame in frames]
    sessions, info = compute.replay_prepared_batch(prepared, settings, device="cpu")
    assert_parity(frames, sessions, settings)
    assert info["selected_device"] == "cpu"
    assert info["backend"] == "numpy" and not info["cuda_work"]
    assert info["bars"] == sum(len(frame) for frame in frames)
    assert info["cuda_batches"] == 0 and info["cuda_elapsed_ms"] is None
    json.dumps(info, allow_nan=False)


def test_preparation_does_not_create_position_or_confirmation_maps(frames, monkeypatch):
    def forbidden_position(*args, **kwargs):
        raise AssertionError("Preparation must not construct Position")
    monkeypatch.setattr(compute.CzscSession, "new_position", forbidden_position)
    prepared = compute.prepare_replay(frames[0], strategy_config())
    assert len(prepared.weekly_latest) == len(frames[0])
    assert np.isnan(prepared.weekly_latest[:7]).all()
    assert len(pickle.dumps(prepared)) > 0


def test_missing_torch_falls_back_only_for_auto(frames, monkeypatch):
    monkeypatch.setattr(compute, "_torch_runtime", lambda: (None, "PyTorch 不可用"))
    monkeypatch.setenv("KHQUANT_COMPUTE_DEVICE", "auto")
    settings = strategy_config()
    prepared = compute.prepare_replay(frames[0], settings)
    sessions, info = compute.replay_prepared_batch([prepared], settings)
    assert info["selected_device"] == "cpu" and "PyTorch" in info["fallback_reason"]
    assert len(sessions[0].decisions) == len(frames[0])
    assert compute.compute_status("cuda:1")["available"] is False
    with pytest.raises(RuntimeError, match="PyTorch"):
        compute.replay_prepared_batch([prepared], settings, device="cuda:1")
    with pytest.raises(ValueError, match="计算设备"):
        compute.compute_status("cuda:bogus")


def test_batch_signals_are_used_by_native_position(frames, monkeypatch):
    monkeypatch.setattr(compute, "_torch_runtime", lambda: (None, "未安装 Torch"))
    settings = strategy_config()
    prepared = compute.prepare_replay(frames[0], settings)
    def closed_week_always_down(latest, previous, ids, warmup):
        return np.full(len(ids), -1, dtype=np.int8), ids + 1 >= warmup
    monkeypatch.setattr(compute, "_numpy_signals", closed_week_always_down)
    sessions, _ = compute.replay_prepared_batch([prepared], settings, "cpu")
    assert all(decision["signals"]["周线_闭合价格_方向V1"] == "向下_任意_任意_0" for decision in sessions[0].decisions)
    assert not any(event["action"] == "BUY" for event in sessions[0].events)
    assert any(event["action"] == "BUY" for event in run_session(frames[0], settings).events)


def test_prepared_configuration_and_shape_are_checked(frames, monkeypatch):
    monkeypatch.setattr(compute, "_torch_runtime", lambda: (None, "未安装 Torch"))
    settings = strategy_config()
    prepared = compute.prepare_replay(frames[0], settings)
    changed = copy.deepcopy(settings)
    changed["warmup_bars"] += 1
    with pytest.raises(ValueError, match="配置哈希"):
        compute.replay_prepared_batch([prepared], changed, "cpu")
    prepared.bar_ids = prepared.bar_ids[:-1]
    with pytest.raises(ValueError, match="长度"):
        compute.replay_prepared_batch([prepared], settings, "cpu")


def test_prepared_stock_and_warmup_ids_cannot_be_mixed(frames, monkeypatch):
    monkeypatch.setattr(compute, "_torch_runtime", lambda: (None, "未安装 Torch"))
    settings = strategy_config()
    mixed = frames[0].copy()
    mixed.loc[1, "symbol"] = "600036.SH"
    with pytest.raises(ValueError, match="混合"):
        compute.prepare_replay(mixed, settings)
    prepared = compute.prepare_replay(frames[0], settings)
    prepared.symbol = "600036.SH"
    with pytest.raises(ValueError, match="股票"):
        compute.replay_prepared_batch([prepared], settings, "cpu")
    prepared.symbol = "000001.SZ"
    prepared.bar_ids[0] += 60
    with pytest.raises(ValueError, match="编号"):
        compute.replay_prepared_batch([prepared], settings, "cpu")


def test_empty_cpu_batch_has_no_claimed_cuda_work(monkeypatch):
    monkeypatch.setattr(compute, "_torch_runtime", lambda: (None, "未安装 Torch"))
    sessions, info = compute.replay_prepared_batch([], strategy_config(), "cpu")
    assert not sessions and info["bars"] == 0 and not info["cuda_work"]


def test_cuda_execution_failure_is_visible_and_explicit_device_does_not_fallback(frames, monkeypatch):
    settings = strategy_config()
    prepared = compute.prepare_replay(frames[0], settings)
    def available_cuda(device):
        return {"available": True, "requested_device": device, "selected_device": "cuda:1",
                "fallback_reason": None, "stages": [{}, {"device": "cuda:1"}, {}]}
    def failing_cuda(*args):
        raise RuntimeError("CUDA test failure")
    monkeypatch.setattr(compute, "compute_status", available_cuda)
    monkeypatch.setattr(compute, "_cuda_signals", failing_cuda)
    sessions, info = compute.replay_prepared_batch([prepared], settings, "auto")
    assert info["selected_device"] == "cpu" and not info["cuda_work"]
    assert "CUDA test failure" in info["fallback_reason"] and info["stages"][1]["device"] == "cpu"
    assert sessions[0].decisions == run_session(frames[0], settings).decisions
    with pytest.raises(RuntimeError, match="未回退 CPU"):
        compute.replay_prepared_batch([prepared], settings, "cuda:1")


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
def test_cuda_batch_drives_equivalent_native_intents_and_ledger(frames, device):
    if not compute.compute_status(device)["available"]:
        pytest.skip(f"{device} is unavailable")
    settings = strategy_config()
    prepared = [compute.prepare_replay(frame, settings) for frame in frames]
    sessions, info = compute.replay_prepared_batch(prepared, settings, device)
    assert_parity(frames, sessions, settings)
    assert info["selected_device"] == device and info["backend"] == "pytorch"
    assert info["cuda_work"] and info["cuda_batches"] == 1
    assert info["cuda_elapsed_ms"] >= 0
    # A float32 implementation would erase these adjacent float64 price differences.
    values = np.array([np.nextafter(1., 2.), np.nextafter(1., 0.), 1., np.nan], dtype=np.float64)
    directions, eligible, _ = compute._cuda_signals(values, np.ones(4, dtype=np.float64), np.array([58, 59, 60, 61]), 60, device)
    assert directions.tolist() == [1, -1, 0, 0]
    assert eligible.tolist() == [False, True, True, True]
    json.dumps(info, allow_nan=False)


def test_mps_selection_runs_exact_cpu_structure_and_ledger_without_claiming_mps_work(frames, monkeypatch):
    from my_strategy.tests.test_torch_devices import fake_torch
    monkeypatch.setattr(compute, "_torch_runtime", lambda: (fake_torch(mps=True, built=True), None))
    settings = strategy_config()
    prepared = [compute.prepare_replay(frame, settings) for frame in frames]
    sessions, info = compute.replay_prepared_batch(prepared, settings, "mps")
    assert_parity(frames, sessions, settings)
    assert info["selected_device"] == "mps" and info["actual_device"] == "cpu"
    assert not info["mps_work"] and not info["cuda_work"] and not info["gpu_work"]
    assert next(stage for stage in info["stages"] if stage["stage"] == "batched_signals")["device"] == "cpu"
    values = np.array([np.nextafter(1., 2.), np.nextafter(1., 0.), 1., np.nan], dtype=np.float64)
    directions, _ = compute._numpy_signals(values, np.ones(4, dtype=np.float64), np.arange(4), 1)
    assert directions.tolist() == [1, -1, 0, 0]
