"""Batch screening retains point-in-time vetoes and parent-only inference."""
from concurrent.futures import Future
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from my_strategy.services import czsc_research as service
from my_strategy.services import czsc_research_features as features_service
from my_strategy.services import czsc_research_ml as ml
from my_strategy.services import czsc_research_runtime as runtime
from my_strategy.services import czsc_research_models as models


def prepared(symbol, *, eligible=True, date="2026-09-30", schema="same", cached=False):
    frame = pd.DataFrame([{"symbol": symbol, "date": date, "input_eligible": eligible,
        "point_confirmed": True, "point_type": "确认二买辅助", "rule_buy": True, "rule_sell": False,
        "reason_codes": [] if eligible else ["unverified_source"], "available_at": date + "T15:00:00+08:00",
        "x": float(symbol[:6]) / 1000000}])
    frame.attrs = {"feature_version": "fixture", "schema_hash": schema, "data_version": symbol}
    dates = pd.to_datetime([date])
    raw = pd.DataFrame({"symbol": symbol, "date": dates, "dt": dates + pd.Timedelta(hours=15), "id": [0], "close": 10.})
    raw.attrs["data_version"] = symbol
    return {"symbol": symbol, "features": frame, "cache_hit": cached, "data_end": date,
            "raw": raw, "reference_close": 10., "data_version": symbol}


@pytest.fixture
def environment(tmp_path, monkeypatch):
    from my_strategy.core import config_loader
    original_load_config = config_loader.load_config
    monkeypatch.setattr(config_loader, "load_config", lambda name: {"version": "czsc_price_volume_mlp_v1", "feature_profile": "legacy"}
                        if name == "czsc_research" else original_load_config(name))
    training_root = tmp_path / "training"
    (training_root / "reports").mkdir(parents=True)
    training = {"data_end": "2026-09-30", "calendar": {"run_id": "verified"}, "dataset_records": [],
        "model_dir": str(training_root / "models/production"), "model_gate": {"passed": True},
        "model_manifest": {"train_label_end": "2026-06-30", "available_at": "2026-09-30"},
        "config": {"probability_threshold": .55}}
    (training_root / "reports/research.json").write_text(json.dumps(training), encoding="utf-8")
    saved = []
    def subdir(*parts):
        path = tmp_path.joinpath("screen", *parts)
        path.mkdir(parents=True, exist_ok=True)
        return path
    context = SimpleNamespace(run_id="batch-screen", subdir=subdir, to_dict=lambda: {}, write_metadata=saved.append)
    monkeypatch.setattr(runtime, "create_run_context", lambda **kwargs: context)
    monkeypatch.setattr(service, "run_path", lambda value: training_root)
    monkeypatch.setattr(service, "_calendar", lambda value: (["2026-09-29", "2026-09-30", "2026-10-08"], {"run_id": "verified"}))
    monkeypatch.setattr(service, "research_status", lambda: {"calendar_run_id": "verified", "models": []})
    monkeypatch.setattr(runtime, "latest_market_date", lambda value: "2026-09-30")
    monkeypatch.setattr(runtime, "_prepare_stock", lambda symbol, *args: prepared(symbol))
    # This fixture mocks inference; its CUDA accounting must not require hardware.
    from my_strategy.services import czsc_compute
    monkeypatch.setattr(czsc_compute, "compute_status", lambda device: {"available": True,
        "selected_device": "cuda:1" if device in {None, "auto"} else device})
    class Resolver:
        def __init__(self, **kwargs):
            self.model_run_id = kwargs.get("model_run_id")
        def resolve_dates(self, dates):
            return [{"date": str(day), "model_dir": str(training_root / "models/production") if self.model_run_id and str(day) >= "2026-09-30" else None,
                     "model_run_id": self.model_run_id, "checkpoint": "production", "probability_threshold": .55,
                     "status": "historical_shadow" if self.model_run_id else "rules_no_model", "applied_to_entry": False} for day in dates]
    class Prediction:
        def __init__(self, device=None):
            self.device, self.rows, self.batches, self.loaded = device, 0, 0, set()
        def predict(self, frame, directory, **kwargs):
            result = ml.predict_model(frame, directory, device=self.device, **kwargs)
            self.rows += len(frame)
            self.batches += 1
            self.loaded.add(str(directory))
            return result
        @property
        def diagnostics(self):
            return {"device": self.device, "rows": self.rows, "batches": self.batches, "model_loads": len(self.loaded),
                    "cache_hits": max(0, self.batches - len(self.loaded)), "actual_cuda_inference": bool(self.rows and str(self.device).startswith("cuda"))}
    monkeypatch.setattr(models, "ModelResolver", Resolver)
    monkeypatch.setattr(ml, "PredictorSession", Prediction)
    return context, saved


def test_latest_rows_are_fused_per_microbatch_with_vetoes_and_attributes(environment, monkeypatch):
    symbols = ["000001.SZ", "300001.SZ", "600001.SH", "600002.SH", "000002.SZ", "600003.SH"]
    data = {symbol: prepared(symbol, cached=True) for symbol in symbols}
    data["600002.SH"] = prepared("600002.SH", eligible=False)
    data["000002.SZ"] = prepared("000002.SZ", date="2026-09-29")
    monkeypatch.setattr(runtime, "_prepare_stock", lambda symbol, *args: data[symbol])
    calls = []
    parent = os.getpid()
    def predict(frame, path, **kwargs):
        assert os.getpid() == parent
        assert frame.attrs["feature_version"] == "fixture" and frame.attrs["schema_hash"] == "same"
        assert set(frame.date) == {"2026-09-30"}
        calls.append(frame.symbol.tolist())
        return frame.x.to_numpy() + .6
    monkeypatch.setattr(ml, "predict_model", predict)
    progress = []
    result = service.scan_research(end="2026-09-30", symbols=symbols, model_run_id="training", cpu_workers=1, batch_size=3,
                                   progress=lambda *args: progress.append(args))
    assert calls == [["000001.SZ", "600001.SH"], ["600003.SH"]]
    rows = {row["symbol"]: row for row in result["rows"]}
    assert rows["000001.SZ"]["model_probability"] == pytest.approx(.600001)
    assert rows["600001.SH"]["model_probability"] == pytest.approx(1.200001)
    for symbol in ("300001.SZ", "600002.SH", "000002.SZ"):
        assert rows[symbol]["model_probability"] is None and rows[symbol]["category"] == "excluded"
    info = result["compute_info"]
    assert info["model_inference_rows"] == 3 and info["model_inference_batches"] == 2 and info["cuda_work"]
    assert info["cache_hits"] == 4 and info["cache_misses"] == 2
    assert [p[1] for p in progress] == sorted(p[1] for p in progress)
    assert all(p[2] == len(symbols) for p in progress)
    assert any(p[0] == "GPU批量ML推理" for p in progress)


def test_no_model_preserves_rule_evidence_and_records_no_cuda(environment, monkeypatch):
    monkeypatch.setattr(ml, "predict_model", lambda *args, **kwargs: pytest.fail("No-model scan ran inference"))
    symbols = ["000001.SZ", "600001.SH"]
    result = service.scan_research(end="2026-09-30", symbols=symbols, cpu_workers=1, batch_size=1)
    assert result["coverage"] == {"requested": 2, "success": 2, "failed": 0}
    for row in result["rows"]:
        assert row["category"] == "buy" and row["action"] == row["position_intent"] == "BUY"
        assert row["candidate_action"] == "BUY" and not row["model_validated"]
        assert row["model_probability"] is None and row["model_resolution"]["status"] == "rules_no_model"
        assert row["agent_evidence"][2]["judgment"] == "unavailable"
        assert row["agent_evidence"][3]["judgment"] == "eligible_signal"
    info = result["compute_info"]
    assert not info["cuda_work"] and info["model_inference_rows"] == info["model_inference_batches"] == 0
    assert info["cache_hits"] == 0 and info["cache_misses"] == 2 and info["cpu_preparation_executor"] == "parent_serial"


def test_withdrawn_release_does_not_qualify_request_using_prewarm_days(environment, monkeypatch):
    item = prepared("000001.SZ", date="2026-09-29")
    last = prepared("000001.SZ")
    item["features"] = pd.concat([item["features"], last["features"]], ignore_index=True)
    item["raw"] = pd.concat([item["raw"], last["raw"]], ignore_index=True)
    item["raw"]["id"] = range(2)
    item["data_end"] = "2026-09-30"
    monkeypatch.setattr(runtime, "_prepare_stock", lambda *args: item)
    monkeypatch.setattr(ml, "predict_model", lambda frame, *args, **kwargs: np.full(len(frame), .99))
    def routes(self, dates):
        return [{"date": str(day), "model_dir": "model" if index == 0 else None,
                 "model_run_id": "released" if index == 0 else None, "probability_threshold": .55,
                 "status": "production_release" if index == 0 else "rules_no_model",
                 "applied_to_entry": index == 0} for index, day in enumerate(dates)]
    monkeypatch.setattr(models.ModelResolver, "resolve_dates", routes)
    executed = []
    def execute(symbol, raw, decisions, *args, **kwargs):
        executed.extend(decisions)
        return {"daily": [{"date": "2026-09-30", "equity": 100000}], "ledger": [], "trades": [], "rejections": []}
    monkeypatch.setattr(runtime, "execute_decisions", execute)
    result = runtime.run_research(kind="backtest", start="2026-09-30", end="2026-09-30",
                                  symbols=["000001.SZ"], cpu_workers=1, device="cpu", usage_mode="production")
    assert executed[0]["ml_filter_applied"] and not executed[1]["ml_filter_applied"]
    assert not result["model_gate"]["passed"] and result["model_summary"]["ml_applied_days"] == 0
    scan = runtime.run_research(kind="scan", end="2026-09-30", symbols=["000001.SZ"], cpu_workers=1,
                                device="cpu", usage_mode="production")
    assert not scan["model_gate"]["passed"] and not scan["rows"][0]["model_validated"]


def test_stale_scan_veto_clears_probability_and_latest_evidence(environment, monkeypatch):
    item = prepared("000001.SZ", date="2026-09-29")
    item["features"]["reason_codes"] = [np.array(["calendar_gap"])]
    monkeypatch.setattr(runtime, "_prepare_stock", lambda *args: item)
    monkeypatch.setattr(models.ModelResolver, "resolve_dates", lambda self, dates: [
        {"date": str(dates[0]), "model_dir": "model", "applied_to_entry": True, "probability_threshold": .55}])
    monkeypatch.setattr(ml, "predict_model", lambda *args, **kwargs: np.array([.99]))
    result = runtime.run_research(kind="scan", end="2026-09-30", symbols=["000001.SZ"], cpu_workers=1,
                                  device="cpu", usage_mode="production")
    row = result["rows"][0]
    assert row["category"] == "excluded" and row["action"] == "WAIT" and row["target_weight"] == 0
    assert row["reason_codes"] == ["calendar_gap", "missing_target_date"]
    assert not row["model_validated"] and not result["model_gate"]["passed"]
    assert row["model_resolution"]["probability"] is None and row["model_resolution"]["status"] == "input_veto"
    assert row["model_resolution"]["reason_codes"] == row["reason_codes"]
    assert row["agent_evidence"][2]["judgment"] == "unavailable"
    assert row["agent_evidence"][3]["judgment"] == "veto"


def test_runtime_resolves_auto_before_loading_predictor(environment, monkeypatch):
    from my_strategy.services import czsc_compute
    monkeypatch.setattr(czsc_compute, "compute_status", lambda value: {"available": True, "selected_device": "cpu"})
    worker = runtime.ResearchRuntime(end="2026-09-30", device="auto")
    assert worker.predictor.device == "cpu"


def test_runtime_rejects_explicit_unavailable_accelerator_before_rule_only_work(environment, monkeypatch):
    from my_strategy.services import czsc_compute
    monkeypatch.setattr(czsc_compute, "compute_status", lambda value: {"available": False,
        "selected_device": None, "fallback_reason": "explicit MPS unavailable"})
    monkeypatch.setattr(runtime, "_prepare_stock", lambda *args: pytest.fail("Unavailable device prepared rules"))
    with pytest.raises(RuntimeError, match="explicit MPS unavailable"):
        runtime.ResearchRuntime(end="2026-09-30", device="mps")


def test_read_only_worker_requires_cache_hash_and_raw_identity(tmp_path, monkeypatch):
    raw = pd.DataFrame({"symbol": ["000001.SZ"], "date": ["2026-09-30"], "close": [10.]})
    raw.attrs["data_version"] = "unchanged"
    frame = prepared("000001.SZ")["features"]
    frame.attrs.update(feature_version=features_service.FEATURE_VERSION, schema_hash=features_service.FEATURE_SCHEMA_HASH)
    dates = ["2026-09-30"]
    frame.attrs["calendar_hash"] = service.stable_hash(dates)
    path = tmp_path / "cache.parquet"
    frame.to_parquet(path, index=False)
    cache = {"symbol": "000001.SZ", "path": str(path), "sha256": service._file_hash(path), "data_version": "unchanged",
             "feature_schema_hash": features_service.FEATURE_SCHEMA_HASH}
    monkeypatch.setattr(service, "load_bars", lambda *args, **kwargs: raw)
    monkeypatch.setattr(features_service, "build_features", lambda value: pytest.fail("Valid cache was rebuilt"))
    hit = service._prepare_scan_stock("000001.SZ", "2026-09-30", dates, cache)
    assert hit["cache_hit"] and len(hit["features"]) == 1
    pd.testing.assert_frame_equal(hit["features"], frame)
    with pytest.raises(ValueError, match="哈希不匹配"):
        service._prepare_scan_stock("000001.SZ", "2026-09-30", dates, {**cache, "sha256": "invalid"})
    with pytest.raises(ValueError, match="股票身份不匹配"):
        service._prepare_scan_stock("000001.SZ", "2026-09-30", dates, {**cache, "symbol": "600001.SH"})
    with pytest.raises(ValueError, match="schema不匹配"):
        service._prepare_scan_stock("000001.SZ", "2026-09-30", dates, {**cache, "feature_schema_hash": "old"})
    frame.attrs["feature_version"] = "old"
    frame.to_parquet(path, index=False)
    cache["sha256"] = service._file_hash(path)
    with pytest.raises(ValueError, match="schema不匹配"):
        service._prepare_scan_stock("000001.SZ", "2026-09-30", dates, cache)
    rebuilt = []
    monkeypatch.setattr(features_service, "build_features", lambda value: rebuilt.append(True) or frame)
    monkeypatch.setattr(service, "_calendar_quality", lambda data, *args: data)
    raw.attrs["data_version"] = "changed"
    miss = service._prepare_scan_stock("000001.SZ", "2026-09-30", dates, cache)
    assert not miss["cache_hit"] and rebuilt == [True]


def test_bad_stock_is_retained_as_failure_but_model_integrity_error_aborts(environment, monkeypatch):
    def prepare(symbol, *args):
        if symbol == "000002.SZ":
            raise ValueError("冻结研究特征文件哈希不匹配")
        return prepared(symbol)
    monkeypatch.setattr(runtime, "_prepare_stock", prepare)
    result = service.scan_research(end="2026-09-30", symbols=["000001.SZ", "000002.SZ"], cpu_workers=1)
    assert result["coverage"] == {"requested": 2, "success": 1, "failed": 1}
    assert result["failures"] == [{"symbol": "000002.SZ", "error": "冻结研究特征文件哈希不匹配"}]
    context, saved = environment
    saved.clear()
    def corrupted(*args, **kwargs):
        raise ValueError("model checkpoint hash mismatch")
    monkeypatch.setattr(ml, "predict_model", corrupted)
    with pytest.raises(ValueError, match="checkpoint hash"):
        service.scan_research(end="2026-09-30", symbols=["000001.SZ"], model_run_id="training", cpu_workers=1)
    assert not saved


def test_mixed_semantic_versions_cannot_be_hidden_by_concat(environment, monkeypatch):
    monkeypatch.setattr(runtime, "_prepare_stock", lambda symbol, *args: prepared(symbol, schema=symbol))
    monkeypatch.setattr(ml, "predict_model", lambda *args, **kwargs: pytest.fail("Mixed schema reached model"))
    with pytest.raises(ValueError, match="schema不一致"):
        service.scan_research(end="2026-09-30", symbols=["000001.SZ", "000002.SZ"], model_run_id="training", cpu_workers=1)


def test_parallel_completion_order_keeps_mapping_and_bounded_batches(environment, monkeypatch):
    calls, shutdowns, replay_symbols, pools = [], [], [], []
    parent = os.getpid()
    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 4 and kwargs["mp_context"].get_start_method() == "spawn"
            pools.append(self)
        def submit(self, worker, symbol, *args):
            future = Future()
            if worker is runtime._replay_stock:
                replay_symbols.append(symbol["symbol"])
            future.set_result(worker(symbol, *args))
            return future
        def shutdown(self, **kwargs):
            shutdowns.append(kwargs)
    monkeypatch.setattr(runtime, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(runtime, "wait", lambda futures, **kwargs: ({list(futures)[-1]}, set(list(futures)[:-1])))
    def predict(frame, *args, **kwargs):
        assert os.getpid() == parent
        calls.append(frame.symbol.tolist())
        return frame.x.to_numpy()
    monkeypatch.setattr(ml, "predict_model", predict)
    symbols = [f"{i:06d}.SZ" for i in range(1, 66)]
    result = service.scan_research(end="2026-09-30", symbols=symbols, model_run_id="training")
    assert list(map(len, calls)) == [32, 32, 1]
    assert [symbol for batch in calls for symbol in batch] == symbols
    assert all(row["model_probability"] == pytest.approx(float(row["symbol"][:6]) / 1000000) for row in result["rows"])
    assert result["compute_info"]["cpu_preparation_executor"] == "spawn_process_pool"
    assert result["compute_info"]["cpu_replay_executor"] == "spawn_process_pool"
    assert replay_symbols == symbols and len(pools) == 1
    assert shutdowns == [{"wait": True, "cancel_futures": False}]


def test_parallel_replay_retains_serial_position_events_and_model_evidence(environment, monkeypatch):
    symbols = ["000001.SZ", "600001.SH", "000002.SZ", "300001.SZ"]
    items = {}
    for symbol in symbols:
        item = prepared(symbol)
        dates = pd.bdate_range("2026-09-21", periods=8)
        item["features"] = pd.concat([item["features"]] * len(dates), ignore_index=True)
        item["features"]["date"] = dates.strftime("%Y-%m-%d")
        item["features"]["rule_buy"] = [True, False, False, True, False, False, True, False]
        item["features"]["rule_sell"] = [False, False, True, False, False, True, False, False]
        item["features"]["input_eligible"] = [True, True, True, False, True, True, True, True]
        item["raw"] = pd.concat([item["raw"]] * len(dates), ignore_index=True)
        item["raw"]["date"], item["raw"]["dt"] = dates, dates + pd.Timedelta(hours=15)
        item["raw"]["id"] = range(len(dates))
        item["data_end"] = dates[-1].date().isoformat()
        items[symbol] = item
    monkeypatch.setattr(runtime, "_prepare_stock", lambda symbol, *args: copy.deepcopy(items[symbol]))
    monkeypatch.setattr(models.ModelResolver, "resolve_dates", lambda self, dates: [
        {"date": str(day), "model_dir": "model", "applied_to_entry": index % 2 == 0,
         "checkpoint": "dated", "probability_threshold": .55} for index, day in enumerate(dates)])
    parent = os.getpid()
    def predict(frame, *args, **kwargs):
        assert os.getpid() == parent
        return np.where(pd.to_datetime(frame.date).dt.day.eq(21), .4, .99)
    monkeypatch.setattr(ml, "predict_model", predict)
    serial = runtime.ResearchRuntime(end="2026-09-30", device="cpu")
    expected = serial.replay_batch([copy.deepcopy(items[s]) for s in symbols])
    class Pool:
        def __init__(self, **kwargs):
            pass
        def submit(self, worker, *args):
            future = Future()
            future.set_result(worker(*args))
            return future
        def shutdown(self, **kwargs):
            pass
    monkeypatch.setattr(runtime, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(runtime, "wait", lambda futures, **kwargs: ({list(futures)[-1]}, set(list(futures)[:-1])))
    parallel = runtime.ResearchRuntime(end="2026-09-30", device="cpu")
    actual = []
    for batch, errors in parallel.prepare_batches(symbols, workers=2, batch_size=3):
        assert not errors
        actual.extend(parallel.replay_batch(batch))
    assert [item["replay"] for item in actual] == [item["replay"] for item in expected]
    assert all(item["replay"]["events"] for item in actual)
    assert expected[0]["replay"]["decisions"][0]["action"] == "WAIT"
    assert expected[-1]["replay"]["decisions"][0]["action"] == "BUY"
    assert parallel._replay_pool is None and parallel.compute_info()["cpu_replay_executor"] == "spawn_process_pool"
    assert serial.compute_info()["cpu_replay_executor"] == "parent_serial"
    latest = serial.replay_batch([copy.deepcopy(items[s]) for s in symbols], include_decisions=False)
    assert [item["replay"] for item in latest] == [
        {key: value for key, value in item["replay"].items() if key != "decisions"} for item in expected]


def test_cancel_stops_submissions_and_inference_and_cancels_pending(environment, monkeypatch):
    submissions, shutdowns = [], []
    class Pool:
        def __init__(self, **kwargs):
            pass
        def submit(self, worker, symbol, *args):
            submissions.append(symbol)
            return Future()
        def shutdown(self, **kwargs):
            shutdowns.append(kwargs)
    monkeypatch.setattr(runtime, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(ml, "predict_model", lambda *args, **kwargs: pytest.fail("Cancelled scan ran inference"))
    def check():
        if submissions:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        service.scan_research(end="2026-09-30", symbols=[f"{i:06d}.SZ" for i in range(1, 10)], model_run_id="training",
                              cpu_workers=4, batch_size=3, check_cancel=check)
    assert submissions == ["000001.SZ", "000002.SZ", "000003.SZ"]
    assert shutdowns == [{"wait": False, "cancel_futures": True}] and not environment[1]


def test_cancel_during_replay_stops_dispatch_and_does_not_publish(environment, monkeypatch):
    replay_submissions, shutdowns, inference = [], [], []
    class Pool:
        def __init__(self, **kwargs):
            pass
        def submit(self, worker, *args):
            future = Future()
            if worker is runtime._replay_stock:
                replay_submissions.append(args[0]["symbol"])
            else:
                future.set_result(worker(*args))
            return future
        def shutdown(self, **kwargs):
            shutdowns.append(kwargs)
    monkeypatch.setattr(runtime, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(ml, "predict_model", lambda frame, *args, **kwargs: inference.append(os.getpid()) or np.full(len(frame), .99))
    def check():
        if replay_submissions:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        service.scan_research(end="2026-09-30", symbols=["000001.SZ", "000002.SZ", "600001.SH"],
                              model_run_id="training", cpu_workers=2, check_cancel=check)
    assert replay_submissions == ["000001.SZ"] and inference == [os.getpid()]
    assert shutdowns == [{"wait": False, "cancel_futures": True}] and not environment[1]


def test_cancel_after_last_adjudication_does_not_publish_complete_result(environment, monkeypatch):
    cancelled, writes = False, []
    def progress(stage, *args):
        nonlocal cancelled
        cancelled = stage == "多角色证据扫描"
    def check():
        if cancelled:
            raise RuntimeError("cancelled")
    monkeypatch.setattr(service, "_save", lambda *args: writes.append("json"))
    monkeypatch.setattr(runtime, "_write_frame", lambda *args: writes.append("csv"))
    with pytest.raises(RuntimeError, match="cancelled"):
        service.scan_research(end="2026-09-30", symbols=["000001.SZ"], cpu_workers=1, progress=progress, check_cancel=check)
    assert not writes and not environment[1]


def test_worker_replay_does_not_initialize_torch():
    script = """
import sys
from my_strategy.services.czsc_research import _prepare_scan_stock
from my_strategy.services.czsc_research_runtime import _replay_stock
from my_strategy.tests.test_czsc_research_scan_perf import prepared
import numpy as np
item = prepared('000001.SZ')
item.update(routes=[{'probability_threshold': .55, 'applied_to_entry': False}], probabilities=np.array([.9]))
assert _replay_stock(item)['latest_decision']['action'] == 'BUY'
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_batch_predictions_match_single_rows_on_cpu_and_available_gpu(tmp_path):
    import torch
    dates = pd.bdate_range("2024-01-02", periods=130)
    x = np.sin(np.arange(len(dates)) / 3)
    learning = pd.DataFrame({"date": dates, "symbol": "000001.SZ", "x": x, "z": np.cos(np.arange(len(dates)) / 7),
        "label": (x > 0).astype(float), "label_end": dates + pd.offsets.BDay(11), "label_available": True, "input_eligible": True})
    ml.train_model(learning, ["x", "z"], tmp_path / "model", "2024-03-29", "2024-04-01", "2024-05-31",
                   device="cpu", epochs=1, batch_size=32, hidden_sizes=(8, 4), min_train_rows=20, min_validation_rows=20)
    inference = learning[learning.date >= "2024-06-03"].iloc[:6].copy()
    expected = np.concatenate([ml.predict_model(inference.iloc[[i]], tmp_path / "model", device="cpu") for i in range(len(inference))])
    actual = ml.predict_model(inference, tmp_path / "model", device="cpu")
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    if torch.cuda.is_available():
        gpu = ml.predict_model(inference, tmp_path / "model", device="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        np.testing.assert_allclose(gpu, expected, atol=1e-6, rtol=0)
