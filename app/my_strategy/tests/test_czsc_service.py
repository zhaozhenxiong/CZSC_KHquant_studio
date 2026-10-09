"""New task lifecycle, portfolio accounting and HTTP contract checks."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
import pandas as pd

from my_strategy.storage.czsc_results import ResultStore
from my_strategy.web_dashboard.app import create_app
from my_strategy.web_dashboard.tasks import TaskManager, aggregate_accounts, aggregate_compute


@pytest.fixture
def manager(tmp_path):
    store = ResultStore(tmp_path / "runs.db", tmp_path / "runs")
    manager = TaskManager(tmp_path / "jobs.db", store, workers=1)
    yield manager
    manager.close()


def wait_terminal(manager, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["status"] not in {"queued", "running", "cancelling"}:
            return job
        time.sleep(.01)
    raise AssertionError("task did not terminate")


def test_cancel_and_idempotency(manager, monkeypatch):
    entered = threading.Event()
    def work(job_id, kind, spec):
        entered.set()
        while True:
            manager._check(job_id)
            time.sleep(.01)
    monkeypatch.setattr(manager, "_run", work)
    first = manager.submit("scan", {"symbols": ["600519.SH"]})
    assert entered.wait(2)
    second = manager.submit("scan", {"symbols": ["600519.SH"]})
    assert first["job_id"] == second["job_id"]
    manager.cancel(first["job_id"])
    assert wait_terminal(manager, first["job_id"])["status"] == "cancelled"


def test_failure_is_visible_and_result_survives_restart(manager, monkeypatch):
    def broken(*args):
        raise RuntimeError("bad source")
    monkeypatch.setattr(manager, "_run", broken)
    job = manager.submit("scan", {})
    assert "bad source" in wait_terminal(manager, job["job_id"])["error"]
    result = {"run_id": "czsc-test", "metrics": {"net_return": .1}, "daily": []}
    manager.results.save("backtest", result)
    another = ResultStore(manager.results.db_path, manager.results.runs_root)
    assert another.get("czsc-test") == result
    with pytest.raises(ValueError):
        another.save("backtest", {"run_id": "../escape"})


def test_equal_accounts_keep_failed_budget_in_cash():
    accounts = [{"daily": [{"date": "2026-09-01", "equity": 50}, {"date": "2026-09-02", "equity": 45}], "trades": [{"date": "2026-09-02", "symbol": "600519.SH", "fee": 1}], "rejections": []}]
    result = aggregate_accounts(accounts, 100, 2)
    assert result["daily"][-1]["equity"] == 95
    assert result["metrics"]["net_return"] == pytest.approx(-.05)
    assert result["metrics"]["max_drawdown"] == pytest.approx(.05)
    assert result["metrics"]["fees"] == 1


def test_new_http_contract_and_old_routes_reject(manager):
    with TestClient(create_app(manager)) as client:
        assert client.get("/api/health").json()["method"] == "CZSC"
        assert client.get("/api/tasks/no-such-job").status_code == 404
        assert client.get("/models").status_code == 404
        assert client.get("/api/v1/jobs").status_code == 404
        assert client.post("/api/tasks", json={"kind": "retrain"}).status_code == 422
        assert client.post("/api/tasks", json={"kind": "backtest", "spec": {"start": "2026-09-28", "end": "2026-09-01"}}).status_code == 422
        assert client.post("/api/tasks", json={"kind": "scan", "spec": {"symbols": ["bad"]}}).status_code == 422
        assert client.post("/api/tasks", json={"kind": "scan"}, headers={"Origin": "https://other.example"}).status_code == 403


def test_optional_token_does_not_expose_secret(manager, monkeypatch):
    monkeypatch.setenv("KHQUANT_API_TOKEN", "test-secret")
    with TestClient(create_app(manager)) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/tasks").status_code == 401
        response = client.get("/api/tasks", headers={"Authorization": "Bearer test-secret"})
        assert response.status_code == 200
        assert "test-secret" not in response.text


def test_scan_rejects_stale_last_event(manager, monkeypatch, tmp_path):
    from my_strategy.adapters import czsc_adapter
    from my_strategy.services import czsc_batch
    from my_strategy.core import paths
    monkeypatch.setattr(paths, "ARTIFACT_RUNS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(czsc_adapter, "data_status", lambda: {"latest_date": "2026-09-28"})
    monkeypatch.setattr(czsc_adapter, "latest_market_date", lambda end: "2026-09-28")
    def batches(symbols, **kwargs):
        assert kwargs["require_latest"] and kwargs["end"] == "2026-09-28"
        assert kwargs["device"] == "cpu"
        kwargs["check_cancel"]()
        kwargs["progress"](2, 2, 1)
        frame = pd.DataFrame([{"date": "2026-09-28"}])
        frame.attrs["data_version"] = "fixture"
        yield {"prepared": [SimpleNamespace(symbol="000001.SZ", frame=frame)],
               "sessions": [SimpleNamespace(events=[{"available_at": "2026-09-28", "action": "buy"}])],
               "failures": [{"symbol": "600519.SH", "error": "行情缺少扫描日 2026-09-28，标的最后日期为 2026-09-25"}],
               "compute_info": {"selected_device": "cpu", "backend": "numpy", "cuda_work": False, "bars": 1, "tensor_operations": ["comparison"], "cuda_elapsed_ms": None}}
    monkeypatch.setattr(czsc_batch, "iter_replay_batches", batches)
    job = manager.submit("scan", {"symbols": ["000001.SZ", "600519.SH"], "end": "2026-09-28", "device": "cpu"})
    finished = wait_terminal(manager, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    result = finished["result"]
    assert result["coverage"] == {"requested": 2, "success": 1, "failed": 1}
    assert [row["symbol"] for row in result["rows"]] == ["000001.SZ"]
    assert "行情缺少扫描日" in result["failures"][0]["error"]
    assert result["compute_info"]["cuda_work"] is False
    assert result["compute_info"]["selected_device"] == "cpu"


def test_interactive_analysis_saves_a_reproducible_run(manager, monkeypatch):
    from my_strategy.core import paths
    from my_strategy.services import czsc_analysis
    monkeypatch.setattr(paths, "ARTIFACT_RUNS_ROOT", manager.results.runs_root)
    def calculate(**kwargs):
        assert kwargs["config"]["czsc_version"] == "1.0.1"
        return {"symbol": kwargs["symbol"], "data_end": "2025-09-30", "data_version": "test-input",
                "as_of": kwargs["as_of"], "frequencies": {}, "events": []}
    monkeypatch.setattr(czsc_analysis, "analyze_stock", calculate)
    with TestClient(create_app(manager)) as client:
        response = client.post("/api/analysis", json={"symbol": "000001.SZ", "start": "2025-01-01", "as_of": "2025-09-30"})
        assert response.status_code == 200
        result = response.json()
        assert result["request"]["as_of"] == "2025-09-30"
        assert result["run_context"]["data_version"] == "test-input"
        assert client.get("/api/runs/" + result["run_id"]).json() == result
        assert client.get("/api/runs").json()["runs"][0]["kind"] == "analysis"


def test_default_dates_device_and_batch_http_contract(manager, monkeypatch):
    monkeypatch.setattr(manager, "submit", lambda kind, spec: {"kind": kind, "spec": spec})
    with TestClient(create_app(manager)) as client:
        for kind in ("scan", "backtest"):
            response = client.post("/api/tasks", json={"kind": kind, "spec": {"device": "cuda:1", "cpu_workers": 4, "batch_size": 32}})
            assert response.status_code == 202
            assert response.json()["spec"]["start"] == "2026-01-01"
            assert response.json()["spec"]["device"] == "cuda:1"
        historical = client.post("/api/tasks", json={"kind": "backtest", "spec": {"start": "2024-01-01", "end": "2024-09-30", "device": "cpu"}})
        assert historical.status_code == 202
        assert historical.json()["spec"]["start"] == "2024-01-01"
        assert "start" not in client.post("/api/tasks", json={"kind": "update"}).json()["spec"]
        for field, value in (("device", "gpu:1"), ("cpu_workers", True), ("cpu_workers", 17), ("batch_size", 0), ("batch_size", 129)):
            assert client.post("/api/tasks", json={"kind": "scan", "spec": {field: value}}).status_code == 422


def test_compute_capability_endpoint_does_not_claim_execution(manager, monkeypatch):
    from my_strategy.services import czsc_compute
    def capability(device=None):
        return {"available": True, "requested_device": device or "cuda:1", "selected_device": device or "cuda:1", "devices": [{"index": 1, "name": "test GPU"}]}
    monkeypatch.setattr(czsc_compute, "compute_status", capability)
    with TestClient(create_app(manager)) as client:
        result = client.get("/api/compute/status?device=cpu").json()
        assert result["selected_device"] == "cpu"
        assert "cuda_work" not in result


def test_compute_aggregation_keeps_cpu_and_cuda_provenance():
    cpu = {"selected_device": "cpu", "backend": "numpy", "cuda_work": False, "cuda_batches": 0, "batch_size": 2,
           "bars": 200, "tensor_operations": ["compare", "warmup"], "cuda_elapsed_ms": None, "tensor_seconds": .01,
           "stages": [{"stage": "batched_signals", "device": "cpu"}]}
    only_cpu = aggregate_compute([cpu, cpu])
    assert not only_cpu["cuda_work"] and only_cpu["cuda_elapsed_ms"] is None
    assert only_cpu["bars"] == 400 and only_cpu["tensor_operations"] == ["compare", "warmup"]
    gpu = {**cpu, "selected_device": "cuda:1", "backend": "pytorch", "cuda_work": True, "cuda_batches": 1,
           "cuda_elapsed_ms": 4.5, "stages": [{"stage": "batched_signals", "device": "cuda:1"}]}
    only_gpu = aggregate_compute([gpu, gpu])
    assert only_gpu["selected_device"] == "cuda:1" and only_gpu["cuda_elapsed_ms"] == 9
    assert only_gpu["cuda_batches"] == 2 and only_gpu["tensor_seconds"] == pytest.approx(.02)
    fallback = {**cpu, "fallback_reason": "auto CUDA failure, CPU fallback"}
    mixed = aggregate_compute([gpu, fallback])
    assert mixed["selected_device"] == "mixed" and mixed["selected_devices"] == ["cuda:1", "cpu"]
    assert mixed["cuda_work"] and mixed["cuda_elapsed_ms"] == 4.5 and mixed["cuda_batches"] == 1
    assert mixed["fallback_reason"] == fallback["fallback_reason"]
    assert mixed["stages"][0]["device"] == "mixed"
    assert gpu["stages"][0]["device"] == "cuda:1"
