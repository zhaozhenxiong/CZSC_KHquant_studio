"""Entry policies cross HTTP, services, runtime, account start and model routing."""
from __future__ import annotations

import copy
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_analysis, czsc_compute, czsc_entry_plan
from my_strategy.services import czsc_research as service
from my_strategy.services import czsc_research_ml as ml
from my_strategy.services import czsc_research_models as models
from my_strategy.services import czsc_research_runtime as runtime
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.web_dashboard.app import create_app
from my_strategy.web_dashboard.tasks import TaskManager
from my_strategy.tests.test_czsc_research_scan_perf import environment


@pytest.fixture
def manager(tmp_path):
    instance = TaskManager(tmp_path / "jobs.db", ResultStore(tmp_path / "results.db", tmp_path / "runs"), workers=1)
    yield instance
    instance.close()


@pytest.fixture
def planned_data(environment, monkeypatch):
    days = ["2026-09-25", "2026-09-28", "2026-09-29", "2026-09-30"]
    dates = pd.to_datetime(days)
    features = pd.DataFrame({"symbol": "600027.SH", "date": days, "input_eligible": True,
        "rule_buy": True, "rule_sell": False, "rule_price_volume_buy": True, "rule_price_volume_sell": False,
        "reason_codes": [[] for _ in days], "point_type": "二买辅助", "point_anchor": "2026-09-24",
        "point_confirmed_at": "2026-09-25T15:00:00+08:00", "confirmed_age": 0.,
        "ma20": 98., "atr14_ratio": .02, "structure_low": 96., "x": .5})
    features.attrs = {"feature_version": "fixture", "schema_hash": "same", "data_version": "600027.SH"}
    raw = pd.DataFrame({"symbol": "600027.SH", "date": dates, "dt": dates + pd.Timedelta(hours=15),
        "id": range(len(days)), "close": 100.})
    raw.attrs = {"data_version": "600027.SH", "price_basis": "unadjusted", "legacy_fuyao_bars": 0,
                 "unverified_trade_price_bars": 0}
    item = {"symbol": "600027.SH", "raw": raw, "features": features,
            "cache_hit": False, "data_end": days[-1], "data_version": "600027.SH"}
    monkeypatch.setattr(runtime, "_prepare_stock", lambda *args: copy.deepcopy(item))
    monkeypatch.setattr(service, "_calendar", lambda *args: ([*days, "2026-10-08"], {"run_id": "verified"}))
    return item


def test_http_rejects_risk_production_before_work_and_preserves_legacy_default(manager, monkeypatch):
    monkeypatch.setattr(manager, "submit", lambda kind, spec: {"kind": kind, "spec": spec})
    with TestClient(create_app(manager, personal_store=SimpleNamespace())) as client:
        for kind in ("scan", "backtest"):
            response = client.post("/api/tasks", json={"kind": kind, "spec": {
                "research": True, "entry_policy": "risk", "usage_mode": "production"}})
            assert response.status_code == 422
            assert "尚为实验" in response.text
            default = client.post("/api/tasks", json={"kind": kind})
            assert default.status_code == 202
            assert default.json()["spec"]["entry_policy"] == "legacy"
        assert client.post("/api/analysis", json={"symbol": "600027.SH", "entry_policy": "risk",
                            "usage_mode": "production"}).status_code == 422
        assert client.post("/api/tasks", json={"kind": "scan", "spec": {
            "research": False, "entry_policy": "fresh"}}).status_code == 422


@pytest.mark.parametrize("kind", ["scan", "backtest"])
@pytest.mark.parametrize("policy", ["legacy", "fresh", "risk"])
def test_task_dispatch_preserves_policy_and_user_start_without_personal_tables(manager, monkeypatch, kind, policy):
    from my_strategy.storage import personal_portfolio
    monkeypatch.setattr(personal_portfolio, "PersonalStore", lambda: SimpleNamespace(holdings=lambda: {"items": []}))
    monkeypatch.setattr(czsc_compute, "compute_status", lambda *args: {"available": True, "selected_device": "cpu"})
    monkeypatch.setattr(service, "scan_research", lambda **kwargs: kwargs)
    monkeypatch.setattr(service, "backtest_research", lambda **kwargs: kwargs)
    spec = {"research": True, "symbols": ["600027.SH"], "entry_policy": policy,
            "start": "2026-09-29", "end": "2026-09-30", "device": "cpu"}
    result = manager._run("fixture", kind, spec)
    assert result["entry_policy"] == policy and result["start"] == spec["start"]


@pytest.mark.parametrize("policy", ["legacy", "fresh", "risk"])
def test_services_forward_entry_policy_and_start(monkeypatch, policy):
    monkeypatch.setattr(runtime, "run_research", lambda **kwargs: kwargs)
    common = {"symbols": ["600027.SH"], "start": "2026-09-29", "end": "2026-09-30", "entry_policy": policy}
    scan = service.scan_research(**common)
    backtest = service.backtest_research(**common, initial_cash=100000)
    assert scan["kind"] == "scan" and backtest["kind"] == "backtest"
    assert scan["entry_policy"] == backtest["entry_policy"] == policy
    assert scan["start"] == backtest["start"] == common["start"]


@pytest.mark.parametrize("policy", ["fresh", "risk"])
def test_old_fixed_horizon_release_is_shadow_for_planned_entry_contract(planned_data, monkeypatch, policy):
    monkeypatch.setattr(models.ModelResolver, "resolve_dates", lambda self, dates: [
        {"date": str(day), "model_dir": "old-fixed-horizon-model", "model_run_id": "old-model",
         "checkpoint": "production", "probability_threshold": .55,
         "applied_to_entry": True, "status": "production_active", "release_id": "old-release"} for day in dates])
    monkeypatch.setattr(ml, "predict_model", lambda frame, *args, **kwargs: np.full(len(frame), .01))
    worker = runtime.ResearchRuntime(end="2026-09-30", device="cpu", entry_policy=policy, position_start="2026-09-30")
    item = worker.replay_batch([copy.deepcopy(planned_data)])[0]
    assert all(not row["ml_filter_applied"] for row in item["replay"]["decisions"])
    latest = item["replay"]["latest_decision"]
    assert latest["action"] == "BUY" and latest["probability"] == pytest.approx(.01)
    assert latest["model_resolution"]["status"] == "entry_contract_shadow"
    assert latest["model_resolution"]["release_id"] is None
    assert "entry_contract_mismatch" in latest["model_resolution"]["reason_codes"]
    assert not latest["model_used"]


@pytest.mark.parametrize("kind", ["scan", "backtest"])
def test_runtime_binds_plan_parameters_and_keeps_user_account_flat_during_prewarm(planned_data, environment, monkeypatch, kind):
    executed = []
    def execute(symbol, raw, decisions, start, initial_cash, config, run_id, **kwargs):
        executed.append({"decisions": decisions, "start": start, **kwargs})
        return {"daily": [{"date": "2026-09-29", "equity": initial_cash},
                          {"date": "2026-09-30", "equity": initial_cash}],
                "ledger": [], "trades": [], "rejections": []}
    monkeypatch.setattr(runtime, "execute_decisions", execute)
    result = runtime.run_research(kind=kind, symbols=["600027.SH"], start="2026-09-29", end="2026-09-30",
                                  entry_policy="fresh", cpu_workers=1, device="cpu")
    parameters = czsc_entry_plan.entry_plan_parameters()
    assert result["entry_policy"] == result["request"]["entry_policy"] == "fresh"
    assert result["position_start"] == "2026-09-29"
    assert result["entry_parameters"] == result["request"]["entry_parameters"] == parameters
    expected_hash = stable_hash({"version": czsc_entry_plan.PLAN_VERSION, "entry_policy": "fresh", "parameters": parameters})
    assert result["entry_plan_config_hash"] == result["request"]["entry_plan_config_hash"] == expected_hash
    assert environment[1][-1]["request"]["entry_plan_config_hash"] == expected_hash
    if kind == "backtest":
        call = executed[0]
        assert call["entry_policy"] == "fresh" and call["start"] == "2026-09-29"
        decisions = call["decisions"]
        assert all(row["action"] == "WAIT" and row["eligible"] for row in decisions[:2])
        assert decisions[2]["action"] == "BUY" and decisions[3]["action"] == "HOLD"
        assert decisions[2]["entry_plan"]["parameters"] == parameters
        assert decisions[2]["account_context"] == "flat_at_start"
    else:
        row = result["rows"][0]
        assert row["action"] == "HOLD" and row["entry_plan"]["status"] == "holding"
        assert row["entry_plan"]["signal_date"] == "2026-09-29"
        assert row["entry_plan"]["parameters"] == parameters


def test_runtime_freezes_parameters_before_workers_and_hash_changes_with_risk_assumptions(planned_data, monkeypatch):
    initial = czsc_entry_plan.entry_plan_parameters()
    selected = {"parameters": initial}
    monkeypatch.setattr(czsc_entry_plan, "entry_plan_parameters", lambda parameters=None:
                        dict(selected["parameters"] if parameters is None else parameters))
    worker = runtime.ResearchRuntime(end="2026-09-30", device="cpu", entry_policy="risk", position_start="2026-09-30")
    initial_hash = worker.entry_plan_config_hash
    selected["parameters"] = {**initial, "risk_fraction": .005}
    result = worker.replay_batch([copy.deepcopy(planned_data)])[0]
    assert result["replay"]["latest_decision"]["entry_plan"]["parameters"] == initial
    second = runtime.run_research(kind="scan", symbols=["600027.SH"], start="2026-09-30", end="2026-09-30",
                                 entry_policy="risk", cpu_workers=1, device="cpu")
    assert second["entry_plan_config_hash"] != initial_hash
    second_hash = second["config_hash"]
    selected["parameters"] = initial
    first = runtime.run_research(kind="scan", symbols=["600027.SH"], start="2026-09-30", end="2026-09-30",
                                entry_policy="risk", cpu_workers=1, device="cpu")
    assert first["config_hash"] != second_hash


def test_analysis_forwards_start_and_records_same_frozen_plan_contract(planned_data, monkeypatch):
    raw = planned_data["raw"]
    monkeypatch.setattr(czsc_analysis, "load_bars", lambda *args, **kwargs: raw)
    monkeypatch.setattr(czsc_analysis, "run_session", lambda *args: SimpleNamespace(payload=lambda lower: {}, events=[]))
    monkeypatch.setattr(czsc_analysis, "_next_session_decision", lambda *args: {"action": "WAIT"})
    result = czsc_analysis.analyze_stock("600027.SH", start="2026-09-29", end="2026-09-30",
                                         research=True, entry_policy="fresh", device="cpu")
    assert result["research"]["position_start"] == "2026-09-29"
    assert result["events"][0]["time"] == "2026-09-29"
    assert result["next_session_decision"]["action"] == "HOLD"
    parameters = result["research"]["entry_parameters"]
    assert parameters == result["next_session_decision"]["entry_plan"]["parameters"]
    expected_hash = stable_hash({"version": czsc_entry_plan.PLAN_VERSION, "entry_policy": "fresh", "parameters": parameters})
    assert result["research"]["entry_plan_config_hash"] == expected_hash


def test_analysis_http_preserves_policy_start_and_binds_risk_parameters_to_run_hash(planned_data, manager, monkeypatch):
    from my_strategy.core import paths
    monkeypatch.setattr(paths, "ARTIFACT_RUNS_ROOT", manager.results.runs_root)
    monkeypatch.setattr(czsc_analysis, "load_bars", lambda *args, **kwargs: planned_data["raw"])
    monkeypatch.setattr(czsc_analysis, "run_session", lambda *args: SimpleNamespace(payload=lambda lower: {}, events=[]))
    monkeypatch.setattr(czsc_analysis, "_next_session_decision", lambda *args: {"action": "WAIT"})
    initial = czsc_entry_plan.entry_plan_parameters()
    selected = {"parameters": initial}
    monkeypatch.setattr(czsc_entry_plan, "entry_plan_parameters", lambda parameters=None:
                        dict(selected["parameters"] if parameters is None else parameters))
    with TestClient(create_app(manager, personal_store=SimpleNamespace())) as client:
        spec = {"symbol": "600027.SH", "start": "2026-09-29", "end": "2026-09-30", "device": "cpu"}
        default = client.post("/api/analysis", json=spec)
        assert default.status_code == 200
        assert default.json()["request"]["entry_policy"] == "legacy"
        fresh = client.post("/api/analysis", json={**spec, "entry_policy": "fresh"})
        assert fresh.status_code == 200
        assert fresh.json()["request"]["start"] == fresh.json()["research"]["position_start"] == spec["start"]
        assert fresh.json()["events"][0]["time"] == spec["start"]
        risk = client.post("/api/analysis", json={**spec, "start": "2026-09-30", "entry_policy": "risk"})
        assert risk.status_code == 200
        selected["parameters"] = {**initial, "risk_fraction": .005}
        changed = client.post("/api/analysis", json={**spec, "start": "2026-09-30", "entry_policy": "risk"})
        assert changed.status_code == 200
        assert changed.json()["config_hash"] != risk.json()["config_hash"]
        assert changed.json()["run_context"]["config_hash"] != risk.json()["run_context"]["config_hash"]


def test_legacy_runtime_default_retains_prefix_account_and_request_hash_shape(planned_data):
    result = runtime.run_research(kind="scan", symbols=["600027.SH"], start="2026-09-29", end="2026-09-30",
                                  cpu_workers=1, device="cpu")
    assert result["entry_policy"] == "legacy" and result["position_start"] is None
    assert not {"entry_policy", "entry_parameters", "entry_plan_config_hash"} & result["request"].keys()
    row = result["rows"][0]
    assert row["account_context"] == "theoretical_prefix_position"
    assert row["entry_plan"]["signal_date"] == "2026-09-25"
    assert row["action"] == "HOLD" and row["entry_gate_passed"]


@pytest.fixture
def study_record(tmp_path, planned_data):
    from my_strategy.services.czsc_research_features import FEATURE_COLUMNS, FEATURE_VERSION, FEATURE_SCHEMA_HASH
    frame = planned_data["features"].copy()
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = 0.
    for column in ("open", "high", "low", "close"):
        frame[column] = 100.
    frame["volume"], frame["amount"], frame["source"], frame["has_trade_price"] = 1e6, 1e8, "tushare", 1
    frame.attrs = {"feature_version": FEATURE_VERSION, "schema_hash": FEATURE_SCHEMA_HASH}
    path = tmp_path / "frozen_features.parquet"
    frame.to_parquet(path, index=False)
    record = {"symbol": "600027.SH", "path": str(path), "sha256": service._file_hash(path), "data_version": "study-fixture"}
    def subdir(*parts):
        destination = tmp_path.joinpath("study", *parts)
        destination.mkdir(parents=True, exist_ok=True)
        return destination
    context = SimpleNamespace(run_id="frozen-study", subdir=subdir)
    return record, context, frame


def test_study_prepare_worker_forwards_captured_parameters_to_actual_exit_labels(study_record, monkeypatch):
    from my_strategy.adapters import czsc_adapter
    from my_strategy.scripts import study_czsc_executable_entries as study
    from my_strategy.services import czsc_research_strategy_labels as labels_service
    record, context, frame = study_record
    frozen = czsc_entry_plan.entry_plan_parameters()
    config = {"entry_parameters": dict(frozen), "initial_cash": 100000, "feature_start": "2026-01-01"}
    live = study._raw(frame)
    live.attrs["data_version"] = record["data_version"]
    monkeypatch.setattr(czsc_adapter, "load_bars", lambda *args, **kwargs: live)
    # Parent already captured the protocol. No worker may read a later local file.
    monkeypatch.setattr(czsc_entry_plan, "load_config", lambda *args: pytest.fail("worker reread local entry configuration"))
    calls = []
    def labels(features, raw, calendar, **kwargs):
        calls.append(kwargs)
        result = pd.DataFrame({"label": 1., "label_end": "2026-09-30", "label_available": True,
                               "net_return": .01, "invested_net_return": .02}, index=features.index)
        result.attrs = {"execution": {}, "unavailable_counts": {}, "label_contract": {"entry_parameters": kwargs["entry_parameters"]}}
        return result
    monkeypatch.setattr(labels_service, "build_strategy_labels", labels)
    evidence, sample, contract = study._prepare_record((record, context, frame.date.tolist(), config))
    assert evidence["raw_verified"] and evidence["mature_labels"] == len(frame)
    assert len(sample) == len(frame)
    assert calls[0]["entry_policy"] == "fresh"
    assert calls[0]["entry_parameters"] == frozen == contract["entry_parameters"]


def test_study_evaluation_workers_use_frozen_parameters_after_local_config_changes(study_record, monkeypatch):
    from my_strategy.scripts import study_czsc_executable_entries as study
    from my_strategy.services import czsc_backtest
    record, context, frame = study_record
    frozen = czsc_entry_plan.entry_plan_parameters()
    config = {"entry_parameters": dict(frozen), "initial_cash": 100000, "probability_threshold": .55}
    # A changed local configuration would halve position risk. Explicit parent
    # parameters must instead preserve the original plan for every policy.
    monkeypatch.setattr(czsc_entry_plan, "load_config", lambda *args: {"risk": {**frozen, "risk_fraction": .005}})
    calls = []
    def execute(symbol, raw, decisions, start, initial_cash, config, run_id, **kwargs):
        calls.append({"decisions": decisions, **kwargs})
        return {"daily": [{"date": start, "equity": initial_cash}], "ledger": [], "trades": [],
                "rejections": [], "metrics": {"total_return": 0.}}
    monkeypatch.setattr(czsc_backtest, "execute_decisions", execute)
    fold = {"name": "fixture", "test_start": "2026-09-30", "test_end": "2026-09-30"}
    result = study._evaluate_record((record, fold, context, config, frame.date.tolist(), np.full(len(frame), .99)))
    assert set(result["accounts"]) == {"legacy", "fresh", "risk", "fresh_ml"}
    assert len(calls) == 4
    for call in calls:
        assert all(row["entry_plan"]["parameters"] == frozen for row in call["decisions"])
    risk = next(call for call in calls if call["entry_policy"] == "risk")
    latest = risk["decisions"][-1]
    assert latest["action"] == "BUY"
    assert latest["entry_plan"]["risk_fraction"] == .01
    assert latest["target_weight"] == pytest.approx(.01 * 100 / (100 - 95.5))
