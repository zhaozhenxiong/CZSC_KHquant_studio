"""Artifact audit catches altered executions and capital denominators without ML runtime."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.scripts.audit_czsc_research import MODES, _file_hash, _hash, audit_run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


@pytest.fixture
def artifact_run(tmp_path):
    root = tmp_path / "training-test"
    dates = pd.bdate_range("2024-01-02", periods=12).strftime("%Y-%m-%d").tolist()
    calendar_path = tmp_path / "calendar-test/reports/calendar.json"
    calendar = {"verified": True, "source": "synthetic-test-calendar", "dates": dates}
    write(calendar_path, calendar)
    feature_version, schema_hash = "synthetic-feature-v1", "frozen-synthetic-formulas"
    frame = pd.DataFrame({"symbol": "000001.SZ", "date": dates, "label": [1.0, *([np.nan] * 11)],
                          "label_end": [dates[11], *([None] * 11)], "label_available": [True, *([False] * 11)],
                          "label_reason": ["available", *(["label_immature"] * 11)], "input_eligible": True,
                          "entry_date": [dates[1], *([None] * 11)], "exit_date": [dates[11], *([None] * 11)]})
    frame.attrs.update(feature_version=feature_version, schema_hash=schema_hash, data_version="frozen-test-prices")
    dataset = root / "dataset/000001_SZ.parquet"
    dataset.parent.mkdir(parents=True)
    frame.to_parquet(dataset, index=False)
    record = {"symbol": "000001.SZ", "path": str(dataset), "data_version": "frozen-test-prices", "feature_schema_hash": schema_hash,
              "bars": 12, "data_end": dates[-1], "sha256": _file_hash(dataset), "label_available_rows": 1,
              "label_reason_counts": {"available": 1, "label_immature": 11}}
    fold = {"name": "fixture", "test_start": dates[5], "test_end": dates[8]}
    config = {"horizon": 10, "initial_cash": 10000, "probability_threshold": .55, "folds": [fold]}
    models = {}
    for name in ("fixture", "production"):
        checkpoint = root / "models" / name / "model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic-checkpoint-hash-fixture")
        schema, preprocess = {"columns": ["x"]}, {"median": [0.0]}
        manifest = {"schema": schema, "schema_sha256": _hash(schema), "preprocess": preprocess, "preprocess_sha256": _hash(preprocess),
                    "checkpoint": str(checkpoint), "checkpoint_sha256": _file_hash(checkpoint), "feature_schema_binding": "bound",
                    "feature_version": feature_version, "feature_schema_hash": schema_hash, "train_label_end": "2023-09-29",
                    "train_end": "2023-09-30", "validation_start": "2023-10-01", "validation_label_end": "2023-12-29",
                    "validation_end": "2023-12-31", "available_at": "2023-12-31", "device": "synthetic-no-runtime",
                    "calibration": {"fit_start": "2023-10-01", "fit_end": "2023-12-31"}, "sample_counts": {"train": 200, "validation": 100}}
        manifest["manifest_sha256"] = _hash(manifest)
        write(checkpoint.parent / "manifest.json", manifest)
        models[name] = manifest
    ledger = pd.DataFrame([
        {"date": dates[5], "signal_date": dates[4], "symbol": "000001.SZ", "action": "BUY", "price": 10., "shares": 100,
         "fee": 5., "cash_flow": -1005., "position_before": 0, "position_after": 100, "cash_after": 8995.},
        {"date": dates[7], "signal_date": dates[6], "symbol": "000001.SZ", "action": "SELL", "price": 12., "shares": 100,
         "fee": 5.6, "cash_flow": 1194.4, "position_before": 100, "position_after": 0, "cash_after": 10189.4},
    ])
    daily = pd.DataFrame({"date": dates[5:9], "cash": [8995., 8995., 10189.4, 10189.4], "shares": [100, 100, 0, 0],
                          "close": [11., 12., 12., 12.], "equity": [10095., 10195., 10189.4, 10189.4]})
    aggregate = pd.DataFrame({"date": dates[5:9], "equity": daily.equity + 10000})
    metrics = {"final_equity": 20189.4, "total_return": 20189.4 / 20000 - 1, "fees": 10.6, "trade_count": 2}
    variants = {}
    for mode in MODES:
        location = root / "evaluation/fixture" / mode / "000001_SZ"
        location.mkdir(parents=True)
        daily.to_csv(location / "daily.csv", index=False)
        ledger.to_csv(location / "ledger.csv", index=False)
        (location / "rejections.csv").write_text("\n")
        aggregate.to_csv(location.parent / "aggregate_daily.csv", index=False)
        variants[mode] = {"accounts": 1, "metrics": metrics}
    context = {"stocks": ["000001.SZ", "000002.SZ"]}
    training = {"run_id": root.name, "data_end": dates[-1], "run_context": context, "config": config, "model_gate": {"passed": False},
                "coverage": {"requested": 2, "success": 1, "failed": 1}, "dataset_records": [record],
                "failures": [{"symbol": "000002.SZ", "error": "synthetic preparation failure"}],
                "calendar": {"path": str(calendar_path), "hash": stable_hash(calendar)}, "model_manifest": models["production"],
                "evaluation": [{"fold": fold, "model_manifest": models["fixture"], "variants": variants,
                                "frozen_mainboard_accounts": 2, "unavailable_accounts_cash_retained": [{"symbol": "000002.SZ"}]}]}
    write(root / "reports/research.json", training)
    write(root / "metadata.json", {**context, "status": "complete", "research_config": config, "config_hash": stable_hash(config)})
    return tmp_path, root


def test_audit_passes_with_failed_budget_retained_and_empty_rejection_csv(artifact_run):
    base, root = artifact_run
    report = audit_run(root.name, runs_root=base, workers=4)
    assert report["passed"], report["errors"]
    assert report["dataset"]["bars"] == 12
    assert report["dataset"]["label_reason_counts"] == {"available": 1, "label_immature": 11}
    assert report["evaluation"][0]["modes"]["ml"]["failed_cash_budget"] == 10000
    assert (root / "reports/execution_audit.json").is_file()
    result = subprocess.run([sys.executable, "-c", "import sys; from my_strategy.scripts.audit_czsc_research import audit_run; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_corrupt_signal_cash_and_failed_capital_are_detected(artifact_run):
    base, root = artifact_run
    location = root / "evaluation/fixture/ml/000001_SZ"
    ledger = pd.read_csv(location / "ledger.csv")
    ledger.loc[0, "signal_date"] = ledger.loc[0, "date"]
    ledger.loc[0, "shares"] = 101
    ledger.to_csv(location / "ledger.csv", index=False)
    daily = pd.read_csv(location / "daily.csv")
    daily.loc[0, "cash"] += 100
    daily.to_csv(location / "daily.csv", index=False)
    aggregate = pd.read_csv(location.parent / "aggregate_daily.csv")
    aggregate["equity"] -= 10000
    aggregate.to_csv(location.parent / "aggregate_daily.csv", index=False)
    report = audit_run(root.name, runs_root=base)
    rules = {error["rule"] for error in report["errors"]}
    assert not report["passed"]
    assert {"ledger_next_verified_session", "buy_round_lot", "daily_matches_actual_ledger_cash_shares", "aggregate_daily_includes_failed_cash"} <= rules


def test_schema_checkpoint_and_coverage_integrity_detected(artifact_run):
    base, root = artifact_run
    path = root / "dataset/000001_SZ.parquet"
    frame = pd.read_parquet(path)
    frame.attrs["schema_hash"] = "same-column-different-semantics"
    frame.to_parquet(path, index=False)
    checkpoint = root / "models/production/model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    (root / "evaluation/fixture/ml/000001_SZ/rejections.csv").unlink()
    report = audit_run(root.name, runs_root=base)
    rules = {error["rule"] for error in report["errors"]}
    assert {"dataset_sha256", "dataset_feature_semantics", "model_checkpoint_sha256_or_path", "missing_execution_file"} <= rules
    assert not report["passed"]


def test_audit_cannot_read_outside_run(artifact_run):
    base, root = artifact_run
    with pytest.raises(ValueError, match="invalid run identity"):
        audit_run("../training-test", runs_root=base)
    with pytest.raises(ValueError, match="workers"):
        audit_run(root.name, runs_root=base, workers=9)


def test_subcent_inventory_and_half_percent_summary_error_cannot_pass(artifact_run):
    base, root = artifact_run
    path = root / "evaluation/fixture/ml/000001_SZ/daily.csv"
    frame = pd.read_csv(path)
    frame["shares"] = frame["shares"].astype(float)
    frame.loc[0, "shares"] += .001
    frame.to_csv(path, index=False)
    source = root / "reports/research.json"
    training = json.loads(source.read_text())
    training["evaluation"][0]["variants"]["ml"]["metrics"]["total_return"] += .005
    write(source, training)
    report = audit_run(root.name, runs_root=base)
    assert {"daily_integer_round_lot_inventory", "daily_matches_actual_ledger_cash_shares", "aggregate_final_summary"} <= {error["rule"] for error in report["errors"]}


def test_screen_keeps_shadow_and_missing_dates_and_catches_gate_promotion(artifact_run):
    base, root = artifact_run
    training = json.loads((root / "reports/research.json").read_text())
    target = training["data_end"]
    roles = [{"role": role, "judgment": "shadow" if role == "ml" else "observe"} for role in ("czsc", "price_volume", "ml", "data_execution")]
    row = {"symbol": "000001.SZ", "category": "watch", "model_probability": .9, "model_validated": False,
           "data_end": target, "agent_evidence": roles, "model_run_id": root.name,
           "model_train_end": training["model_manifest"]["train_end"]}
    screen = {"rows": [row], "failures": [{"symbol": "000002.SZ", "error": "no verified target data"}],
              "coverage": {"requested": 2, "success": 1, "failed": 1}, "category_counts": {"watch": 1},
              "model_gate": {"passed": False}, "model_run_id": root.name, "data_range": {"requested_end": target}}
    path = base / "screen-test/reports/research-screen.json"
    write(path, screen)
    assert audit_run(root.name, "screen-test", runs_root=base)["passed"]
    screen["model_gate"]["passed"] = True
    screen["rows"][0].update(category="buy", model_validated=True, point_confirmed_at=target + "T15:00:00+08:00")
    screen["category_counts"] = {"buy": 1}
    screen["rows"][0]["agent_evidence"][-1]["judgment"] = "veto"
    write(path, screen)
    report = audit_run(root.name, "screen-test", runs_root=base)
    assert {"screen_frozen_model_gate", "screen_risk_veto", "screen_buy_has_confirmed_validated_evidence"} <= {error["rule"] for error in report["errors"]}
