"""Development audit detects ledger tampering and dropped failed-account capital."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.scripts.audit_czsc_research import MODES, _file_hash, _hash
from my_strategy.scripts.audit_czsc_candidate_study import audit_study


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.fixture
def candidate_run(tmp_path):
    parent, study = tmp_path / "parent", tmp_path / "study"
    dates = pd.bdate_range("2024-01-02", periods=12).strftime("%Y-%m-%d").tolist()
    calendar = {"verified": True, "source": "synthetic", "dates": dates}
    calendar_path = tmp_path / "calendar/reports/calendar.json"
    write(calendar_path, calendar)
    fold = {"name": "fixture", "train_end": dates[2], "validation_start": dates[3], "validation_end": dates[5],
            "test_start": dates[6], "test_end": dates[9]}
    config = {"feature_start": dates[0], "horizon": 1, "initial_cash": 10000, "probability_threshold": .55,
              "folds": [fold], "production": {"train_end": dates[5], "validation_start": dates[6], "validation_end": dates[8]}}
    contract = {"horizon": 1, "initial_cash": 10000, "calendar_hash": stable_hash(dates)}
    frame = pd.DataFrame({"symbol": "000001.SZ", "date": dates, "label": np.nan, "label_end": None,
                          "label_available": False, "input_eligible": True, "rule_buy": False,
                          "label_reason": "label_immature", "entry_date": None, "exit_date": None,
                          "net_return": np.nan, "x": np.arange(12, dtype=float)})
    for index in (0, 3, 6):
        frame.loc[index, ["label", "label_end", "label_available", "rule_buy", "label_reason", "entry_date", "exit_date", "net_return"]] = [
            float(index != 0), dates[index + 2], True, True, "available", dates[index + 1], dates[index + 2], .02 if index else -.02]
    frame.attrs.update(feature_version="synthetic-v1", schema_hash="synthetic-schema", data_version="parent-data")
    snapshot = parent / "dataset/000001_SZ.parquet"
    snapshot.parent.mkdir(parents=True)
    frame.to_parquet(snapshot, index=False)
    record = {"symbol": "000001.SZ", "path": str(snapshot), "sha256": _file_hash(snapshot), "bars": 12,
              "data_end": dates[-1], "data_version": "parent-data", "feature_schema_hash": "synthetic-schema"}
    schema = {"feature_version": "synthetic-v1", "feature_schema_hash": "synthetic-schema", "schema": {"columns": ["x"]}, "label_contract": contract}
    stocks = ["000001.SZ", "000002.SZ"]
    training = {"run_id": parent.name, "data_end": dates[-1], "data_version": "parent-data", "config": config,
                "model_manifest": schema, "run_context": {"stocks": stocks}, "dataset_records": [record],
                "failures": [{"symbol": "000002.SZ", "error": "synthetic missing input"}],
                "coverage": {"requested": 2, "success": 1, "failed": 1}, "calendar": {"path": str(calendar_path), "hash": stable_hash(calendar)}}
    write(parent / "reports/research.json", training)
    write(parent / "metadata.json", {"status": "complete", "stocks": stocks, "config_hash": stable_hash(config), "research_config": config})
    protocol = {"stage": "development_only", "parent_run_id": parent.name, "unchanged_label_horizon": 1,
                "known_test_windows_are_development_diagnostics": True, "seed": 42, "epochs": 1, "threshold_grid": [.5, .55]}
    write(study / "reports/protocol.json", protocol)
    columns = ["symbol", "date", "label", "label_end", "label_available", "input_eligible", "rule_buy", "net_return", "x"]
    data = frame.loc[frame.label_available, columns].copy().reset_index(drop=True)
    data.attrs.update(feature_version="synthetic-v1", schema_hash="synthetic-schema", label_contract=contract,
                      data_version=stable_hash({"parent": "parent-data", "population": "eligible_rule_buy"}))
    (study / "dataset").mkdir(parents=True)
    data.to_parquet(study / "dataset/entry_candidates.parquet", index=False)
    checkpoints = []
    for settings in (fold, {**config["production"], "name": "production"}):
        name = settings["name"]
        path = study / "models" / name / "model.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"synthetic-weight-file")
        train = data[data.label_end <= settings["train_end"]]
        validation = data[(data.date >= settings["validation_start"]) & (data.label_end <= settings["validation_end"])]
        preprocess = {"fit_scope": "purged_training_only", "median": [0.]}
        manifest = {**schema, "schema_sha256": _hash(schema["schema"]), "preprocess": preprocess,
                    "preprocess_sha256": _hash(preprocess), "feature_schema_binding": "bound", "checkpoint": str(path),
                    "checkpoint_sha256": _file_hash(path), "train_end": settings["train_end"], "train_label_end": train.label_end.max(),
                    "validation_start": settings["validation_start"], "validation_end": settings["validation_end"],
                    "validation_label_end": validation.label_end.max(), "available_at": settings["validation_end"],
                    "calibration": {"fit_start": settings["validation_start"], "fit_end": settings["validation_end"]},
                    "sample_counts": {"train": len(train), "validation": len(validation)}, "device": "synthetic-no-gpu",
                    "seed": 42, "hyperparameters": {"epochs": 1}, "data_version": data.attrs["data_version"]}
        manifest["manifest_sha256"] = _hash(manifest)
        write(path.parent / "manifest.json", manifest)
        checkpoints.append({"name": name, "manifest": manifest, "threshold_used_for_development_ledger": .55,
                            "validation_threshold_grid": [{"threshold": value, "rows": 1} for value in protocol["threshold_grid"]]})
    ledger = pd.DataFrame([
        {"date": dates[6], "signal_date": dates[5], "symbol": "000001.SZ", "action": "BUY", "price": 10., "shares": 100,
         "fee": 5., "cash_flow": -1005., "position_before": 0, "position_after": 100, "cash_after": 8995.},
        {"date": dates[8], "signal_date": dates[7], "symbol": "000001.SZ", "action": "SELL", "price": 12., "shares": 100,
         "fee": 5.6, "cash_flow": 1194.4, "position_before": 100, "position_after": 0, "cash_after": 10189.4}])
    daily = pd.DataFrame({"date": dates[6:10], "cash": [8995., 8995., 10189.4, 10189.4], "shares": [100, 100, 0, 0],
                          "close": [11., 12., 12., 12.], "equity": [10095., 10195., 10189.4, 10189.4]})
    aggregate = pd.DataFrame({"date": dates[6:10], "equity": daily.equity + 10000})
    metrics = {"final_equity": 20189.4, "total_return": 20189.4 / 20000 - 1, "net_return": 20189.4 / 20000 - 1, "fees": 10.6, "trade_count": 2}
    for mode in MODES:
        location = study / "evaluation/fixture" / mode / "000001_SZ"
        location.mkdir(parents=True)
        ledger.to_csv(location / "ledger.csv", index=False)
        daily.to_csv(location / "daily.csv", index=False)
        (location / "rejections.csv").write_text("\n", encoding="utf-8")
        aggregate.to_csv(location.parent / "aggregate_daily.csv", index=False)
    report = {"study_run_id": study.name, "parent_run_id": parent.name, "status": "development_only", "production_eligible": False,
              "independent_certification": False, "known_test_windows": "previously inspected; diagnostic only", "candidate_rows": 3,
              "requested_stock_count": 2, "hashed_records": 1, "parent_research_sha256": _file_hash(parent / "reports/research.json"),
              "protocol_sha256": _file_hash(study / "reports/protocol.json"), "checkpoints": checkpoints,
              "evaluation": [{"fold": fold, "frozen_mainboard_accounts": 2, "unavailable_accounts_cash_retained": [{"symbol": "000002.SZ"}],
                              "variants": {mode: {"accounts": 1, "metrics": metrics} for mode in MODES}}]}
    write(study / "reports/study.json", report)
    write(study / "metadata.json", {"status": "development_study_complete", "protocol": protocol, "config_hash": stable_hash(protocol), "candidate_rows": 3})
    return tmp_path, study


def test_development_audit_retains_failed_budget_without_model_registration(candidate_run):
    base, study = candidate_run
    report = audit_study(study.name, runs_root=base, workers=2)
    assert report["passed"], report["errors"]
    assert report["diagnostic_only"] and not report["independent_certification"] and not report["production_eligible"]
    assert report["dataset"]["candidate_rows"] == 3
    assert report["evaluation"][0]["modes"]["ml"]["failed_cash_budget"] == 10000
    assert (study / "reports/development_execution_audit.json").is_file()
    assert not (study / "reports/research.json").exists()
    result = subprocess.run([sys.executable, "-c", "import sys; import my_strategy.scripts.audit_czsc_candidate_study; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_ledger_tampering_and_dropped_failed_capital_are_detected(candidate_run):
    base, study = candidate_run
    location = study / "evaluation/fixture/ml/000001_SZ"
    ledger = pd.read_csv(location / "ledger.csv")
    ledger.loc[0, "signal_date"] = ledger.loc[0, "date"]
    ledger.loc[0, "cash_flow"] -= 100
    ledger.to_csv(location / "ledger.csv", index=False)
    aggregate = pd.read_csv(location.parent / "aggregate_daily.csv")
    aggregate.equity -= 10000
    aggregate.to_csv(location.parent / "aggregate_daily.csv", index=False)
    report = audit_study(study.name, runs_root=base)
    assert not report["passed"]
    assert {"ledger_next_verified_session", "ledger_cash_flow_or_position_chain", "aggregate_daily_includes_failed_cash"} <= {error["rule"] for error in report["errors"]}


def test_candidate_population_tampering_and_path_escape_are_detected(candidate_run):
    base, study = candidate_run
    path = study / "dataset/entry_candidates.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "rule_buy"] = False
    frame.to_parquet(path, index=False)
    report = audit_study(study.name, runs_root=base)
    assert {"candidate_population_matches_frozen_rule_buy", "candidate_eligibility_labels_or_counts"} <= {error["rule"] for error in report["errors"]}
    with pytest.raises(ValueError, match="invalid run identity"):
        audit_study("../study", runs_root=base)
    with pytest.raises(ValueError, match="workers"):
        audit_study(study.name, runs_root=base, workers=9)
