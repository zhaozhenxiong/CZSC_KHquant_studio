"""Independent audit rejects future scores and fictitious holding targets."""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys

import numpy as np
import pandas as pd

from my_strategy.scripts.audit_czsc_dual_research import (
    _check_bundle, _check_exit_dataset, _check_exit_reference, _check_model, _check_oof,
    _check_training_preprocessing, _completed_dual_identity, _execution_prices, _round_trip_metrics,
)
from my_strategy.scripts.audit_czsc_research import _file_hash, _hash


def _oof():
    return pd.DataFrame({"symbol": ["000001.SZ", "000002.SZ"], "date": ["2025-01-02", "2025-01-03"],
                         "label": [1, 0], "label_end": ["2025-01-17", "2025-01-20"],
                         "p_ma": [.6, .4], "p_structure": [.7, .3], "expert_train_end": "2024-09-30",
                         "expert_validation_end": "2024-12-31", "expert_available_at": "2024-12-31", "oof_fold": "forward"})


def _exit():
    return pd.DataFrame({"symbol": ["000001.SZ", "000001.SZ"], "date": ["2025-01-02", "2025-01-03"],
                         "actual_shares": [100, 100], "actual_entry_date": "2024-12-31", "actual_cost_per_share": [10.05, 10.05],
                         "snapshot_available_at": ["2025-01-02T15:00:00+08:00", "2025-01-03T15:00:00+08:00"], "ledger_entries_seen": [1, 1],
                         "forced_net_proceeds": [1100., np.nan], "continuation_net_proceeds": [1090., np.nan],
                         "forced_exit_date": ["2025-01-03", None], "continuation_exit_date": ["2025-01-06", None],
                         "label": [1., np.nan], "label_available": [True, False], "label_end": ["2025-01-06", None]})


def test_forward_oof_with_mature_target_and_held_actual_states_pass():
    assert not _check_oof(_oof(), "2025-03-31", fold="outer")
    assert not _check_exit_dataset(_exit())


def test_task_manager_success_still_requires_complete_matching_run_report():
    report = {"run_id": "dual-unit", "status": "complete"}
    for status in ("complete", "succeeded"):
        metadata = {"run_id": "dual-unit", "status": status}
        assert _completed_dual_identity(report, metadata, "dual-unit")
        assert not _completed_dual_identity({**report, "status": "running"}, metadata, "dual-unit")
        assert not _completed_dual_identity({"run_id": "dual-unit"}, metadata, "dual-unit")
        assert not _completed_dual_identity(report, {**metadata, "run_id": "other"}, "dual-unit")
    for status in ("running", "failed", "cancelled", "queued", None):
        assert not _completed_dual_identity(report, {"run_id": "dual-unit", "status": status}, "dual-unit")


def test_oof_rejects_in_sample_prediction_future_maturity_and_missing_provenance():
    frame = _oof()
    frame.loc[0, "expert_available_at"] = frame.loc[0, "date"]
    frame.loc[1, "label_end"] = "2025-04-01"
    result = _check_oof(frame, "2025-03-31", fold="outer")
    assert {x["rule"] for x in result} == {"oof_forward_prediction_and_label_maturity"}
    assert result[0]["invalid_rows"] == 2
    assert _check_oof(frame.drop(columns="expert_train_end"), "2025-03-31", fold="outer")[0]["rule"] == "oof_required_provenance"


def test_exit_rejects_theoretical_inventory_and_entry_probability_as_label():
    frame = _exit()
    frame.loc[0, "actual_shares"] = 0
    frame.loc[0, "label"] = 0
    frame.loc[0, "label_end"] = frame.loc[0, "forced_exit_date"]
    assert {x["rule"] for x in _check_exit_dataset(frame)} == {
        "exit_actual_held_positive_cost_state", "exit_label_compares_net_remaining_share_proceeds", "exit_label_two_path_full_exit_maturity"}


def test_round_trip_metrics_include_partial_sell_fees_and_exclude_open_position():
    ledger = pd.DataFrame([
        {"action": "BUY", "shares": 200, "cash_flow": -2005.},
        {"action": "REDUCE", "shares": 100, "cash_flow": 1195.},
        {"action": "SELL", "shares": 100, "cash_flow": 995.},
        {"action": "BUY", "shares": 100, "cash_flow": -1005.},
    ])
    result = _round_trip_metrics(ledger)
    assert result["completed_round_trips"] == 1
    assert result["win_rate"] == 1
    assert result["net_pnl"] == [185.]
    assert np.isclose(result["trade_expectancy"], 2190. / 2005. - 1)


def test_hash_only_model_audit_detects_checkpoint_and_future_label(tmp_path):
    root = tmp_path / "run"
    directory = root / "models/expert"
    directory.mkdir(parents=True)
    checkpoint = directory / "model.pt"
    checkpoint.write_bytes(b"no-deserialization-needed")
    schema, preprocess = {"columns": ["x"]}, {"median": [0.]}
    manifest = {"schema": schema, "schema_sha256": _hash(schema), "preprocess": preprocess, "preprocess_sha256": _hash(preprocess),
                "checkpoint": str(checkpoint), "checkpoint_sha256": _file_hash(checkpoint), "feature_schema_binding": "bound",
                "feature_version": "unit", "feature_schema_hash": "unit-hash", "train_end": "2024-09-30",
                "train_label_end": "2024-09-27", "validation_start": "2024-10-01", "validation_end": "2024-12-31",
                "validation_label_end": "2024-12-30", "available_at": "2024-12-31", "sample_counts": {"train": 2, "validation": 2},
                "calibration": {"fit_start": "2024-10-01", "fit_end": "2024-12-31"},
                "training_started_at": "2026-10-09T15:00:00+08:00", "training_completed_at": "2026-10-09T15:01:00+08:00",
                "promotion_status": "candidate"}
    manifest["manifest_sha256"] = _hash(manifest)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert not _check_model(path, root)["errors"]
    checkpoint.write_bytes(b"modified")
    manifest["train_label_end"] = "2024-10-01"
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = _hash(manifest)
    path.write_text(json.dumps(manifest))
    assert {x["rule"] for x in _check_model(path, root)["errors"]} == {"model_checkpoint_sha256_or_path", "model_purged_time_boundaries"}
    result = subprocess.run([sys.executable, "-c", "import sys; import my_strategy.scripts.audit_czsc_dual_research; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_bundle_audit_rejects_rehashed_future_oof_and_target_substitution(tmp_path):
    from my_strategy.services.czsc_research_profiles import get_feature_profile
    root = tmp_path / "run"
    directory = root / "models/outer"
    target = {"label_version": "czsc_fixed_horizon_broker_v1", "horizon": 10}
    def model(location, name, inner=False):
        location.mkdir(parents=True)
        (location / "model.pt").write_bytes(b"frozen-synthetic-model")
        profile = get_feature_profile("ma_trend_v1" if name == "ma" else "czsc_structure_v1") if name != "fusion" else {"version": "dual_probability_oof_v1", "schema_hash": "fusion-hash", "columns": ["p_ma", "p_structure"]}
        schema = {"columns": profile["columns"]}
        preprocess = {"median": [0.] * len(schema["columns"])}
        value = {"feature_version": profile["version"], "feature_schema_hash": profile["schema_hash"], "feature_schema_binding": "bound",
                 "schema": schema, "schema_sha256": _hash(schema), "preprocess": preprocess, "preprocess_sha256": _hash(preprocess),
                 "checkpoint": str(location / "model.pt"), "checkpoint_sha256": _file_hash(location / "model.pt"),
                 "train_end": "2024-09-30" if inner else "2025-03-31", "train_label_end": "2024-09-27" if inner else "2025-01-20",
                 "validation_start": "2024-10-01" if inner else "2025-04-01", "validation_end": "2024-12-31" if inner else "2025-06-30",
                 "validation_label_end": "2024-12-30" if inner else "2025-06-27", "available_at": "2024-12-31" if inner else "2025-06-30",
                 "calibration": {"fit_start": "2024-10-01" if inner else "2025-04-01", "fit_end": "2024-12-31" if inner else "2025-06-30"},
                 "label_contract": target, "sample_counts": {"train": 2, "validation": 2}, "training_started_at": "2026-10-09T15:00:00+08:00",
                 "training_completed_at": "2026-10-09T15:01:00+08:00", "promotion_status": "candidate"}
        value["manifest_sha256"] = _hash(value)
        (location / "manifest.json").write_text(json.dumps(value))
        return value
    experts = {name: model(directory / "experts" / name, name) for name in ("ma", "structure")}
    fusion = model(directory / "fusion", "fusion")
    shared = root / "models/oof-shared/forward/cache-identity"
    inner = {name: model(shared / "experts" / name, name, True) for name in ("ma", "structure")}
    oof = _oof()
    oof["label_available"] = True
    oof_path = directory / "oof/oof.parquet"
    oof_path.parent.mkdir(parents=True)
    oof.to_parquet(oof_path, index=False)
    def binding(location, value):
        return {"path": str(location), "manifest_file_sha256": _file_hash(location / "manifest.json"), "manifest_sha256": value["manifest_sha256"],
                "checkpoint_sha256": value["checkpoint_sha256"], "preprocess_sha256": value["preprocess_sha256"]}
    split = {"name": "forward", "train_end": "2024-09-30", "validation_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-03-31"}
    cached_path = shared / "predictions.parquet"
    oof.to_parquet(cached_path, index=False)
    header = {"identity": shared.name, "fold": split, "experts": inner, "rows_path": str(cached_path), "rows_sha256": _file_hash(cached_path)}
    header["cache_sha256"] = _hash(header)
    (shared / "cache.json").write_text(json.dumps(header))
    threshold = {"threshold": .55, "method": "validation_fixed_horizon_mean_net_return_proxy", "qualification": "fitted_validation_diagnostic_not_strategy_acceptance"}
    bundle = {"version": "czsc_dual_experts_v1", "status": "shadow", "publication_allowed": False, "target": target, "target_hash": _hash(target),
              "train_end": "2025-03-31", "train_label_end": "2025-01-20", "validation_start": "2025-04-01", "validation_end": "2025-06-30", "available_at": "2025-06-30",
              "experts": {name: binding(directory / "experts" / name, value) for name, value in experts.items()}, "fusion": binding(directory / "fusion", fusion),
              "thresholds": {name: threshold for name in ("ma", "structure", "fusion")},
              "oof": {"path": str(oof_path), "sha256": _file_hash(oof_path), "rows": len(oof), "label_end_max": "2025-01-20", "folds": [{**header, "reused": False}]}}
    def save():
        bundle.pop("bundle_sha256", None)
        bundle["bundle_sha256"] = _hash(bundle)
        (directory / "bundle.json").write_text(json.dumps(bundle))
    save()
    training = oof[["symbol", "date", "label", "label_end"]].copy()
    assert not _check_bundle(directory, root, training)["errors"]
    oof.loc[0, "expert_available_at"] = "2025-01-02"
    oof.to_parquet(oof_path, index=False)
    bundle["oof"]["sha256"] = _file_hash(oof_path)
    save()
    rules = {x["rule"] for x in _check_bundle(directory, root, training)["errors"]}
    assert {"oof_forward_prediction_and_label_maturity", "oof_recorded_fold_boundaries"} <= rules
    bundle["target"]["horizon"] = 1
    bundle["target_hash"] = _hash(bundle["target"])
    save()
    assert "bundle_fixed_ten_session_target" in {x["rule"] for x in _check_bundle(directory, root, training)["errors"]}


def test_preprocessing_refit_excludes_validation_and_future_maturing_rows():
    frame = pd.DataFrame({"x": [0., 2., 1000., 2000.], "date": ["2024-09-01", "2024-09-02", "2024-10-01", "2024-10-02"],
                          "label": [0, 1, 0, 1], "label_end": ["2024-09-15", "2024-09-16", "2024-10-16", "2025-04-01"],
                          "label_available": True, "input_eligible": True})
    manifest = {"feature_version": "test", "train_end": "2024-09-30", "validation_start": "2024-10-01", "validation_end": "2024-12-31",
                "sample_counts": {"train": 2, "validation": 1}, "schema": {"columns": ["x"]}, "checkpoint": "test-model",
                "train_signal_start": "2024-09-01", "train_signal_end": "2024-09-02", "train_label_end": "2024-09-16", "validation_label_end": "2024-10-16",
                "preprocess": {"median": [1.], "mean": [1.], "scale": [1.], "all_missing": [False], "fit_scope": "purged_training_only", "clip_standard_deviations": 8.},
                "dataset_sha256": hashlib.sha256(pd.util.hash_pandas_object(frame.iloc[:3], index=True).to_numpy().tobytes()).hexdigest()}
    assert not _check_training_preprocessing(frame, manifest)
    frame.loc[3, "x"] = 9e10
    assert not _check_training_preprocessing(frame, manifest)
    manifest["preprocess"]["median"] = [1000.]
    assert "model_training_only_preprocess_refit" in {x["rule"] for x in _check_training_preprocessing(frame, manifest)}


def test_actual_reference_replays_state_forced_sell_outcome_and_trade_fees(tmp_path):
    root = tmp_path / "run"
    path = root / "evaluation/exit_reference/000001_SZ"
    path.mkdir(parents=True)
    dates = pd.bdate_range("2024-01-02", periods=65).strftime("%Y-%m-%d").tolist()
    symbol = "000001.SZ"
    raw = pd.DataFrame({"symbol": symbol, "date": dates, "open": 10., "close": 10., "volume": 100000., "amount": 1000000., "has_trade_price": 1, "source": "baostock"})
    raw.loc[62:64, ["open", "close"]] = [[11., 11.], [12., 12.], [13., 13.]]
    buy_flow, sell_flow = -1005., 1293.3505
    ledger = pd.DataFrame([
        {"date": dates[61], "signal_date": dates[60], "symbol": symbol, "action": "BUY", "price": 10., "shares": 100, "fee": 5., "cash_flow": buy_flow, "position_before": 0, "position_after": 100, "cash_after": 8995.},
        {"date": dates[64], "signal_date": dates[63], "symbol": symbol, "action": "SELL", "price": 12.99, "shares": 100, "fee": 5.6495, "cash_flow": sell_flow, "position_before": 100, "position_after": 0, "cash_after": 10288.3505}])
    ledger.to_csv(path / "ledger.csv", index=False)
    daily = pd.DataFrame({"date": dates[61:], "cash": [8995., 8995., 8995., 10288.3505], "shares": [100, 100, 100, 0], "close": [10., 11., 12., 13.], "equity": [9995., 10095., 10195., 10288.3505]})
    daily.to_csv(path / "daily.csv", index=False)
    (path / "rejections.csv").write_text("\n")
    samples = pd.DataFrame({"symbol": symbol, "date": dates[61:64], "actual_shares": 100, "actual_cost_per_share": 10.05,
                            "actual_entry_date": dates[61], "ledger_entries_seen": 1, "label_available": True,
                            "forced_net_proceeds": [1093.4505, 1193.4005, 1293.3505], "continuation_net_proceeds": sell_flow,
                            "forced_exit_date": dates[62:65], "continuation_exit_date": dates[64], "forced_rejection_count": 0})
    settings = {"commission": .0003, "stamp_tax": .0005, "min_commission": 5., "slippage": .0005,
                "seasoned_bars": 60, "mainboard_conservative_limit": .05, "max_volume_participation": .01, "max_weight": .95}
    assert not _check_exit_reference(path, samples, raw, 10000., dates, settings)["errors"]
    source = root / "dataset/source.parquet"
    source.parent.mkdir(parents=True)
    raw.to_parquet(source, index=False)
    record = {"symbol": symbol, "path": str(source)}
    assert not _execution_prices(record, root, [], settings, dates)
    samples.loc[0, "actual_cost_per_share"] = 10.
    samples.loc[1, "forced_net_proceeds"] += 5.
    assert {x["rule"] for x in _check_exit_reference(path, samples, raw, 10000., dates, settings)["errors"]} == {"exit_actual_state_matches_prior_ledger", "exit_two_path_net_proceeds_replayed"}
    ledger.loc[1, "fee"] = 0.
    ledger.to_csv(path / "ledger.csv", index=False)
    assert "execution_actual_open_slippage_and_fees" in {x["rule"] for x in _execution_prices(record, root, [], settings, dates)}
