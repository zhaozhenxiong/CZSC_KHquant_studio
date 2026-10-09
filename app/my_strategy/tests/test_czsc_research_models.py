"""Models and production qualification cannot travel backwards in time."""
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.adapters.czsc_adapter import SOURCE_SHA256
from my_strategy.core.tz import LOCAL_TZ
from my_strategy.services.czsc_research_features import FEATURE_COLUMNS, FEATURE_VERSION, FEATURE_SCHEMA_HASH
from my_strategy.services.czsc_research_ml import MODEL_VERSION, LABEL_VERSION
from my_strategy.services.czsc_research_models import ModelResolver, file_hash, model_hash
from my_strategy.services.czsc_research_profiles import get_feature_profile
from my_strategy.storage.czsc_model_releases import ModelReleaseStore


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def window(name, start, end):
    base = {"total_return": .01, "max_drawdown": .03, "trade_expectancy": .01, "completed_round_trips": 40}
    ml = {"total_return": .02, "max_drawdown": .02, "trade_expectancy": .02, "completed_round_trips": 40}
    return {"fold": {"name": name, "test_start": start, "test_end": end},
        "variants": {"czsc_price_volume": {"metrics": base}, "ml": {"metrics": ml}},
        "paired_daily_return_uncertainty": {"11": {"lower": .0001, "block_length": 11, "dates": 60, "confidence": .95}}}


@pytest.fixture
def models(tmp_path, monkeypatch):
    from my_strategy.storage import czsc_model_releases
    monkeypatch.setattr(czsc_model_releases, "local_now", lambda: datetime(2026, 10, 2, 16, tzinfo=LOCAL_TZ))
    runs, root = tmp_path / "runs", tmp_path / "runs" / "training"
    calendar = {"verified": True, "source": "verified_fixture", "dates": ["2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2026-09-30"]}
    save(runs / "calendar" / "reports" / "calendar.json", calendar)
    folds = [window("early", "2025-01-01", "2025-03-31"), window("middle", "2025-04-01", "2025-06-30"), window("late", "2025-07-01", "2025-09-30")]
    for name, available in [("early", "2024-12-31"), ("middle", "2025-03-31"), ("late", "2025-06-30"), ("production", "2025-09-30")]:
        directory = root / "models" / name
        directory.mkdir(parents=True)
        (directory / "model.pt").write_bytes(("checkpoint-" + name).encode())
        schema, preprocess = {"columns": list(FEATURE_COLUMNS)}, {"fit_scope": "training_only"}
        manifest = {"model_version": MODEL_VERSION, "label_version": LABEL_VERSION,
            "feature_schema_binding": "bound", "feature_version": FEATURE_VERSION, "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "schema": schema, "schema_sha256": model_hash(schema), "preprocess": preprocess, "preprocess_sha256": model_hash(preprocess),
            "checkpoint_sha256": file_hash(directory / "model.pt"), "data_version": "frozen_dataset",
            "label_contract": {"calendar_hash": stable_hash(calendar["dates"])},
            "available_at": available, "train_label_end": available, "validation_label_end": available, "validation_end": available}
        manifest["manifest_sha256"] = model_hash(manifest)
        save(directory / "manifest.json", manifest)
    training = {"run_id": "training", "strategy_version": "czsc_price_volume_mlp_v1", "data_version": "frozen_dataset",
        "calendar": {"run_id": "calendar", "hash": stable_hash(calendar)}, "config": {"probability_threshold": .55},
        "coverage": {"requested": 2, "success": 2, "failed": 0}, "failures": [], "evaluation": folds,
        "dataset_records": [{"symbol": "600000.SH"}, {"symbol": "000001.SZ"}],
        "model_gate": {"passed": True, "full_universe": True, "preparation_failures": 0, "windows": 3, "passing_windows": 3}}
    save(root / "reports" / "research.json", training)
    save(root / "metadata.json", {"created_at": "2025-10-01T17:00:00+08:00", "git_commit": "test-source", "stocks": ["600000.SH", "000001.SZ"]})
    save(root / "reports" / "execution_audit.json", {"training_run_id": "training", "passed": True, "error_count": 0, "errors": [], "frozen_stocks": 2})
    selected = json.loads((root / "models" / "production" / "manifest.json").read_text(encoding="utf-8"))
    bindings = {"training_run_id": "training", "checkpoint": "production", "manifest_sha256": selected["manifest_sha256"],
        "feature_schema_hash": FEATURE_SCHEMA_HASH, "strategy_version": training["strategy_version"],
        "research_config_sha256": stable_hash(training["config"]), "native_source_sha256": SOURCE_SHA256}
    freeze = runs / "freeze"
    save(freeze / "metadata.json", {"run_id": "freeze", "created_at": "2025-10-01T19:00:00+08:00", "scope": "implementation"})
    save(freeze / "reports" / "production_freeze.json", {**bindings, "stage": "frozen"})
    independent = []
    for name, start, end in [("I1", "2025-10-02", "2025-12-31"), ("I2", "2026-01-01", "2026-03-31"), ("I3", "2026-04-01", "2026-06-30")]:
        report = {**window(name, start, end), **bindings, "independent": True, "coverage": training["coverage"],
            "execution_audit": {"passed": True, "error_count": 0, "errors": [], "training_run_id": "training",
                "manifest_sha256": selected["manifest_sha256"], "frozen_stocks": 2, "audited_at": end + "T18:00:00+08:00"}}
        path = root / "reports" / (name + ".json")
        save(path, report)
        independent.append({"test_start": start, "test_end": end, "report_path": path.relative_to(runs).as_posix(), "sha256": file_hash(path)})
    manifest = json.loads((root / "models" / "production" / "manifest.json").read_text(encoding="utf-8"))
    certificate = {"passed": True, "independent": True, "training_run_id": "training", "checkpoint": "production",
        "completed_at": "2026-07-01T18:00:00+08:00", "training_completed_at": "2025-10-01T18:00:00+08:00",
        "frozen_before_evaluation_at": "2025-10-01T19:00:00+08:00", "independent_windows": independent,
        "manifest_sha256": manifest["manifest_sha256"], "research_sha256": file_hash(root / "reports" / "research.json"),
        "execution_audit_sha256": file_hash(root / "reports" / "execution_audit.json"), "freeze_run_id": "freeze",
        "freeze_metadata_sha256": file_hash(freeze / "metadata.json"), "freeze_policy_sha256": file_hash(freeze / "reports" / "production_freeze.json")}
    save(root / "reports" / "production_certification.json", certificate)
    return ModelReleaseStore(tmp_path / "results.db", runs), root, training, certificate


def _bind_ma_model_fixture(models):
    """Rebind every signed artifact, so profile tests reach semantic checks."""
    store, root, training, certificate = models
    profile = get_feature_profile("ma_trend_v1")
    training["strategy_version"] = profile["strategy_version"]
    training["config"]["feature_profile"] = profile["name"]
    for directory in (root / "models").iterdir():
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.pop("manifest_sha256")
        manifest.update(feature_profile=profile["name"], feature_version=profile["version"], feature_schema_hash=profile["schema_hash"])
        manifest["schema"]["columns"] = list(profile["columns"])
        manifest["schema_sha256"] = model_hash(manifest["schema"])
        manifest["manifest_sha256"] = model_hash(manifest)
        save(path, manifest)
    save(root / "reports" / "research.json", training)
    manifest = json.loads((root / "models" / "production" / "manifest.json").read_text(encoding="utf-8"))
    bindings = {"feature_profile": profile["name"], "feature_schema_hash": profile["schema_hash"],
                "strategy_version": profile["strategy_version"], "manifest_sha256": manifest["manifest_sha256"],
                "research_config_sha256": stable_hash(training["config"])}
    freeze_path = store.runs_root / "freeze" / "reports" / "production_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze.update(bindings)
    save(freeze_path, freeze)
    for window_record in certificate["independent_windows"]:
        path = store.runs_root / window_record["report_path"]
        report = json.loads(path.read_text(encoding="utf-8"))
        report.update(bindings)
        report["execution_audit"]["manifest_sha256"] = manifest["manifest_sha256"]
        save(path, report)
        window_record["sha256"] = file_hash(path)
    certificate.update(manifest_sha256=manifest["manifest_sha256"], research_sha256=file_hash(root / "reports" / "research.json"),
                       freeze_policy_sha256=file_hash(freeze_path))
    save(root / "reports" / "production_certification.json", certificate)
    return profile


@pytest.fixture
def ma_models(models):
    _bind_ma_model_fixture(models)
    return models


def test_ma_profile_routes_and_releases_without_borrowing_legacy_identity(ma_models):
    store, _, _, _ = ma_models
    profile = get_feature_profile("ma_trend_v1")
    arguments = {"feature_version": profile["version"], "feature_schema_hash": profile["schema_hash"],
                 "feature_columns": profile["columns"], "strategy_version": profile["strategy_version"],
                 "model_run_id": "training", "store": store}
    assert ModelResolver(model_run_id="training", store=store).resolve("2026-10-08")["model_dir"] is None
    assert ModelResolver(**arguments).resolve("2026-10-08")["status"] == "production_shadow"
    release = store.promote("training", reason="independent MA ledger certification passed")
    resolver = ModelResolver(**arguments)
    assert not resolver.resolve("2026-10-02")["applied_to_entry"]
    route = resolver.resolve("2026-10-08")
    assert route["applied_to_entry"] and route["release_id"] == release["release_id"]
    assert route["feature_profile"] == profile["name"]
    assert ModelResolver(model_run_id="training", store=store).resolve("2026-10-08")["model_dir"] is None
    arguments["feature_columns"] = profile["columns"][::-1]
    mismatched = ModelResolver(**arguments)
    assert mismatched.catalog_errors and mismatched.resolve("2026-10-08")["model_dir"] is None


@pytest.mark.parametrize("change", ["column_order", "schema_hash", "declared_profile", "strategy_version", "missing_strategy"])
def test_ma_signed_but_incompatible_schema_cannot_route_or_publish(ma_models, change):
    store, root, training, _ = ma_models
    profile = get_feature_profile("ma_trend_v1")
    path = root / "models" / "production" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256")
    if change == "column_order":
        manifest["schema"]["columns"].reverse()
        manifest["schema_sha256"] = model_hash(manifest["schema"])
    elif change == "schema_hash":
        manifest["feature_schema_hash"] = FEATURE_SCHEMA_HASH
    elif change == "declared_profile":
        training["config"]["feature_profile"] = "unknown"
    elif change == "missing_strategy":
        training.pop("strategy_version")
    else:
        training["strategy_version"] = "czsc_price_volume_mlp_v1"
    manifest["manifest_sha256"] = model_hash(manifest)
    save(path, manifest)
    save(root / "reports" / "research.json", training)
    resolver = ModelResolver(model_run_id="training", store=store, feature_version=profile["version"],
                             feature_schema_hash=profile["schema_hash"], feature_columns=profile["columns"])
    assert resolver.catalog_errors and resolver.resolve("2026-10-08")["model_dir"] is None
    with pytest.raises(ValueError, match="profile"):
        store.promote("training", reason="attempt")
    assert not store.events() and store.active() is None


@pytest.mark.parametrize("change", ["freeze_profile", "window_profile", "window_schema"])
def test_ma_certification_semantics_are_checked_after_all_artifact_hashes_match(ma_models, change):
    store, root, _, certificate = ma_models
    if change == "freeze_profile":
        path = store.runs_root / "freeze" / "reports" / "production_freeze.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["feature_profile"] = "legacy"
        save(path, value)
        certificate["freeze_policy_sha256"] = file_hash(path)
    else:
        window_record = certificate["independent_windows"][0]
        path = store.runs_root / window_record["report_path"]
        value = json.loads(path.read_text(encoding="utf-8"))
        value["feature_profile" if change == "window_profile" else "feature_schema_hash"] = "legacy"
        save(path, value)
        window_record["sha256"] = file_hash(path)
    save(root / "reports" / "production_certification.json", certificate)
    with pytest.raises(ValueError, match="bind|identity"):
        store.promote("training", reason="attempt")
    assert not store.events() and store.active() is None


def test_history_routes_by_data_date_and_does_not_apply_future_aggregate_gate(models):
    store, _, _, _ = models
    resolver = ModelResolver(usage_mode="historical", model_run_id="training", store=store)
    routes = resolver.resolve_dates(["2024-12-01", "2025-01-01", "2025-05-01", "2025-08-01", "2026-01-01"])
    assert [entry["checkpoint"] for entry in routes] == [None, "early", "middle", "late", "production"]
    assert not any(entry["applied_to_entry"] for entry in routes)
    assert routes[-1]["status"] == "historical_shadow"
    assert routes[-1]["actual_training_completed_at"] is None
    assert routes[-1]["actual_training_started_at"] == "2025-10-01T17:00:00+08:00"


def test_pinned_future_remains_explicitly_unavailable_and_retrospective_never_production(models):
    store, _, _, _ = models
    pinned = dict(model_policy="pinned", model_run_id="training", checkpoint="production", store=store)
    early = ModelResolver(usage_mode="historical", **pinned).resolve("2025-01-01")
    assert early["model_dir"] is None and early["reason_codes"] == ["pinned_model_unavailable"]
    retro = ModelResolver(usage_mode="retrospective", **pinned).resolve("2025-01-01")
    assert retro["model_dir"] and retro["retrospective_overlap"]
    assert not retro["applied_to_entry"]


def test_missing_incompatible_and_tampered_models_fail_closed(models):
    store, root, _, _ = models
    assert ModelResolver(model_run_id="absent", store=store).resolve("2026-09-30")["model_dir"] is None
    assert ModelResolver(model_run_id="training", feature_schema_hash="wrong", store=store).resolve("2026-09-30")["model_dir"] is None
    (root / "models" / "production" / "model.pt").write_bytes(b"tampered")
    resolver = ModelResolver(model_run_id="training", store=store)
    assert resolver.catalog_errors and resolver.resolve("2026-09-30")["model_dir"] is None


def test_real_release_timestamp_is_not_retroactive_and_restart_preserves_identity(models):
    store, _, _, _ = models
    release = store.promote("training", reason="independent ledger certification passed")
    resolver = ModelResolver(model_run_id="training", store=store)
    assert not resolver.resolve("2026-10-02")["applied_to_entry"]  # daily close is 15:00, release 16:00
    route = resolver.resolve("2026-10-08")
    assert route["applied_to_entry"] and route["release_id"] == release["release_id"]
    reopened = ModelReleaseStore(store.db_path, store.runs_root)
    assert reopened.active()["release_id"] == release["release_id"]
    assert reopened.active("2026-09-30") is None
    historic = ModelResolver(usage_mode="historical", model_run_id="training", store=reopened)
    assert not historic.resolve("2026-10-02")["applied_to_entry"]
    assert historic.resolve("2026-10-08")["applied_to_entry"]
    assert historic.resolve("2026-10-08")["status"] == "historical_released"


def test_history_cannot_inherit_release_from_another_checkpoint_and_retrospective_is_shadow(models):
    store, _, _, _ = models
    store.promote("training", reason="certified")
    historical = ModelResolver(usage_mode="historical", model_policy="pinned", model_run_id="training", checkpoint="middle", store=store)
    assert not historical.resolve("2026-10-08")["applied_to_entry"]
    retro = ModelResolver(usage_mode="retrospective", model_policy="pinned", model_run_id="training", checkpoint="production", store=store)
    assert not retro.resolve("2026-10-08")["applied_to_entry"]


def test_history_latest_candidate_cannot_borrow_older_active_release(models):
    store, root, training, _ = models
    store.promote("training", reason="certified")
    newer_root = root.parent / "newer"
    shutil.copytree(root, newer_root)
    training["run_id"] = "newer"
    training["model_gate"]["passed"] = False
    save(newer_root / "reports" / "research.json", training)
    path = newer_root / "models" / "production" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256")
    manifest.update(available_at="2026-09-30", validation_end="2026-09-30", validation_label_end="2026-09-30")
    manifest["manifest_sha256"] = model_hash(manifest)
    save(path, manifest)
    historical = ModelResolver(usage_mode="historical", store=store).resolve("2026-10-08")
    assert historical["model_run_id"] == "newer" and not historical["applied_to_entry"]
    production = ModelResolver(usage_mode="production", store=store).resolve("2026-10-08")
    assert production["model_run_id"] == "training" and production["applied_to_entry"]


def test_history_release_withdrawal_is_dated_and_integrity_failure_is_closed(models, monkeypatch):
    from my_strategy.storage import czsc_model_releases
    store, root, _, _ = models
    release = store.promote("training", reason="certified")
    monkeypatch.setattr(czsc_model_releases, "local_now", lambda: datetime(2026, 10, 3, 12, tzinfo=LOCAL_TZ))
    store.rollback(reason="withdraw")
    resolver = ModelResolver(usage_mode="historical", model_run_id="training", store=store)
    assert resolver.resolve("2026-10-02T17:00:00+08:00")["applied_to_entry"]
    assert not resolver.resolve("2026-10-03")["applied_to_entry"]
    store.rollback(release["release_id"], reason="restore")
    (root / "reports" / "I1.json").write_text("{}", encoding="utf-8")
    tampered = ModelResolver(usage_mode="historical", model_run_id="training", store=store).resolve("2026-10-08")
    assert not tampered["applied_to_entry"]
    assert tampered["reason_codes"] == ["production_release_integrity_failure"]


def test_failed_gate_or_forged_aggregate_metrics_never_updates_alias(models):
    store, root, training, _ = models
    training["model_gate"]["passed"] = False
    save(root / "reports" / "research.json", training)
    with pytest.raises(ValueError, match="complete-universe"):
        store.promote("training", reason="attempt")
    training["model_gate"]["passed"] = True
    training["evaluation"][0]["variants"]["ml"]["metrics"]["completed_round_trips"] = 2
    save(root / "reports" / "research.json", training)
    with pytest.raises(ValueError, match="recomputation"):
        store.promote("training", reason="attempt")
    assert store.active() is None and not store.events()


@pytest.mark.parametrize("change", ["failed_audit", "missing_certificate", "wrong_hash", "backdated_freeze", "future_completion", "duplicate_window", "failed_independent_metrics", "wrong_window_source", "failed_window_audit", "false_freeze_timestamp"])
@pytest.mark.parametrize("profile_name", ["legacy", "ma_trend_v1"])
def test_certification_requires_complete_current_and_independent_evidence(models, change, profile_name):
    if profile_name == "ma_trend_v1":
        _bind_ma_model_fixture(models)
    store, root, _, certificate = models
    cert_path = root / "reports" / "production_certification.json"
    if change == "failed_audit":
        save(root / "reports" / "execution_audit.json", {"passed": False})
    elif change == "missing_certificate":
        cert_path.unlink()
    elif change == "wrong_hash":
        certificate["research_sha256"] = "bad"
    elif change == "backdated_freeze":
        certificate["frozen_before_evaluation_at"] = "2025-01-01T19:00:00+08:00"
    elif change == "future_completion":
        certificate["completed_at"] = "2027-01-01T18:00:00+08:00"
    elif change == "duplicate_window":
        certificate["independent_windows"][1] = copy.deepcopy(certificate["independent_windows"][0])
    elif change == "false_freeze_timestamp":
        freeze_metadata = store.runs_root / "freeze" / "metadata.json"
        save(freeze_metadata, {"run_id": "freeze", "created_at": "2025-10-02T20:00:00+08:00", "scope": "implementation"})
        certificate["freeze_metadata_sha256"] = file_hash(freeze_metadata)
    else:
        path = store.runs_root / certificate["independent_windows"][0]["report_path"]
        report = json.loads(path.read_text(encoding="utf-8"))
        if change == "wrong_window_source":
            report["native_source_sha256"] = "different vendor"
        elif change == "failed_window_audit":
            report["execution_audit"]["passed"] = False
        else:
            report["paired_daily_return_uncertainty"]["11"]["lower"] = -.01
        save(path, report)
        certificate["independent_windows"][0]["sha256"] = file_hash(path)
    if change not in {"failed_audit", "missing_certificate"}:
        save(cert_path, certificate)
    with pytest.raises((ValueError, KeyError)):
        store.promote("training", reason="attempt")
    assert not store.events() and store.active() is None


def test_release_events_are_immutable_rollback_appends_and_tamper_blocks_reactivation(models):
    store, root, _, _ = models
    release = store.promote("training", reason="certified")
    store.rollback(reason="withdraw")
    assert store.active() is None and len(store.events()) == 2
    store.rollback(release["release_id"], reason="restore certified version")
    assert store.active()["release_id"] == release["release_id"] and len(store.events()) == 3
    with sqlite3.connect(store.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE czsc_model_release_events SET event_type='other'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("DELETE FROM czsc_model_release_events")
    (root / "models" / "production" / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact integrity"):
        store.rollback(release["release_id"], reason="tampered restore")
    assert len(store.events()) == 3


def test_alias_failure_rolls_back_event_in_same_transaction(models):
    store, _, _, _ = models
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("CREATE TRIGGER deny_alias BEFORE INSERT ON czsc_model_aliases BEGIN SELECT RAISE(ABORT,'alias blocked'); END")
    with pytest.raises(sqlite3.IntegrityError, match="alias blocked"):
        store.promote("training", reason="attempt")
    assert not store.events() and store.active() is None


@pytest.mark.parametrize("identity", ["../x", "x/y", "..", "/x"])
def test_model_identity_cannot_escape_run_storage(models, identity):
    store, _, _, _ = models
    with pytest.raises(ValueError, match="identity"):
        ModelResolver(model_run_id=identity, store=store)
