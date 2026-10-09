"""Portable CZSC packages retain committed data and exclude retired payloads."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from my_strategy.scripts.package_project import build_package, main, verify_package
from my_strategy.scripts.snapshot_sqlite_databases import CANONICAL_DATABASE_PATHS, runtime_layout
from my_strategy.storage.personal_portfolio import PersonalStore
from my_strategy.core.run_context import stable_hash
from my_strategy.scripts.migration_manifest import file_sha256, write_integrity_manifest
from my_strategy.services.czsc_research_models import checkpoint_catalog, model_hash


def source_repo(root: Path) -> Path:
    repo = root / "source"
    app = repo / "app"
    (app / "my_strategy/core").mkdir(parents=True)
    (app / "my_strategy/core/paths.py").write_text("# marker", encoding="utf-8")
    (app / "AGENTS.md").write_text("# source project instructions", encoding="utf-8")
    (repo / "install.py").write_text("# installer", encoding="utf-8")
    (app / "vendor/czsc").mkdir(parents=True)
    (app / "vendor/czsc/LICENSE").write_text("source license", encoding="utf-8")
    return repo


def test_package_only_current_databases_and_czsc_runs(tmp_path):
    repo = source_repo(tmp_path)
    app = repo / "app"
    for relative in CANONICAL_DATABASE_PATHS:
        database = app / relative
        database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("CREATE TABLE evidence (value TEXT)")
            connection.execute("INSERT INTO evidence VALUES ('current')")
            connection.commit()
    for relative in ("my_strategy/data/processed/warehouse/old.db", "my_strategy/artifacts/vector_datasets/old.csv",
                     ".scratch/secret.txt", "my_strategy/__pycache__/old.pyc", ".venv/lib/old.txt"):
        path = app / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retired", encoding="utf-8")
    run = app / "my_strategy/artifacts/runs/current/reports"
    run.mkdir(parents=True)
    (run / "czsc.json").write_text(json.dumps({"run_id": "current"}), encoding="utf-8")
    old_run = app / "my_strategy/artifacts/runs/old/reports"
    old_run.mkdir(parents=True)
    (old_run / "report.json").write_text("{}", encoding="utf-8")
    with closing(sqlite3.connect(app / CANONICAL_DATABASE_PATHS[1])) as connection:
        connection.execute("CREATE TABLE czsc_runs(run_id TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO czsc_runs VALUES (?)", [("current",), ("unselected",)])
        connection.commit()
    target = tmp_path / "package"
    manifest = build_package(source=app, target=target, include_data=True, include_artifacts="all")
    assert manifest["run_ids"] == ["current"]
    assert verify_package(target) == []
    assert (target / "app/vendor/czsc/LICENSE").is_file()
    for relative in CANONICAL_DATABASE_PATHS[:-1]:
        assert (target / "app" / relative).is_file()
    assert not (target / "app" / CANONICAL_DATABASE_PATHS[-1]).exists()
    assert (target / "app/my_strategy/artifacts/runs/current/reports/czsc.json").is_file()
    assert not (target / "app/my_strategy/artifacts/runs/old").exists()
    assert not (target / "app/my_strategy/data/processed/warehouse").exists()
    assert not (target / "app/.scratch").exists()
    assert not (target / "app/.venv").exists()
    assert not (target / "app/my_strategy/__pycache__").exists()
    assert runtime_layout(target)["app_root"] == target / "app"
    assert runtime_layout(target / "app") == runtime_layout(target)
    with closing(sqlite3.connect(target / "app" / CANONICAL_DATABASE_PATHS[1])) as connection:
        assert connection.execute("SELECT run_id FROM czsc_runs").fetchall() == [("current",)]


def test_package_online_backup_captures_active_wal(tmp_path):
    repo = source_repo(tmp_path)
    raw = repo / "app" / CANONICAL_DATABASE_PATHS[0]
    raw.parent.mkdir(parents=True)
    with closing(sqlite3.connect(raw)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE evidence (value TEXT)")
        writer.execute("INSERT INTO evidence VALUES ('committed-in-wal')")
        writer.commit()
        assert raw.with_name(raw.name + "-wal").exists()
        target = tmp_path / "package"
        build_package(source=repo, target=target, include_data=True, include_artifacts="none")
    copied = target / "app" / CANONICAL_DATABASE_PATHS[0]
    with closing(sqlite3.connect(copied)) as connection:
        assert connection.execute("SELECT value FROM evidence").fetchone()[0] == "committed-in-wal"
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert not copied.with_name(copied.name + "-wal").exists()


def test_external_paths_become_portable_and_integrity_is_checked(tmp_path):
    repo = source_repo(tmp_path)
    external = tmp_path / "external"
    raw = external / "raw/khquant_raw.db"
    raw.parent.mkdir(parents=True)
    with closing(sqlite3.connect(raw)) as connection:
        connection.execute("CREATE TABLE stock_daily_normalized(stock TEXT, date TEXT)")
        connection.commit()
    (repo / ".env").write_text(f'KHQUANT_PROJECT_ROOT=./app\nKHQUANT_DATA_ROOT="{external.as_posix()}"\n', encoding="utf-8")
    target = tmp_path / "portable"
    build_package(source=repo, target=target, include_data=True, include_artifacts="none")
    assert str(external) not in (target / ".env").read_text(encoding="utf-8")
    assert verify_package(target) == []
    (target / "app/my_strategy/core/paths.py").write_text("tampered", encoding="utf-8")
    assert any("mismatch" in issue for issue in verify_package(target))


def test_package_preserves_saved_personal_records_from_external_metadata(tmp_path):
    repo = source_repo(tmp_path)
    metadata = tmp_path / "personal metadata"
    (repo / ".env").write_text(
        f'KHQUANT_PROJECT_ROOT=./app\nKHQUANT_METADATA_ROOT="{metadata.as_posix()}"\n', encoding="utf-8")
    unavailable_raw = tmp_path / "no-market.db"
    personal = PersonalStore(metadata / "personal_portfolio.db", unavailable_raw)
    personal.add_watchlist(["000001.SZ", "600036.SH"])
    personal.save_holding("000001.SZ", 137, 10.25)
    target = tmp_path / "personal-package"
    build_package(source=repo, target=target, include_personal=True)
    assert verify_package(target) == []
    saved = target / "app/my_strategy/data/metadata/personal_portfolio.db"
    reopened = PersonalStore(saved, unavailable_raw)
    assert [row["symbol"] for row in reopened.watchlist()["items"]] == ["000001.SZ", "600036.SH"]
    holding = reopened.holdings()["items"][0]
    assert (holding["symbol"], holding["shares"], holding["average_cost"]) == ("000001.SZ", 137, 10.25)
    assert holding["market_value"] is None
    assert not (target / "app" / CANONICAL_DATABASE_PATHS[0]).exists()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def research_state(repo: Path) -> tuple[Path, Path]:
    from my_strategy.services.czsc_research_ml import MODEL_VERSION, LABEL_VERSION
    from my_strategy.services.czsc_research_profiles import get_feature_profile
    profile = get_feature_profile("legacy")
    runs = repo / "app/my_strategy/artifacts/runs"
    calendar = {"verified": True, "source": "verified fixture", "dates": ["2026-01-05", "2026-01-06"]}
    write_json(runs / "calendar/reports/calendar.json", calendar)
    model = runs / "training/models/production"
    model.mkdir(parents=True)
    checkpoint = model / "model.pt"
    checkpoint.write_bytes(b"immutable CPU checkpoint fixture")
    schema = {"columns": profile["columns"]}
    preprocess = {"mean": [0.0] * len(profile["columns"]), "scale": [1.0] * len(profile["columns"])}
    manifest = {"model_version": MODEL_VERSION, "label_version": LABEL_VERSION,
                "schema": schema, "schema_sha256": model_hash(schema),
                "preprocess": preprocess, "preprocess_sha256": model_hash(preprocess),
                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
                "available_at": "2026-01-06", "train_label_end": "2026-01-05",
                "validation_label_end": "2026-01-06", "validation_end": "2026-01-06",
                "feature_schema_binding": "bound", "feature_profile": profile["name"],
                "feature_version": profile["version"], "feature_schema_hash": profile["schema_hash"],
                "label_contract": {"calendar_hash": stable_hash(calendar["dates"])}, "data_version": "fixture-data"}
    manifest["manifest_sha256"] = model_hash(manifest)
    write_json(model / "manifest.json", manifest)
    write_json(runs / "training/reports/research.json", {"run_id": "training", "data_version": "fixture-data",
               "strategy_version": profile["strategy_version"], "config": {"feature_profile": profile["name"]},
               "calendar": {"run_id": "calendar", "hash": stable_hash(calendar)}, "evaluation": []})
    return runs, model


def add_release_database(repo: Path, runs: Path) -> Path:
    write_json(runs / "independent/reports/evaluation.json", {"training_run_id": "training"})
    artifacts = [{"path": path.relative_to(runs).as_posix(), "sha256": file_sha256(path)}
                 for path in [runs / "training/reports/research.json", runs / "calendar/reports/calendar.json",
                              runs / "training/models/production/manifest.json", runs / "training/models/production/model.pt",
                              runs / "independent/reports/evaluation.json"]]
    event = {"event_id": "release-1", "event_type": "promote", "target_release_id": "release-1",
             "model_run_id": "training", "artifacts": artifacts}
    db = repo / "app" / CANONICAL_DATABASE_PATHS[1]
    db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db)) as connection:
        connection.executescript("""CREATE TABLE czsc_model_release_events (
            sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL);
            CREATE TABLE czsc_model_aliases (alias TEXT,release_id TEXT,event_id TEXT);
            CREATE TABLE czsc_runs (run_id TEXT PRIMARY KEY,report_path TEXT);
            CREATE TRIGGER immutable_release BEFORE DELETE ON czsc_model_release_events BEGIN
                SELECT RAISE(ABORT, 'immutable'); END;""")
        connection.execute("INSERT INTO czsc_model_release_events VALUES (1,?,?)", (json.dumps(event), model_hash(event)))
        connection.execute("INSERT INTO czsc_model_aliases VALUES ('production','release-1','release-1')")
        connection.executemany("INSERT INTO czsc_runs VALUES (?,?)", [("scan", "scan/reports/czsc.json"),
                               ("unselected", "unselected/reports/czsc.json")])
        connection.commit()
    return db


def test_code_only_default_carries_installation_and_current_source_without_private_state(tmp_path):
    repo = source_repo(tmp_path)
    runs, _model = research_state(repo)
    write_json(repo / "runtime-manifest.json", {"schema": "fixture"})
    for name in ("setup.py", "pyproject.toml", "MANIFEST.in"):
        (repo / name).write_text("fixture", encoding="utf-8")
    for relative in (".github/workflows/portable.yml", "app/my_strategy/services/new_business.py",
                     "app/my_strategy/configs/new_business.json", "app/my_strategy/knowledge_base_system/prompts/private.md",
                     "app/my_strategy/knowledge_base/private.md", "app/AI_CHANGELOG.md", "app/.env.production",
                     "app/.pytest-old/private.txt", "app/build/lib/stale_business.py",
                     "app/dist/old.whl", "app/my_strategy/khquant.egg-info/PKG-INFO"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    target = tmp_path / "code"
    manifest = build_package(source=repo, target=target)
    assert manifest["run_ids"] == [] and manifest["include_personal"] is False
    assert list((target / "app/my_strategy/artifacts/runs").iterdir()) == []
    assert (target / "runtime-manifest.json").exists()
    assert (target / "setup.py").exists()
    assert (target / ".github/workflows/portable.yml").exists()
    assert (target / "app/my_strategy/services/new_business.py").exists()
    assert (target / "app/my_strategy/configs/new_business.json").exists()
    for relative in ("app/my_strategy/knowledge_base_system", "app/my_strategy/knowledge_base",
                     "app/AI_CHANGELOG.md", "app/.env.production", "app/.pytest-old",
                     "app/build", "app/dist", "app/my_strategy/khquant.egg-info"):
        assert not (target / relative).exists()
    assert (runs / "training/models/production/model.pt").exists()


@pytest.mark.parametrize("missing", ["install.py", "app/AGENTS.md"])
def test_export_rejects_non_source_layout_without_creating_target(tmp_path, missing):
    repo = source_repo(tmp_path)
    (repo / missing).unlink()
    target = tmp_path / "exports/package"
    with pytest.raises(RuntimeError, match="Use --from-project"):
        build_package(source=repo, target=target)
    assert not target.parent.exists()


def test_wheel_layout_cannot_export_site_packages_but_explicit_source_can(tmp_path):
    site_packages = tmp_path / "environment/lib/site-packages"
    marker = site_packages / "my_strategy/core/paths.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("# installed wheel module", encoding="utf-8")
    third_party = site_packages / "third_party/private.txt"
    third_party.parent.mkdir()
    third_party.write_text("unrelated installed package", encoding="utf-8")
    target = tmp_path / "exports/package"
    with pytest.raises(RuntimeError, match="wheel installations cannot be used as source"):
        main(["--from-project", str(site_packages), "--target", str(target)])
    assert not target.exists()
    repo = source_repo(tmp_path)
    assert main(["--from-project", str(repo), "--target", str(target)]) == 0
    assert (target / "install.py").is_file()
    assert not (target / "app/third_party").exists()


def test_research_and_calendar_without_czsc_report_survive_relocation_unchanged(tmp_path):
    repo = source_repo(tmp_path)
    runs, model = research_state(repo)
    write_json(runs / "study/metadata.json", {"task": "czsc-executable-entry-study"})
    write_json(runs / "study/reports/summary.json", {"training_run_id": "training"})
    original = (model / "manifest.json").read_bytes()
    target = tmp_path / "other path/portable"
    manifest = build_package(source=repo, target=target, include_artifacts="all")
    assert manifest["run_ids"] == ["calendar", "study", "training"]
    copied_runs = target / "app/my_strategy/artifacts/runs"
    assert (copied_runs / "training/models/production/manifest.json").read_bytes() == original
    catalog = checkpoint_catalog("training", runs_root=copied_runs)
    assert Path(catalog[0]["model_dir"]) == (copied_runs / "training/models/production").resolve()
    assert verify_package(target) == []


def test_latest_closes_model_calendar_release_evidence_and_keeps_immutable_events(tmp_path):
    repo = source_repo(tmp_path)
    runs, _model = research_state(repo)
    write_json(runs / "scan/reports/czsc.json", {"run_id": "scan", "model_run_id": "training"})
    database = add_release_database(repo, runs)
    with closing(sqlite3.connect(database)) as connection:
        original_event = connection.execute("SELECT payload,payload_sha256 FROM czsc_model_release_events").fetchone()
    target = tmp_path / "latest"
    manifest = build_package(source=repo, target=target, include_data=True, include_artifacts="latest")
    assert manifest["run_ids"] == ["calendar", "independent", "scan", "training"]
    assert manifest["release_event_count"] == 1
    with closing(sqlite3.connect(target / "app" / CANONICAL_DATABASE_PATHS[1])) as connection:
        assert connection.execute("SELECT run_id FROM czsc_runs").fetchall() == [("scan",)]
        assert connection.execute("SELECT payload,payload_sha256 FROM czsc_model_release_events").fetchone() == original_event
    assert verify_package(target) == []
    with pytest.raises(ValueError, match="immutable model releases"):
        build_package(source=repo, target=tmp_path / "broken-data-only", include_data=True)


@pytest.mark.parametrize("damage", ["calendar", "checkpoint", "manifest", "unsafe-reference"])
def test_package_rejects_broken_research_state_without_replacing_previous(tmp_path, damage):
    repo = source_repo(tmp_path)
    runs, model = research_state(repo)
    if damage == "calendar":
        (runs / "calendar/reports/calendar.json").unlink()
    elif damage == "checkpoint":
        (model / "model.pt").write_bytes(b"damaged")
    elif damage == "manifest":
        value = json.loads((model / "manifest.json").read_text(encoding="utf-8"))
        value["available_at"] = "2030-01-01"
        write_json(model / "manifest.json", value)
    else:
        write_json(runs / "scan/reports/czsc.json", {"run_id": "scan", "model_run_id": "../outside"})
    target = tmp_path / "previous"
    target.mkdir()
    (target / "keep.txt").write_text("previous", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        build_package(source=repo, target=target, include_artifacts="all", force=True)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "previous"


def test_verify_reports_release_hash_failure_even_with_rebuilt_outer_manifest(tmp_path):
    repo = source_repo(tmp_path)
    runs, _model = research_state(repo)
    write_json(runs / "scan/reports/czsc.json", {"run_id": "scan", "model_run_id": "training"})
    add_release_database(repo, runs)
    target = tmp_path / "released"
    build_package(source=repo, target=target, include_data=True, include_artifacts="all")
    (target / "app/my_strategy/artifacts/runs/independent/reports/evaluation.json").write_text("{}", encoding="utf-8")
    write_integrity_manifest(target, source_project_root=repo / "app", filename="image-manifest.json", reject_external_symlinks=True)
    assert any("release artifact hash mismatch" in issue for issue in verify_package(target))


def test_force_does_not_remove_previous_package_on_build_failure(tmp_path, monkeypatch):
    from my_strategy.scripts import package_project
    repo = source_repo(tmp_path)
    target = tmp_path / "package"
    target.mkdir()
    (target / "keep.txt").write_text("previous", encoding="utf-8")
    def fail(*args):
        raise RuntimeError("failed stage")
    monkeypatch.setattr(package_project, "_contents", fail)
    with pytest.raises(RuntimeError, match="failed stage"):
        build_package(source=repo, target=target, include_data=False, include_artifacts="none", force=True)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "previous"


def test_package_refuses_overlapping_target(tmp_path):
    repo = source_repo(tmp_path)
    with pytest.raises(RuntimeError, match="overlap"):
        build_package(source=repo, target=repo / "app/nested", include_data=False, include_artifacts="none")


def test_relative_verify_uses_source_repository_after_cli_changes_directory(tmp_path, monkeypatch):
    repo = source_repo(tmp_path)
    target = repo / "exports/portable"
    build_package(source=repo, target=target, include_data=False, include_artifacts="none")
    monkeypatch.chdir(tmp_path)
    assert main(["--from-project", str(repo), "--verify", "exports/portable"]) == 0
