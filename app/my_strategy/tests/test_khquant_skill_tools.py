from __future__ import annotations

import importlib.util
import gzip
import json
import sqlite3
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = APP_ROOT / "my_strategy" / "skills" / "khquant-ai-agent"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_app_root(root: Path) -> Path:
    app = root / "app"
    (app / "my_strategy").mkdir(parents=True)
    (app / "AGENTS.md").write_text("# test", encoding="utf-8")
    return app


def test_verifier_finds_nested_application_and_checks_duplicate_keys(tmp_path: Path) -> None:
    verifier = load_module("khquant_verifier", SKILL_ROOT / "scripts" / "verify_khquant_workspace.py")
    app = make_app_root(tmp_path)
    for relative in verifier.REQUIRED_PATHS:
        path = app / relative
        if "." in path.name:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        else:
            path.mkdir(parents=True, exist_ok=True)
    database = app / "my_strategy/data/raw/khquant_raw.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE stock_daily_normalized (stock TEXT, date TEXT)")
        connection.execute("CREATE TABLE securities (stock TEXT)")
        connection.executemany("INSERT INTO stock_daily_normalized VALUES (?, ?)", [("000001.SZ", "2026-09-10"), ("000001.SZ", "2026-09-10")])
        connection.execute("INSERT INTO securities VALUES ('000001.SZ')")
    assert verifier.find_application_root(tmp_path) == app
    report, missing = verifier.verify(app, "2026-09-10", 1, 24, None, environ={})
    assert missing == []
    assert report["raw_database"]["target_stocks"] == 1
    assert report["raw_database"]["duplicate_stock_dates"] == 1


def test_verifier_resolves_repo_and_project_relative_env_paths(tmp_path: Path) -> None:
    verifier = load_module("khquant_verifier_portable", SKILL_ROOT / "scripts" / "verify_khquant_workspace.py")
    app = make_app_root(tmp_path)
    for relative in verifier.REQUIRED_PATHS:
        path = app / relative
        if "." in path.name:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        else:
            path.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".env").write_text(
        "\n".join(
            (
                'KHQUANT_PROJECT_ROOT="./app"',
                'KHQUANT_DATA_ROOT="../portable-data"',
                'KHQUANT_ARTIFACT_ROOT="../portable-artifacts"',
                'KHQUANT_RAW_DB="../portable-data/raw/custom.db"',
                'KHQUANT_VENV_DIR="./runtime-venv"',
                'PYTHONPATH="./app"',
            )
        ),
        encoding="utf-8",
    )
    database = tmp_path / "portable-data/raw/custom.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE stock_daily_normalized (stock TEXT, date TEXT)")
        connection.execute("INSERT INTO stock_daily_normalized VALUES ('000001.SZ', '2026-09-24')")

    report, missing = verifier.verify(app, "2026-09-24", 1, 24, None, environ={})

    assert missing == []
    assert report["runtime_layout"]["application_root_matches"] is True
    assert report["runtime_layout"]["data_root"] == str(tmp_path / "portable-data")
    assert report["runtime_layout"]["artifact_root"] == str(tmp_path / "portable-artifacts")
    assert report["runtime_layout"]["venv_root"] == str(tmp_path / "runtime-venv")
    assert report["runtime_layout"]["python_path_contains_application"] is True
    assert report["raw_database"]["target_stocks"] == 1


def test_verifier_defaults_to_flat_application_root(tmp_path: Path) -> None:
    verifier = load_module("khquant_verifier_flat", SKILL_ROOT / "scripts" / "verify_khquant_workspace.py")
    (tmp_path / "my_strategy").mkdir()
    (tmp_path / "AGENTS.md").write_text("# test", encoding="utf-8")

    layout = verifier.resolve_runtime_layout(tmp_path, environ={})

    assert layout["repo_root"] == str(tmp_path)
    assert layout["configured_project_root"] == str(tmp_path)
    assert layout["application_root_matches"] is True
    assert layout["python_paths"] == [str(tmp_path)]
    assert layout["python_path_contains_application"] is True


def test_verifier_closes_read_only_database_connections(tmp_path: Path, monkeypatch) -> None:
    verifier = load_module("khquant_verifier_closing", SKILL_ROOT / "scripts" / "verify_khquant_workspace.py")
    raw_database = tmp_path / "raw.db"
    with sqlite3.connect(raw_database) as connection:
        connection.execute("CREATE TABLE stock_daily_normalized (stock TEXT, date TEXT)")

    real_connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(verifier.sqlite3, "connect", tracking_connect)

    verifier.inspect_raw_database(raw_database, None, 60)

    assert len(opened) == 1
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")


def test_verifier_strict_mode_rejects_stale_project_binding(tmp_path: Path) -> None:
    verifier = load_module("khquant_verifier_stale", SKILL_ROOT / "scripts" / "verify_khquant_workspace.py")
    app = make_app_root(tmp_path)
    (tmp_path / ".env").write_text('KHQUANT_PROJECT_ROOT="./old-app"\n', encoding="utf-8")

    report, _ = verifier.verify(app, None, 60, 24, None, environ={})
    errors = verifier.strict_errors(report, None, None)

    assert any("KHQUANT_PROJECT_ROOT points to" in error for error in errors)


def test_record_audit_finds_missing_note_and_unreferenced_dispatch(tmp_path: Path) -> None:
    auditor = load_module("record_auditor", APP_ROOT / "my_strategy/scripts/audit_project_records.py")
    app = make_app_root(tmp_path)
    change = app / "my_strategy/knowledge_base/04_AI_Changes/2026/09/20260910-000000-aaaaaaaa-change.md"
    change.parent.mkdir(parents=True)
    change.write_text("# change", encoding="utf-8")
    dispatch = app / "my_strategy/knowledge_base_system/dispatch_packets/old.md"
    dispatch.parent.mkdir(parents=True)
    dispatch.write_text("# planned", encoding="utf-8")
    sessions = app / "my_strategy/knowledge_base_system/sessions"
    sessions.mkdir(parents=True)
    (sessions / "20260910-000000-aaaaaaaa.json").write_text(
        json.dumps({"session_id": "20260910-000000-aaaaaaaa", "status": "success"}),
        encoding="utf-8",
    )
    report = auditor.audit(app)
    assert report["changes_without_session_note"] == ["20260910-000000-aaaaaaaa"]
    assert report["unreferenced_dispatch_packets"][0]["name"] == "old.md"
    assert report["unreferenced_dispatch_packets"][0]["sha256"]


def test_large_ai_diff_uses_redacted_compressed_sidecar(tmp_path: Path) -> None:
    logger = load_module("ai_change_logger_for_test", APP_ROOT / "my_strategy/scripts/ai_change_logger.py")
    logger.PROJECT_ROOT = tmp_path
    logger.DIFF_ROOT = tmp_path / "my_strategy/knowledge_base_system/diffs"
    logger.DIFF_ROOT.mkdir(parents=True)
    logger.MAX_DIFF_CHARS = 10
    redacted = logger.redact("api_key=secret-value\n" + "x" * 40)
    relative = logger.write_diff_artifact("session", redacted)
    assert relative == "my_strategy/knowledge_base_system/diffs/session.patch.gz"
    with gzip.open(tmp_path / relative, "rt", encoding="utf-8") as handle:
        content = handle.read()
    assert "secret-value" not in content
    assert "<REDACTED>" in content


def test_ai_change_status_parser_keeps_first_filename_character() -> None:
    logger = load_module("ai_change_logger_status_test", APP_ROOT / "my_strategy/scripts/ai_change_logger.py")
    assert logger.parse_status_paths(" M AGENTS.md\n?? new-file.md") == ["AGENTS.md", "new-file.md"]


def test_ai_changelog_append_is_idempotent_for_session(tmp_path: Path) -> None:
    logger = load_module("ai_change_logger_append_test", APP_ROOT / "my_strategy/scripts/ai_change_logger.py")
    logger.PROJECT_ROOT = tmp_path
    logger.CHANGELOG = tmp_path / "AI_CHANGELOG.md"
    note = tmp_path / "my_strategy/knowledge_base/04_AI_Changes/change.md"
    note.parent.mkdir(parents=True)
    note.touch()
    data = {
        "session_id": "20260911-000000-aaaaaaaa",
        "ended_at": "2026-09-11T00:00:00+08:00",
        "agent": "ops-agent",
        "status": "success",
        "task": "test",
        "changed_files": [],
        "final_commit": "UNCOMMITTED",
    }
    logger.append_changelog(data, note)
    logger.append_changelog(data, note)
    assert logger.CHANGELOG.read_text(encoding="utf-8").count(data["session_id"]) == 1
