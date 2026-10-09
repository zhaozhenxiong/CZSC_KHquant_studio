"""CZSC runtime checks cannot create or fall back to a legacy database."""
from contextlib import closing
from pathlib import Path
import sqlite3

from my_strategy.scripts import docker_runtime_doctor as doctor


def isolate(monkeypatch, tmp_path):
    directories = {"PROJECT_ROOT": tmp_path / "app", "DATA_ROOT": tmp_path / "data",
                   "METADATA_ROOT": tmp_path / "data/metadata", "PROCESSED_DATA_ROOT": tmp_path / "data/processed",
                   "ARTIFACT_RUNS_ROOT": tmp_path / "artifacts/runs"}
    for name, path in directories.items():
        monkeypatch.setattr(doctor, name, path)
    directories["PROJECT_ROOT"].mkdir()
    raw = directories["DATA_ROOT"] / "raw/khquant_raw.db"
    monkeypatch.setattr(doctor, "raw_db_path", lambda: raw)
    monkeypatch.setattr(doctor, "inside_docker", lambda: False)
    monkeypatch.setattr(doctor, "module_version", lambda name: {"name": name, "ok": True})
    monkeypatch.setattr(doctor, "czsc_report", lambda: {"name": "czsc", "ok": True})
    monkeypatch.setattr(doctor, "compute_status", lambda: {"available": True, "selected_device": "cpu"})
    return raw


def test_read_only_check_does_not_create_directories_or_databases(monkeypatch, tmp_path):
    raw = isolate(monkeypatch, tmp_path)
    payload = doctor.build_payload(doctor.parse_args(["--json"]))
    assert payload["status"] == "failed"
    assert not raw.parent.exists()
    assert all(not item["exists"] for item in payload["databases"])


def test_write_probe_creates_only_current_directories(monkeypatch, tmp_path):
    raw = isolate(monkeypatch, tmp_path)
    payload = doctor.build_payload(doctor.parse_args(["--check-write", "--expect-native", "--strict"]))
    assert payload["status"] == "success"
    assert not raw.exists()
    assert {item["name"] for item in payload["databases"]} == {"raw_db", "czsc_results", "czsc_tasks"}
    assert not (tmp_path / "data/processed/warehouse").exists()
    assert not list(tmp_path.rglob(".czsc-doctor-*"))


def test_invalid_current_database_fails_strict_validation(monkeypatch, tmp_path):
    raw = isolate(monkeypatch, tmp_path)
    raw.parent.mkdir(parents=True)
    with closing(sqlite3.connect(raw)) as connection:
        connection.execute("CREATE TABLE wrong_schema(value TEXT)")
        connection.commit()
    payload = doctor.build_payload(doctor.parse_args(["--check-write", "--strict"]))
    assert any("raw_db" in failure for failure in payload["failures"])
    report = payload["databases"][0]
    assert report["missing_tables"] == ["securities", "stock_daily_normalized"]
    with closing(sqlite3.connect(raw)) as connection:
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name='securities'").fetchone()


def test_frozen_attachment_failure_is_required(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(doctor, "czsc_report", lambda: {"name": "czsc", "ok": False, "error": "wrong source"})
    payload = doctor.build_payload(doctor.parse_args(["--check-write"]))
    assert any("czsc: wrong source" in failure for failure in payload["failures"])


def test_explicit_cuda_unavailable_fails_doctor(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(doctor, "compute_status", lambda: {"available": False, "selected_device": None,
                                                          "fallback_reason": "cuda:1 不可用"})
    payload = doctor.build_payload(doctor.parse_args(["--check-write", "--strict"]))
    assert payload["status"] == "failed"
    assert "compute unavailable: cuda:1 不可用" in payload["failures"]
