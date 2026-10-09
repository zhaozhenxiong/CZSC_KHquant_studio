from __future__ import annotations

from my_strategy import runtime_env


def test_normalize_native_runtime_paths_resolves_project_relative_data(monkeypatch, tmp_path) -> None:
    repo = tmp_path / "repo"
    monkeypatch.setenv("KHQUANT_ENV_SCHEMA", "2")
    monkeypatch.setenv("KHQUANT_PROJECT_ROOT", "./app")
    monkeypatch.setenv("KHQUANT_DATA_ROOT", "./my_strategy/data")
    monkeypatch.setenv("KHQUANT_ARTIFACT_ROOT", "./my_strategy/artifacts")
    monkeypatch.setenv("KHQUANT_RAW_DB", "./my_strategy/data/raw/khquant_raw.db")
    monkeypatch.setenv("KHQUANT_VENV_DIR", "./.venv")
    monkeypatch.setenv("PYTHONPATH", "./app")

    runtime_env.normalize_native_runtime_paths(repo)

    assert runtime_env.os.environ["KHQUANT_PROJECT_ROOT"] == str(repo / "app")
    assert runtime_env.os.environ["KHQUANT_DATA_ROOT"] == str(repo / "app" / "my_strategy" / "data")
    assert runtime_env.os.environ["KHQUANT_ARTIFACT_ROOT"] == str(repo / "app" / "my_strategy" / "artifacts")
    assert runtime_env.os.environ["KHQUANT_RAW_DB"] == str(repo / "app" / "my_strategy" / "data" / "raw" / "khquant_raw.db")
    assert runtime_env.os.environ["KHQUANT_VENV_DIR"] == str(repo / ".venv")
    assert runtime_env.os.environ["PYTHONPATH"] == str(repo / "app")


def test_normalize_native_runtime_paths_keeps_legacy_launcher_business_paths(
    monkeypatch, tmp_path
) -> None:
    repo = tmp_path / "repo"
    monkeypatch.delenv("KHQUANT_ENV_SCHEMA", raising=False)
    monkeypatch.setenv("KHQUANT_PROJECT_ROOT", "./app")
    monkeypatch.setenv("KHQUANT_DATA_ROOT", "./data")

    runtime_env.normalize_native_runtime_paths(repo)

    assert runtime_env.os.environ["KHQUANT_PROJECT_ROOT"] == str(repo / "app")
    assert runtime_env.os.environ["KHQUANT_DATA_ROOT"] == str(repo / "app" / "data")


def test_normalize_native_runtime_paths_accepts_legacy_installer_app_prefix(
    monkeypatch, tmp_path
) -> None:
    repo = tmp_path / "repo"
    monkeypatch.delenv("KHQUANT_ENV_SCHEMA", raising=False)
    monkeypatch.setenv("KHQUANT_PROJECT_ROOT", "./app")
    monkeypatch.setenv("KHQUANT_DATA_ROOT", "./app/my_strategy/data")

    runtime_env.normalize_native_runtime_paths(repo)

    assert runtime_env.os.environ["KHQUANT_DATA_ROOT"] == str(repo / "app" / "my_strategy" / "data")
