"""Installed wheels must retain frozen provenance and use writable user state."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from my_strategy import runtime_env
from my_strategy.adapters import czsc_adapter


def test_wheel_defaults_keep_state_outside_code(monkeypatch, tmp_path):
    package = tmp_path / "site-packages" / "my_strategy"
    monkeypatch.setattr(runtime_env, "__file__", str(package / "runtime_env.py"))
    for key in list(runtime_env.os.environ):
        if key.startswith("KHQUANT_") or key == "PYTHONPATH":
            monkeypatch.delenv(key)
    home = tmp_path / "user-state"
    monkeypatch.setenv("KHQUANT_HOME", str(home))
    runtime_env.initialize_runtime_environment()
    assert runtime_env.os.environ["KHQUANT_PROJECT_ROOT"] == str(package.parent)
    assert runtime_env.os.environ["KHQUANT_RAW_DB"] == str(home / "data" / "raw" / "khquant_raw.db")
    assert runtime_env.os.environ["KHQUANT_ARTIFACT_ROOT"] == str(home / "artifacts")
    assert runtime_env.runtime_resource_root() == package / "_runtime"
    assert not home.exists()


def test_checkout_environment_does_not_override_explicit_state(monkeypatch, tmp_path):
    repo = tmp_path / "checkout"
    app = repo / "app"
    app.mkdir(parents=True)
    (app / "AGENTS.md").write_text("source checkout", encoding="utf-8")
    (repo / ".env").write_text('KHQUANT_PROJECT_ROOT=./app\nKHQUANT_DATA_ROOT=./my_strategy/data\n', encoding="utf-8")
    monkeypatch.setattr(runtime_env, "__file__", str(app / "my_strategy" / "runtime_env.py"))
    monkeypatch.delenv("KHQUANT_PROJECT_ROOT", raising=False)
    explicit = tmp_path / "selected-data"
    monkeypatch.setenv("KHQUANT_DATA_ROOT", str(explicit))
    runtime_env.initialize_runtime_environment()
    assert runtime_env.os.environ["KHQUANT_PROJECT_ROOT"] == str(app)
    assert runtime_env.os.environ["KHQUANT_DATA_ROOT"] == str(explicit)


def test_platform_wheel_provenance_is_bound_to_frozen_source(monkeypatch, tmp_path):
    source = tmp_path / "vendor" / "czsc"
    source.mkdir(parents=True)
    (source / "engine.py").write_bytes(b"frozen algorithm")
    digest = hashlib.sha256(b"frozen algorithm").hexdigest()
    manifest = {"archive_sha256": czsc_adapter.SOURCE_SHA256, "built_wheel_sha256": "windows-wheel",
                "files": [{"path": "engine.py", "sha256": digest}]}
    (tmp_path / "czsc-source-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    wheels = tmp_path / "vendor" / "wheels"
    wheels.mkdir()
    source_digest = hashlib.sha256(f"engine.py {digest}".encode()).hexdigest()
    index = {"package": "czsc", "version": "1.0.1", "source_tree_sha256": source_digest,
             "wheels": [{"filename": "mac.whl", "sha256": "mac-wheel"}]}
    index_path = wheels / "wheel-index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    distribution = SimpleNamespace(version="1.0.1", read_text=lambda _: json.dumps(
        {"url": "file:///mac.whl", "archive_info": {"hashes": {"sha256": "mac-wheel"}}}))
    monkeypatch.setattr(czsc_adapter.importlib.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(czsc_adapter, "runtime_resource_root", lambda: tmp_path)
    expected = SimpleNamespace(__version__="1.0.1")
    monkeypatch.setitem(sys.modules, "czsc", expected)
    czsc_adapter.native_runtime.cache_clear()
    try:
        assert czsc_adapter.native_runtime() is expected
        index["source_tree_sha256"] = "different-source"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        czsc_adapter.native_runtime.cache_clear()
        with pytest.raises(RuntimeError, match="来源索引"):
            czsc_adapter.native_runtime()
        index["source_tree_sha256"] = source_digest
        index_path.write_text(json.dumps(index), encoding="utf-8")
        (source / "engine.py").write_bytes(b"changed algorithm")
        with pytest.raises(RuntimeError, match="源码校验失败"):
            czsc_adapter.native_runtime()
    finally:
        czsc_adapter.native_runtime.cache_clear()
