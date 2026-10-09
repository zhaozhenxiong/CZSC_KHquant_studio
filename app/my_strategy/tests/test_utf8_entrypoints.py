"""Exercise real entrypoint pipes under the Windows CI's cp1252 setting."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from my_strategy import runtime_env


ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "app"


def _installer():
    spec = importlib.util.spec_from_file_location("khquant_utf8_installer", ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _child(arguments, tmp_path):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("KHQUANT_") and key != "PYTHONHOME"}
    environment.update(
        PYTHONPATH=str(APP), PYTHONUTF8="0", PYTHONIOENCODING="cp1252",
        PYTHONDONTWRITEBYTECODE="1", KHQUANT_PROJECT_ROOT=str(APP),
        KHQUANT_DATA_ROOT=str(tmp_path / "data"),
        KHQUANT_RAW_DB=str(tmp_path / "data/raw/khquant_raw.db"),
        KHQUANT_ARTIFACT_ROOT=str(tmp_path / "artifacts"),
        KHQUANT_METADATA_ROOT=str(tmp_path / "metadata"),
        KHQUANT_LOG_ROOT=str(tmp_path / "logs"), KHQUANT_COMPUTE_DEVICE="cpu",
    )
    return subprocess.run([sys.executable, *arguments], cwd=tmp_path, env=environment,
                          capture_output=True, check=False, timeout=30)


@pytest.mark.parametrize("entrypoint", ["runtime", "installer"])
def test_utf8_configuration_keeps_stringio_capture_streams(monkeypatch, entrypoint):
    configure = (runtime_env.configure_utf8_stdio if entrypoint == "runtime"
                 else _installer().configure_utf8_stdio)
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    configure()
    print("量价模型")
    print("中文错误", file=sys.stderr)
    assert sys.stdout is stdout and stdout.getvalue() == "量价模型\n"
    assert sys.stderr is stderr and stderr.getvalue() == "中文错误\n"


def test_doctor_json_roundtrip_from_real_cp1252_pipe(tmp_path):
    code = """
import sys
assert sys.stdout.encoding.lower() == 'cp1252'
from my_strategy.scripts import docker_runtime_doctor as doctor
assert sys.stdout.encoding.lower() == 'cp1252', 'import must not reconfigure streams'
doctor.build_payload = lambda args: {
    'status': 'success', 'failures': [], '说明': '量价模型保持影子',
    'path': 'C:/行情数据/研究模型',
}
raise SystemExit(doctor.main(['--json', '--strict']))
"""
    result = _child(["-c", code], tmp_path)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    payload = json.loads(result.stdout.decode("utf-8"))
    assert payload["说明"] == "量价模型保持影子"
    assert payload["path"] == "C:/行情数据/研究模型"
    assert "量价模型".encode("utf-8") in result.stdout


@pytest.mark.parametrize("arguments, expected", [
    (["--help"], "结构研究"),
    (["analyze", "--help"], "计算设备"),
])
def test_real_cli_help_is_utf8_under_cp1252(tmp_path, arguments, expected):
    code = "import sys, runpy; assert sys.stdout.encoding.lower() == 'cp1252'; runpy.run_module('my_strategy.cli', run_name='__main__')"
    result = _child(["-c", code, *arguments], tmp_path)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert expected in result.stdout.decode("utf-8")


def test_real_cli_parser_error_is_utf8_under_cp1252(tmp_path):
    result = _child(["-m", "my_strategy.cli", "analyze", "--model-family", "不存在"], tmp_path)
    assert result.returncode == 2
    assert "不存在" in result.stderr.decode("utf-8")
    assert b"UnicodeEncodeError" not in result.stderr


def test_real_installer_dry_run_roundtrips_chinese_paths(tmp_path):
    data = tmp_path / "行情 数据"
    artifacts = tmp_path / "模型研究"
    environment = tmp_path / "运行环境"
    result = _child([str(ROOT / "install.py"), "--dry-run", "--device", "cpu",
                     "--venv-dir", str(environment), "--data-dir", str(data),
                     "--artifact-dir", str(artifacts), "--raw-db", str(data / "原始.db")], tmp_path)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    payload = json.loads(result.stdout.decode("utf-8"))
    assert Path(payload["data"]) == data.resolve()
    assert Path(payload["artifacts"]) == artifacts.resolve()
    assert Path(payload["raw_db"]) == (data / "原始.db").resolve()
    assert "行情 数据".encode("utf-8") in result.stdout
    assert not data.exists() and not artifacts.exists() and not environment.exists()


def test_cli_delegates_propagate_utf8_without_mutating_parent_environment(monkeypatch):
    from my_strategy import cli
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    monkeypatch.setenv("PYTHONUTF8", "0")
    calls = []
    monkeypatch.setattr(cli.subprocess, "call", lambda command, **kwargs: calls.append(kwargs) or 0)
    assert cli.main(["doctor", "--json"]) == 0
    assert calls[0]["env"]["PYTHONIOENCODING"] == "utf-8"
    assert calls[0]["env"]["PYTHONUTF8"] == "1"
    assert os.environ["PYTHONIOENCODING"] == "cp1252"
    assert os.environ["PYTHONUTF8"] == "0"


def test_installer_children_override_legacy_cp1252_configuration(monkeypatch, tmp_path):
    installer = _installer()
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    monkeypatch.setattr(installer, "APP", tmp_path / "app")
    installer.APP.mkdir()
    (installer.APP / "czsc-source-manifest.json").write_text(
        json.dumps({"version": "1.0.1", "files": []}), encoding="utf-8")
    (tmp_path / ".env").write_text('PYTHONIOENCODING="cp1252"\nPYTHONUTF8="0"\n', encoding="utf-8")
    monkeypatch.setattr(installer.platform, "system", lambda: "Windows")
    monkeypatch.setattr(installer.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(installer.venv.EnvBuilder, "create", lambda *args: None)
    monkeypatch.setattr(installer, "compatible_wheel", lambda *args: tmp_path / "verified-czsc.whl")
    monkeypatch.setattr(installer, "record_wheel_provenance", lambda *args: None)
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    assert installer.install(["--device", "cpu"]) == 0
    assert any(command[-4:] == ["doctor", "--check-write", "--strict", "--json"]
               for command, _ in calls)
    assert all(kwargs["env"]["PYTHONUTF8"] == "1" and kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
               for _, kwargs in calls)
    assert 'PYTHONIOENCODING="cp1252"' in (tmp_path / ".env").read_text(encoding="utf-8")
