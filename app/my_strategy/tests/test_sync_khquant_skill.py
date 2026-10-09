from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "khquant-ai-agent"
    / "scripts"
    / "sync_khquant_skill.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("sync_khquant_skill", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_skill(root: Path) -> Path:
    source = root / "source" / "khquant-ai-agent"
    (source / "references").mkdir(parents=True)
    (source / "SKILL.md").write_text("source skill\n", encoding="utf-8")
    (source / "references" / "migration.md").write_text("portable\n", encoding="utf-8")
    return source


def _create_junction(link: Path, target: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction test")
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot create a junction on this host: {result.stderr or result.stdout}")


def test_dry_run_reports_drift_without_writing(tmp_path: Path) -> None:
    module = _load_module()
    source = _source_skill(tmp_path)
    destination = tmp_path / "user" / "skills" / "khquant-ai-agent"

    results = module.sync_skill(source, [destination], dry_run=True)

    assert results == [
        {
            "destination": str(destination.resolve()),
            "file_count": 2,
            "status": "would-sync",
        }
    ]
    assert not destination.exists()


def test_sync_replaces_stale_directory_and_verifies_hashes(tmp_path: Path) -> None:
    module = _load_module()
    source = _source_skill(tmp_path)
    destination = tmp_path / "user" / "skills" / "khquant-ai-agent"
    destination.mkdir(parents=True)
    (destination / "SKILL.md").write_text("stale\n", encoding="utf-8")
    (destination / "obsolete.md").write_text("remove me\n", encoding="utf-8")

    first = module.sync_skill(source, [destination])
    second = module.sync_skill(source, [destination])

    assert first[0]["status"] == "synced"
    assert second[0]["status"] == "up-to-date"
    assert module.build_manifest(destination) == module.build_manifest(source)
    assert not (destination / "obsolete.md").exists()
    assert not list(destination.parent.glob(".khquant-ai-agent.*-*"))


def test_cli_json_accepts_an_explicit_destination(tmp_path: Path, capsys) -> None:
    module = _load_module()
    source = _source_skill(tmp_path)
    destination = tmp_path / "custom" / "khquant-ai-agent"

    exit_code = module.main(
        [
            "--source",
            str(source),
            "--destination",
            str(destination),
            "--dry-run",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload[0]["destination"] == str(destination.resolve())
    assert payload[0]["status"] == "would-sync"


def test_source_cannot_be_used_as_destination(tmp_path: Path) -> None:
    module = _load_module()
    source = _source_skill(tmp_path)

    try:
        module.sync_skill(source, [source])
    except ValueError as exc:
        assert "source" in str(exc).lower()
    else:
        raise AssertionError("source/destination overlap must be rejected")


@pytest.mark.parametrize("linked_root", ["source", "destination"])
def test_lexical_symlink_root_is_rejected_before_resolution(
    monkeypatch, tmp_path: Path, linked_root: str
) -> None:
    module = _load_module()
    source = _source_skill(tmp_path)
    destination = tmp_path / "user" / "skills" / "khquant-ai-agent"
    link = source if linked_root == "source" else destination
    real_is_symlink = Path.is_symlink
    real_resolve = Path.resolve

    def pretend_link(path: Path) -> bool:
        return path == link or real_is_symlink(path)

    def reject_early_resolution(path: Path, *args, **kwargs):
        if path == link:
            raise AssertionError("link identity must be checked before resolve()")
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", pretend_link)
    monkeypatch.setattr(Path, "resolve", reject_early_resolution)

    with pytest.raises(ValueError, match="symlink|junction|reparse"):
        module.sync_skill(source, [destination], dry_run=True)


def test_source_tree_rejects_a_windows_junction(tmp_path: Path) -> None:
    module = _load_module()
    source = _source_skill(tmp_path)
    external = tmp_path / "external-source"
    external.mkdir()
    (external / "unexpected.md").write_text("outside\n", encoding="utf-8")
    junction = source / "references" / "external"
    _create_junction(junction, external)

    try:
        with pytest.raises(ValueError, match="symlink|junction|reparse"):
            module.build_manifest(source)
    finally:
        if junction.is_junction():
            junction.rmdir()


def test_destination_windows_junction_is_rejected_without_touching_target(tmp_path: Path) -> None:
    module = _load_module()
    source = _source_skill(tmp_path)
    external = tmp_path / "external-destination"
    external.mkdir()
    marker = external / "SKILL.md"
    marker.write_text("external target\n", encoding="utf-8")
    destination = tmp_path / "user" / "skills" / "khquant-ai-agent"
    destination.parent.mkdir(parents=True)
    _create_junction(destination, external)

    try:
        with pytest.raises(ValueError, match="symlink|junction|reparse"):
            module.sync_skill(source, [destination], dry_run=True)
        assert marker.read_text(encoding="utf-8") == "external target\n"
    finally:
        if destination.is_junction():
            destination.rmdir()


def test_failed_swap_restores_the_previous_skill(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    source = _source_skill(tmp_path)
    destination = tmp_path / "user" / "skills" / "khquant-ai-agent"
    destination.mkdir(parents=True)
    (destination / "SKILL.md").write_text("previous skill\n", encoding="utf-8")
    real_replace = module.os.replace

    def fail_stage_swap(source_path, destination_path):
        if Path(source_path).name.startswith(".khquant-ai-agent.stage-"):
            raise OSError("simulated swap failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(module.os, "replace", fail_stage_swap)

    with pytest.raises(OSError, match="simulated"):
        module.sync_skill(source, [destination])

    assert (destination / "SKILL.md").read_text(encoding="utf-8") == "previous skill\n"
    assert not list(destination.parent.glob(".khquant-ai-agent.*-*"))


def test_second_destination_failure_restores_first_destination(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    source = _source_skill(tmp_path)
    first = tmp_path / "codex" / "khquant-ai-agent"
    second = tmp_path / "agents" / "khquant-ai-agent"
    for destination, content in ((first, "first old\n"), (second, "second old\n")):
        destination.mkdir(parents=True)
        (destination / "SKILL.md").write_text(content, encoding="utf-8")
    real_replace = module.os.replace

    def fail_second_stage(source_path, destination_path):
        if (
            Path(source_path).name.startswith(".khquant-ai-agent.stage-")
            and Path(destination_path).resolve() == second.resolve()
        ):
            raise OSError("second destination failed")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(module.os, "replace", fail_second_stage)

    with pytest.raises(OSError, match="second destination"):
        module.sync_skill(source, [first, second])

    assert (first / "SKILL.md").read_text(encoding="utf-8") == "first old\n"
    assert (second / "SKILL.md").read_text(encoding="utf-8") == "second old\n"
