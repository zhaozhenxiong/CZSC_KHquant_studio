#!/usr/bin/env python3
"""Atomically synchronize the repository-owned KHQuant skill to user homes."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import uuid


SKILL_NAME = "khquant-ai-agent"
_IGNORED_DIRS = {"__pycache__", ".pytest_cache"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


def _is_link_or_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _checked_lexical_root(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if _is_link_or_reparse_point(lexical):
        raise ValueError(f"{label} must not be a symlink, junction, or reparse point: {lexical}")
    return lexical


def _manifest_files(root: Path) -> Iterator[Path]:
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames.sort()
        filenames.sort()
        directory_path = Path(directory)
        for name in [*dirnames, *filenames]:
            candidate = directory_path / name
            if _is_link_or_reparse_point(candidate):
                raise ValueError(
                    f"skill trees must not contain symlinks, junctions, or reparse points: {candidate}"
                )
        for name in filenames:
            path = directory_path / name
            if path.is_file():
                yield path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _include(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return not any(part in _IGNORED_DIRS for part in relative.parts) and path.suffix not in _IGNORED_SUFFIXES


def build_manifest(root: Path) -> dict[str, str]:
    """Return a stable relative-path to SHA256 manifest for a skill tree."""
    root = _checked_lexical_root(root, label="skill tree root")
    if not (root / "SKILL.md").is_file():
        raise ValueError(f"skill source is missing SKILL.md: {root}")
    manifest: dict[str, str] = {}
    for path in _manifest_files(root):
        if _include(path, root):
            manifest[path.relative_to(root).as_posix()] = _sha256(path)
    return manifest


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _paths_overlap(source: Path, destination: Path) -> bool:
    try:
        destination.relative_to(source)
        return True
    except ValueError:
        pass
    try:
        source.relative_to(destination)
        return True
    except ValueError:
        return False


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in _IGNORED_DIRS or Path(name).suffix in _IGNORED_SUFFIXES
    }


def _sync_one(source: Path, destination: Path, source_manifest: dict[str, str], *, dry_run: bool) -> dict[str, object]:
    source = _checked_lexical_root(source, label="skill source")
    destination = _checked_lexical_root(destination, label="skill destination")
    if _paths_overlap(source.resolve(), destination.resolve()):
        raise ValueError(f"skill source and destination must not overlap: {source} -> {destination}")

    destination_manifest = build_manifest(destination) if (destination / "SKILL.md").is_file() else None
    status = "up-to-date" if destination_manifest == source_manifest else "would-sync"
    result: dict[str, object] = {
        "destination": str(destination),
        "file_count": len(source_manifest),
        "status": status,
    }
    if dry_run or status == "up-to-date":
        return result

    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stage = destination.parent / f".{destination.name}.stage-{token}"
    backup = destination.parent / f".{destination.name}.backup-{token}"
    moved_existing = False
    try:
        shutil.copytree(source, stage, copy_function=shutil.copy2, ignore=_copy_ignore)
        if build_manifest(stage) != source_manifest:
            raise RuntimeError(f"staged skill hash verification failed: {stage}")
        if destination.exists() or destination.is_symlink():
            os.replace(destination, backup)
            moved_existing = True
        os.replace(stage, destination)
        if build_manifest(destination) != source_manifest:
            raise RuntimeError(f"installed skill hash verification failed: {destination}")
    except Exception:
        _remove_path(stage)
        if moved_existing:
            _remove_path(destination)
            os.replace(backup, destination)
        raise
    else:
        _remove_path(backup)

    result["status"] = "synced"
    return result


def sync_skill(source: Path, destinations: list[Path], *, dry_run: bool = False) -> list[dict[str, object]]:
    """Synchronize all destinations as one staged, compensating transaction."""
    source = _checked_lexical_root(source, label="skill source")
    source_manifest = build_manifest(source)
    resolved_source = source.resolve()
    unique: list[Path] = []
    seen: set[str] = set()
    for destination in destinations:
        lexical = _checked_lexical_root(destination, label="skill destination")
        resolved = lexical.resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            unique.append(lexical)
    results: list[dict[str, object]] = []
    changed: list[Path] = []
    for destination in unique:
        if _paths_overlap(resolved_source, destination.resolve()):
            raise ValueError(f"skill source and destination must not overlap: {source} -> {destination}")
        destination_manifest = build_manifest(destination) if (destination / "SKILL.md").is_file() else None
        status = "up-to-date" if destination_manifest == source_manifest else "would-sync"
        results.append({"destination": str(destination), "file_count": len(source_manifest), "status": status})
        if status != "up-to-date":
            changed.append(destination)
    if dry_run or not changed:
        return results

    transaction = uuid.uuid4().hex
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    swapped: list[Path] = []
    try:
        for destination in changed:
            _checked_lexical_root(destination, label="skill destination")
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage = destination.parent / f".{destination.name}.stage-{transaction}"
            shutil.copytree(source, stage, copy_function=shutil.copy2, ignore=_copy_ignore)
            if build_manifest(stage) != source_manifest:
                raise RuntimeError(f"staged skill hash verification failed: {stage}")
            staged[destination] = stage

        for destination in changed:
            _checked_lexical_root(destination, label="skill destination")
            backup = destination.parent / f".{destination.name}.backup-{transaction}"
            previous: Path | None = None
            if destination.exists() or destination.is_symlink():
                os.replace(destination, backup)
                previous = backup
            backups[destination] = previous
            os.replace(staged[destination], destination)
            swapped.append(destination)
            if build_manifest(destination) != source_manifest:
                raise RuntimeError(f"installed skill hash verification failed: {destination}")
    except Exception:
        for destination in reversed(swapped):
            _remove_path(destination)
            previous = backups.get(destination)
            if previous is not None and previous.exists():
                os.replace(previous, destination)
        for destination, previous in backups.items():
            if destination not in swapped and previous is not None and previous.exists():
                os.replace(previous, destination)
        for stage in staged.values():
            _remove_path(stage)
        for backup in backups.values():
            if backup is not None:
                _remove_path(backup)
        raise
    else:
        for backup in backups.values():
            if backup is not None:
                _remove_path(backup)
        for result in results:
            if Path(str(result["destination"])) in changed:
                result["status"] = "synced"
    return results


def default_destinations() -> list[Path]:
    home = Path.home()
    codex_home = Path(os.environ.get("CODEX_HOME", str(home / ".codex")))
    agents_home = Path(os.environ.get("AGENTS_HOME", str(home / ".agents")))
    return [
        codex_home / "skills" / SKILL_NAME,
        agents_home / "skills" / SKILL_NAME,
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository skill directory (default: the parent skill of this script)",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        action="append",
        help="Destination skill directory; repeat to sync multiple homes",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report drift without writing")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    destinations = args.destination or default_destinations()
    try:
        results = sync_skill(args.source, destinations, dry_run=args.dry_run)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"skill sync failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for result in results:
            print(f"{result['status']}: {result['destination']} ({result['file_count']} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
