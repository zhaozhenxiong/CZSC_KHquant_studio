"""Integrity manifests for portable KHQuant packages and workspaces."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_FILE_MANIFEST = "workspace-files.json"
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def file_sha256(path: Path) -> str:
    """Hash a regular file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _iter_payload_paths(root: Path) -> Iterable[Path]:
    """Yield files and links without following directory symlinks."""
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        directory_names[:] = [name for name in directory_names if name != "__pycache__"]
        file_names = [name for name in file_names if not name.endswith((".pyc", ".pyo"))]
        linked_directories = [name for name in directory_names if _is_link_like(current_path / name)]
        directory_names[:] = [name for name in directory_names if name not in linked_directories]
        for name in sorted([*linked_directories, *file_names]):
            yield current_path / name


def build_file_entries(
    root: Path,
    *,
    exclude: Iterable[str] = (),
    reject_external_symlinks: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Build deterministic file evidence and report whether it is portable."""
    root = root.resolve()
    excluded = {Path(value).as_posix() for value in exclude}
    entries: list[dict[str, Any]] = []
    portable = True

    for path in sorted(_iter_payload_paths(root), key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if _is_link_like(path):
            link_text = os.readlink(path)
            resolved_target = (path.parent / link_text).resolve() if not Path(link_text).is_absolute() else Path(link_text).resolve()
            external = not _is_within(resolved_target, root)
            portable = portable and not external and not Path(link_text).is_absolute()
            if reject_external_symlinks and (external or Path(link_text).is_absolute()):
                raise RuntimeError(
                    f"non-portable symlink or junction escapes workspace: {relative} -> {link_text}"
                )
            entries.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "target": link_text,
                    "external": external,
                    "bytes": len(link_text.encode("utf-8")),
                    "sha256": hashlib.sha256(link_text.encode("utf-8")).hexdigest(),
                }
            )
            continue
        if path.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return entries, portable


def write_integrity_manifest(
    root: Path,
    *,
    source_project_root: Path,
    filename: str = WORKSPACE_FILE_MANIFEST,
    metadata: dict[str, Any] | None = None,
    reject_external_symlinks: bool,
) -> dict[str, Any]:
    """Atomically write a manifest for every payload file below *root*."""
    root = root.resolve()
    entries, portable = build_file_entries(
        root,
        exclude={filename},
        reject_external_symlinks=reject_external_symlinks,
    )
    payload: dict[str, Any] = {
        "schema": "khquant-file-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_project_root": str(source_project_root.resolve()),
        "hash_algorithm": "sha256",
        "portable": portable,
        "excluded_from_hashes": [filename],
        "files": entries,
    }
    if metadata:
        payload.update(metadata)

    destination = root / filename
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return payload


def verify_integrity_manifest(
    root: Path,
    *,
    filename: str = WORKSPACE_FILE_MANIFEST,
    check_sqlite: bool = False,
    check_unexpected: bool = True,
    mutable_paths: Iterable[str] = (),
) -> list[str]:
    """Return integrity issues for a previously written manifest."""
    root = root.resolve()
    manifest_path = root / filename
    if not manifest_path.is_file():
        return [f"missing integrity manifest: {filename}"]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid integrity manifest: {filename}: {exc}"]
    records = payload.get("files")
    if not isinstance(records, list):
        return [f"invalid integrity manifest: {filename}"]

    issues: list[str] = []
    mutable = {Path(value).as_posix() for value in mutable_paths}
    expected_paths: set[str] = set()
    for record in records:
        relative = str(record.get("path", ""))
        expected_paths.add(relative)
        relative_path = Path(relative)
        raw_path = root / relative_path
        resolved_path = raw_path.resolve(strict=False)
        if relative_path.is_absolute() or not _is_within(resolved_path, root):
            issues.append(f"unsafe manifest path: {relative}")
            continue
        expected_type = record.get("type")
        if expected_type == "symlink":
            if not _is_link_like(raw_path):
                issues.append(f"missing symlink: {relative}")
                continue
            link_text = os.readlink(raw_path)
            digest = hashlib.sha256(link_text.encode("utf-8")).hexdigest()
        else:
            if not raw_path.is_file() or _is_link_like(raw_path):
                issues.append(f"missing file: {relative}")
                continue
            if relative not in mutable and raw_path.stat().st_size != int(record.get("bytes", -1)):
                issues.append(f"size mismatch: {relative}")
                continue
            digest = record.get("sha256") if relative in mutable else file_sha256(raw_path)
        if relative not in mutable and digest != record.get("sha256"):
            issues.append(f"sha256 mismatch: {relative}")
            continue
        if check_sqlite and expected_type == "file" and raw_path.suffix.lower() in SQLITE_SUFFIXES:
            try:
                uri = f"{raw_path.resolve().as_uri()}?mode=ro"
                with closing(sqlite3.connect(uri, uri=True)) as connection:
                    quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
                if quick_check != "ok":
                    issues.append(f"sqlite quick_check failed: {relative}: {quick_check}")
            except sqlite3.Error as exc:
                issues.append(f"sqlite unreadable: {relative}: {exc}")

    if check_unexpected:
        actual_entries, _ = build_file_entries(
            root,
            exclude={filename},
            reject_external_symlinks=False,
        )
        actual_paths = {str(entry["path"]) for entry in actual_entries}
        for relative in sorted(actual_paths - expected_paths):
            issues.append(f"unexpected file: {relative}")
    if payload.get("portable") is False:
        issues.append(f"manifest is not portable: {filename}")
    return issues
