"""Create consistent, portable SQLite snapshots for a KHQuant workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable


SNAPSHOT_PURPOSES = {"cross_drive_migration", "schema_migration", "maintenance"}
CANONICAL_DATABASE_PATHS = (
    Path("my_strategy/data/raw/khquant_raw.db"),
    Path("my_strategy/data/processed/czsc/results.db"),
    Path("my_strategy/data/metadata/czsc_tasks.db"),
    Path("my_strategy/data/metadata/personal_portfolio.db"),
)
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def sha256(path: Path) -> str:
    """Return a SHA-256 hash without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_database_metadata(path: Path) -> dict[str, object]:
    """Return the target database evidence retained by a runtime manifest."""
    target_uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(target_uri, uri=True)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        page_count = connection.execute("PRAGMA page_count").fetchone()[0]
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        has_meta = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='warehouse_meta'"
        ).fetchone()
        application_schema_version = None
        if has_meta:
            row = connection.execute("SELECT value FROM warehouse_meta WHERE key='schema_version'").fetchone()
            application_schema_version = row[0] if row else None
    finally:
        connection.close()
    return {
        "installed_size_bytes": path.stat().st_size,
        "installed_sha256": sha256(path),
        "quick_check": quick_check,
        "page_count": page_count,
        "page_size": page_size,
        "sqlite_user_version": user_version,
        "application_schema_version": application_schema_version,
    }


def snapshot_database(source: Path, target: Path) -> dict[str, object]:
    """Back up one read-only SQLite database and atomically publish the copy."""
    partial = target.with_suffix(f"{target.suffix}.partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists() or target.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot target: {target}")

    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    source_connection = sqlite3.connect(source_uri, uri=True)
    target_connection: sqlite3.Connection | None = None
    completed = False
    try:
        target_connection = sqlite3.connect(partial)
        source_connection.backup(target_connection, pages=8192, sleep=0.05)
        # A source in WAL mode persists that journal setting in the backup.
        # Normalize the staged database so the published snapshot is one
        # self-contained file and does not require copied -wal/-shm sidecars.
        journal_mode = target_connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        if str(journal_mode).lower() != "delete":
            raise RuntimeError(f"failed to normalize snapshot journal mode for {source}: {journal_mode}")
        quick_check = target_connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise RuntimeError(f"quick_check failed for {source}: {quick_check}")
        page_count = target_connection.execute("PRAGMA page_count").fetchone()[0]
        page_size = target_connection.execute("PRAGMA page_size").fetchone()[0]
        completed = True
    finally:
        if target_connection is not None:
            target_connection.close()
        source_connection.close()
        if not completed:
            partial.unlink(missing_ok=True)

    os.replace(partial, target)
    return {
        "source_size_bytes": source.stat().st_size,
        "snapshot_size_bytes": target.stat().st_size,
        "snapshot_sha256": sha256(target),
        "page_count": page_count,
        "page_size": page_size,
        "quick_check": quick_check,
    }


def _looks_like_sqlite(path: Path) -> bool:
    return path.suffix.lower() in SQLITE_SUFFIXES


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def copy_path_consistently(
    source: Path,
    target: Path,
    *,
    ignore: Callable[[str, list[str]], set[str]] | None = None,
) -> None:
    """Copy a migration path, snapshotting SQLite files instead of copying WAL state."""
    if _is_link_like(source):
        raise RuntimeError(f"refusing to copy symlink or junction into portable payload: {source}")
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if _looks_like_sqlite(source):
            snapshot_database(source, target)
        else:
            shutil.copy2(source, target)
        return
    if not source.is_dir():
        raise FileNotFoundError(source)

    target.mkdir(parents=True, exist_ok=True)
    names = sorted(path.name for path in source.iterdir())
    ignored = ignore(str(source), names) if ignore else set()
    for name in names:
        if name in ignored or name.endswith(SQLITE_SIDECAR_SUFFIXES):
            continue
        copy_path_consistently(source / name, target / name, ignore=ignore)


def discover_app_root(project_root: Path) -> Path:
    """Accept either a Git repository root or the application root."""
    project_root = project_root.resolve()
    candidates = (project_root, project_root / "app")
    for candidate in candidates:
        if (candidate / "my_strategy" / "core" / "paths.py").is_file():
            return candidate
    raise RuntimeError(f"source does not look like a KHQuant project: {project_root}")


def _read_env_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip():
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _resolve_business_path(value: str | Path, *, app_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (app_root / path).resolve()


def runtime_layout(project_root: Path) -> dict[str, Path]:
    """Resolve repo/app roots and the current app-relative runtime paths."""
    source = project_root.resolve()
    app_root = discover_app_root(source)
    if source != app_root:
        repo_root = source
    elif app_root.name.casefold() == "app" and (app_root.parent / "app").resolve() == app_root:
        repo_root = app_root.parent
    else:
        repo_root = app_root

    values = _read_env_values(repo_root / ".env")
    current_app_root = Path(__file__).resolve().parents[2]
    if app_root == current_app_root:
        for key in (
            "KHQUANT_PROJECT_ROOT",
            "KHQUANT_DATA_ROOT",
            "KHQUANT_ARTIFACT_ROOT",
            "KHQUANT_METADATA_ROOT",
            "KHQUANT_RAW_DB",
        ):
            if os.environ.get(key):
                values[key] = os.environ[key]
    configured = Path(values.get("KHQUANT_PROJECT_ROOT", str(app_root))).expanduser()
    configured = configured.resolve() if configured.is_absolute() else (repo_root / configured).resolve()
    if configured != app_root:
        raise RuntimeError(f"KHQUANT_PROJECT_ROOT points to {configured}, expected {app_root}")
    data_root = _resolve_business_path(
        values.get("KHQUANT_DATA_ROOT", "./my_strategy/data"),
        app_root=app_root,
    )
    artifact_root = _resolve_business_path(
        values.get("KHQUANT_ARTIFACT_ROOT", "./my_strategy/artifacts"),
        app_root=app_root,
    )
    metadata_root = _resolve_business_path(values.get("KHQUANT_METADATA_ROOT", str(data_root / "metadata")), app_root=app_root)
    return {
        "repo_root": repo_root,
        "app_root": app_root,
        "data_root": data_root,
        "artifact_root": artifact_root,
        "metadata_root": metadata_root,
        "raw_database": _resolve_business_path(
            values.get("KHQUANT_RAW_DB", str(data_root / "raw" / "khquant_raw.db")),
            app_root=app_root,
        ),
        "results_database": data_root / "processed" / "czsc" / "results.db",
        "tasks_database": metadata_root / "czsc_tasks.db",
        "personal_database": metadata_root / "personal_portfolio.db",
    }


def _canonical_database_pairs(project_root: Path) -> list[tuple[Path, Path]]:
    """Select raw input, CZSC results/tasks, and saved personal data."""
    layout = runtime_layout(project_root)
    configured = [layout[key] for key in ("raw_database", "results_database", "tasks_database", "personal_database")]
    return [
        (source_path, relative)
        for source_path, relative in zip(configured, CANONICAL_DATABASE_PATHS, strict=True)
        if source_path.is_file()
    ]


def discover_canonical_databases(project_root: Path) -> list[Path]:
    """Return only the formal runtime databases eligible for migration backup."""
    return [source for source, _relative in _canonical_database_pairs(project_root)]


def verify_snapshot_targets(manifest_path: Path, target_root: Path) -> list[dict[str, object]]:
    """Verify copied snapshot databases before their staging directory is removed."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"snapshot manifest has no files: {manifest_path}")

    target_root = target_root.resolve()
    allowed = {
        Path(str(value)).as_posix()
        for value in (manifest.get("selection_policy") or {}).get("allowed_relative_paths", [])
    }
    results: list[dict[str, object]] = []
    for record in records:
        relative = Path(str(record["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe snapshot manifest path: {relative}")
        relative_text = relative.as_posix()
        if allowed and relative_text not in allowed:
            raise ValueError(f"snapshot path is outside the manifest whitelist: {relative}")
        target = (target_root / relative).resolve()
        if not target.is_relative_to(target_root):
            raise ValueError(f"snapshot target escapes verification root: {relative}")
        if not target.is_file():
            raise FileNotFoundError(f"snapshot target is missing: {target}")
        expected_size = int(record["snapshot_size_bytes"])
        if target.stat().st_size != expected_size:
            raise RuntimeError(f"snapshot target size mismatch: {target}")
        target_metadata = installed_database_metadata(target)
        if target_metadata["installed_sha256"] != record.get("snapshot_sha256"):
            raise RuntimeError(f"snapshot target sha256 mismatch: {target}")
        quick_check = target_metadata["quick_check"]
        page_count = target_metadata["page_count"]
        page_size = target_metadata["page_size"]
        if quick_check != "ok":
            raise RuntimeError(f"snapshot target quick_check failed for {target}: {quick_check}")
        if page_count != int(record["page_count"]) or page_size != int(record["page_size"]):
            raise RuntimeError(f"snapshot target page layout mismatch: {target}")
        results.append(
            {
                "relative_path": relative.as_posix(),
                "quick_check": quick_check,
                "page_count": page_count,
                "page_size": page_size,
                **target_metadata,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="KHQuant repository root.")
    parser.add_argument("--destination", help="New temporary staging directory; it must not exist.")
    parser.add_argument("--purpose", choices=sorted(SNAPSHOT_PURPOSES), help="Why this temporary snapshot is required.")
    parser.add_argument("--manifest", help="Existing snapshot manifest to verify.")
    parser.add_argument("--verify-target-root", help="Copied runtime root to verify against --manifest.")
    args = parser.parse_args()

    if args.verify_target_root or args.manifest:
        if not args.verify_target_root or not args.manifest:
            parser.error("--verify-target-root and --manifest must be supplied together")
        if args.destination or args.purpose:
            parser.error("verification cannot be combined with snapshot creation options")
        results = verify_snapshot_targets(Path(args.manifest).resolve(), Path(args.verify_target_root).resolve())
        print(json.dumps({"status": "ok", "files": results}, ensure_ascii=False), flush=True)
        return 0

    if not args.destination or not args.purpose:
        parser.error("--destination and --purpose are required to create a temporary snapshot")

    project_root = Path(args.project_root).resolve()
    destination = Path(args.destination).resolve()
    if destination.exists():
        parser.error(f"destination already exists: {destination}")

    database_pairs = _canonical_database_pairs(project_root)
    if not database_pairs:
        parser.error(f"no canonical SQLite databases found below {project_root}")

    destination.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for index, (source, relative) in enumerate(database_pairs, start=1):
        print(f"[{index}/{len(database_pairs)}] snapshot {relative}", flush=True)
        record = snapshot_database(source, destination / relative)
        record["relative_path"] = relative.as_posix()
        records.append(record)
        print(f"[{index}/{len(database_pairs)}] verified {relative}", flush=True)

    manifest = {
        "schema": "khquant-czsc-snapshot-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_project": str(project_root),
        "purpose": args.purpose,
        "retention": "temporary_staging",
        "selection_policy": {
            "mode": "canonical_whitelist",
            "allowed_relative_paths": [str(record["relative_path"]) for record in records],
            "excluded_by_design": ["all non-canonical SQLite files"],
        },
        "method": "sqlite3.Connection.backup with read-only source connections",
        "files": records,
    }
    (destination / "sqlite-snapshot-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"complete: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
