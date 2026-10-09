"""Consistent SQLite backup with an isolated restore drill before publication."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from uuid import uuid4


def _check(connection: sqlite3.Connection) -> dict[str, int]:
    checks = [row[0] for row in connection.execute("PRAGMA integrity_check")]
    if checks != ["ok"]:
        raise ValueError(f"SQLite integrity check failed: {checks}")
    tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {name: connection.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0] for name in tables}


def create_verified_backup(source: Path, backup_root: Path) -> dict:
    source, backup_root = source.resolve(), backup_root.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.parent == backup_root:
        raise ValueError("Backup destination must differ from the source directory")
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
        estimated_bytes = src.execute("PRAGMA page_count").fetchone()[0] * src.execute("PRAGMA page_size").fetchone()[0]
    existing_parent = backup_root
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    # Snapshot and restored copy coexist during the restore drill; retain 1 GiB.
    required_bytes = estimated_bytes * 2 + 1024**3
    free_bytes = shutil.disk_usage(existing_parent).free
    if free_bytes < required_bytes:
        raise OSError(f"Insufficient backup space: free={free_bytes}, required={required_bytes}")
    backup_root.mkdir(parents=True, exist_ok=True)
    destination = backup_root / f"{source.stem}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex}.db"
    with tempfile.TemporaryDirectory(prefix=".backup-verify-", dir=backup_root) as temp:
        snapshot = Path(temp) / "snapshot.db"
        restored = Path(temp) / "restored.db"
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src, closing(sqlite3.connect(snapshot)) as dst:
            src.backup(dst)
            counts = _check(dst)
        # Restore the closed, checkpointed backup file. Copying an immutable
        # snapshot uses large sequential I/O; copying a live source is forbidden.
        shutil.copyfile(snapshot, restored)
        with closing(sqlite3.connect(restored.as_uri() + "?mode=ro", uri=True)) as recovery:
            if _check(recovery) != counts:
                raise ValueError("Restore table counts differ from backup")
        with snapshot.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        payload = {
            "status": "success", "source": str(source), "backup": str(destination),
            "created_at": datetime.now(timezone.utc).isoformat(), "bytes": snapshot.stat().st_size,
            "sha256": digest, "table_rows": counts, "integrity_check": "ok", "restore_verified": True,
            "scope": "single_sqlite_database", "consistency": "SQLite online backup snapshot; separate databases are not one atomic snapshot",
            "space_preflight": {"estimated_database_bytes": estimated_bytes, "required_bytes": required_bytes, "free_bytes": free_bytes},
        }
        manifest = destination.with_suffix(".manifest.json")
        pending_manifest = Path(temp) / "manifest.json"
        pending_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        snapshot.replace(destination)
        pending_manifest.replace(manifest)
    return {**payload, "manifest": str(manifest)}
