from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import unittest

from my_strategy.scripts.snapshot_sqlite_databases import (
    CANONICAL_DATABASE_PATHS,
    copy_path_consistently,
    discover_canonical_databases,
    snapshot_database,
    verify_snapshot_targets,
)


class SnapshotSqliteDatabasesTests(unittest.TestCase):
    def test_discovery_uses_canonical_database_whitelist(self) -> None:
        self.assertTrue(all(not path.is_absolute() for path in CANONICAL_DATABASE_PATHS))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            marker = root / "my_strategy" / "core" / "paths.py"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("# marker\n", encoding="utf-8")
            expected = [root / path for path in CANONICAL_DATABASE_PATHS]
            excluded = root / "my_strategy/data/processed/retired/old.db"
            for path in (*expected, excluded):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"placeholder")

            discovered = discover_canonical_databases(root)

            self.assertEqual(discovered, expected)

    def test_discovery_honors_external_runtime_paths_from_repo_or_app(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            repo = root / "repo"
            app = repo / "app"
            marker = app / "my_strategy" / "core" / "paths.py"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("# marker\n", encoding="utf-8")
            external = root / "external data"
            raw = external / "raw" / "khquant_raw.db"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"placeholder")
            (repo / ".env").write_text(
                "KHQUANT_ENV_SCHEMA=2\n"
                'KHQUANT_PROJECT_ROOT="./app"\n'
                f'KHQUANT_DATA_ROOT="{external.as_posix()}"\n',
                encoding="utf-8",
            )

            self.assertEqual(discover_canonical_databases(repo), [raw])
            self.assertEqual(discover_canonical_databases(app), [raw])

    def test_verify_snapshot_targets_accepts_verified_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "project"
            source = project / "my_strategy" / "data" / "raw" / "sample.db"
            source.parent.mkdir(parents=True)
            conn = sqlite3.connect(source)
            try:
                conn.execute("CREATE TABLE bars (date TEXT PRIMARY KEY, close REAL)")
                conn.execute("INSERT INTO bars VALUES ('2026-07-30', 10.5)")
                conn.commit()
            finally:
                conn.close()

            staging = root / "staging"
            target = root / "runtime" / "app"
            snapshot = staging / "my_strategy" / "data" / "raw" / "sample.db"
            record = snapshot_database(source, snapshot)
            record["relative_path"] = "my_strategy/data/raw/sample.db"
            manifest = staging / "sqlite-snapshot-manifest.json"
            manifest.write_text(json.dumps({"files": [record]}), encoding="utf-8")
            copied = target / record["relative_path"]
            copied.parent.mkdir(parents=True)
            shutil.copy2(snapshot, copied)

            result = verify_snapshot_targets(manifest, target)

            self.assertEqual(result[0]["relative_path"], record["relative_path"])
            self.assertEqual(result[0]["quick_check"], "ok")
            self.assertEqual(result[0]["installed_size_bytes"], copied.stat().st_size)
            self.assertTrue(result[0]["installed_sha256"])

    def test_snapshot_captures_committed_rows_from_active_wal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "live.db"
            target = root / "snapshot" / "live.db"
            writer = sqlite3.connect(source)
            try:
                writer.execute("PRAGMA journal_mode=WAL")
                writer.execute("PRAGMA wal_autocheckpoint=0")
                writer.execute("CREATE TABLE evidence (value TEXT)")
                writer.execute("INSERT INTO evidence VALUES ('committed-in-active-wal')")
                writer.commit()
                self.assertTrue(source.with_name(source.name + "-wal").exists())

                record = snapshot_database(source, target)
            finally:
                writer.close()

            self.assertEqual(record["quick_check"], "ok")
            self.assertFalse(target.with_name(target.name + "-wal").exists())
            self.assertFalse(target.with_name(target.name + "-shm").exists())
            connection = sqlite3.connect(target)
            try:
                self.assertEqual(
                    connection.execute("SELECT value FROM evidence").fetchone()[0],
                    "committed-in-active-wal",
                )
                self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "delete")
            finally:
                connection.close()

    def test_portable_copy_rejects_windows_junction(self) -> None:
        if os.name != "nt":
            self.skipTest("Windows junction test")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.txt").write_text("outside", encoding="utf-8")
            junction = root / "junction"
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                self.skipTest(f"junction creation unavailable: {completed.stderr}")
            with self.assertRaisesRegex(RuntimeError, "junction"):
                copy_path_consistently(junction, root / "copied")

    def test_verify_snapshot_targets_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.db"
            connection = sqlite3.connect(source)
            try:
                connection.execute("CREATE TABLE evidence (value TEXT)")
                connection.execute("INSERT INTO evidence VALUES ('source')")
                connection.commit()
            finally:
                connection.close()

            staging = root / "staging"
            snapshot = staging / "data" / "source.db"
            record = snapshot_database(source, snapshot)
            record["relative_path"] = "data/source.db"
            manifest = staging / "sqlite-snapshot-manifest.json"
            manifest.write_text(json.dumps({"files": [record]}), encoding="utf-8")
            target_root = root / "target"
            copied = target_root / "data" / "source.db"
            copied.parent.mkdir(parents=True)
            shutil.copy2(snapshot, copied)
            connection = sqlite3.connect(copied)
            try:
                connection.execute("INSERT INTO evidence VALUES ('tampered')")
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
                verify_snapshot_targets(manifest, target_root)


if __name__ == "__main__":
    unittest.main()
