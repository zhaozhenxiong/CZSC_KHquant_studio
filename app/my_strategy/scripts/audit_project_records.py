#!/usr/bin/env python3
"""Audit KHQuant AI records and optionally delete exact unreferenced dispatch packets."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_id(path: Path) -> str:
    return path.name[:24]


def references_to(root: Path, target: Path) -> int:
    needle = target.name
    count = 0
    for path in root.rglob("*"):
        if not path.is_file() or path == target or path.suffix.lower() not in {".md", ".json", ".py", ".yaml", ".yml"}:
            continue
        if path.name.startswith("Project_Record_Audit_"):
            continue
        try:
            if needle in path.read_text(encoding="utf-8", errors="ignore"):
                count += 1
        except OSError:
            continue
    return count


def audit(app_root: Path, *, stale_hours: int = 24, oversized_bytes: int = 100_000) -> dict[str, Any]:
    strategy = app_root / "my_strategy"
    kb = strategy / "knowledge_base"
    system = strategy / "knowledge_base_system"
    changes = [p for p in (kb / "04_AI_Changes").rglob("*.md") if p.name != "_Index.md"]
    notes = [p for p in (kb / "05_AI_Sessions").rglob("*.md") if p.name != "_Index.md"]
    json_paths = sorted((system / "sessions").glob("*.json"))
    change_ids = {record_id(path) for path in changes}
    note_ids = {record_id(path) for path in notes}
    reconciled_note_ids = {
        record_id(path)
        for path in notes
        if "reconciled" in path.read_text(encoding="utf-8", errors="ignore")
    }
    json_ids: set[str] = set()
    invalid_json: list[str] = []
    stale_running: list[dict[str, str]] = []
    statuses: dict[str, int] = {}
    now = dt.datetime.now().astimezone()
    for path in json_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sid = str(data.get("session_id") or path.stem)
            json_ids.add(sid)
            status = str(data.get("status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
            if status == "running" and data.get("started_at"):
                started = dt.datetime.fromisoformat(str(data["started_at"]))
                if now - started.astimezone() > dt.timedelta(hours=stale_hours):
                    stale_running.append({"session_id": sid, "started_at": str(data["started_at"]), "task": str(data.get("task", ""))})
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            invalid_json.append(path.relative_to(app_root).as_posix())

    dispatch = []
    for path in sorted((system / "dispatch_packets").glob("*.md")):
        dispatch.append(
            {
                "name": path.name,
                "path": path.relative_to(app_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "references": references_to(strategy, path),
            }
        )

    index_state = []
    for folder in ("04_AI_Changes", "05_AI_Sessions", "06_Experiments", "07_Backtests", "08_Data_Lineage", "09_Decisions"):
        root = kb / folder
        index = root / "_Index.md"
        records = [p for p in root.rglob("*.md") if p.name != "_Index.md"]
        latest = max((p.stat().st_mtime for p in records), default=0)
        index_state.append({"folder": folder, "exists": index.is_file(), "stale": not index.is_file() or index.stat().st_mtime < latest})

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "application_root": str(app_root.resolve()),
        "counts": {"changes": len(changes), "session_notes": len(notes), "session_json": len(json_paths), "dispatch_packets": len(dispatch)},
        "session_statuses": statuses,
        "stale_running": stale_running,
        "invalid_session_json": invalid_json,
        "changes_without_session_note": sorted(change_ids - note_ids),
        "historical_changes_without_session_json": sorted(change_ids - json_ids),
        "session_notes_without_change": sorted(note_ids - change_ids - reconciled_note_ids),
        "reconciled_session_notes_without_change": sorted((note_ids - change_ids) & reconciled_note_ids),
        "unreferenced_dispatch_packets": [item for item in dispatch if item["references"] == 0],
        "dispatch_packets": dispatch,
        "oversized_change_notes": [
            {"path": p.relative_to(app_root).as_posix(), "bytes": p.stat().st_size}
            for p in changes if p.stat().st_size > oversized_bytes
        ],
        "indexes": index_state,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Application root containing my_strategy.")
    parser.add_argument("--stale-hours", type=int, default=24)
    parser.add_argument("--oversized-bytes", type=int, default=100_000)
    parser.add_argument("--delete-dispatch", action="append", default=[], metavar="FILENAME")
    parser.add_argument("--execute", action="store_true", help="Apply exact approved dispatch deletions.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    app_root = Path(args.project_root).resolve()
    if not (app_root / "my_strategy").is_dir() and (app_root / "app/my_strategy").is_dir():
        app_root = app_root / "app"
    report = audit(app_root, stale_hours=args.stale_hours, oversized_bytes=args.oversized_bytes)
    candidates = {item["name"]: item for item in report["unreferenced_dispatch_packets"]}
    actions = []
    for name in args.delete_dispatch:
        if Path(name).name != name:
            parser.error("--delete-dispatch accepts a filename, not a path")
        if name not in candidates:
            parser.error(f"dispatch packet is missing or referenced: {name}")
        action = {"name": name, "status": "would-delete"}
        if args.execute:
            (app_root / candidates[name]["path"]).unlink()
            action["status"] = "deleted"
        actions.append(action)
    report["actions"] = actions
    report["mode"] = "execute" if args.execute else "dry-run"
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"AI records: {report['counts']}")
        print(f"Stale running: {len(report['stale_running'])}")
        print(f"Missing session notes: {len(report['changes_without_session_note'])}")
        print(f"Unreferenced dispatch packets: {len(report['unreferenced_dispatch_packets'])}")
        print(f"Oversized change notes: {len(report['oversized_change_notes'])}")
        print(f"Mode: {report['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
