"""New CZSC run index and atomic reports; no legacy result fallback."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import uuid
from typing import Any

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, PROCESSED_DATA_ROOT
from my_strategy.core.tz import local_now


class ResultStore:
    def __init__(self, db_path: Path | None = None, runs_root: Path | None = None) -> None:
        self.db_path = db_path or PROCESSED_DATA_ROOT / "czsc" / "results.db"
        self.runs_root = runs_root or ARTIFACT_RUNS_ROOT
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS czsc_runs (run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, symbol TEXT, created_at TEXT NOT NULL, summary TEXT NOT NULL, report_path TEXT NOT NULL)")
            conn.commit()

    def save(self, kind: str, result: dict[str, Any]) -> dict[str, Any]:
        run_id = result["run_id"]
        if not re.fullmatch(r"[\w.-]+", run_id) or run_id in {".", ".."}:
            raise ValueError("Invalid run identity")
        target = self.runs_root / run_id / "reports" / "czsc.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(target)
        summary = {key: result[key] for key in ("metrics", "coverage", "data_range", "strategy_version", "model_gate", "model_run_id", "category_counts") if key in result}
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("INSERT INTO czsc_runs VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET summary=excluded.summary, report_path=excluded.report_path", (run_id, kind, result.get("symbol", ""), local_now().isoformat(), json.dumps(summary, ensure_ascii=False), target.relative_to(self.runs_root).as_posix()))
            conn.commit()
        return result

    def list(self, kind: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            where, args = ("WHERE kind=?", [kind]) if kind else ("", [])
            rows = conn.execute(f"SELECT run_id,kind,symbol,created_at,summary FROM czsc_runs {where} ORDER BY created_at DESC LIMIT ?", [*args, min(max(limit, 1), 200)]).fetchall()
        return [{**dict(row), "summary": json.loads(row["summary"])} for row in rows]

    def get(self, run_id: str) -> dict[str, Any]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute("SELECT report_path FROM czsc_runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError(run_id)
        path = (self.runs_root / row[0]).resolve()
        if not path.is_relative_to(self.runs_root.resolve()):
            raise ValueError("Report path outside run storage")
        return json.loads(path.read_text(encoding="utf-8"))
