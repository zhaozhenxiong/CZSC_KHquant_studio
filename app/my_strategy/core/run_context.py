"""Run identity and artifact metadata helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from my_strategy.core.paths import PROJECT_ROOT, artifact_run_dir, artifact_subdir
from my_strategy.core.tz import local_now


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def current_git_commit(short: bool = False) -> str:
    commit = _git_output("rev-parse", "--short" if short else "HEAD")
    return commit or "NO_GIT"


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def slugify(value: str, limit: int = 48) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", str(value), flags=re.UNICODE)
    text = re.sub(r"-+", "-", text).strip("-_.")
    return (text or "run")[:limit]


_VALID_SCOPES = frozenset({"single", "full", "batch", "train", "daily_update", "implementation"})
_VALID_SOURCES = frozenset({"web", "cli", "api", "scheduled"})


def _validate_choice(value: str, name: str, valid: frozenset[str]) -> None:
    if value and value not in valid:
        raise ValueError(f"{name} must be one of {sorted(valid)}, got {value!r}")


def scope_from_task(task: str) -> str:
    """Infer a canonical run scope from a free-form task name."""
    task_lower = task.lower()
    if "daily" in task_lower or "update" in task_lower:
        return "daily_update"
    if "train" in task_lower:
        return "train"
    if "batch" in task_lower:
        return "batch"
    # "full" backtests/screenings take precedence over the generic "backtest"
    # keyword so task names like "web-full-backtest" map correctly.
    if "screen" in task_lower or "full" in task_lower:
        return "full"
    if "backtest" in task_lower:
        return "single"
    if "impl" in task_lower:
        return "implementation"
    return "single"


@dataclass(frozen=True)
class RunContext:
    run_id: str
    as_of_date: str
    git_commit: str
    config_hash: str
    data_version: str
    seed: int = 42
    created_at: str = field(default_factory=lambda: local_now().isoformat(timespec="seconds"))
    task: str = ""
    artifact_root: str = ""
    scope: str = ""
    stocks: list[str] = field(default_factory=list)
    source: str = ""
    start_date: str = ""
    end_date: str = ""

    def run_dir(self) -> Path:
        return artifact_run_dir(self.run_id)

    def subdir(self, *parts: str) -> Path:
        return artifact_subdir(self.run_id, *parts)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifact_root"] = str(self.run_dir())
        return data

    def write_metadata(self, extra: dict[str, Any] | None = None) -> Path:
        payload = self.to_dict()
        if extra:
            payload.update(extra)
        path = self.run_dir() / "metadata.json"
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path


def make_run_id(task: str = "run", as_of_date: str | date | None = None, commit: str | None = None) -> str:
    stamp = local_now().strftime("%Y%m%d-%H%M%S-%f")
    date_part = str(as_of_date or local_now().date()).replace("-", "")
    git_part = (commit or current_git_commit(short=True))[:12]
    return f"{stamp}-{date_part}-{slugify(task, 36)}-{git_part}"


def create_run_context(
    *,
    task: str = "run",
    as_of_date: str | date | None = None,
    config: dict[str, Any] | None = None,
    data_version: str = "",
    seed: int = 42,
    run_id: str | None = None,
    scope: str | None = None,
    stocks: list[str] | None = None,
    source: str | None = None,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
) -> RunContext:
    commit = current_git_commit()
    effective_date = str(as_of_date or local_now().date())
    inferred_scope = scope_from_task(task) if scope is None else scope
    _validate_choice(inferred_scope, "scope", _VALID_SCOPES)
    inferred_source = source or "cli"
    _validate_choice(inferred_source, "source", _VALID_SOURCES)

    context = RunContext(
        run_id=run_id or make_run_id(task=task, as_of_date=effective_date, commit=commit),
        as_of_date=effective_date,
        git_commit=commit,
        config_hash=stable_hash(config or {}),
        data_version=data_version or "local",
        seed=int(seed),
        task=task,
        scope=inferred_scope,
        stocks=list(stocks) if stocks is not None else [],
        source=inferred_source,
        start_date=str(start_date or effective_date),
        end_date=str(end_date or effective_date),
    )
    context.run_dir()
    for name in ["config", "data_snapshots", "signals", "backtests", "reports", "logs", "validation"]:
        context.subdir(name)
    context.write_metadata({"config": config or {}})
    return context


def load_run_metadata(run_id: str) -> dict[str, Any]:
    path = artifact_run_dir(run_id, create=False) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))
