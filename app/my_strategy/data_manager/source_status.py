"""Data source status and fallback lineage records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from my_strategy.core.paths import METADATA_ROOT, PROJECT_ROOT

DEFAULT_STATUS_PATH = METADATA_ROOT / "data_source_status.jsonl"


@dataclass
class DataSourceStatus:
    source: str
    status: str
    fetch_time: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))
    fallback_used: bool = False
    fallback_source: str = ""
    error: str = ""
    row_count: int = 0
    date_range: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def append_source_status(status: DataSourceStatus, path: str | Path | None = DEFAULT_STATUS_PATH) -> Path:
    target = Path(path) if path is not None else DEFAULT_STATUS_PATH
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(status.to_dict(), ensure_ascii=False) + "\n")
    return target


def load_source_status(path: str | Path = DEFAULT_STATUS_PATH) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.exists():
        return []
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def clear_source_status(path: str | Path | None = DEFAULT_STATUS_PATH) -> bool:
    """Delete all recorded data-source status lines.

    Returns True if a status file existed and was removed, False if there was
    nothing to clear.
    """
    target = Path(path) if path is not None else DEFAULT_STATUS_PATH
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.exists():
        return False
    target.unlink()
    return True
