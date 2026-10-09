"""Resolve a configured runtime path from the application root."""
from pathlib import Path


def resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()
