"""Raw market-data location and ingestion modes."""
from __future__ import annotations

import os
from pathlib import Path

from my_strategy.core.paths import PROJECT_ROOT, RAW_DATA_ROOT
from my_strategy.core.storage_layout import resolve_path

DEFAULT_RAW_DB = RAW_DATA_ROOT / "khquant_raw.db"


def raw_db_path() -> Path:
    return resolve_path(os.environ.get("KHQUANT_RAW_DB") or str(DEFAULT_RAW_DB), PROJECT_ROOT)


def storage_mode() -> str:
    """Raw ingestion may use SQLite, source files, or both."""
    value = os.environ.get("KHQUANT_STORAGE_MODE", "db").strip().lower()
    return value if value in {"db", "dual", "file"} else "db"


def read_db_first() -> bool:
    return storage_mode() in {"db", "dual"}


def write_db() -> bool:
    return storage_mode() in {"db", "dual"}


def write_files() -> bool:
    return storage_mode() in {"file", "dual"}
