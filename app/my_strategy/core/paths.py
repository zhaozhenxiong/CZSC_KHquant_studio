"""Single path boundary for raw inputs and CZSC outputs."""
from __future__ import annotations

import os
from pathlib import Path
import re


PROJECT_ROOT = Path(os.environ.get("KHQUANT_PROJECT_ROOT") or Path(__file__).resolve().parents[2]).resolve()
MY_STRATEGY_ROOT = PROJECT_ROOT / "my_strategy"


def _root(name: str, default: Path) -> Path:
    value = Path(os.environ.get(name) or default).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


DATA_ROOT = _root("KHQUANT_DATA_ROOT", MY_STRATEGY_ROOT / "data")
RAW_DATA_ROOT = DATA_ROOT / "raw"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
ARTIFACT_ROOT = _root("KHQUANT_ARTIFACT_ROOT", MY_STRATEGY_ROOT / "artifacts")
ARTIFACT_RUNS_ROOT = ARTIFACT_ROOT / "runs"
LOG_ROOT = _root("KHQUANT_LOG_ROOT", DATA_ROOT / "logs")
METADATA_ROOT = _root("KHQUANT_METADATA_ROOT", DATA_ROOT / "metadata")
STOCK_METADATA_CSV_PATH = METADATA_ROOT / "stock_metadata.csv"
ALL_A_STOCK_LIST_PATH = RAW_DATA_ROOT / "stock_basic" / "all_a_stock_list.parquet"
CUSTOM_POOL_PATH = RAW_DATA_ROOT / "custom_pool" / "my_pool.csv"


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def artifact_run_dir(run_id: str, *, create: bool = True) -> Path:
    if not re.fullmatch(r"[\w.-]+", run_id) or run_id in {".", ".."}:
        raise ValueError("run_id must be a single safe path component")
    path = ARTIFACT_RUNS_ROOT / run_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_subdir(run_id: str, *parts: str, create: bool = True) -> Path:
    root = artifact_run_dir(run_id, create=create)
    original = root.joinpath(*parts)
    # Windows realpath can retain a \\?\ prefix if a shared parent appears
    # during resolution. Retry the original path; true escapes still fail.
    for _ in range(3):
        path = original.resolve()
        if path.is_relative_to(root.resolve()):
            break
    else:
        raise ValueError("artifact path must remain inside its run")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
