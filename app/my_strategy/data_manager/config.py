"""Configuration helpers for the local data center."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any

from my_strategy.core.config_loader import get_config_path, load_yaml_config
from my_strategy.core.paths import DATA_ROOT, PROJECT_ROOT

DEFAULT_CONFIG_PATH = get_config_path("data_config")


def _data_root_child(data_root: str, *parts: str) -> str:
    if data_root.startswith("/"):
        return str(PurePosixPath(data_root, *parts))
    return str(Path(data_root, *parts))


def load_data_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the data center configuration.

    Defaults to ``my_strategy/configs/data_config.yaml``. Environment variable
    ``KHQUANT_DATA_ROOT`` overrides the configured ``data_root`` and updates
    dependent stock-pool paths.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config = load_yaml_config(config_path)
    config.setdefault("data_root", str(DATA_ROOT))
    data_root_override = os.getenv("KHQUANT_DATA_ROOT", "").strip()
    if data_root_override:
        config["data_root"] = data_root_override
        stock_pool = config.setdefault("stock_pool", {})
        stock_pool["custom_pool_path"] = _data_root_child(data_root_override, "raw", "custom_pool", "my_pool.csv")
        stock_pool["all_a_cache_path"] = _data_root_child(
            data_root_override,
            "raw",
            "stock_basic",
            "all_a_stock_list.parquet",
        )
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_project_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
