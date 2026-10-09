#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initialize KHQuant Local Data Center directories and default pool."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.data_manager.config import load_data_config
from my_strategy.data_manager.local_cache import LocalCache
from my_strategy.data_manager.stock_pool import StockPoolManager


def main() -> None:
    config = load_data_config()
    cache = LocalCache(config)
    pool_path = StockPoolManager(config).ensure_default_custom_pool()
    print(f"KHQuant Local Data Center initialized: {cache.data_root}")
    print(f"Custom pool: {pool_path}")


if __name__ == "__main__":
    main()
