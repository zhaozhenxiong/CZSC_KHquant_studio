#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Print local data center update status."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from my_strategy.core.paths import PROJECT_ROOT
from my_strategy.data_manager.config import load_data_config
from my_strategy.data_manager.local_cache import LocalCache


def main() -> None:
    cache = LocalCache(load_data_config())
    path = cache.metadata_dir / "update_status.csv"
    if not path.exists():
        print(f"No update status yet: {path}")
        return
    status = pd.read_csv(path)
    print(status.sort_values(["status", "code"]).to_string(index=False))


if __name__ == "__main__":
    main()
