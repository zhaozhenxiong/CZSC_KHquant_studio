#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create an explicit SQLite backup of the canonical raw database.

The default policy is disabled.  A backup needs both ``raw_backup_enabled:
true`` in the policy file and the explicit ``--execute`` command flag.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from my_strategy.core.config_loader import get_config_path, load_yaml_config
from my_strategy.core.paths import PROJECT_ROOT
from my_strategy.storage.db_paths import raw_db_path
from my_strategy.storage.sqlite_backup import create_verified_backup


DEFAULT_POLICY = get_config_path("backup_policy")



def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_policy(path: Path) -> dict[str, Any]:
    data = load_yaml_config(path)
    if not isinstance(data, dict):
        raise ValueError("backup policy must be a mapping")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--backup-root", type=Path, default=os.environ.get("KHQUANT_BACKUP_ROOT"), help="Explicit destination or KHQUANT_BACKUP_ROOT; required with --execute")
    parser.add_argument("--execute", action="store_true", help="Perform the backup after policy approval")
    args = parser.parse_args(argv)
    policy = load_policy(_resolve(args.policy))
    source = _resolve(args.source) if args.source else raw_db_path()
    result: dict[str, Any] = {
        "source": str(source),
        "policy": str(_resolve(args.policy)),
        "enabled": bool(policy.get("raw_backup_enabled", False)),
        "execute": bool(args.execute),
    }
    if not result["enabled"]:
        result.update(status="disabled", message="Backup policy is disabled; no copy was created.")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not args.execute:
        result.update(status="dry_run", message="Policy permits backup but --execute was not supplied.")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not source.exists():
        raise FileNotFoundError(source)
    if args.backup_root is None:
        parser.error("--backup-root is required for an executed backup")
    result.update(create_verified_backup(source, _resolve(args.backup_root)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
