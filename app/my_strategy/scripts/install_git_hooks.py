#!/usr/bin/env python3
"""Install KHQuant Git hooks without silently overwriting custom hooks."""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import stat
import subprocess
import sys

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SOURCE_DIR = PROJECT_ROOT / "tools" / "git_hooks"


def git_dir() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError("Not a Git repository. Run git init first.")
    path = Path(result.stdout.strip())
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def install_hook(name: str) -> None:
    source = SOURCE_DIR / name
    target = git_dir() / "hooks" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(source)

    if target.exists() and target.read_bytes() != source.read_bytes():
        backup = target.with_name(target.name + ".pre-khquant-backup")
        if not backup.exists():
            shutil.copy2(target, backup)
            print(f"Backed up existing hook to {backup}")

    shutil.copy2(source, target)
    if os.name != "nt":
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
    print(f"Installed {name}: {target}")


def main() -> int:
    try:
        install_hook("post-commit")
    except Exception as exc:
        print(f"Hook installation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
