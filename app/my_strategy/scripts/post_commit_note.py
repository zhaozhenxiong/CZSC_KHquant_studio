#!/usr/bin/env python3
"""Create an Obsidian note for the latest Git commit."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
import subprocess

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
KB_ROOT = PROJECT_ROOT / "my_strategy" / "knowledge_base"
COMMIT_ROOT = KB_ROOT / "10_Commits"


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> int:
    sha = git("rev-parse", "HEAD")
    if not sha:
        return 0
    stamp = dt.datetime.now().astimezone()
    target = COMMIT_ROOT / stamp.strftime("%Y") / stamp.strftime("%m")
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{stamp.strftime('%Y%m%d-%H%M%S')}-{sha[:8]}.md"
    if path.exists():
        return 0

    subject = git("show", "-s", "--format=%s", "HEAD")
    author = git("show", "-s", "--format=%an <%ae>", "HEAD")
    authored = git("show", "-s", "--format=%aI", "HEAD")
    files = git("show", "--name-only", "--format=", "HEAD")
    stat = git("show", "--stat", "--format=", "HEAD")
    file_lines = "\n".join(f"- `{x}`" for x in files.splitlines() if x.strip()) or "- 无"

    path.write_text(
        f"""---
type: git-commit
commit: "{sha}"
created_at: "{stamp.isoformat(timespec='seconds')}"
tags: [khquant, git-commit]
---

# {subject}

- Commit: `{sha}`
- Author: {author}
- Authored: {authored}

## Files

{file_lines}

## Stat

```text
{stat}
```

## Links

- [[../../00_Home/KHQuant_Home]]
- [[../../04_AI_Changes/_Index]]
""",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
