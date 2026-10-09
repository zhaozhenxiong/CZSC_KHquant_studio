#!/usr/bin/env python3
"""Rebuild simple Obsidian indexes without requiring Dataview."""
from __future__ import annotations

from pathlib import Path
import datetime as dt

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
KB_ROOT = PROJECT_ROOT / "my_strategy" / "knowledge_base"


def note_title(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except OSError:
        pass
    return path.stem.replace("_", " ")


def rel_link(source: Path, target: Path) -> str:
    rel = target.relative_to(KB_ROOT).with_suffix("").as_posix()
    return f"[[{rel}|{note_title(target)}]]"


def collect(folder: str) -> list[Path]:
    root = KB_ROOT / folder
    if not root.exists():
        return []
    return sorted(
        [
            p for p in root.rglob("*.md")
            if p.name not in {"_Index.md"} and not p.name.startswith(".")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def write_index(folder: str, title: str) -> None:
    root = KB_ROOT / folder
    root.mkdir(parents=True, exist_ok=True)
    notes = collect(folder)
    lines = [
        "---",
        "type: index",
        f"updated_at: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        "---",
        "",
        f"# {title}",
        "",
    ]
    if notes:
        lines.extend(f"- {rel_link(root, note)}" for note in notes)
    else:
        lines.append("- 暂无记录")
    lines.append("")
    with (root / "_Index.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    KB_ROOT.mkdir(parents=True, exist_ok=True)
    write_index("04_AI_Changes", "AI 修改索引")
    write_index("05_AI_Sessions", "AI 会话索引")
    write_index("06_Experiments", "实验索引")
    write_index("07_Backtests", "回测索引")
    write_index("08_Data_Lineage", "数据血缘索引")
    write_index("09_Decisions", "架构决策索引")
    print(f"Obsidian indexes rebuilt under: {KB_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
