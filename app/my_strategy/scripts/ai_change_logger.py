#!/usr/bin/env python3
"""Record AI-assisted project changes into an Obsidian audit trail.

Typical usage:
    python ai_change_logger.py start --agent data-agent --task "Update daily data"
    python ai_change_logger.py finish --session-id <id> --summary "Done" --tests "pytest -q"
    python ai_change_logger.py run --agent factor-agent --task "Add factor" -- command args...
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import textwrap
import uuid
from typing import Any, Iterable

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
KB_ROOT = PROJECT_ROOT / "my_strategy" / "knowledge_base"
SYSTEM_ROOT = PROJECT_ROOT / "my_strategy" / "knowledge_base_system"
SESSION_JSON_ROOT = SYSTEM_ROOT / "sessions"
PROMPT_ROOT = SYSTEM_ROOT / "prompts"
CHANGE_ROOT = KB_ROOT / "04_AI_Changes"
SESSION_NOTE_ROOT = KB_ROOT / "05_AI_Sessions"
DIFF_ROOT = SYSTEM_ROOT / "diffs"
CHANGELOG = PROJECT_ROOT / "AI_CHANGELOG.md"

REDACTIONS = [
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)\S+"),
    re.compile(r"(?i)(token\s*[=:]\s*)\S+"),
    re.compile(r"(?i)(password\s*[=:]\s*)\S+"),
    re.compile(r"(?i)(cookie\s*[=:]\s*)\S+"),
]
MAX_DIFF_CHARS = 20_000
MAX_OUTPUT_CHARS = 50_000


def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()


def iso_now() -> str:
    return now_local().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in [SESSION_JSON_ROOT, PROMPT_ROOT, CHANGE_ROOT, SESSION_NOTE_ROOT, DIFF_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def redact(text: str) -> str:
    result = text
    for pattern in REDACTIONS:
        result = pattern.sub(r"\1<REDACTED>", result)
    return result


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n\n... <TRUNCATED {omitted} CHARS> ...\n"


def write_diff_artifact(sid: str, diff_text: str) -> str | None:
    """Store a large redacted diff as a compressed, hash-addressed sidecar."""
    if len(diff_text) <= MAX_DIFF_CHARS:
        return None
    path = DIFF_ROOT / f"{sid}.patch.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(diff_text)
    return path.relative_to(PROJECT_ROOT).as_posix()


def run_process(
    args: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=check,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(args=args, returncode=127, stdout="")


def git_output(*args: str) -> str:
    result = run_process(["git", *args])
    if result.returncode != 0:
        return ""
    # Keep the first status column: leading whitespace is meaningful in
    # porcelain output (for example " M AGENTS.md").
    return result.stdout.rstrip()


def git_available() -> bool:
    return bool(git_output("rev-parse", "--show-toplevel"))


def git_head() -> str:
    return git_output("rev-parse", "HEAD") or "NO_GIT"


def git_branch() -> str:
    return git_output("branch", "--show-current") or "NO_GIT"


def git_status() -> str:
    return git_output("status", "--short")


def git_diff() -> str:
    unstaged = git_output("diff", "--no-ext-diff")
    staged = git_output("diff", "--cached", "--no-ext-diff")
    parts = []
    if unstaged:
        parts.append("# Unstaged\n" + unstaged)
    if staged:
        parts.append("# Staged\n" + staged)
    return "\n\n".join(parts)


def git_diff_stat() -> str:
    unstaged = git_output("diff", "--stat")
    staged = git_output("diff", "--cached", "--stat")
    parts = []
    if unstaged:
        parts.append("Unstaged:\n" + unstaged)
    if staged:
        parts.append("Staged:\n" + staged)
    return "\n\n".join(parts)


def parse_status_paths(status_text: str) -> list[str]:
    paths: list[str] = []
    for line in status_text.splitlines():
        if len(line) < 4:
            continue
        raw = line[3:].strip()
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.append(raw.strip('"'))
    return sorted(set(paths))


def task_slug(task: str, limit: int = 48) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", task, flags=re.UNICODE)
    value = re.sub(r"-+", "-", value).strip("-_")
    return (value or "ai-task")[:limit]


def session_id() -> str:
    stamp = now_local().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def session_json_path(sid: str) -> Path:
    return SESSION_JSON_ROOT / f"{sid}.json"


def load_session(sid: str) -> dict[str, Any]:
    path = session_json_path(sid)
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {sid}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_session(data: dict[str, Any]) -> None:
    ensure_dirs()
    path = session_json_path(str(data["session_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def yaml_scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_prompt_snapshot(sid: str, task: str, prompt: str | None) -> str | None:
    if not prompt:
        return None
    path = PROMPT_ROOT / f"{sid}.md"
    path.write_text(
        "\n".join(
            [
                "---",
                f"session_id: {yaml_scalar(sid)}",
                f"task: {yaml_scalar(task)}",
                f"created_at: {yaml_scalar(iso_now())}",
                "type: ai-prompt",
                "---",
                "",
                "# AI Prompt",
                "",
                redact(prompt),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def start_session(
    *,
    agent: str,
    task: str,
    prompt: str | None = None,
    target_paths: Iterable[str] = (),
    command: list[str] | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    sid = session_id()
    baseline_status = git_status() if git_available() else ""
    prompt_path = write_prompt_snapshot(sid, task, prompt)
    data: dict[str, Any] = {
        "session_id": sid,
        "task": task,
        "agent": agent,
        "status": "running",
        "started_at": iso_now(),
        "ended_at": None,
        "project_root": str(PROJECT_ROOT),
        "git_available": git_available(),
        "git_branch": git_branch(),
        "baseline_commit": git_head(),
        "baseline_status": baseline_status,
        "target_paths": list(target_paths),
        "prompt_path": prompt_path,
        "command": command or [],
        "command_exit_code": None,
        "command_output": "",
        "summary": "",
        "tests": [],
        "risks": [],
        "changed_files": [],
        "final_commit": None,
        "diff_stat": "",
        "diff_sha256": "",
        "note_path": None,
    }
    save_session(data)
    return data


def markdown_list(items: Iterable[str], empty: str = "- 无") -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    return "\n".join(f"- {value}" for value in values) if values else empty


def split_multi(values: list[str] | None) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values:
        for part in re.split(r"\s*\|\s*|\s*;\s*", value):
            if part.strip():
                result.append(part.strip())
    return result


def create_change_note(data: dict[str, Any], diff_text: str) -> Path:
    ended = dt.datetime.fromisoformat(str(data["ended_at"]))
    year = ended.strftime("%Y")
    month = ended.strftime("%m")
    target_dir = CHANGE_ROOT / year / month
    target_dir.mkdir(parents=True, exist_ok=True)
    slug = task_slug(str(data["task"]))
    note_path = target_dir / f"{data['session_id']}-{slug}.md"

    changed_files = data.get("changed_files", [])
    links = []
    for file_path in changed_files:
        clean = str(file_path).replace("\\", "/")
        links.append(f"- `[[../../../../{clean}|{clean}]]`")

    diff_text = "\n".join(
        line.rstrip()
        for line in truncate(redact(diff_text), MAX_DIFF_CHARS).splitlines()
    )
    command_output = truncate(
        redact(str(data.get("command_output", ""))),
        MAX_OUTPUT_CHARS,
    )
    command_display = " ".join(
        shlex.quote(str(part)) for part in data.get("command", [])
    )

    content = f"""---
type: ai-change
session_id: {yaml_scalar(str(data['session_id']))}
task: {yaml_scalar(str(data['task']))}
agent: {yaml_scalar(str(data['agent']))}
status: {yaml_scalar(str(data['status']))}
started_at: {yaml_scalar(str(data['started_at']))}
ended_at: {yaml_scalar(str(data['ended_at']))}
baseline_commit: {yaml_scalar(str(data['baseline_commit']))}
final_commit: {yaml_scalar(str(data.get('final_commit') or 'UNCOMMITTED'))}
tags:
  - khquant
  - ai-change
  - agent/{data['agent']}
---

# {data['task']}

## 摘要

{data.get('summary') or '未填写'}

## Agent

- 主责：[[../../02_Agents/Agent_Router|{data['agent']}]]
- 会话：`{data['session_id']}`
- 状态：`{data['status']}`
- 分支：`{data.get('git_branch', '')}`
- 基线提交：`{data.get('baseline_commit', '')}`
- 最终提交：`{data.get('final_commit') or 'UNCOMMITTED'}`

## 修改文件

{chr(10).join(links) if links else '- 未检测到文件变化'}

## 测试与验证

{markdown_list(data.get('tests', []))}

## 风险与限制

{markdown_list(data.get('risks', []))}

## 执行命令

```text
{command_display or '手工/外部调用'}
```

退出码：`{data.get('command_exit_code')}`

## Git Diff Stat

```text
{data.get('diff_stat') or '无'}
```

## Diff Evidence

- SHA-256: `{data.get('diff_sha256') or '无'}`
- 完整压缩 diff：`{data.get('diff_artifact_path') or '未生成（diff 未超过内联上限）'}`

## 命令输出

```text
{command_output or '无'}
```

## Git Diff

```diff
{diff_text or '# No diff captured'}
```

## 关联页面

- [[../../00_Home/KHQuant_Home]]
- [[../../01_Project/Architecture]]
- [[../../03_Workflows/AI_Change_Workflow]]
- [[../../09_Decisions/Architecture_Decision_Log]]
"""
    note_path.write_text(content, encoding="utf-8")
    return note_path


def append_changelog(data: dict[str, Any], note_path: Path) -> None:
    if not CHANGELOG.exists():
        CHANGELOG.write_text(
            "# KHQuant AI Change Log\n\n"
            "> 自动记录由 AI、Agent 或包装命令产生的项目修改。\n\n",
            encoding="utf-8",
        )
    rel_note = note_path.relative_to(PROJECT_ROOT).as_posix()
    sid = str(data["session_id"])
    line = (
        f"- {data['ended_at']} | `{data['agent']}` | session `{sid}` | "
        f"`{data['status']}` | [{data['task']}]({rel_note}) | "
        f"{len(data.get('changed_files', []))} files | "
        f"{data.get('final_commit') or 'UNCOMMITTED'}\n"
    )
    existing = [
        item for item in CHANGELOG.read_text(encoding="utf-8").splitlines()
        if sid not in item
    ]
    with CHANGELOG.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(existing).rstrip() + "\n" + line)


def rebuild_obsidian_indexes() -> bool:
    script = SCRIPT_PATH.with_name("rebuild_obsidian_indexes.py")
    if not script.is_file():
        return False
    return run_process([sys.executable, str(script)]).returncode == 0


def create_session_note(data: dict[str, Any]) -> Path:
    ended = dt.datetime.fromisoformat(str(data["ended_at"]))
    target_dir = SESSION_NOTE_ROOT / ended.strftime("%Y") / ended.strftime("%m")
    target_dir.mkdir(parents=True, exist_ok=True)
    note_path = target_dir / f"{data['session_id']}.md"
    change_note = data.get("note_path", "")
    content = f"""---
type: ai-session
session_id: {yaml_scalar(str(data['session_id']))}
agent: {yaml_scalar(str(data['agent']))}
status: {yaml_scalar(str(data['status']))}
started_at: {yaml_scalar(str(data['started_at']))}
ended_at: {yaml_scalar(str(data['ended_at']))}
tags: [khquant, ai-session]
---

# AI Session {data['session_id']}

- 任务：{data['task']}
- Agent：{data['agent']}
- 状态：{data['status']}
- 变更记录：`{change_note}`
- Prompt 快照：`{data.get('prompt_path') or '无'}`
- 命令退出码：`{data.get('command_exit_code')}`
"""
    note_path.write_text(content, encoding="utf-8")
    return note_path


def finish_session(
    *,
    sid: str,
    summary: str = "",
    tests: Iterable[str] = (),
    risks: Iterable[str] = (),
    exit_code: int | None = None,
    command_output: str | None = None,
    explicit_status: str | None = None,
) -> dict[str, Any]:
    data = load_session(sid)
    current_status = git_status() if data.get("git_available") else ""
    diff_text = git_diff() if data.get("git_available") else ""
    changed_files = parse_status_paths(current_status)
    final_commit = git_head() if data.get("git_available") else "NO_GIT"

    if explicit_status:
        status = explicit_status
    elif exit_code is None or exit_code == 0:
        status = "success"
    else:
        status = "failed"

    redacted_diff = redact(diff_text)
    data.update(
        {
            "status": status,
            "ended_at": iso_now(),
            "summary": summary,
            "tests": list(tests),
            "risks": list(risks),
            "command_exit_code": exit_code,
            "command_output": command_output or data.get("command_output", ""),
            "changed_files": changed_files,
            "final_commit": final_commit,
            "diff_stat": git_diff_stat() if data.get("git_available") else "",
            "diff_sha256": hashlib.sha256(redacted_diff.encode("utf-8", errors="replace")).hexdigest(),
            "diff_artifact_path": write_diff_artifact(sid, redacted_diff),
        }
    )

    note_path = create_change_note(data, diff_text)
    data["note_path"] = str(note_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    session_note = create_session_note(data)
    data["session_note_path"] = str(
        session_note.relative_to(PROJECT_ROOT)
    ).replace("\\", "/")
    append_changelog(data, note_path)
    data["indexes_rebuilt"] = rebuild_obsidian_indexes()
    save_session(data)
    return data


def command_start(args: argparse.Namespace) -> int:
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    data = start_session(
        agent=args.agent,
        task=args.task,
        prompt=prompt,
        target_paths=args.path or [],
    )
    print(data["session_id"])
    return 0


def command_finish(args: argparse.Namespace) -> int:
    data = finish_session(
        sid=args.session_id,
        summary=args.summary or "",
        tests=split_multi(args.tests),
        risks=split_multi(args.risks),
        exit_code=args.exit_code,
        explicit_status=args.status,
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def command_run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("No wrapped command supplied.", file=sys.stderr)
        return 2

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    data = start_session(
        agent=args.agent,
        task=args.task,
        prompt=prompt,
        target_paths=args.path or [],
        command=command,
    )
    sid = str(data["session_id"])
    print(f"[KHQuant] session={sid}", flush=True)

    result = run_process(command, cwd=PROJECT_ROOT)
    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")

    summary = args.summary or (
        "包装命令执行完成。" if result.returncode == 0 else "包装命令执行失败。"
    )
    final = finish_session(
        sid=sid,
        summary=summary,
        tests=split_multi(args.tests),
        risks=split_multi(args.risks),
        exit_code=result.returncode,
        command_output=output,
    )
    print(f"[KHQuant] change_note={final['note_path']}")
    return int(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    start = sub.add_parser("start", help="Start an AI modification session.")
    start.add_argument("--agent", required=True)
    start.add_argument("--task", required=True)
    start.add_argument("--prompt")
    start.add_argument("--prompt-file")
    start.add_argument("--path", action="append")
    start.set_defaults(func=command_start)

    finish = sub.add_parser("finish", help="Finish and record a session.")
    finish.add_argument("--session-id", required=True)
    finish.add_argument("--summary")
    finish.add_argument("--tests", action="append")
    finish.add_argument("--risks", action="append")
    finish.add_argument("--exit-code", type=int)
    finish.add_argument(
        "--status",
        choices=["success", "partial", "failed", "blocked"],
    )
    finish.set_defaults(func=command_finish)

    run = sub.add_parser("run", help="Wrap a command and record its changes.")
    run.add_argument("--agent", required=True)
    run.add_argument("--task", required=True)
    run.add_argument("--prompt")
    run.add_argument("--prompt-file")
    run.add_argument("--path", action="append")
    run.add_argument("--summary")
    run.add_argument("--tests", action="append")
    run.add_argument("--risks", action="append")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=command_run)

    return parser


def main() -> int:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
