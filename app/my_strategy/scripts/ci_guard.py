"""Run the local checks for the current CZSC workbench."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent


def sh(*cmd: str, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding="utf-8", errors="replace", check=False)


def syntax_checks() -> int:
    files = [REPO_ROOT / "install.py"]
    files += [path for path in (PROJECT_ROOT / "my_strategy").rglob("*.py")
              if not any(part in {"knowledge_base", "knowledge_base_system", "data", "artifacts", "__pycache__"} for part in path.parts)]
    for path in files:
        try:
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
        except (SyntaxError, UnicodeError) as exc:
            print(f"Syntax failed: {path}: {exc}")
            return 1
    return 0


def runtime_tests() -> int:
    tests = [path.relative_to(PROJECT_ROOT).as_posix() for path in sorted((PROJECT_ROOT / "my_strategy/tests").rglob("test_*.py"))]
    scratch = REPO_ROOT / ".scratch"
    scratch.mkdir(exist_ok=True)
    temporary = tempfile.mkdtemp(prefix="pytest-ci-", dir=scratch)
    result = sh(sys.executable, "-m", "pytest", *tests, "-q", "--basetemp", temporary)
    print(result.stdout)
    return result.returncode


def doctor_smoke() -> int:
    result = sh(sys.executable, "-m", "my_strategy.cli", "doctor", "--json", "--strict")
    if result.returncode:
        print(result.stdout)
        return result.returncode
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(result.stdout)
        return 1
    return int(payload.get("status") != "success")


def docker_smoke() -> int:
    if shutil.which("docker") is None:
        print("Docker CLI unavailable; Docker check skipped.")
        return 0
    result = sh("docker", "compose", "run", "--rm", "khquant-cli", "doctor", "--expect-docker", "--check-write", "--strict")
    print(result.stdout)
    return result.returncode


BLOCKING_GATES = [syntax_checks, runtime_tests, doctor_smoke]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--skip-doctor", action="store_true")
    parser.add_argument("--docker-only", action="store_true")
    args = parser.parse_args(argv)
    if args.docker_only:
        return docker_smoke()
    for gate in BLOCKING_GATES:
        if args.skip_doctor and gate is doctor_smoke:
            continue
        print(f"CI: {gate.__name__}")
        code = gate()
        if code:
            return code
    if not args.skip_docker:
        code = docker_smoke()
        if code:
            return code
    print("All CI gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
