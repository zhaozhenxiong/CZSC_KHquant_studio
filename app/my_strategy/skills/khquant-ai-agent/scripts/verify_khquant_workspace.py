"""Read-only CZSC application, raw input and AI audit verifier."""

from __future__ import annotations

import argparse
from contextlib import closing
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


REQUIRED_PATHS = (
    "AGENTS.md",
    "my_strategy/README.md",
    "my_strategy/configs",
    "my_strategy/cli.py",
    "my_strategy/scripts/update_daily_data.py",
    "my_strategy/adapters/czsc_adapter.py",
    "my_strategy/services/czsc_analysis.py",
)

PATH_KEYS = (
    "KHQUANT_ENV_SCHEMA",
    "KHQUANT_PROJECT_ROOT",
    "KHQUANT_DATA_ROOT",
    "KHQUANT_ARTIFACT_ROOT",
    "KHQUANT_LOG_ROOT",
    "KHQUANT_METADATA_ROOT",
    "KHQUANT_RAW_DB",
    "KHQUANT_VENV_DIR",
    "PYTHONPATH",
)


def find_application_root(candidate: Path) -> Path:
    """Find the app root from either the Git root or an app child path."""
    resolved = candidate.resolve()
    if resolved.is_file():
        resolved = resolved.parent
    for directory in (resolved, *resolved.parents):
        if (directory / "AGENTS.md").is_file() and (directory / "my_strategy").is_dir():
            return directory
        nested = directory / "app"
        if (nested / "AGENTS.md").is_file() and (nested / "my_strategy").is_dir():
            return nested
    raise FileNotFoundError(f"No KHQuant application root found from {candidate}")


def find_git_root(app_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=app_root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def read_env_values(path: Path) -> dict[str, str]:
    """Read the simple KEY=VALUE subset used by the repository .env file."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _resolve_business(
    value: str | Path,
    *,
    repo_root: Path,
    project_root: Path,
    schema: str | None,
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def resolve_runtime_layout(
    app_root: Path,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the same repo-relative/project-relative contract as native launchers."""
    app_root = app_root.resolve()
    repo_root = app_root.parent if app_root.name == "app" else app_root
    values = read_env_values(repo_root / ".env")
    process_env = os.environ if environ is None else environ
    for key in PATH_KEYS:
        if process_env.get(key):
            values[key] = process_env[key]

    default_application_path = "./app" if repo_root != app_root else "."
    configured_project = _resolve(values.get("KHQUANT_PROJECT_ROOT", default_application_path), repo_root)
    schema = values.get("KHQUANT_ENV_SCHEMA")
    business = {"repo_root": repo_root, "project_root": configured_project, "schema": schema}
    data_root = _resolve_business(values.get("KHQUANT_DATA_ROOT", "./my_strategy/data"), **business)
    artifact_root = _resolve_business(values.get("KHQUANT_ARTIFACT_ROOT", "./my_strategy/artifacts"), **business)
    log_root = _resolve_business(
        values.get("KHQUANT_LOG_ROOT", str(data_root / "logs")),
        **business,
    )
    metadata_root = _resolve_business(
        values.get("KHQUANT_METADATA_ROOT", str(data_root / "metadata")),
        **business,
    )
    raw_database = _resolve_business(
        values.get("KHQUANT_RAW_DB", str(data_root / "raw" / "khquant_raw.db")),
        **business,
    )
    venv_root = _resolve(values.get("KHQUANT_VENV_DIR", "./.venv-win"), repo_root)
    python_path_values = values.get("PYTHONPATH", default_application_path).split(os.pathsep)
    python_paths = [str(_resolve(value, repo_root)) for value in python_path_values if value]
    return {
        "repo_root": str(repo_root),
        "env_file": str(repo_root / ".env"),
        "env_file_exists": (repo_root / ".env").is_file(),
        "configured_project_root": str(configured_project),
        "application_root_matches": configured_project == app_root,
        "data_root": str(data_root),
        "artifact_root": str(artifact_root),
        "log_root": str(log_root),
        "metadata_root": str(metadata_root),
        "raw_database": str(raw_database),
        "results_database": str(data_root / "processed" / "czsc" / "results.db"),
        "tasks_database": str(metadata_root / "czsc_tasks.db"),
        "venv_root": str(venv_root),
        "python_paths": python_paths,
        "python_path_contains_application": str(app_root) in python_paths,
    }


def inspect_python_binding(layout: dict[str, Any]) -> dict[str, Any]:
    """Verify that the configured venv imports this checkout even under -I."""
    venv_root = Path(layout["venv_root"])
    python = venv_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    result: dict[str, Any] = {"venv_root": str(venv_root), "python": str(python), "python_exists": python.is_file()}
    if not python.is_file():
        return result
    try:
        completed = subprocess.run(
            [str(python), "-I", "-c", "import my_strategy; print(my_strategy.__file__)"],
            cwd=layout["repo_root"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.update(returncode=None, imported_file="", import_matches_application=False, error=str(exc))
        return result
    imported_file = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    expected_root = Path(layout["configured_project_root"]).resolve()
    try:
        import_matches = Path(imported_file).resolve().is_relative_to(expected_root)
    except (OSError, ValueError):
        import_matches = False
    result.update(
        returncode=completed.returncode,
        imported_file=imported_file,
        import_matches_application=bool(completed.returncode == 0 and import_matches),
        error=completed.stderr.strip()[-1000:] if completed.returncode else "",
    )
    return result


def scalar(connection: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> int | str | None:
    row = connection.execute(query, params).fetchone()
    return None if row is None else row[0]


def inspect_raw_database(database_path: Path, target_date: str | None, min_history: int) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(database_path), "exists": database_path.is_file()}
    if not database_path.is_file():
        return result
    with closing(sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        result["tables"] = sorted(tables & {"stock_daily_normalized", "securities", "data_update_runs"})
        if "stock_daily_normalized" not in tables:
            return result
        result.update(
            daily_rows=scalar(connection, "SELECT COUNT(*) FROM stock_daily_normalized"),
            daily_stocks=scalar(connection, "SELECT COUNT(DISTINCT stock) FROM stock_daily_normalized"),
            latest_date=scalar(connection, "SELECT MAX(date) FROM stock_daily_normalized"),
            duplicate_stock_dates=scalar(connection, "SELECT COUNT(*) FROM (SELECT stock, date FROM stock_daily_normalized GROUP BY stock, date HAVING COUNT(*) > 1)"),
        )
        if "securities" in tables:
            result["securities"] = scalar(connection, "SELECT COUNT(*) FROM securities")
        if target_date:
            result["target_date"] = target_date
            result["target_stocks"] = scalar(connection, "SELECT COUNT(DISTINCT stock) FROM stock_daily_normalized WHERE date = ?", (target_date,))
            result["warmup_eligible_stocks"] = scalar(
                connection,
                """SELECT COUNT(*) FROM (
                    SELECT stock FROM stock_daily_normalized WHERE date <= ? GROUP BY stock
                    HAVING MAX(CASE WHEN date = ? THEN 1 ELSE 0 END) = 1 AND COUNT(*) >= ?
                )""",
                (target_date, target_date, min_history),
            )
    return result


def inspect_sessions(session_root: Path, stale_hours: int) -> dict[str, Any]:
    now = dt.datetime.now().astimezone()
    statuses: dict[str, int] = {}
    stale: list[dict[str, str]] = []
    invalid: list[str] = []
    for path in sorted(session_root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            status = str(data.get("status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
            if status == "running" and data.get("started_at"):
                started = dt.datetime.fromisoformat(str(data["started_at"]))
                if now - started.astimezone() > dt.timedelta(hours=stale_hours):
                    stale.append({"session_id": str(data.get("session_id", path.stem)), "started_at": str(data["started_at"])})
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            invalid.append(path.name)
    return {"path": str(session_root), "statuses": statuses, "stale_running": stale, "invalid_json": invalid}


def inspect_dashboard(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=3) as response:
            return {"url": url, "status_code": response.status, "healthy": 200 <= response.status < 400}
    except (OSError, URLError) as exc:
        return {"url": url, "healthy": False, "error": str(exc)}


def verify(
    app_root: Path,
    target_date: str | None,
    min_history: int,
    stale_hours: int,
    dashboard_url: str | None,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    missing = [path for path in REQUIRED_PATHS if not (app_root / path).exists()]
    layout = resolve_runtime_layout(app_root, environ=environ)
    raw_database = Path(layout["raw_database"])
    artifact_root = Path(layout["artifact_root"])
    report: dict[str, Any] = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_root": find_git_root(app_root),
        "application_root": str(app_root),
        "runtime_layout": layout,
        "python_binding": inspect_python_binding(layout),
        "required_paths_missing": missing,
        "raw_database": inspect_raw_database(raw_database, target_date, min_history),
        "results_database": {"path": layout["results_database"], "exists": Path(layout["results_database"]).is_file()},
        "tasks_database": {"path": layout["tasks_database"], "exists": Path(layout["tasks_database"]).is_file()},
        "artifact_root": {"path": str(artifact_root), "exists": artifact_root.is_dir()},
        "artifacts_runs_exists": (artifact_root / "runs").is_dir(),
        "audit_sessions": inspect_sessions(app_root / "my_strategy/knowledge_base_system/sessions", stale_hours),
    }
    if dashboard_url:
        report["dashboard"] = inspect_dashboard(dashboard_url)
    return report, missing


def strict_errors(report: dict[str, Any], target_date: str | None, dashboard_url: str | None) -> list[str]:
    """Return current runtime and raw-input failures without mutating the report."""
    errors = list(report["required_paths_missing"])
    raw = report["raw_database"]
    layout = report["runtime_layout"]
    binding = report["python_binding"]
    if not layout["application_root_matches"]:
        errors.append(
            f"KHQUANT_PROJECT_ROOT points to {layout['configured_project_root']}, expected {report['application_root']}"
        )
    if not layout["python_path_contains_application"]:
        errors.append("PYTHONPATH does not include the current application root")
    if not binding["python_exists"]:
        errors.append(f"configured venv Python missing: {binding['python']}")
    elif not binding.get("import_matches_application"):
        errors.append("configured venv imports a different or unavailable my_strategy package")
    if not raw["exists"]:
        errors.append(raw["path"])
    if not report["artifact_root"]["exists"]:
        errors.append(report["artifact_root"]["path"])
    if target_date and not raw.get("target_stocks"):
        errors.append(f"stock_daily_normalized target date {target_date}")
    if raw.get("duplicate_stock_dates"):
        errors.append(f"duplicate stock/date keys: {raw['duplicate_stock_dates']}")
    if report["audit_sessions"]["stale_running"]:
        errors.append("stale running audit sessions")
    if dashboard_url and not report.get("dashboard", {}).get("healthy"):
        errors.append(f"dashboard unavailable: {dashboard_url}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Git root, application root, or any child path.")
    parser.add_argument("--target-date", help="Optional YYYY-MM-DD coverage date.")
    parser.add_argument("--min-history", type=int, default=60)
    parser.add_argument("--stale-session-hours", type=int, default=24)
    parser.add_argument("--dashboard-url", help="Optional URL such as http://localhost:8124/api/overview.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.min_history < 1 or args.stale_session_hours < 1:
        parser.error("history and stale-session thresholds must be positive")
    try:
        root = find_application_root(Path(args.project_root))
        report, missing = verify(root, args.target_date, args.min_history, args.stale_session_hours, args.dashboard_url)
    except (FileNotFoundError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raw = report["raw_database"]
    errors = strict_errors(report, args.target_date, args.dashboard_url) if args.strict else list(missing)
    report["status"] = "ok" if not errors else "failed"
    report["errors"] = errors
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"KHQuant app: {report['application_root']}")
        print(f"Git root: {report['git_root'] or 'unavailable'}")
        print(f"Configured project root: {report['runtime_layout']['configured_project_root']}")
        print(f"Venv import bound here: {report['python_binding'].get('import_matches_application', False)}")
        print(f"Required paths missing: {len(missing)}")
        print(f"Raw rows/stocks/latest: {raw.get('daily_rows')}/{raw.get('daily_stocks')}/{raw.get('latest_date')}")
        if args.target_date:
            print(f"Target {args.target_date}: stocks={raw.get('target_stocks')}, warmup>={args.min_history}={raw.get('warmup_eligible_stocks')}")
        print(f"Duplicate stock/date keys: {raw.get('duplicate_stock_dates')}")
        print(f"CZSC results: {report['results_database']['path']}")
        print(f"Stale running sessions: {len(report['audit_sessions']['stale_running'])}")
        print(f"Status: {report['status']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
