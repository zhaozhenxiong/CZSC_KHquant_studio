"""Build and verify a portable package of the current CZSC application."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
import sqlite3
import tempfile
from typing import Any

from my_strategy.core.paths import PROJECT_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.scripts.migration_manifest import file_sha256, verify_integrity_manifest, write_integrity_manifest
from my_strategy.scripts.snapshot_sqlite_databases import _canonical_database_pairs, copy_path_consistently, runtime_layout
from my_strategy.services.czsc_research_models import model_hash


ROOT_FILES = (".env.example", ".env.native.example", ".gitignore", "README.md", "INSTALL_NATIVE.md",
              ".gitattributes", "pyproject.toml", "setup.py", "MANIFEST.in", "runtime-manifest.json",
              "LICENSE", "install.py", "install.bat", "install.command", "khquant-native.bat", "khquant-native.sh", "khquant.sh")
EXCLUDED_NAMES = {".git", ".scratch", ".venv", ".venv-win", "venv", "__pycache__", ".pytest_cache",
                  ".ruff_cache", ".mypy_cache", ".claude", ".codex", ".serena", "node_modules", ".env",
                  "target", "build", "dist", ".runtime-build", ".runtime-wheels", "AI_CHANGELOG.md"}


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in EXCLUDED_NAMES or name.startswith(".venv")
            or name.startswith((".pytest", "_tmp_pytest", ".env.")) and not name.endswith(".example")
            or name.endswith((".egg-info", ".pyc", ".pyo", ".log", "-wal", "-shm", "-journal"))}


def _copy_application(app: Path, target: Path) -> None:
    target.mkdir(parents=True)
    for source in sorted(app.iterdir()):
        if source.name in {"data", "artifacts", "logs"} or source.name in _ignore(str(app), [source.name]):
            continue
        destination = target / source.name
        if source.name == "my_strategy":
            def strategy_ignore(directory: str, names: list[str]) -> set[str]:
                excluded = _ignore(directory, names)
                if Path(directory) == source:
                    excluded |= {"data", "artifacts", "logs", "knowledge_base", "knowledge_base_system"}
                return excluded
            copy_path_consistently(source, destination, ignore=strategy_ignore)
        else:
            copy_path_consistently(source, destination, ignore=_ignore)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run_path(runs: Path, identity: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", identity):
        raise ValueError(f"unsafe artifact run identity: {identity}")
    path = runs / identity
    if path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)()):
        raise ValueError(f"artifact run cannot be a link: {identity}")
    if not path.resolve().is_relative_to(runs.resolve()):
        raise ValueError(f"artifact run escapes storage: {identity}")
    if not path.is_dir():
        raise ValueError(f"missing artifact run dependency: {identity}")
    return path


def _artifact_path(runs: Path, relative: str) -> Path:
    path = Path(relative)
    if (not relative or path.is_absolute() or PureWindowsPath(relative).drive
            or "\\" in relative or ".." in path.parts or len(path.parts) < 2):
        raise ValueError(f"unsafe artifact path: {relative}")
    target = _run_path(runs, path.parts[0]) / Path(*path.parts[1:])
    if not target.resolve().is_relative_to(runs.resolve()) or target.is_symlink():
        raise ValueError(f"unsafe artifact path: {relative}")
    if not target.is_file():
        raise ValueError(f"missing artifact dependency: {relative}")
    return target


def _references(value: Any, runs: Path) -> set[str]:
    references: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"model_run_id", "training_run_id", "calendar_run_id", "freeze_run_id"} and item:
                references.add(_run_path(runs, str(item)).name)
            elif key == "calendar" and isinstance(item, dict) and item.get("run_id"):
                references.add(_run_path(runs, str(item["run_id"])).name)
            elif key == "report_path" and item:
                references.add(_artifact_path(runs, str(item)).relative_to(runs).parts[0])
            references.update(_references(item, runs))
    elif isinstance(value, list):
        for item in value:
            references.update(_references(item, runs))
    return references


def _czsc_runs(artifacts: Path, selection: str) -> list[Path]:
    if selection == "none":
        return []
    runs = artifacts / "runs"
    if not runs.is_dir():
        return []
    candidates = []
    for run in runs.iterdir():
        if not run.is_dir():
            continue
        reports = [run / "reports" / name for name in ("czsc.json", "research.json", "calendar.json")]
        metadata = run / "metadata.json"
        supported = any(path.is_file() for path in reports)
        if metadata.is_file():
            supported |= str(_read_json(metadata).get("task", "")).startswith("czsc-")
        if supported:
            _run_path(runs, run.name)
            candidates.append(run)
    # Preserve the existing latest CZSC-report selection when one is available.
    if selection == "latest":
        candidates = [run for run in candidates if (run / "reports/czsc.json").is_file()] or candidates
    candidates.sort(key=lambda run: ((run / "reports/czsc.json").stat().st_mtime_ns
                                    if (run / "reports/czsc.json").is_file() else run.stat().st_mtime_ns, run.name))
    return candidates[-1:] if selection == "latest" else candidates


def _release_events(database: Path) -> list[dict[str, Any]]:
    if not database.is_file():
        return []
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "czsc_model_release_events" not in tables:
            return []
        events = []
        for raw, digest in connection.execute("SELECT payload,payload_sha256 FROM czsc_model_release_events ORDER BY sequence"):
            event = json.loads(raw)
            if model_hash(event) != digest:
                raise ValueError("release event integrity mismatch")
            events.append(event)
        promotions = {event["event_id"] for event in events if event["event_type"] == "promote"}
        for event in events:
            if event.get("target_release_id") and event["target_release_id"] not in promotions:
                raise ValueError("release event references unknown promotion")
        if "czsc_model_aliases" in tables:
            for release_id, event_id in connection.execute("SELECT release_id,event_id FROM czsc_model_aliases"):
                if (not events or event_id != events[-1]["event_id"]
                        or release_id != events[-1].get("target_release_id")):
                    raise ValueError("release alias does not match the latest event")
        return events


def _artifact_closure(runs: Path, selected: list[Path], events: list[dict[str, Any]]) -> list[Path]:
    identities = {run.name for run in selected}
    for event in events:
        identities.update(_references(event, runs))
        for artifact in event.get("artifacts", []):
            path = _artifact_path(runs, artifact["path"])
            if file_sha256(path) != artifact["sha256"]:
                raise ValueError(f"release artifact hash mismatch: {artifact['path']}")
            identities.add(path.relative_to(runs).parts[0])
    pending, visited = list(sorted(identities)), set()
    while pending:
        identity = pending.pop()
        if identity in visited:
            continue
        run = _run_path(runs, identity)
        visited.add(identity)
        reports = list((run / "reports").glob("*.json"))
        metadata = run / "metadata.json"
        if metadata.is_file():
            reports.append(metadata)
        for report in reports:
            # Row-level datasets do not route to model or calendar artifacts.
            if report.name == "dataset.json":
                continue
            references = _references(json.loads(report.read_text(encoding="utf-8")), runs)
            identities.update(references)
            pending.extend(references - visited)
    return [_run_path(runs, identity) for identity in sorted(identities)]


def _verify_artifact_state(runs: Path, database: Path, selected: list[Path] | None = None) -> None:
    events = _release_events(database)
    if selected is None:
        selected = sorted(path for path in runs.iterdir() if path.is_dir()) if runs.exists() else []
    _artifact_closure(runs, selected, events)
    for run in selected:
        czsc_report = run / "reports/czsc.json"
        if czsc_report.is_file() and _read_json(czsc_report).get("run_id") != run.name:
            raise ValueError(f"CZSC report identity mismatch: {run.name}")
        model_directories = {path.parent for filename in ("manifest.json", "model.pt")
                             for path in (run / "models").glob(f"*/{filename}")}
        for directory in sorted(model_directories):
            path = directory / "manifest.json"
            manifest = _read_json(path)
            integrity = dict(manifest)
            recorded = integrity.pop("manifest_sha256", None)
            if (recorded != model_hash(integrity)
                    or manifest.get("schema_sha256") != model_hash(manifest.get("schema"))
                    or manifest.get("preprocess_sha256") != model_hash(manifest.get("preprocess"))):
                raise ValueError(f"model manifest/schema/preprocess integrity mismatch: {path}")
            checkpoint = path.parent / "model.pt"
            if not checkpoint.is_file() or file_sha256(checkpoint) != manifest.get("checkpoint_sha256"):
                raise ValueError(f"model checkpoint hash mismatch: {checkpoint}")
        report = run / "reports/research.json"
        if report.is_file():
            research = _read_json(report)
            if research.get("run_id") != run.name:
                raise ValueError(f"research report identity mismatch: {run.name}")
            calendar = research.get("calendar", {})
            calendar_root = _run_path(runs, str(calendar.get("run_id", "")))
            value = _read_json(calendar_root / "reports/calendar.json")
            dates = value.get("dates", [])
            if (not value.get("verified") or not value.get("source")
                    or stable_hash(value) != calendar.get("hash") or dates != sorted(set(dates))):
                raise ValueError(f"verified model trading calendar hash mismatch: {run.name}")
            for name in dict.fromkeys(["production", *[item["fold"]["name"] for item in research.get("evaluation", [])]]):
                model = _run_path(run / "models", name)
                manifest = _read_json(model / "manifest.json")
                if (manifest.get("label_contract", {}).get("calendar_hash") != stable_hash(dates)
                        or manifest.get("data_version") != research.get("data_version")):
                    raise ValueError(f"checkpoint/research/calendar binding mismatch: {run.name}/{name}")
    if database.is_file():
        with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            if connection.execute("SELECT 1 FROM sqlite_master WHERE name='czsc_runs'").fetchone():
                columns = {row[1] for row in connection.execute("PRAGMA table_info(czsc_runs)")}
                if "report_path" in columns:
                    for run_id, relative in connection.execute("SELECT run_id,report_path FROM czsc_runs"):
                        report = _artifact_path(runs, relative)
                        if report.relative_to(runs).parts[0] != run_id or _read_json(report).get("run_id") != run_id:
                            raise ValueError(f"result index/report identity mismatch: {run_id}")


def verify_package(package_root: Path, *, allow_runtime_additions: bool = False) -> list[str]:
    """Verify immutable payloads, allowing databases to change after installation."""
    root = package_root.resolve()
    mutable = [f"app/{relative}" for _source, relative in _canonical_database_pairs(root)] if allow_runtime_additions else []
    issues = verify_integrity_manifest(root, filename="image-manifest.json", check_sqlite=True,
                                       check_unexpected=not allow_runtime_additions, mutable_paths=mutable)
    if not issues:
        try:
            _verify_artifact_state(root / "app/my_strategy/artifacts/runs",
                                   root / "app/my_strategy/data/processed/czsc/results.db")
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            issues.append(f"artifact state invalid: {exc}")
    return issues


def _contents(source: Path, target: Path, include_data: bool, include_artifacts: str,
              include_personal: bool = False) -> dict[str, Any]:
    layout = runtime_layout(source)
    repo, app = layout["repo_root"], layout["app_root"]
    target.mkdir()
    for filename in ROOT_FILES:
        if (repo / filename).is_file():
            copy_path_consistently(repo / filename, target / filename)
    for directory in ("services", ".github"):
        if (repo / directory).is_dir():
            copy_path_consistently(repo / directory, target / directory, ignore=_ignore)
    _copy_application(app, target / "app")
    environment = ("KHQUANT_ENV_SCHEMA=2\nKHQUANT_PROJECT_ROOT=./app\nKHQUANT_VENV_DIR=./.venv\n"
                   "KHQUANT_DATA_ROOT=./my_strategy/data\nKHQUANT_ARTIFACT_ROOT=./my_strategy/artifacts\n"
                   "KHQUANT_METADATA_ROOT=./my_strategy/data/metadata\n"
                   "KHQUANT_RAW_DB=./my_strategy/data/raw/khquant_raw.db\n"
                   "KHQUANT_LOG_ROOT=./my_strategy/data/logs\nPYTHONPATH=./app\n")
    environment += "KHQUANT_COMPUTE_DEVICE=auto\nKHQUANT_CPU_WORKERS=4\nKHQUANT_GPU_BATCH_SIZE=32\n"
    (target / ".env").write_text(environment, encoding="utf-8")
    for relative in ("my_strategy/data/raw", "my_strategy/data/processed/czsc", "my_strategy/data/metadata", "my_strategy/artifacts/runs"):
        (target / "app" / relative).mkdir(parents=True, exist_ok=True)
    selected_runs = _czsc_runs(layout["artifact_root"], include_artifacts)
    copied_results = target / "app/my_strategy/data/processed/czsc/results.db"
    if include_data or include_personal:
        for database, relative in _canonical_database_pairs(source):
            is_personal = database == layout["personal_database"]
            if is_personal and not include_personal or not is_personal and not include_data:
                continue
            copy_path_consistently(database, target / "app" / relative)
    events = _release_events(copied_results)
    if events and include_artifacts == "none":
        raise ValueError("data contains immutable model releases; use --include-artifacts latest or all")
    selected_runs = _artifact_closure(layout["artifact_root"] / "runs", selected_runs, events)
    if include_data:
        if copied_results.is_file():
            with closing(sqlite3.connect(copied_results)) as connection:
                if connection.execute("SELECT name FROM sqlite_master WHERE name='czsc_runs'").fetchone():
                    keep = [run.name for run in selected_runs]
                    if keep:
                        connection.execute(f"DELETE FROM czsc_runs WHERE run_id NOT IN ({','.join('?' for _ in keep)})", keep)
                    else:
                        connection.execute("DELETE FROM czsc_runs")
                    connection.commit()
    for run in selected_runs:
        copy_path_consistently(run, target / "app/my_strategy/artifacts/runs" / run.name, ignore=_ignore)
    (target / "RUNBOOK.md").write_text(
        "# KHQuant CZSC portable package\n\nRun `python install.py` from this directory. "
        "Use `python -m my_strategy.cli package --verify .` from this directory to verify the payload. "
        "Runtime paths resolve from `app`; the raw market database is an input. "
        "Code-only is the default. Data and model/report/calendar state are optional; "
        "saved watchlist/holdings require the separate --include-personal export option. "
        "Run `python install.py --device auto` to select available CPU, CUDA or Apple MPS support.\n", encoding="utf-8")
    return write_integrity_manifest(target, source_project_root=app, filename="image-manifest.json",
                                    reject_external_symlinks=True, metadata={"architecture": "czsc",
                                    "include_data": include_data, "include_artifacts": include_artifacts,
                                     "include_personal": include_personal,
                                     "artifact_selection": "czsc_dependency_closure_v1",
                                     "release_event_count": len(events),
                                    "run_ids": [run.name for run in selected_runs]})


def build_package(*, source: Path, target: Path, include_data: bool = False, include_artifacts: str = "none",
                  include_personal: bool = False, force: bool = False) -> dict[str, Any]:
    """Stage and verify before atomically replacing an explicitly selected target."""
    if include_artifacts not in {"none", "latest", "all"}:
        raise ValueError("include_artifacts must be none, latest or all")
    layout = runtime_layout(source)
    if not (layout["repo_root"] / "install.py").is_file() or not (layout["app_root"] / "AGENTS.md").is_file():
        raise RuntimeError("package export requires a KHQuant source checkout with install.py and app/AGENTS.md; "
                           "wheel installations cannot be used as source. Use --from-project /path/to/KHQuant "
                           "to select a cloned repository or extracted source Release.")
    lexical_target = target.absolute()
    for path in (lexical_target, *lexical_target.parents):
        if path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)()):
            raise RuntimeError(f"package target cannot use symlink or junction: {path}")
    target = lexical_target.resolve()
    repo = layout["repo_root"]
    if repo.is_relative_to(target) or target.is_relative_to(layout["app_root"]):
        raise RuntimeError("package target cannot overlap the application or replace its repository")
    if target.exists() and not force:
        raise FileExistsError(f"target exists; pass --force to replace: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent)).resolve()
    staging, previous = stage_root / "package", stage_root / "previous"
    preserve_previous = False
    try:
        manifest = _contents(source, staging, include_data, include_artifacts, include_personal)
        issues = verify_package(staging)
        if issues:
            raise RuntimeError("staged package verification failed: " + "; ".join(issues))
        if target.exists():
            target.rename(previous)
        try:
            staging.rename(target)
        except BaseException:
            if previous.exists():
                preserve_previous = True
                previous.rename(target)
                preserve_previous = False
            raise
        return manifest
    finally:
        # Both paths belong to this newly-created, verified staging directory.
        if not preserve_previous and stage_root.parent == target.parent and stage_root.name.startswith(f".{target.name}.stage-"):
            shutil.rmtree(stage_root)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--target")
    destination.add_argument("--verify")
    parser.add_argument("--from-project", default=str(PROJECT_ROOT),
                        help="Source checkout or extracted source Release; required for export from a wheel installation.")
    parser.add_argument("--include-data", action="store_true")
    parser.add_argument("--include-artifacts", choices=["none", "latest", "all"], default="none",
                        help="Optional CZSC/research/calendar state with model and release evidence dependencies.")
    parser.add_argument("--include-personal", action="store_true",
                        help="Explicitly include saved watchlist and manual holdings (private by default).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify:
        verify = Path(args.verify)
        if not verify.is_absolute():
            verify = runtime_layout(Path(args.from_project))["repo_root"] / verify
        issues = verify_package(verify)
        print(json.dumps({"ok": not issues, "issues": issues}, ensure_ascii=False))
        return int(bool(issues))
    target = Path(args.target)
    if not target.is_absolute():
        target = runtime_layout(Path(args.from_project))["repo_root"] / target
    manifest = build_package(source=Path(args.from_project), target=target, include_data=args.include_data,
                             include_artifacts=args.include_artifacts, include_personal=args.include_personal,
                             force=args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2) if args.json else f"Package created: {target.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
