"""Normalize portable native-runtime paths before importing path-dependent code."""

from __future__ import annotations

import os
from pathlib import Path
import sysconfig


def runtime_resource_root() -> Path:
    """Locate frozen resources in a checkout or an installed KHQuant wheel."""
    package = Path(__file__).resolve().parent
    source = package.parent
    return source if (source / "czsc-source-manifest.json").is_file() else package / "_runtime"


def initialize_runtime_environment() -> None:
    """Keep checkout configuration, and put wheel state outside site-packages."""
    package = Path(__file__).resolve().parent
    application = package.parent
    if (application / "AGENTS.md").is_file():
        repository = application.parent if application.name == "app" else application
        environment = repository / ".env"
        if environment.is_file():
            for line in environment.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        normalize_native_runtime_paths(repository)
        return
    state = Path(os.environ.get("KHQUANT_HOME") or Path.home() / ".khquant").expanduser().resolve()
    os.environ.setdefault("KHQUANT_PROJECT_ROOT", str(application))
    data = Path(os.environ.get("KHQUANT_DATA_ROOT") or state / "data")
    for key, value in {
        "KHQUANT_DATA_ROOT": data,
        "KHQUANT_ARTIFACT_ROOT": state / "artifacts",
        "KHQUANT_METADATA_ROOT": data / "metadata",
        "KHQUANT_LOG_ROOT": data / "logs",
        "KHQUANT_RAW_DB": data / "raw" / "khquant_raw.db",
    }.items():
        os.environ.setdefault(key, str(value))
    normalize_native_runtime_paths(application)


_PROJECT_PATH_VARS = (
    "KHQUANT_DATA_ROOT",
    "KHQUANT_ARTIFACT_ROOT",
    "KHQUANT_LOG_ROOT",
    "KHQUANT_METADATA_ROOT",
    "KHQUANT_RAW_DB",
)


def normalize_native_runtime_paths(repo_root: str | Path) -> None:
    """Resolve portable ``.env`` paths to absolute paths in the process.

    ``KHQUANT_PROJECT_ROOT`` is relative to the repository. Other project
    paths are relative to that resolved project root, matching the generated
    native ``.env`` contract (for example ``./my_strategy/data``).
    """

    repo = Path(repo_root).resolve()
    # Some task hosts omit this standard Windows variable; restore the value
    # from the interpreter's verified target before Polars performs CPUID checks.
    if sysconfig.get_platform() == "win-amd64":
        os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
    project_value = os.environ.get("KHQUANT_PROJECT_ROOT", "")
    project_root = _resolve(project_value, repo) if project_value else repo / "app"
    os.environ["KHQUANT_PROJECT_ROOT"] = str(project_root)

    schema = os.environ.get("KHQUANT_ENV_SCHEMA")
    for name in _PROJECT_PATH_VARS:
        value = os.environ.get(name, "")
        if value:
            os.environ[name] = str(_resolve_business(value, repo, project_root, schema))

    for name in ("KHQUANT_VENV_DIR", "PYTHONPATH"):
        value = os.environ.get(name, "")
        if value and os.pathsep not in value:
            os.environ[name] = str(_resolve(value, repo))


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _resolve_business(value: str, repo: Path, project_root: Path, schema: str | None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    parts = tuple(part.casefold() for part in path.parts if part not in (".", ""))
    base = repo if schema != "2" and parts and parts[0] == "app" else project_root
    return (base / path).resolve()
