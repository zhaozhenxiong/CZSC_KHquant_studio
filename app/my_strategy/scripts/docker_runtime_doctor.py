"""Diagnose the CZSC native or Docker environment without opening legacy stores."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import tempfile
from typing import Any

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, DATA_ROOT, METADATA_ROOT, PROJECT_ROOT, PROCESSED_DATA_ROOT
from my_strategy.storage.db_paths import raw_db_path
from my_strategy.services.czsc_compute import compute_status
from my_strategy.runtime_env import configure_utf8_stdio


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("json", "strict", "check-write"):
        parser.add_argument(f"--{flag}", action="store_true")
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--expect-docker", action="store_true")
    runtime.add_argument("--expect-native", action="store_true")
    return parser.parse_args(argv)


def inside_docker() -> bool:
    if Path("/.dockerenv").exists():
        return True
    try:
        return any(word in Path("/proc/1/cgroup").read_text().lower() for word in ("docker", "containerd"))
    except OSError:
        return False


def module_version(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"name": name, "ok": True, "version": str(getattr(module, "__version__", "")), "error": ""}
    except Exception as exc:
        return {"name": name, "ok": False, "version": "", "error": str(exc)}


def czsc_report() -> dict[str, Any]:
    try:
        from my_strategy.adapters.czsc_adapter import native_runtime
        module = native_runtime()
        return {"name": "czsc", "ok": True, "version": str(module.__version__), "source": "frozen_attachment", "error": ""}
    except Exception as exc:
        return {"name": "czsc", "ok": False, "error": str(exc)}


def torch_report(compute: dict[str, Any]) -> dict[str, Any]:
    """Exercise the same float32 training operators and inference as the MLP."""
    report: dict[str, Any] = {"name": "torch", "ok": False, "version": "", "error": "",
                              "device": compute.get("selected_device"), "training_work": False, "inference_work": False}
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from my_strategy.core.device import synchronize_torch_device
        report["version"] = str(torch.__version__)
        if not compute["available"]:
            raise RuntimeError(compute.get("fallback_reason") or "requested device unavailable")
        selected = torch.device(compute["selected_device"])
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            values = torch.tensor([[0.5, -0.2], [-0.5, 0.2]], dtype=torch.float32, device=selected)
            labels = torch.tensor([[1.0], [0.0]], dtype=torch.float32, device=selected)
            model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1)).to(selected)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss = torch.nn.BCEWithLogitsLoss()(model(values), labels)
            loss.backward()
            optimizer.step()
            with torch.inference_mode():
                prediction = torch.sigmoid(model(values)).cpu()
            synchronize_torch_device(torch, selected)
            if prediction.dtype != torch.float32 or not torch.isfinite(prediction).all().item():
                raise RuntimeError("float32 MLP probe produced invalid output")
            report.update(ok=True, training_work=True, inference_work=True, dtype="float32",
                          operators=["Linear", "ReLU", "BCEWithLogitsLoss", "backward", "Adam", "sigmoid"],
                          actual_cuda_work=selected.type == "cuda", actual_mps_work=selected.type == "mps")
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def path_report(name: str, path: Path, *, check_write: bool) -> dict[str, Any]:
    error = ""
    writable = None
    if check_write:
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".czsc-doctor-", dir=path):
                pass
            writable = True
        except OSError as exc:
            writable, error = False, str(exc)
    return {"name": name, "path": str(path), "exists": path.is_dir(), "writable": writable, "error": error}


def database_report(name: str, path: Path, required_tables: set[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"name": name, "path": str(path), "exists": path.is_file(), "quick_check": None,
                              "missing_tables": [], "error": ""}
    if path.is_file():
        try:
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
                connection.execute("PRAGMA query_only=ON")
                report["quick_check"] = connection.execute("PRAGMA quick_check").fetchone()[0]
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                report["missing_tables"] = sorted(required_tables - tables)
        except sqlite3.Error as exc:
            report["error"] = str(exc)
    return report


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    directories = [("project_root", PROJECT_ROOT), ("data_root", DATA_ROOT), ("raw_db_parent", raw_db_path().parent),
                   ("results_root", PROCESSED_DATA_ROOT / "czsc"), ("metadata_root", METADATA_ROOT),
                   ("runs_root", ARTIFACT_RUNS_ROOT)]
    paths = [path_report(name, path, check_write=args.check_write and name != "project_root") for name, path in directories]
    dependencies = [module_version(name) for name in ("numpy", "pandas", "yaml", "fastapi", "uvicorn", "plotly")]
    dependencies.append(czsc_report())
    compute = compute_status()
    dependencies.append(torch_report(compute))
    databases = [database_report("raw_db", raw_db_path(), {"stock_daily_normalized", "securities"}),
                 database_report("czsc_results", PROCESSED_DATA_ROOT / "czsc/results.db", {"czsc_runs"}),
                 database_report("czsc_tasks", METADATA_ROOT / "czsc_tasks.db", {"czsc_tasks"})]
    docker = inside_docker()
    failures = []
    if not compute["available"]:
        failures.append(f"compute unavailable: {compute['fallback_reason']}")
    if args.expect_docker and not docker:
        failures.append("expected Docker runtime")
    if args.expect_native and docker:
        failures.append("expected native runtime")
    for item in paths:
        if not item["exists"] or item["writable"] is False:
            failures.append(f"directory unavailable: {item['path']} {item['error']}")
    for item in dependencies:
        if not item["ok"]:
            failures.append(f"dependency unavailable: {item['name']}: {item['error']}")
    # A fresh installation can have no market data or result DB yet.
    for item in databases:
        if item["exists"] and (item["quick_check"] != "ok" or item["missing_tables"] or item["error"]):
            failures.append(f"database invalid: {item['name']}: {item['error']} {item['missing_tables']}")
    return {"status": "failed" if failures else "success", "architecture": "czsc", "failures": failures,
            "runtime": {"created_at": datetime.now().astimezone().isoformat(), "inside_docker": docker,
                        "python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform()},
            "environment": {key: os.environ.get(key, "") for key in ("KHQUANT_PROJECT_ROOT", "KHQUANT_DATA_ROOT",
                            "KHQUANT_RAW_DB", "KHQUANT_ARTIFACT_ROOT", "KHQUANT_METADATA_ROOT")},
            "paths": paths, "dependencies": dependencies, "databases": databases, "compute": compute}


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    args = parse_args(argv)
    payload = build_payload(args)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"CZSC runtime doctor: {payload['status']} ({payload['runtime']['executable']})")
        for failure in payload["failures"]:
            print(f"FAIL: {failure}")
    return int(args.strict and bool(payload["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
