"""Run manifest helpers for reproducibility and lineage.

A manifest is a small JSON summary of a backtest run: config hash, seed,
models used, date range, and stock list. It is written for every CLI single
run, Web single run, and CLI batch run so downstream tools can answer
"what changed between these two runs?" without diffing full configs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def stable_hash(data: Any) -> str:
    """Deterministic sha256 for JSON-serialisable data."""
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_name(agents: dict[str, Any], key: str) -> str:
    cfg = agents.get(key, {})
    if not cfg.get("enabled", False):
        return ""
    return str(cfg.get("model_name", "") or "")


def build_manifest(
    config: dict[str, Any],
    stocks: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest dict from the effective config.

    ``config`` should be the resolved config after best-model selection and
    after ``run_context`` has been attached.
    """
    run_context = config.get("run_context", {}) or {}
    run_id = str(run_context.get("run_id", ""))
    seed = run_context.get("seed", 42)

    data_cfg = config.get("data", {})
    agents = config.get("agents", {})

    # Hash the resolved config minus the volatile run_context so the hash is
    # stable across re-runs with the same parameters.
    hashable = json.loads(json.dumps(config, default=str))
    hashable.pop("run_context", None)
    hashable.pop("_effective_config_snapshot", None)

    models = {
        "ml": _model_name(agents, "ml"),
        "deep_learning": _model_name(agents, "deep_learning"),
        "reinforcement_learning": _model_name(agents, "reinforcement_learning"),
        "probabilistic": _model_name(agents, "probabilistic"),
    }
    trend_model = _model_name(agents, "trend_score")
    if trend_model:
        strategy_variant = str((config.get("decision", {}) or {}).get("strategy_variant", "") or "")
        # Keep archive/gross trend runs distinguishable while making the
        # production net dependency explicit for catalog registration.
        trend_role = (
            "trend_score_net"
            if strategy_variant.startswith("trend_score_net") or trend_model == "trend_entry_mlp_net_v1"
            else "trend_score"
        )
        models[trend_role] = trend_model

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "config_hash": stable_hash(hashable),
        "seed": seed,
        "models": models,
        "data_range": {
            "start": str(data_cfg.get("start", "")),
            "end": str(data_cfg.get("end", "")),
            "stocks": list(stocks or data_cfg.get("stocks", [])),
        },
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    effective_snapshot = config.get("_effective_config_snapshot")
    if isinstance(effective_snapshot, dict):
        manifest["effective_config"] = dict(effective_snapshot)
    if extra:
        manifest.update(extra)
    return manifest


def build_effective_config_snapshot(
    config: dict[str, Any],
    runtime_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a full, serialisable configuration snapshot for one run."""

    resolved_config = json.loads(json.dumps(config, default=str))
    resolved_config.pop("_effective_config_snapshot", None)
    hashable = json.loads(json.dumps(resolved_config, default=str))
    hashable.pop("run_context", None)
    run_context = resolved_config.get("run_context", {}) or {}
    return {
        "schema_version": "effective-config-snapshot-v1",
        "run_id": str(run_context.get("run_id", "")),
        "config_hash": stable_hash(hashable),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "resolved_config": resolved_config,
        "runtime_resolution": dict(runtime_resolution or {}),
    }


def write_effective_config_snapshot(
    path: str | Path,
    config: dict[str, Any],
    runtime_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically persist the full configuration used after runtime resolution."""

    snapshot = build_effective_config_snapshot(config, runtime_resolution)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{__import__('os').getpid()}.tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    snapshot["path"] = str(target)
    return snapshot


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Atomically write a manifest to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{__import__('os').getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
