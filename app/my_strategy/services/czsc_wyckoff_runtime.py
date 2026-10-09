"""Expose complete Wyckoff candidates and actual-holding shadow exit scores."""
from __future__ import annotations

from datetime import datetime
import json

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT


def wyckoff_research_status() -> dict:
    models, failures = [], []
    paths = sorted(ARTIFACT_RUNS_ROOT.glob("*/reports/wyckoff-research.json"),
                   key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths[:20]:
        root = path.parent.parent
        try:
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("status") not in {"complete", "succeeded"}:
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("status") != "complete" or report.get("model_family") != "wyckoff":
                continue
            from my_strategy.services.czsc_wyckoff_models import verified_bundle
            checkpoints = []
            for bundle in report.get("model_bundles", []):
                name = bundle["name"]
                directory = root / "models" / name
                if not directory.resolve().is_relative_to(root.resolve()):
                    raise ValueError("bundle outside run")
                manifest = verified_bundle(directory)
                checkpoints.append({"name": name, "available_at": manifest["available_at"],
                    "train_label_end": manifest.get("train_label_end"),
                    "validation_end": manifest.get("validation_end"), "model_family": "wyckoff",
                    "training_completed_at": manifest["training_completed_at"]})
            if not checkpoints:
                continue
            models.append({"run_id": root.name, "model_family": "wyckoff", "feature_profile": "wyckoff_bundle_v1",
                "model_gate": report.get("model_gate", {"passed": False}),
                "coverage": report.get("coverage", {}), "checkpoints": checkpoints,
                "source_training_run_id": report["source_snapshot"]["source_run_id"],
                "training_completed_at": max((item["training_completed_at"] for item in checkpoints),
                                             key=datetime.fromisoformat),
                "quality": report.get("quality", {}), "independent_window": report.get("independent_window", {})})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"run_id": root.name, "reason": str(exc)})
    return {"models": models, "active_release": None, "qualification": "shadow_only", "failures": failures,
            "entry_target": "fixed_10_sessions_net_positive", "policy_entry_target": "frozen_policy_roundtrip_net_positive",
            "exit_target": "exit_next_verified_open_vs_frozen_rule_continue"}


def shadow_exit_policy(runtime, features):
    from my_strategy.services.czsc_wyckoff_exit import make_runtime_exit_policy
    return make_runtime_exit_policy(runtime, features)


def shadow_entry_policy(runtime, features):
    from my_strategy.services.czsc_wyckoff_exit import make_runtime_entry_policy
    return make_runtime_entry_policy(runtime, features)
