"""Discover complete dual research candidates without creating a release."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT


def dual_research_status() -> dict:
    """Only completed, persisted research packages are exposed to the UI."""
    from my_strategy.services.czsc_dual_models import verified_bundle
    models, failures = [], []
    paths = sorted(ARTIFACT_RUNS_ROOT.glob("*/reports/dual-research.json"),
                   key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths[:20]:
        root = path.parent.parent
        try:
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("status") not in {"complete", "succeeded"}:
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            checkpoints = []
            for bundle in report.get("model_bundles", []):
                name = bundle["name"]
                manifest_path = root / "models" / name / "manifest.json"
                if not manifest_path.resolve().is_relative_to(root.resolve()):
                    raise ValueError("bundle outside run")
                manifest = verified_bundle(manifest_path.parent)
                checkpoints.append({"name": name, "available_at": manifest["available_at"],
                    "train_label_end": manifest.get("train_label_end"),
                    "validation_end": manifest.get("validation_end"), "model_family": "dual",
                    "training_completed_at": manifest["training_completed_at"]})
            if not checkpoints:
                continue
            models.append({"run_id": root.name, "model_family": "dual", "feature_profile": "dual_bundle_v1",
                "model_gate": report.get("model_gate", {"passed": False}),
                "coverage": report.get("coverage", {}), "checkpoints": checkpoints,
                "source_training_run_id": report["source_snapshot"]["source_run_id"],
                "training_completed_at": max((item["training_completed_at"] for item in checkpoints),
                                             key=datetime.fromisoformat)})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"run_id": root.name, "reason": str(exc)})
    return {"models": models, "active_release": None, "qualification": "shadow_only",
            "failures": failures, "entry_target": "fixed_10_sessions_net_positive",
            "exit_target": "exit_next_verified_open_vs_frozen_rule_continue"}


def shadow_exit_policy(runtime, features):
    """Score only actual holdings, using each signal day's compatible head."""
    from my_strategy.services.czsc_analysis import strategy_config
    from my_strategy.services.czsc_dual_exit import ExitPredictor, exit_contract, make_exit_policy
    if runtime.entry_policy == "legacy":
        return None
    if not hasattr(runtime, "exit_predictors"):
        runtime.exit_predictors = {}
    expected = exit_contract(strategy_config(), runtime.market_dates, runtime.entry_policy)
    def resolve(day):
        route = runtime.resolver.resolve(day)
        binding = route.get("exit_model")
        if not route.get("model_dir") or not binding or binding.get("unavailable") or not binding.get("model_dir"):
            return None
        # A risk plan and a fresh plan are different exit target contracts.
        if binding.get("entry_policy", "fresh") != runtime.entry_policy:
            return None
        directory = binding["model_dir"]
        if directory not in runtime.exit_predictors:
            runtime.exit_predictors[directory] = ExitPredictor(binding, device="cpu", expected_contract=expected)
        return runtime.exit_predictors[directory]
    return make_exit_policy(features=features, predictor_resolver=resolve, apply_exit=False)
