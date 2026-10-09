"""Verified CZSC checkpoint identities and chronological model routing.

Data availability is not a production release. Historical candidates need a
matching actual release; retrospective candidates always stay shadow.
"""
from __future__ import annotations

from datetime import datetime, time
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import LOCAL_TZ


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def model_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def identity_path(root: Path, identity: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", identity):
        raise ValueError("invalid CZSC model identity")
    path = (root / identity).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("CZSC model path outside run storage")
    return path


def signal_time(value: Any) -> datetime:
    text = str(value)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if len(text) == 10:
        parsed = datetime.combine(parsed.date(), time(15), LOCAL_TZ)
    elif parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(LOCAL_TZ)


def verified_checkpoint(directory: Path) -> dict[str, Any]:
    from my_strategy.services.czsc_research_ml import MODEL_VERSION, LABEL_VERSION
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    integrity = dict(manifest)
    recorded = integrity.pop("manifest_sha256", None)
    if (recorded != model_hash(integrity)
            or manifest.get("schema_sha256") != model_hash(manifest["schema"])
            or manifest.get("preprocess_sha256") != model_hash(manifest["preprocess"])):
        raise ValueError("model manifest/schema/preprocess integrity mismatch")
    if file_hash(directory / "model.pt") != manifest.get("checkpoint_sha256"):
        raise ValueError("model checkpoint hash mismatch")
    # Label maturity, not merely signal date, defines chronological availability.
    available = signal_time(manifest["available_at"])
    for field in ("train_label_end", "validation_label_end", "validation_end"):
        if field not in manifest or signal_time(manifest[field]) > available:
            raise ValueError("model availability precedes mature training/validation labels")
    if (manifest.get("feature_schema_binding") != "bound"
            or not manifest.get("feature_version") or not manifest.get("feature_schema_hash")):
        raise ValueError("model requires bound feature semantics")
    if manifest.get("model_version") != MODEL_VERSION or manifest.get("label_version") != LABEL_VERSION:
        raise ValueError("checkpoint model/label contract is incompatible")
    return manifest


def checkpoint_catalog(model_run_id: str, *, runs_root: Path | None = None) -> list[dict[str, Any]]:
    from my_strategy.adapters.czsc_adapter import SOURCE_SHA256
    from my_strategy.services.czsc_research_profiles import recognized_feature_profile
    root = identity_path(runs_root or ARTIFACT_RUNS_ROOT, model_run_id)
    training = json.loads((root / "reports" / "research.json").read_text(encoding="utf-8"))
    if training.get("run_id") != model_run_id:
        raise ValueError("research report identity mismatch")
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    calendar = training.get("calendar", {})
    calendar_root = identity_path(runs_root or ARTIFACT_RUNS_ROOT, calendar.get("run_id", ""))
    calendar_path = calendar_root / "reports" / "calendar.json"
    calendar_value = json.loads(calendar_path.read_text(encoding="utf-8"))
    if not calendar_value.get("verified") or not calendar_value.get("source") or stable_hash(calendar_value) != calendar.get("hash"):
        raise ValueError("verified model trading calendar hash mismatch")
    dates = calendar_value.get("dates", [])
    if dates != sorted(set(dates)):
        raise ValueError("model calendar must be ascending and unique")
    threshold = float(training.get("config", {}).get("probability_threshold", .55))
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("invalid model probability threshold")
    names = ["production", *[item["fold"]["name"] for item in training.get("evaluation", [])]]
    entries = []
    for name in dict.fromkeys(names):
        directory = identity_path(root / "models", name)
        manifest = verified_checkpoint(directory)
        profile = recognized_feature_profile(manifest["feature_version"], manifest["feature_schema_hash"],
                                             manifest["schema"]["columns"], training.get("strategy_version"))
        if (training.get("strategy_version") != profile["strategy_version"]
                or training.get("config", {}).get("feature_profile", profile["name"]) != profile["name"]
                or manifest.get("feature_profile", profile["name"]) != profile["name"]):
            raise ValueError("checkpoint and research feature profile binding mismatch")
        if manifest.get("label_contract", {}).get("calendar_hash") != stable_hash(dates):
            raise ValueError("checkpoint and research calendar binding mismatch")
        if manifest.get("data_version") != training.get("data_version"):
            raise ValueError("checkpoint and research dataset binding mismatch")
        entries.append({"model_run_id": model_run_id, "checkpoint": name, "name": name,
            "kind": "production_candidate" if name == "production" else "historical_reconstruction",
            "model_dir": str(directory), "available_at": manifest["available_at"],
            "probability_threshold": threshold, "manifest_sha256": manifest["manifest_sha256"],
            "checkpoint_sha256": manifest["checkpoint_sha256"], "preprocess_sha256": manifest["preprocess_sha256"],
            "schema_sha256": manifest["schema_sha256"], "model_version": manifest["model_version"],
            "label_version": manifest["label_version"], "train_label_end": manifest["train_label_end"],
            "validation_label_end": manifest["validation_label_end"], "validation_end": manifest["validation_end"],
            "feature_version": manifest["feature_version"], "feature_schema_hash": manifest["feature_schema_hash"],
            "feature_profile": profile["name"],
            "strategy_version": training.get("strategy_version"), "data_version": training.get("data_version"),
            "calendar_run_id": calendar["run_id"], "calendar_hash": calendar["hash"],
            "config_hash": metadata.get("config_hash") or stable_hash(training.get("config", {})),
            "source_hash": metadata.get("source_hash") or SOURCE_SHA256, "native_source_sha256": SOURCE_SHA256,
            "source_git_commit": metadata.get("git_commit"),
            "actual_training_started_at": manifest.get("training_started_at") or metadata.get("created_at") or training.get("run_context", {}).get("created_at"),
            "training_started_provenance": "model_recorded" if manifest.get("training_started_at") else "run_creation_only",
            "actual_training_completed_at": manifest.get("training_completed_at") or metadata.get("training_completed_at"),
            "training_time_provenance": "recorded" if manifest.get("training_completed_at") or metadata.get("training_completed_at") else "not_recorded",
            "manifest": manifest})
    return entries


class ModelResolver:
    def __init__(self, *, usage_mode: str = "production", model_policy: str = "auto",
                 model_run_id: str | None = None, checkpoint: str | None = None,
                 calendar_run_id: str | None = None, store=None, runs_root: Path | None = None,
                 feature_version: str | None = None, feature_schema_hash: str | None = None,
                 feature_columns: Sequence[str] | None = None,
                 strategy_version: str | None = None) -> None:
        from my_strategy.services.czsc_research_profiles import get_feature_profile
        from my_strategy.storage.czsc_model_releases import ModelReleaseStore
        default_profile = get_feature_profile()
        requested_version = feature_version or default_profile["version"]
        requested_hash = feature_schema_hash or default_profile["schema_hash"]
        requested_columns = list(feature_columns) if feature_columns is not None else None
        if requested_columns is None:
            for name in ("legacy", "ma_trend_v1"):
                profile = get_feature_profile(name)
                if (requested_version, requested_hash) == (profile["version"], profile["schema_hash"]):
                    requested_columns = profile["columns"]
                    break
        if usage_mode not in {"production", "historical", "retrospective"} or model_policy not in {"auto", "pinned"}:
            raise ValueError("invalid model usage mode or selection policy")
        if model_policy == "pinned" and (not model_run_id or not checkpoint):
            raise ValueError("pinned model requires run and checkpoint identities")
        if usage_mode == "retrospective" and model_policy != "pinned":
            raise ValueError("retrospective research requires an explicit pinned checkpoint")
        self.usage_mode, self.model_policy = usage_mode, model_policy
        self.model_run_id, self.checkpoint = model_run_id, checkpoint
        self.runs_root = Path(runs_root or getattr(store, "runs_root", ARTIFACT_RUNS_ROOT))
        self.store = store or ModelReleaseStore(runs_root=self.runs_root)
        self.catalog, self.catalog_errors = [], []
        self._releases = self.store.events(limit=None)
        self._release_checks = {}
        if model_run_id:
            roots = [identity_path(self.runs_root, model_run_id)]
        else:
            roots = sorted(self.runs_root.iterdir()) if self.runs_root.exists() else []
        for root in roots:
            if not (root / "reports" / "research.json").is_file():
                continue
            try:
                entries = checkpoint_catalog(root.name, runs_root=self.runs_root)
                for entry in entries:
                    if (entry["feature_version"] != requested_version
                            or entry["feature_schema_hash"] != requested_hash
                            or (strategy_version is not None and entry["strategy_version"] != strategy_version)
                            or (calendar_run_id is not None and entry["calendar_run_id"] != calendar_run_id)):
                        continue
                    if requested_columns is None or entry["manifest"]["schema"]["columns"] != requested_columns:
                        raise ValueError("checkpoint columns mismatch bound feature schema")
                    self.catalog.append(entry)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                self.catalog_errors.append({"model_run_id": root.name, "reason": str(exc)})
        self.catalog.sort(key=lambda entry: (entry["available_at"], entry["model_run_id"], entry["checkpoint"]))

    def _active(self, moment: datetime):
        events = [event for event in self._releases if signal_time(event["created_at"]) <= moment]
        if not events or not events[-1]["target_release_id"]:
            return None
        release_id = events[-1]["target_release_id"]
        if release_id not in self._release_checks:
            self._release_checks[release_id] = self.store.verify_release(release_id)
        return self._release_checks[release_id]

    def resolve(self, day: Any) -> dict[str, Any]:
        moment = signal_time(day)
        route = {"date": moment.date().isoformat(), "usage_mode": self.usage_mode,
            "model_policy": self.model_policy, "model_run_id": self.model_run_id, "checkpoint": self.checkpoint,
            "model_dir": None, "probability_threshold": .55, "status": "rules_no_model",
            "applied_to_entry": False, "reason": "no compatible checkpoint available at signal time",
            "reason_codes": ["model_unavailable"], "release_id": None}
        candidates = [entry for entry in self.catalog if self.model_policy != "pinned"
                      or entry["model_run_id"] == self.model_run_id and entry["checkpoint"] == self.checkpoint]
        if self.usage_mode != "retrospective":
            candidates = [entry for entry in candidates if signal_time(entry["available_at"]) <= moment]
        if not candidates:
            if self.model_policy == "pinned":
                route["reason_codes"] = ["pinned_model_unavailable"]
                route["reason"] = "pinned checkpoint is missing, incompatible, damaged, or unavailable at signal time"
            return route
        chosen, release = candidates[-1], None
        if self.usage_mode in {"production", "historical"}:
            try:
                release = self._active(moment)
            except (ValueError, KeyError, OSError) as exc:
                route.update(reason=str(exc), reason_codes=["production_release_integrity_failure"])
                return route
            matching = [entry for entry in (candidates if self.usage_mode == "production" else [chosen])
                        if release and entry["model_run_id"] == release["model_run_id"]
                        and entry["checkpoint"] == release["checkpoint"]]
            if matching:
                chosen = matching[0]
            else:
                release = None
        route.update({key: value for key, value in chosen.items() if key != "manifest"})
        if release:
            route.update(status="production_active" if self.usage_mode == "production" else "historical_released",
                applied_to_entry=True, release_id=release["release_id"],
                promoted_at=release["created_at"], reason="actual production release effective at signal time",
                reason_codes=[])
        elif self.usage_mode == "retrospective":
            route.update(status="retrospective_shadow", reason="latest-model retrospective research; excluded from production and unbiased certification",
                reason_codes=["retrospective_only"], retrospective_overlap=moment < signal_time(chosen["available_at"]))
        else:
            route.update(status="historical_shadow" if self.usage_mode == "historical" else "production_shadow",
                reason="checkpoint is data-available, but has no eligible actual release; aggregate future gate is not historical qualification",
                reason_codes=["model_shadow", "production_release_missing"])
        return route

    def resolve_dates(self, dates: Iterable[Any]) -> list[dict[str, Any]]:
        return [self.resolve(day) for day in dates]
