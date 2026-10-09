"""Append-only model release events and an atomic alias in CZSC results.db."""
from __future__ import annotations

from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import uuid
from typing import Any

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, PROCESSED_DATA_ROOT
from my_strategy.core.tz import local_now


def _verify_window(item: dict[str, Any], config: dict[str, Any]) -> None:
    base, ml = item["variants"]["czsc_price_volume"]["metrics"], item["variants"]["ml"]["metrics"]
    interval = item.get("paired_daily_return_uncertainty", {}).get("11", {})
    lower = interval.get("lower", -1)
    values = [ml.get("total_return"), base.get("total_return"), ml.get("max_drawdown"), base.get("max_drawdown"),
        ml.get("trade_expectancy"), base.get("trade_expectancy"), lower]
    if (any(value is None or not math.isfinite(float(value)) for value in values)
            or ml["total_return"] <= base["total_return"]
            or ml["max_drawdown"] > base["max_drawdown"] + min(.02, float(config.get("drawdown_tolerance", .02)))
            or ml.get("completed_round_trips", 0) < max(30, int(config.get("min_round_trips", 30)))
            or ml["trade_expectancy"] <= base["trade_expectancy"] or lower <= 0
            or interval.get("block_length") != 11 or interval.get("dates", 0) < 22
            or not .95 <= interval.get("confidence", 0) < 1
            or int(config.get("horizon", 10)) >= 11):
        raise ValueError("production gate metrics failed independent recomputation")


class ModelReleaseStore:
    def __init__(self, db_path: Path | None = None, runs_root: Path | None = None) -> None:
        self.db_path = Path(db_path or PROCESSED_DATA_ROOT / "czsc" / "results.db")
        self.runs_root = Path(runs_root or ARTIFACT_RUNS_ROOT)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS czsc_model_release_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL, event_type TEXT NOT NULL,
                    target_release_id TEXT, previous_release_id TEXT, payload TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS czsc_model_aliases (
                    alias TEXT PRIMARY KEY, release_id TEXT, event_id TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS czsc_model_events_no_update
                    BEFORE UPDATE ON czsc_model_release_events BEGIN
                    SELECT RAISE(ABORT, 'CZSC release events are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS czsc_model_events_no_delete
                    BEFORE DELETE ON czsc_model_release_events BEGIN
                    SELECT RAISE(ABORT, 'CZSC release events are immutable'); END;
            """)
            conn.commit()

    def events(self, limit: int | None = 100) -> list[dict[str, Any]]:
        from my_strategy.services.czsc_research_models import model_hash
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute("SELECT payload,payload_sha256 FROM czsc_model_release_events ORDER BY sequence").fetchall()
        values = []
        for raw, recorded in rows:
            value = json.loads(raw)
            if model_hash(value) != recorded:
                raise ValueError("release event integrity mismatch")
            values.append(value)
        return values[-max(1, limit):] if limit is not None else values

    list = events

    def active(self, as_of: Any = None) -> dict[str, Any] | None:
        from my_strategy.services.czsc_research_models import signal_time
        values = self.events(limit=None)
        if as_of is not None:
            values = [event for event in values if signal_time(event["created_at"]) <= signal_time(as_of)]
        if not values or not values[-1]["target_release_id"]:
            return None
        return self.verify_release(values[-1]["target_release_id"])

    def verify_release(self, release_id: str) -> dict[str, Any]:
        from my_strategy.services.czsc_research_models import file_hash
        event = next((event for event in self.events(limit=None) if event["event_id"] == release_id and event["event_type"] == "promote"), None)
        if event is None:
            raise ValueError("unknown original production release")
        for artifact in event["artifacts"]:
            path = (self.runs_root / artifact["path"]).resolve()
            if not path.is_relative_to(self.runs_root.resolve()) or file_hash(path) != artifact["sha256"]:
                raise ValueError("production release artifact integrity mismatch")
        return event

    def _append(self, kind: str, target: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        from my_strategy.services.czsc_research_models import model_hash
        event_id, created_at = uuid.uuid4().hex, local_now().isoformat()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            previous = conn.execute("SELECT release_id FROM czsc_model_aliases WHERE alias='production'").fetchone()
            event = {**payload, "event_id": event_id, "release_id": event_id if kind == "promote" else target,
                "created_at": created_at, "event_type": kind, "target_release_id": event_id if kind == "promote" else target,
                "previous_release_id": previous[0] if previous else None}
            conn.execute("INSERT INTO czsc_model_release_events (event_id,created_at,event_type,target_release_id,previous_release_id,payload,payload_sha256) VALUES (?,?,?,?,?,?,?)",
                (event_id, created_at, kind, event["target_release_id"], event["previous_release_id"],
                 json.dumps(event, ensure_ascii=False, allow_nan=False), model_hash(event)))
            conn.execute("INSERT INTO czsc_model_aliases VALUES ('production',?,?) ON CONFLICT(alias) DO UPDATE SET release_id=excluded.release_id,event_id=excluded.event_id",
                (event["target_release_id"], event_id))
            conn.commit()
        return event

    def promote(self, model_run_id: str, checkpoint: str = "production", *, reason: str,
                certification_path: Path | None = None) -> dict[str, Any]:
        from my_strategy.services.czsc_research_models import checkpoint_catalog, file_hash, identity_path, signal_time
        from my_strategy.core.run_context import stable_hash
        from my_strategy.adapters.czsc_adapter import SOURCE_SHA256, native_runtime
        if not reason.strip():
            raise ValueError("production release requires an explicit reason")
        root = identity_path(self.runs_root, model_run_id)
        training_path = root / "reports" / "research.json"
        training = json.loads(training_path.read_text(encoding="utf-8"))
        gate, coverage = training.get("model_gate", {}), training.get("coverage", {})
        if (gate.get("passed") is not True or gate.get("full_universe") is not True
                or gate.get("preparation_failures") != 0 or training.get("failures")
                or coverage.get("failed") != 0 or not coverage.get("requested")
                or coverage.get("requested") != coverage.get("success")):
            raise ValueError("model has not passed the complete-universe ledger gate; keep shadow")
        config, evaluations = training.get("config", {}), training.get("evaluation", [])
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        frozen_stocks = metadata.get("stocks", [])
        records = training.get("dataset_records", [])
        if (len(frozen_stocks) != coverage["requested"] or len(set(frozen_stocks)) != len(frozen_stocks)
                or len(records) != coverage["requested"]
                or {record.get("symbol") for record in records} != set(frozen_stocks)):
            raise ValueError("complete-universe frozen stock and dataset records do not match")
        if len(evaluations) < 3 or gate.get("windows") != len(evaluations) or gate.get("passing_windows") != len(evaluations):
            raise ValueError("production gate requires at least three complete passing windows")
        for item in evaluations:
            _verify_window(item, config)
        catalog = checkpoint_catalog(model_run_id, runs_root=self.runs_root)
        selected = next((entry for entry in catalog if entry["checkpoint"] == checkpoint), None)
        if not selected:
            raise ValueError("selected checkpoint absent")
        from my_strategy.services.czsc_research_profiles import recognized_feature_profile
        profile = recognized_feature_profile(selected["feature_version"], selected["feature_schema_hash"],
                                             selected["manifest"]["schema"]["columns"], selected["strategy_version"])
        native_runtime()  # Verify the frozen attachment before granting a production identity.
        audit_path = root / "reports" / "execution_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (audit.get("passed") is not True or audit.get("error_count") != 0 or audit.get("errors")
                or audit.get("training_run_id") != model_run_id or audit.get("frozen_stocks") != coverage["requested"]):
            raise ValueError("independent execution audit incomplete or failed")
        cert_path = Path(certification_path or root / "reports" / "production_certification.json").resolve()
        if not cert_path.is_relative_to(self.runs_root.resolve()):
            raise ValueError("certification must be an immutable run artifact")
        if not cert_path.is_file():
            raise ValueError("independent production certification is missing")
        cert = json.loads(cert_path.read_text(encoding="utf-8"))
        now = local_now()
        if (cert.get("passed") is not True or cert.get("independent") is not True
                or cert.get("training_run_id") != model_run_id or cert.get("checkpoint") != checkpoint
                or cert.get("manifest_sha256") != selected["manifest_sha256"]
                or cert.get("research_sha256") != file_hash(training_path)
                or cert.get("execution_audit_sha256") != file_hash(audit_path)):
            raise ValueError("independent certification identity/hash mismatch")
        completed = signal_time(cert["completed_at"])
        fitted = signal_time(cert["training_completed_at"])
        if (len(cert["completed_at"]) <= 10 or len(cert["training_completed_at"]) <= 10
                or fitted > completed or completed > now
                or fitted < signal_time(selected["available_at"])
                or any(signal_time(item["fold"]["test_end"]) > completed for item in evaluations)):
            raise ValueError("certification/train timestamps cannot precede data or follow actual release time")
        if selected.get("actual_training_started_at") and fitted < signal_time(selected["actual_training_started_at"]):
            raise ValueError("training completion cannot precede actual run start")
        if selected.get("actual_training_completed_at") and fitted != signal_time(selected["actual_training_completed_at"]):
            raise ValueError("certified training completion mismatches checkpoint provenance")
        independent_windows = cert.get("independent_windows", [])
        if len(independent_windows) < 3 or not cert.get("frozen_before_evaluation_at"):
            raise ValueError("certification requires at least three untouched evaluation windows frozen in advance")
        frozen = signal_time(cert["frozen_before_evaluation_at"])
        if frozen < fitted:
            raise ValueError("certification cannot freeze a model before training completed")
        freeze_root = identity_path(self.runs_root, cert.get("freeze_run_id", ""))
        freeze_metadata_path = freeze_root / "metadata.json"
        freeze_policy_path = freeze_root / "reports" / "production_freeze.json"
        if (file_hash(freeze_metadata_path) != cert.get("freeze_metadata_sha256")
                or file_hash(freeze_policy_path) != cert.get("freeze_policy_sha256")):
            raise ValueError("pre-evaluation freeze run integrity mismatch")
        freeze_metadata = json.loads(freeze_metadata_path.read_text(encoding="utf-8"))
        freeze_policy = json.loads(freeze_policy_path.read_text(encoding="utf-8"))
        bindings = {"training_run_id": model_run_id, "checkpoint": checkpoint,
            "manifest_sha256": selected["manifest_sha256"], "feature_schema_hash": profile["schema_hash"],
            "strategy_version": selected["strategy_version"], "research_config_sha256": stable_hash(config),
            "native_source_sha256": SOURCE_SHA256}
        if profile["name"] != "legacy":
            bindings["feature_profile"] = profile["name"]
        if (freeze_metadata.get("run_id") != freeze_root.name
                or len(str(freeze_metadata.get("created_at", ""))) <= 10
                or signal_time(freeze_metadata["created_at"]) != frozen
                or freeze_metadata.get("scope") != "implementation"
                or freeze_policy.get("stage") != "frozen"
                or any(freeze_policy.get(key) != value for key, value in bindings.items())):
            raise ValueError("freeze RunContext does not bind actual time and model/config/source policy")
        previous_end = None
        seen_paths = set()
        for window in independent_windows:
            if not frozen < signal_time(window["test_start"]) <= signal_time(window["test_end"]) <= completed:
                raise ValueError("independent window was not frozen before evaluation")
            if previous_end is not None and signal_time(window["test_start"]) <= previous_end:
                raise ValueError("independent evaluation windows must be unique, ordered and disjoint")
            if window["report_path"] in seen_paths:
                raise ValueError("independent windows cannot repeat one report")
            previous_end = signal_time(window["test_end"])
            seen_paths.add(window["report_path"])
        paths = [training_path, audit_path, cert_path, root / "metadata.json", freeze_metadata_path, freeze_policy_path,
                 Path(selected["model_dir"]) / "manifest.json", Path(selected["model_dir"]) / "model.pt",
                 identity_path(self.runs_root, selected["calendar_run_id"]) / "reports" / "calendar.json"]
        for entry in catalog:
            paths.extend([Path(entry["model_dir"]) / "manifest.json", Path(entry["model_dir"]) / "model.pt"])
        for window in independent_windows:
            evidence = (self.runs_root / window["report_path"]).resolve()
            if not evidence.is_relative_to(self.runs_root.resolve()) or file_hash(evidence) != window["sha256"]:
                raise ValueError("independent certification window evidence hash mismatch")
            report = json.loads(evidence.read_text(encoding="utf-8"))
            if (report.get("training_run_id") != model_run_id or report.get("checkpoint") != checkpoint
                    or report.get("independent") is not True
                    or report.get("fold", {}).get("test_start") != window["test_start"]
                    or report.get("fold", {}).get("test_end") != window["test_end"]
                    or report.get("coverage") != coverage):
                raise ValueError("independent window model, dates or full-universe coverage mismatch")
            if any(report.get(key) != value for key, value in bindings.items()):
                raise ValueError("independent window strategy/config/source bindings mismatch")
            window_audit = report.get("execution_audit", {})
            if (window_audit.get("passed") is not True or window_audit.get("error_count") != 0
                    or window_audit.get("errors") or window_audit.get("training_run_id") != model_run_id
                    or window_audit.get("manifest_sha256") != selected["manifest_sha256"]
                    or window_audit.get("frozen_stocks") != coverage["requested"]
                    or not signal_time(window["test_end"]) <= signal_time(window_audit["audited_at"]) <= completed):
                raise ValueError("independent window execution audit is missing, failed or unbound")
            _verify_window(report, config)
            paths.append(evidence)
        artifacts = [{"path": path.resolve().relative_to(self.runs_root.resolve()).as_posix(), "sha256": file_hash(path)} for path in dict.fromkeys(paths)]
        return self._append("promote", None, {"model_run_id": model_run_id, "checkpoint": checkpoint, "reason": reason,
            "identity": {key: value for key, value in selected.items() if key != "manifest"},
            "training_completed_at": cert["training_completed_at"], "certification_completed_at": cert["completed_at"],
            "artifacts": artifacts})

    def rollback(self, release_id: str | None = None, *, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("production rollback requires an explicit reason")
        if release_id is not None:
            self.verify_release(release_id)
        return self._append("rollback", release_id, {"reason": reason})
