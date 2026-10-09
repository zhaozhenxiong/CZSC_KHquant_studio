"""Isolated shadow experts and a strictly forward OOF probability stack."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_research_ml import (
    PredictorSession, _file_hash, _hash, _verified_manifest, train_model,
)
from my_strategy.services.czsc_research_profiles import (
    bind_feature_profile, get_feature_profile, prepare_structure_inputs,
)

DUAL_VERSION = "czsc_dual_experts_v1"
FUSION_COLUMNS = ["p_ma", "p_structure"]
FUSION_HASH = stable_hash({"version": "dual_probability_oof_v1", "columns": FUSION_COLUMNS,
                           "fit": "strict_forward_oof_only", "target": "czsc_fixed_horizon_broker_v1"})


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def expert_inputs(frame: pd.DataFrame, expert: str) -> pd.DataFrame:
    if expert == "ma":
        result = bind_feature_profile(frame, "ma_trend_v1")
        if "model_input_eligible" not in result:
            raise ValueError("MA input qualification missing")
        result["input_eligible"] = result["model_input_eligible"].fillna(False).astype(bool)
    elif expert == "structure":
        result = prepare_structure_inputs(frame)
        result["input_eligible"] = result["structure_input_eligible"].astype(bool)
    else:
        raise ValueError("unknown dual expert")
    columns = result.attrs["feature_columns"]
    metadata = [c for c in ("date", "symbol", "label", "label_end", "label_available", "net_return", "input_eligible") if c in result]
    return result[list(dict.fromkeys([*columns, *metadata]))].copy()


def fusion_inputs(frame: pd.DataFrame, ma, structure) -> pd.DataFrame:
    columns = [c for c in ("date", "symbol", "label", "label_end", "label_available", "net_return") if c in frame]
    result = frame[columns].copy()
    result["p_ma"], result["p_structure"] = np.asarray(ma, dtype=float), np.asarray(structure, dtype=float)
    result["input_eligible"] = np.isfinite(result[FUSION_COLUMNS]).all(axis=1)
    result.attrs = {**frame.attrs, "feature_version": "dual_probability_oof_v1",
                    "schema_hash": FUSION_HASH, "feature_columns": FUSION_COLUMNS}
    return result


def oof_splits(feature_start: str, train_end: str) -> list[dict]:
    """Six initial train months, one calibration quarter, then forward quarters."""
    first = pd.Period(feature_start, freq="Q") + 3
    last = pd.Period(train_end, freq="Q")
    output = []
    for quarter in pd.period_range(first, last, freq="Q"):
        calibration = quarter - 1
        output.append({"name": str(quarter), "train_end": (quarter - 2).end_time.date().isoformat(),
                       "validation_start": calibration.start_time.date().isoformat(),
                       "validation_end": calibration.end_time.date().isoformat(),
                       "test_start": quarter.start_time.date().isoformat(),
                       "test_end": min(quarter.end_time.date().isoformat(), train_end)})
    return output


def freeze_threshold(frame: pd.DataFrame, probabilities, config: dict) -> dict:
    """Validation-only fixed-horizon net-return proxy; never a release gate."""
    values = np.asarray(probabilities, dtype=float)
    valid = (frame["label_available"].astype(bool).to_numpy() & np.isfinite(values)
             & np.isfinite(pd.to_numeric(frame["net_return"], errors="coerce").to_numpy()))
    returns = pd.to_numeric(frame["net_return"], errors="coerce").to_numpy(dtype=float)
    grid = [float(x) for x in config.get("threshold_grid", [.5, .55, .6, .65])]
    if not grid or any(not 0 < x < 1 for x in grid):
        raise ValueError("invalid dual threshold grid")
    minimum = int(config.get("threshold_min_rows", 30))
    scores = []
    for threshold in grid:
        selected = valid & (values >= threshold)
        scores.append({"threshold": threshold, "rows": int(selected.sum()),
                       "mean_net_return": float(returns[selected].mean()) if selected.any() else None})
    candidates = [s for s in scores if s["rows"] >= minimum]
    chosen = max(candidates, key=lambda s: (s["mean_net_return"], -s["threshold"])) if candidates else None
    return {"threshold": chosen["threshold"] if chosen else float(config.get("default_threshold", .55)),
            "method": "validation_fixed_horizon_mean_net_return_proxy", "scores": scores,
            "minimum_rows": minimum, "fallback": chosen is None,
            "qualification": "fitted_validation_diagnostic_not_strategy_acceptance"}


def _fit_experts(dataset, directory, split, config, device, progress, check_cancel):
    result = {}
    for name, profile, hidden in (("ma", "ma_trend_v1", config.get("ma_hidden_sizes", [32, 16])),
                                   ("structure", "czsc_structure_v1", [])):
        inputs = expert_inputs(dataset, name)
        result[name] = train_model(inputs, get_feature_profile(profile)["columns"], directory / name,
            split["train_end"], split["validation_start"], split["validation_end"], device=device,
            seed=int(config.get("seed", 42)), epochs=int(config.get("epochs", 20)),
            batch_size=int(config.get("batch_size", 2048)), hidden_sizes=tuple(hidden),
            min_train_rows=int(config.get("min_train_rows", 100)),
            min_validation_rows=int(config.get("min_validation_rows", 30)),
            progress=progress, check_cancel=check_cancel)
    return result


def _cached_oof(dataset, root, split, config, device, predictor, progress, check_cancel):
    """Run-local immutable cache shared by outer folds; no cross-source reuse."""
    identity = _hash({"data_version": dataset.attrs.get("data_version"),
        "target": dataset.attrs.get("label_contract"), "split": split, "config": config, "device": device,
        "ma": get_feature_profile("ma_trend_v1"), "structure": get_feature_profile("czsc_structure_v1")})
    directory = root / split["name"] / identity
    header_path, rows_path = directory / "cache.json", directory / "predictions.parquet"
    if header_path.exists():
        header = json.loads(header_path.read_text())
        unsigned = dict(header); recorded = unsigned.pop("cache_sha256", None)
        if recorded != _hash(unsigned) or header["identity"] != identity or _file_hash(rows_path) != header["rows_sha256"]:
            raise ValueError("shared OOF cache integrity mismatch")
        for name, recorded_model in header["experts"].items():
            actual, _ = _verified_manifest(directory / "experts" / name)
            if actual != recorded_model:
                raise ValueError("shared OOF checkpoint binding mismatch")
        return pd.read_parquet(rows_path), {**header, "reused": True}, []
    models = _fit_experts(dataset, directory / "experts", split, config, device, progress, check_cancel)
    dates = pd.to_datetime(dataset["date"]).dt.strftime("%Y-%m-%d")
    test = dataset.loc[(dates >= split["test_start"]) & (dates <= split["test_end"])].copy()
    if test.empty:
        raise ValueError("OOF block has no forward observations")
    predictions = {name: predictor.predict(expert_inputs(test, name), directory / "experts" / name,
                                           as_of=split["test_end"]) for name in models}
    meta = fusion_inputs(test, predictions["ma"], predictions["structure"])
    meta = meta.loc[meta["input_eligible"]].copy()
    meta["oof_fold"] = split["name"]
    meta["expert_train_end"] = split["train_end"]
    meta["expert_validation_end"] = split["validation_end"]
    meta["expert_available_at"] = split["validation_end"]
    meta["target_label_end"] = meta["label_end"]
    if (pd.to_datetime(meta["date"]) <= pd.Timestamp(split["validation_end"])).any():
        raise ValueError("OOF expert was unavailable at its feature date")
    directory.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(rows_path, index=False)
    header = {"identity": identity, "fold": split, "experts": models,
              "rows_path": str(rows_path.resolve()), "rows_sha256": _file_hash(rows_path)}
    header["cache_sha256"] = _hash(header)
    _save(header_path, header)
    return meta, {**header, "reused": False}, list(models.values())


def train_dual_bundle(dataset, output_dir, fold, *, device="mps", config=None,
                      progress=None, check_cancel=None) -> dict[str, Any]:
    """No test labels enter expert fitting, OOF stacking, calibration or thresholds."""
    from my_strategy.core.tz import local_now
    config = dict(config or {})
    path = Path(output_dir)
    if (path / "bundle.json").exists():
        raise FileExistsError("dual bundle already exists")
    check = check_cancel or (lambda: None)
    dates = pd.to_datetime(dataset["date"]).dt.strftime("%Y-%m-%d")
    ends = pd.to_datetime(dataset["label_end"], errors="coerce").dt.strftime("%Y-%m-%d")
    oof_parts, oof_models, newly_trained_oof = [], [], []
    predictor = PredictorSession(device=device)
    for split in oof_splits(config.get("feature_start", "2023-01-01"), fold["train_end"]):
        check()
        meta, evidence, fresh_models = _cached_oof(dataset, path.parent / "oof-shared", split, config,
            device, predictor, progress, check_cancel)
        mature = (meta["label_available"].astype(bool)
                  & (pd.to_datetime(meta["label_end"], errors="coerce") <= pd.Timestamp(fold["train_end"])))
        meta = meta.loc[mature].copy()
        oof_parts.append(meta)
        oof_models.append(evidence)
        newly_trained_oof.extend(fresh_models)
    if not oof_parts:
        raise ValueError("strict forward OOF history unavailable")
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(["date", "symbol"])
    if oof.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate OOF outcomes")
    oof.attrs = {**dataset.attrs, "feature_version": "dual_probability_oof_v1", "schema_hash": FUSION_HASH}
    oof_path = path / "oof" / "oof.parquet"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_path, index=False)
    experts = _fit_experts(dataset, path / "experts", fold, config, device, progress, check_cancel)
    validation_mask = ((dates >= fold["validation_start"]) & (dates <= fold["validation_end"])
                       & dataset["label_available"].astype(bool) & (ends <= fold["validation_end"]))
    validation = dataset.loc[validation_mask].copy()
    # Calibration has observed the validation outcomes. Explicit fitted validation
    # predictions are never called OOF, never replayed as earlier available models.
    validation_p = {name: predictor.predict_retrospective(expert_inputs(validation, name), path / "experts" / name,
                       as_of=fold["validation_end"]) for name in experts}
    validation_meta = fusion_inputs(validation, validation_p["ma"], validation_p["structure"])
    fusion_data = pd.concat([oof, validation_meta], ignore_index=True)
    fusion_data.attrs = dict(oof.attrs)
    fusion_dataset_path = path / "fusion-training.parquet"
    path.mkdir(parents=True, exist_ok=True)
    fusion_data.to_parquet(fusion_dataset_path, index=False)
    fusion = train_model(fusion_data, FUSION_COLUMNS, path / "fusion", fold["train_end"],
        fold["validation_start"], fold["validation_end"], device=device, seed=int(config.get("seed", 42)),
        epochs=int(config.get("epochs", 20)), batch_size=int(config.get("batch_size", 2048)), hidden_sizes=(),
        min_train_rows=int(config.get("min_train_rows", 100)), min_validation_rows=int(config.get("min_validation_rows", 30)),
        progress=progress, check_cancel=check_cancel)
    validation_p["fusion"] = predictor.predict_retrospective(validation_meta, path / "fusion", as_of=fold["validation_end"])
    thresholds = {name: freeze_threshold(validation, values, config) for name, values in validation_p.items()}
    expert_bindings = {name: {"path": str((path / "experts" / name).resolve()),
        "manifest_file_sha256": _file_hash(path / "experts" / name / "manifest.json"),
        "manifest_sha256": value["manifest_sha256"], "checkpoint_sha256": value["checkpoint_sha256"],
        "feature_version": value["feature_version"], "feature_schema_hash": value["feature_schema_hash"],
        "preprocess_sha256": value["preprocess_sha256"]} for name, value in experts.items()}
    models = [*experts.values(), fusion, *newly_trained_oof]
    bundle = {"version": DUAL_VERSION, "status": "shadow", "publication_allowed": False,
        "available_at": fold["validation_end"], "trained_at": local_now().isoformat(),
        "training_completed_at": local_now().isoformat(), "train_end": fold["train_end"],
        "train_label_end": max(m["train_label_end"] for m in experts.values()),
        "validation_start": fold["validation_start"], "validation_end": fold["validation_end"],
        "probability_threshold": thresholds["fusion"]["threshold"], "thresholds": thresholds,
        "target": dataset.attrs.get("label_contract", {}), "target_hash": _hash(dataset.attrs.get("label_contract", {})),
        "data_version": dataset.attrs.get("data_version"), "experts": expert_bindings,
        "fusion": {"path": str((path / "fusion").resolve()), "manifest_sha256": fusion["manifest_sha256"],
                   "dataset_path": str(fusion_dataset_path.resolve()), "dataset_file_sha256": _file_hash(fusion_dataset_path),
                   "manifest_file_sha256": _file_hash(path / "fusion" / "manifest.json"),
                   "checkpoint_sha256": fusion["checkpoint_sha256"], "preprocess_sha256": fusion["preprocess_sha256"]},
        "oof": {"path": str(oof_path.resolve()), "sha256": _file_hash(oof_path), "rows": len(oof),
                "date_start": str(oof["date"].min()), "date_end": str(oof["date"].max()),
                "label_end_max": str(oof["label_end"].max()), "folds": oof_models,
                "method": "expanding_train_prior_quarter_calibration_forward_quarter_predictions",
                "fusion_fit": "OOF only; all outcomes mature by outer train_end",
                "validation_predictions": "retrospective_fitted_validation_only_not_OOF"},
        "compute_info": {"training_batches": sum(m["training_batches"] for m in models),
                         "trained_models": len(models), "reused_oof_blocks": sum(x["reused"] for x in oof_models),
                         "referenced_models": 3 + 2 * len(oof_models), "devices": sorted({m["device"] for m in models}),
                         "actual_mps_training": any(m["actual_mps_training"] for m in models),
                         "actual_cuda_training": any(m["actual_cuda_training"] for m in models),
                         "inference": predictor.diagnostics},
        "limitations": ["All four historical windows have been observed and are development diagnostics.",
                         "Fixed ten-session entry probabilities are not exit probabilities.",
                         "No new independent future window has occurred; publication is prohibited."]}
    bundle["bundle_sha256"] = _hash(bundle)
    _save(path / "bundle.json", bundle)
    _save(path / "manifest.json", bundle)
    return bundle


def attach_exit_head(directory, binding):
    """Bind the independent holding target without changing any fitted weights."""
    path = Path(directory)
    bundle = verified_bundle(path)
    value = dict(binding)
    if not value.get("unavailable"):
        model_path = Path(value["model_dir"])
        manifest, _ = _verified_manifest(model_path)
        value.update(manifest_file_sha256=_file_hash(model_path / "manifest.json"),
            checkpoint_sha256=manifest["checkpoint_sha256"], preprocess_sha256=manifest["preprocess_sha256"],
            label_version=manifest["label_version"], target_hash=_hash(manifest["label_contract"]),
            actual_training_device=manifest["device"], training_batches=manifest["training_batches"])
        for name in ("head.binding.json", "exit-linear.json"):
            if (model_path / name).exists():
                value[name + "_sha256"] = _file_hash(model_path / name)
        if (model_path / "head.binding.json").exists():
            value["head_binding_sha256"] = json.loads((model_path / "head.binding.json").read_text())["binding_sha256"]
    bundle["exit_head"] = value
    from my_strategy.core.tz import local_now
    bundle["trained_at"] = local_now().isoformat()
    bundle["training_completed_at"] = bundle["trained_at"]
    bundle.pop("bundle_sha256", None); bundle["bundle_sha256"] = _hash(bundle)
    _save(path / "bundle.json", bundle); _save(path / "manifest.json", bundle)
    return bundle


def verified_bundle(directory, *, check_oof=False):
    root = Path(directory).resolve()
    value = json.loads((root / "bundle.json").read_text())
    if (root / "manifest.json").exists() and json.loads((root / "manifest.json").read_text()) != value:
        raise ValueError("dual bundle manifest alias mismatch")
    unsigned = dict(value); recorded = unsigned.pop("bundle_sha256", None)
    if recorded != _hash(unsigned) or value.get("version") != DUAL_VERSION or value.get("status") != "shadow" or value.get("publication_allowed") is not False:
        raise ValueError("dual bundle identity/integrity/shadow mismatch")
    if value["target_hash"] != _hash(value["target"]) or value["target"].get("label_version") != "czsc_fixed_horizon_broker_v1":
        raise ValueError("dual fixed-horizon target mismatch")
    if not value["train_label_end"] <= value["train_end"] < value["validation_start"] <= value["validation_end"] == value["available_at"]:
        raise ValueError("dual chronological cutoffs invalid")
    for name, binding in [*value["experts"].items(), ("fusion", value["fusion"])]:
        path = Path(binding["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("dual checkpoint path escapes bundle")
        manifest, _ = _verified_manifest(path)
        profile = get_feature_profile("ma_trend_v1" if name == "ma" else "czsc_structure_v1") if name != "fusion" else None
        expected = (profile["version"], profile["schema_hash"], profile["columns"]) if profile else ("dual_probability_oof_v1", FUSION_HASH, FUSION_COLUMNS)
        if (manifest["feature_version"], manifest["feature_schema_hash"], manifest["schema"]["columns"]) != expected:
            raise ValueError("dual expert feature schema mismatch")
        if (_file_hash(path / "manifest.json") != binding["manifest_file_sha256"]
                or manifest["manifest_sha256"] != binding["manifest_sha256"]
                or manifest["checkpoint_sha256"] != binding["checkpoint_sha256"]
                or manifest["preprocess_sha256"] != binding["preprocess_sha256"]
                or manifest["label_contract"] != value["target"]
                or manifest["available_at"] != value["available_at"]):
            raise ValueError("dual expert artifact/target binding mismatch")
    if check_oof and _file_hash(Path(value["oof"]["path"])) != value["oof"]["sha256"]:
        raise ValueError("dual OOF artifact hash mismatch")
    if check_oof and _file_hash(Path(value["fusion"]["dataset_path"])) != value["fusion"]["dataset_file_sha256"]:
        raise ValueError("dual fusion training artifact hash mismatch")
    exit_head = value.get("exit_head")
    if exit_head and not exit_head.get("unavailable"):
        exit_path = Path(exit_head["model_dir"]).resolve()
        if not exit_path.is_relative_to(root):
            raise ValueError("exit head path escapes bundle")
        exit_manifest, _ = _verified_manifest(exit_path)
        if (_file_hash(exit_path / "manifest.json") != exit_head["manifest_file_sha256"]
                or exit_manifest["checkpoint_sha256"] != exit_head["checkpoint_sha256"]
                or _hash(exit_manifest["label_contract"]) != exit_head["target_hash"]):
            raise ValueError("independent exit-head artifact binding mismatch")
        for name in ("head.binding.json", "exit-linear.json"):
            if name + "_sha256" in exit_head and _file_hash(exit_path / name) != exit_head[name + "_sha256"]:
                raise ValueError("exit CPU export hash mismatch")
    return value


class DualPredictorSession:
    def __init__(self, device=None, batch_size=8192):
        self.predictor = PredictorSession(device=device, batch_size=batch_size)
        self.last_expert_values = {}
        self.last_prediction = {}

    @property
    def diagnostics(self):
        return self.predictor.diagnostics

    def _predict(self, frame, bundle_dir, as_of, retrospective):
        bundle = verified_bundle(bundle_dir)
        method = self.predictor.predict_retrospective if retrospective else self.predictor.predict
        values = {name: method(expert_inputs(frame, name), entry["path"], as_of=as_of)
                  for name, entry in bundle["experts"].items()}
        values["fusion"] = method(fusion_inputs(frame, values["ma"], values["structure"]), bundle["fusion"]["path"], as_of=as_of)
        self.last_expert_values = values
        self.last_prediction = dict(self.predictor.last_prediction)
        return values["fusion"]

    def predict(self, frame, bundle_dir, *, as_of=None):
        return self._predict(frame, bundle_dir, as_of, False)

    def predict_retrospective(self, frame, bundle_dir, *, as_of=None):
        return self._predict(frame, bundle_dir, as_of, True)


def predict_dual_bundle(features, bundle_dir, *, as_of=None, device=None):
    session = DualPredictorSession(device=device)
    session.predict(features, bundle_dir, as_of=as_of)
    return {**session.last_expert_values, "compute_info": session.diagnostics}


class DualResolver:
    """Independent catalog. Never consults or mutates the legacy active alias."""
    def __init__(self, *, usage_mode="production", model_policy="auto", model_run_id=None,
                 checkpoint=None, calendar_run_id=None, runs_root=None, expected_calendar_hash=None, **_):
        if usage_mode not in {"production", "historical", "retrospective"} or model_policy not in {"auto", "pinned"}:
            raise ValueError("invalid dual usage mode or selection policy")
        if model_policy == "pinned" and (not model_run_id or not checkpoint):
            raise ValueError("pinned dual model requires run/checkpoint")
        if usage_mode == "retrospective" and model_policy != "pinned":
            raise ValueError("retrospective dual use requires explicit pin")
        self.usage_mode, self.model_policy = usage_mode, model_policy
        self.model_run_id, self.checkpoint = model_run_id, checkpoint
        self.catalog, self.catalog_errors, self.cache = [], [], {}
        self.calendar, self.market_dates = {}, []
        root = Path(runs_root or ARTIFACT_RUNS_ROOT)
        roots = [root / model_run_id] if model_run_id else sorted(root.iterdir()) if root.exists() else []
        for run in roots:
            report_path = run / "reports" / "dual-research.json"
            if not report_path.exists():
                continue
            try:
                report = json.loads(report_path.read_text())
                metadata = json.loads((run / "metadata.json").read_text())
                if report.get("status") != "complete" or metadata.get("status") != "complete" or report.get("publication_allowed") is not False:
                    raise ValueError("dual run incomplete or not shadow")
                if calendar_run_id and report["calendar"]["run_id"] != calendar_run_id:
                    continue
                pending_cache, calendar, dates = _verified_catalog_context(report)
                if expected_calendar_hash and expected_calendar_hash != stable_hash(dates):
                    raise ValueError("dual catalog and requested calendar mismatch")
                pending_entries = []
                for entry in report["model_bundles"]:
                    bundle = verified_bundle(entry["path"], check_oof=True)
                    if bundle["target"]["calendar_hash"] != stable_hash(dates) or bundle["data_version"] != report["data_version"]:
                        raise ValueError("dual bundle and catalog calendar/data mismatch")
                    pending_entries.append({"model_run_id": run.name, "checkpoint": entry["name"],
                        "name": entry["name"], "model_dir": entry["path"], "available_at": bundle["available_at"],
                        "actual_training_completed_at": bundle["trained_at"],
                        "probability_threshold": bundle["probability_threshold"], "bundle_sha256": bundle["bundle_sha256"],
                        "exit_model": bundle.get("exit_head"),
                        "manifest_sha256": bundle["bundle_sha256"], "target_hash": bundle["target_hash"],
                        "feature_profile": "ma_trend_v1", "manifest": bundle})
                # Routing-only guard added 2026-10-09 15:53 +08:00. Commit no
                # cache/calendar until every checkpoint in this run is valid.
                self.catalog.extend(pending_entries)
                self.cache.update(pending_cache)
                self.calendar, self.market_dates = calendar, dates
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                self.catalog_errors.append({"model_run_id": run.name, "error": str(exc)})
        self.catalog.sort(key=lambda x: (x["available_at"], x["actual_training_completed_at"]))

    def resolve(self, day):
        from my_strategy.services.czsc_research_models import signal_time
        moment = signal_time(day)
        entries = [x for x in self.catalog if self.model_policy != "pinned"
                   or x["model_run_id"] == self.model_run_id and x["checkpoint"] == self.checkpoint]
        if self.usage_mode != "retrospective":
            entries = [x for x in entries if signal_time(x["available_at"]) <= moment
                       and (self.usage_mode != "production" or pd.Timestamp(x["actual_training_completed_at"]) <= moment)]
        route = {"date": pd.Timestamp(day).date().isoformat(), "usage_mode": self.usage_mode, "model_policy": self.model_policy, "model_family": "dual",
                 "model_dir": None, "probability_threshold": .55, "applied_to_entry": False,
                 "status": "rules_no_model", "reason_codes": ["model_unavailable"], "release_id": None}
        if entries:
            chosen = entries[-1]
            route.update({k: v for k, v in chosen.items() if k != "manifest"})
            route.update(status="retrospective_shadow" if self.usage_mode == "retrospective" else
                         "historical_shadow" if self.usage_mode == "historical" else "production_shadow",
                         reason="dual experts are shadow-only; no independent future window or production release",
                         reason_codes=["dual_model_shadow", "independent_future_window_missing"], applied_to_entry=False)
        return route

    def resolve_dates(self, dates):
        return [self.resolve(day) for day in dates]


def _verified_catalog_context(report):
    """Verify cloned calendar bytes and immutable source/cache identities."""
    import hashlib
    import pyarrow.parquet as pq
    snapshot = report["source_snapshot"]
    source_path = Path(snapshot["source_report_path"])
    if _file_hash(source_path) != snapshot["source_report_sha256"]:
        raise ValueError("dual catalog frozen source report hash mismatch")
    source = json.loads(source_path.read_text())
    calendar = report["calendar"]
    original_path = Path(calendar["source_path"])
    if _file_hash(original_path) != calendar["source_file_sha256"]:
        raise ValueError("dual catalog frozen calendar file hash mismatch")
    original = json.loads(original_path.read_text())
    path = Path(calendar["path"])
    # The driver cloned the verified source with its deterministic JSON writer.
    # This derives exact expected clone bytes even for already-running jobs
    # whose report predates an explicit cloned-file SHA field.
    clone_hash = hashlib.sha256(json.dumps(original, ensure_ascii=False, indent=2, allow_nan=False).encode()).hexdigest()
    expected_file_hash = calendar.get("file_sha256", clone_hash)
    if _file_hash(path) != expected_file_hash:
        raise ValueError("dual catalog cloned calendar file hash mismatch")
    payload = json.loads(path.read_text())
    dates = report["market_dates"]
    if not isinstance(dates, list) or any(not isinstance(day, str) for day in dates):
        raise ValueError("dual calendar dates must be ISO string sessions")
    if (not payload.get("verified") or not isinstance(payload.get("source"), str) or not payload["source"].strip() or payload != original
            or stable_hash(payload) != calendar["hash"] or source["calendar"]["hash"] != calendar["hash"]
            or dates != sorted(set(dates)) or dates != payload["dates"]
            or source["model_manifest"]["label_contract"]["calendar_hash"] != stable_hash(dates)):
        raise ValueError("dual catalog verified calendar identity mismatch")
    source_records = {record["symbol"]: record for record in source["dataset_records"]}
    profile = get_feature_profile("ma_trend_v1")
    cache = {}
    for record in report["dataset_records"]:
        old = source_records.get(record["symbol"])
        if old is None or any(record.get(key) != old.get(key) for key in ("path", "sha256", "data_version", "data_end")):
            raise ValueError("dual cache and frozen source record mismatch")
        meta = pq.read_schema(record["path"]).metadata or {}
        attrs = json.loads(meta.get(b"PANDAS_ATTRS", b"{}"))
        if (attrs.get("feature_version") != profile["version"] or attrs.get("schema_hash") != profile["schema_hash"]
                or attrs.get("feature_columns") != profile["columns"] or attrs.get("feature_profile") != profile["name"]
                or attrs.get("calendar_hash") != stable_hash(dates) or attrs.get("data_version") != record["data_version"]):
            raise ValueError("dual cached feature attrs/schema/calendar mismatch")
        cache[record["symbol"]] = record
    return cache, calendar, dates
