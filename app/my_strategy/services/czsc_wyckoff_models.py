"""Independent Wyckoff expert, strict forward OOF stack, and immutable routing.

Existing MA/structure bytes are reused only with exactly the same source/y10
contract. New L2 fits live in their own family and can never publish themselves.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
import pandas as pd
from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_dual_models import (expert_inputs as dual_expert_inputs,
    oof_splits, verified_bundle as verified_dual_bundle, _verified_catalog_context as verified_source_context)
from my_strategy.services.czsc_research_ml import (PredictorSession, MODEL_VERSION, LABEL_VERSION,
    _date, _hash, _file_hash, _features, _fit_preprocess, _transform, _metrics, _network, _device, _verified_manifest)
from my_strategy.services.czsc_research_profiles import get_feature_profile

WYCKOFF_VERSION = "czsc_wyckoff_three_experts_v1"
FUSION_COLUMNS = ["p_ma", "p_structure", "p_wyckoff"]
FUSION_VERSION = "wyckoff_three_probability_oof_v1"
FUSION_HASH = stable_hash({"version": FUSION_VERSION, "columns": FUSION_COLUMNS,
    "fit": "strict_forward_oof_only", "target": "czsc_fixed_horizon_broker_v1"})


def _save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def train_regularized_model(dataset: pd.DataFrame, feature_columns: Sequence[str], output_dir: str | Path,
                train_end: str, validation_start: str, validation_end: str,
                device: str = "auto", seed: int = 42, *, epochs: int = 30,
                batch_size: int = 2048, learning_rate: float = 0.001, weight_decay: float = 0.001,
                hidden_sizes: Sequence[int] = (32, 16), min_train_rows: int = 100,
                min_validation_rows: int = 30, progress: Callable[[int, int], None] | None = None,
                check_cancel: Callable[[], None] | None = None) -> dict[str, Any]:
    """Fixed architecture/epochs; training-only preprocessing, validation-only calibration.

    train_end is the last available training label date, not just signal date.
    Validation labels mature by validation_end. The checkpoint is consequently
    available at validation_end close and cannot predict earlier replay rows.
    Test rows are never used for fitting, calibration or hyperparameter choice.
    progress receives (completed_epochs, total_epochs); check_cancel may raise
    the caller's cancellation exception, which propagates without a checkpoint.
    """
    import torch
    from my_strategy.core.tz import local_now
    training_started_at = local_now().isoformat()
    check = check_cancel or (lambda: None)
    check()
    train_end, validation_start, validation_end = map(_date, (train_end, validation_start, validation_end))
    if not train_end < validation_start <= validation_end:
        raise ValueError("training and validation periods must be chronological and disjoint")
    if not math.isfinite(float(weight_decay)) or weight_decay < 0:
        raise ValueError("invalid L2 penalty")
    if epochs < 1 or batch_size < 1 or not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise ValueError("invalid training hyperparameters")
    if not {"date", "label", "label_end", "label_available"}.issubset(dataset.columns):
        raise ValueError("dataset missing chronological label metadata")
    feature_version = dataset.attrs.get("feature_version")
    feature_schema_hash = dataset.attrs.get("schema_hash")
    if (feature_version is None) != (feature_schema_hash is None):
        raise ValueError("feature semantic binding requires both feature_version and schema_hash")
    values = _features(dataset, feature_columns)
    dates = pd.to_datetime(dataset["date"]).dt.strftime("%Y-%m-%d")
    label_ends = pd.to_datetime(dataset["label_end"], errors="coerce").dt.strftime("%Y-%m-%d")
    labels = pd.to_numeric(dataset["label"], errors="coerce").to_numpy(dtype=float)
    eligible = dataset.get("input_eligible", pd.Series(True, index=dataset.index)).fillna(False).astype(bool).to_numpy()
    available = dataset["label_available"].fillna(False).astype(bool).to_numpy() & eligible & np.isin(labels, [0, 1]) & label_ends.notna().to_numpy()
    if (available & (label_ends <= dates).to_numpy()).any():
        raise ValueError("available label end must be later than its signal date")
    train_mask = available & (dates <= train_end).to_numpy() & (label_ends <= train_end).to_numpy() & (label_ends < validation_start).to_numpy()
    validation_mask = available & (dates >= validation_start).to_numpy() & (dates <= validation_end).to_numpy() & (label_ends <= validation_end).to_numpy()
    if int(train_mask.sum()) < min_train_rows or int(validation_mask.sum()) < min_validation_rows:
        raise ValueError(f"insufficient mature samples: train={train_mask.sum()}, validation={validation_mask.sum()}")
    if len(np.unique(labels[train_mask])) < 2 or len(np.unique(labels[validation_mask])) < 2:
        raise ValueError("training and validation each require both label classes")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from my_strategy.core.device import requested_torch_device
    requested_device = requested_torch_device(device)
    selected = _device(device)
    torch.manual_seed(seed)
    if selected.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    # Small matrix GEMMs are reproducible without TF32 approximation.
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    preprocess = _fit_preprocess(values[train_mask])
    transformed = _transform(values, preprocess)
    train_x = torch.from_numpy(transformed[train_mask])
    train_y = torch.from_numpy(labels[train_mask].astype(np.float32)).reshape(-1, 1)
    model = _network(len(feature_columns), hidden_sizes).to(selected)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    training_batches = 0
    model.train()
    for epoch in range(epochs):
        check()
        permutation = torch.randperm(len(train_x), generator=generator)
        total = 0.0
        for start in range(0, len(train_x), batch_size):
            check()
            indices = permutation[start:start + batch_size]
            batch_x, batch_y = train_x[indices].to(selected), train_y[indices].to(selected)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            training_batches += 1
            total += float(loss.detach().item()) * len(indices)
        losses.append(total / len(train_x))
        if progress:
            progress(epoch + 1, epochs)
    model.eval()
    def logits(mask: np.ndarray) -> np.ndarray:
        chunks = []
        with torch.no_grad():
            selected_values = transformed[mask]
            for start in range(0, len(selected_values), batch_size):
                check()
                chunks.append(model(torch.from_numpy(selected_values[start:start + batch_size]).to(selected)).cpu().numpy().reshape(-1))
        return np.concatenate(chunks).astype(float)
    train_logits, validation_logits = logits(train_mask), logits(validation_mask)
    temperatures = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    def probability(z: np.ndarray, temperature: float) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z / temperature, -40, 40)))
    temperature_scores = []
    for temperature in temperatures:
        check()
        temperature_scores.append(_metrics(labels[validation_mask], probability(validation_logits, temperature))["log_loss"])
    temperature = temperatures[int(np.argmin(temperature_scores))]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists() or (destination / "model.pt").exists():
        raise FileExistsError("model output already exists; use an isolated run directory")
    checkpoint = destination / "model.pt"
    check()
    torch.save({"state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                "feature_columns": list(feature_columns), "hidden_sizes": list(hidden_sizes), "model_version": MODEL_VERSION,
                "feature_version": feature_version, "feature_schema_hash": feature_schema_hash}, checkpoint)
    schema = {"columns": list(feature_columns), "model_version": MODEL_VERSION, "hidden_sizes": list(hidden_sizes)}
    manifest = {"model_version": MODEL_VERSION, "architecture": "ma_logistic" if not hidden_sizes else "mlp",
                "label_version": dataset.attrs.get("label_version", LABEL_VERSION),
                "label_contract": dataset.attrs.get("label_contract", {}), "feature_version": feature_version,
                "feature_schema_hash": feature_schema_hash, "feature_schema_binding": "bound" if feature_version is not None else "unbound_research_fixture",
                "data_version": dataset.attrs.get("data_version"), "schema": schema, "schema_sha256": _hash(schema),
                "preprocess": preprocess, "preprocess_sha256": _hash(preprocess), "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _file_hash(checkpoint), "available_at": validation_end, "train_end": train_end,
                "validation_start": validation_start, "validation_end": validation_end,
                "train_signal_start": str(dates[train_mask].min()), "train_signal_end": str(dates[train_mask].max()),
                "train_label_end": str(label_ends[train_mask].max()), "validation_label_end": str(label_ends[validation_mask].max()),
                "seed": seed, "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda, "device": str(selected),
                "training_started_at": training_started_at, "training_completed_at": local_now().isoformat(),
                "device_name": torch.cuda.get_device_name(selected) if selected.type == "cuda" else "Apple MPS" if selected.type == "mps" else "CPU",
                "actual_cuda_training": selected.type == "cuda", "actual_mps_training": selected.type == "mps",
                "actual_gpu_training": selected.type in {"cuda", "mps"}, "requested_device": requested_device,
                "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "weight_decay": weight_decay, "regularization": "L2 Adam"},
                "training_loss": losses, "training_batches": training_batches,
                "calibration": {"method": "validation_temperature_grid", "temperature": temperature,
                    "temperatures": list(temperatures), "validation_log_losses": temperature_scores, "fit_start": validation_start, "fit_end": validation_end},
                "train_metrics": _metrics(labels[train_mask], probability(train_logits, temperature)),
                "validation_metrics_uncalibrated": _metrics(labels[validation_mask], probability(validation_logits, 1)),
                "validation_metrics": _metrics(labels[validation_mask], probability(validation_logits, temperature)),
                "sample_counts": {"total": len(dataset), "unavailable_or_ineligible": int((~available).sum()), "train": int(train_mask.sum()), "validation": int(validation_mask.sum())},
                "dataset_sha256": hashlib.sha256(pd.util.hash_pandas_object(dataset.loc[train_mask | validation_mask], index=True).to_numpy().tobytes()).hexdigest(),
                "promotion_status": "shadow_pending_complete_universe_ledger_gate",
                "limitations": ["Validation calibration metrics are fitted evidence, not independent test results.",
                    "Feasible-label selection can bias training; evaluation must retain rejected trades and unexited positions.",
                    "No claim of improvement follows from classifier accuracy or model availability."]}
    manifest["manifest_sha256"] = _hash(manifest)
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return manifest

def expert_inputs(frame, expert):
    if expert in {"ma", "structure"}:
        return dual_expert_inputs(frame, expert)
    if expert != "wyckoff":
        raise ValueError("unknown Wyckoff expert")
    from my_strategy.services.czsc_wyckoff_features import bind_wyckoff_profile, get_wyckoff_profile
    result = bind_wyckoff_profile(frame)
    result["input_eligible"] = result["wyckoff_input_eligible"].fillna(False).astype(bool)
    metadata = [c for c in ("date", "symbol", "label", "label_end", "label_available", "net_return", "input_eligible") if c in result]
    return result[list(dict.fromkeys([*get_wyckoff_profile()["columns"], *metadata]))].copy()


def fusion_inputs(frame, ma, structure, wyckoff):
    columns = [c for c in ("date", "symbol", "label", "label_end", "label_available", "net_return") if c in frame]
    result = frame[columns].copy()
    result["p_ma"], result["p_structure"], result["p_wyckoff"] = (np.asarray(x, float) for x in (ma, structure, wyckoff))
    result["input_eligible"] = np.isfinite(result[FUSION_COLUMNS]).all(axis=1)
    result.attrs = {**frame.attrs, "feature_version": FUSION_VERSION, "schema_hash": FUSION_HASH,
                    "feature_columns": FUSION_COLUMNS, "feature_profile": "wyckoff_three_probability_oof_v1"}
    return result


def freeze_broker_threshold(scores, config):
    """Only fixed validation candidate Broker ledgers may select a threshold."""
    grid = [float(x) for x in config.get("threshold_grid", [.5, .55, .6, .65])]
    minimum = int(config.get("threshold_min_completed_round_trips", 30))
    minimum_coverage = float(config.get("threshold_min_candidate_coverage", .01))
    if not grid or any(not 0 < x < 1 for x in grid):
        raise ValueError("invalid Wyckoff threshold grid")
    if {float(s["threshold"]) for s in scores} != set(grid):
        raise ValueError("Broker threshold evidence must cover exactly the frozen grid")
    checked = []
    for score in scores:
        if score.get("evidence") != "complete_validation_broker_ledger":
            raise ValueError("threshold requires real complete Broker ledger evidence")
        row = dict(score)
        row["selectable"] = (row.get("completed_round_trips", 0) >= minimum
            and row.get("candidate_coverage", 0.) >= minimum_coverage
            and math.isfinite(float(row.get("net_return", float("nan"))))
            and math.isfinite(float(row.get("max_drawdown", float("nan")))))
        checked.append(row)
    candidates = [s for s in checked if s["selectable"]]
    selected = max(candidates, key=lambda s: (s["net_return"], -abs(s["max_drawdown"]), -s["threshold"])) if candidates else None
    return {"threshold": selected["threshold"] if selected else None,
        "method": "validation_fixed_candidates_complete_broker_return_coverage_drawdown",
        "scores": checked, "minimum_completed_round_trips": minimum, "minimum_candidate_coverage": minimum_coverage,
        "selected": selected is not None, "unavailable": None if selected else "no_threshold_meets_trade_and_coverage_requirements",
        "qualification": "fitted_validation_diagnostic_not_independent_policy_qualification"}


def _fit_wyckoff(data, directory, split, config, device, progress, check_cancel):
    from my_strategy.services.czsc_wyckoff_features import get_wyckoff_profile
    return train_regularized_model(expert_inputs(data, "wyckoff"), get_wyckoff_profile()["columns"], directory,
        split["train_end"], split["validation_start"], split["validation_end"], device=device,
        seed=int(config.get("seed", 42)), epochs=int(config.get("epochs", 20)),
        batch_size=int(config.get("batch_size", 2048)), hidden_sizes=(),
        weight_decay=float(config.get("wyckoff_l2", .001)),
        min_train_rows=int(config.get("min_train_rows", 100)), min_validation_rows=int(config.get("min_validation_rows", 30)),
        progress=progress, check_cancel=check_cancel)


def _cached_wyckoff_oof(data, root, split, old_block, config, device, predictor, progress, check_cancel):
    identity = _hash({"data_version": data.attrs["data_version"], "target": data.attrs["label_contract"],
        "split": split, "config": config, "device": device, "old_oof_cache_sha256": old_block["cache_sha256"]})
    directory = root / split["name"] / identity
    header_path, rows_path = directory / "cache.json", directory / "predictions.parquet"
    if header_path.exists():
        header = json.loads(header_path.read_text()); unsigned = dict(header); recorded = unsigned.pop("cache_sha256", None)
        if recorded != _hash(unsigned) or header["identity"] != identity or _file_hash(rows_path) != header["rows_sha256"]:
            raise ValueError("Wyckoff OOF cache integrity mismatch")
        model, _ = _verified_manifest(directory / "expert")
        if model != header["wyckoff_model"]:
            raise ValueError("Wyckoff OOF model binding mismatch")
        if _file_hash(Path(header["old_rows_path"])) != header["old_rows_sha256"]:
            raise ValueError("reused dual OOF changed")
        return pd.read_parquet(rows_path), {**header, "reused": True}, None
    if old_block["fold"] != split or _file_hash(Path(old_block["rows_path"])) != old_block["rows_sha256"]:
        raise ValueError("old dual OOF split/hash mismatch")
    model = _fit_wyckoff(data, directory / "expert", split, config, device, progress, check_cancel)
    dates = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    forward = data.loc[(dates >= split["test_start"]) & (dates <= split["test_end"])].copy()
    p = predictor.predict(expert_inputs(forward, "wyckoff"), directory / "expert", as_of=split["test_end"])
    old = pd.read_parquet(old_block["rows_path"])
    old["date"] = pd.to_datetime(old["date"]).dt.strftime("%Y-%m-%d")
    keys = forward[["date", "symbol"]].copy(); keys["date"] = pd.to_datetime(keys["date"]).dt.strftime("%Y-%m-%d")
    aligned = keys.merge(old[["date", "symbol", "p_ma", "p_structure"]], how="left", on=["date", "symbol"], validate="one_to_one")
    meta = fusion_inputs(forward, aligned.p_ma, aligned.p_structure, p)
    meta = meta.loc[meta.input_eligible].copy()
    meta["oof_fold"], meta["expert_train_end"] = split["name"], split["train_end"]
    meta["expert_validation_end"] = meta["expert_available_at"] = split["validation_end"]
    meta["target_label_end"] = meta.label_end
    if (pd.to_datetime(meta.date) <= pd.Timestamp(split["validation_end"])).any():
        raise ValueError("OOF model not available at signal time")
    directory.mkdir(parents=True, exist_ok=True); meta.to_parquet(rows_path, index=False)
    header = {"identity": identity, "fold": split, "wyckoff_model": model,
        "old_rows_path": old_block["rows_path"], "old_rows_sha256": old_block["rows_sha256"],
        "old_cache_sha256": old_block["cache_sha256"], "rows_path": str(rows_path.resolve()), "rows_sha256": _file_hash(rows_path)}
    header["cache_sha256"] = _hash(header); _save(header_path, header)
    return meta, {**header, "reused": False}, model


def train_wyckoff_bundle(data, output_dir, fold, *, baseline_bundle_dir, device="mps", config=None,
                         threshold_evaluator=None, progress=None, check_cancel=None):
    """Fit new W/stack using source-bound immutable MA/S and forward OOF only."""
    from my_strategy.core.tz import local_now
    cfg = dict(config or {}); path = Path(output_dir)
    if (path / "bundle.json").exists():
        raise FileExistsError("Wyckoff bundle already exists")
    baseline = verified_dual_bundle(baseline_bundle_dir, check_oof=True)
    if baseline["target"] != data.attrs["label_contract"] or any(baseline[k] != fold[k] for k in ("train_end", "validation_start", "validation_end")):
        raise ValueError("reused dual experts must have same target and exact fold cutoffs")
    check = check_cancel or (lambda: None); predictor = PredictorSession(device=device)
    oof_parts, oof_evidence, new_models = [], [], []
    old_blocks = {x["fold"]["name"]: x for x in baseline["oof"]["folds"]}
    for split in oof_splits(cfg.get("feature_start", "2023-01-01"), fold["train_end"]):
        check()
        if split["name"] not in old_blocks:
            raise ValueError("baseline forward OOF block missing")
        meta, evidence, model = _cached_wyckoff_oof(data, path.parent / "oof-shared", split,
            old_blocks[split["name"]], cfg, device, predictor, progress, check_cancel)
        mature = meta.label_available.astype(bool) & (pd.to_datetime(meta.label_end) <= pd.Timestamp(fold["train_end"]))
        oof_parts.append(meta.loc[mature].copy()); oof_evidence.append(evidence)
        if model is not None: new_models.append(model)
    if not oof_parts: raise ValueError("strict forward OOF unavailable")
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(["date", "symbol"])
    if oof.duplicated(["date", "symbol"]).any(): raise ValueError("duplicate OOF signal identity")
    oof.attrs = {**data.attrs, "feature_version": FUSION_VERSION, "schema_hash": FUSION_HASH, "feature_columns": FUSION_COLUMNS}
    op = path / "oof/oof.parquet"; op.parent.mkdir(parents=True, exist_ok=True); oof.to_parquet(op, index=False)
    wyckoff = _fit_wyckoff(data, path / "experts/wyckoff", fold, cfg, device, progress, check_cancel)
    new_models.append(wyckoff)
    dates = pd.to_datetime(data.date).dt.strftime("%Y-%m-%d")
    mask = (dates >= fold["validation_start"]) & (dates <= fold["validation_end"])
    validation = data.loc[mask].copy()
    expert_bindings = {name: {**binding, "reuse": "immutable_same_source_y10_dual_expert",
        "source_bundle_dir": str(Path(baseline_bundle_dir).resolve()), "source_bundle_sha256": baseline["bundle_sha256"]}
        for name, binding in baseline["experts"].items()}
    expert_bindings["wyckoff"] = _model_binding(path / "experts/wyckoff", wyckoff)
    def predict_validation(frame):
        values = {name: predictor.predict_retrospective(expert_inputs(frame, name), binding["path"], as_of=fold["validation_end"])
                  for name, binding in expert_bindings.items()}
        if (path / "fusion/manifest.json").exists():
            values["fusion"] = predictor.predict_retrospective(fusion_inputs(frame, values["ma"], values["structure"], values["wyckoff"]),
                path / "fusion", as_of=fold["validation_end"])
        return values
    vp = predict_validation(validation)
    vmeta = fusion_inputs(validation, vp["ma"], vp["structure"], vp["wyckoff"])
    stacked = pd.concat([oof, vmeta], ignore_index=True); stacked.attrs = dict(oof.attrs)
    sp = path / "fusion-training.parquet"; stacked.to_parquet(sp, index=False)
    fusion = train_regularized_model(stacked, FUSION_COLUMNS, path / "fusion", fold["train_end"],
        fold["validation_start"], fold["validation_end"], device=device, seed=cfg.get("seed", 42),
        epochs=cfg.get("epochs", 20), batch_size=cfg.get("batch_size", 2048), hidden_sizes=(),
        weight_decay=cfg.get("fusion_l2", .01), min_train_rows=cfg.get("min_train_rows", 100),
        min_validation_rows=cfg.get("min_validation_rows", 30), progress=progress, check_cancel=check_cancel)
    new_models.append(fusion)
    thresholds = ({name: freeze_broker_threshold(scores, cfg) for name, scores in
        threshold_evaluator(predict_validation, fold).items()} if threshold_evaluator else
        {name: {"threshold": None, "selected": False, "unavailable": "broker_validation_evaluator_not_provided",
                "method": "none_no_proxy_threshold", "qualification": "shadow_only"} for name in ("wyckoff", "fusion")})
    completed = local_now().isoformat()
    bundle = {"version": WYCKOFF_VERSION, "model_family": "wyckoff", "status": "shadow", "publication_allowed": False,
        "available_at": fold["validation_end"], "trained_at": completed, "training_completed_at": completed,
        "train_end": fold["train_end"], "train_label_end": max(x["train_label_end"] for x in new_models),
        "validation_start": fold["validation_start"], "validation_end": fold["validation_end"],
        "probability_threshold": thresholds["fusion"]["threshold"] if thresholds["fusion"]["selected"] else .55,
        "thresholds": thresholds, "target": data.attrs["label_contract"], "target_hash": _hash(data.attrs["label_contract"]),
        "data_version": data.attrs["data_version"], "experts": expert_bindings,
        "fusion": {**_model_binding(path / "fusion", fusion), "dataset_path": str(sp.resolve()), "dataset_file_sha256": _file_hash(sp)},
        "baseline_bundle": {"path": str(Path(baseline_bundle_dir).resolve()), "sha256": baseline["bundle_sha256"],
            "data_version": baseline["data_version"], "source_identity": data.attrs.get("source_snapshot_hash")},
        "oof": {"path": str(op.resolve()), "sha256": _file_hash(op), "rows": len(oof), "folds": oof_evidence,
            "method": "strict_forward_three_expert_same_y10", "fusion_fit": "only outcomes mature by outer train_end",
            "validation_predictions": "explicit_fitted_validation_retrospective_never_OOF"},
        "compute_info": {"training_batches": sum(x["training_batches"] for x in new_models), "trained_models": len(new_models),
            "reused_experts": 2, "reused_oof_blocks": sum(x["reused"] for x in oof_evidence),
            "actual_mps_training": any(x["actual_mps_training"] for x in new_models),
            "actual_cuda_training": any(x["actual_cuda_training"] for x in new_models), "inference": predictor.diagnostics},
        "limitations": ["Observed history is development only.", "Fixed ten-day probabilities are not policy/exit qualification.",
            "Critical corporate-action/ST/source boundaries remain explicit; publication prohibited."]}
    _seal_bundle(path, bundle); return bundle


def _model_binding(path, manifest):
    return {"path": str(Path(path).resolve()), "manifest_file_sha256": _file_hash(Path(path) / "manifest.json"),
        **{key: manifest[key] for key in ("manifest_sha256", "checkpoint_sha256", "preprocess_sha256", "feature_version")},
        "feature_schema_hash": manifest["feature_schema_hash"]}


def _seal_bundle(path, bundle):
    bundle.pop("bundle_sha256", None); bundle["bundle_sha256"] = _hash(bundle)
    _save(Path(path) / "bundle.json", bundle); _save(Path(path) / "manifest.json", bundle)


def attach_heads(directory, *, exit_head=None, policy_heads=None, exit_heads=None):
    """After every head finishes, bind it and advance actual complete clock."""
    from my_strategy.core.tz import local_now
    path = Path(directory); bundle = verified_bundle(path)
    def binding(value):
        if not value or value.get("unavailable"): return value
        result = dict(value); mp = Path(result["model_dir"]).resolve()
        if not mp.is_relative_to(path.resolve()): raise ValueError("new head must live in its Wyckoff bundle")
        manifest, _ = _verified_manifest(mp)
        result.update(manifest_file_sha256=_file_hash(mp / "manifest.json"), manifest_sha256=manifest["manifest_sha256"],
            checkpoint_sha256=manifest["checkpoint_sha256"], preprocess_sha256=manifest["preprocess_sha256"],
            target_hash=_hash(manifest["label_contract"]), label_version=manifest["label_version"],
            **{key: manifest["label_contract"][key] for key in ("entry_policy", "candidate_policy", "behavior_policy", "continuation_policy_hash", "source_snapshot_hash", "calendar_hash")},
            training_batches=manifest["training_batches"], actual_training_device=manifest["device"])
        if (mp / "head.binding.json").exists(): result["head_binding_sha256"] = json.loads((mp / "head.binding.json").read_text())["binding_sha256"]
        for ep in ("head.binding.json", "exit-linear.json", "policy-linear.json"):
            if (mp / ep).exists(): result[ep + "_sha256"] = _file_hash(mp / ep)
        return result
    if exit_head is not None: bundle["exit_head"] = binding(exit_head); bundle["exit_model"] = bundle["exit_head"]
    if exit_heads is not None: bundle["exit_heads"] = {k: binding(v) for k,v in exit_heads.items()}
    if policy_heads is not None: bundle["policy_heads"] = {k: binding(v) for k,v in policy_heads.items()}
    bundle["training_completed_at"] = bundle["trained_at"] = local_now().isoformat(); _seal_bundle(path, bundle)
    return bundle


def verified_bundle(directory, *, check_oof=False):
    from my_strategy.services.czsc_wyckoff_features import get_wyckoff_profile
    root = Path(directory).resolve(); value = json.loads((root / "bundle.json").read_text())
    if json.loads((root / "manifest.json").read_text()) != value: raise ValueError("Wyckoff manifest alias mismatch")
    unsigned = dict(value); recorded = unsigned.pop("bundle_sha256", None)
    if recorded != _hash(unsigned) or value.get("version") != WYCKOFF_VERSION or value.get("status") != "shadow" or value.get("publication_allowed") is not False:
        raise ValueError("Wyckoff identity/integrity/shadow mismatch")
    if value["target_hash"] != _hash(value["target"]) or value["target"].get("label_version") != "czsc_fixed_horizon_broker_v1":
        raise ValueError("Wyckoff y10 target mismatch")
    if not value["train_label_end"] <= value["train_end"] < value["validation_start"] <= value["validation_end"] == value["available_at"]:
        raise ValueError("Wyckoff chronological cutoff invalid")
    base = verified_dual_bundle(value["baseline_bundle"]["path"], check_oof=check_oof)
    if base["bundle_sha256"] != value["baseline_bundle"]["sha256"] or base["target"] != value["target"]:
        raise ValueError("reused dual baseline binding mismatch")
    profiles = {"ma": get_feature_profile("ma_trend_v1"), "structure": get_feature_profile("czsc_structure_v1"), "wyckoff": get_wyckoff_profile()}
    for name, binding in [*value["experts"].items(), ("fusion", value["fusion"])]:
        path = Path(binding["path"]).resolve(); manifest, _ = _verified_manifest(path)
        if name in {"ma", "structure"}:
            if path != Path(base["experts"][name]["path"]).resolve(): raise ValueError("external expert is not bound immutable baseline")
        elif not path.is_relative_to(root): raise ValueError("new checkpoint escapes Wyckoff bundle")
        expected = profiles[name] if name != "fusion" else {"version": FUSION_VERSION, "schema_hash": FUSION_HASH, "columns": FUSION_COLUMNS}
        if (manifest["feature_version"], manifest["feature_schema_hash"], manifest["schema"]["columns"]) != (expected["version"], expected["schema_hash"], expected["columns"]):
            raise ValueError("Wyckoff expert feature schema mismatch")
        if any(manifest[k] != binding[k] for k in ("manifest_sha256", "checkpoint_sha256", "preprocess_sha256")) or _file_hash(path / "manifest.json") != binding["manifest_file_sha256"] or manifest["label_contract"] != value["target"] or manifest["available_at"] != value["available_at"]:
            raise ValueError("Wyckoff artifact/target binding mismatch")
    if check_oof:
        for item, pkey, hkey in ((value["oof"], "path", "sha256"), (value["fusion"], "dataset_path", "dataset_file_sha256")):
            if _file_hash(Path(item[pkey])) != item[hkey]: raise ValueError("Wyckoff OOF training data hash mismatch")
    from my_strategy.services.czsc_wyckoff_exit import ExitPredictor, PolicyPredictor
    tagged = [("fresh", value.get("exit_head"), "exit"), *[(key,head,"exit") for key,head in value.get("exit_heads", {}).items()], *[(key,head,"policy") for key,head in value.get("policy_heads", {}).items()]]
    for key,head,kind in tagged:
        if not head or head.get("unavailable"): continue
        hp = Path(head["model_dir"]).resolve()
        if not hp.is_relative_to(root): raise ValueError("Wyckoff head path escapes bundle")
        manifest, _ = _verified_manifest(hp)
        if _file_hash(hp / "manifest.json") != head["manifest_file_sha256"] or manifest["checkpoint_sha256"] != head["checkpoint_sha256"] or _hash(manifest["label_contract"]) != head["target_hash"]:
            raise ValueError("Wyckoff head target/artifact mismatch")
        contract=manifest["label_contract"]
        expected_entry,expected_candidate=("fresh","fresh_v1") if key=="fresh" else ("risk","fresh_wyckoff_union_v1")
        if (key not in {"fresh","union"} or contract["calendar_hash"] != value["target"]["calendar_hash"]
                or contract["source_snapshot_hash"] != value["baseline_bundle"]["source_identity"]
                or manifest["available_at"] != value["available_at"]
                or contract["entry_policy"] != expected_entry or contract["candidate_policy"] != expected_candidate):
            raise ValueError("Wyckoff head source/calendar/availability/entry identity mismatch")
        (ExitPredictor if kind=="exit" else PolicyPredictor)(head,device="cpu")
        for ep in ("head.binding.json", "exit-linear.json", "policy-linear.json"):
            if ep + "_sha256" in head and _file_hash(hp / ep) != head[ep + "_sha256"]: raise ValueError("Wyckoff CPU head export mismatch")
    return value


class WyckoffPredictorSession:
    def __init__(self, device=None, batch_size=8192):
        self.predictor = PredictorSession(device=device, batch_size=batch_size)
        self.last_expert_values, self.last_prediction = {}, {}
        self.last_dual_fallback_values, self.last_fallback_model = np.array([]), None
        self.fallback_diagnostics = {"rows": 0, "batches": 0, "model_loads": 0}
        self.expert_diagnostics = {name:{"rows":0,"batches":0,"model_loads":0} for name in ("ma","structure","wyckoff","fusion","dual_fusion")}
    @property
    def diagnostics(self): return self.predictor.diagnostics
    def _tracked(self,expert,method,frame,path,as_of):
        before=self.predictor.diagnostics;result=method(frame,path,as_of=as_of);after=self.predictor.diagnostics
        for key in ("rows","batches","model_loads"):self.expert_diagnostics[expert][key]+=after[key]-before[key]
        return result
    def _predict(self, frame, bundle_dir, as_of, retrospective):
        bundle = verified_bundle(bundle_dir); method = self.predictor.predict_retrospective if retrospective else self.predictor.predict
        values = {name: self._tracked(name,method,expert_inputs(frame, name),binding["path"],as_of) for name,binding in bundle["experts"].items()}
        values["fusion"] = self._tracked("fusion",method,fusion_inputs(frame, values["ma"], values["structure"], values["wyckoff"]), bundle["fusion"]["path"], as_of)
        self.last_prediction = dict(self.predictor.last_prediction)
        self.last_dual_fallback_values = np.full(len(frame), np.nan)
        self.last_fallback_compute_info = {"rows": 0, "batches": 0, "model_loads": 0}
        fallback_mask = ~np.isfinite(values["fusion"]) & np.isfinite(values["ma"]) & np.isfinite(values["structure"])
        self.last_fallback_model = None
        if fallback_mask.any():
            from my_strategy.services.czsc_dual_models import fusion_inputs as dual_fusion_inputs
            baseline = verified_dual_bundle(bundle["baseline_bundle"]["path"])
            projected = dual_fusion_inputs(frame.loc[fallback_mask].copy(), values["ma"][fallback_mask], values["structure"][fallback_mask])
            before = self.predictor.diagnostics
            self.last_dual_fallback_values[fallback_mask] = self._tracked("dual_fusion",method,projected,baseline["fusion"]["path"],as_of)
            after = self.predictor.diagnostics
            self.last_fallback_compute_info = {k: after[k] - before[k] for k in ("rows", "batches", "model_loads")}
            for key,value in self.last_fallback_compute_info.items(): self.fallback_diagnostics[key] += value
            self.last_fallback_model = {"bundle_dir": bundle["baseline_bundle"]["path"], "bundle_sha256": baseline["bundle_sha256"], "family": "dual", "status": "dual_fallback_shadow",
                "model_run_id": Path(bundle["baseline_bundle"]["path"]).parent.parent.name,
                "checkpoint": Path(bundle["baseline_bundle"]["path"]).name,
                "model_dir": bundle["baseline_bundle"]["path"], "available_at": baseline["available_at"],
                "probability_threshold": baseline["probability_threshold"], "target": baseline["target"],
                "target_hash": baseline["target_hash"], "data_version": baseline["data_version"],
                "fusion_manifest": baseline["fusion"]}
        self.last_expert_values = values
        return values["fusion"]
    def predict(self, frame, bundle_dir, *, as_of=None): return self._predict(frame, bundle_dir, as_of, False)
    def predict_retrospective(self, frame, bundle_dir, *, as_of=None): return self._predict(frame, bundle_dir, as_of, True)

    def predict_dual_fallback(self, frame, bundle_dir, *, as_of=None, retrospective=False):
        from my_strategy.services.czsc_dual_models import fusion_inputs as dual_fusion_inputs
        baseline=verified_dual_bundle(bundle_dir);method=self.predictor.predict_retrospective if retrospective else self.predictor.predict
        before=self.predictor.diagnostics
        values={name:self._tracked(name,method,dual_expert_inputs(frame,name),binding["path"],as_of) for name,binding in baseline["experts"].items()}
        fallback=self._tracked("dual_fusion",method,dual_fusion_inputs(frame,values["ma"],values["structure"]),baseline["fusion"]["path"],as_of)
        after=self.predictor.diagnostics;self.last_fallback_compute_info={k:after[k]-before[k] for k in ("rows","batches","model_loads")}
        for key,value in self.last_fallback_compute_info.items():self.fallback_diagnostics[key]+=value
        self.last_expert_values={**values,"wyckoff":np.full(len(frame),np.nan),"fusion":np.full(len(frame),np.nan)}
        self.last_dual_fallback_values=fallback;self.last_prediction=dict(self.predictor.last_prediction)
        self.last_fallback_model={"family":"dual","status":"dual_fallback_shadow","model_run_id":Path(bundle_dir).parent.parent.name,
            "checkpoint":Path(bundle_dir).name,"model_dir":str(bundle_dir),"available_at":baseline["available_at"],
            "probability_threshold":baseline["probability_threshold"],"target_hash":baseline["target_hash"],"target":baseline["target"],
            "data_version":baseline["data_version"],"bundle_sha256":baseline["bundle_sha256"],"fusion_manifest":baseline["fusion"]}
        return np.full(len(frame),np.nan)


class WyckoffResolver:
    def __init__(self, *, usage_mode="production", model_policy="auto", model_run_id=None, checkpoint=None,
                 calendar_run_id=None, runs_root=None, expected_calendar_hash=None, **_):
        if usage_mode not in {"production", "historical", "retrospective"} or model_policy not in {"auto", "pinned"}: raise ValueError("invalid Wyckoff routing policy")
        if model_policy == "pinned" and (not model_run_id or not checkpoint): raise ValueError("pinned Wyckoff requires run/checkpoint")
        if usage_mode == "retrospective" and model_policy != "pinned": raise ValueError("retrospective Wyckoff requires explicit pin")
        self.usage_mode, self.model_policy, self.model_run_id, self.checkpoint = usage_mode, model_policy, model_run_id, checkpoint
        self.catalog, self.catalog_errors, self.cache, self.calendar, self.market_dates = [], [], {}, {}, []
        root=Path(runs_root or ARTIFACT_RUNS_ROOT); roots=[root/model_run_id] if model_run_id else sorted(root.iterdir()) if root.exists() else []
        for run in roots:
            report_path=run/"reports/wyckoff-research.json"
            if not report_path.exists(): continue
            try:
                report=json.loads(report_path.read_text()); metadata=json.loads((run/"metadata.json").read_text())
                if report.get("status") != "complete" or metadata.get("status") != "complete" or report.get("publication_allowed") is not False: raise ValueError("Wyckoff run incomplete or not shadow")
                if calendar_run_id and report["calendar"]["run_id"] != calendar_run_id: continue
                cache,calendar,dates=verified_source_context({**report,"dataset_records": report["source_dataset_records"]})
                if expected_calendar_hash and expected_calendar_hash != stable_hash(dates): raise ValueError("Wyckoff requested calendar mismatch")
                projections={r["symbol"]:r for r in report["dataset_records"]}; pending=[]
                for entry in report["model_bundles"]:
                    bundle=verified_bundle(entry["path"],check_oof=True)
                    if bundle["target"]["calendar_hash"] != stable_hash(dates) or bundle["data_version"] != report["data_version"]: raise ValueError("Wyckoff catalog calendar/data mismatch")
                    pending.append({"model_run_id":run.name,"checkpoint":entry["name"],"name":entry["name"],"model_dir":entry["path"],
                        "available_at":bundle["available_at"],"actual_training_completed_at":bundle["training_completed_at"],
                        "probability_threshold":bundle["probability_threshold"],"manifest_sha256":bundle["bundle_sha256"],"bundle_sha256":bundle["bundle_sha256"],
                        "exit_model":bundle.get("exit_head"),"policy_models":bundle.get("policy_heads"),"source_snapshot_hash":report["source_snapshot"]["source_report_sha256"],"target_hash":bundle["target_hash"],"feature_profile":"ma_trend_v1","manifest":bundle})
                for symbol,record in projections.items():
                    if _file_hash(Path(record["path"])) != record["sha256"] or record["source_sha256"] != cache[symbol]["sha256"]: raise ValueError("Wyckoff projected cache/source mismatch")
                self.catalog.extend(pending); self.cache.update(projections); self.calendar,self.market_dates=calendar,dates
            except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc: self.catalog_errors.append({"model_run_id":run.name,"error":str(exc)})
        self.catalog.sort(key=lambda x:(x["available_at"],x["actual_training_completed_at"]))
        self.dual_fallback_resolver=None
        if model_policy=="auto" and usage_mode!="retrospective":
            from my_strategy.core.config_loader import load_config
            from my_strategy.services.czsc_dual_models import DualResolver
            frozen=load_config("czsc_wyckoff_research")["baseline_dual_run_id"]
            self.dual_fallback_resolver=DualResolver(usage_mode=usage_mode,model_policy="auto",model_run_id=frozen,
                calendar_run_id=calendar_run_id,runs_root=runs_root,expected_calendar_hash=expected_calendar_hash)
            if not self.catalog and self.dual_fallback_resolver.catalog:
                self.cache=dict(self.dual_fallback_resolver.cache);self.calendar=dict(self.dual_fallback_resolver.calendar)
                self.market_dates=list(self.dual_fallback_resolver.market_dates)
    def resolve(self, day):
        from my_strategy.services.czsc_research_models import signal_time
        moment=signal_time(day); entries=[x for x in self.catalog if self.model_policy != "pinned" or x["model_run_id"] == self.model_run_id and x["checkpoint"] == self.checkpoint]
        if self.usage_mode != "retrospective": entries=[x for x in entries if signal_time(x["available_at"]) <= moment and (self.usage_mode != "production" or pd.Timestamp(x["actual_training_completed_at"]) <= moment)]
        route={"date":pd.Timestamp(day).date().isoformat(),"usage_mode":self.usage_mode,"model_policy":self.model_policy,"model_family":"wyckoff",
            "model_dir":None,"probability_threshold":.55,"applied_to_entry":False,"status":"rules_no_model","reason_codes":["model_unavailable"],"release_id":None}
        if entries:
            route.update({k:v for k,v in entries[-1].items() if k != "manifest"}); route.update(status=self.usage_mode+"_shadow",
                reason="Wyckoff experts remain shadow pending independent source/ledger qualification",reason_codes=["wyckoff_model_shadow","independent_future_window_missing"],applied_to_entry=False)
        elif self.model_policy=="auto" and self.dual_fallback_resolver is not None:
            fallback=self.dual_fallback_resolver.resolve(day)
            if fallback.get("model_dir"):
                route.update(fallback);route.update(model_family="wyckoff",bundle_family="dual",primary_model_dir=None,
                    probability_source="dual_fallback",status="dual_fallback_shadow",fallback_model=dict(fallback),
                    reason_codes=["wyckoff_checkpoint_unavailable","dual_fallback_shadow"],applied_to_entry=False)
        return route
    def resolve_dates(self, dates): return [self.resolve(day) for day in dates]


def get_bundle_profile():
    from my_strategy.services.czsc_wyckoff_features import get_wyckoff_profile
    from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS, STRUCTURE_COLUMNS
    columns = [*MA_TREND_COLUMNS, *STRUCTURE_COLUMNS, *get_wyckoff_profile()["columns"]]
    return {"name": "wyckoff_bundle_v1", "version": WYCKOFF_VERSION, "columns": columns,
        "schema_hash": stable_hash({"version": WYCKOFF_VERSION, "experts": [get_feature_profile("ma_trend_v1"), get_feature_profile("czsc_structure_v1"), get_wyckoff_profile()]})}
