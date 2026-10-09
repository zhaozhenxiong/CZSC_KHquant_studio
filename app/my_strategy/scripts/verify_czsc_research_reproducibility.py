"""Retrain a completed frozen production model in a new, independent run.

Source artifacts are read only. This does not evaluate trading performance or
change thresholds; it compares model tensors and real-row inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.tz import local_now
from my_strategy.scripts.audit_czsc_research import _bounded, _file_hash, _hash, _read, _run


def _frozen_device(manifest: dict, requested: str | None = None) -> str:
    """Reproduce the recorded actual device, never an auto-selection preference."""
    frozen = manifest.get("device")
    if not isinstance(frozen, str) or not re.fullmatch(r"cpu|mps|cuda:\d+", frozen):
        raise ValueError("source requires an explicit frozen actual device: cpu, mps or cuda:N")
    if requested is not None and requested != frozen:
        raise ValueError("reproduction device must match the frozen actual device; no fallback")
    return frozen


def _source(run_id: str, runs_root: Path) -> tuple[Path, dict, dict]:
    root = _run(runs_root, run_id)
    metadata = _read(_bounded(root / "metadata.json", root))
    if metadata.get("status") != "complete":
        raise ValueError("source research run is not complete; no retraining permitted")
    research = _read(_bounded(root / "reports" / "research.json", root))
    if research["run_id"] != run_id or research["run_context"]["stocks"] != metadata["stocks"]:
        raise ValueError("source completed run identity mismatch")
    config = research["config"]
    if metadata.get("config_hash") != stable_hash(config) or metadata.get("research_config") != config:
        raise ValueError("source frozen configuration mismatch")
    manifest = _read(_bounded(root / "models" / "production" / "manifest.json", root))
    integrity = dict(manifest)
    recorded = integrity.pop("manifest_sha256", None)
    if recorded != _hash(integrity) or research["model_manifest"] != manifest:
        raise ValueError("source production manifest integrity mismatch")
    if manifest["schema_sha256"] != _hash(manifest["schema"]) or manifest["preprocess_sha256"] != _hash(manifest["preprocess"]):
        raise ValueError("source model schema/preprocess integrity mismatch")
    checkpoint = _bounded(root / "models" / "production" / "model.pt", root)
    if _file_hash(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("source production checkpoint hash mismatch")
    production = config["production"]
    if (manifest["train_end"] != production["train_end"] or manifest["validation_start"] != production["validation_start"]
            or manifest["validation_end"] != min(production["validation_end"], research["data_end"])
            or manifest["seed"] != config["seed"] or manifest["hyperparameters"]["epochs"] != config["epochs"]):
        raise ValueError("source production training differs from frozen configuration")
    for key in ("batch_size", "learning_rate"):
        if key in config and manifest["hyperparameters"].get(key) != config[key]:
            raise ValueError("source production training differs from frozen configuration")
    if "hidden_sizes" in config and manifest["schema"]["hidden_sizes"] != config["hidden_sizes"]:
        raise ValueError("source production architecture differs from frozen configuration")
    _frozen_device(manifest)
    if manifest.get("feature_schema_binding") != "bound":
        raise ValueError("real production model requires bound feature semantics")
    return root, research, manifest


def _semantics(frame: pd.DataFrame, manifest: dict) -> list[str]:
    restored = []
    for key, expected in (("feature_version", manifest["feature_version"]), ("schema_hash", manifest["feature_schema_hash"])):
        if frame.attrs.get(key) is None:
            frame.attrs[key] = expected
            restored.append(key)
        elif frame.attrs[key] != expected:
            raise ValueError(f"frozen parquet semantic mismatch: {key}")
    return restored


def _fitting_hash(dataset: pd.DataFrame, manifest: dict, index: pd.Index | None = None) -> str:
    dates = pd.to_datetime(dataset["date"]).dt.strftime("%Y-%m-%d")
    ends = pd.to_datetime(dataset["label_end"], errors="coerce").dt.strftime("%Y-%m-%d")
    available = dataset["label_available"].fillna(False).astype(bool) & dataset["input_eligible"].fillna(False).astype(bool)
    available &= pd.to_numeric(dataset["label"], errors="coerce").isin([0, 1]) & ends.notna()
    train = (dates <= manifest["train_end"]) & (ends <= manifest["train_end"]) & (ends < manifest["validation_start"])
    validation = (dates >= manifest["validation_start"]) & (dates <= manifest["validation_end"]) & (ends <= manifest["validation_end"])
    selected = dataset.loc[available & (train | validation)].copy()
    if index is not None:
        selected.index = index[(available & (train | validation)).to_numpy()]
    return hashlib.sha256(pd.util.hash_pandas_object(selected, index=True).to_numpy().tobytes()).hexdigest()


def _original_index(dataset: pd.DataFrame, records: list[dict]) -> pd.Index:
    """Recover concat(ignore_index=True) labels before the frozen date/symbol sort."""
    sizes = dataset["symbol"].value_counts().to_dict()
    offsets, total = {}, 0
    for record in records:
        symbol = record["symbol"]
        if symbol in offsets:
            raise ValueError("duplicate frozen dataset record")
        offsets[symbol] = total
        total += sizes.get(symbol, 0)
    if total != len(dataset) or not dataset["symbol"].isin(offsets).all():
        raise ValueError("training symbols differ from frozen records")
    if dataset[["date", "symbol"]].duplicated().any() or not dataset[["date", "symbol"]].equals(dataset[["date", "symbol"]].sort_values(["date", "symbol"])):
        raise ValueError("training rows differ from frozen chronological ordering")
    values = dataset["symbol"].map(offsets).to_numpy() + dataset.groupby("symbol", sort=False).cumcount().to_numpy()
    return pd.Index(values, dtype="int64")


def _probe(root: Path, research: dict, manifest: dict, minimum: int = 20) -> tuple[pd.DataFrame, list[dict]]:
    rows, records = [], []
    profile_name = research.get("config", {}).get("feature_profile", manifest.get("feature_profile", "legacy"))
    if profile_name not in {"legacy", "ma_trend_v1"}:
        raise ValueError("unrecognized frozen model feature profile")
    eligibility = "model_input_eligible" if profile_name == "ma_trend_v1" else "input_eligible"
    columns = ["symbol", "date", "input_eligible", *manifest["schema"]["columns"]]
    if eligibility != "input_eligible":
        columns.insert(3, eligibility)
    for record in sorted(research["dataset_records"], key=lambda item: item["symbol"]):
        symbol = record["symbol"]
        if not (symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ")):
            continue
        path = _bounded(record["path"], root)
        if _file_hash(path) != record["sha256"]:
            raise ValueError(f"frozen feature file hash mismatch: {symbol}")
        frame = pd.read_parquet(path, columns=columns)
        restored = _semantics(frame, manifest)
        selected = frame.loc[(frame["date"] == research["data_end"]) & frame[eligibility].fillna(False).astype(bool)].copy()
        selected["input_eligible"] = selected[eligibility].fillna(False).astype(bool)
        if len(selected) > 1 or not frame["symbol"].eq(symbol).all():
            raise ValueError(f"frozen feature row identity mismatch: {symbol}")
        if len(selected):
            rows.append(selected)
            records.append({"symbol": symbol, "date": research["data_end"], "file_sha256": record["sha256"], "restored_attrs": restored,
                            "eligibility_column": eligibility})
        if len(rows) >= minimum:
            break
    if len(rows) < minimum or research["data_end"] < manifest["available_at"]:
        raise ValueError("at least 20 eligible latest real rows available to the production model are required")
    probe = pd.concat(rows, ignore_index=True)
    _semantics(probe, manifest)
    return probe, records


def _tensor_comparison(left: dict, right: dict) -> dict:
    import torch
    tensors = []
    same_keys = list(left) == list(right)
    for name in left.keys() | right.keys():
        a, b = left.get(name), right.get(name)
        compatible = a is not None and b is not None and a.shape == b.shape and a.dtype == b.dtype
        item: dict[str, Any] = {"name": name, "shape": list(a.shape) if a is not None else None, "compatible": compatible}
        if compatible:
            delta = (a.to(torch.float64) - b.to(torch.float64)).abs()
            item.update(exact=torch.equal(a, b), max_absolute_difference=float(delta.max()),
                        max_relative_difference=float((delta / a.to(torch.float64).abs().clamp_min(1e-12)).max()))
        else:
            item["exact"] = False
        tensors.append(item)
    tensors.sort(key=lambda item: item["name"])
    return {"same_keys": same_keys, "all_tensors_exact": same_keys and all(item["exact"] for item in tensors), "tensors": tensors}


def _probability_comparison(left: np.ndarray, right: np.ndarray) -> dict:
    finite = left.shape == right.shape and bool(np.isfinite(left).all() and np.isfinite(right).all())
    return {"rows": len(left), "all_finite": finite, "absolute_tolerance": 1e-6, "relative_tolerance": 0,
            "max_absolute_difference": float(np.max(np.abs(left - right))) if finite else None,
            "passed": finite and bool(np.allclose(left, right, atol=1e-6, rtol=0)),
            "left": [float(value) if np.isfinite(value) else None for value in left],
            "right": [float(value) if np.isfinite(value) else None for value in right]}


def verify_run(training_run_id: str, *, runs_root: Path | None = None, device: str | None = None, progress=None) -> dict:
    """No Torch/GPU activity until complete-run, hash and real-input guards pass."""
    base = (runs_root or ARTIFACT_RUNS_ROOT).resolve()
    root, research, manifest = _source(training_run_id, base)
    reproduction_device = _frozen_device(manifest, device)
    path = _bounded(root / "dataset" / "training.parquet", root)
    source_paths = [path, root / "models" / "production" / "manifest.json", root / "models" / "production" / "model.pt",
                    root / "reports" / "research.json", root / "metadata.json"]
    source_hashes = {str(item): _file_hash(_bounded(item, root)) for item in source_paths}
    dataset = pd.read_parquet(path)
    restored = _semantics(dataset, manifest)
    for key in ("data_version", "label_contract"):
        if dataset.attrs.get(key) is None:
            dataset.attrs[key] = manifest[key]
            restored.append(key)
        elif dataset.attrs[key] != manifest[key]:
            raise ValueError(f"frozen training metadata mismatch: {key}")
    if len(dataset) != research["training_rows"] or list(dataset.columns[-len(manifest["schema"]["columns"]):]) != manifest["schema"]["columns"]:
        raise ValueError("frozen training row count or feature order differs")
    read_hash = _fitting_hash(dataset, manifest)
    reconstructed_hash = _fitting_hash(dataset, manifest, _original_index(dataset, research["dataset_records"]))
    if manifest["dataset_sha256"] not in {read_hash, reconstructed_hash}:
        raise ValueError("frozen training content differs from the production fitting fingerprint")
    probe, probe_records = _probe(root, research, manifest)
    from my_strategy.services.czsc_research_ml import train_model, predict_model
    import torch
    if str(torch.__version__) != manifest["torch_version"] or torch.version.cuda != manifest["cuda_version"]:
        raise ValueError("same-environment reproduction requires the production Torch/CUDA versions")
    from my_strategy.core.device import select_torch_device
    if select_torch_device(torch, reproduction_device) != reproduction_device:
        raise RuntimeError("reproduction selected a different device; no fallback")
    settings = {"source_run_id": training_run_id, "source_manifest_sha256": manifest["manifest_sha256"],
                "source_checkpoint_sha256": manifest["checkpoint_sha256"], "training": manifest["hyperparameters"],
                "production": research["config"]["production"], "device": reproduction_device, "seed": manifest["seed"]}
    context = create_run_context(task="czsc-research-reproducibility", as_of_date=research["data_end"], config=settings,
                                 data_version=manifest["data_version"], seed=manifest["seed"], scope="train", source="cli",
                                 stocks=research["run_context"]["stocks"], start_date=manifest["train_signal_start"], end_date=research["data_end"])
    context.write_metadata({"config": settings, "status": "running", "source_run_id": training_run_id})
    try:
        output = context.subdir("models", "production_retrain")
        retrained = train_model(dataset, manifest["schema"]["columns"], output, manifest["train_end"], manifest["validation_start"],
                                manifest["validation_end"], device=reproduction_device, seed=manifest["seed"],
                                hidden_sizes=manifest["schema"]["hidden_sizes"], progress=progress, **manifest["hyperparameters"])
        if retrained.get("device") != reproduction_device:
            raise RuntimeError("retrained actual device differs from frozen device; no fallback")
        source_checkpoint = torch.load(root / "models" / "production" / "model.pt", map_location="cpu", weights_only=True)
        new_checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        tensors = _tensor_comparison(source_checkpoint["state_dict"], new_checkpoint["state_dict"])
        production_gpu = predict_model(probe, root / "models" / "production", device=reproduction_device, as_of=research["data_end"])
        replay_gpu = predict_model(probe, output, device=reproduction_device, as_of=research["data_end"])
        production_cpu = predict_model(probe, root / "models" / "production", device="cpu", as_of=research["data_end"])
        inference = _probability_comparison(production_gpu, replay_gpu)
        cpu_gpu = _probability_comparison(production_gpu, production_cpu)
        compared_fields = ["schema", "preprocess", "calibration", "training_loss", "sample_counts", "train_metrics", "validation_metrics"]
        compared_fields += ["device", "seed", "hyperparameters", "feature_version", "feature_schema_hash", "label_contract", "data_version"]
        compared_fields += [key for key in ("feature_profile", "actual_mps_training", "actual_cuda_training", "actual_gpu_training", "training_batches") if key in manifest]
        fields = {key: manifest[key] == retrained[key] for key in compared_fields}
        source_unchanged = all(_file_hash(Path(item)) == digest for item, digest in source_hashes.items())
        report = {"run_id": context.run_id, "source_run_id": training_run_id, "verified_at": local_now().isoformat(),
                  "source_manifest_sha256": manifest["manifest_sha256"], "source_checkpoint_sha256": manifest["checkpoint_sha256"],
                  "retrained_manifest_sha256": retrained["manifest_sha256"], "retrained_checkpoint_sha256": retrained["checkpoint_sha256"],
                  "training_parquet_sha256": source_hashes[str(path)], "production_config": settings, "restored_training_attrs": restored,
                  "training_fingerprint": {"source": manifest["dataset_sha256"], "read_parquet": read_hash,
                       "reconstructed_original_index": reconstructed_hash, "retrained": retrained["dataset_sha256"],
                       "index_changed_after_index_false_parquet": manifest["dataset_sha256"] != read_hash,
                       "index_is_model_input": False, "reconstruction": "record-order concat offsets plus chronological within-symbol row counts"},
                  "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device": manifest["device"],
                       "device_name": retrained["device_name"], "same_device_name": retrained["device_name"] == manifest["device_name"],
                       "cpu_threads": torch.get_num_threads()}, "manifest_fields_exact": fields,
                  "state_dict": tensors, "same_device_inference": inference, "cpu_gpu_inference": cpu_gpu,
                  "cpu_device_inference": cpu_gpu, "actual_training_batches": retrained.get("training_batches", 0),
                  "probe": {"rows": len(probe), "data_end": research["data_end"], "real_records": probe_records,
                       "sha256": hashlib.sha256(pd.util.hash_pandas_object(probe, index=False).to_numpy().tobytes()).hexdigest()},
                  "source_artifacts_modified": not source_unchanged, "source_file_hashes": source_hashes,
                  "limitations": ["Reproducibility is not evidence of predictive or trading improvement."]}
        report["passed"] = tensors["all_tensors_exact"] and all(fields.values()) and inference["passed"] and cpu_gpu["passed"] and report["environment"]["same_device_name"] and source_unchanged and report["actual_training_batches"] > 0
        report_path = context.subdir("reports") / "reproducibility.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        context.write_metadata({"config": settings, "status": "complete", "source_run_id": training_run_id, "passed": report["passed"]})
        return report
    except Exception as exc:
        context.write_metadata({"config": settings, "status": "failed", "source_run_id": training_run_id, "error": str(exc)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--device", help="Must match the frozen actual device; defaults to that recorded device")
    args = parser.parse_args()
    def progress(completed, total):
        print(json.dumps({"stage": "production_retrain", "completed": completed, "total": total}), flush=True)
    report = verify_run(args.training_run_id, device=args.device, progress=progress)
    print(json.dumps({"passed": report["passed"], "run_id": report["run_id"],
                      "report": str(ARTIFACT_RUNS_ROOT / report["run_id"] / "reports" / "reproducibility.json")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
