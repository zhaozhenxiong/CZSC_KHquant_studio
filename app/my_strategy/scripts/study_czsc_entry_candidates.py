"""Development-only candidate-population experiment on frozen research inputs.

Known historical test windows are diagnostics, never independent certification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


def study(parent_run_id: str, study_run_id: str, *, device="cuda:1", workers=4):
    from my_strategy.services.czsc_research import run_path, _file_hash, _save, evaluate_fold
    from my_strategy.services.czsc_research_features import FEATURE_COLUMNS, FEATURE_VERSION, FEATURE_SCHEMA_HASH
    from my_strategy.services.czsc_research_ml import train_model, PredictorSession
    from my_strategy.core.run_context import RunContext, stable_hash
    from my_strategy.core.tz import local_now
    parent, root = run_path(parent_run_id), run_path(study_run_id)
    protocol_path = root / "reports/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("parent_run_id") != parent_run_id or protocol.get("stage") != "development_only":
        raise ValueError("study requires a frozen development-only protocol")
    if (root / "reports/study.json").exists():
        raise FileExistsError("use an isolated study run")
    training = json.loads((parent / "reports/research.json").read_text(encoding="utf-8"))
    config = training["config"]
    records = training["dataset_records"]
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    fields = RunContext.__dataclass_fields__
    context = RunContext(**{key: metadata[key] for key in fields if key in metadata})
    columns = ["symbol", "date", "label", "label_end", "label_available", "input_eligible", "rule_buy", "net_return", *FEATURE_COLUMNS]
    def read(record):
        path = Path(record["path"])
        if _file_hash(path) != record["sha256"]:
            raise ValueError("frozen feature hash mismatch")
        mainboard = record["symbol"].startswith("60") and record["symbol"].endswith(".SH") or record["symbol"].startswith("00") and record["symbol"].endswith(".SZ")
        if not mainboard:
            return None
        frame = pd.read_parquet(path, columns=columns)
        if frame.attrs.get("schema_hash") != FEATURE_SCHEMA_HASH or frame.attrs.get("feature_version") != FEATURE_VERSION:
            raise ValueError("candidate feature semantics mismatch")
        frame = frame[(frame.date >= config["feature_start"]) & frame.input_eligible & frame.rule_buy & frame.label_available & frame.label.notna()].copy()
        frame.attrs = {}
        return frame
    frames = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, frame in enumerate(pool.map(read, records), 1):
            if frame is not None and len(frame):
                frames.append(frame)
            if index % 250 == 0 or index == len(records):
                print(f"candidate preparation {index}/{len(records)}", flush=True)
    data = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])
    parent_manifest = json.loads((parent / "models/production/manifest.json").read_text(encoding="utf-8"))
    data.attrs.update(feature_version=FEATURE_VERSION, schema_hash=FEATURE_SCHEMA_HASH,
                      data_version=stable_hash({"parent": training["data_version"], "population": "eligible_rule_buy"}),
                      label_contract=parent_manifest["label_contract"])
    data.to_parquet(context.subdir("dataset") / "entry_candidates.parquet", index=False)
    output = {"parent_run_id": parent_run_id, "study_run_id": study_run_id,
              "status": "development_only", "production_eligible": False,
              "independent_certification": False, "known_test_windows": "previously inspected; diagnostic only",
              "requested_stock_count": training["coverage"]["requested"], "hashed_records": len(records),
              "candidate_rows": len(data), "feature_version": FEATURE_VERSION,
              "parent_research_sha256": _file_hash(parent / "reports/research.json"),
              "protocol_sha256": _file_hash(protocol_path), "checkpoints": [], "evaluation": []}
    predictor = PredictorSession(device=device)
    for fold in [*config["folds"], config["holdout"], {**config["production"], "name": "production"}]:
        name = fold["name"]
        model_dir = context.subdir("models") / name
        started = time.perf_counter()
        manifest = train_model(data, FEATURE_COLUMNS, model_dir, fold["train_end"], fold["validation_start"],
                               fold["validation_end"], device=device, seed=protocol["seed"], epochs=protocol["epochs"],
                               progress=lambda n, total: print(f"training {name} {n}/{total}", flush=True))
        validation = data[(data.date >= fold["validation_start"]) & (data.label_end <= fold["validation_end"])].copy()
        probability = predictor.predict_retrospective(validation, model_dir, as_of=fold["validation_end"])
        grid = []
        for threshold in protocol["threshold_grid"]:
            chosen = validation[np.isfinite(probability) & (probability >= threshold)]
            grid.append({"threshold": threshold, "rows": len(chosen), "retained_fraction": len(chosen) / len(validation),
                         "mean_fixed_horizon_net_return": float(chosen.net_return.mean()) if len(chosen) else None,
                         "positive_rate": float(chosen.label.mean()) if len(chosen) else None})
        # The grid is a diagnostic, not an optimized ledger policy or promotion.
        checkpoint = {"name": name, "manifest": manifest, "validation_threshold_grid": grid,
                      "baseline_validation_net_return": float(validation.net_return.mean()),
                      "training_seconds": time.perf_counter() - started, "threshold_used_for_development_ledger": config["probability_threshold"]}
        output["checkpoints"].append(checkpoint)
        _save(context.subdir("reports") / (name + "-candidate.json"), checkpoint)
        _save(context.subdir("reports") / "study-progress.json", output)
        if "test_start" in fold:
            from my_strategy.services.czsc_research import _calendar
            dates, _ = _calendar(training["calendar"]["run_id"])
            evaluation = evaluate_fold(records, model_dir, fold, context, config, device=device,
                                       requested_symbols=training["run_context"]["stocks"], market_dates=dates, cpu_workers=workers,
                                       progress=lambda stage,n,total,failed: print(f"{stage} {n}/{total} failed={failed}", flush=True) if n % 250 == 0 or n == total else None)
            output["evaluation"].append(evaluation)
            _save(context.subdir("reports") / "study-progress.json", output)
    output.update(completed_at=local_now().isoformat(), compute_info=predictor.diagnostics)
    _save(context.subdir("reports") / "study.json", output)
    context.write_metadata({"status": "development_study_complete", "protocol": protocol,
                            "candidate_rows": len(data), "production_eligible": False,
                            "training_completed_at": output["completed_at"]})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run-id", required=True)
    parser.add_argument("--study-run-id", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = study(args.parent_run_id, args.study_run_id, device=args.device, workers=args.workers)
    print(json.dumps({"run_id": result["study_run_id"], "candidate_rows": result["candidate_rows"],
                      "production_eligible": False, "windows": len(result["evaluation"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
