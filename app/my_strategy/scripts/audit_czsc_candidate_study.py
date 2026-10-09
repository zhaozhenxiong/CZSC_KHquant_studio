"""Reconcile completed development candidate studies without training or promotion.

Only reports/development_execution_audit.json is written. Passing this audit
establishes accounting and input integrity, never independent certification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now
from my_strategy.scripts.audit_czsc_research import (
    MODES, _account, _bounded, _close, _csv, _dataset, _error, _file_hash,
    _model, _read, _run,
)


def _mainboard(symbol: str) -> bool:
    return symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ")


def _candidate_record(record, parent, schema, calendar, config, columns):
    audit = _dataset(record, parent, schema, calendar, int(config["horizon"]))
    candidates = None
    if _mainboard(record["symbol"]):
        try:
            frame = pd.read_parquet(_bounded(record["path"], parent), columns=columns)
            candidates = frame.loc[(frame.date >= config["feature_start"]) & frame.input_eligible
                                   & frame.rule_buy & frame.label_available & frame.label.notna()].copy()
            candidates.attrs = {}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            audit["errors"].append(_error("candidate_parent_read", symbol=record["symbol"], error=str(exc)))
    return audit, candidates


def _evaluation(root, evaluation, frozen, initial, following, pool, errors):
    fold = evaluation["fold"]
    unavailable = [item["symbol"] for item in evaluation["unavailable_accounts_cash_retained"]]
    if (evaluation["frozen_mainboard_accounts"] != len(frozen) or len(unavailable) != len(set(unavailable))
            or not set(unavailable).issubset(frozen)):
        errors.append(_error("fold_frozen_capital_denominator", fold=fold["name"]))
    per_mode = {}
    fold_root = _run(_bounded(root / "evaluation", root), fold["name"])
    for mode in MODES:
        location = _bounded(fold_root / mode, root)
        present = [symbol for symbol in frozen if _bounded(location / symbol.replace(".", "_") / "daily.csv", root).is_file()]
        actual = [path.name.replace("_", ".") for path in location.iterdir() if path.is_dir()] if location.is_dir() else []
        if set(frozen) - set(present) != set(unavailable):
            errors.append(_error("fold_missing_accounts_cash_record", fold=fold["name"], mode=mode,
                                 missing=sorted(set(frozen) - set(present)), recorded=sorted(unavailable)))
        if set(actual) != set(present):
            errors.append(_error("unexpected_or_incomplete_account_files", fold=fold["name"], mode=mode))
        accounts = list(pool.map(lambda symbol: _account(_bounded(location / symbol.replace(".", "_"), root),
                                                         symbol, initial, following), present))
        for item in accounts:
            errors.extend({**error, "fold": fold["name"], "mode": mode} for error in item.pop("errors"))
        valid = [item for item in accounts if "_equity" in item]
        summary = evaluation["variants"][mode]
        if len(valid) != summary["accounts"]:
            errors.append(_error("fold_account_count", fold=fold["name"], mode=mode))
        try:
            aggregate = _csv(_bounded(location / "aggregate_daily.csv", root))
            if not valid or aggregate.empty:
                raise ValueError("missing nonempty aggregate or account daily series")
            wide = pd.concat([item["_equity"] for item in valid], axis=1).sort_index().ffill().fillna(initial)
            expected = wide.sum(axis=1) + (len(frozen) - len(valid)) * initial
            dates = pd.to_datetime(aggregate["date"]).dt.strftime("%Y-%m-%d")
            if (dates.duplicated().any() or list(dates) != list(expected.index)
                    or not _close(aggregate["equity"], expected)):
                errors.append(_error("aggregate_daily_includes_failed_cash", fold=fold["name"], mode=mode))
            if expected.index[0] < fold["test_start"] or expected.index[-1] > fold["test_end"]:
                errors.append(_error("aggregate_test_boundaries", fold=fold["name"], mode=mode))
            final, metrics = float(expected.iloc[-1]), summary["metrics"]
            total_return = final / (len(frozen) * initial) - 1
            if (not _close(metrics["final_equity"], final)
                    or not np.isclose(metrics["total_return"], total_return, rtol=1e-8, atol=1e-10)
                    or "net_return" in metrics and not np.isclose(metrics["net_return"], total_return, rtol=1e-8, atol=1e-10)):
                errors.append(_error("aggregate_final_summary", fold=fold["name"], mode=mode))
            if (not _close(metrics["fees"], sum(item["fees"] for item in valid))
                    or metrics["trade_count"] != sum(item["ledger_rows"] for item in valid)):
                errors.append(_error("aggregate_trade_count_fees", fold=fold["name"], mode=mode))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(_error("aggregate_read_or_reconcile", fold=fold["name"], mode=mode, error=str(exc)))
        per_mode[mode] = {"accounts": len(valid), "daily_files": len(present),
                          "daily_rows": sum(item.get("daily_rows", 0) for item in accounts),
                          "ledger_rows": sum(item.get("ledger_rows", 0) for item in accounts),
                          "rejection_rows": sum(item.get("rejection_rows", 0) for item in accounts),
                          "fees": sum(item.get("fees", 0) for item in accounts),
                          "failed_cash_accounts": len(unavailable), "failed_cash_budget": len(unavailable) * initial,
                          "frozen_initial_budget": len(frozen) * initial,
                          "open_positions": sum(item.get("final_shares", 0) > 0 for item in accounts)}
    return {"name": fold["name"], "modes": per_mode}


def audit_study(study_run_id: str, *, runs_root: Path | None = None, workers: int = 8) -> dict:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be within 1..8")
    base = (runs_root or ARTIFACT_RUNS_ROOT).resolve()
    root = _run(base, study_run_id)
    study = _read(_bounded(root / "reports/study.json", root))
    protocol_path = _bounded(root / "reports/protocol.json", root)
    protocol, metadata = _read(protocol_path), _read(_bounded(root / "metadata.json", root))
    parent = _run(base, study["parent_run_id"])
    parent_path = _bounded(parent / "reports/research.json", parent)
    training, parent_metadata = _read(parent_path), _read(_bounded(parent / "metadata.json", parent))
    config, schema = training["config"], training["model_manifest"]
    errors = []
    if (study.get("study_run_id") != study_run_id or study.get("status") != "development_only"
            or study.get("production_eligible") is not False or study.get("independent_certification") is not False
            or not study.get("known_test_windows") or metadata.get("status") != "development_study_complete"):
        errors.append(_error("completed_development_only_identity"))
    if (protocol.get("stage") != "development_only" or protocol.get("parent_run_id") != parent.name
            or protocol.get("known_test_windows_are_development_diagnostics") is not True
            or protocol.get("unchanged_label_horizon") != config["horizon"]
            or metadata.get("config_hash") != stable_hash(protocol)
            or metadata.get("protocol") != protocol):
        errors.append(_error("frozen_development_protocol"))
    if _file_hash(protocol_path) != study["protocol_sha256"] or _file_hash(parent_path) != study["parent_research_sha256"]:
        errors.append(_error("parent_research_or_protocol_sha256"))
    if training.get("run_id") != parent.name or parent_metadata.get("status") != "complete":
        errors.append(_error("completed_parent_identity"))
    if parent_metadata.get("config_hash") != stable_hash(config) or parent_metadata.get("research_config") != config:
        errors.append(_error("frozen_parent_configuration"))
    stocks = training["run_context"]["stocks"]
    records, failures = training["dataset_records"], training.get("failures", [])
    covered = [item["symbol"] for item in [*records, *failures]]
    if (len(stocks) != len(set(stocks)) or parent_metadata.get("stocks") != stocks
            or set(covered) != set(stocks) or len(covered) != len(set(covered))
            or training["coverage"] != {"requested": len(stocks), "success": len(records), "failed": len(failures)}
            or study["requested_stock_count"] != len(stocks) or study["hashed_records"] != len(records)):
        errors.append(_error("frozen_parent_dataset_coverage"))
    frozen = [symbol for symbol in stocks if _mainboard(symbol)]
    calendar_value = _read(_bounded(training["calendar"]["path"], base))
    calendar = calendar_value["dates"]
    if (not calendar_value.get("verified") or not calendar_value.get("source")
            or calendar != sorted(set(calendar)) or stable_hash(calendar_value) != training["calendar"]["hash"]):
        errors.append(_error("verified_frozen_calendar"))
    columns = ["symbol", "date", "label", "label_end", "label_available", "input_eligible", "rule_buy", "net_return", *schema["schema"]["columns"]]
    report = {"study_run_id": study_run_id, "parent_run_id": parent.name, "audited_at": local_now().isoformat(),
              "read_only_inputs": True, "diagnostic_only": True, "production_eligible": False,
              "independent_certification": False, "known_test_windows": study["known_test_windows"],
              "limitations": ["Passing reconciles frozen inputs and arithmetic; inspected historical tests cannot certify production.",
                              "Feasible fixed-horizon labels are a selected population; ledger rejects and unexited positions remain in evaluation."],
              "model_payload_check": "hash_only_without_torch_deserialization", "workers": workers,
              "numeric_tolerances": {"money_absolute": 0.01, "money_relative": 1e-9,
                                     "ratio_absolute": 1e-10, "ratio_relative": 1e-8, "inventory": "exact_round_lots"},
              "frozen_stocks": len(stocks), "frozen_mainboard_accounts": len(frozen),
              "parent_research_sha256": _file_hash(parent_path), "protocol_sha256": _file_hash(protocol_path),
              "errors": errors}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="czsc-candidate-read-audit") as pool:
        inspected = list(pool.map(lambda record: _candidate_record(record, parent, schema, calendar, config, columns), records))
        for item, _ in inspected:
            errors.extend(item["errors"])
        selected = [frame for _, frame in inspected if frame is not None and not frame.empty]
        expected = pd.concat(selected, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True) if selected else pd.DataFrame(columns=columns)
        candidate_path = _bounded(root / "dataset/entry_candidates.parquet", root)
        data = pd.read_parquet(candidate_path)
        try:
            pd.testing.assert_frame_equal(data.reset_index(drop=True), expected, check_dtype=False, check_exact=True)
        except (AssertionError, ValueError) as exc:
            errors.append(_error("candidate_population_matches_frozen_rule_buy", error=str(exc)[:500]))
        if (len(data) != study["candidate_rows"] or len(data) != metadata.get("candidate_rows")
                or not (data.rule_buy & data.input_eligible & data.label_available).all()
                or not data.label.isin([0, 1]).all() or not (data.label_end > data.date).all()
                or (data.label_end > training["data_end"]).any()
                or data[["symbol", "date"]].duplicated().any()):
            errors.append(_error("candidate_eligibility_labels_or_counts"))
        if (data.attrs.get("feature_version") != schema["feature_version"] or data.attrs.get("schema_hash") != schema["feature_schema_hash"]
                or data.attrs.get("label_contract") != schema["label_contract"]
                or data.attrs.get("data_version") != stable_hash({"parent": training["data_version"], "population": "eligible_rule_buy"})):
            errors.append(_error("candidate_feature_label_or_data_semantics"))
        report["dataset"] = {"parent_records": len(inspected), "parent_bars": sum(item["bars"] for item, _ in inspected),
                             "candidate_rows": len(data), "candidate_stocks": int(data.symbol.nunique()), "candidate_sha256": _file_hash(candidate_path)}
        folds = [*config["folds"], *([config["holdout"]] if "holdout" in config else [])]
        configured = {fold["name"]: fold for fold in [*folds, {**config["production"], "name": "production"}]}
        embedded = {item["name"]: item for item in study["checkpoints"]}
        paths = sorted((root / "models").glob("*/manifest.json"))
        if set(embedded) != set(configured) or len(embedded) != len(study["checkpoints"]) or {path.parent.name for path in paths} != set(configured):
            errors.append(_error("complete_configured_models", expected=sorted(configured)))
        models = list(pool.map(lambda path: _model(path, root, schema), paths))
        for item in models:
            errors.extend(item.pop("errors"))
        for path in paths:
            name, manifest = path.parent.name, _read(_bounded(path, root))
            checkpoint, fold = embedded.get(name), configured.get(name)
            if not checkpoint or not fold or manifest != checkpoint["manifest"]:
                errors.append(_error("frozen_embedded_model_manifest", model=name))
                continue
            train = (data.date <= fold["train_end"]) & (data.label_end <= fold["train_end"]) & (data.label_end < fold["validation_start"])
            validation = (data.date >= fold["validation_start"]) & (data.date <= fold["validation_end"]) & (data.label_end <= fold["validation_end"])
            if (manifest["sample_counts"]["train"] != int(train.sum()) or manifest["sample_counts"]["validation"] != int(validation.sum())
                    or manifest["preprocess"].get("fit_scope") != "purged_training_only"
                    or manifest["seed"] != protocol["seed"] or manifest["hyperparameters"]["epochs"] != protocol["epochs"]
                    or manifest["label_contract"] != schema["label_contract"]
                    or manifest["data_version"] != data.attrs["data_version"]):
                errors.append(_error("candidate_purged_model_samples_or_contract", model=name))
            if (checkpoint["threshold_used_for_development_ledger"] != config["probability_threshold"]
                    or [row["threshold"] for row in checkpoint["validation_threshold_grid"]] != protocol["threshold_grid"]
                    or any(row["rows"] < 0 or row["rows"] > int(validation.sum()) for row in checkpoint["validation_threshold_grid"])):
                errors.append(_error("validation_only_threshold_diagnostic", model=name))
        report["models"] = models
        if [item["fold"] for item in study["evaluation"]] != folds:
            errors.append(_error("all_configured_development_evaluations"))
        following = dict(zip(calendar[:-1], calendar[1:]))
        report["evaluation"] = [_evaluation(root, evaluation, frozen, float(config["initial_cash"]), following, pool, errors)
                                for evaluation in study["evaluation"]]
    report.update(passed=not errors, error_count=len(errors))
    output = _bounded(root / "reports/development_execution_audit.json", root)
    temporary = _bounded(output.with_suffix(".json.tmp"), root)
    temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-run-id", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--runs-root", type=Path, help="Isolated audit fixture root")
    args = parser.parse_args()
    report = audit_study(args.study_run_id, runs_root=args.runs_root, workers=args.workers)
    print(json.dumps({"study_run_id": args.study_run_id, "passed": report["passed"], "error_count": report["error_count"],
                      "diagnostic_only": True, "independent_certification": False, "candidate_rows": report["dataset"]["candidate_rows"],
                      "report": str((args.runs_root or ARTIFACT_RUNS_ROOT) / args.study_run_id / "reports/development_execution_audit.json")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
