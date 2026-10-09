"""Frozen-policy fee/slippage sensitivity using new real Broker replays.

Models remain bound to their original base-cost targets. Changed execution
costs change cash, lots, actual cost stops and live policy features; this is not
a relabelling, retraining, parameter selection, or release.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions, _write_frame
from my_strategy.services.czsc_dual_models import DualPredictorSession, verified_bundle as dual_bundle
from my_strategy.services.czsc_dual_research import _raw, _mainboard, round_trip_statistics
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services.czsc_research_decisions import replay_decisions
from my_strategy.services.czsc_wyckoff_models import WyckoffPredictorSession, verified_bundle
from my_strategy.services.czsc_wyckoff_research import bounded_replay_frame, frozen_candidate_filter

ARMS = ("rules", "fusion_exit", "three_fusion_exit", "policy_fresh", "union_rules", "policy_union")
COST_FIELDS = ("commission", "stamp_tax", "min_commission", "slippage")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stressed_strategy(base, multiplier):
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("stress multiplier must be finite and at least one")
    value = copy.deepcopy(base)
    for key in COST_FIELDS:
        value["execution"][key] *= float(multiplier)
    return value


def verify_base_account(account, directory):
    """Verify real base replay against previously persisted immutable CSVs."""
    for key in ("ledger", "daily", "rejections"):
        path = Path(directory) / (key + ".csv")
        try:
            expected = pd.read_csv(path, float_precision="round_trip")
        except pd.errors.EmptyDataError:
            expected = pd.DataFrame()
        actual = pd.DataFrame(account[key])
        if expected.empty and actual.empty:
            continue
        expected = expected.drop(columns=["run_id"], errors="ignore")
        actual = actual.drop(columns=["run_id"], errors="ignore")
        if set(actual.columns) != set(expected.columns):
            raise ValueError("base replay columns differ: " + str(path))
        actual = actual[expected.columns]
        # CSV blanks and Python None have the same missing-value semantics.
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False, check_exact=True)


def rolling_exit(features, catalog):
    from my_strategy.services.czsc_wyckoff_exit import ExitPredictor, make_exit_policy
    predictors = {}
    def resolve(day):
        allowed = [entry for entry in catalog if entry["available_at"] <= day and entry["label_cutoff"] < day]
        if not allowed:
            return None
        binding = allowed[-1]
        key = binding["manifest_sha256"]
        if key not in predictors:
            predictors[key] = ExitPredictor(binding, device="cpu")
        return predictors[key]
    return make_exit_policy(features=features, predictor_resolver=resolve, apply_exit=True), predictors


def _worker(payload):
    (record, fold, bundle_path, probabilities, catalog, identities, multipliers, cash,
        destination, old_root, new_root, validate_base, base_settings, run_id) = payload
    from my_strategy.services.czsc_dual_exit import ExitPredictor as OldExit, make_exit_policy as old_exit_policy
    from my_strategy.services.czsc_wyckoff_exit import ExitPredictor, PolicyPredictor, make_exit_policy, make_entry_policy, make_union_decisions
    if file_hash(record["path"]) != record["sha256"]:
        raise ValueError("stress input snapshot changed: " + record["symbol"])
    full = pd.read_parquet(record["path"])
    full = full.loc[pd.to_datetime(full.date) <= pd.Timestamp(fold["test_end"])].copy()
    if not (pd.to_datetime(full.date) >= pd.Timestamp(fold["test_start"])).any():
        return {"symbol": record["symbol"], "cash_reason": "no_test_bars", "accounts": {}}
    frame, offset = bounded_replay_frame(full, fold["test_start"])
    values = {key: np.asarray(value)[offset:] for key, value in probabilities.items()}
    raw = _raw(frame)
    bundle = verified_bundle(bundle_path)
    previous = dual_bundle(bundle["baseline_bundle"]["path"])
    calendar = catalog["calendar"]
    base = replay_decisions(frame, raw, entry_policy="fresh", position_start=fold["test_start"], config=base_settings)["decisions"]
    old_decisions = replay_decisions(frame, raw, probabilities=values["dual_fusion"], thresholds=previous["probability_threshold"],
        apply_ml_mask=True, mode="rules", entry_policy="fresh", position_start=fold["test_start"], config=base_settings)["decisions"]
    threshold = bundle["thresholds"]["fusion"]["threshold"]
    new_values = values["fusion"] if threshold is not None else np.full(len(raw), np.nan)
    three = frozen_candidate_filter(base, new_values, threshold if threshold is not None else .55)
    union = make_union_decisions(frame, base, entry_policy="risk", config=base_settings, candidate_policy="fresh_wyckoff_union_v1")
    intent = {"rules": base, "fusion_exit": old_decisions, "three_fusion_exit": three,
        "policy_fresh": base, "union_rules": union, "policy_union": union}
    output = {}
    for multiplier in multipliers:
        stressed = stressed_strategy(base_settings, multiplier)
        factor = str(float(multiplier))
        output[factor] = {}
        for arm in ARMS:
            exit_callback, gate, predictors = None, None, []
            entry = "risk" if arm in {"union_rules", "policy_union"} else "fresh"
            if arm == "fusion_exit":
                head = previous.get("exit_head")
                if head and not head.get("unavailable"):
                    model = OldExit(head, device="cpu")
                    predictors.append(model)
                    exit_callback = old_exit_policy(features=frame, predictor=model, apply_exit=True)
            elif arm == "three_fusion_exit":
                head = bundle.get("exit_heads", {}).get("fresh")
                if head and not head.get("unavailable"):
                    model = ExitPredictor(head, device="cpu")
                    predictors.append(model)
                    exit_callback = make_exit_policy(features=frame, predictor=model, apply_exit=True)
            elif arm in {"policy_fresh", "policy_union"}:
                policy = "fresh" if arm == "policy_fresh" else "union"
                exit_callback, rolling = rolling_exit(frame, catalog[policy])
                head = bundle.get("policy_heads", {}).get(policy)
                if head and not head.get("unavailable"):
                    model = PolicyPredictor(head, device="cpu")
                    predictors.append(model)
                    gate = make_entry_policy(features=frame, predictor=model, apply_entry=True,
                        candidate_policy="fresh_v1" if policy == "fresh" else "fresh_wyckoff_union_v1",
                        continuation_policy_hash=identities[policy])
            account = execute_decisions(record["symbol"], raw, intent[arm], fold["test_start"], cash, stressed,
                run_id, market_dates=calendar, verified_sources=VERIFIED_SOURCES,
                entry_policy=entry, exit_policy=exit_callback, entry_gate=gate)
            if arm in {"policy_fresh", "policy_union"}:
                predictors.extend(rolling.values())
            if multiplier == 1. and validate_base:
                root = Path(old_root) if arm in {"rules", "fusion_exit"} else Path(new_root)
                verify_base_account(account, root / "evaluation" / fold["name"] / arm / record["symbol"].replace(".", "_"))
            target = Path(destination) / fold["name"] / factor / arm / record["symbol"].replace(".", "_")
            target.mkdir(parents=True, exist_ok=True)
            for key in ("ledger", "daily", "rejections", "entry_diagnostics"):
                _write_frame(pd.DataFrame(account.get(key, [])), target / (key + ".csv"))
            for key in ("exit_policy_diagnostics", "entry_gate_diagnostics"):
                if key in account:
                    (target / (key + ".json")).write_text(json.dumps(account[key], ensure_ascii=False, indent=2))
            diagnostics = [predictor.diagnostics for predictor in predictors]
            # Complete daily/state records remain persisted above. Aggregation
            # needs only equity by date; avoiding repeated Python state dicts
            # bounds memory for 3,197 accounts x 18 fixed-policy variants.
            compact_daily = [{"date": row["date"], "equity": row["equity"]} for row in account["daily"]]
            output[factor][arm] = {"ledger": account["ledger"], "daily": compact_daily, "rejections": account["rejections"],
                "metrics": account["metrics"], "cpu_rows": sum(row["rows"] for row in diagnostics),
                "cpu_batches": sum(row["batches"] for row in diagnostics), "base_ledger_equal": multiplier == 1. and validate_base}
    return {"symbol": record["symbol"], "accounts": output}


def _aggregate(items, cash, requested):
    from my_strategy.web_dashboard.tasks import aggregate_accounts
    accounts = [{**item, "trades": item["ledger"]} for item in items]
    summary = aggregate_accounts(accounts, requested * cash, requested)
    summary["metrics"].update(round_trip_statistics([item["ledger"] for item in items]))
    summary["cpu_rows"] = sum(item["cpu_rows"] for item in items)
    summary["cpu_batches"] = sum(item["cpu_batches"] for item in items)
    return summary


def run_stress(training_run_id, *, multipliers=(1., 1.5, 2.), cpu_workers=4, device="mps", symbols=None, validate_base=True):
    import torch
    torch.set_num_threads(1)
    root = artifact_run_dir(training_run_id, create=False)
    report_path = root / "reports/wyckoff-research.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete" or report.get("publication_allowed") is not False:
        raise ValueError("cost stress requires a complete immutable shadow training run")
    cfg = report["config"]
    strategy_path = root / "config/czsc_strategy.json"
    base_settings = json.loads(strategy_path.read_text())
    if stable_hash(base_settings) != stable_hash(strategy_config()):
        raise ValueError("current strategy differs from frozen training strategy; immutable replay unavailable")
    records = [record for record in report["dataset_records"] if _mainboard(record["symbol"])]
    if symbols is not None:
        requested = set(symbols)
        if not requested or requested - {record["symbol"] for record in records}:
            raise ValueError("stress symbols must belong to the new frozen mainboard projection")
        records = [record for record in records if record["symbol"] in requested]
    if not records or not multipliers or any(not np.isfinite(x) or x < 1 for x in multipliers):
        raise ValueError("invalid stress universe or multipliers")
    source_root = Path(report["baseline_reference"]["report_path"]).parent.parent
    frozen = {str(path): file_hash(path) for path in root.glob("models/**/*.pt")}
    frozen.update({str(path): file_hash(path) for path in root.glob("models/**/*.json")})
    frozen[str(report_path)] = file_hash(report_path)
    frozen[str(strategy_path)] = file_hash(strategy_path)
    frozen[report["baseline_reference"]["report_path"]] = file_hash(report["baseline_reference"]["report_path"])
    for path in (*source_root.glob("models/**/*.pt"), *source_root.glob("models/**/*.json")):
        frozen[str(path)] = file_hash(path)
    for record in records:
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError("projected input changed before stress")
        frozen[record["path"]] = record["sha256"]
    for record in report["source_snapshot"]["all_source_records"]:
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError("original data changed before stress")
        frozen[record["path"]] = record["sha256"]
    context = create_run_context(task="czsc-wyckoff-cost-stress", as_of_date=report["data_end"],
        config={"training_run_id": training_run_id, "base_strategy": base_settings, "multipliers": list(multipliers),
            "arms": list(ARMS), "policy": "fixed_models_thresholds_candidates_not_retrained", "publication_allowed": False},
        seed=cfg["seed"], scope="batch", stocks=[record["symbol"] for record in records], source="cli",
        data_version=report["data_version"], start_date=min(fold["test_start"] for fold in cfg["development_windows"]), end_date=report["data_end"])
    print("RUN_ID", context.run_id, flush=True)
    predictor, old_predictor = WyckoffPredictorSession(device=device), DualPredictorSession(device=device)
    summaries = []
    try:
        with ProcessPoolExecutor(max_workers=cpu_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for fold in cfg["development_windows"]:
                bundle_path = next(item["path"] for item in report["model_bundles"] if item["name"] == fold["name"])
                bundle = verified_bundle(bundle_path)
                gathered = {str(float(factor)): {arm: [] for arm in ARMS} for factor in multipliers}
                failed_cash = []
                for begin in range(0, len(records), 32):
                    tasks = []
                    for record in records[begin:begin + 32]:
                        if file_hash(record["path"]) != record["sha256"]:
                            raise ValueError("projected input changed before stress inference")
                        frame = pd.read_parquet(record["path"])
                        frame = frame.loc[pd.to_datetime(frame.date) <= pd.Timestamp(fold["test_end"])].copy()
                        mask = pd.to_datetime(frame.date) >= pd.Timestamp(bundle["available_at"])
                        probabilities = {name: np.full(len(frame), np.nan) for name in ("fusion", "dual_fusion")}
                        if mask.any():
                            probabilities["fusion"][mask.to_numpy()] = predictor.predict(frame.loc[mask].copy(), bundle_path, as_of=fold["test_end"])
                            probabilities["dual_fusion"][mask.to_numpy()] = old_predictor.predict(frame.loc[mask].copy(), bundle["baseline_bundle"]["path"], as_of=fold["test_end"])
                        catalog = {**report["frozen_exit_routers"]["catalog"], "calendar": report["market_dates"]}
                        tasks.append((record, fold, bundle_path, probabilities, catalog, report["frozen_exit_routers"]["identity"],
                            list(multipliers), cfg["initial_cash"], str(context.subdir("evaluation")), str(source_root), str(root),
                            validate_base, base_settings, context.run_id))
                    for result in pool.map(_worker, tasks, chunksize=1):
                        if not result["accounts"]:
                            failed_cash.append({"symbol": result["symbol"], "reason": result["cash_reason"]})
                        else:
                            for factor, accounts in result["accounts"].items():
                                for arm, account in accounts.items():
                                    gathered[factor][arm].append(account)
                    print("PROGRESS", fold["name"], min(begin + 32, len(records)), len(records), flush=True)
                variants = {}
                for factor, arms in gathered.items():
                    variants[factor] = {}
                    for arm, accounts in arms.items():
                        summary = _aggregate(accounts, cfg["initial_cash"], len(records))
                        _write_frame(pd.DataFrame(summary["daily"]), context.subdir("evaluation", fold["name"], factor, arm) / "aggregate_daily.csv")
                        variants[factor][arm] = {"metrics": summary["metrics"], "accounts": len(accounts),
                            "cpu_rows": summary["cpu_rows"], "cpu_batches": summary["cpu_batches"]}
                summaries.append({"fold": fold, "variants": variants, "unavailable_accounts_cash_retained": failed_cash})
        for path, expected in frozen.items():
            if file_hash(path) != expected:
                raise ValueError("frozen source/model changed during stress: " + path)
        output = {"status": "complete", "run_id": context.run_id, "training_run_id": training_run_id,
            "data_end": report["data_end"], "source_training_report_sha256": file_hash(report_path),
            "mainboard_accounts": len(records), "multipliers": list(multipliers), "arms": list(ARMS),
            "cost_fields_multiplied": list(COST_FIELDS), "base_strategy": base_settings,
            "source_strategy_path": str(strategy_path), "source_strategy_sha256": frozen[str(strategy_path)],
            "stressed_strategies": {str(float(x)): stressed_strategy(base_settings, x) for x in multipliers},
            "policy": "fixed_models_thresholds_candidates_actual_broker_cash_lots_stops_replayed",
            "model_target_scope": "original_base_cost_target_no_retraining_changed_cost_sensitivity_only",
            "publication_allowed": False, "qualification": "development_cost_sensitivity_not_independent_profit_qualification",
            "base_ledger_validation": "exact_all_account_ledger_daily_rejections_except_run_id" if validate_base and 1. in multipliers else "not_requested",
            "base_accounts_validated": sum(v["accounts"] for result in summaries for factor, variants in result["variants"].items()
                if validate_base and factor == "1.0" for v in variants.values()),
            "source_model_files_unchanged": len(frozen), "frozen_file_sha256": frozen, "evaluation": summaries,
            "compute_info": {"new_training_models": 0, "entry_inference": predictor.diagnostics,
                "old_dual_inference": old_predictor.diagnostics,
                "exit_and_policy_cpu_rows": sum(v["cpu_rows"] for result in summaries for variants in result["variants"].values() for v in variants.values()),
                "exit_and_policy_cpu_batches": sum(v["cpu_batches"] for result in summaries for variants in result["variants"].values() for v in variants.values())}}
        (context.subdir("reports") / "wyckoff-cost-stress.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
        context.write_metadata({"status": "complete", "training_run_id": training_run_id, "coverage": {"requested": len(records), "failed": 0}, "compute_info": output["compute_info"]})
        print("COMPLETE", context.run_id, flush=True)
        return output
    except Exception as exc:
        context.write_metadata({"status": "failed", "error": str(exc), "training_run_id": training_run_id})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--multipliers", type=float, nargs="+", default=[1., 1.5, 2.])
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--skip-base-validation", action="store_true")
    args = parser.parse_args()
    run_stress(args.training_run_id, multipliers=args.multipliers, cpu_workers=args.cpu_workers, device=args.device,
        symbols=args.symbols, validate_base=not args.skip_base_validation)


if __name__ == "__main__":
    main()
