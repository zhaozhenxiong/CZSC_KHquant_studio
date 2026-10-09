"""Independently reconcile actual-exit labels and complete development ledgers.

Reads frozen features/models/accounts; writes only this study's audit report.
Passing establishes arithmetic and lineage, never production certification.
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
from my_strategy.scripts.audit_czsc_research import _account, _close, _csv, _file_hash, _model, _read, _run
from my_strategy.services.czsc_research_strategy_labels import LABEL_VERSION
from my_strategy.services.czsc_research_features import FEATURE_COLUMNS

VARIANTS = ("legacy", "fresh", "risk", "fresh_ml")


def _evaluated_account(location, old, variant, name, symbol, initial, following):
    account = _account(location / symbol.replace(".", "_"), symbol, initial, following)
    if variant == "legacy":
        for table in ("daily", "ledger", "rejections"):
            current = _csv(location / symbol.replace(".", "_") / (table + ".csv"))
            previous = _csv(old / "evaluation" / name / "czsc_price_volume" / symbol.replace(".", "_") / (table + ".csv"))
            current, previous = (value.drop(columns=["run_id"], errors="ignore") for value in (current, previous))
            try:
                pd.testing.assert_frame_equal(current, previous, check_dtype=False, check_exact=False, rtol=1e-9, atol=.00000001)
            except AssertionError as exc:
                account["errors"].append({"rule": "legacy_account_preserved", "symbol": symbol, "table": table, "error": str(exc)[:200]})
    return account


def _rounds(ledger):
    result, opened, inventory, paid, received, fees = [], None, 0, 0., 0., 0.
    for trade in ledger:
        if trade["action"] == "BUY":
            if inventory:
                raise ValueError("unexpected position addition")
            opened, inventory = trade, int(trade["shares"])
            paid, received, fees = -float(trade["cash_flow"]), 0., float(trade["fee"])
        else:
            inventory -= int(trade["shares"])
            received += float(trade["cash_flow"])
            fees += float(trade["fee"])
            if inventory == 0:
                result.append({"date": opened["signal_date"], "entry_date": opened["date"],
                    "exit_date": trade["date"], "label_end": trade["date"], "entry_shares": int(opened["shares"]),
                    "label": float(received > paid), "net_pnl": received - paid,
                    "invested_cash": paid, "invested_net_return": received / paid - 1, "fees": fees})
                opened = None
    return result


def _labels(root, record, initial, contract):
    symbol, errors = record["symbol"], []
    mainboard = symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ")
    if _file_hash(Path(record["path"])) != record["sha256"]:
        errors.append({"rule": "frozen_feature_hash", "symbol": symbol})
    if not mainboard:
        return {"symbol": symbol, "errors": errors, "sample": None, "mature_labels": 0}
    try:
        labels = pd.read_parquet(root / "labels" / (symbol.replace(".", "_") + ".parquet"))
        execution = _read(root / "label_audit" / symbol.replace(".", "_") / "execution.json")
        frozen = pd.read_parquet(record["path"], columns=["symbol", "date", "input_eligible", "close", *FEATURE_COLUMNS])
        available = labels.label_available.astype(bool)
        if (len(labels) != record["bars"] or labels.attrs.get("label_version") != LABEL_VERSION
                or labels.attrs.get("label_contract") != contract or labels.attrs.get("entry_policy") != "fresh"
                or not np.array_equal(available, labels.label.notna())
                or not list(labels.date) == list(frozen.date)
                or not labels.loc[available, "label"].isin([0, 1]).all()
                or not (labels.loc[available, "label_end"] > labels.loc[available, "date"]).all()
                or (labels.loc[available, "label_end"] > record["data_end"]).any()):
            errors.append({"rule": "actual_label_identity_contract_maturity", "symbol": symbol})
        daily = pd.DataFrame(execution["daily"])
        flows, inventory_changes, cash, inventory = {}, {}, initial, 0
        for trade in execution["ledger"]:
            quantity = int(trade["shares"])
            change = quantity if trade["action"] == "BUY" else -quantity
            expected_flow = -(quantity * trade["price"] + trade["fee"]) if change > 0 else quantity * trade["price"] - trade["fee"]
            cash += trade["cash_flow"]
            if (trade["position_before"] != inventory or trade["position_after"] != inventory + change
                    or not _close(trade["cash_flow"], expected_flow) or not _close(trade["cash_after"], cash)):
                errors.append({"rule": "label_execution_cash_inventory_chain", "symbol": symbol, "date": trade["date"]})
            inventory += change
            flows[trade["date"]] = flows.get(trade["date"], 0.) + trade["cash_flow"]
            inventory_changes[trade["date"]] = inventory_changes.get(trade["date"], 0) + change
        if (list(daily.date) != list(labels.date) or not _close(daily.close, frozen.close)
                or not _close(daily.cash, initial + daily.date.map(flows).fillna(0).cumsum())
                or not np.array_equal(daily.shares, daily.date.map(inventory_changes).fillna(0).cumsum())
                or not _close(daily.equity, daily.cash + daily.shares * daily.close)):
            errors.append({"rule": "label_full_prefix_account_reconciliation", "symbol": symbol})
        rounds = _rounds(execution["ledger"])
        actual = labels.loc[available].set_index("date")
        if len(rounds) != len(actual):
            errors.append({"rule": "actual_label_roundtrip_count", "symbol": symbol})
        for trip in rounds:
            row = actual.loc[trip["date"]]
            for key, value in {**trip, "net_return": trip["net_pnl"] / initial}.items():
                if key == "date":
                    continue
                if isinstance(value, (int, float)):
                    agrees = np.isclose(float(row[key]), value, rtol=1e-9, atol=1e-10)
                else:
                    agrees = row[key] == value
                if not agrees:
                    errors.append({"rule": "actual_label_ledger_value", "symbol": symbol, "date": trip["date"], "column": key})
        sample = frozen.loc[available & frozen.input_eligible.astype(bool), ["symbol", "date", "input_eligible", *FEATURE_COLUMNS]].copy()
        for key in ("label", "label_end", "label_available", "net_return", "invested_net_return"):
            sample[key] = labels.loc[sample.index, key]
        sample.attrs = {}
        return {"symbol": symbol, "errors": errors, "sample": sample, "mature_labels": len(actual)}
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return {"symbol": symbol, "errors": [*errors, {"rule": "label_audit_read", "symbol": symbol, "error": str(exc)}], "sample": None, "mature_labels": 0}


def audit_study(run_id: str, *, legacy_run_id: str, workers: int = 8) -> dict:
    root, old = _run(ARTIFACT_RUNS_ROOT, run_id), _run(ARTIFACT_RUNS_ROOT, legacy_run_id)
    study, protocol, metadata = (_read(root / name) for name in ("reports/study.json", "reports/protocol.json", "metadata.json"))
    parent = _run(ARTIFACT_RUNS_ROOT, protocol["parent_run_id"])
    training = _read(parent / "reports/research.json")
    records, config = training["dataset_records"], training["config"]
    data = pd.read_parquet(root / "dataset/actual_entry_labels.parquet")
    contract, initial = data.attrs["label_contract"], float(config["initial_cash"])
    calendar = _read(Path(training["calendar"]["path"]))["dates"]
    following = dict(zip(calendar[:-1], calendar[1:]))
    frozen = [x for x in training["run_context"]["stocks"] if x.startswith("60") and x.endswith(".SH") or x.startswith("00") and x.endswith(".SZ")]
    errors, evaluations = [], []
    if (study.get("status") != "development_complete" or study.get("production_eligible") is not False
            or not protocol.get("known_windows") or protocol.get("production_eligible") is not False
            or protocol.get("label_version") != LABEL_VERSION or metadata.get("config_hash") != stable_hash(protocol)
            or _file_hash(parent / "reports/research.json") != protocol["parent_sha256"]):
        errors.append({"rule": "frozen_development_protocol"})
    evidence = {x["symbol"]: x for x in study["coverage"]}
    if len(evidence) != len(records) or set(evidence) != {x["symbol"] for x in records} or any(
            not evidence[x["symbol"]].get("raw_verified") or evidence[x["symbol"]]["sha256"] != x["sha256"] for x in records):
        errors.append({"rule": "all_input_records_verified"})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        print(f"audit frozen inputs + actual labels {len(records)}", flush=True)
        inspected = list(pool.map(lambda record: _labels(root, record, initial, contract), records))
        for item in inspected:
            errors.extend(item["errors"])
        print(f"audited actual labels; errors={len(errors)}", flush=True)
        expected = pd.concat([x["sample"] for x in inspected if x["sample"] is not None], ignore_index=True)
        expected = expected.loc[expected.date >= config["feature_start"]].sort_values(["date", "symbol"]).reset_index(drop=True)
        columns = list(expected)
        try:
            pd.testing.assert_frame_equal(data[columns].reset_index(drop=True), expected, check_dtype=False, check_exact=True)
        except AssertionError as exc:
            errors.append({"rule": "training_population_matches_actual_strategy_labels", "error": str(exc)[:500]})
        configured = [*config["folds"], config["holdout"]]
        if [x["fold"] for x in study["evaluation"]] != configured:
            errors.append({"rule": "all_configured_windows"})
        models = []
        for item in study["checkpoints"]:
            manifest = item["manifest"]
            audit = _model(root / "models" / item["name"] / "manifest.json", root, training["model_manifest"])
            errors.extend(audit.pop("errors"))
            models.append(audit)
            train = (data.date <= manifest["train_end"]) & (data.label_end <= manifest["train_end"]) & (data.label_end < manifest["validation_start"])
            validation = (data.date >= manifest["validation_start"]) & (data.date <= manifest["validation_end"]) & (data.label_end <= manifest["validation_end"])
            if (manifest.get("label_version") != LABEL_VERSION or manifest.get("label_contract") != contract
                    or manifest["sample_counts"]["train"] != int(train.sum())
                    or manifest["sample_counts"]["validation"] != int(validation.sum())):
                errors.append({"rule": "actual_label_model_purged_samples", "model": item["name"]})
        for fold in study["evaluation"]:
            name, absent = fold["fold"]["name"], {x["symbol"] for x in fold["unavailable_accounts_cash_retained"]}
            fold_result = {"name": name, "variants": {}}
            for variant in VARIANTS:
                print(f"audit account ledgers {name} {variant}", flush=True)
                location = root / "evaluation" / name / variant
                present = {p.name.replace("_", ".") for p in location.iterdir() if p.is_dir()}
                if present | absent != set(frozen) or present & absent:
                    errors.append({"rule": "all_frozen_accounts_or_explicit_cash", "fold": name, "variant": variant})
                accounts = list(pool.map(lambda symbol: _evaluated_account(location, old, variant, name, symbol, initial, following), sorted(present)))
                for account in accounts:
                    errors.extend({**item, "fold": name, "variant": variant} for item in account.pop("errors"))
                valid = [x for x in accounts if "_equity" in x]
                expected_equity = pd.concat([x["_equity"] for x in valid], axis=1).sort_index().ffill().fillna(initial).sum(axis=1) + len(absent) * initial
                aggregate = _csv(location / "aggregate_daily.csv")
                summary = fold["variants"][variant]
                metrics = summary["metrics"]
                if (summary["frozen_accounts"] != len(frozen) or summary["accounts"] != len(valid)
                        or list(aggregate.date) != list(expected_equity.index) or not _close(aggregate.equity, expected_equity)
                        or not _close(metrics["final_equity"], expected_equity.iloc[-1])
                        or not np.isclose(metrics["total_return"], expected_equity.iloc[-1] / (len(frozen) * initial) - 1)
                        or metrics["trade_count"] != sum(x["ledger_rows"] for x in valid)
                        or not _close(metrics["fees"], sum(x["fees"] for x in valid))):
                    errors.append({"rule": "all_account_summary_fees_cash", "fold": name, "variant": variant})
                if variant == "legacy":
                    previous = _read(old / "reports" / (name + ".json"))["variants"]["czsc_price_volume"]
                    for key in ("final_equity", "total_return", "max_drawdown", "fees", "trade_count", "completed_round_trips", "trade_expectancy"):
                        if not np.isclose(metrics[key], previous["metrics"][key], rtol=1e-8, atol=1e-10 if "return" in key or "expectancy" in key or key == "max_drawdown" else .01):
                            errors.append({"rule": "legacy_summary_preserved", "fold": name, "metric": key})
                fold_result["variants"][variant] = {"accounts": len(valid), "cash_retained_accounts": len(absent),
                    "cash_retained_budget": len(absent) * initial, "initial_budget": len(frozen) * initial,
                    "ledger_rows": sum(x["ledger_rows"] for x in valid), "fees": sum(x["fees"] for x in valid),
                    "open_positions": sum(x["final_shares"] > 0 for x in valid), "final_equity": float(expected_equity.iloc[-1])}
            evaluations.append(fold_result)
    report = {"study_run_id": run_id, "parent_run_id": parent.name, "legacy_comparison_run_id": legacy_run_id,
        "audited_at": local_now().isoformat(), "passed": not errors, "error_count": len(errors), "errors": errors,
        "diagnostic_only": True, "independent_certification": False, "production_eligible": False,
        "frozen_stocks": len(records), "frozen_mainboard_accounts": len(frozen), "mature_actual_labels": sum(x["mature_labels"] for x in inspected),
        "training_rows": len(data), "models": models, "evaluation": evaluations,
        "limitations": ["Known development windows cannot certify a release.", "ML training conditions on complete actual strategy episodes; unfilled/unclosed rows are retained in per-stock labels and all-account evaluation.",
                        "Legacy preservation compares every account daily, ledger and rejections excluding run_id, plus full-window summary metrics."]}
    output = root / "reports/actual_entry_execution_audit.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-run-id", required=True)
    parser.add_argument("--legacy-run-id", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    result = audit_study(args.study_run_id, legacy_run_id=args.legacy_run_id, workers=args.workers)
    print(json.dumps({"study_run_id": args.study_run_id, "passed": result["passed"], "error_count": result["error_count"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
