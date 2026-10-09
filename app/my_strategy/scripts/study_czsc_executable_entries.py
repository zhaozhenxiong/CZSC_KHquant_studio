"""Development-only executable-entry and actual-exit-label study.

Previously inspected windows cannot certify production. Frozen feature inputs
are verified against the local raw prefix before any outcome is computed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd


def _raw(frame):
    columns = ["symbol", "date", "open", "high", "low", "close", "volume", "amount", "source", "has_trade_price"]
    raw = frame[columns].copy()
    raw["dt"] = pd.to_datetime(raw.date) + pd.Timedelta(hours=15)
    raw["id"] = np.arange(len(raw))
    return raw


def _prepare_record(args):
    from my_strategy.adapters.czsc_adapter import load_bars
    from my_strategy.services.czsc_research import _file_hash, _save
    from my_strategy.services.czsc_research_features import FEATURE_COLUMNS, FEATURE_VERSION, FEATURE_SCHEMA_HASH
    from my_strategy.services.czsc_research_strategy_labels import build_strategy_labels
    record, context, calendar, config = args
    path = Path(record["path"])
    if _file_hash(path) != record["sha256"]:
        raise ValueError("frozen feature hash mismatch: " + record["symbol"])
    frame = pd.read_parquet(path)
    if frame.attrs.get("schema_hash") != FEATURE_SCHEMA_HASH or frame.attrs.get("feature_version") != FEATURE_VERSION:
        raise ValueError("frozen feature semantics mismatch")
    live = load_bars(record["symbol"], end=frame.date.iloc[-1])
    raw = _raw(frame)
    if live.attrs["data_version"] != record["data_version"] or len(live) != len(raw):
        raise ValueError("raw prefix changed: " + record["symbol"])
    for column in raw.columns:
        if column in {"id", "dt"}:
            continue
        if column == "date":
            left = pd.to_datetime(live[column]).dt.strftime("%Y-%m-%d").to_numpy()
            right = pd.to_datetime(raw[column]).dt.strftime("%Y-%m-%d").to_numpy()
        else:
            left, right = live[column].to_numpy(), raw[column].to_numpy()
        if not np.all((left == right) | (pd.isna(left) & pd.isna(right))):
            raise ValueError("raw column changed: " + column)
    mainboard = record["symbol"].startswith("60") and record["symbol"].endswith(".SH") or record["symbol"].startswith("00") and record["symbol"].endswith(".SZ")
    evidence = {"symbol": record["symbol"], "sha256": record["sha256"], "data_version": record["data_version"], "bars": len(raw), "raw_verified": True, "mainboard": mainboard}
    if not mainboard:
        return evidence, None, None
    labels = build_strategy_labels(frame, raw, calendar, entry_policy="fresh", initial_cash=config["initial_cash"],
        entry_parameters=config["entry_parameters"])
    execution = labels.attrs.pop("execution")
    _save(context.subdir("label_audit", record["symbol"].replace(".", "_")) / "execution.json", execution)
    labels.to_parquet(context.subdir("labels") / (record["symbol"].replace(".", "_") + ".parquet"), index=False)
    selected = frame[(frame.date >= config["feature_start"]) & frame.input_eligible.astype(bool) & labels.label_available].copy()
    selected = selected[["symbol", "date", "input_eligible", *FEATURE_COLUMNS]]
    for column in ("label", "label_end", "label_available", "net_return", "invested_net_return"):
        selected[column] = labels.loc[selected.index, column]
    selected.attrs = {}
    evidence.update(mature_labels=int(labels.label_available.sum()), unavailability=labels.attrs["unavailable_counts"])
    return evidence, selected, labels.attrs["label_contract"]


def _evaluate_record(args):
    from my_strategy.services.czsc_research import _save, _round_trips, _file_hash
    from my_strategy.services.czsc_research_decisions import replay_decisions
    from my_strategy.services.czsc_analysis import strategy_config
    from my_strategy.services.czsc_backtest import execute_decisions, _write_frame
    from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
    record, fold, context, config, calendar, probabilities = args
    if _file_hash(Path(record["path"])) != record["sha256"]:
        raise ValueError("frozen evaluation input changed")
    frame = pd.read_parquet(record["path"])
    frame = frame[frame.date <= fold["test_end"]].copy()
    if frame.empty or not (frame.date >= fold["test_start"]).any():
        return {"symbol": record["symbol"], "missing": "no_test_bars_cash_retained", "accounts": {}}
    raw, accounts = _raw(frame), {}
    for variant, policy, mode in (("legacy", "legacy", "rules"), ("fresh", "fresh", "rules"), ("risk", "risk", "rules"), ("fresh_ml", "fresh", "ml")):
        replay = replay_decisions(frame, raw, mode=mode, probabilities=probabilities,
            thresholds=config["probability_threshold"], entry_policy=policy,
            position_start=fold["test_start"] if policy != "legacy" else None,
            entry_parameters=config["entry_parameters"])
        result = execute_decisions(record["symbol"], raw, replay["decisions"], fold["test_start"], config["initial_cash"], strategy_config(), context.run_id,
            market_dates=calendar, verified_sources=VERIFIED_SOURCES, entry_policy=policy)
        destination = context.subdir("evaluation", fold["name"], variant, record["symbol"].replace(".", "_"))
        for name in ("daily", "ledger", "rejections"):
            _write_frame(pd.DataFrame(result[name]), destination / (name + ".csv"))
        _save(destination / "entry_diagnostics.json", {key: result.get(key) for key in ("entry_diagnostics", "actual_position", "sell_pending", "consumed_plan_ids")})
        count, returns = _round_trips(result["ledger"])
        accounts[variant] = {"symbol": record["symbol"], "daily": result["daily"], "ledger": result["ledger"], "metrics": result["metrics"],
            "trades": result["trades"], "rejections": result["rejections"], "limitations": result.get("limitations", []), "completed_round_trips": count, "round_returns": returns}
    return {"symbol": record["symbol"], "missing": None, "accounts": accounts}


def study(parent_run_id, *, device="cuda:1", workers=4):
    from my_strategy.services.czsc_research import run_path, _save, _calendar, _file_hash
    from my_strategy.services.czsc_research_features import FEATURE_COLUMNS, FEATURE_VERSION, FEATURE_SCHEMA_HASH
    from my_strategy.services.czsc_research_strategy_labels import LABEL_VERSION
    from my_strategy.services.czsc_research_ml import train_model, PredictorSession
    from my_strategy.services.czsc_entry_plan import entry_plan_parameters, PLAN_VERSION
    from my_strategy.core.run_context import create_run_context, stable_hash
    from my_strategy.core.tz import local_now
    from my_strategy.web_dashboard.tasks import aggregate_accounts
    parent = run_path(parent_run_id)
    training_path = parent / "reports/research.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    records = training["dataset_records"]
    config = {**training["config"], "entry_parameters": entry_plan_parameters()}
    calendar, calendar_evidence = _calendar(training["calendar"]["run_id"])
    protocol = {"parent_run_id": parent_run_id, "parent_sha256": _file_hash(training_path), "stage": "development_only", "production_eligible": False,
        "known_windows": True, "policies": ["legacy", "fresh", "risk", "fresh_ml"], "ml_entry_policy": "fresh", "label_version": LABEL_VERSION,
        "plan_version": PLAN_VERSION, "risk_parameters": config["entry_parameters"], "seed": 42, "epochs": 20, "threshold": config["probability_threshold"],
        "config": config, "calendar": calendar_evidence, "confidence": "not_certifiable: known windows and 120-bar holding dependence exceed quarterly windows"}
    context = create_run_context(task="czsc-executable-entry-study", as_of_date=training["run_context"]["as_of_date"], config=protocol,
        stocks=training["run_context"]["stocks"], scope="train", data_version=training["data_version"], seed=42)
    _save(context.subdir("reports") / "protocol.json", protocol)
    print("RUN_ID=" + context.run_id, flush=True)
    output = {"run_id": context.run_id, "protocol": protocol, "coverage": [], "checkpoints": [], "evaluation": [], "production_eligible": False}
    frames, contract = [], None
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for index, (evidence, sample, item_contract) in enumerate(pool.map(_prepare_record, ((r, context, calendar, config) for r in records)), 1):
            output["coverage"].append(evidence)
            if sample is not None and len(sample):
                frames.append(sample)
                if contract is not None and contract != item_contract:
                    raise ValueError("label contracts differ across stocks")
                contract = item_contract
            if index % 100 == 0 or index == len(records):
                print(f"verified raw + actual labels {index}/{len(records)}", flush=True)
                _save(context.subdir("reports") / "study-progress.json", {**output, "stage": "labels"})
        data = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])
        data.attrs.update(feature_version=FEATURE_VERSION, schema_hash=FEATURE_SCHEMA_HASH, label_version=LABEL_VERSION,
            label_contract=contract, data_version=stable_hash({"parent": training["data_version"], "contract": contract}))
        data.to_parquet(context.subdir("dataset") / "actual_entry_labels.parquet", index=False)
        output["candidate_rows"] = len(data)
        mainboard = {e["symbol"] for e in output["coverage"] if e["mainboard"]}
        selected = [r for r in records if r["symbol"] in mainboard]
        predictor = PredictorSession(device=device)
        for fold in [*config["folds"], config["holdout"], {**config["production"], "name": "production"}]:
            name, directory = fold["name"], context.subdir("models", fold["name"])
            manifest = train_model(data, FEATURE_COLUMNS, directory, fold["train_end"], fold["validation_start"], fold["validation_end"], device=device,
                seed=42, epochs=20, progress=lambda n, total: print(f"aligned training {name} {n}/{total}", flush=True))
            output["checkpoints"].append({"name": name, "manifest": manifest})
            _save(context.subdir("reports") / "study-progress.json", {**output, "stage": "training"})
            if "test_start" not in fold:
                continue
            accounts = {key: [] for key in protocol["policies"]}
            missing = []
            for offset in range(0, len(selected), 32):
                batch, arrays, parts = selected[offset:offset + 32], [], []
                for record in batch:
                    frame = pd.read_parquet(record["path"])
                    frame = frame[frame.date <= fold["test_end"]]
                    mask = frame.date >= manifest["available_at"]
                    arrays.append((np.full(len(frame), np.nan), mask.to_numpy()))
                    part = frame.loc[mask, ["date", "input_eligible", *FEATURE_COLUMNS]].copy()
                    part.attrs = {}
                    parts.append(part)
                merged = pd.concat(parts, ignore_index=True)
                merged.attrs.update(feature_version=FEATURE_VERSION, schema_hash=FEATURE_SCHEMA_HASH)
                values = predictor.predict(merged, directory, as_of=fold["test_end"]) if len(merged) else np.array([])
                cursor, tasks = 0, []
                for record, (array, mask) in zip(batch, arrays, strict=True):
                    length = int(mask.sum())
                    array[mask] = values[cursor:cursor + length]
                    cursor += length
                    tasks.append((record, fold, context, config, calendar, array))
                for value in pool.map(_evaluate_record, tasks):
                    if value["missing"]:
                        missing.append({"symbol": value["symbol"], "reason": value["missing"]})
                    for key, account in value["accounts"].items():
                        accounts[key].append(account)
                if offset % 256 == 0 or offset + 32 >= len(selected):
                    print(f"all-account ledger {name} {min(offset+32,len(selected))}/{len(selected)}", flush=True)
            variants = {}
            for key, items in accounts.items():
                total = aggregate_accounts(items, len(selected) * config["initial_cash"], len(selected))
                returns = [x for item in items for x in item["round_returns"]]
                total["metrics"].update(completed_round_trips=len(returns), trade_expectancy=float(np.mean(returns)) if returns else None)
                total["metrics"]["label_return_denominator"] = "actual_buy_cash_debit_including_fee"
                pd.DataFrame(total["daily"]).to_csv(context.subdir("evaluation", name, key) / "aggregate_daily.csv", index=False)
                variants[key] = {"metrics": total["metrics"], "accounts": len(items), "frozen_accounts": len(selected)}
            evaluation = {"fold": fold, "variants": variants, "unavailable_accounts_cash_retained": missing,
                "confidence": protocol["confidence"], "independent_certification": False}
            output["evaluation"].append(evaluation)
            _save(context.subdir("reports") / (name + ".json"), evaluation)
            _save(context.subdir("reports") / "study-progress.json", {**output, "stage": "evaluation"})
    output.update(completed_at=local_now().isoformat(), compute_info=predictor.diagnostics, status="development_complete")
    _save(context.subdir("reports") / "study.json", output)
    context.write_metadata({"status": "development_complete", "protocol": protocol, "production_eligible": False, "candidate_rows": len(data)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run-id", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = study(args.parent_run_id, device=args.device, workers=args.workers)
    print(json.dumps({"run_id": result["run_id"], "labels": result["candidate_rows"], "production_eligible": False}))


if __name__ == "__main__":
    # Load project runtime environment in the parent; spawned CPU workers inherit it.
    import my_strategy.cli
    main()
