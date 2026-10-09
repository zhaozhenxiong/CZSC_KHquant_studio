"""Frozen-source, four-entry-arm plus independent-exit shadow research."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import copy
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.config_loader import load_config
from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions, _write_frame
from my_strategy.services.czsc_dual_models import DualPredictorSession, attach_exit_head, train_dual_bundle
from my_strategy.services.czsc_research import _file_hash, _json, _save
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services.czsc_research_decisions import replay_decisions
from my_strategy.services.czsc_research_ml import date_block_bootstrap
from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS, STRUCTURE_COLUMNS
from my_strategy.web_dashboard.tasks import aggregate_accounts

ENTRY_ARMS = ("rules", "ma_only", "structure_only", "fusion")
RAW_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount", "source", "has_trade_price"]


def _mainboard(symbol):
    return symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ")


def _raw(frame):
    raw = frame[RAW_COLUMNS].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["dt"] = raw["date"] + pd.Timedelta(hours=15)
    raw["id"] = np.arange(len(raw))
    raw.attrs = dict(frame.attrs)
    return raw


def prepare_frozen_dataset(source_training_run_id, context, config, *, symbols=None, end=None,
                           progress=None, check_cancel=None):
    """Reference immutable all-universe source files; project only chosen samples."""
    root = artifact_run_dir(source_training_run_id, create=False).resolve()
    report_path = root / "reports/research.json"
    source = json.loads(report_path.read_text())
    source_hash = _file_hash(report_path)
    if source.get("coverage", {}).get("failed") or not source.get("model_manifest"):
        raise ValueError("source research preparation incomplete")
    actual_end = source["data_end"]
    if end is not None and end != actual_end:
        raise ValueError("dual frozen-source end must exactly match source data_end")
    if source["model_manifest"]["label_contract"] != {
            "horizon": config["horizon"], "initial_cash": config["initial_cash"],
            "calendar_hash": source["model_manifest"]["label_contract"]["calendar_hash"],
            "label_version": "czsc_fixed_horizon_broker_v1"}:
        raise ValueError("source fixed-horizon broker target mismatch")
    calendar_path = Path(source["calendar"]["path"])
    calendar_value = json.loads(calendar_path.read_text())
    market_dates = calendar_value["dates"]
    if (not calendar_value.get("verified") or not calendar_value.get("source")
            or market_dates != sorted(set(market_dates))
            or stable_hash(calendar_value) != source["calendar"]["hash"]
            or stable_hash(market_dates) != source["model_manifest"]["label_contract"]["calendar_hash"]):
        raise ValueError("source verified calendar binding mismatch")
    all_records = source["dataset_records"]
    all_symbols = [r["symbol"] for r in all_records]
    if len(set(all_symbols)) != len(all_symbols) or len(all_symbols) != source["coverage"]["requested"]:
        raise ValueError("source universe duplicate/incomplete")
    requested = list(all_symbols if symbols is None else symbols)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(all_symbols):
        raise ValueError("selected symbols must be a unique subset of the frozen source")
    selected = set(requested)
    check = check_cancel or (lambda: None)
    records, samples = [], []
    columns = list(dict.fromkeys([*MA_TREND_COLUMNS, *STRUCTURE_COLUMNS, "date", "symbol", "available_at",
        "structure_confirmed_at", "zone_confirmed_at", "weekly_available_at", "monthly_available_at",
        "input_eligible", "reason_codes", "model_input_eligible", "model_reason_codes",
        "label", "label_end", "label_available", "net_return"]))
    expected_strategy_hash = stable_hash(strategy_config())
    for index, record in enumerate(all_records):
        check()
        path = Path(record["path"]).resolve()
        if not path.is_relative_to(root) or _file_hash(path) != record["sha256"]:
            raise ValueError("source snapshot path/hash mismatch: " + record["symbol"])
        if record["symbol"] in selected:
            frame = pd.read_parquet(path)
            if frame.attrs.get("price_basis") != "unadjusted" or frame.attrs.get("config_hash") != expected_strategy_hash:
                raise ValueError("source price/execution strategy binding mismatch")
            dates = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
            if (dates.duplicated().any() or not dates.is_monotonic_increasing
                    or not frame["symbol"].eq(record["symbol"]).all() or dates.iloc[-1] != record["data_end"]):
                raise ValueError("source snapshot identity/date mismatch")
            missing = set([*columns, *RAW_COLUMNS]) - set(frame)
            if missing:
                raise ValueError("source dual inputs missing: " + str(sorted(missing)))
            records.append({**record, "path": str(path), "source_run_id": source_training_run_id,
                            "snapshot_mode": "immutable_reference_sha256_verified"})
            if _mainboard(record["symbol"]):
                # Retain both eligibility fields. Each expert projects only its
                # own model copy; full frozen rule qualifications are unchanged.
                mask = dates >= config["feature_start"]
                sample = frame.loc[mask & frame["label_available"].astype(bool), columns].copy()
                samples.append(sample)
        if progress:
            progress("核验全源冻结快照", index + 1, len(all_records), 0)
    if not samples or not any(len(x) for x in samples):
        raise ValueError("selected mainboard universe has no mature labels")
    data = pd.concat(samples, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    data.attrs.update(data_version=stable_hash({r["symbol"]: r["data_version"] for r in records}),
        label_version="czsc_fixed_horizon_broker_v1", label_contract=source["model_manifest"]["label_contract"],
        source_training_run_id=source_training_run_id)
    data.to_parquet(context.subdir("dataset") / "training.parquet", index=False)
    own_calendar = context.subdir("data_snapshots") / "calendar.json"
    _save(own_calendar, calendar_value)
    calendar = {**source["calendar"], "path": str(own_calendar), "source_path": str(calendar_path),
                "source_file_sha256": _file_hash(calendar_path)}
    snapshot = {"source_run_id": source_training_run_id, "source_report_path": str(report_path),
        "source_report_sha256": source_hash, "source_universe_count": len(all_records),
        "all_source_records": all_records, "selected_symbols": requested,
        "selected_count": len(requested), "all_source_sha256_verified": True,
        "selected_full_source_universe": set(requested) == set(all_symbols),
        "strategy_hash": expected_strategy_hash, "mode": "immutable_references_no_original_changes"}
    _save(context.subdir("dataset") / "source-references.json", snapshot)
    return data, records, snapshot, market_dates, calendar


def round_trip_statistics(ledgers):
    """Actual fee-inclusive cash debits/credits; incomplete inventory stays open."""
    returns, pnls, unclosed = [], [], 0
    for ledger in ledgers:
        inventory, paid, received = 0, 0., 0.
        for row in ledger:
            if row["action"] == "BUY":
                inventory += int(row["shares"]); paid -= float(row["cash_flow"])
            else:
                inventory -= int(row["shares"]); received += float(row["cash_flow"])
                if inventory < 0:
                    raise ValueError("exit exceeds actual inventory")
                if inventory == 0 and paid > 0:
                    returns.append(received / paid - 1); pnls.append(received - paid)
                    paid, received = 0., 0.
        unclosed += int(inventory > 0)
    wins, losses = [x for x in pnls if x > 0], [x for x in pnls if x < 0]
    return {"completed_round_trips": len(returns), "win_rate": float(np.mean(np.array(pnls) > 0)) if pnls else None,
        "trade_expectancy": float(np.mean(returns)) if returns else None,
        "mean_net_pnl_CNY": float(np.mean(pnls)) if pnls else None,
        "payoff_ratio": float(np.mean(wins) / -np.mean(losses)) if wins and losses else None,
        "profit_factor": float(sum(wins) / -sum(losses)) if wins and losses else None,
        "wins": len(wins), "losses": len(losses), "breakeven": len(pnls) - len(wins) - len(losses),
        "unclosed_positions": unclosed, "payoff_definition": "mean_winning_net_cash_pnl/abs(mean_losing_net_cash_pnl)"}


def _evaluate_record(payload):
    record, fold, path, settings, cash, calendar, probabilities, thresholds, exit_model = payload
    symbol = record["symbol"]
    source = Path(record["path"])
    if _file_hash(source) != record["sha256"]:
        raise ValueError("frozen evaluation source hash mismatch")
    features = pd.read_parquet(source)
    features = features.loc[pd.to_datetime(features["date"]) <= pd.Timestamp(fold["test_end"])].copy()
    if not (pd.to_datetime(features["date"]) >= pd.Timestamp(fold["test_start"])).any():
        return {"symbol": symbol, "accounts": {}, "cash_reason": "no_test_bars"}
    raw = _raw(features)
    accounts = {}
    for arm in ENTRY_ARMS:
        expert = {"ma_only": "ma", "structure_only": "structure", "fusion": "fusion"}.get(arm)
        replay = replay_decisions(features, raw, probabilities=probabilities.get(expert) if expert else None,
            thresholds=thresholds[expert] if expert else .55, apply_ml_mask=expert is not None,
            mode="rules", entry_policy="fresh", position_start=fold["test_start"], config=settings)
        decisions = replay["decisions"]
        account = execute_decisions(symbol, raw, decisions, fold["test_start"], cash, settings,
            path.name, market_dates=calendar, verified_sources=VERIFIED_SOURCES, entry_policy="fresh")
        accounts[arm] = account
        if arm in {"rules", "fusion"} and exit_model is not None:
            if exit_model.get("unavailable"):
                exit_account = copy.deepcopy(account)
                exit_account["exit_policy_diagnostics"] = {"status": "rules_fallback", "reason": exit_model["unavailable"]}
                exit_account["exit_compute_info"] = {"rows": 0, "batches": 0, "actual_mps_inference": False}
            else:
                from my_strategy.services.czsc_dual_exit import evaluate_exit_arm
                exit_account = evaluate_exit_arm(features=features, raw=raw, base_decisions=decisions,
                    market_dates=calendar, start=fold["test_start"], end=fold["test_end"], initial_cash=cash,
                    config=settings, model_bundle=exit_model, device="cpu", apply_exit=True)
            accounts[arm + "_exit"] = exit_account
    for arm, account in accounts.items():
        destination = path / "evaluation" / fold["name"] / arm / symbol.replace(".", "_")
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("ledger", "daily", "rejections"):
            _write_frame(pd.DataFrame(account[name]), destination / (name + ".csv"))
    return {"symbol": symbol, "accounts": accounts}


def evaluate_dual_fold(records, bundle_dir, fold, context, config, market_dates, *, cpu_workers=4,
                       device="mps", progress=None, check_cancel=None, exit_model=None):
    predictor = DualPredictorSession(device=device)
    bundle = json.loads((Path(bundle_dir) / "bundle.json").read_text())
    thresholds = {name: item["threshold"] for name, item in bundle["thresholds"].items()}
    selected = [r for r in records if _mainboard(r["symbol"])]
    settings = strategy_config()
    tasks = []
    check = check_cancel or (lambda: None)
    for index, record in enumerate(selected):
        check()
        source = Path(record["path"])
        if _file_hash(source) != record["sha256"]:
            raise ValueError("frozen prediction source hash mismatch")
        frame = pd.read_parquet(source)
        frame = frame.loc[pd.to_datetime(frame["date"]) <= pd.Timestamp(fold["test_end"])].copy()
        mask = pd.to_datetime(frame["date"]) >= pd.Timestamp(bundle["available_at"])
        values = {name: np.full(len(frame), np.nan) for name in ("ma", "structure", "fusion")}
        if mask.any():
            predictor.predict(frame.loc[mask].copy(), bundle_dir, as_of=fold["test_end"])
            for name in values:
                values[name][mask.to_numpy()] = predictor.last_expert_values[name]
        tasks.append((record, fold, context.run_dir(), settings, config["initial_cash"], market_dates,
                      values, thresholds, exit_model))
        if progress:
            progress("独立双专家批量概率", index + 1, len(selected), 0)
    arm_names = [*ENTRY_ARMS, *(["rules_exit", "fusion_exit"] if exit_model is not None else [])]
    accounts, cash_reasons = {arm: [] for arm in arm_names}, []
    if cpu_workers > 1:
        with ProcessPoolExecutor(max_workers=cpu_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = pool.map(_evaluate_record, tasks, chunksize=1)
            for i, result in enumerate(results):
                check()
                if result["accounts"]:
                    for arm, account in result["accounts"].items(): accounts[arm].append(account)
                else: cash_reasons.append({"symbol": result["symbol"], "reason": result["cash_reason"]})
                if progress: progress("同候选池fresh独立账本", i + 1, len(tasks), 0)
    else:
        for i, task in enumerate(tasks):
            check(); result = _evaluate_record(task)
            if result["accounts"]:
                for arm, account in result["accounts"].items(): accounts[arm].append(account)
            else: cash_reasons.append({"symbol": result["symbol"], "reason": result["cash_reason"]})
            if progress: progress("同候选池fresh独立账本", i + 1, len(tasks), 0)
    totals, variants = {}, {}
    for arm, values in accounts.items():
        if not values:
            raise ValueError("no executable evaluation dates; cash-only arm evidence unavailable")
        total = aggregate_accounts(values, len(selected) * config["initial_cash"], len(selected))
        total["metrics"].update(round_trip_statistics([account["ledger"] for account in values]))
        totals[arm] = total
        variants[arm] = {"metrics": total["metrics"], "accounts": len(values)}
        _write_frame(pd.DataFrame(total["daily"]), context.subdir("evaluation", fold["name"], arm) / "aggregate_daily.csv")
    initial = len(selected) * config["initial_cash"]
    def daily_returns(arm):
        equity = pd.DataFrame(totals[arm]["daily"]).set_index("date")["equity"]
        prior = equity.shift(1); prior.iloc[0] = initial
        return equity / prior - 1
    baseline = daily_returns("rules")
    comparisons = {}
    for arm in arm_names[1:]:
        daily = daily_returns(arm)
        paired = pd.concat({"rules": baseline, arm: daily}, axis=1)
        paired["difference"] = paired[arm] - paired["rules"]
        comparisons[arm] = {}
        for block in config.get("ledger_bootstrap_blocks", [160, 240]):
            try:
                comparisons[arm][str(block)] = date_block_bootstrap(paired.reset_index(), "difference", block_length=block,
                    dependence_horizon=config.get("ledger_dependence_bars", 120), seed=config["seed"])
            except ValueError as exc:
                comparisons[arm][str(block)] = {"unavailable": str(exc)}
    result = {"fold": fold, "qualification": "observed_historical_development_diagnostic",
        "variants": variants, "paired_daily_return_uncertainty": comparisons,
        "frozen_mainboard_accounts": len(selected), "unavailable_accounts_cash_retained": cash_reasons,
        "candidate_contract": "same_original_rule_buy_fresh_candidates_expert_entry_filters_only",
        "exit_model": exit_model,
        "bootstrap_contract": {"blocks": config.get("ledger_bootstrap_blocks", [160, 240]),
            "dependence_bars": config.get("ledger_dependence_bars", 120), "quarterly_intervals_expected_unavailable": True},
        "compute_info": {**predictor.diagnostics,
            "exit_inference": {"rows": sum(a.get("exit_compute_info", {}).get("rows", 0) for arm in ("rules_exit", "fusion_exit") for a in accounts.get(arm, [])),
                "batches": sum(a.get("exit_compute_info", {}).get("batches", 0) for arm in ("rules_exit", "fusion_exit") for a in accounts.get(arm, [])),
                "actual_mps_inference": False, "device": "cpu_numpy"}}}
    _save(context.subdir("reports") / (fold["name"] + "-dual.json"), result)
    return result


def prepare_exit_training(records, context, config, market_dates, *, progress=None, check_cancel=None):
    from my_strategy.services.czsc_dual_exit import build_exit_dataset
    parts = []
    selected = [r for r in records if _mainboard(r["symbol"])]
    settings = strategy_config(); check = check_cancel or (lambda: None)
    attrs = None
    for i, record in enumerate(selected):
        check()
        if _file_hash(Path(record["path"])) != record["sha256"]:
            raise ValueError("frozen exit source hash mismatch")
        features = pd.read_parquet(record["path"]); raw = _raw(features)
        decisions = replay_decisions(features, raw, entry_policy="fresh", position_start=config["feature_start"], config=settings)["decisions"]
        def reference_callback(account):
            destination = context.subdir("evaluation", "exit_reference", record["symbol"].replace(".", "_"))
            for name in ("ledger", "daily", "rejections", "holding_snapshots"):
                _write_frame(pd.DataFrame(account.get(name, [])), destination / (name + ".csv"))
            _save(destination / "reference-metadata.json", {"source_path": record["path"], "source_sha256": record["sha256"],
                "entry_policy": "fresh", "decision_hash": stable_hash(decisions), "strategy_hash": stable_hash(settings),
                "metrics": account.get("metrics", {}), "exit_policy_diagnostics": account.get("exit_policy_diagnostics", {})})
        data = build_exit_dataset(features=features, raw=raw, base_decisions=decisions,
            market_dates=market_dates, start=config["feature_start"], initial_cash=config["initial_cash"], config=settings,
            entry_policy="fresh", reference_callback=reference_callback)
        if len(data):
            if attrs is not None and data.attrs.get("label_contract") != attrs.get("label_contract"):
                raise ValueError("exit target contract differs between accounts")
            attrs = dict(data.attrs); parts.append(data)
        if progress: progress("独立实际持仓退出标签", i + 1, len(selected), 0)
    if not parts:
        return pd.DataFrame()
    data = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    data.attrs = attrs
    data.attrs.update(data_version=stable_hash({"source_records": {r["symbol"]: r["sha256"] for r in selected},
        "contract": attrs["label_contract"], "entry_policy": "fresh", "strategy_hash": stable_hash(settings),
        "reference_start": config["feature_start"]}), actual_held_samples=len(data),
        label_reason_counts=dict(Counter(data["label_reason"])))
    data.attrs.pop("reference_account_metrics", None)
    data.attrs.pop("reference_rejections", None)
    data.attrs["reference_accounts"] = len(selected)
    data.to_parquet(context.subdir("dataset") / "exit-training.parquet", index=False)
    return data


def train_dual_research(*, source_training_run_id, end=None, symbols=None, device="mps", cpu_workers=4,
                        progress=None, check_cancel=None, run_callback=None, config=None, exit_evaluator=None):
    """All historical arms are diagnostics. This API cannot publish a model."""
    if exit_evaluator is not None:
        raise ValueError("use the exact isolated exit-head contract; arbitrary evaluators are unsupported")
    cfg = dict(load_config("czsc_dual_research") if config is None else config)
    if cfg["horizon"] != 10 or cfg["initial_cash"] != 100000 or cfg.get("publication_allowed") is not False:
        raise ValueError("dual fixed-horizon/shadow contract is immutable")
    source = json.loads((artifact_run_dir(source_training_run_id, create=False) / "reports/research.json").read_text())
    requested_symbols = [r["symbol"] for r in source["dataset_records"]] if symbols is None else list(symbols)
    version = stable_hash({r["symbol"]: r["data_version"] for r in source["dataset_records"] if r["symbol"] in set(requested_symbols)})
    context = create_run_context(task="czsc-dual-research-train", as_of_date=end or source["data_end"],
        config={"research": cfg, "strategy": strategy_config()}, seed=cfg["seed"], scope="train", stocks=requested_symbols,
        source="api", data_version=version, start_date=cfg["feature_start"], end_date=end or source["data_end"])
    if run_callback: run_callback(context.run_id)
    try:
        data, records, snapshot, market_dates, calendar = prepare_frozen_dataset(source_training_run_id, context, cfg,
            symbols=symbols, end=end, progress=progress, check_cancel=check_cancel)
        _save(context.subdir("config") / "czsc_dual_research.json", cfg)
        _save(context.subdir("config") / "czsc_strategy.json", strategy_config())
        exits = prepare_exit_training(records, context, cfg, market_dates, progress=progress, check_cancel=check_cancel) if cfg.get("exit_enabled", True) else None
        evaluation, bundles, exit_manifests = [], [], []
        for fold in [*cfg["development_windows"], {"name": "production", **cfg["production"]}]:
            directory = context.subdir("models", fold["name"])
            bundle = train_dual_bundle(data, directory, fold, device=device, config=cfg,
                progress=(lambda current, total: progress("双专家与严格时序OOF训练 " + fold["name"], current, total, 0)) if progress else None,
                check_cancel=check_cancel)
            exit_binding = None
            if exits is not None:
                from my_strategy.services.czsc_dual_exit import train_exit_model
                exit_directory = context.subdir("models", fold["name"], "exit")
                if exits.empty:
                    exit_binding = {"unavailable": "no_actual_holdings_labels", "rules_exit_retained": True}
                else:
                    try:
                        exit_manifest = train_exit_model(exits, exit_directory, fold["train_end"], fold["validation_start"], fold["validation_end"],
                            device=device, seed=cfg["seed"], epochs=cfg["epochs"], hidden_sizes=())
                        exit_manifests.append(exit_manifest)
                        exit_binding = {"model_dir": str(exit_directory), "device": "cpu",
                            "contract_hash": exit_manifest["label_contract"]["contract_hash"],
                            "threshold": cfg.get("exit_threshold", .55)}
                    except ValueError as exc:
                        if not any(text in str(exc) for text in ("insufficient mature samples", "each require both label classes", "insufficient feasible purged")):
                            raise
                        exit_binding = {"unavailable": str(exc), "rules_exit_retained": True}
                bundle = attach_exit_head(directory, exit_binding)
                exit_binding = bundle["exit_head"]
            bundles.append({"name": fold["name"], "path": str(directory), "available_at": bundle["available_at"],
                            "bundle_sha256": bundle["bundle_sha256"], "exit_model": exit_binding})
            if "test_start" in fold:
                evaluation.append(evaluate_dual_fold(records, directory, fold, context, cfg, market_dates,
                    cpu_workers=cpu_workers, device=device, progress=progress, check_cancel=check_cancel, exit_model=exit_binding))
        if _file_hash(Path(snapshot["source_report_path"])) != snapshot["source_report_sha256"]:
            raise ValueError("source report changed during isolated research")
        for record in snapshot["all_source_records"]:
            if _file_hash(Path(record["path"])) != record["sha256"]:
                raise ValueError("source snapshot changed during isolated research")
        manifests = [json.loads((Path(item["path"]) / "bundle.json").read_text()) for item in bundles]
        report = {"run_id": context.run_id, "status": "complete", "model_family": "dual", "strategy_version": cfg["version"],
            "publication_allowed": False, "promotion_status": "shadow_future_independent_window_missing",
            "independent_window": cfg["independent_window"], "observed_windows_diagnostic_only": True,
            "source_snapshot": snapshot, "source_artifacts_unchanged": True, "data_end": source["data_end"],
            "data_version": data.attrs["data_version"], "dataset_records": records, "calendar": calendar,
            "market_dates": market_dates, "config": cfg, "run_context": context.to_dict(), "training_rows": len(data),
            "coverage": {"requested": len(snapshot["selected_symbols"]), "success": len(records), "failed": 0,
                         "source_universe": snapshot["source_universe_count"], "full_universe": snapshot["selected_full_source_universe"]},
            "model_bundles": bundles, "evaluation": evaluation,
            "compute_info": {"training_batches": sum(m["compute_info"]["training_batches"] for m in manifests) + sum(m["training_batches"] for m in exit_manifests),
                             "trained_models": sum(m["compute_info"]["trained_models"] for m in manifests) + len(exit_manifests),
                             "exit_training_batches": sum(m["training_batches"] for m in exit_manifests),
                             "exit_trained_models": len(exit_manifests),
                             "actual_mps_training": any(m["compute_info"]["actual_mps_training"] for m in manifests),
                             "actual_cuda_training": any(m["compute_info"]["actual_cuda_training"] for m in manifests),
                             "inference_rows": sum(e["compute_info"]["rows"] for e in evaluation),
                             "inference_batches": sum(e["compute_info"]["batches"] for e in evaluation),
                             "exit_cpu_numpy_rows": sum(e["compute_info"]["exit_inference"]["rows"] for e in evaluation),
                             "exit_cpu_numpy_batches": sum(e["compute_info"]["exit_inference"]["batches"] for e in evaluation)},
            "model_gate": {"passed": False, "reason": "historical_windows_already_observed_no_new_independent_future_window"},
            "limitations": ["All four historical windows are development diagnostics; no independent future window has occurred.",
                "Current-security-pool survivorship and unadjusted prices; cash dividends/splits and historical ST identities absent.",
                "Only SH60/SZ00 long-only execution; unsupported boards remain visible as prepared sources.",
                "Separate equal-capital accounts; no shared portfolio cash or forced terminal exits.",
                "Exit head learns baseline fresh-rule holdings; fusion holdings can have a different state distribution."]}
        _save(context.subdir("reports") / "dual-research.json", report)
        context.write_metadata({"config": {"research": cfg, "strategy": strategy_config()}, "status": "complete",
            "model_family": "dual", "publication_allowed": False, "coverage": report["coverage"], "compute_info": report["compute_info"]})
        return _json(report)
    except Exception as exc:
        context.write_metadata({"config": {"research": cfg, "strategy": strategy_config()}, "status": "failed", "error": str(exc), "publication_allowed": False})
        raise
