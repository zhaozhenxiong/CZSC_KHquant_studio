"""Independent, chronological CZSC learning runs and evidence-based screening."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
from dataclasses import replace
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from my_strategy.adapters.czsc_adapter import load_bars, list_symbols, latest_market_date, native_runtime
from my_strategy.core.config_loader import load_config
from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions, _write_frame
from my_strategy.web_dashboard.tasks import aggregate_accounts


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_path(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", run_id):
        raise ValueError("无效研究运行标识")
    return ARTIFACT_RUNS_ROOT / run_id


def _research_checkpoint(training: dict, root: Path, name: str):
    fold = next((item["fold"] for item in training.get("evaluation", []) if item["fold"]["name"] == name), None)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", name) or (name != "production" and fold is None):
        raise ValueError("此研究运行没有所选模型检查点")
    directory = (root / "models" / name).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("模型检查点必须位于所选研究运行内")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = {"name": name, "kind": "production" if name == "production" else "evaluation",
                  "available_at": manifest["available_at"], "train_label_end": manifest.get("train_label_end"),
                  "validation_end": manifest["validation_end"],
                  "test_start": fold["test_start"] if fold else None, "test_end": fold["test_end"] if fold else None}
    return checkpoint, directory, manifest


def _calendar_quality(features: pd.DataFrame, raw: pd.DataFrame, market_dates: list[str]) -> pd.DataFrame:
    """Keep sparse rows visible while vetoing windows with unverified missing sessions."""
    dates = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d").tolist()
    following = dict(zip(market_dates[:-1], market_dates[1:]))
    known = set(market_dates)
    gaps = np.array([day not in known or (i > 0 and following.get(dates[i - 1]) != day) for i, day in enumerate(dates)])
    window = int(features.attrs.get("quality_window_bars", 120))
    affected = pd.Series(gaps).rolling(window, min_periods=1).max().astype(bool).to_numpy()
    result = features.copy()
    for i in np.flatnonzero(affected):
        reasons = list(result.iloc[i]["reason_codes"])
        result.iat[i, result.columns.get_loc("reason_codes")] = list(dict.fromkeys([*reasons, "calendar_missing_market_bar_window"]))
    result.loc[affected, "input_eligible"] = False
    for column in ("rule_czsc_buy", "rule_czsc_sell", "rule_price_volume_buy", "rule_price_volume_sell", "rule_buy", "rule_sell"):
        result.loc[affected, column] = False
    result.attrs.update(calendar_hash=stable_hash(market_dates), calendar_gap_policy="veto_trailing_quality_window_preserve_rows")
    return result


def _prepare(symbol: str, end: str, market_dates: list[str], config: dict[str, Any], db_path: str | None):
    from my_strategy.services.czsc_research_features import build_features
    from my_strategy.services.czsc_research_ml import build_labels
    raw = load_bars(symbol, end=end, db_path=db_path)
    features = _calendar_quality(build_features(raw), raw, market_dates)
    labels = build_labels(raw, market_dates=market_dates, horizon=config["horizon"], initial_cash=config["initial_cash"])
    joined = features.copy()
    for column in labels:
        if column not in {"date", "symbol"}:
            joined[column] = labels[column]
    for column in ("open", "high", "low", "close", "volume", "amount", "source", "has_trade_price"):
        joined[column] = raw[column].to_numpy()
    joined["date"] = pd.to_datetime(joined["date"]).dt.strftime("%Y-%m-%d")
    joined.attrs = {**features.attrs, "data_version": raw.attrs["data_version"], "price_basis": raw.attrs["price_basis"]}
    return joined


def _calendar(calendar_run_id: str) -> tuple[list[str], dict[str, Any]]:
    root = run_path(calendar_run_id)
    path = root / "reports" / "calendar.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("verified") or not value.get("source"):
        raise ValueError("研究标签需要已核验的交易日历")
    dates = sorted(set(value["dates"]))
    return dates, {"run_id": calendar_run_id, "source": value["source"], "hash": stable_hash(value), "path": str(path)}


def build_dataset(symbols: list[str], end: str, context, config: dict[str, Any], market_dates: list[str],
                  *, cpu_workers: int = 4, db_path: str | None = None,
                  progress: Callable | None = None, check_cancel: Callable | None = None):
    from my_strategy.services.czsc_research_features import FEATURE_COLUMNS, FEATURE_VERSION, FEATURE_SCHEMA_HASH
    check = check_cancel or (lambda: None)
    records, failures, samples = [], [], []
    pool = ProcessPoolExecutor(max_workers=cpu_workers, mp_context=multiprocessing.get_context("spawn")) if cpu_workers > 1 else None
    finished = False
    def consume(symbol, data):
        path = context.subdir("dataset") / (symbol.replace(".", "_") + ".parquet")
        data.to_parquet(path, index=False)
        records.append({"symbol": symbol, "path": str(path), "data_version": data.attrs["data_version"], "bars": len(data),
                        "data_end": data.iloc[-1]["date"], "feature_schema_hash": data.attrs["schema_hash"], "sha256": _file_hash(path),
                        "input_eligible_rows": int(data["input_eligible"].sum()), "label_available_rows": int(data["label_available"].sum()),
                        "label_reason_counts": data["label_reason"].value_counts().to_dict()})
        if symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ"):
            selected = data[(data["date"] >= config["feature_start"]) & data["input_eligible"].astype(bool) & data["label"].notna()]
            samples.append(selected[["symbol", "date", "label", "label_end", "label_available", "input_eligible", *FEATURE_COLUMNS]])
    try:
        for offset in range(0, len(symbols), 32):
            check()
            batch = symbols[offset:offset + 32]
            if pool:
                pending = {pool.submit(_prepare, symbol, end, market_dates, config, db_path): symbol for symbol in batch}
                while pending:
                    check()
                    done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        symbol = pending.pop(future)
                        try:
                            consume(symbol, future.result())
                        except (ValueError, FileNotFoundError, RuntimeError) as exc:
                            failures.append({"symbol": symbol, "error": str(exc)})
                        if progress:
                            progress("逐日特征与标签", len(records) + len(failures), len(symbols), len(failures))
            else:
                for symbol in batch:
                    check()
                    try:
                        consume(symbol, _prepare(symbol, end, market_dates, config, db_path))
                    except (ValueError, FileNotFoundError, RuntimeError) as exc:
                        failures.append({"symbol": symbol, "error": str(exc)})
                    if progress:
                        progress("逐日特征与标签", len(records) + len(failures), len(symbols), len(failures))
        finished = True
    finally:
        if pool:
            pool.shutdown(wait=finished, cancel_futures=not finished)
    if not samples or not any(len(item) for item in samples):
        raise ValueError("没有通过来源、时点及可执行标签检查的训练样本")
    dataset = pd.concat(samples, ignore_index=True).sort_values(["date", "symbol"])
    dataset.attrs.update(feature_version=FEATURE_VERSION, schema_hash=FEATURE_SCHEMA_HASH, data_version=stable_hash({r["symbol"]: r["data_version"] for r in records}),
                         label_contract={"horizon": config["horizon"], "initial_cash": config["initial_cash"], "calendar_hash": stable_hash(market_dates), "label_version": "czsc_fixed_horizon_broker_v1"})
    _save(context.subdir("reports") / "dataset.json", {"records": records, "failures": failures, "rows": len(dataset), "config": config})
    dataset.to_parquet(context.subdir("dataset") / "training.parquet", index=False)
    return dataset, records, failures


def rule_decisions(features: pd.DataFrame, raw: pd.DataFrame, mode: str,
                   probabilities: np.ndarray | None = None, threshold: float = .55):
    from my_strategy.services.czsc_research_decisions import replay_decisions
    return replay_decisions(features, raw, mode=mode, probabilities=probabilities,
                            thresholds=threshold)["decisions"]


def evaluate_fold(records: list[dict], model_dir: Path, fold: dict, context, config: dict,
                  *, device: str, requested_symbols=None, market_dates=None, cpu_workers=1, progress=None, check_cancel=None):
    from my_strategy.services.czsc_research_ml import PredictorSession
    from my_strategy.services.czsc_research_evaluation import evaluate_record
    modes = ("czsc", "price_volume", "czsc_price_volume", "ml")
    model_available = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))["available_at"]
    accounts = {mode: [] for mode in modes}
    requested = requested_symbols if requested_symbols is not None else [r["symbol"] for r in records]
    frozen_pool = [s for s in requested if s.startswith("60") and s.endswith(".SH") or s.startswith("00") and s.endswith(".SZ")]
    selected = [r for r in records if r["symbol"] in frozen_pool]
    unavailable = [{"symbol": s, "reason": "dataset_preparation_failed_cash_retained"} for s in frozen_pool if s not in {r["symbol"] for r in selected}]
    check = check_cancel or (lambda: None)
    completed, finished = 0, False
    predictor = PredictorSession(device=device)
    pool = ProcessPoolExecutor(max_workers=cpu_workers, mp_context=multiprocessing.get_context("spawn")) if cpu_workers > 1 else None
    def consume(value):
        nonlocal completed
        if value["missing"]:
            unavailable.append(value["missing"])
        for mode, account in value["accounts"].items():
            accounts[mode].append(account)
        completed += 1
        if progress:
            progress("样本外账本 " + fold["name"], completed, len(selected), 0)
    try:
        for offset in range(0, len(selected), 32):
            check()
            pending = []
            for record in selected[offset:offset + 32]:
                check()
                if _file_hash(Path(record["path"])) != record["sha256"]:
                    raise ValueError("冻结特征文件哈希不匹配")
                features = pd.read_parquet(record["path"])
                features = features[features["date"] <= fold["test_end"]].copy()
                probabilities = np.full(len(features), np.nan)
                mask = features["date"] >= model_available
                if mask.any():
                    probabilities[mask] = predictor.predict(features.loc[mask], model_dir, as_of=fold["test_end"])
                args = (record, fold, context, config, probabilities, market_dates)
                if pool:
                    pending.append(pool.submit(evaluate_record, *args))
                else:
                    consume(evaluate_record(*args))
            while pending:
                check()
                done, waiting = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                pending = list(waiting)
                for future in done:
                    consume(future.result())
        finished = True
    finally:
        if pool:
            pool.shutdown(wait=finished, cancel_futures=not finished)
    totals = {}
    for mode, values in accounts.items():
        if not values:
            raise ValueError("样本外股票池没有可评估账户")
        totals[mode] = aggregate_accounts(values, len(frozen_pool) * config["initial_cash"], len(frozen_pool))
        totals[mode]["accounts"] = len(values)
        totals[mode]["metrics"]["completed_round_trips"] = sum(_round_trips(account["ledger"])[0] for account in values)
        round_returns = [value for account in values for value in _round_trips(account["ledger"])[1]]
        totals[mode]["metrics"]["trade_expectancy"] = float(np.mean(round_returns)) if round_returns else None
        _write_frame(pd.DataFrame(totals[mode]["daily"]), context.subdir("evaluation", fold["name"], mode) / "aggregate_daily.csv")
    comparisons = {mode: {"metrics": total["metrics"], "accounts": total["accounts"]} for mode, total in totals.items()}
    from my_strategy.services.czsc_research_ml import date_block_bootstrap
    equity = {mode: pd.DataFrame(totals[mode]["daily"]).set_index("date")["equity"] for mode in modes}
    paired = pd.concat({mode: series.pct_change() for mode, series in equity.items()}, axis=1)
    paired["difference"] = paired["ml"] - paired["czsc_price_volume"]
    uncertainty = {}
    for block in (11, 20, 40):
        try:
            uncertainty[str(block)] = date_block_bootstrap(paired.reset_index(), "difference", block_length=block, dependence_horizon=config["horizon"])
        except ValueError as exc:
            uncertainty[str(block)] = {"unavailable": str(exc)}
    result = {"fold": fold, "variants": comparisons, "paired_daily_return_uncertainty": uncertainty,
              "frozen_mainboard_accounts": len(frozen_pool), "unavailable_accounts_cash_retained": unavailable}
    _save(context.subdir("reports") / (fold["name"] + ".json"), result)
    return result


def _round_trips(ledger):
    inventory, paid, received, returns = 0, 0.0, 0.0, []
    for entry in ledger:
        if entry["action"] == "BUY":
            inventory += int(entry["shares"])
            paid += -float(entry["cash_flow"])
        else:
            inventory -= int(entry["shares"])
            received += float(entry["cash_flow"])
            if inventory == 0 and paid > 0:
                returns.append(received / paid - 1)
                paid, received = 0.0, 0.0
    return len(returns), returns


def train_research(*, end: str, calendar_run_id: str, symbols: list[str] | None = None, device: str = "auto",
                   cpu_workers: int = 4, progress=None, check_cancel=None, run_callback=None) -> dict:
    from my_strategy.services.czsc_research_features import FEATURE_COLUMNS
    from my_strategy.services.czsc_research_ml import train_model
    config = load_config("czsc_research")
    end = pd.Timestamp(end).date().isoformat()
    if latest_market_date(end) != end:
        raise ValueError("请求目标日行情缺失，研究训练不得静默回退")
    symbols = symbols or [row["symbol"] for row in list_symbols()]
    dates, calendar = _calendar(calendar_run_id)
    context = create_run_context(task="czsc-research-train", as_of_date=end, config=config, stocks=symbols, source="cli", scope="train", end_date=end)
    if run_callback:
        run_callback(context.run_id)
    context.write_metadata({"research_config": config, "calendar": calendar, "status": "running"})
    dataset, records, failures = build_dataset(symbols, end, context, config, dates, cpu_workers=cpu_workers, progress=progress, check_cancel=check_cancel)
    context = replace(context, data_version=dataset.attrs["data_version"])
    results = []
    for fold in [*config["folds"], config["holdout"]]:
        if check_cancel:
            check_cancel()
        if fold["test_end"] > end:
            continue
        model_dir = context.subdir("models", fold["name"])
        if progress:
            progress("训练 " + fold["name"], 0, 1, 0)
        manifest = train_model(dataset, FEATURE_COLUMNS, model_dir, fold["train_end"], fold["validation_start"], fold["validation_end"], device=device, seed=config["seed"], epochs=config["epochs"],
                               progress=(lambda c, t: progress("训练 " + fold["name"], c, t, 0)) if progress else None, check_cancel=check_cancel)
        result = evaluate_fold(records, model_dir, fold, context, config, device=device, requested_symbols=symbols, market_dates=dates, cpu_workers=cpu_workers, progress=progress, check_cancel=check_cancel)
        result["model_manifest"] = manifest
        results.append(result)
    wins = []
    for item in results:
        base, ml = item["variants"]["czsc_price_volume"]["metrics"], item["variants"]["ml"]["metrics"]
        interval = item["paired_daily_return_uncertainty"].get("11", {})
        wins.append(ml["total_return"] > base["total_return"] and ml["max_drawdown"] <= base["max_drawdown"] + config["drawdown_tolerance"]
                    and ml["completed_round_trips"] >= config["min_round_trips"] and ml["trade_expectancy"] is not None
                    and base["trade_expectancy"] is not None and ml["trade_expectancy"] > base["trade_expectancy"] and interval.get("lower", -1) > 0)
    full_universe = set(symbols) == {item["symbol"] for item in list_symbols()}
    gate = {"passed": full_universe and not failures and len(wins) >= 3 and all(wins), "full_universe": full_universe, "preparation_failures": len(failures), "windows": len(wins), "passing_windows": sum(wins), "rule": "完整冻结股票池且无准备失败；费用后收益及交易期望改善、回撤不恶化超过2个百分点、至少30次完整交易且配对日期块置信下限>0；至少3窗口全部通过"}
    production = config["production"]
    manifest = train_model(dataset, FEATURE_COLUMNS, context.subdir("models", "production"), production["train_end"], production["validation_start"], min(production["validation_end"], end), device=device, seed=config["seed"], epochs=config["epochs"],
                           progress=(lambda c, t: progress("训练 production", c, t, 0)) if progress else None, check_cancel=check_cancel)
    training_completed_at = manifest["training_completed_at"]
    from my_strategy.core.device import requested_torch_device
    trained_manifests = [item["model_manifest"] for item in results] + [manifest]
    devices = list(dict.fromkeys(item["device"] for item in trained_manifests))
    compute_info = {"research": True, "training": True, "requested_device": requested_torch_device(device),
                    "selected_devices": devices, "selected_device": devices[0] if len(devices) == 1 else "mixed",
                    "actual_devices": devices, "actual_device": devices[0] if len(devices) == 1 else "mixed",
                    "cuda_work": any(item.get("actual_cuda_training", False) for item in trained_manifests),
                    "mps_work": any(item.get("actual_mps_training", False) for item in trained_manifests),
                    "gpu_work": any(item.get("actual_gpu_training", item.get("actual_cuda_training", False)) for item in trained_manifests),
                    "model_training_models": len(trained_manifests),
                    "model_training_batches": sum(item.get("training_batches", 0) for item in trained_manifests)}
    result = {"run_id": context.run_id, "strategy_version": config["version"], "data_version": dataset.attrs["data_version"], "data_end": end, "data_range": {"start": config["feature_start"], "end": end},
              "coverage": {"requested": len(symbols), "success": len(records), "failed": len(failures)}, "failures": failures, "calendar": calendar,
              "dataset_records": records, "training_rows": len(dataset), "evaluation": results, "model_gate": gate, "model_manifest": manifest,
              "model_dir": str(context.subdir("models", "production")), "run_context": context.to_dict(), "config": config,
              "training_completed_at": training_completed_at, "compute_info": compute_info,
              "limitations": ["当前证券列表存在存活偏差；未复权、历史ST和公司行动边界仍适用。", "LLM复核不计入历史收益。", "固定等额独立账户，无共享现金。"]}
    _save(context.subdir("reports") / "research.json", result)
    context.write_metadata({"research_config": config, "calendar": calendar, "coverage": result["coverage"], "model_gate": gate, "training_completed_at": training_completed_at, "compute_info": compute_info, "status": "complete"})
    return result


def adjudicate(features: dict, *, probability: float | None, model_gate: bool, held: bool, data_end: str, requested_end: str, threshold: float = .55):
    reason_values = features.get("reason_codes")
    reasons = list(reason_values) if reason_values is not None else []
    supported = bool(features.get("point_confirmed"))
    eligible = bool(features.get("input_eligible")) and data_end == requested_end
    symbol = features["symbol"]
    mainboard = symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ")
    if data_end != requested_end:
        reasons.append("missing_target_date")
    if not mainboard:
        reasons.append("unsupported_board")
    evidence = [
        {"role": "czsc", "judgment": "sell" if features.get("rule_czsc_sell") else "support" if supported and features.get("rule_czsc_buy", features.get("rule_buy")) else "observe", "point_type": features.get("point_type"), "confirmed_at": features.get("point_confirmed_at"), "input_hash": features.get("input_hash")},
        {"role": "price_volume", "judgment": "support" if features.get("rule_price_volume_buy") else "observe", "values": {k: features.get(k) for k in ("ma20_bias", "ma60_bias", "vol_ratio20")}},
        {"role": "ml", "judgment": "support" if model_gate and probability is not None and probability >= threshold else "shadow", "probability": probability, "threshold": threshold, "validated": model_gate},
        {"role": "data_execution", "judgment": "allow" if eligible and mainboard else "veto", "reasons": list(reasons), "source": features.get("source"), "actual_data_end": data_end}]
    if not eligible or not mainboard:
        category, action = "excluded", "excluded"
    elif held and features.get("rule_sell"):
        category, action = "exit", "SELL"
    elif supported and features.get("rule_buy") and model_gate and probability is not None and probability >= threshold:
        category, action = "buy", "BUY"
    else:
        category, action = "watch", "watch"
        if not model_gate:
            reasons.append("model_shadow_gate_not_passed")
        if not supported:
            reasons.append("unconfirmed_point")
        if not features.get("rule_buy"):
            reasons.append("combined_rule_not_met")
        if probability is None:
            reasons.append("model_probability_unavailable")
        elif probability < threshold:
            reasons.append("model_probability_below_threshold")
    return category, action, reasons, evidence


def _prepare_scan_stock(symbol: str, end: str, market_dates: list[str], cache: dict | None):
    """Read-only CPU preparation; return only the latest point-in-time feature."""
    from my_strategy.services.czsc_research_features import build_features, FEATURE_VERSION, FEATURE_SCHEMA_HASH
    raw = load_bars(symbol, end=end)
    if cache and _file_hash(Path(cache["path"])) != cache["sha256"]:
        raise ValueError("冻结研究特征文件哈希不匹配")
    cache_hit = bool(cache and cache["data_version"] == raw.attrs["data_version"])
    features = pd.read_parquet(cache["path"]) if cache_hit else _calendar_quality(build_features(raw), raw, market_dates)
    if cache_hit:
        if cache.get("symbol") != symbol or "symbol" not in features or not features["symbol"].eq(symbol).all():
            raise ValueError("冻结研究特征股票身份不匹配")
        if (cache.get("feature_schema_hash") != FEATURE_SCHEMA_HASH or features.attrs.get("schema_hash") != FEATURE_SCHEMA_HASH
                or features.attrs.get("feature_version") != FEATURE_VERSION):
            raise ValueError("冻结研究特征语义版本/schema不匹配")
    if features.attrs.get("calendar_hash") != stable_hash(market_dates):
        features = _calendar_quality(features, raw, market_dates)
    features = features[pd.to_datetime(features["date"]) <= pd.Timestamp(end)]
    if features.empty:
        raise ValueError("请求时点无研究特征")
    return {"symbol": symbol, "features": features.iloc[[-1]].copy(), "cache_hit": cache_hit,
            "data_end": pd.Timestamp(raw.iloc[-1]["date"]).date().isoformat(),
            "reference_close": float(raw.iloc[-1]["close"]), "data_version": raw.attrs["data_version"]}


def scan_research(*, end: str, symbols: list[str] | None = None, model_run_id: str | None = None,
                  device: str = "auto", cpu_workers: int | None = None, batch_size: int | None = None,
                  progress=None, check_cancel=None, held_symbols: set[str] | None = None, run_callback=None,
                  usage_mode: str = "historical", model_policy: str = "auto", model_fold: str | None = None,
                  calendar_run_id: str | None = None, entry_policy: str = "legacy", start: str | None = None):
    from my_strategy.services.czsc_research_runtime import run_research
    return run_research(kind="scan", end=end, symbols=symbols, model_run_id=model_run_id, device=device,
                        cpu_workers=cpu_workers, batch_size=batch_size, progress=progress,
                        check_cancel=check_cancel, held_symbols=held_symbols, run_callback=run_callback,
                        usage_mode=usage_mode, model_policy=model_policy, model_fold=model_fold,
                        calendar_run_id=calendar_run_id, entry_policy=entry_policy, start=start)


def research_status():
    from my_strategy.storage.czsc_results import ResultStore
    from my_strategy.storage.czsc_model_releases import ModelReleaseStore
    calendars = sorted(ARTIFACT_RUNS_ROOT.glob("*/reports/calendar.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    calendar_run_id = None
    for path in calendars:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("verified"):
            calendar_run_id = path.parent.parent.name
            break
    models = []
    store = ResultStore()
    for item in store.list(kind="research_train", limit=20):
        summary = item["summary"]
        root = run_path(item["run_id"])
        training = json.loads((root / "reports" / "research.json").read_text(encoding="utf-8"))
        names = ["production", *(value["fold"]["name"] for value in training.get("evaluation", []))]
        checkpoints = [_research_checkpoint(training, root, name)[0] for name in names]
        models.append({"run_id": item["run_id"], "created_at": item["created_at"], "model_gate": summary.get("model_gate", {}),
                       "coverage": summary.get("coverage", {}), "checkpoints": checkpoints})
    releases = ModelReleaseStore()
    return {"models": models, "calendar_run_id": calendar_run_id,
            "active_release": releases.active(), "releases": releases.events(limit=30)}


def backtest_research(*, symbols: list[str], start: str, end: str, initial_cash: float,
                      model_run_id: str | None = None, model_fold: str | None = None, device: str = "auto",
                      progress=None, check_cancel=None, run_callback=None, usage_mode: str = "historical",
                      model_policy: str = "auto", calendar_run_id: str | None = None,
                      cpu_workers: int | None = None, batch_size: int | None = None, entry_policy: str = "legacy"):
    from my_strategy.services.czsc_research_runtime import run_research
    return run_research(kind="backtest", symbols=symbols, start=start, end=end, initial_cash=initial_cash,
                        model_run_id=model_run_id, model_fold=model_fold, device=device, progress=progress,
                        check_cancel=check_cancel, run_callback=run_callback, usage_mode=usage_mode,
                        model_policy=model_policy, calendar_run_id=calendar_run_id,
                        cpu_workers=cpu_workers, batch_size=batch_size, entry_policy=entry_policy)
