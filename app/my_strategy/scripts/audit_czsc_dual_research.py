"""Independent, read-only audit of dual entry experts and held-position exits.

The audit never imports Torch or deserializes weights. It verifies frozen source
files, forward OOF maturity, model contracts and actual Broker ledger cash flows.
Only reports/dual-execution-audit.json is written in the audited run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now
from my_strategy.scripts.audit_czsc_research import (
    _account, _bounded, _close, _csv, _error, _file_hash, _hash, _read, _run,
)


def _check_model(path: Path, root: Path) -> dict:
    """Check contracts and physical weights without executing the model."""
    errors = []
    try:
        path = _bounded(path, root)
        manifest = _read(path)
        integrity = dict(manifest)
        recorded = integrity.pop("manifest_sha256", None)
        if recorded != _hash(integrity):
            errors.append(_error("model_manifest_sha256", path=str(path)))
        for field in ("schema", "preprocess"):
            if manifest.get(field + "_sha256") != _hash(manifest[field]):
                errors.append(_error("model_" + field + "_sha256", path=str(path)))
        checkpoint = _bounded(manifest["checkpoint"], root)
        if checkpoint != (path.parent / "model.pt").resolve() or _file_hash(checkpoint) != manifest["checkpoint_sha256"]:
            errors.append(_error("model_checkpoint_sha256_or_path", path=str(path)))
        columns = manifest["schema"]["columns"]
        if len(columns) != len(set(columns)) or not columns:
            errors.append(_error("model_unique_ordered_columns", path=str(path)))
        if manifest.get("feature_schema_binding") != "bound" or not manifest.get("feature_version") or not manifest.get("feature_schema_hash"):
            errors.append(_error("model_bound_feature_schema", path=str(path)))
        train_end, validation_start = manifest["train_end"], manifest["validation_start"]
        validation_end, available_at = manifest["validation_end"], manifest["available_at"]
        if not (manifest["train_label_end"] <= train_end < validation_start <= manifest["validation_label_end"] <= validation_end == available_at):
            errors.append(_error("model_purged_time_boundaries", path=str(path)))
        if manifest["calibration"]["fit_start"] != validation_start or manifest["calibration"]["fit_end"] != validation_end:
            errors.append(_error("model_validation_only_calibration", path=str(path)))
        if not manifest.get("training_completed_at") or not manifest.get("training_started_at"):
            errors.append(_error("model_actual_training_time_recorded", path=str(path)))
        elif manifest["training_started_at"] > manifest["training_completed_at"]:
            errors.append(_error("model_actual_training_time_order", path=str(path)))
        if manifest.get("promotion_status") not in {"candidate", "shadow", "research_candidate", "unqualified_candidate", "shadow_pending_complete_universe_ledger_gate"}:
            errors.append(_error("model_candidate_only", path=str(path), status=manifest.get("promotion_status")))
        return {"path": str(path), "manifest_sha256": manifest.get("manifest_sha256"),
                "checkpoint_sha256": manifest.get("checkpoint_sha256"), "available_at": available_at,
                "feature_version": manifest.get("feature_version"), "label_version": manifest.get("label_version"),
                "sample_counts": manifest["sample_counts"], "device": manifest.get("device"),
                "manifest": manifest, "errors": errors}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"path": str(path), "errors": [*errors, _error("model_read", path=str(path), error=str(exc))]}


def _source_record(record: dict, base: Path) -> dict:
    """Verify complete physical input snapshots, including unavailable rows."""
    errors = []
    symbol = record.get("symbol")
    try:
        path = _bounded(record["path"], base)
        if _file_hash(path) != record["sha256"]:
            errors.append(_error("source_parquet_sha256", symbol=symbol, path=str(path)))
        columns = ["symbol", "date", "input_eligible", "model_input_eligible", "label", "label_available", "label_end",
                   "open", "high", "low", "close", "volume", "amount", "has_trade_price", "structure_confirmed_at"]
        frame = pd.read_parquet(path, columns=columns)
        dates = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        if len(frame) != record["bars"] or frame.empty or dates.duplicated().any() or not dates.is_monotonic_increasing or set(frame["symbol"]) != {symbol} or dates.iloc[-1] != record["data_end"]:
            errors.append(_error("source_symbol_date_coverage", symbol=symbol))
        if frame.attrs.get("schema_hash") != record["feature_schema_hash"] or frame.attrs.get("data_version") != record["data_version"]:
            errors.append(_error("source_schema_data_binding", symbol=symbol))
        available = frame["label_available"].fillna(False).astype(bool)
        if not np.array_equal(available, frame["label"].notna()) or not frame.loc[available, "label"].isin([0, 1]).all():
            errors.append(_error("source_label_availability", symbol=symbol))
        if available.any() and (frame.loc[available, "label_end"].astype(str) > dates.iloc[-1]).any():
            errors.append(_error("source_label_maturity", symbol=symbol))
        return {"symbol": symbol, "rows": len(frame), "input_eligible_rows": int(frame["input_eligible"].sum()),
                "model_input_eligible_rows": int(frame["model_input_eligible"].sum()),
                "label_available_rows": int(available.sum()), "errors": errors}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"symbol": symbol, "rows": 0, "input_eligible_rows": 0, "model_input_eligible_rows": 0,
                "label_available_rows": 0, "errors": [*errors, _error("source_read", symbol=symbol, error=str(exc))]}


def _check_oof(frame: pd.DataFrame, outer_train_end: str, *, fold: str) -> list[dict]:
    """Reject in-sample scores and future-maturing rows in fusion fitting."""
    errors = []
    required = {"date", "symbol", "label", "label_end", "p_ma", "p_structure", "expert_train_end", "expert_validation_end", "expert_available_at", "oof_fold"}
    if not required <= set(frame):
        return [_error("oof_required_provenance", fold=fold, missing=sorted(required - set(frame)))]
    if frame.empty or frame[["symbol", "date"]].duplicated().any():
        errors.append(_error("oof_unique_nonempty_observations", fold=fold))
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    ends = pd.to_datetime(frame["label_end"], errors="coerce").dt.strftime("%Y-%m-%d")
    available = frame["expert_available_at"].astype(str)
    valid = (dates.notna() & ends.notna() & (frame["expert_train_end"].astype(str) < frame["expert_validation_end"].astype(str))
             & (frame["expert_validation_end"].astype(str) == available) & (available < dates) & (ends >= dates) & (ends <= outer_train_end))
    if not valid.all():
        errors.append(_error("oof_forward_prediction_and_label_maturity", fold=fold, invalid_rows=int((~valid).sum())))
    probabilities = frame[["p_ma", "p_structure"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any() or not frame["label"].isin([0, 1]).all():
        errors.append(_error("oof_probability_binary_target", fold=fold))
    return errors


def _round_trip_metrics(ledger: pd.DataFrame) -> dict:
    """Recompute after-cost closed trades, including partial sells."""
    shares, paid, received = 0, 0.0, 0.0
    returns = []
    profits = []
    for trade in ledger.to_dict(orient="records"):
        if trade["action"] == "BUY":
            shares += int(trade["shares"])
            paid -= float(trade["cash_flow"])
        elif trade["action"] in {"SELL", "REDUCE"}:
            shares -= int(trade["shares"])
            received += float(trade["cash_flow"])
            if shares == 0 and paid > 0:
                profits.append(received - paid)
                returns.append(received / paid - 1)
                paid, received = 0.0, 0.0
    return {"completed_round_trips": len(returns), "trade_expectancy": float(np.mean(returns)) if returns else None,
            "win_rate": float(np.mean(np.asarray(profits) > 0)) if profits else None,
            "net_pnl": profits, "net_returns": returns}


def _check_exit_dataset(frame: pd.DataFrame) -> list[dict]:
    """An exit target must belong to actual inventory and have both outcomes."""
    errors = []
    # Field names are the persisted audit contract, not native Position states.
    required = {"symbol", "date", "actual_shares", "actual_entry_date", "actual_cost_per_share", "label", "label_available", "label_end",
                "snapshot_available_at", "ledger_entries_seen", "forced_net_proceeds", "continuation_net_proceeds", "forced_exit_date", "continuation_exit_date"}
    if not required <= set(frame):
        return [_error("exit_actual_inventory_and_outcome_columns", missing=sorted(required - set(frame)))]
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    entries = pd.to_datetime(frame["actual_entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if frame[["symbol", "date"]].duplicated().any() or not dates.notna().all():
        errors.append(_error("exit_unique_dated_actual_states"))
    state = frame[["actual_shares", "actual_cost_per_share"]].to_numpy(dtype=float)
    if not np.isfinite(state).all() or (state <= 0).any() or not (entries <= dates).all() or (frame["actual_shares"] % 100 != 0).any():
        errors.append(_error("exit_actual_held_positive_cost_state"))
    observed = pd.to_datetime(frame["snapshot_available_at"], utc=True, errors="coerce")
    if not observed.notna().all() or not (observed.dt.strftime("%Y-%m-%d") == dates).all() or (frame["ledger_entries_seen"] < 1).any():
        errors.append(_error("exit_contemporaneous_snapshot_provenance"))
    available = frame["label_available"].fillna(False).astype(bool)
    if not np.array_equal(available, frame["label"].notna()) or not frame.loc[available, "label"].isin([0, 1]).all():
        errors.append(_error("exit_label_availability_binary"))
    if available.any():
        selected = frame.loc[available]
        values = selected[["forced_net_proceeds", "continuation_net_proceeds"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any() or not np.array_equal(selected["label"].to_numpy(), (values[:, 0] > values[:, 1]).astype(float)):
            errors.append(_error("exit_label_compares_net_remaining_share_proceeds"))
        expected_end = selected[["forced_exit_date", "continuation_exit_date"]].astype(str).max(axis=1)
        if not np.array_equal(selected["label_end"].astype(str), expected_end) or not (selected["label_end"].astype(str) > dates.loc[available]).all():
            errors.append(_error("exit_label_two_path_full_exit_maturity"))
    return errors


def _check_bundle(path: Path, root: Path, training: pd.DataFrame | None = None) -> dict:
    """Independently bind the two experts, fusion and its forward-score history."""
    from my_strategy.services.czsc_research_profiles import get_feature_profile
    errors = []
    try:
        path = _bounded(path, root)
        directory = path if path.is_dir() else path.parent
        bundle = _read(directory / "bundle.json")
        alias = directory / "manifest.json"
        if alias.exists() and _read(alias) != bundle:
            errors.append(_error("bundle_manifest_alias_identity", path=str(directory)))
        unsigned = dict(bundle)
        recorded = unsigned.pop("bundle_sha256", None)
        if recorded != _hash(unsigned):
            errors.append(_error("bundle_sha256", path=str(directory)))
        if bundle.get("version") != "czsc_dual_experts_v1" or bundle.get("status") != "shadow" or bundle.get("publication_allowed") is not False:
            errors.append(_error("bundle_shadow_only", path=str(directory)))
        if bundle.get("target_hash") != _hash(bundle["target"]) or bundle["target"].get("label_version") != "czsc_fixed_horizon_broker_v1" or bundle["target"].get("horizon") != 10:
            errors.append(_error("bundle_fixed_ten_session_target", path=str(directory)))
        if not (bundle["train_label_end"] <= bundle["train_end"] < bundle["validation_start"] <= bundle["validation_end"] == bundle["available_at"]):
            errors.append(_error("bundle_purged_availability", path=str(directory)))
        if set(bundle["experts"]) != {"ma", "structure"}:
            errors.append(_error("bundle_two_experts", path=str(directory)))
        models = {}
        for name, binding in [*bundle["experts"].items(), ("fusion", bundle["fusion"])]:
            model_directory = _bounded(binding["path"], directory)
            model_path = model_directory / "manifest.json"
            checked = _check_model(model_path, root)
            errors.extend(checked.pop("errors"))
            manifest = checked.get("manifest")
            if manifest is None:
                continue
            models[name] = manifest
            if (binding["manifest_file_sha256"] != _file_hash(model_path) or binding["manifest_sha256"] != manifest["manifest_sha256"]
                    or binding["checkpoint_sha256"] != manifest["checkpoint_sha256"] or binding["preprocess_sha256"] != manifest["preprocess_sha256"]
                    or manifest["label_contract"] != bundle["target"] or manifest["available_at"] != bundle["available_at"]):
                errors.append(_error("bundle_model_target_artifact_binding", path=str(directory), expert=name))
            expected = get_feature_profile("ma_trend_v1" if name == "ma" else "czsc_structure_v1") if name != "fusion" else None
            if expected:
                if (manifest["feature_version"], manifest["feature_schema_hash"], manifest["schema"]["columns"]) != (expected["version"], expected["schema_hash"], expected["columns"]):
                    errors.append(_error("bundle_independent_expert_schema", path=str(directory), expert=name))
            elif manifest["feature_version"] != "dual_probability_oof_v1" or manifest["schema"]["columns"] != ["p_ma", "p_structure"]:
                errors.append(_error("bundle_fusion_probability_schema", path=str(directory)))
            threshold = bundle["thresholds"][name]
            if (threshold["method"] != "validation_fixed_horizon_mean_net_return_proxy" or threshold["qualification"] != "fitted_validation_diagnostic_not_strategy_acceptance"
                    or not math.isfinite(float(threshold["threshold"])) or not 0 < threshold["threshold"] < 1):
                errors.append(_error("bundle_validation_only_frozen_threshold", path=str(directory), expert=name))
        oof_path = _bounded(bundle["oof"]["path"], directory)
        if _file_hash(oof_path) != bundle["oof"]["sha256"]:
            errors.append(_error("bundle_oof_physical_sha256", path=str(directory)))
        oof = pd.read_parquet(oof_path)
        errors.extend(_check_oof(oof, bundle["train_end"], fold=directory.name))
        if len(oof) != bundle["oof"]["rows"] or str(oof["label_end"].max()) != bundle["oof"]["label_end_max"]:
            errors.append(_error("bundle_oof_row_maturity_summary", path=str(directory)))
        if "fusion" in models and models["fusion"]["sample_counts"]["train"] != len(oof):
            errors.append(_error("fusion_training_only_mature_oof_rows", path=str(directory)))
        if "dataset_path" in bundle["fusion"]:
            fusion_path = _bounded(bundle["fusion"]["dataset_path"], directory)
            if _file_hash(fusion_path) != bundle["fusion"]["dataset_file_sha256"]:
                errors.append(_error("fusion_fit_dataset_physical_hash", path=str(directory)))
            fusion_data = pd.read_parquet(fusion_path)
            train_rows, _, _, _ = _training_masks(fusion_data, models["fusion"])
            try:
                pd.testing.assert_frame_equal(fusion_data.loc[train_rows].reset_index(drop=True), oof.reset_index(drop=True), check_dtype=False, check_exact=True)
            except AssertionError:
                errors.append(_error("fusion_fit_rows_are_only_recorded_oof", path=str(directory)))
        for item in bundle["oof"]["folds"]:
            split = item["fold"]
            cached_path = _bounded(item["rows_path"], root / "models/oof-shared")
            header = _read(cached_path.parent / "cache.json")
            unsigned_header = dict(header)
            cache_hash = unsigned_header.pop("cache_sha256", None)
            evidence = {key: value for key, value in item.items() if key != "reused"}
            if header != evidence or cache_hash != _hash(unsigned_header) or cached_path.parent.name != header["identity"] or _file_hash(cached_path) != header["rows_sha256"]:
                errors.append(_error("oof_shared_cache_artifact_binding", path=str(directory), fold=split["name"]))
            cached = pd.read_parquet(cached_path)
            rows = oof.loc[oof["oof_fold"] == split["name"]]
            if rows.empty or not ((rows["expert_train_end"] == split["train_end"]) & (rows["expert_validation_end"] == split["validation_end"]) & (rows["expert_available_at"] == split["validation_end"])
                                  & (rows["date"].astype(str) >= split["test_start"]) & (rows["date"].astype(str) <= split["test_end"])).all():
                errors.append(_error("oof_recorded_fold_boundaries", path=str(directory), fold=split["name"]))
            expected_rows = cached.loc[cached["label_available"].astype(bool) & (pd.to_datetime(cached["label_end"]) <= pd.Timestamp(bundle["train_end"]))]
            try:
                pd.testing.assert_frame_equal(rows.reset_index(drop=True), expected_rows.reset_index(drop=True), check_dtype=False, check_exact=True)
            except AssertionError:
                errors.append(_error("oof_shared_forward_scores_only_mature_subset", path=str(directory), fold=split["name"]))
            for name, embedded in item["experts"].items():
                manifest_path = cached_path.parent / "experts" / name / "manifest.json"
                checked = _check_model(manifest_path, root)
                errors.extend(checked.pop("errors"))
                actual = checked.get("manifest", {})
                if (actual.get("manifest_sha256") != embedded.get("manifest_sha256") or actual.get("available_at") != split["validation_end"]
                        or actual.get("train_end") != split["train_end"] or actual.get("label_contract") != bundle["target"]):
                    errors.append(_error("oof_expert_embedded_chronological_binding", path=str(directory), fold=split["name"], expert=name))
        if set(oof["oof_fold"]) != {item["fold"]["name"] for item in bundle["oof"]["folds"]}:
            errors.append(_error("oof_complete_declared_folds", path=str(directory)))
        if training is not None:
            target = training[["symbol", "date", "label", "label_end"]].copy()
            target["date"] = pd.to_datetime(target["date"]).dt.strftime("%Y-%m-%d")
            target["label_end"] = pd.to_datetime(target["label_end"], errors="coerce").dt.strftime("%Y-%m-%d")
            observed = oof[["symbol", "date", "label", "label_end"]].copy()
            observed["date"] = pd.to_datetime(observed["date"]).dt.strftime("%Y-%m-%d")
            observed["label_end"] = pd.to_datetime(observed["label_end"], errors="coerce").dt.strftime("%Y-%m-%d")
            joined = observed.merge(target, on=["symbol", "date"], how="left", suffixes=("_oof", "_source"), validate="one_to_one")
            if len(joined) != len(oof) or not ((joined["label_oof"] == joined["label_source"]) & (joined["label_end_oof"] == joined["label_end_source"])).all():
                errors.append(_error("oof_targets_match_frozen_training_snapshot", path=str(directory)))
        return {"path": str(directory), "available_at": bundle["available_at"], "oof_rows": len(oof),
                "oof_blocks": len(bundle["oof"]["folds"]), "model_count": len(models) + 2 * len(bundle["oof"]["folds"]),
                "bundle_sha256": recorded, "errors": errors}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"path": str(path), "errors": [*errors, _error("bundle_read", path=str(path), error=str(exc))]}


def _audit_evaluation(evaluation: dict, root: Path, frozen: list[str], initial: float, calendar: list[str], pool) -> dict:
    errors = []
    fold = evaluation["fold"]
    following = dict(zip(calendar[:-1], calendar[1:]))
    unavailable = {item["symbol"] for item in evaluation["unavailable_accounts_cash_retained"]}
    if evaluation["frozen_mainboard_accounts"] != len(frozen) or not unavailable.issubset(frozen):
        errors.append(_error("fold_complete_cash_denominator", fold=fold["name"]))
    modes = {}
    for mode, summary in evaluation["variants"].items():
        location = root / "evaluation" / fold["name"] / mode
        present = [symbol for symbol in frozen if (location / symbol.replace(".", "_") / "daily.csv").is_file()]
        actual = [path.name.replace("_", ".") for path in location.iterdir() if path.is_dir()] if location.is_dir() else []
        if set(frozen) - set(present) != unavailable or set(actual) != set(present):
            errors.append(_error("fold_all_accounts_or_explicit_cash", fold=fold["name"], mode=mode))
        accounts = list(pool.map(lambda symbol: _account(location / symbol.replace(".", "_"), symbol, initial, following), present))
        for account in accounts:
            errors.extend({**item, "fold": fold["name"], "mode": mode} for item in account.pop("errors"))
        valid = [account for account in accounts if "_equity" in account]
        if len(valid) != summary["accounts"]:
            errors.append(_error("fold_account_summary", fold=fold["name"], mode=mode))
        round_trips = [_round_trip_metrics(_csv(location / symbol.replace(".", "_") / "ledger.csv")) for symbol in present]
        returns = [value for account in round_trips for value in account["net_returns"]]
        profits = [value for account in round_trips for value in account["net_pnl"]]
        try:
            aggregate = _csv(location / "aggregate_daily.csv")
            if not valid or aggregate.empty:
                raise ValueError("no complete account or aggregate")
            wide = pd.concat([account["_equity"] for account in valid], axis=1).sort_index().ffill().fillna(initial)
            expected = wide.sum(axis=1) + (len(frozen) - len(valid)) * initial
            observed_dates = pd.to_datetime(aggregate["date"]).dt.strftime("%Y-%m-%d")
            if list(observed_dates) != list(expected.index) or not _close(aggregate["equity"], expected):
                errors.append(_error("aggregate_all_failed_cash_retained", fold=fold["name"], mode=mode))
            metrics = summary["metrics"]
            capital = len(frozen) * initial
            recomputed = {"final_equity": float(expected.iloc[-1]), "total_return": float(expected.iloc[-1] / capital - 1),
                          "max_drawdown": float(-(expected / expected.cummax().clip(lower=capital) - 1).min()),
                          "fees": sum(account.get("fees", 0.) for account in valid), "trade_count": sum(account.get("ledger_rows", 0) for account in valid),
                          "completed_round_trips": len(returns), "trade_expectancy": float(np.mean(returns)) if returns else None,
                          "win_rate": float(np.mean(np.asarray(profits) > 0)) if profits else None}
            winners, losers = [value for value in profits if value > 0], [value for value in profits if value < 0]
            recomputed.update(wins=len(winners), losses=len(losers), breakeven=len(profits) - len(winners) - len(losers),
                              unclosed_positions=sum(account.get("final_shares", 0) > 0 for account in valid),
                              mean_net_pnl_CNY=float(np.mean(profits)) if profits else None,
                              payoff_ratio=float(np.mean(winners) / -np.mean(losers)) if winners and losers else None,
                              profit_factor=float(sum(winners) / -sum(losers)) if winners and losers else None)
            for name in recomputed:
                if name not in metrics:
                    errors.append(_error("aggregate_missing_net_trade_metric", fold=fold["name"], mode=mode, metric=name))
                elif recomputed[name] is None or metrics[name] is None:
                    if recomputed[name] != metrics[name]:
                        errors.append(_error("aggregate_net_trade_metric", fold=fold["name"], mode=mode, metric=name))
                elif name in {"final_equity", "fees"}:
                    if not _close(metrics[name], recomputed[name]):
                        errors.append(_error("aggregate_net_trade_metric", fold=fold["name"], mode=mode, metric=name))
                elif not np.isclose(metrics[name], recomputed[name], rtol=1e-8, atol=1e-10):
                    errors.append(_error("aggregate_net_trade_metric", fold=fold["name"], mode=mode, metric=name))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            recomputed = {}
            errors.append(_error("aggregate_read", fold=fold["name"], mode=mode, error=str(exc)))
        for account in valid:
            account.pop("_equity")
        modes[mode] = {"accounts": len(valid), "cash_retained_accounts": len(unavailable), "cash_retained_budget": initial * len(unavailable),
                       "daily_rows": sum(account.get("daily_rows", 0) for account in valid), "ledger_rows": sum(account.get("ledger_rows", 0) for account in valid),
                       "recomputed_metrics": recomputed}
    return {"name": fold["name"], "modes": modes, "errors": errors}


def _forced_proceeds(sample: dict, rows: list[dict], positions: dict, seasoned: np.ndarray, following: dict, settings: dict, sources: set) -> tuple:
    """Independent arithmetic replay of frozen flat-cost sell execution."""
    shares, proceeds, rejected = int(sample["actual_shares"]), 0., 0
    start = positions[sample["date"]]
    tick = lambda value: float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    for index in range(start + 1, len(rows)):
        row, previous = rows[index], rows[index - 1]
        reject = (seasoned[index] < settings["seasoned_bars"] or row["source"] not in sources or previous["source"] not in sources
                  or row["has_trade_price"] != 1 or previous["has_trade_price"] != 1 or row["volume"] <= 0 or row["amount"] <= 0
                  or following.get(previous["date"]) != row["date"] or row["date"] == sample["actual_entry_date"]
                  or row["open"] <= tick(previous["close"] * (1 - settings["mainboard_conservative_limit"])))
        capacity = int(previous["volume"] * settings["max_volume_participation"]) // 100 * 100
        quantity = min(shares, capacity)
        if reject or quantity <= 0:
            rejected += 1
            continue
        price = tick(row["open"] * (1 - settings["slippage"]))
        notional = quantity * price
        proceeds += notional - max(notional * settings["commission"], settings["min_commission"]) - notional * settings["stamp_tax"]
        shares -= quantity
        if shares == 0:
            return row["date"], proceeds, rejected
    return None, None, rejected


def _execution_prices(record: dict, root: Path, evaluations: list[dict], settings: dict, calendar: list[str]) -> list[dict]:
    """Verify ledger price, fee, source, band and capacity against frozen OHLCV."""
    errors = []
    symbol = record["symbol"]
    try:
        source = pd.read_parquet(record["path"], columns=["date", "open", "close", "volume", "amount", "has_trade_price", "source"])
        source["date"] = pd.to_datetime(source["date"]).dt.strftime("%Y-%m-%d")
        rows = source.to_dict("records")
        lookup = {row["date"]: index for index, row in enumerate(rows)}
        valid = np.asarray([row["volume"] > 0 and row["amount"] > 0 and row["has_trade_price"] == 1 for row in rows])
        seasoned = np.r_[0, np.cumsum(valid[:-1])]
        from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
        following = dict(zip(calendar[:-1], calendar[1:]))
        paths = [(root / "evaluation" / item["fold"]["name"] / mode / symbol.replace(".", "_") / "ledger.csv", item["fold"]["name"], mode)
                 for item in evaluations for mode in item["variants"]]
        reference = root / "evaluation/exit_reference" / symbol.replace(".", "_") / "ledger.csv"
        if reference.exists():
            paths.append((reference, "exit_reference", "rules"))
        tick = lambda value: float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        for path, fold, mode in paths:
            if not path.exists():
                continue  # Coverage checks separately require an explicit cash account.
            for trade in _csv(path).to_dict("records"):
                day = str(trade["date"])
                index = lookup.get(day)
                if index is None or index < 1:
                    errors.append(_error("execution_has_exact_frozen_bar", symbol=symbol, date=day, fold=fold, mode=mode))
                    continue
                row, previous = rows[index], rows[index - 1]
                buy = trade["action"] == "BUY"
                quantity = int(trade["shares"])
                price = tick(row["open"] * (1 + settings["slippage"] if buy else 1 - settings["slippage"]))
                notional = quantity * price
                fee = max(notional * settings["commission"], settings["min_commission"]) + (0. if buy else notional * settings["stamp_tax"])
                if not np.isclose(trade["price"], price, rtol=0., atol=1e-10) or not np.isclose(trade["fee"], fee, rtol=1e-10, atol=1e-10):
                    errors.append(_error("execution_actual_open_slippage_and_fees", symbol=symbol, date=day, fold=fold, mode=mode))
                boundary = tick(previous["close"] * (1 + settings["mainboard_conservative_limit"] if buy else 1 - settings["mainboard_conservative_limit"]))
                guard = (row["open"] >= boundary if buy else row["open"] <= boundary)
                if (guard or seasoned[index] < settings["seasoned_bars"] or following.get(previous["date"]) != day
                        or row["source"] not in VERIFIED_SOURCES or previous["source"] not in VERIFIED_SOURCES
                        or row["has_trade_price"] != 1 or previous["has_trade_price"] != 1 or row["volume"] <= 0 or row["amount"] <= 0):
                    errors.append(_error("execution_frozen_sell_buy_guards", symbol=symbol, date=day, fold=fold, mode=mode))
                if quantity > int(previous["volume"] * settings["max_volume_participation"]) // 100 * 100:
                    errors.append(_error("execution_prior_volume_capacity", symbol=symbol, date=day, fold=fold, mode=mode))
                if buy:
                    cash_before = float(trade["cash_after"]) - float(trade["cash_flow"])
                    equity_before = cash_before + float(trade["position_before"]) * row["open"]
                    if notional + fee > min(cash_before, equity_before * settings["max_weight"]) + 1e-8:
                        errors.append(_error("execution_cash_and_max_weight_budget", symbol=symbol, date=day, fold=fold, mode=mode))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(_error("execution_frozen_price_read", symbol=symbol, error=str(exc)))
    return errors


def _check_exit_reference(path: Path, frame: pd.DataFrame, raw: pd.DataFrame, initial: float, calendar: list[str], settings: dict) -> dict:
    """Reconstruct actual state and both exit outcomes from independent artifacts."""
    symbol = str(raw.iloc[0]["symbol"])
    errors = []
    try:
        checked = _account(path, symbol, initial, dict(zip(calendar[:-1], calendar[1:])))
        errors.extend(checked.pop("errors"))
        checked.pop("_equity", None)
        daily, ledger = _csv(path / "daily.csv"), _csv(path / "ledger.csv")
        observed = frame.loc[frame["symbol"] == symbol].copy()
        expected_dates = set(daily.loc[daily["shares"] > 0, "date"].astype(str))
        if set(observed["date"].astype(str)) != expected_dates:
            errors.append(_error("exit_every_actual_held_close_retained", symbol=symbol))
        states, shares, cost, entered, entry_shares = {}, 0, 0., None, 0
        trade_rows = ledger.to_dict("records")
        flows = np.cumsum([float(row["cash_flow"]) for row in trade_rows])
        future_flat = {}
        next_flat = None
        for index in range(len(trade_rows) - 1, -1, -1):
            if int(trade_rows[index]["position_after"]) == 0:
                next_flat = index
            future_flat[index] = next_flat
        seen = 0
        for day in daily.to_dict("records"):
            while seen < len(trade_rows) and trade_rows[seen]["date"] <= day["date"]:
                trade = trade_rows[seen]
                quantity = int(trade["shares"])
                if trade["action"] == "BUY":
                    cost = (shares * cost - float(trade["cash_flow"])) / (shares + quantity)
                    shares += quantity
                    entered, entry_shares = trade["date"], quantity
                else:
                    shares -= quantity
                    if shares == 0:
                        cost, entered, entry_shares = 0., None, 0
                seen += 1
            states[str(day["date"])] = {"shares": shares, "cost": cost, "entry_date": entered, "entry_shares": entry_shares, "seen": seen, **day}
        rows = raw[["date", "open", "close", "volume", "amount", "has_trade_price", "source"]].copy()
        rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
        rows = rows.to_dict("records")
        positions = {row["date"]: index for index, row in enumerate(rows)}
        sources = set(raw.attrs.get("verified_sources", []))
        if not sources:
            from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
            sources = set(VERIFIED_SOURCES)
        valid = np.asarray([row["volume"] > 0 and row["amount"] > 0 and row["has_trade_price"] == 1 and row["source"] in sources for row in rows])
        seasoned = np.r_[0, np.cumsum(valid[:-1])]
        following = dict(zip(calendar[:-1], calendar[1:]))
        for sample in observed.to_dict("records"):
            day = str(sample["date"])
            state = states.get(day)
            if state is None or (sample["actual_shares"] != state["shares"] or not _close(sample["actual_cost_per_share"], state["cost"])
                                 or sample["actual_entry_date"] != state["entry_date"] or sample["ledger_entries_seen"] != state["seen"]):
                errors.append(_error("exit_actual_state_matches_prior_ledger", symbol=symbol, date=day))
                continue
            expected = {"holding_return": state["close"] / state["cost"] - 1,
                        "position_weight": state["shares"] * state["close"] / state["equity"],
                        "cash_weight": state["cash"] / state["equity"], "remaining_fraction": state["shares"] / state["entry_shares"],
                        "holding_bars": positions[day] - positions[state["entry_date"]]}
            for key, value in expected.items():
                if key in sample and not np.isclose(sample[key], value, rtol=1e-8, atol=1e-10):
                    errors.append(_error("exit_feature_matches_contemporaneous_state", symbol=symbol, date=day, feature=key))
            flat = future_flat.get(state["seen"])
            continue_end = trade_rows[flat]["date"] if flat is not None else None
            continue_proceeds = float(flows[flat] - flows[state["seen"] - 1]) if flat is not None else None
            forced_end, forced, rejections = _forced_proceeds(sample, rows, positions, seasoned, following, settings, sources) if continue_end else (None, None, 0)
            matured = continue_end is not None and forced_end is not None
            if bool(sample["label_available"]) != matured:
                errors.append(_error("exit_all_outcomes_availability_replayed", symbol=symbol, date=day))
            elif matured and (sample["continuation_exit_date"] != continue_end or sample["forced_exit_date"] != forced_end
                             or not _close(sample["continuation_net_proceeds"], continue_proceeds) or not _close(sample["forced_net_proceeds"], forced)
                             or int(sample["forced_rejection_count"]) != rejections):
                errors.append(_error("exit_two_path_net_proceeds_replayed", symbol=symbol, date=day))
        return {"symbol": symbol, "held_samples": len(observed), "reference_ledger_rows": len(ledger), "errors": errors}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"symbol": symbol, "held_samples": 0, "errors": [*errors, _error("exit_reference_read", symbol=symbol, error=str(exc))]}


def _check_exit_export(directory: Path, binding: dict, manifest: dict, root: Path) -> list[dict]:
    errors = []
    try:
        directory = _bounded(directory, root)
        header = _read(directory / "head.binding.json")
        unsigned = dict(header)
        recorded = unsigned.pop("binding_sha256", None)
        exported_path = directory / "exit-linear.json"
        exported = _read(exported_path)
        if (recorded != stable_hash(unsigned) or recorded != binding["head_binding_sha256"] or header.get("head") != "holding_exit"
                or header.get("model_manifest_sha256") != manifest["manifest_sha256"] or header.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
                or header.get("contract_hash") != manifest["label_contract"]["contract_hash"] or header.get("available_at") != manifest["available_at"]
                or header.get("runtime") != "cpu_numpy" or header.get("export") != "exit-linear.json"):
            errors.append(_error("exit_linear_export_head_binding", path=str(directory)))
        if (header.get("export_sha256") != _file_hash(exported_path)
                or binding.get("head.binding.json_sha256") != _file_hash(directory / "head.binding.json")
                or binding.get("exit-linear.json_sha256") != _file_hash(exported_path)
                or binding.get("manifest_file_sha256") != _file_hash(directory / "manifest.json")
                or binding.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
                or binding.get("target_hash") != _hash(manifest["label_contract"])):
            errors.append(_error("exit_linear_export_physical_artifacts", path=str(directory)))
        if (exported.get("format") != "holding_exit_linear_cpu_numpy_v1" or exported.get("columns") != manifest["schema"]["columns"]
                or exported.get("feature_schema_hash") != manifest["feature_schema_hash"] or exported.get("feature_version") != manifest["feature_version"]
                or exported.get("contract_hash") != manifest["label_contract"]["contract_hash"] or exported.get("preprocess") != manifest["preprocess"]
                or exported.get("temperature") != manifest["calibration"]["temperature"]
                or len(exported.get("weight", [])) != len(manifest["schema"]["columns"])
                or not np.isfinite([*exported["weight"], exported["bias"], exported["temperature"]]).all() or exported["temperature"] <= 0):
            errors.append(_error("exit_linear_export_semantic_identity", path=str(directory)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(_error("exit_linear_export_read", path=str(directory), error=str(exc)))
    return errors


def _training_masks(dataset: pd.DataFrame, manifest: dict, eligible: np.ndarray | None = None):
    dates = pd.to_datetime(dataset["date"]).dt.strftime("%Y-%m-%d")
    ends = pd.to_datetime(dataset["label_end"], errors="coerce").dt.strftime("%Y-%m-%d")
    qualification = dataset["input_eligible"].fillna(False).to_numpy(dtype=bool) if eligible is None else eligible
    available = dataset["label_available"].fillna(False).to_numpy(dtype=bool) & qualification & dataset["label"].isin([0, 1]).to_numpy() & ends.notna().to_numpy()
    train = available & (dates <= manifest["train_end"]).to_numpy() & (ends <= manifest["train_end"]).to_numpy() & (ends < manifest["validation_start"]).to_numpy()
    validation = available & (dates >= manifest["validation_start"]).to_numpy() & (dates <= manifest["validation_end"]).to_numpy() & (ends <= manifest["validation_end"]).to_numpy()
    return train, validation, dates, ends


def _check_training_preprocessing(dataset: pd.DataFrame, manifest: dict, *, eligible=None, cache=None, verify_validation=True) -> list[dict]:
    """Refit preprocessing independently using mature training rows only."""
    errors = []
    train, validation, dates, ends = _training_masks(dataset, manifest, eligible)
    identity = (manifest["feature_version"], manifest["train_end"], manifest.get("data_version"))
    counts = manifest["sample_counts"]
    if int(train.sum()) != counts["train"] or verify_validation and int(validation.sum()) != counts["validation"]:
        errors.append(_error("model_sample_counts_independent_purging", checkpoint=manifest["checkpoint"]))
    if not train.any() or verify_validation and not validation.any():
        return [*errors, _error("model_mature_samples_unavailable", checkpoint=manifest["checkpoint"])]
    if verify_validation:
        fitted_hash = hashlib.sha256(pd.util.hash_pandas_object(dataset.loc[train | validation], index=True).to_numpy().tobytes()).hexdigest()
        if fitted_hash != manifest.get("dataset_sha256"):
            errors.append(_error("model_exact_training_validation_dataset_hash", checkpoint=manifest["checkpoint"]))
    expected = {"train_signal_start": dates.loc[train].min(), "train_signal_end": dates.loc[train].max(),
                "train_label_end": ends.loc[train].max()}
    if verify_validation:
        expected["validation_label_end"] = ends.loc[validation].max()
    for field, value in expected.items():
        if manifest.get(field) != value:
            errors.append(_error("model_independent_signal_label_cutoff", checkpoint=manifest["checkpoint"], field=field))
    if cache is None or identity not in cache:
        values = dataset.loc[train, manifest["schema"]["columns"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
        values[~np.isfinite(values)] = np.nan
        missing = np.isnan(values).all(axis=0)
        medians = np.asarray([0. if absent else np.nanmedian(values[:, index]) for index, absent in enumerate(missing)])
        filled = np.where(np.isnan(values), medians, values)
        scales = filled.std(axis=0)
        scales[scales < 1e-8] = 1.
        fitted = {"median": medians, "mean": filled.mean(axis=0), "scale": scales, "all_missing": missing}
        if cache is not None:
            cache[identity] = fitted
    else:
        fitted = cache[identity]
    preprocessing = manifest["preprocess"]
    if preprocessing.get("fit_scope") != "purged_training_only" or preprocessing.get("clip_standard_deviations") != 8.:
        errors.append(_error("model_training_only_preprocess_scope", checkpoint=manifest["checkpoint"]))
    for field, expected_values in fitted.items():
        if len(preprocessing.get(field, [])) != len(expected_values) or not np.allclose(preprocessing.get(field, []), expected_values, rtol=1e-10, atol=1e-12):
            errors.append(_error("model_training_only_preprocess_refit", checkpoint=manifest["checkpoint"], field=field))
    return errors


def _structure_eligibility(frame: pd.DataFrame) -> np.ndarray:
    """Independent vector expression of currently observable structure inputs."""
    core = [f"bi{i}_{field}" for i in (1, 2, 3) for field in ("direction", "return", "length")]
    now = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    confirmed = pd.to_datetime(frame["structure_confirmed_at"], utc=True, errors="coerce")
    eligible = (frame["input_eligible"].fillna(False).to_numpy(dtype=bool)
                & np.isfinite(frame[core].to_numpy(dtype=float)).all(axis=1)
                & (pd.to_numeric(frame["finished_bi_count"], errors="coerce").to_numpy() >= 3)
                & now.notna().to_numpy() & confirmed.notna().to_numpy() & (confirmed <= now).to_numpy())
    for field in ("zone_confirmed_at", "weekly_available_at", "monthly_available_at"):
        present = frame[field].notna().to_numpy()
        visible = pd.to_datetime(frame[field], utc=True, errors="coerce")
        eligible &= ~present | (visible.notna().to_numpy() & (visible <= now).to_numpy())
    return eligible


def _training_projection(record: dict, training: pd.DataFrame, feature_start: str) -> list[dict]:
    symbol = record["symbol"]
    try:
        source = pd.read_parquet(record["path"], columns=list(training.columns))
        dates = pd.to_datetime(source["date"]).dt.strftime("%Y-%m-%d")
        expected = source.loc[(dates >= feature_start) & source["label_available"].astype(bool)].reset_index(drop=True)
        observed = training.loc[training["symbol"] == symbol].reset_index(drop=True)
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False, check_exact=True, check_index_type=False)
        return []
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        return [_error("training_rows_equal_frozen_source_projection", symbol=symbol, error=str(exc)[:500])]


def _completed_dual_identity(report: dict, metadata: dict, run_id: str) -> bool:
    """API completion and TaskManager success are both persisted final states."""
    return (report.get("run_id") == metadata.get("run_id") == run_id and report.get("status") == "complete"
            and metadata.get("status") in {"complete", "succeeded"})


def audit_dual_run(training_run_id: str, *, runs_root: Path | None = None, workers: int = 4) -> dict:
    """Audit a completed isolated research run; never create a release event."""
    if not 1 <= workers <= 8:
        raise ValueError("workers must be within 1..8")
    base = (runs_root or ARTIFACT_RUNS_ROOT).resolve()
    root = _run(base, training_run_id)
    report = _read(_bounded(root / "reports/dual-research.json", root))
    metadata = _read(_bounded(root / "metadata.json", root))
    errors = []
    if not _completed_dual_identity(report, metadata, training_run_id):
        errors.append(_error("completed_dual_run_identity"))
    if report.get("publication_allowed") is not False or metadata.get("publication_allowed") is not False or report["model_gate"].get("passed") is not False or report.get("observed_windows_diagnostic_only") is not True:
        errors.append(_error("shadow_historical_diagnostics_never_released"))
    config = report["config"]
    if metadata["config"]["research"] != config or metadata.get("config_hash") != stable_hash(metadata["config"]):
        errors.append(_error("frozen_dual_configuration"))
    strategy = _read(root / "config/czsc_strategy.json")
    if metadata["config"]["strategy"] != strategy or _read(root / "config/czsc_dual_research.json") != config:
        errors.append(_error("frozen_strategy_configuration"))
    snapshot = _read(root / "dataset/source-references.json")
    if snapshot != report["source_snapshot"] or snapshot.get("strategy_hash") != stable_hash(strategy) or report.get("source_artifacts_unchanged") is not True:
        errors.append(_error("source_reference_report_binding"))
    source_report_path = _bounded(snapshot["source_report_path"], base)
    source_report = _read(source_report_path)
    if _file_hash(source_report_path) != snapshot["source_report_sha256"] or snapshot["source_run_id"] != source_report["run_id"] or snapshot["all_source_records"] != source_report["dataset_records"]:
        errors.append(_error("original_source_report_identity_sha256"))
    source_records = snapshot["all_source_records"]
    all_symbols = [item["symbol"] for item in source_records]
    selected = snapshot["selected_symbols"]
    if (len(all_symbols) != len(set(all_symbols)) or len(selected) != len(set(selected)) or not set(selected).issubset(all_symbols)
            or snapshot["selected_count"] != len(selected) or snapshot["source_universe_count"] != len(all_symbols)
            or snapshot["selected_full_source_universe"] != (set(selected) == set(all_symbols))):
        errors.append(_error("frozen_complete_requested_universe"))
    if metadata.get("stocks") != selected or report["run_context"].get("stocks") != selected:
        errors.append(_error("requested_stocks_enter_run_lineage"))
    records = report["dataset_records"]
    if {item["symbol"] for item in records} != set(selected) or len(records) != len(selected) or report["coverage"].get("requested") != len(selected) or report["coverage"].get("success") != len(records) or report["coverage"].get("failed") != 0:
        errors.append(_error("selected_dataset_complete_coverage"))
    original = {item["symbol"]: item for item in source_records}
    for record in records:
        baseline = original.get(record["symbol"], {})
        if any(record.get(key) != value for key, value in baseline.items()) or record.get("source_run_id") != snapshot["source_run_id"]:
            errors.append(_error("selected_source_record_unchanged", symbol=record["symbol"]))
    calendar_path = _bounded(report["calendar"]["path"], root)
    calendar_value = _read(calendar_path)
    calendar = calendar_value["dates"]
    if (not calendar_value.get("verified") or not calendar_value.get("source") or calendar != sorted(set(calendar))
            or stable_hash(calendar_value) != report["calendar"]["hash"] or calendar != report["market_dates"]
            or stable_hash(calendar) != source_report["model_manifest"]["label_contract"]["calendar_hash"]):
        errors.append(_error("verified_calendar_and_ten_day_target_binding"))
    source_calendar = _bounded(report["calendar"]["source_path"], base)
    if _file_hash(source_calendar) != report["calendar"]["source_file_sha256"] or _read(source_calendar) != calendar_value:
        errors.append(_error("source_calendar_unchanged"))
    independent = report.get("independent_window", {})
    if independent != config.get("independent_window") or independent.get("start", "") <= report["data_end"] or independent.get("status") != "future_unobserved":
        errors.append(_error("future_independent_window_not_backfilled"))
    training = pd.read_parquet(root / "dataset/training.parquet")
    frozen = [symbol for symbol in selected if symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ")]
    if (training[["symbol", "date"]].duplicated().any() or set(training["symbol"]) - set(frozen) or len(training) != report["training_rows"]
            or not training["label_available"].all() or training.attrs.get("label_contract") != source_report["model_manifest"]["label_contract"]):
        errors.append(_error("training_frozen_mature_mainboard_targets"))
    result = {"training_run_id": training_run_id, "audited_at": local_now().isoformat(), "read_only_inputs": True,
              "model_payload_check": "hash_only_without_torch_deserialization", "source_report_sha256": snapshot["source_report_sha256"],
              "source_universe_count": len(all_symbols), "selected_symbols": len(selected), "frozen_mainboard_accounts": len(frozen),
              "training_rows": len(training), "training_parquet_sha256": _file_hash(root / "dataset/training.parquet"), "errors": errors}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dual-independent-audit") as pool:
        datasets = list(pool.map(lambda record: _source_record(record, base), source_records))
        for item in datasets:
            errors.extend(item.pop("errors"))
        result["source_dataset"] = {"files": len(datasets), "bars": sum(item["rows"] for item in datasets),
                                    "input_eligible_rows": sum(item["input_eligible_rows"] for item in datasets),
                                    "model_input_eligible_rows": sum(item["model_input_eligible_rows"] for item in datasets),
                                    "label_available_rows": sum(item["label_available_rows"] for item in datasets)}
        training_indices = training.groupby("symbol", sort=False).indices
        projections = list(pool.map(lambda record: _training_projection(record, training.iloc[training_indices.get(record["symbol"], [])], config["feature_start"]), [item for item in records if item["symbol"] in frozen]))
        for items in projections:
            errors.extend(items)
        bundles = [_check_bundle(Path(item["path"]), root, training) for item in report["model_bundles"]]
        for bundle in bundles:
            errors.extend(bundle.pop("errors"))
        result["bundles"] = bundles
        ma_eligible = training["model_input_eligible"].fillna(False).to_numpy(dtype=bool)
        structural_eligible = _structure_eligibility(training)
        preprocess_cache = {}
        checkpoint_files = sorted((root / "models").rglob("model.pt"))
        checked_models = []
        for checkpoint in checkpoint_files:
            manifest = _read(checkpoint.parent / "manifest.json")
            if manifest["feature_version"] in {"ma_trend_v1", "czsc_structure_v1"}:
                eligible = ma_eligible if manifest["feature_version"] == "ma_trend_v1" else structural_eligible
                columns = list(dict.fromkeys([*manifest["schema"]["columns"], "date", "symbol", "label", "label_end", "label_available", "net_return", "input_eligible"]))
                fit_inputs = training[columns].copy()
                fit_inputs["input_eligible"] = eligible
                errors.extend(_check_training_preprocessing(fit_inputs, manifest, cache=preprocess_cache))
            elif manifest["feature_version"] == "dual_probability_oof_v1":
                fusion_frame = pd.read_parquet(checkpoint.parent.parent / "fusion-training.parquet")
                errors.extend(_check_training_preprocessing(fusion_frame, manifest, cache=preprocess_cache))
            elif manifest["label_version"] != "czsc_actual_holding_exit_advantage_v1":
                errors.append(_error("unexpected_model_weight_contract", checkpoint=str(checkpoint)))
            checked_models.append({"checkpoint": str(checkpoint), "training_batches": manifest["training_batches"],
                                   "device": manifest["device"], "feature_version": manifest["feature_version"]})
        if len(checked_models) != report["compute_info"]["trained_models"] or sum(item["training_batches"] for item in checked_models) != report["compute_info"]["training_batches"]:
            errors.append(_error("actual_unique_model_and_training_batch_counts"))
        result["unique_models"] = checked_models
        expected_names = [item["name"] for item in config["development_windows"]] + ["production"]
        if [item["name"] for item in report["model_bundles"]] != expected_names:
            errors.append(_error("all_frozen_window_and_production_bundles"))
        if [item["fold"] for item in report["evaluation"]] != config["development_windows"]:
            errors.append(_error("all_frozen_development_windows_evaluated"))
        for evaluation in report["evaluation"]:
            if set(evaluation["variants"]) != {"rules", "ma_only", "structure_only", "fusion", "rules_exit", "fusion_exit"}:
                errors.append(_error("six_frozen_entry_exit_comparison_arms", fold=evaluation["fold"]["name"]))
            if evaluation.get("bootstrap_contract", {}).get("dependence_bars") != config.get("ledger_dependence_bars", 120) or evaluation.get("bootstrap_contract", {}).get("blocks") != config.get("ledger_bootstrap_blocks", [160, 240]):
                errors.append(_error("variable_holding_ledger_bootstrap_contract", fold=evaluation["fold"]["name"]))
        evaluations = [_audit_evaluation(item, root, frozen, float(config["initial_cash"]), calendar, pool) for item in report["evaluation"]]
        for item in evaluations:
            errors.extend(item.pop("errors"))
        result["evaluation"] = evaluations
        price_checks = list(pool.map(lambda record: _execution_prices(record, root, report["evaluation"], strategy["execution"], calendar), [item for item in records if item["symbol"] in frozen]))
        for items in price_checks:
            errors.extend(items)
        result["actual_open_cost_and_guard_accounts"] = len(price_checks)
        exit_path = root / "dataset/exit-training.parquet"
        if not exit_path.is_file():
            errors.append(_error("independent_exit_training_dataset_missing"))
        else:
            exits = pd.read_parquet(exit_path)
            errors.extend(_check_exit_dataset(exits))
            contract = exits.attrs.get("label_contract", {})
            content = {key: value for key, value in contract.items() if key != "contract_hash"}
            if (contract.get("contract_hash") != stable_hash(content) or contract.get("head") != "holding_exit"
                    or contract.get("label_version") != "czsc_actual_holding_exit_advantage_v1"
                    or contract.get("strategy_hash") != stable_hash(strategy) or contract.get("calendar_hash") != stable_hash(calendar)):
                errors.append(_error("independent_exit_contract_hash_strategy_calendar"))
            if set(exits["symbol"]) - set(frozen):
                errors.append(_error("exit_samples_remain_in_frozen_accounts"))
            exit_models = []
            for item in report["model_bundles"]:
                model_directory = _bounded(item["path"], root)
                bundle = _read(model_directory / "bundle.json")
                binding = bundle.get("exit_head", {})
                if binding != item.get("exit_model"):
                    errors.append(_error("exit_head_embedded_bundle_report_binding", model=item["name"]))
                if binding.get("unavailable"):
                    if not binding.get("rules_exit_retained") or (model_directory / "exit/model.pt").exists():
                        errors.append(_error("exit_unavailable_keeps_rule_arms_no_weights", model=item["name"]))
                    fold = next((fold for fold in config["development_windows"] if fold["name"] == item["name"]), config["production"])
                    unavailable_manifest = {**fold}
                    train, validation, _, _ = _training_masks(exits, unavailable_manifest)
                    minimum_train, minimum_validation = 100, 30
                    feasible = (train.sum() >= minimum_train and validation.sum() >= minimum_validation
                                and exits.loc[train, "label"].nunique() == 2 and exits.loc[validation, "label"].nunique() == 2)
                    if feasible:
                        errors.append(_error("exit_unavailable_reason_matches_mature_sample_shortage", model=item["name"]))
                    exit_models.append({"model": item["name"], "unavailable": binding["unavailable"], "train_rows": int(train.sum()), "validation_rows": int(validation.sum())})
                    continue
                checked = _check_model(model_directory / "exit/manifest.json", root)
                errors.extend(checked.pop("errors"))
                manifest = checked.pop("manifest", {})
                if manifest.get("label_contract") != contract or manifest.get("label_version") != contract.get("label_version") or manifest.get("feature_schema_hash") != exits.attrs.get("schema_hash"):
                    errors.append(_error("exit_head_independent_target_binding", model=item["name"]))
                errors.extend(_check_exit_export(model_directory / "exit", binding, manifest, root))
                errors.extend(_check_training_preprocessing(exits, manifest, cache=preprocess_cache))
                exit_models.append(checked)
            exit_indices = exits.groupby("symbol", sort=False).indices
            def audit_reference(record):
                raw = pd.read_parquet(record["path"])
                return _check_exit_reference(root / "evaluation/exit_reference" / record["symbol"].replace(".", "_"), exits.iloc[exit_indices.get(record["symbol"], [])], raw,
                                             float(config["initial_cash"]), calendar, strategy["execution"])
            references = list(pool.map(audit_reference, [item for item in records if item["symbol"] in frozen]))
            for item in references:
                errors.extend(item.pop("errors"))
            result["exit_head"] = {"training_rows": len(exits), "label_available_rows": int(exits["label_available"].sum()),
                                   "label_reason_counts": {str(key): int(value) for key, value in exits["label_reason"].value_counts().items()},
                                   "contract_hash": contract.get("contract_hash"), "sha256": _file_hash(exit_path), "models": exit_models,
                                   "reference_accounts": len(references), "reference_held_samples": sum(item["held_samples"] for item in references),
                                   "reference_ledger_rows": sum(item.get("reference_ledger_rows", 0) for item in references)}
    result["error_count"] = len(errors)
    result["passed"] = not errors
    output = _bounded(root / "reports/dual-execution-audit.json", root)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    result = audit_dual_run(args.training_run_id, runs_root=args.runs_root, workers=args.workers)
    print(json.dumps({"training_run_id": args.training_run_id, "passed": result["passed"], "error_count": result["error_count"],
                      "source_files": result["source_dataset"]["files"], "source_bars": result["source_dataset"]["bars"],
                      "training_rows": result["training_rows"], "bundles": len(result["bundles"]),
                      "report": str((args.runs_root or ARTIFACT_RUNS_ROOT) / args.training_run_id / "reports/dual-execution-audit.json")}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
