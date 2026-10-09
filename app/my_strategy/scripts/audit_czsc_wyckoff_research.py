"""Read-only independent Wyckoff artifact, cash, target, and chronology audit.

Audit arithmetic comes from the standalone raw-ledger audit, never the trainer.
Linear weight storage is read as inert float data without Torch or unpickling.
The only output is a separate audit report; no model is executed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now
from my_strategy.scripts.audit_czsc_research import _account, _csv, _file_hash, _read, _error, _close, _hash
from my_strategy.scripts.audit_czsc_dual_research import (
    _check_model, _check_training_preprocessing, _check_exit_dataset,
    _check_exit_reference, _execution_prices, _round_trip_metrics,
)
from my_strategy.services.czsc_wyckoff_features import W_INPUT_COLUMNS, WYCKOFF_SCHEMA_HASH, prepare_wyckoff_inputs
from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS, STRUCTURE_COLUMNS

ARMS = ("rules", "ma_only", "structure_only", "fusion", "rules_exit", "fusion_exit", "wyckoff_rule",
        "wyckoff_only", "three_fusion", "fusion_wyckoff_exit", "three_fusion_exit", "policy_fresh", "union_rules", "policy_union")
MARKET_INPUT_COLUMNS = (*MA_TREND_COLUMNS, *STRUCTURE_COLUMNS, *W_INPUT_COLUMNS)


def _mainboard(symbol):
    return symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ")


def check_oof(frame, cutoff):
    errors = []
    required = {"date", "symbol", "label", "label_end", "expert_train_end", "expert_validation_end", "expert_available_at",
                "p_ma", "p_structure", "p_wyckoff", "target_label_end", "oof_fold"}
    if not required <= set(frame):
        return [_error("three_expert_oof_provenance", missing=sorted(required - set(frame)))]
    dates = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    ends = pd.to_datetime(frame.label_end).dt.strftime("%Y-%m-%d")
    chronology = ((frame.expert_train_end.astype(str) < frame.expert_validation_end.astype(str))
                  & frame.expert_validation_end.astype(str).eq(frame.expert_available_at.astype(str))
                  & (frame.expert_available_at.astype(str) < dates) & (ends > dates) & (ends <= cutoff))
    if not chronology.all():
        errors.append(_error("strict_forward_oof_mature_target", invalid_rows=int((~chronology).sum())))
    if frame.empty or frame.duplicated(["date", "symbol"]).any():
        errors.append(_error("oof_nonempty_unique_signal_identity"))
    values = frame[["p_ma", "p_structure", "p_wyckoff"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any() or not frame.label.isin([0, 1]).all():
        errors.append(_error("oof_real_three_probabilities_same_binary_target"))
    if not frame.target_label_end.astype(str).eq(frame.label_end.astype(str)).all():
        errors.append(_error("oof_target_maturity_identity"))
    return errors


def check_policy_rows(frame):
    errors = []
    required = {"date", "candidate_id", "behavior_cash", "behavior_shares", "clone_initial_cash", "clone_ledger_hash",
                "label", "label_available", "label_end", "net_return", "entry_date", "exit_date", "label_reason",
                "entry_shares", "state_observed_at", "candidate_cash_ratio", "candidate_account_return"}
    if not required <= set(frame):
        return [_error("policy_all_candidate_actual_state_provenance", missing=sorted(required - set(frame)))]
    if frame.duplicated(["symbol", "date", "candidate_id"]).any():
        errors.append(_error("policy_unique_frozen_candidate"))
    available = frame.label_available.fillna(False).astype(bool)
    if not np.array_equal(available, frame.label.notna()) or not frame.loc[available, "label"].isin([0, 1]).all():
        errors.append(_error("policy_rejected_censored_not_negative"))
    selected = frame.loc[available]
    if (not selected.behavior_shares.eq(0).all() or not selected.entry_shares.gt(0).all()
            or not (selected.entry_date.astype(str) > selected.date.astype(str)).all()
            or not selected.exit_date.astype(str).eq(selected.label_end.astype(str)).all()
            or not (selected.exit_date.astype(str) > selected.entry_date.astype(str)).all()
            or not np.array_equal(selected.label.to_numpy(dtype=float), selected.net_return.gt(0).to_numpy(dtype=float))):
        errors.append(_error("policy_flat_actual_complete_fee_roundtrip_target"))
    if not _close(frame.behavior_cash, frame.clone_initial_cash):
        errors.append(_error("policy_clone_exact_observed_cash"))
    if not frame.state_observed_at.astype(str).str[:10].eq(frame.date.astype(str).str[:10]).all():
        errors.append(_error("policy_contemporaneous_state_observation"))
    return errors


def check_linear_export_storage(checkpoint, exported):
    """Compare CPU exports to immutable float32 checkpoint storage, no pickle.

    The frozen first-stage heads have one weight matrix and one scalar bias.
    PyTorch's ZIP archive stores these two tensors separately. Unknown formats
    fail this audit instead of trusting a self-consistent exported JSON hash.
    """
    errors = []
    try:
        with zipfile.ZipFile(checkpoint) as archive:
            order_names = [name for name in archive.namelist() if name.endswith("/byteorder")]
            tensor_names = [name for name in archive.namelist() if "/data/" in name and name.rsplit("/", 1)[-1].isdigit()]
            if len(order_names) != 1 or len(tensor_names) != 2:
                raise ValueError("not the frozen two-storage linear checkpoint format")
            byteorder = archive.read(order_names[0]).decode("ascii")
            if byteorder not in {"little", "big"}:
                raise ValueError("unknown checkpoint byte order")
            tensors = [np.frombuffer(archive.read(name), dtype="<f4" if byteorder == "little" else ">f4") for name in tensor_names]
        width = len(exported["columns"])
        weights = [array for array in tensors if len(array) == width]
        biases = [array for array in tensors if len(array) == 1]
        if len(weights) != 1 or len(biases) != 1 or not np.array_equal(weights[0], np.asarray(exported["weight"], dtype=np.float32)) or biases[0][0] != np.float32(exported["bias"]):
            errors.append(_error("cpu_export_exact_original_linear_checkpoint_float_storage", path=str(checkpoint)))
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        errors.append(_error("cpu_export_checkpoint_storage_read", path=str(checkpoint), error=str(exc)))
    return errors


def check_market_inputs(samples, source):
    """Every market feature must equal its frozen symbol/date observation."""
    errors = []
    try:
        frozen = source.copy()
        frozen["date"] = pd.to_datetime(frozen.date).dt.strftime("%Y-%m-%d")
        if frozen.duplicated(["date", "symbol"]).any():
            raise ValueError("non-unique frozen market observation")
        frozen = frozen.set_index(["date", "symbol"])
        keys = pd.MultiIndex.from_arrays([pd.to_datetime(samples.date).dt.strftime("%Y-%m-%d"), samples.symbol])
        expected = frozen.loc[keys, list(MARKET_INPUT_COLUMNS)].to_numpy(dtype=float)
        actual = samples[list(MARKET_INPUT_COLUMNS)].to_numpy(dtype=float)
        if not np.array_equal(expected, actual, equal_nan=True):
            bad = ~np.isclose(expected, actual, rtol=0, atol=0, equal_nan=True)
            errors.append(_error("all_74_market_inputs_exact_frozen_observation", rows=int(bad.any(axis=1).sum()),
                                 features=[name for index, name in enumerate(MARKET_INPUT_COLUMNS) if bad[:, index].any()]))
    except (KeyError, ValueError, TypeError) as exc:
        errors.append(_error("market_input_source_identity", error=str(exc)))
    return errors


def _projection(record, source_record, run_root, source_root, training=None, feature_start=None):
    errors = []; symbol = record["symbol"]
    try:
        path, original = Path(record["path"]).resolve(), Path(source_record["path"]).resolve()
        if not path.is_relative_to(run_root) or not original.is_relative_to(source_root):
            raise ValueError("projection/source escapes own artifact root")
        if _file_hash(path) != record["sha256"] or _file_hash(original) != source_record["sha256"] or record["source_sha256"] != source_record["sha256"]:
            errors.append(_error("projection_exact_immutable_source_hash", symbol=symbol))
        source = pd.read_parquet(original); frame = pd.read_parquet(path)
        if len(source) != len(frame) or not set(source) <= set(frame):
            raise ValueError("projection changed source row/column contract")
        for key in source:
            if not frame[key].equals(source[key]):
                errors.append(_error("projection_raw_or_y10_changed", symbol=symbol, column=key))
        if frame.attrs.get("wyckoff_schema_hash") != WYCKOFF_SCHEMA_HASH or frame.attrs.get("feature_version") != source.attrs.get("feature_version"):
            errors.append(_error("projection_independent_w_schema_old_profile_unchanged", symbol=symbol))
        available = pd.to_datetime(frame.wyckoff_available_at, utc=True, errors="coerce")
        dates = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
        if not available.notna().all() or not available.dt.strftime("%Y-%m-%d").eq(dates).all():
            errors.append(_error("wyckoff_current_close_available", symbol=symbol))
        for key in ("wyckoff_observed_at", "wyckoff_range_formed_at"):
            time = pd.to_datetime(frame[key], utc=True, errors="coerce", format="mixed")
            present = frame[key].notna()
            if (present & (time.isna() | (time > available))).any():
                errors.append(_error("wyckoff_event_or_range_future_visible", symbol=symbol, column=key))
            if key == "wyckoff_range_formed_at" and (present & (time >= available)).any():
                errors.append(_error("wyckoff_range_includes_current_bar", symbol=symbol))
        buys = frame.wyckoff_rule_buy.fillna(False).astype(bool)
        if not frame.loc[buys, "wyckoff_event"].isin(["Test", "LPS"]).all() or not frame.loc[buys, "wyckoff_input_eligible"].all():
            errors.append(_error("wyckoff_buy_requires_later_confirmed_test_or_lps", symbol=symbol))
        if frame.wyckoff_formal_qualified.any():
            errors.append(_error("raw_identity_gaps_never_qualified", symbol=symbol))
        if training is not None:
            expected = frame.loc[(pd.to_datetime(frame.date) >= pd.Timestamp(feature_start)) & frame.label_available.astype(bool), training.columns]
            try:
                pd.testing.assert_frame_equal(training.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False,
                                              check_exact=True, check_index_type=False)
            except AssertionError as exc:
                errors.append(_error("training_rows_exact_frozen_source_projection", symbol=symbol, error=str(exc)[:400]))
        return {"symbol": symbol, "rows": len(frame), "eligible": int(frame.wyckoff_input_eligible.sum()), "errors": errors}
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return {"symbol": symbol, "rows": 0, "eligible": 0, "errors": [*errors, _error("projection_read", symbol=symbol, error=str(exc))]}


def _bundle(path, root, training, preprocess_cache, old_models):
    errors = []; directory = Path(path).resolve(); bundle = _read(directory / "bundle.json")
    unsigned = dict(bundle); recorded = unsigned.pop("bundle_sha256", None)
    if not directory.is_relative_to(root) or recorded != _hash(unsigned) or _read(directory / "manifest.json") != bundle:
        errors.append(_error("bundle_alias_sha256_own_path", path=str(directory)))
    if bundle.get("publication_allowed") is not False or bundle.get("status") != "shadow":
        errors.append(_error("bundle_shadow_only", path=str(directory)))
    baseline_path = Path(bundle["baseline_bundle"]["path"]).resolve(); baseline = _read(baseline_path / "bundle.json")
    if baseline["bundle_sha256"] != bundle["baseline_bundle"]["sha256"] or baseline["target"] != bundle["target"]:
        errors.append(_error("reuse_same_exact_y10_target_and_baseline", path=str(directory)))
    if bundle["target"].get("horizon") != 10 or bundle["target"].get("label_version") != "czsc_fixed_horizon_broker_v1":
        errors.append(_error("y10_not_next_day_or_policy_exit", path=str(directory)))
    oof_path = Path(bundle["oof"]["path"])
    if _file_hash(oof_path) != bundle["oof"]["sha256"]:
        errors.append(_error("oof_physical_hash", path=str(directory)))
    oof = pd.read_parquet(oof_path); errors.extend(check_oof(oof, bundle["train_end"]))
    for name in ("ma", "structure"):
        binding = bundle["experts"][name]
        if binding["path"] != baseline["experts"][name]["path"]:
            errors.append(_error("expert_immutable_same_source_reuse", expert=name))
        manifest_path = Path(binding["path"]) / "manifest.json"
        if str(manifest_path) not in old_models:
            old_models[str(manifest_path)] = _check_model(manifest_path, ARTIFACT_RUNS_ROOT)
        checked = old_models[str(manifest_path)]; errors.extend(checked["errors"])
        manifest = checked.get("manifest", {})
        if (_file_hash(manifest_path) != binding["manifest_file_sha256"]
                or any(manifest.get(key) != binding[key] for key in ("manifest_sha256", "checkpoint_sha256", "preprocess_sha256", "feature_version", "feature_schema_hash"))):
            errors.append(_error("old_ten_experts_physical_manifest_and_checkpoint_binding", expert=name, path=str(manifest_path)))
    for evidence in bundle["oof"]["folds"]:
        if _file_hash(Path(evidence["rows_path"])) != evidence["rows_sha256"] or _file_hash(Path(evidence["old_rows_path"])) != evidence["old_rows_sha256"]:
            errors.append(_error("oof_new_old_rows_physical_hash"))
        new, old = pd.read_parquet(evidence["rows_path"]), pd.read_parquet(evidence["old_rows_path"])
        for f in (new, old): f["date"] = pd.to_datetime(f.date).dt.strftime("%Y-%m-%d")
        joined = new.merge(old[["date", "symbol", "p_ma", "p_structure"]], on=["date", "symbol"], suffixes=("_new", "_old"), validate="one_to_one")
        if len(joined) != len(new) or not np.allclose(joined[["p_ma_new", "p_structure_new"]], joined[["p_ma_old", "p_structure_old"]], rtol=0, atol=0):
            errors.append(_error("strict_forward_old_expert_probability_identity"))
    fusion_data = pd.read_parquet(bundle["fusion"]["dataset_path"])
    if _file_hash(Path(bundle["fusion"]["dataset_path"])) != bundle["fusion"]["dataset_file_sha256"]:
        errors.append(_error("fusion_actual_dataset_sha256"))
    fusion_manifest = _read(Path(bundle["fusion"]["path"]) / "manifest.json")
    errors.extend(_check_training_preprocessing(fusion_data, fusion_manifest, cache=preprocess_cache))
    targets = training[["date", "symbol", "label", "label_end"]].copy()
    targets["date"] = pd.to_datetime(targets.date).dt.strftime("%Y-%m-%d")
    targets["label_end"] = pd.to_datetime(targets.label_end).dt.strftime("%Y-%m-%d")
    for name, rows in (("oof", oof), ("stack", fusion_data)):
        observed = rows[["date", "symbol", "label", "label_end"]].copy()
        observed["date"] = pd.to_datetime(observed.date).dt.strftime("%Y-%m-%d")
        observed["label_end"] = pd.to_datetime(observed.label_end).dt.strftime("%Y-%m-%d")
        joined = observed.merge(targets, on=["date", "symbol"], how="left", validate="one_to_one", suffixes=("_model", "_source"))
        if not joined.label_model.eq(joined.label_source).all() or not joined.label_end_model.eq(joined.label_end_source).all():
            errors.append(_error("oof_and_stack_exact_frozen_y10_target", dataset=name, path=str(directory)))
    for column in ("date", "symbol", "p_ma", "p_structure", "p_wyckoff", "label", "label_end"):
        if not fusion_data.iloc[:len(oof)][column].reset_index(drop=True).equals(oof[column].reset_index(drop=True)):
            errors.append(_error("fusion_training_uses_exact_original_forward_oof_prefix", column=column, path=str(directory)))
    for name, threshold in bundle["thresholds"].items():
        scores = threshold.get("scores", [])
        selectable = [s for s in scores if s.get("selectable")]
        if scores and any(s.get("evidence") != "complete_validation_broker_ledger" for s in scores):
            errors.append(_error("threshold_requires_real_validation_ledger", expert=name))
        if threshold.get("selected"):
            best = max(selectable, key=lambda s: (s["net_return"], -s["max_drawdown"], -s["threshold"])) if selectable else None
            if best is None or threshold["threshold"] != best["threshold"]:
                errors.append(_error("threshold_frozen_validation_selection", expert=name))
        elif threshold.get("threshold") is not None:
            errors.append(_error("zero_trade_threshold_not_excellent_or_invented", expert=name))
    return {"path": str(directory), "oof_rows": len(oof), "available_at": bundle["available_at"], "errors": errors}


def _policy_clones(policy, data, root, calendar, initial_cash):
    errors = []; ledger_rows = 0; clone_count = 0; behavior_checked = 0
    following = dict(zip(calendar[:-1], calendar[1:]))
    errors.extend(check_policy_rows(data))
    for symbol, group in data.groupby("symbol", sort=False):
        behavior_path = root / "evaluation/policy_behavior" / policy / str(symbol).replace(".", "_")
        if not behavior_path.exists():
            errors.append(_error("policy_behavior_cash_reference_missing", symbol=symbol, policy=policy)); continue
        checked = _account(behavior_path, symbol, initial_cash, following); errors.extend(checked["errors"]); behavior_checked += 1
        daily = _csv(behavior_path / "daily.csv").set_index("date")
        for sample in group.to_dict("records"):
            state = daily.loc[sample["date"]]
            if not _close(sample["behavior_cash"], state["cash"]) or int(sample["behavior_shares"]) != int(state["shares"]):
                errors.append(_error("policy_cash_and_flat_state_reconciled", symbol=symbol, date=sample["date"]))
            if not _close(sample["candidate_cash_ratio"], state["cash"] / initial_cash) or not _close(sample["candidate_account_return"], state["equity"] / initial_cash - 1):
                errors.append(_error("policy_features_actual_cash_original_account_denominator", symbol=symbol, date=sample["date"]))
            if not sample.get("clone_ledger_hash"):
                if bool(sample["label_available"]) or int(state["shares"]) == 0:
                    errors.append(_error("decidable_candidate_clone_missing", symbol=symbol, date=sample["date"]))
                continue
            folder = root / "evaluation/policy_candidate_clones" / policy / str(symbol).replace(".", "_") / stable_hash({"candidate": sample["candidate_id"], "date": sample["date"]})[:20]
            checked = _account(folder, symbol, float(sample["behavior_cash"]), following)
            errors.extend(checked["errors"]); clone_count += 1
            ledger = _csv(folder / "ledger.csv"); ledger_rows += len(ledger)
            trades = ledger.where(pd.notna(ledger), None).to_dict("records")
            # CSV losslessly round-trips cash to audit tolerance, but floating
            # formatting may change exact JSON hash; use persisted record if
            # supplied and recompute amounts independently below.
            if ledger.empty:
                if bool(sample["label_available"]): errors.append(_error("rejected_clone_not_mature", symbol=symbol))
                continue
            buys = ledger.loc[ledger.action.eq("BUY")]
            if len(buys) != 1:
                errors.append(_error("candidate_clone_no_reinvestment", symbol=symbol, date=sample["date"]))
            if bool(sample["label_available"]):
                net = float(ledger.cash_flow.sum()); end = str(ledger.iloc[-1]["date"])
                if checked.get("final_shares") != 0 or not _close(sample["net_return"], net / sample["behavior_cash"]) or sample["label_end"] != end or float(sample["label"]) != float(net > 0):
                    errors.append(_error("policy_clone_fee_positive_maturity_recomputed", symbol=symbol, date=sample["date"]))
    return {"policy": policy, "rows": len(data), "behavior_accounts": behavior_checked, "clones": clone_count, "clone_ledger_rows": ledger_rows, "errors": errors}


def _execution_account(item, initial, following):
    symbol, directory = item
    checked = _account(directory, symbol, initial, following)
    checked["_trades"] = _round_trip_metrics(_csv(directory / "ledger.csv"))
    return checked


def audit_wyckoff_run(run_id, *, output=None, workers=4, runs_root=None):
    base = Path(runs_root or ARTIFACT_RUNS_ROOT).resolve(); root = (base / run_id).resolve()
    if not root.is_relative_to(base) or root == base: raise ValueError("run id must stay within artifact root")
    report_path = root / "reports/wyckoff-research.json"; before = _file_hash(report_path)
    report, metadata = _read(report_path), _read(root / "metadata.json"); errors = []
    if report.get("run_id") != run_id or metadata.get("run_id") != run_id or report.get("status") != "complete" or metadata.get("status") not in {"complete", "succeeded"}:
        errors.append(_error("completed_run_exact_identity"))
    if report.get("publication_allowed") is not False or report.get("model_gate", {}).get("passed") is not False or tuple(report["config"]["arms"]) != ARMS:
        errors.append(_error("fourteen_frozen_arms_shadow_only"))
    independent = report.get("independent_window", {})
    if independent.get("start") is not None and pd.Timestamp(independent["start"]).date() <= pd.Timestamp(independent["actual_complete_freeze_at"]).date():
        errors.append(_error("new_future_window_never_backdated"))
    calendar_value = _read(Path(report["calendar"]["path"])); calendar = calendar_value["dates"]
    if not calendar_value.get("verified") or calendar != sorted(set(calendar)) or stable_hash(calendar_value) != report["calendar"]["hash"]:
        errors.append(_error("verified_calendar_immutable_binding"))
    source_path = Path(report["source_snapshot"]["source_report_path"]); source_root = source_path.parent.parent
    source = _read(source_path); source_hash = _file_hash(source_path)
    if source_hash != report["source_snapshot"]["source_report_sha256"]: errors.append(_error("source_report_immutable_sha256"))
    baseline_path = Path(report["baseline_reference"]["report_path"])
    if _file_hash(baseline_path) != report["baseline_reference"]["report_sha256"]: errors.append(_error("baseline_report_immutable_sha256"))
    original_by_symbol = {r["symbol"]: r for r in source["dataset_records"]}
    records = report["dataset_records"]
    if len(records) != report["coverage"]["requested"] or len({r["symbol"] for r in records}) != len(records) or report["coverage"]["failed"]:
        errors.append(_error("full_requested_projection_coverage"))
    training = pd.read_parquet(root / "dataset/training.parquet")
    training_indices = training.groupby("symbol", sort=False).indices
    with ThreadPoolExecutor(max_workers=workers) as pool:
        projections = list(pool.map(lambda r: _projection(r, original_by_symbol[r["symbol"]], root, source_root,
            training.iloc[training_indices.get(r["symbol"], [])] if _mainboard(r["symbol"]) else None, report["config"]["feature_start"]), records))
    for item in projections: errors.extend(item["errors"])
    print(f"projection_audit {len(projections)} symbols", flush=True)
    # Recheck every original source, including unsupported/cash-only securities.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        unchanged = list(pool.map(lambda r: _file_hash(Path(r["path"])) == r["sha256"], source["dataset_records"]))
    if not all(unchanged): errors.append(_error("all_original_source_hashes_unchanged"))
    cache = {}; bundles = []; old_models = {}
    for item in report["model_bundles"]:
        value = _bundle(item["path"], root, training, cache, old_models); bundles.append(value); errors.extend(value["errors"])
    models = []; datasets = {path.name: pd.read_parquet(path) for path in sorted((root / "dataset").glob("*-*-training.parquet"))
                             if path.name.endswith(("-exit-training.parquet", "-policy-training.parquet"))}
    wdata = training[[*W_INPUT_COLUMNS, "date", "symbol", "label", "label_end", "label_available", "net_return"]].copy()
    wdata["input_eligible"] = training.wyckoff_input_eligible.fillna(False).astype(bool)
    for path in sorted((root / "models").rglob("manifest.json")):
        raw = _read(path)
        if "checkpoint" not in raw: continue
        checked = _check_model(path, root); errors.extend(checked["errors"]); models.append(checked)
        version = raw.get("feature_version", "")
        if version == "wyckoff_pv_v1": data = wdata
        elif version == "wyckoff_three_probability_oof_v1": continue  # exact stack checked per bundle
        else:
            contract = raw["label_contract"]; policy = "union" if contract["entry_policy"] == "risk" else "fresh"
            key = policy + ("-exit-training.parquet" if contract["head"] == "holding_exit" else "-policy-training.parquet")
            if key not in datasets: datasets[key] = pd.read_parquet(root / "dataset" / key)
            data = datasets[key]
            if contract != data.attrs["label_contract"]: errors.append(_error("head_independent_dataset_target_contract", path=str(path)))
            export_name = "exit-linear.json" if contract["head"] == "holding_exit" else "policy-linear.json"
            binding = _read(path.parent / "head.binding.json"); export = _read(path.parent / export_name)
            unsigned = dict(binding); signature = unsigned.pop("binding_sha256", None)
            if signature != stable_hash(unsigned) or _file_hash(path.parent / export_name) != binding["export_sha256"] or binding["checkpoint_sha256"] != raw["checkpoint_sha256"] or export["preprocess"] != raw["preprocess"] or export["columns"] != raw["schema"]["columns"]:
                errors.append(_error("head_cpu_export_weights_target_binding", path=str(path)))
            errors.extend(check_linear_export_storage(path.parent / "model.pt", export))
        errors.extend(_check_training_preprocessing(data, raw, cache=cache))
    if len(models) != report["compute_info"]["trained_models"] or sum(m["manifest"]["training_batches"] for m in models) != report["compute_info"]["training_batches"]:
        errors.append(_error("actual_unique_training_model_batch_counts"))
    print(f"model_audit {len(models)} models", flush=True)
    source_map = {r["symbol"]: r for r in records}; initial = report["config"]["initial_cash"]
    exit_checks = []
    baseline_root = baseline_path.parent.parent
    settings = _read(root / "config/czsc_strategy.json")["execution"]
    for key, data in list(datasets.items()):
        if not key.endswith("exit-training.parquet"): continue
        for behavior, partition in data.groupby("behavior_policy", sort=False):
            for symbol, frame in partition.groupby("symbol", sort=False):
                if behavior == "immutable_audited_rules_fresh_v1": folder = baseline_root / "evaluation/exit_reference" / symbol.replace(".", "_")
                elif "union" in behavior: folder = root / "evaluation/exit_reference/union" / symbol.replace(".", "_")
                else: folder = root / "evaluation/exit_reference/forward_three_fusion" / symbol.replace(".", "_")
                errors.extend(_check_exit_dataset(frame))
                raw = pd.read_parquet(source_map[symbol]["path"])
                errors.extend(check_market_inputs(frame, raw))
                checked = _check_exit_reference(folder, frame, raw, initial, calendar, settings); exit_checks.append(checked); errors.extend(checked["errors"])
    policy_checks = []
    for key, data in list(datasets.items()):
        if key.endswith("policy-training.parquet"):
            for symbol, samples in data.groupby("symbol", sort=False):
                errors.extend(check_market_inputs(samples, pd.read_parquet(source_map[symbol]["path"])))
            checked = _policy_clones(key.split("-")[0], data, root, calendar, initial); policy_checks.append(checked); errors.extend(checked["errors"])
    evaluations = []; following = dict(zip(calendar[:-1], calendar[1:]))
    auxiliary_names = report["config"].get("auxiliary_controls", ["matched_dual_fixed_candidates"])
    frozen_accounts = {r["symbol"] for r in records if _mainboard(r["symbol"])}
    for evaluation in report["evaluation"]:
        fold = evaluation["fold"]["name"]; reference_path = root / "reports" / (fold + "-old-six-ledger-references.json")
        references = _read(reference_path)
        if stable_hash(references) != evaluation["old_six_reference_manifest_sha256"]:
            errors.append(_error("immutable_old_six_reference_manifest", fold=fold))
        cash_only = {item["symbol"] for item in evaluation["unavailable_accounts_cash_retained"]}
        if evaluation["frozen_mainboard_accounts"] != len(frozen_accounts) or not cash_only <= frozen_accounts:
            errors.append(_error("all_arms_frozen_capital_denominator", fold=fold))
        if set(evaluation["variants"]) != set(ARMS): errors.append(_error("all_fourteen_ledger_arms_present", fold=fold))
        for arm in (*ARMS, *auxiliary_names):
            daily_rows = ledger_rows = accounts = 0
            completed_returns = []; completed_pnls = []; account_fees = 0.; equities = []; unclosed = 0; present = set()
            if arm in references:
                groups = {}
                for binding in references[arm]["bindings"]:
                    if _file_hash(Path(binding["path"])) != binding["sha256"]: errors.append(_error("old_baseline_ledger_reference_sha256", fold=fold, arm=arm))
                    groups[binding["symbol"]] = Path(binding["path"]).parent
            else:
                groups = {r["symbol"]: root / "evaluation" / fold / arm / r["symbol"].replace(".", "_") for r in records if _mainboard(r["symbol"])}
            located = [(symbol, directory) for symbol, directory in groups.items() if directory.exists()]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for checked in pool.map(lambda item: _execution_account(item, initial, following), located):
                    errors.extend(checked["errors"])
                    accounts += 1; daily_rows += checked.get("daily_rows", 0); ledger_rows += checked.get("ledger_rows", 0)
                    if "_equity" in checked: equities.append(checked["_equity"])
                    unclosed += int(checked.get("final_shares", 0) > 0); present.add(checked["symbol"])
                    trades = checked["_trades"]
                    completed_returns.extend(trades["net_returns"]); completed_pnls.extend(trades["net_pnl"])
                    account_fees += checked.get("fees", 0.)
            if arm in auxiliary_names:
                if "auxiliary_variants" in evaluation:
                    expected = evaluation["auxiliary_variants"][arm]["accounts"]
                else:
                    expected = evaluation["auxiliary_matched_dual_fixed_candidates"]["accounts"]
            else:
                expected = evaluation["variants"][arm]["accounts"]
            if accounts != expected: errors.append(_error("arm_real_accounts_count", fold=fold, arm=arm, actual=accounts, expected=expected))
            if frozen_accounts - present != cash_only:
                errors.append(_error("every_frozen_account_present_or_explicit_cash", fold=fold, arm=arm))
            variant = (evaluation.get("auxiliary_variants", {}).get(arm, evaluation.get("auxiliary_matched_dual_fixed_candidates", {}))
                       if arm in auxiliary_names else evaluation["variants"][arm])
            metrics = variant["metrics"]
            actual = {"completed_round_trips": len(completed_returns), "fees": account_fees,
                      "win_rate": float(np.mean(np.asarray(completed_pnls) > 0)) if completed_pnls else None,
                      "trade_expectancy": float(np.mean(completed_returns)) if completed_returns else None}
            wins = [value for value in completed_pnls if value > 0]; losses = [value for value in completed_pnls if value < 0]
            actual.update(wins=len(wins), losses=len(losses), breakeven=len(completed_pnls) - len(wins) - len(losses),
                          unclosed_positions=unclosed, trade_count=ledger_rows,
                          mean_net_pnl_CNY=float(np.mean(completed_pnls)) if completed_pnls else None,
                          payoff_ratio=float(np.mean(wins) / -np.mean(losses)) if wins and losses else None,
                          profit_factor=float(sum(wins) / -sum(losses)) if wins and losses else None)
            try:
                aggregate = _csv(root / "evaluation" / fold / arm / "aggregate_daily.csv")
                wide = pd.concat(equities, axis=1).sort_index().ffill().fillna(initial)
                equity = wide.sum(axis=1) + (len(frozen_accounts) - len(equities)) * initial
                dates = pd.to_datetime(aggregate.date).dt.strftime("%Y-%m-%d")
                if list(dates) != list(equity.index) or not _close(aggregate.equity, equity):
                    errors.append(_error("all_account_equity_and_missing_cash_aggregate", fold=fold, arm=arm))
                capital = len(frozen_accounts) * initial
                actual.update(final_equity=float(equity.iloc[-1]), total_return=float(equity.iloc[-1] / capital - 1),
                              net_return=float(equity.iloc[-1] / capital - 1),
                              max_drawdown=float(-(equity / equity.cummax().clip(lower=capital) - 1).min()))
            except (OSError, KeyError, ValueError, TypeError) as exc:
                errors.append(_error("aggregate_equity_audit_read", fold=fold, arm=arm, error=str(exc)))
            for key, value in actual.items():
                reported = metrics.get(key)
                if (value is None and reported is not None) or (value is not None and (reported is None or not np.isclose(value, reported, rtol=1e-8, atol=1e-7))):
                    errors.append(_error("reported_complete_fee_roundtrip_statistics_recomputed", fold=fold, arm=arm, metric=key))
            evaluations.append({"fold": fold, "arm": arm, "accounts": accounts, "daily_rows": daily_rows, "ledger_rows": ledger_rows,
                                "closed_round_trip_metrics": actual})
            print(f"execution_audit {fold} {arm} {accounts} accounts", flush=True)
    price_evaluations = [{**evaluation, "variants": {**evaluation["variants"], **{name: {} for name in auxiliary_names}}}
                         for evaluation in report["evaluation"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for checked in pool.map(lambda r: _execution_prices(r, root, price_evaluations, settings, calendar), [r for r in records if _mainboard(r["symbol"])]): errors.extend(checked)
    if _file_hash(report_path) != before or _file_hash(source_path) != source_hash:
        errors.append(_error("reports_unchanged_during_audit"))
    result = {"audit_version": "wyckoff_independent_execution_v1", "training_run_id": run_id,
        "training_report_sha256": before, "audited_at": local_now().isoformat(), "passed": not errors,
        "errors": errors, "error_count": len(errors), "source_universe": len(unchanged),
        "source_rows": sum(r["bars"] for r in source["dataset_records"]), "projection_coverage": len(projections),
        "projection_rows": sum(x["rows"] for x in projections), "quality_eligible_rows": sum(x["eligible"] for x in projections),
        "models": [{k: v for k, v in x.items() if k not in {"manifest", "errors"}} for x in models],
        "model_count": len(models), "actual_training_batches": sum(x["manifest"]["training_batches"] for x in models),
        "reused_old_expert_count": len(old_models),
        "reused_old_experts": [{key: value for key, value in item.items() if key not in {"manifest", "errors"}} for item in old_models.values()],
        "bundles": [{k: v for k, v in x.items() if k != "errors"} for x in bundles],
        "exit_reference_accounts": len(exit_checks), "actual_held_states": sum(x["held_samples"] for x in exit_checks),
        "policy": [{k: v for k, v in x.items() if k != "errors"} for x in policy_checks],
        "evaluation": evaluations, "formal_qualification": False,
        "scope": "artifact integrity, unchanged source/y10, forward OOF, training-only preprocessing, actual cash/fees/T1/open execution, true-held exit outcomes, all-candidate policy clones",
        "limitations": ["No profitability qualification; historical development data, corporate action/ST/delisting gaps remain.",
                        "Linear CPU exported weights exactly match original inert checkpoint storage; real-device tests separately cover forward arithmetic tolerance."]}
    destination = Path(output) if output else root / "reports/wyckoff-execution-audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"path": str(destination), "passed": result["passed"], "error_count": len(errors), "model_count": len(models)}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--output"); parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8: parser.error("workers must be 1..8")
    result = audit_wyckoff_run(args.training_run_id, output=args.output, workers=args.workers)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
