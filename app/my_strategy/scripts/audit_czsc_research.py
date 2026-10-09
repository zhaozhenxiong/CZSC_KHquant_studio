"""Read frozen research artifacts and reconcile execution without Torch or GPU.

Only reports/execution_audit.json is written. Source parquet, models, ledgers,
training configuration, thresholds and screening results are never changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now


MODES = ("czsc", "price_volume", "czsc_price_volume", "ml")


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bounded(path: str | Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"artifact outside expected run: {resolved}")
    return resolved


def _run(root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", run_id):
        raise ValueError("invalid run identity")
    return _bounded(root / run_id, root)


def _close(left: Any, right: Any) -> bool:
    return bool(np.allclose(left, right, rtol=1e-9, atol=0.01, equal_nan=False))


def _csv(path: Path) -> pd.DataFrame:
    # No-column empty ledgers and rejection tables are emitted as a newline.
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _error(rule: str, **context: Any) -> dict:
    return {"rule": rule, **context}


def _dataset(record: dict, root: Path, schema: dict, calendar: list[str], horizon: int) -> dict:
    errors = []
    symbol = record["symbol"]
    try:
        path = _bounded(record["path"], root)
        if _file_hash(path) != record["sha256"]:
            errors.append(_error("dataset_sha256", symbol=symbol, path=str(path)))
        columns = ["symbol", "date", "label", "label_end", "label_available", "label_reason", "input_eligible", "entry_date", "exit_date"]
        frame = pd.read_parquet(path, columns=columns)
        if frame.attrs.get("feature_version") != schema["feature_version"] or frame.attrs.get("schema_hash") != schema["feature_schema_hash"] or record.get("feature_schema_hash") != schema["feature_schema_hash"]:
            errors.append(_error("dataset_feature_semantics", symbol=symbol))
        if frame.attrs.get("data_version") != record["data_version"]:
            errors.append(_error("dataset_data_version", symbol=symbol))
        dates = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        if frame.empty or dates.duplicated().any() or not dates.is_monotonic_increasing or set(frame["symbol"]) != {symbol}:
            errors.append(_error("dataset_symbol_dates", symbol=symbol))
        if len(frame) != record["bars"] or (not frame.empty and dates.iloc[-1] != record["data_end"]):
            errors.append(_error("dataset_record_counts_or_end", symbol=symbol))
        available = frame["label_available"].fillna(False).astype(bool)
        if not np.array_equal(available, frame["label"].notna()) or not frame.loc[available, "label"].isin([0, 1]).all():
            errors.append(_error("label_availability_binary_value", symbol=symbol))
        reasons = {str(key): int(value) for key, value in frame["label_reason"].value_counts().items()}
        if record.get("label_reason_counts") is not None and reasons != record["label_reason_counts"]:
            errors.append(_error("label_reason_counts", symbol=symbol))
        if record.get("label_available_rows") is not None and int(available.sum()) != record["label_available_rows"]:
            errors.append(_error("label_available_counts", symbol=symbol))
        positions = {day: index for index, day in enumerate(calendar)}
        available_dates = dates.loc[available]
        signal_positions = available_dates.map(positions)
        in_calendar = signal_positions.notna() & (signal_positions + horizon + 1 < len(calendar))
        if not in_calendar.all():
            errors.append(_error("available_label_calendar", symbol=symbol, date=available_dates.loc[~in_calendar].iloc[0]))
        elif len(available_dates):
            indices = signal_positions.to_numpy(dtype=int)
            sessions = np.asarray(calendar)
            entry = frame.loc[available, "entry_date"].to_numpy()
            exit_days = frame.loc[available, "exit_date"].to_numpy()
            ends = frame.loc[available, "label_end"].to_numpy()
            valid = (entry == sessions[indices + 1]) & (exit_days == sessions[indices + horizon + 1]) & (ends == exit_days) & (ends <= dates.iloc[-1])
            if not valid.all():
                errors.append(_error("available_label_session_or_maturity", symbol=symbol, date=available_dates.iloc[int(np.flatnonzero(~valid)[0])]))
        return {"symbol": symbol, "bars": len(frame), "input_eligible_rows": int(frame["input_eligible"].sum()),
                "label_available_rows": int(available.sum()), "label_reason_counts": reasons, "errors": errors}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"symbol": symbol, "bars": 0, "input_eligible_rows": 0, "label_available_rows": 0,
                "label_reason_counts": {}, "errors": [*errors, _error("dataset_read", symbol=symbol, error=str(exc))]}


def _model(path: Path, root: Path, schema: dict) -> dict:
    errors = []
    try:
        path = _bounded(path, root)
        manifest = _read(path)
        integrity = dict(manifest)
        recorded = integrity.pop("manifest_sha256")
        if recorded != _hash(integrity):
            errors.append(_error("model_manifest_sha256", model=path.parent.name))
        for key in ("schema", "preprocess"):
            if manifest[key + "_sha256"] != _hash(manifest[key]):
                errors.append(_error("model_" + key + "_sha256", model=path.parent.name))
        checkpoint = _bounded(manifest["checkpoint"], root)
        if checkpoint != (path.parent / "model.pt").resolve() or _file_hash(checkpoint) != manifest["checkpoint_sha256"]:
            errors.append(_error("model_checkpoint_sha256_or_path", model=path.parent.name))
        if manifest.get("feature_schema_binding") != "bound" or manifest.get("feature_version") != schema["feature_version"] or manifest.get("feature_schema_hash") != schema["feature_schema_hash"]:
            errors.append(_error("model_feature_semantics", model=path.parent.name))
        if not manifest["train_label_end"] <= manifest["train_end"] < manifest["validation_start"] <= manifest["validation_label_end"] <= manifest["validation_end"] == manifest["available_at"]:
            errors.append(_error("model_purged_time_boundaries", model=path.parent.name))
        if manifest["calibration"]["fit_start"] != manifest["validation_start"] or manifest["calibration"]["fit_end"] != manifest["validation_end"]:
            errors.append(_error("model_calibration_dates", model=path.parent.name))
        return {"model": path.parent.name, "checkpoint_sha256": manifest["checkpoint_sha256"],
                "train_rows": manifest["sample_counts"]["train"], "validation_rows": manifest["sample_counts"]["validation"],
                "available_at": manifest["available_at"], "device": manifest["device"], "errors": errors}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"model": path.parent.name, "errors": [*errors, _error("model_read", model=path.parent.name, error=str(exc))]}


def _account(path: Path, symbol: str, initial_cash: float, following: dict[str, str]) -> dict:
    errors = []
    try:
        for name in ("daily", "ledger", "rejections"):
            if not (path / (name + ".csv")).is_file():
                errors.append(_error("missing_execution_file", symbol=symbol, path=str(path / (name + ".csv"))))
        if errors:
            return {"symbol": symbol, "errors": errors}
        daily, ledger, rejected = (_csv(path / (name + ".csv")) for name in ("daily", "ledger", "rejections"))
        if daily.empty:
            raise ValueError("daily account table is empty")
        daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
        if daily["date"].duplicated().any() or not daily["date"].is_monotonic_increasing:
            errors.append(_error("daily_unique_ascending_dates", symbol=symbol))
        daily_dates = set(daily["date"])
        numeric = daily[["cash", "shares", "close", "equity"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (daily["shares"] < 0).any() or (daily["cash"] < -0.01).any():
            errors.append(_error("daily_finite_nonnegative_account", symbol=symbol))
        if (daily["shares"] % 100 != 0).any():
            errors.append(_error("daily_integer_round_lot_inventory", symbol=symbol))
        if not _close(daily["cash"] + daily["shares"] * daily["close"], daily["equity"]):
            errors.append(_error("daily_cash_shares_close_equity", symbol=symbol))
        cash, shares, last_buy_date = initial_cash, 0, None
        flows, share_changes = {}, {}
        fees, buys, sells = 0.0, 0, 0
        if not ledger.empty and not pd.to_datetime(ledger["date"]).is_monotonic_increasing:
            errors.append(_error("ledger_chronological_order", symbol=symbol))
        for trade in ledger.to_dict(orient="records"):
            date, signal = str(trade["date"])[:10], str(trade["signal_date"])[:10]
            quantity, price, fee = float(trade["shares"]), float(trade["price"]), float(trade["fee"])
            action = trade["action"]
            if trade["symbol"] != symbol or date not in daily_dates:
                errors.append(_error("ledger_symbol_or_account_date", symbol=symbol, date=date))
            if signal >= date or following.get(signal) != date:
                errors.append(_error("ledger_next_verified_session", symbol=symbol, signal_date=signal, execution_date=date))
            if not all(math.isfinite(value) for value in (quantity, price, fee, float(trade["cash_flow"]))) or quantity <= 0 or quantity != int(quantity) or price <= 0 or fee < 0:
                errors.append(_error("ledger_valid_quantity_price_fee", symbol=symbol, date=date))
            if action == "BUY":
                buys += 1
                last_buy_date = date
                if quantity % 100 != 0:
                    errors.append(_error("buy_round_lot", symbol=symbol, date=date))
                delta, flow = quantity, -(quantity * price + fee)
            elif action in {"SELL", "REDUCE"}:
                sells += 1
                if last_buy_date is not None and date <= last_buy_date:
                    errors.append(_error("ledger_t_plus_one_inventory", symbol=symbol, date=date))
                delta, flow = -quantity, quantity * price - fee
            else:
                errors.append(_error("ledger_action", symbol=symbol, date=date, action=str(action)))
                continue
            if not _close(trade["cash_flow"], flow) or float(trade["position_before"]) != shares or float(trade["position_after"]) != shares + delta:
                errors.append(_error("ledger_cash_flow_or_position_chain", symbol=symbol, date=date))
            shares += int(delta)
            cash += float(trade["cash_flow"])
            if shares < 0 or not _close(trade["cash_after"], cash):
                errors.append(_error("ledger_cash_or_inventory_chain", symbol=symbol, date=date))
            flows[date] = flows.get(date, 0.0) + float(trade["cash_flow"])
            share_changes[date] = share_changes.get(date, 0) + int(delta)
            fees += fee
        expected_cash = initial_cash + daily["date"].map(flows).fillna(0).cumsum()
        expected_shares = daily["date"].map(share_changes).fillna(0).cumsum()
        if not _close(daily["cash"], expected_cash) or not np.array_equal(daily["shares"].to_numpy(), expected_shares.to_numpy()):
            errors.append(_error("daily_matches_actual_ledger_cash_shares", symbol=symbol))
        if not rejected.empty:
            if not {"date", "reason"}.issubset(rejected) or rejected["reason"].isna().any() or not set(rejected["date"].astype(str).str[:10]).issubset(daily_dates):
                errors.append(_error("rejections_retained_with_account_dates", symbol=symbol))
        return {"symbol": symbol, "daily_rows": len(daily), "ledger_rows": len(ledger), "rejection_rows": len(rejected),
                "buy_rows": buys, "sell_rows": sells, "fees": fees, "final_cash": float(daily.iloc[-1]["cash"]),
                "final_shares": int(daily.iloc[-1]["shares"]), "final_equity": float(daily.iloc[-1]["equity"]),
                "data_end": daily.iloc[-1]["date"], "_equity": pd.Series(daily["equity"].to_numpy(), index=daily["date"]), "errors": errors}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"symbol": symbol, "errors": [*errors, _error("account_read", symbol=symbol, path=str(path), error=str(exc))]}


def _screen(root: Path, training: dict) -> dict:
    errors = []
    report = _read(root / "reports" / "research-screen.json")
    rows, failed = report["rows"], report.get("failures", [])
    symbols = [row["symbol"] for row in rows] + [row["symbol"] for row in failed]
    counts = dict(Counter(row["category"] for row in rows))
    if len(symbols) != len(set(symbols)) or len(symbols) != report["coverage"]["requested"] or len(rows) != report["coverage"]["success"] or len(failed) != report["coverage"]["failed"] or counts != report["category_counts"]:
        errors.append(_error("screen_complete_classified_coverage"))
    if report.get("model_run_id") not in (None, training["run_id"]):
        errors.append(_error("screen_training_model_identity"))
    if report.get("model_run_id") == training["run_id"] and report.get("model_gate") != training.get("model_gate"):
        errors.append(_error("screen_frozen_model_gate"))
    requested_end = report["data_range"]["requested_end"]
    for row in rows:
        category, probability = row["category"], row.get("model_probability")
        if category not in {"buy", "watch", "exit", "excluded"} or (row["data_end"] != requested_end and category != "excluded"):
            errors.append(_error("screen_category_or_actual_date", symbol=row["symbol"]))
        if probability is not None and (not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1):
            errors.append(_error("screen_probability_range", symbol=row["symbol"]))
        evidence = {item["role"]: item for item in row.get("agent_evidence", [])}
        if set(evidence) != {"czsc", "price_volume", "ml", "data_execution"}:
            errors.append(_error("screen_four_evidence_roles", symbol=row["symbol"]))
        if evidence.get("data_execution", {}).get("judgment") == "veto" and category != "excluded":
            errors.append(_error("screen_risk_veto", symbol=row["symbol"]))
        if row.get("model_run_id") != report.get("model_run_id") or (report.get("model_run_id") is not None and row.get("model_train_end") != training["model_manifest"]["train_end"]):
            errors.append(_error("screen_row_model_identity", symbol=row["symbol"]))
        if category == "buy" and (not report["model_gate"]["passed"] or not row.get("model_validated") or probability is None
                                  or probability < training["config"]["probability_threshold"] or not row.get("point_confirmed_at")
                                  or str(row["point_confirmed_at"])[:10] > row["data_end"]
                                  or any(evidence.get(role, {}).get("judgment") != "support" for role in ("czsc", "price_volume", "ml"))):
            errors.append(_error("screen_buy_has_confirmed_validated_evidence", symbol=row["symbol"]))
    return {"run_id": root.name, "rows": len(rows), "failed": len(failed), "category_counts": counts, "errors": errors}


def audit_run(training_run_id: str, screen_run_id: str | None = None, *, runs_root: Path | None = None, workers: int = 8) -> dict:
    """Audit completed artifact snapshots, writing just their separate audit report."""
    if not 1 <= workers <= 8:
        raise ValueError("workers must be within 1..8")
    base = (runs_root or ARTIFACT_RUNS_ROOT).resolve()
    root = _run(base, training_run_id)
    training = _read(_bounded(root / "reports" / "research.json", root))
    metadata = _read(_bounded(root / "metadata.json", root))
    errors = []
    if training["run_id"] != training_run_id or metadata.get("status") != "complete":
        errors.append(_error("completed_training_identity"))
    if metadata.get("config_hash") != stable_hash(training["config"]) or metadata.get("research_config") != training["config"]:
        errors.append(_error("frozen_training_configuration"))
    configured_folds = [*training["config"]["folds"], *([training["config"]["holdout"]] if "holdout" in training["config"] else [])]
    expected_folds = [fold["name"] for fold in configured_folds if fold["test_end"] <= training["data_end"]]
    if [item["fold"]["name"] for item in training["evaluation"]] != expected_folds:
        errors.append(_error("all_configured_fold_evaluations", expected=expected_folds))
    stocks = metadata["stocks"]
    if len(stocks) != len(set(stocks)) or stocks != training["run_context"]["stocks"]:
        errors.append(_error("frozen_requested_stock_identity"))
    frozen = [symbol for symbol in stocks if symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ")]
    records, failures = training["dataset_records"], training.get("failures", [])
    covered = [item["symbol"] for item in records] + [item["symbol"] for item in failures]
    if set(covered) != set(stocks) or len(covered) != len(set(covered)) or training["coverage"] != {"requested": len(stocks), "success": len(records), "failed": len(failures)}:
        errors.append(_error("frozen_dataset_coverage"))
    calendar_path = _bounded(training["calendar"]["path"], base)
    calendar_value = _read(calendar_path)
    calendar = calendar_value["dates"]
    if not calendar_value.get("verified") or not calendar_value.get("source") or calendar != sorted(set(calendar)) or stable_hash(calendar_value) != training["calendar"]["hash"]:
        errors.append(_error("verified_frozen_calendar"))
    following = dict(zip(calendar[:-1], calendar[1:]))
    schema = training["model_manifest"]
    report = {"training_run_id": training_run_id, "screen_run_id": screen_run_id, "audited_at": local_now().isoformat(),
              "read_only_inputs": True, "model_payload_check": "hash_only_without_torch_deserialization", "workers": workers,
              "numeric_tolerances": {"money_absolute": 0.01, "money_relative": 1e-9, "ratio_absolute": 1e-10, "ratio_relative": 1e-8, "inventory": "exact_integer_round_lots"},
              "frozen_stocks": len(stocks), "frozen_mainboard_accounts": len(frozen), "errors": errors}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="czsc-read-audit") as pool:
        datasets = list(pool.map(lambda record: _dataset(record, root, schema, calendar, int(training["config"]["horizon"])), records))
        reason_counts = Counter()
        for item in datasets:
            reason_counts.update(item["label_reason_counts"])
            errors.extend(item.pop("errors"))
        report["dataset"] = {"files": len(datasets), "bars": sum(item["bars"] for item in datasets),
                             "input_eligible_rows": sum(item["input_eligible_rows"] for item in datasets),
                             "label_available_rows": sum(item["label_available_rows"] for item in datasets),
                             "label_reason_counts": dict(reason_counts), "records": datasets}
        model_paths = sorted((root / "models").glob("*/manifest.json"))
        models = list(pool.map(lambda path: _model(path, root, schema), model_paths))
        expected_models = {item["fold"]["name"] for item in training["evaluation"]} | {"production"}
        if {item["model"] for item in models} != expected_models:
            errors.append(_error("complete_model_files", expected=sorted(expected_models), observed=[item["model"] for item in models]))
        for item in models:
            errors.extend(item.pop("errors"))
        embedded_models = {item["fold"]["name"]: item.get("model_manifest") for item in training["evaluation"]}
        embedded_models["production"] = training["model_manifest"]
        for path in model_paths:
            expected_manifest = embedded_models.get(path.parent.name)
            if not expected_manifest or _read(path).get("manifest_sha256") != expected_manifest.get("manifest_sha256"):
                errors.append(_error("frozen_embedded_model_manifest", model=path.parent.name))
        report["models"] = models
        fold_reports = []
        initial = float(training["config"]["initial_cash"])
        for evaluation in training["evaluation"]:
            fold = evaluation["fold"]
            unavailable = {item["symbol"] for item in evaluation["unavailable_accounts_cash_retained"]}
            if evaluation["frozen_mainboard_accounts"] != len(frozen) or not unavailable.issubset(frozen):
                errors.append(_error("fold_frozen_capital_denominator", fold=fold["name"]))
            per_mode = {}
            for mode in MODES:
                location = root / "evaluation" / fold["name"] / mode
                present = [symbol for symbol in frozen if (location / symbol.replace(".", "_") / "daily.csv").is_file()]
                if set(frozen) - set(present) != unavailable:
                    errors.append(_error("fold_missing_accounts_cash_record", fold=fold["name"], mode=mode,
                                         missing=sorted(set(frozen) - set(present)), recorded=sorted(unavailable)))
                actual = [path.name.replace("_", ".") for path in location.iterdir() if path.is_dir()] if location.is_dir() else []
                if set(actual) != set(present):
                    errors.append(_error("unexpected_or_incomplete_account_files", fold=fold["name"], mode=mode))
                accounts = list(pool.map(lambda symbol: _account(location / symbol.replace(".", "_"), symbol, initial, following), present))
                for item in accounts:
                    errors.extend({**error, "fold": fold["name"], "mode": mode} for error in item.pop("errors"))
                valid = [item for item in accounts if "_equity" in item]
                summary = evaluation["variants"][mode]
                if len(valid) != summary["accounts"]:
                    errors.append(_error("fold_account_count", fold=fold["name"], mode=mode))
                try:
                    aggregate = _csv(location / "aggregate_daily.csv")
                except (OSError, ValueError) as exc:
                    aggregate = pd.DataFrame()
                    errors.append(_error("aggregate_read", fold=fold["name"], mode=mode, error=str(exc)))
                if valid and not aggregate.empty:
                    wide = pd.concat([item["_equity"] for item in valid], axis=1).sort_index().ffill().fillna(initial)
                    expected = wide.sum(axis=1) + (len(frozen) - len(valid)) * initial
                    dates = pd.to_datetime(aggregate["date"]).dt.strftime("%Y-%m-%d")
                    if list(dates) != list(expected.index) or not _close(aggregate["equity"], expected):
                        errors.append(_error("aggregate_daily_includes_failed_cash", fold=fold["name"], mode=mode))
                    final = float(expected.iloc[-1])
                    if not _close(summary["metrics"]["final_equity"], final) or not np.isclose(summary["metrics"]["total_return"], final / (len(frozen) * initial) - 1, rtol=1e-8, atol=1e-10):
                        errors.append(_error("aggregate_final_summary", fold=fold["name"], mode=mode))
                    if not _close(summary["metrics"]["fees"], sum(item["fees"] for item in valid)) or summary["metrics"]["trade_count"] != sum(item["ledger_rows"] for item in valid):
                        errors.append(_error("aggregate_trade_count_fees", fold=fold["name"], mode=mode))
                else:
                    errors.append(_error("missing_aggregate_daily", fold=fold["name"], mode=mode))
                for item in valid:
                    item.pop("_equity")
                per_mode[mode] = {"accounts": len(valid), "daily_files": len(present), "ledger_files": sum((location / symbol.replace(".", "_") / "ledger.csv").is_file() for symbol in present),
                                  "rejection_files": sum((location / symbol.replace(".", "_") / "rejections.csv").is_file() for symbol in present),
                                  "daily_rows": sum(item.get("daily_rows", 0) for item in accounts), "ledger_rows": sum(item.get("ledger_rows", 0) for item in accounts),
                                  "rejection_rows": sum(item.get("rejection_rows", 0) for item in accounts), "failed_cash_accounts": len(unavailable),
                                  "failed_cash_budget": len(unavailable) * initial, "open_positions": sum(item.get("final_shares", 0) > 0 for item in accounts),
                                  "account_records": accounts}
            fold_reports.append({"name": fold["name"], "modes": per_mode})
        report["evaluation"] = fold_reports
    if screen_run_id:
        try:
            screen = _screen(_run(base, screen_run_id), training)
            errors.extend(screen.pop("errors"))
            report["screen"] = screen
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(_error("screen_read", error=str(exc)))
    report["passed"] = not errors
    report["error_count"] = len(errors)
    output = _bounded(root / "reports" / "execution_audit.json", root)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--screen-run-id")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--runs-root", type=Path, help="Explicit artifact root for isolated audit fixtures")
    args = parser.parse_args()
    report = audit_run(args.training_run_id, args.screen_run_id, runs_root=args.runs_root, workers=args.workers)
    print(json.dumps({"passed": report["passed"], "error_count": report["error_count"], "dataset_bars": report["dataset"]["bars"],
                      "models": len(report["models"]), "training_run_id": args.training_run_id,
                      "report": str((args.runs_root or ARTIFACT_RUNS_ROOT) / args.training_run_id / "reports" / "execution_audit.json")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
