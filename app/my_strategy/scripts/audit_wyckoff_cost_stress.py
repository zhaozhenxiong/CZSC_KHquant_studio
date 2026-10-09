"""Independently reconcile immutable fee/slippage stress ledgers without models."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.tz import local_now
from my_strategy.scripts.audit_czsc_research import _read, _csv, _file_hash, _error, _close
from my_strategy.scripts.audit_czsc_dual_research import _execution_prices
from my_strategy.scripts.audit_czsc_wyckoff_research import _execution_account

ARMS = ("rules", "fusion_exit", "three_fusion_exit", "policy_fresh", "union_rules", "policy_union")
COST_FIELDS = ("commission", "stamp_tax", "min_commission", "slippage")


def check_stress_settings(base, stressed, factor):
    expected = copy.deepcopy(base)
    for name in COST_FIELDS:
        expected["execution"][name] *= factor
    return [] if expected == stressed else [_error("only_frozen_four_cost_fields_multiplied", multiplier=factor)]


def _base_equal(stressed, source):
    errors = []
    for name in ("ledger", "daily", "rejections"):
        try:
            left = _csv(stressed / (name + ".csv")).drop(columns="run_id", errors="ignore")
            right = _csv(source / (name + ".csv")).drop(columns="run_id", errors="ignore")
            if left.empty and right.empty:
                continue
            pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
        except (AssertionError, OSError, ValueError, KeyError) as exc:
            errors.append(_error("one_times_exact_original_account", account=str(stressed), artifact=name, error=str(exc)[:300]))
    return errors


def audit_cost_stress(run_id, *, output, workers=4):
    root = (ARTIFACT_RUNS_ROOT / run_id).resolve()
    if not root.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()) or root == ARTIFACT_RUNS_ROOT.resolve():
        raise ValueError("run id must stay within artifact root")
    path = root / "reports/wyckoff-cost-stress.json"
    report_hash = _file_hash(path); report = _read(path); metadata = _read(root / "metadata.json"); errors = []
    if (report["run_id"] != run_id or report["status"] != "complete" or metadata.get("status") != "complete"
            or report["publication_allowed"] is not False or report["compute_info"]["new_training_models"] != 0
            or tuple(report["arms"]) != ARMS or tuple(report["cost_fields_multiplied"]) != COST_FIELDS):
        errors.append(_error("completed_shadow_fixed_policy_no_training"))
    original_root = ARTIFACT_RUNS_ROOT / report["training_run_id"]
    original_path = original_root / "reports/wyckoff-research.json"; training = _read(original_path)
    if _file_hash(original_path) != report["source_training_report_sha256"]:
        errors.append(_error("frozen_training_report_physical_sha256"))
    if _file_hash(Path(report["source_strategy_path"])) != report["source_strategy_sha256"] or _read(Path(report["source_strategy_path"])) != report["base_strategy"]:
        errors.append(_error("stress_uses_exact_frozen_training_strategy"))
    frozen = report["frozen_file_sha256"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for file, expected, actual in pool.map(lambda pair: (pair[0], pair[1], _file_hash(Path(pair[0]))), frozen.items()):
            if actual != expected: errors.append(_error("stress_frozen_file_physical_sha256", path=file))
    if len(frozen) != report["source_model_files_unchanged"]:
        errors.append(_error("frozen_file_manifest_complete_count"))
    selected = set(metadata["stocks"])
    records = [record for record in training["dataset_records"] if record["symbol"] in selected]
    if len(records) != len(selected) or len(selected) != report["mainboard_accounts"]:
        errors.append(_error("selected_real_frozen_accounts_exact_coverage"))
    calendar = training["market_dates"]; following = dict(zip(calendar[:-1], calendar[1:])); initial = training["config"]["initial_cash"]
    baseline_root = Path(training["baseline_reference"]["report_path"]).parent.parent
    evaluated = []
    for evaluation in report["evaluation"]:
        fold = evaluation["fold"]["name"]; cash_only = {row["symbol"] for row in evaluation["unavailable_accounts_cash_retained"]}
        if not cash_only <= selected or set(evaluation["variants"]) != {str(float(value)) for value in report["multipliers"]}:
            errors.append(_error("stress_complete_factors_and_cash_only", fold=fold))
        for factor, variants in evaluation["variants"].items():
            factor_value = float(factor); settings = report["stressed_strategies"][factor]
            errors.extend(check_stress_settings(report["base_strategy"], settings, factor_value))
            if set(variants) != set(ARMS): errors.append(_error("all_six_fixed_stress_arms", fold=fold, multiplier=factor))
            price_evaluation = [{"fold": {**evaluation["fold"], "name": fold + "/" + factor}, "variants": variants}]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for checked in pool.map(lambda record: _execution_prices(record, root, price_evaluation, settings["execution"], calendar), records):
                    errors.extend(checked)
            for arm, variant in variants.items():
                location = root / "evaluation" / fold / factor / arm
                items = [(symbol, location / symbol.replace(".", "_")) for symbol in sorted(selected)
                         if (location / symbol.replace(".", "_") / "daily.csv").is_file()]
                if selected - {item[0] for item in items} != cash_only:
                    errors.append(_error("stress_all_accounts_or_explicit_cash", fold=fold, multiplier=factor, arm=arm))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    accounts = list(pool.map(lambda item: _execution_account(item, initial, following), items))
                for account in accounts: errors.extend(account["errors"])
                if len(accounts) != variant["accounts"]:
                    errors.append(_error("stress_report_actual_account_count", fold=fold, multiplier=factor, arm=arm))
                if factor_value == 1:
                    for symbol, directory in items:
                        source_root = baseline_root if arm in {"rules", "fusion_exit"} else original_root
                        errors.extend(_base_equal(directory, source_root / "evaluation" / fold / arm / symbol.replace(".", "_")))
                wide = pd.concat([account["_equity"] for account in accounts], axis=1).sort_index().ffill().fillna(initial)
                equity = wide.sum(axis=1) + (len(selected) - len(accounts)) * initial
                aggregate = _csv(location / "aggregate_daily.csv")
                if list(aggregate.date) != list(equity.index) or not _close(aggregate.equity, equity):
                    errors.append(_error("stress_complete_cash_equity_aggregation", fold=fold, multiplier=factor, arm=arm))
                returns = [value for account in accounts for value in account["_trades"]["net_returns"]]
                pnl = [value for account in accounts for value in account["_trades"]["net_pnl"]]
                capital = len(selected) * initial
                actual = {"final_equity": float(equity.iloc[-1]), "total_return": float(equity.iloc[-1] / capital - 1),
                          "max_drawdown": float(-(equity / equity.cummax().clip(lower=capital) - 1).min()),
                          "fees": sum(account["fees"] for account in accounts), "trade_count": sum(account["ledger_rows"] for account in accounts),
                          "completed_round_trips": len(returns), "win_rate": float(np.mean(np.asarray(pnl) > 0)) if pnl else None,
                          "trade_expectancy": float(np.mean(returns)) if returns else None}
                for key, value in actual.items():
                    observed = variant["metrics"].get(key)
                    if (value is None and observed is not None) or (value is not None and (observed is None or not np.isclose(value, observed, rtol=1e-8, atol=1e-7))):
                        errors.append(_error("stress_statistics_independently_recomputed", fold=fold, multiplier=factor, arm=arm, metric=key))
                evaluated.append({"fold": fold, "multiplier": factor_value, "arm": arm, "accounts": len(accounts), "metrics": actual})
            print(f"cost_execution_audit {fold} {factor} {len(selected)} accounts", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if not all(pool.map(lambda pair: _file_hash(Path(pair[0])) == pair[1], frozen.items())):
            errors.append(_error("immutable_stress_sources_unchanged_during_audit"))
    if _file_hash(path) != report_hash: errors.append(_error("stress_report_unchanged_during_audit"))
    result = {"audit_version": "wyckoff_independent_cost_execution_v1", "run_id": run_id, "training_run_id": report["training_run_id"],
              "stress_report_sha256": report_hash, "audited_at": local_now().isoformat(), "passed": not errors,
              "error_count": len(errors), "errors": errors, "frozen_files": len(frozen), "evaluation": evaluated,
              "formal_qualification": False, "scope": "frozen costs/models/thresholds/input integrity, actual source open prices/cash/T1/fees/capacity, 1x exact original ledgers, independent aggregate equity/return/drawdown/roundtrips"}
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"path": str(destination), "passed": result["passed"], "error_count": len(errors)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stress-run-id", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8: parser.error("workers must be 1..8")
    return 0 if audit_cost_stress(args.stress_run_id, output=args.output, workers=args.workers)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
