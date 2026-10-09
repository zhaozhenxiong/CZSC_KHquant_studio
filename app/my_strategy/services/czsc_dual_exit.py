"""Independent, research-only exit head trained on actual Broker close states.

Only outcomes read future ledger events. Holding features are captured while the
Broker advances chronologically, before the next session or final ledger exists.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.execution.broker import BrokerSimulator
from my_strategy.execution.cost_model import CostModel
from my_strategy.execution.order import Order
from my_strategy.services.czsc_backtest import execute_decisions, _guard, _tick
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services.czsc_research_features import FEATURE_COLUMNS, FEATURE_SCHEMA_HASH
from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS, get_feature_profile


EXIT_LABEL_VERSION = "czsc_actual_holding_exit_advantage_v1"
EXIT_FEATURE_VERSION = "czsc_market_actual_holding_exit_v1"
STATE_COLUMNS = ("holding_bars", "holding_return", "stop_distance", "position_weight",
                 "remaining_fraction", "cash_weight", "stop_available")
EXIT_FEATURE_COLUMNS = tuple(dict.fromkeys((*FEATURE_COLUMNS, *MA_TREND_COLUMNS, *STATE_COLUMNS)))
EXIT_FEATURE_SCHEMA_HASH = stable_hash({"version": EXIT_FEATURE_VERSION, "columns": EXIT_FEATURE_COLUMNS,
    "market_schema": FEATURE_SCHEMA_HASH, "ma_schema": get_feature_profile("ma_trend_v1")["schema_hash"],
    "state": "actual_broker_after_open_before_future_close_v1"})
_TARGET = "exit_remaining_shares_next_verified_open_vs_frozen_rule_continuation_net_proceeds"
_EXECUTION = "next_verified_open_T1_broker_guards_partial_exit_pending"
_MATURITY = "later_of_two_complete_actual_liquidation_dates"
_VALUATION = "remaining_share_net_sell_proceeds_common_cash_and_sunk_buy_cost_cancel_no_reinvestment"


def exit_contract(config: dict, market_dates: list[str], entry_policy: str = "fresh") -> dict:
    if entry_policy not in {"fresh", "risk"}:
        raise ValueError("holding exit labels require a planned entry policy")
    if not market_dates or market_dates != sorted(set(market_dates)):
        raise ValueError("holding exit requires a verified ascending unique calendar")
    value = {"label_version": EXIT_LABEL_VERSION, "head": "holding_exit", "entry_policy": entry_policy,
             "target": _TARGET,
             "strategy_hash": stable_hash(config), "calendar_hash": stable_hash(market_dates),
             "feature_schema_hash": EXIT_FEATURE_SCHEMA_HASH,
             "execution": _EXECUTION, "maturity": _MATURITY, "valuation": _VALUATION,
             "qualification": "research_only_shadow_unless_explicit_arm_opt_in"}
    return {**value, "contract_hash": stable_hash(value)}


def _check_contract(value: dict) -> None:
    content = {key: item for key, item in value.items() if key != "contract_hash"}
    if (value.get("contract_hash") != stable_hash(content) or value.get("head") != "holding_exit"
            or value.get("label_version") != EXIT_LABEL_VERSION
            or value.get("feature_schema_hash") != EXIT_FEATURE_SCHEMA_HASH
            or value.get("target") != _TARGET or value.get("execution") != _EXECUTION
            or value.get("maturity") != _MATURITY or value.get("valuation") != _VALUATION
            or value.get("entry_policy") not in {"fresh", "risk"}
            or not value.get("strategy_hash") or not value.get("calendar_hash")
            or value.get("qualification") != "research_only_shadow_unless_explicit_arm_opt_in"):
        raise ValueError("independent holding-exit target contract mismatch")


def _aligned(features: pd.DataFrame, raw: pd.DataFrame, decisions: list[dict]) -> None:
    if (not {"date", "symbol", "open", "high", "low", "close", "volume", "amount", "source", "has_trade_price"} <= set(raw)
            or not {"date", "symbol", "input_eligible", "model_input_eligible", *EXIT_FEATURE_COLUMNS[:-len(STATE_COLUMNS)]} <= set(features)):
        raise ValueError("exit research requires complete frozen market features and verified-price metadata")
    dates = pd.to_datetime(raw.date).dt.strftime("%Y-%m-%d").tolist()
    if (raw.empty or len(raw) != len(features) or len(raw) != len(decisions)
            or dates != sorted(set(dates)) or raw.symbol.nunique() != 1
            or features.symbol.astype(str).tolist() != raw.symbol.astype(str).tolist()
            or pd.to_datetime(features.date).dt.strftime("%Y-%m-%d").tolist() != dates
            or [row["date"] for row in decisions] != dates):
        raise ValueError("exit research requires aligned one-symbol chronological inputs")


def holding_feature_row(feature: dict, snapshot: dict) -> dict:
    """Combine only contemporaneous market values and a live Broker state."""
    cost, close, equity = snapshot["actual_cost_per_share"], snapshot["close"], snapshot["equity"]
    stop = snapshot.get("stop_price")
    values = {column: feature.get(column, np.nan) for column in EXIT_FEATURE_COLUMNS}
    values.update(holding_bars=float(snapshot["holding_bars"]), holding_return=close / cost - 1,
                  stop_distance=close / stop - 1 if stop and stop > 0 else np.nan,
                  position_weight=snapshot["shares"] * close / equity,
                  remaining_fraction=snapshot["shares"] / snapshot["entry_shares"],
                  cash_weight=snapshot["cash"] / equity, stop_available=float(stop is not None))
    bar = snapshot["bar"]
    eligible = (bool(feature.get("input_eligible", False))
                and bool(feature.get("model_input_eligible", feature.get("input_eligible", False))))
    eligible = (eligible and bar["source"] in VERIFIED_SOURCES and bar["has_trade_price"] == 1
                and np.isfinite([bar["volume"], bar["amount"]]).all() and bar["volume"] > 0 and bar["amount"] > 0
                and np.isfinite([values[column] for column in (*MA_TREND_COLUMNS, *STATE_COLUMNS)]).all())
    observed_at = pd.Timestamp(snapshot["available_at"])
    for field in ("available_at", "structure_confirmed_at", "zone_confirmed_at", "weekly_available_at", "monthly_available_at"):
        value = feature.get(field)
        if value is not None and not pd.isna(value):
            timestamp = pd.to_datetime(value, errors="coerce", utc=True)
            eligible = eligible and pd.notna(timestamp) and timestamp <= observed_at
    return {**values, "symbol": snapshot["symbol"], "date": snapshot["date"], "input_eligible": bool(eligible),
            "snapshot_available_at": snapshot["available_at"], "actual_shares": snapshot["shares"],
            "actual_cost_per_share": cost, "actual_entry_date": snapshot["entry_date"],
            "ledger_entries_seen": snapshot["ledger_entries_seen"]}


def _bind(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    frame.attrs.update(feature_version=EXIT_FEATURE_VERSION, schema_hash=EXIT_FEATURE_SCHEMA_HASH,
                       feature_columns=list(EXIT_FEATURE_COLUMNS), label_version=EXIT_LABEL_VERSION,
                       label_contract=dict(contract), feature_profile="actual_holding_exit_v1")
    return frame


def _forced_liquidation(snapshot, rows, seasoned, calendar_next, config):
    """Counterfactual outcomes use the same sell guards, capacity, ticks and fees."""
    settings = config["execution"]
    cost = CostModel(commission=float(settings["commission"]), stamp_tax=float(settings["stamp_tax"]),
                     min_commission=float(settings["min_commission"]), slippage=float(settings["slippage"]),
                     version="flat_v1", impact_tiers=())
    broker = BrokerSimulator(initial_cash=snapshot["cash"], cost_model=cost)
    symbol = snapshot["symbol"]
    broker.positions[symbol] = snapshot["shares"]
    broker.avg_cost[symbol] = snapshot["actual_cost_per_share"]
    rejected = []
    for index in range(snapshot["bar_index"] + 1, len(rows)):
        row, previous = rows[index], rows[index - 1]
        day = pd.Timestamp(row.date).date().isoformat()
        previous_day = pd.Timestamp(previous.date).date().isoformat()
        reason = _guard(symbol, row, previous, "SELL", int(seasoned[index]), settings)
        if calendar_next.get(previous_day) != day:
            reason = "missing_next_market_bar"
        if any(bar.source not in VERIFIED_SOURCES for bar in (row, previous)):
            reason = "unverified_research_source"
        if day == snapshot["entry_date"]:
            reason = "t_plus_one"
        capacity = int(float(previous.volume) * float(settings["max_volume_participation"])) // 100 * 100
        shares = min(broker.shares(symbol), capacity)
        if not reason and shares <= 0:
            reason = "insufficient_cash_or_prior_liquidity"
        if reason:
            rejected.append(reason)
            continue
        price = _tick(cost.execution_price("SELL", float(row.open)))
        entry = broker.submit(Order(date=day, symbol=symbol, side="SELL", shares=shares, price=price,
                            target_weight=0., signal_date=snapshot["date"], reason="forced_exit_label",
                            applied_slippage=cost.slippage), portfolio_prices={symbol: float(row.open)})
        if entry is None:
            rejected.append("broker_rejected")
        if broker.shares(symbol) == 0:
            return {"date": day, "proceeds": sum(entry.cash_flow for entry in broker.ledger.entries),
                    "rejections": rejected, "reason": "available"}
    return {"date": None, "proceeds": None, "rejections": rejected, "reason": "forced_exit_unclosed"}


def build_exit_dataset(*, features, raw, base_decisions, market_dates, start, initial_cash, config,
                       entry_policy="fresh", reference_callback=None) -> pd.DataFrame:
    """Preserve every actual-held sample; future outcomes never create its features."""
    _aligned(features, raw, base_decisions)
    contract = exit_contract(config, market_dates, entry_policy)
    market = {str(pd.Timestamp(row["date"]).date()): row for row in features.to_dict("records")}
    samples = []
    def collect(snapshot):
        samples.append((holding_feature_row(market[snapshot["date"]], snapshot), snapshot))
        return None
    account = execute_decisions(str(raw.iloc[0].symbol), raw, base_decisions, start, initial_cash,
                                config, "holding-exit-label-reference", market_dates=market_dates,
                                verified_sources=VERIFIED_SOURCES, entry_policy=entry_policy, exit_policy=collect)
    if reference_callback is not None:
        reference_callback(account)
    rows = list(raw.itertuples(index=False))
    from my_strategy.adapters.czsc_adapter import legacy_fuyao_source
    valid = np.array([row.volume > 0 and row.amount > 0 and row.has_trade_price == 1 and not legacy_fuyao_source(row.source) for row in rows])
    seasoned = np.r_[0, np.cumsum(valid[:-1])]
    calendar_next = dict(zip(market_dates[:-1], market_dates[1:]))
    ledger = account["ledger"]
    result = []
    for values, snapshot in samples:
        record = {**values, "label": np.nan, "label_end": None, "label_available": False,
                  "label_reason": "continuation_unclosed", "exit_advantage": np.nan,
                  "forced_exit_date": None, "continuation_exit_date": None,
                  "forced_rejection_count": 0, "forced_first_rejection": None,
                  "forced_net_proceeds": np.nan, "continuation_net_proceeds": np.nan}
        remaining, proceeds, continued_end = snapshot["shares"], 0., None
        for entry in ledger[snapshot["ledger_entries_seen"]:]:
            if entry["action"] == "BUY":
                break
            remaining -= int(entry["shares"])
            proceeds += float(entry["cash_flow"])
            if remaining == 0:
                continued_end = entry["date"]
                break
        if continued_end:
            forced = _forced_liquidation(snapshot, rows, seasoned, calendar_next, config)
            record.update(continuation_exit_date=continued_end, forced_exit_date=forced["date"],
                          forced_rejection_count=len(forced["rejections"]),
                          forced_first_rejection=forced["rejections"][0] if forced["rejections"] else None,
                          label_reason=forced["reason"])
            if forced["date"]:
                advantage = forced["proceeds"] - proceeds
                record.update(label=float(advantage > 0), label_end=max(forced["date"], continued_end),
                              label_available=True, label_reason="available", exit_advantage=advantage / initial_cash,
                              forced_net_proceeds=forced["proceeds"], continuation_net_proceeds=proceeds)
        result.append(record)
    frame = pd.DataFrame(result, columns=list(dict.fromkeys([*EXIT_FEATURE_COLUMNS, "symbol", "date", "input_eligible",
        "snapshot_available_at", "actual_shares", "actual_cost_per_share", "actual_entry_date", "ledger_entries_seen",
        "label", "label_end", "label_available", "label_reason", "exit_advantage", "forced_exit_date",
        "continuation_exit_date", "forced_rejection_count", "forced_first_rejection", "forced_net_proceeds", "continuation_net_proceeds"])))
    frame = _bind(frame, contract)
    frame.attrs.update(data_version=stable_hash({"raw": raw.attrs.get("data_version"), "symbol": str(raw.iloc[0].symbol),
                                                "reference_decisions": base_decisions}),
                       actual_held_samples=len(frame), label_reason_counts=dict(Counter(frame.label_reason)),
                       reference_account_metrics=account["metrics"], reference_rejections=account["rejections"])
    return frame


def train_exit_model(dataset, output_dir, train_end, validation_start, validation_end,
                     device="auto", seed=42, epochs=20, hidden_sizes=(), **kwargs):
    """Reuse chronological training machinery under an independently bound target."""
    from my_strategy.services.czsc_research_ml import train_model
    _check_contract(dataset.attrs.get("label_contract", {}))
    if dataset.attrs.get("label_version") != EXIT_LABEL_VERSION or dataset.attrs.get("schema_hash") != EXIT_FEATURE_SCHEMA_HASH:
        raise ValueError("holding-exit training dataset identity mismatch")
    if (dataset.attrs.get("feature_version") != EXIT_FEATURE_VERSION
            or dataset.attrs.get("feature_columns") != list(EXIT_FEATURE_COLUMNS)):
        raise ValueError("holding-exit training feature order/version mismatch")
    manifest = train_model(dataset, EXIT_FEATURE_COLUMNS, output_dir, train_end, validation_start, validation_end,
                           device=device, seed=seed, epochs=epochs, hidden_sizes=hidden_sizes, **kwargs)
    if not hidden_sizes:
        export_exit_head(output_dir)
    return manifest


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_exit_head(output_dir):
    """Parent training process exports immutable linear weights for CPU workers.

    This is a separate binding; the training manifest and checkpoint stay intact.
    """
    import torch
    from my_strategy.services.czsc_research_ml import _verified_manifest
    directory = Path(output_dir)
    manifest, checkpoint = _verified_manifest(directory)
    _check_contract(manifest.get("label_contract", {}))
    if manifest["schema"]["hidden_sizes"]:
        raise ValueError("CPU NumPy exit export requires a linear head")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    parameters = payload["state_dict"]
    exported = {"format": "holding_exit_linear_cpu_numpy_v1", "columns": list(EXIT_FEATURE_COLUMNS),
        "feature_version": EXIT_FEATURE_VERSION, "feature_schema_hash": EXIT_FEATURE_SCHEMA_HASH,
        "contract_hash": manifest["label_contract"]["contract_hash"],
        "weight": parameters["0.weight"].numpy().reshape(-1).tolist(),
        "bias": float(parameters["0.bias"].numpy()[0]), "preprocess": manifest["preprocess"],
        "temperature": manifest["calibration"]["temperature"]}
    export_path = directory / "exit-linear.json"
    serialized = json.dumps(exported, ensure_ascii=False, indent=2, allow_nan=False)
    if export_path.exists() and export_path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError("exit export already exists with a different identity")
    export_path.write_text(serialized, encoding="utf-8")
    binding = {"head": "holding_exit", "label_version": EXIT_LABEL_VERSION,
        "contract_hash": exported["contract_hash"], "feature_schema_hash": EXIT_FEATURE_SCHEMA_HASH,
        "model_manifest_sha256": manifest["manifest_sha256"], "checkpoint_sha256": manifest["checkpoint_sha256"],
        "available_at": manifest["available_at"], "export": export_path.name, "export_sha256": _file_hash(export_path),
        "runtime": "cpu_numpy", "actual_training_device": manifest["device"]}
    binding["binding_sha256"] = stable_hash(binding)
    destination = directory / "head.binding.json"
    serialized = json.dumps(binding, ensure_ascii=False, indent=2, allow_nan=False)
    if destination.exists() and destination.read_text(encoding="utf-8") != serialized:
        raise FileExistsError("exit binding already exists with a different identity")
    destination.write_text(serialized, encoding="utf-8")
    return binding


class ExitPredictor:
    def __init__(self, model_bundle, device=None, *, expected_contract=None):
        from my_strategy.services.czsc_research_ml import PredictorSession, _verified_manifest
        bundle = model_bundle if isinstance(model_bundle, dict) else {"model_dir": str(model_bundle)}
        self.directory = Path(bundle["model_dir"])
        self.manifest, _ = _verified_manifest(self.directory)
        self.contract = self.manifest.get("label_contract", {})
        _check_contract(self.contract)
        if (self.manifest.get("label_version") != EXIT_LABEL_VERSION
                or self.manifest.get("feature_version") != EXIT_FEATURE_VERSION
                or self.manifest.get("feature_schema_hash") != EXIT_FEATURE_SCHEMA_HASH
                or self.manifest.get("schema", {}).get("columns") != list(EXIT_FEATURE_COLUMNS)
                or bundle.get("contract_hash", self.contract["contract_hash"]) != self.contract["contract_hash"]
                or expected_contract and expected_contract["contract_hash"] != self.contract["contract_hash"]):
            raise ValueError("entry weights or incompatible policy cannot be used as an exit head")
        self.threshold = float(bundle.get("threshold", .55))
        if not np.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("exit threshold must be a finite validated value within (0, 1)")
        self.exported = None
        self._counts = {"rows": 0, "input_rows": 0, "batches": 0, "model_loads": 0, "hash_checks": 0, "cache_hits": 0}
        self._seconds = 0.
        self.requested_device = device
        self._expected_binding_hash = bundle.get("head_binding_sha256")
        if (self.directory / "head.binding.json").exists():
            self.exported = self._load_export()
            self._counts["model_loads"] = 1
        # A non-linear extension remains explicitly CPU; spawned workers never
        # infer on a GPU merely because the original checkpoint trained there.
        self.session = None if self.exported is not None else PredictorSession(device="cpu")

    def _load_export(self):
        path = self.directory / "head.binding.json"
        binding = json.loads(path.read_text(encoding="utf-8"))
        content = {key: value for key, value in binding.items() if key != "binding_sha256"}
        if (binding.get("binding_sha256") != stable_hash(content) or binding.get("head") != "holding_exit"
                or self._expected_binding_hash is not None and binding.get("binding_sha256") != self._expected_binding_hash
                or binding.get("label_version") != EXIT_LABEL_VERSION or binding.get("runtime") != "cpu_numpy"
                or binding.get("contract_hash") != self.contract["contract_hash"]
                or binding.get("feature_schema_hash") != EXIT_FEATURE_SCHEMA_HASH
                or binding.get("model_manifest_sha256") != self.manifest["manifest_sha256"]
                or binding.get("checkpoint_sha256") != self.manifest["checkpoint_sha256"]
                or binding.get("available_at") != self.manifest["available_at"]
                or binding.get("export") != "exit-linear.json" or self.manifest["schema"]["hidden_sizes"]):
            raise ValueError("exit CPU export binding integrity mismatch")
        export_path = self.directory / binding["export"]
        if _file_hash(export_path) != binding["export_sha256"]:
            raise ValueError("exit CPU export file hash mismatch")
        value = json.loads(export_path.read_text(encoding="utf-8"))
        if (value.get("format") != "holding_exit_linear_cpu_numpy_v1"
                or value.get("columns") != list(EXIT_FEATURE_COLUMNS)
                or value.get("feature_version") != EXIT_FEATURE_VERSION
                or value.get("feature_schema_hash") != EXIT_FEATURE_SCHEMA_HASH
                or value.get("contract_hash") != self.contract["contract_hash"]
                or value.get("preprocess") != self.manifest["preprocess"]
                or value.get("temperature") != self.manifest["calibration"]["temperature"]
                or len(value.get("weight", [])) != len(EXIT_FEATURE_COLUMNS)
                or not np.isfinite([*value["weight"], value["bias"], value["temperature"]]).all()
                or value["temperature"] <= 0):
            raise ValueError("exit CPU export semantic identity mismatch")
        return value

    @property
    def diagnostics(self):
        work = self.session.diagnostics if self.session is not None else {**self._counts,
            "device": "cpu", "devices": ["cpu"], "requested_device": self.requested_device,
            "actual_cpu_numpy_inference": self._counts["rows"] > 0, "actual_cuda_inference": False,
            "actual_mps_inference": False, "actual_gpu_inference": False, "inference_seconds": self._seconds}
        return {**work, "head": "holding_exit", "contract_hash": self.contract["contract_hash"],
            "actual_training_device": self.manifest["device"],
            "actual_mps_training": self.manifest.get("actual_mps_training", False)}

    def predict(self, frame, *, as_of=None):
        _check_contract(frame.attrs.get("label_contract", {}))
        if frame.attrs["label_contract"]["contract_hash"] != self.contract["contract_hash"]:
            raise ValueError("holding-exit inference target contract mismatch")
        if self.session is not None:
            return self.session.predict(frame, self.directory, as_of=as_of)
        from my_strategy.services.czsc_research_ml import _features, _transform, _verified_manifest
        started = time.perf_counter()
        current, _ = _verified_manifest(self.directory)
        if current["manifest_sha256"] != self.manifest["manifest_sha256"]:
            raise ValueError("exit model changed during research")
        exported = self._load_export()
        self._counts["hash_checks"] += 1
        if frame.attrs.get("feature_version") != EXIT_FEATURE_VERSION or frame.attrs.get("schema_hash") != EXIT_FEATURE_SCHEMA_HASH:
            raise ValueError("exit inference feature semantic identity mismatch")
        dates = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
        if ((dates < self.manifest["available_at"]).any()
                or as_of is not None and (pd.Timestamp(as_of).date().isoformat() < self.manifest["available_at"]
                    or (dates > pd.Timestamp(as_of).date().isoformat()).any())):
            raise ValueError("exit model unavailable at requested replay/as_of date")
        eligible = frame.input_eligible.fillna(False).astype(bool).to_numpy()
        values = _transform(_features(frame, EXIT_FEATURE_COLUMNS), exported["preprocess"])
        result = np.full(len(frame), np.nan)
        logits = (values[eligible] @ np.asarray(exported["weight"], dtype=np.float32)
                  + np.float32(exported["bias"])) / np.float32(exported["temperature"])
        result[eligible] = 1. / (1. + np.exp(-np.clip(logits, -40, 40)))
        self._counts["rows"] += int(eligible.sum())
        self._counts["input_rows"] += len(frame)
        self._counts["batches"] += int(eligible.any())
        self._counts["cache_hits"] += 1
        self._seconds += time.perf_counter() - started
        return result

    def predict_one(self, values: dict, *, as_of: str) -> float:
        """Avoid constructing a pandas frame for each live holding close."""
        if self.session is not None:
            return float(self.predict(_bind(pd.DataFrame([values]), self.contract), as_of=as_of)[0])
        from my_strategy.services.czsc_research_ml import _verified_manifest
        started = time.perf_counter()
        current, _ = _verified_manifest(self.directory)
        if current["manifest_sha256"] != self.manifest["manifest_sha256"]:
            raise ValueError("exit model changed during research")
        exported = self._load_export()
        self._counts["hash_checks"] += 1
        date = pd.Timestamp(values["date"]).date().isoformat()
        signal = pd.Timestamp(as_of).date().isoformat()
        if date < self.manifest["available_at"] or signal < self.manifest["available_at"] or date > signal:
            raise ValueError("exit model unavailable at requested replay/as_of date")
        eligible = bool(values["input_eligible"])
        probability = np.nan
        if eligible:
            numbers = np.asarray([values[key] for key in EXIT_FEATURE_COLUMNS], dtype=np.float64)
            preprocess = exported["preprocess"]
            filled = np.where(np.isfinite(numbers), numbers, np.asarray(preprocess["median"]))
            transformed = np.clip((filled - np.asarray(preprocess["mean"])) / np.asarray(preprocess["scale"]), -8, 8).astype(np.float32)
            logit = (transformed @ np.asarray(exported["weight"], dtype=np.float32) + np.float32(exported["bias"])) / np.float32(exported["temperature"])
            probability = float(1. / (1. + np.exp(-np.clip(logit, -40, 40))))
        self._counts["rows"] += int(eligible)
        self._counts["input_rows"] += 1
        self._counts["batches"] += int(eligible)
        self._counts["cache_hits"] += 1
        self._seconds += time.perf_counter() - started
        return probability


def make_exit_policy(*, features, predictor=None, predictor_resolver=None, apply_exit=False):
    market = {str(pd.Timestamp(row["date"]).date()): row for row in features.to_dict("records")}
    def policy(snapshot):
        selected = predictor_resolver(snapshot["date"]) if predictor_resolver is not None else predictor
        values = holding_feature_row(market[snapshot["date"]], snapshot)
        response = {"head": "holding_exit", "probability": None, "request_exit": False,
                    "apply": False, "research_arm": bool(apply_exit), "status": "exit_model_unavailable"}
        if selected is None:
            return response
        response.update(contract_hash=selected.contract["contract_hash"], threshold=selected.threshold,
                        available_at=selected.manifest["available_at"])
        if snapshot["date"] < selected.manifest["available_at"]:
            return {**response, "status": "exit_checkpoint_unavailable_at_signal"}
        if not values["input_eligible"]:
            return {**response, "status": "exit_input_ineligible"}
        probability = (selected.predict_one(values, as_of=snapshot["date"]) if hasattr(selected, "predict_one")
                       else float(selected.predict(_bind(pd.DataFrame([values]), selected.contract), as_of=snapshot["date"])[0]))
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            return {**response, "status": "exit_probability_unavailable"}
        return {**response, "probability": probability, "request_exit": probability >= selected.threshold,
                "apply": bool(apply_exit), "status": "exit_research_applied" if apply_exit else "exit_shadow"}
    return policy


def evaluate_exit_arm(*, features, raw, base_decisions, market_dates, start, end, initial_cash,
                      config, model_bundle=None, device=None, apply_exit=False, entry_policy="fresh", run_id="dual-exit-arm"):
    _aligned(features, raw, base_decisions)
    contract = exit_contract(config, market_dates, entry_policy)
    predictor = ExitPredictor(model_bundle, device, expected_contract=contract) if model_bundle is not None else None
    mask = pd.to_datetime(raw.date) <= pd.Timestamp(end)
    frame = raw.loc[mask].reset_index(drop=True)
    if frame.empty:
        raise ValueError("exit research end precedes available bars")
    result = execute_decisions(str(frame.iloc[0].symbol), frame, base_decisions[:len(frame)], start, initial_cash,
                               config, run_id, market_dates=market_dates, verified_sources=VERIFIED_SOURCES,
                               entry_policy=entry_policy,
                               exit_policy=make_exit_policy(features=features.loc[mask], predictor=predictor, apply_exit=apply_exit))
    result.update(exit_compute_info=predictor.diagnostics if predictor else {"head": "holding_exit", "rows": 0,
                    "batches": 0, "actual_mps_inference": False, "actual_cuda_inference": False},
                  exit_contract=contract, exit_research_opt_in=bool(apply_exit), qualification="research_only_not_released")
    return result
