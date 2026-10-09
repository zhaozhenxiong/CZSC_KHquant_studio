"""Independent Wyckoff actual-holding exit and frozen-policy entry targets.

Market features are observed at the close; only outcome construction reads
future Broker ledger entries. Ordinary callbacks are shadow by default. This
module never rewrites the dual target, model, or Broker implementation.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_dual_exit import (
    EXIT_FEATURE_COLUMNS as DUAL_EXIT_COLUMNS, STATE_COLUMNS,
    _forced_liquidation, holding_feature_row as dual_holding_feature_row,
)
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS, STRUCTURE_COLUMNS
from my_strategy.services.czsc_wyckoff_features import W_INPUT_COLUMNS, WYCKOFF_SCHEMA_HASH


EXIT_LABEL_VERSION = "wyckoff_actual_holding_exit_advantage_v2"
EXIT_FEATURE_VERSION = "wyckoff_market_actual_holding_exit_v2"
EXIT_FEATURE_COLUMNS = tuple(dict.fromkeys((*DUAL_EXIT_COLUMNS, *W_INPUT_COLUMNS)))
EXIT_FEATURE_SCHEMA_HASH = stable_hash({"version": EXIT_FEATURE_VERSION,
    "columns": EXIT_FEATURE_COLUMNS, "wyckoff_schema": WYCKOFF_SCHEMA_HASH,
    "state": "actual_broker_close_before_next_open_v2"})
POLICY_LABEL_VERSION = "wyckoff_frozen_actual_policy_roundtrip_v1"
POLICY_FEATURE_VERSION = "wyckoff_three_experts_actual_candidate_state_v1"
POLICY_STATE_COLUMNS = ("candidate_cash_ratio", "candidate_account_return", "candidate_stop_distance",
    "candidate_floor_distance", "candidate_ceiling_distance", "candidate_risk_fraction",
    "candidate_target_weight", "candidate_risk_plan")
POLICY_FEATURE_COLUMNS = tuple((*MA_TREND_COLUMNS, *STRUCTURE_COLUMNS, *W_INPUT_COLUMNS, *POLICY_STATE_COLUMNS))
POLICY_FEATURE_SCHEMA_HASH = stable_hash({"version": POLICY_FEATURE_VERSION,
    "columns": POLICY_FEATURE_COLUMNS, "wyckoff_schema": WYCKOFF_SCHEMA_HASH,
    "state": "all_frozen_decidable_candidates_flat_actual_cash_clone_v1"})
_EXIT_TARGET = "exit_remaining_shares_next_verified_open_vs_frozen_continuation_net_proceeds"
_POLICY_TARGET = "candidate_actual_broker_complete_fee_net_roundtrip_positive"
_QUALIFICATION = "research_only_shadow_unless_explicit_arm_opt_in"


def _policy_hash(config):
    return stable_hash({"policy": "native_position_and_actual_cost_stop_v1", "config": config})


def _contract(config, market_dates, entry_policy, candidate_policy, behavior_policy,
              continuation_policy_hash, source_snapshot_hash, *, head):
    if entry_policy not in {"fresh", "risk"}:
        raise ValueError("actual-state research requires a planned fresh or risk entry policy")
    if not market_dates or market_dates != sorted(set(market_dates)):
        raise ValueError("actual-state research requires a verified ascending unique calendar")
    if not all(isinstance(value, str) and value.strip() for value in
               (candidate_policy, behavior_policy, continuation_policy_hash, source_snapshot_hash)):
        raise ValueError("candidate, behavior, continuation and source identities are required")
    exit_head = head == "holding_exit"
    value = {"label_version": EXIT_LABEL_VERSION if exit_head else POLICY_LABEL_VERSION,
        "head": head, "entry_policy": entry_policy, "candidate_policy": candidate_policy,
        "behavior_policy": behavior_policy, "continuation_policy_hash": continuation_policy_hash,
        "source_snapshot_hash": source_snapshot_hash,
        "target": _EXIT_TARGET if exit_head else _POLICY_TARGET,
        "strategy_hash": stable_hash(config), "calendar_hash": stable_hash(market_dates),
        "feature_schema_hash": EXIT_FEATURE_SCHEMA_HASH if exit_head else POLICY_FEATURE_SCHEMA_HASH,
        "execution": "next_verified_open_T1_broker_guards_partial_exit_pending",
        "maturity": "later_of_two_actual_liquidation_dates" if exit_head else "actual_complete_episode_liquidation_date",
        "valuation": "remaining_share_net_sell_proceeds_common_cash_and_sunk_buy_cost_cancel_no_reinvestment" if exit_head
            else "all_actual_buy_debits_and_remaining_share_net_sell_credits_no_reinvestment",
        "selection": "actual_held_behavior_replay_states" if exit_head else "all_frozen_candidates_before_final_model_selection",
        "qualification": _QUALIFICATION}
    return {**value, "contract_hash": stable_hash(value)}


def exit_contract(config, market_dates, entry_policy="fresh", *, candidate_policy="fresh_v1",
                  behavior_policy="rules_fresh_v1", continuation_policy_hash=None,
                  source_snapshot_hash="explicit_fixture_source"):
    return _contract(config, market_dates, entry_policy, candidate_policy, behavior_policy,
        continuation_policy_hash or _policy_hash(config), source_snapshot_hash, head="holding_exit")


def policy_contract(config, market_dates, entry_policy="fresh", *, candidate_policy="fresh_v1",
                    behavior_policy="rules_fresh_v1", continuation_policy_hash=None,
                    source_snapshot_hash="explicit_fixture_source"):
    return _contract(config, market_dates, entry_policy, candidate_policy, behavior_policy,
        continuation_policy_hash or _policy_hash(config), source_snapshot_hash, head="policy_entry")


def _check_contract(value, *, head="holding_exit"):
    label = EXIT_LABEL_VERSION if head == "holding_exit" else POLICY_LABEL_VERSION
    schema = EXIT_FEATURE_SCHEMA_HASH if head == "holding_exit" else POLICY_FEATURE_SCHEMA_HASH
    target = _EXIT_TARGET if head == "holding_exit" else _POLICY_TARGET
    if (value.get("contract_hash") != stable_hash({k: v for k, v in value.items() if k != "contract_hash"})
            or value.get("head") != head or value.get("label_version") != label
            or value.get("feature_schema_hash") != schema or value.get("target") != target
            or value.get("entry_policy") not in {"fresh", "risk"}
            or value.get("qualification") != _QUALIFICATION
            or not all(value.get(key) for key in ("strategy_hash", "calendar_hash", "candidate_policy",
                "behavior_policy", "continuation_policy_hash", "source_snapshot_hash"))):
        raise ValueError("independent Wyckoff target contract mismatch")
    expected = _contract({}, ["fixture"], value["entry_policy"], value["candidate_policy"],
        value["behavior_policy"], value["continuation_policy_hash"], value["source_snapshot_hash"], head=head)
    if any(value.get(key) != expected[key] for key in ("execution", "maturity", "valuation", "selection")):
        raise ValueError("independent Wyckoff target execution/maturity contract mismatch")


def _aligned(features, raw, decisions):
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "amount", "source", "has_trade_price"}
    if not required <= set(raw) or not {"date", "symbol", "input_eligible", "model_input_eligible",
            "wyckoff_input_eligible", *W_INPUT_COLUMNS, *DUAL_EXIT_COLUMNS[:-len(STATE_COLUMNS)]} <= set(features):
        raise ValueError("Wyckoff targets require complete verified market features and price metadata")
    dates = pd.to_datetime(raw.date).dt.strftime("%Y-%m-%d").tolist()
    if (raw.empty or len(raw) != len(features) or len(raw) != len(decisions) or dates != sorted(set(dates))
            or raw.symbol.nunique() != 1 or features.symbol.astype(str).tolist() != raw.symbol.astype(str).tolist()
            or pd.to_datetime(features.date).dt.strftime("%Y-%m-%d").tolist() != dates
            or [row["date"] for row in decisions] != dates):
        raise ValueError("Wyckoff targets require aligned one-symbol chronological inputs")
    _check_behavior_lineage(decisions)


def _check_behavior_lineage(decisions):
    """Reject final/retrospective model states masquerading as forward behavior."""
    for row in decisions:
        if row.get("retrospective") or row.get("retrospective_inference"):
            raise ValueError("behavior holdings must come from strictly forward checkpoints")
        lineage = row.get("behavior_model_lineage", {})
        if lineage.get("retrospective"):
            raise ValueError("behavior holdings must come from strictly forward checkpoints")
        available = lineage.get("available_at") or row.get("behavior_model_available_at")
        cutoff = lineage.get("label_cutoff") or row.get("behavior_model_label_cutoff")
        if available and pd.Timestamp(available).date().isoformat() > row["date"]:
            raise ValueError("behavior checkpoint was not available at its signal")
        if cutoff and pd.Timestamp(cutoff).date().isoformat() >= row["date"]:
            raise ValueError("behavior checkpoint labels overlap its holding replay")


def _wyckoff_visible(feature, date):
    eligible = bool(feature.get("wyckoff_input_eligible", False))
    # No preceding event has a genuinely missing age; its train-only imputation
    # is part of the W schema. Essential price/volume values cannot be imputed.
    eligible = eligible and np.isfinite([feature.get(k, np.nan) for k in W_INPUT_COLUMNS if k != "w_event_age"]).all()
    cutoff = pd.Timestamp(str(date) + "T15:00:00+08:00")
    for key in ("wyckoff_available_at", "wyckoff_observed_at", "wyckoff_range_formed_at"):
        value = feature.get(key)
        if value is not None and not pd.isna(value):
            timestamp = pd.to_datetime(value, errors="coerce", utc=True)
            eligible = eligible and pd.notna(timestamp) and timestamp <= cutoff
    return bool(eligible)


def holding_feature_row(feature, snapshot):
    values = dual_holding_feature_row(feature, snapshot)
    values.update({key: feature.get(key, np.nan) for key in W_INPUT_COLUMNS})
    values["input_eligible"] = bool(values["input_eligible"] and _wyckoff_visible(feature, snapshot["date"]))
    values.update(wyckoff_available_at=feature.get("wyckoff_available_at"),
        episode_id=stable_hash({"symbol": snapshot["symbol"], "entry_date": snapshot["entry_date"], "plan_id": snapshot["plan_id"]}))
    return values


def _bind(frame, contract, *, head="holding_exit"):
    _check_contract(contract, head=head)
    exit_head = head == "holding_exit"
    frame.attrs.update(feature_version=EXIT_FEATURE_VERSION if exit_head else POLICY_FEATURE_VERSION,
        schema_hash=EXIT_FEATURE_SCHEMA_HASH if exit_head else POLICY_FEATURE_SCHEMA_HASH,
        feature_columns=list(EXIT_FEATURE_COLUMNS if exit_head else POLICY_FEATURE_COLUMNS),
        label_version=EXIT_LABEL_VERSION if exit_head else POLICY_LABEL_VERSION,
        label_contract=dict(contract), feature_profile="wyckoff_actual_holding_exit_v2" if exit_head else "wyckoff_actual_policy_entry_v1")
    return frame


def build_exit_dataset(*, features, raw, base_decisions, market_dates, start, initial_cash, config,
                       entry_policy="fresh", candidate_policy="fresh_v1", behavior_policy="rules_fresh_v1",
                       continuation_policy_hash=None, source_snapshot_hash="explicit_fixture_source",
                       reference_exit_policy=None, reference_callback=None, row_behavior_policy=None):
    """Real held states from frozen forward behavior, with independent future outcomes."""
    _aligned(features, raw, base_decisions)
    contract = exit_contract(config, market_dates, entry_policy, candidate_policy=candidate_policy,
        behavior_policy=behavior_policy, continuation_policy_hash=continuation_policy_hash,
        source_snapshot_hash=source_snapshot_hash)
    market = {pd.Timestamp(row["date"]).date().isoformat(): row for row in features.to_dict("records")}
    samples = []
    def collect(snapshot):
        samples.append((holding_feature_row(market[snapshot["date"]], snapshot), snapshot))
        return reference_exit_policy(copy.deepcopy(snapshot)) if reference_exit_policy else None
    account = execute_decisions(str(raw.iloc[0].symbol), raw, base_decisions, start, initial_cash, config,
        "wyckoff-holding-label-reference", market_dates=market_dates, verified_sources=VERIFIED_SOURCES,
        entry_policy=entry_policy, exit_policy=collect)
    if reference_callback is not None:
        reference_callback(account)
    rows = list(raw.itertuples(index=False))
    from my_strategy.adapters.czsc_adapter import legacy_fuyao_source
    valid = np.array([row.volume > 0 and row.amount > 0 and row.has_trade_price == 1
                      and not legacy_fuyao_source(row.source) for row in rows])
    seasoned = np.r_[0, np.cumsum(valid[:-1])]
    calendar_next = dict(zip(market_dates[:-1], market_dates[1:]))
    result = []
    for values, snapshot in samples:
        record = {**values, "label": np.nan, "label_end": None, "label_available": False,
            "label_reason": "continuation_unclosed", "exit_advantage": np.nan,
            "forced_exit_date": None, "continuation_exit_date": None,
            "forced_rejection_count": 0, "forced_first_rejection": None,
            "forced_net_proceeds": np.nan, "continuation_net_proceeds": np.nan,
            "behavior_policy": row_behavior_policy or behavior_policy,
            "candidate_policy": candidate_policy, "continuation_policy_hash": contract["continuation_policy_hash"],
            "source_snapshot_hash": source_snapshot_hash, "entry_policy": entry_policy,
            "plan_id": snapshot["plan_id"], "snapshot_sha256": stable_hash(snapshot)}
        remaining, proceeds, continued_end = snapshot["shares"], 0., None
        for entry in account["ledger"][snapshot["ledger_entries_seen"]:]:
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
    extra = ("symbol", "date", "input_eligible", "snapshot_available_at", "actual_shares", "actual_cost_per_share",
        "actual_entry_date", "ledger_entries_seen", "wyckoff_available_at", "episode_id", "label", "label_end",
        "label_available", "label_reason", "exit_advantage", "forced_exit_date", "continuation_exit_date",
        "forced_rejection_count", "forced_first_rejection", "forced_net_proceeds", "continuation_net_proceeds",
        "behavior_policy", "candidate_policy", "continuation_policy_hash", "source_snapshot_hash", "entry_policy", "plan_id", "snapshot_sha256")
    frame = _bind(pd.DataFrame(result, columns=[*EXIT_FEATURE_COLUMNS, *extra]), contract)
    frame.attrs.update(data_version=stable_hash({"raw": raw.attrs.get("data_version"), "symbol": str(raw.iloc[0].symbol),
        "reference_decisions": base_decisions, "contract": contract["contract_hash"]}),
        actual_held_samples=len(frame), label_reason_counts=dict(Counter(frame.label_reason)),
        state_coverage={"behavior_policy": row_behavior_policy or behavior_policy, "candidate_policy": candidate_policy,
            "episodes": int(frame.episode_id.nunique()), "actual_held_rows": len(frame),
            "mature_rows": int(frame.label_available.sum()), "ineligible_rows": int((~frame.input_eligible).sum())},
        reference_account_metrics=account["metrics"], reference_rejections=account["rejections"])
    return frame


def bind_existing_exit_dataset(existing, wyckoff_features, *, config, market_dates,
        source_dataset_sha256, reference_run_id, entry_policy="fresh", candidate_policy="fresh_v1",
        behavior_policy="rules_fresh_v1", continuation_policy_hash=None,
        source_snapshot_hash="explicit_fixture_source"):
    """Add causal W values to immutable, independently audited actual held states.

    Both outcome paths, remaining shares and live features are retained byte-for-
    value. The caller verifies the cited source file SHA before reading it; the
    new dataset records that SHA instead of pretending these are new replays.
    """
    if (not isinstance(source_dataset_sha256, str) or len(source_dataset_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_dataset_sha256)
            or not isinstance(reference_run_id, str) or not reference_run_id):
        raise ValueError("reused actual state dataset requires a verified source SHA and reference run")
    required = {*DUAL_EXIT_COLUMNS, "symbol", "date", "input_eligible", "actual_shares",
        "actual_entry_date", "label", "label_end", "label_available", "label_reason",
        "forced_net_proceeds", "continuation_net_proceeds"}
    if not required <= set(existing) or not {"symbol", "date", *W_INPUT_COLUMNS, "wyckoff_input_eligible"} <= set(wyckoff_features):
        raise ValueError("reused exit dataset is missing audited actual states or causal W inputs")
    if existing.duplicated(["symbol", "date"]).any() or wyckoff_features.duplicated(["symbol", "date"]).any():
        raise ValueError("reused exit state and W projections require unique symbol/date keys")
    source = existing.copy()
    source["date"] = pd.to_datetime(source.date).dt.strftime("%Y-%m-%d")
    w = wyckoff_features.copy()
    w["date"] = pd.to_datetime(w.date).dt.strftime("%Y-%m-%d")
    metadata = [key for key in ("wyckoff_available_at", "wyckoff_observed_at", "wyckoff_range_formed_at") if key in w]
    frame = source.merge(w[["symbol", "date", *W_INPUT_COLUMNS, "wyckoff_input_eligible", *metadata]],
        how="left", on=["symbol", "date"], sort=False, validate="one_to_one")
    visible = np.array([_wyckoff_visible(row, row["date"]) for row in frame.to_dict("records")])
    frame["input_eligible"] = frame.input_eligible.fillna(False).astype(bool) & visible
    frame["behavior_policy"], frame["candidate_policy"] = behavior_policy, candidate_policy
    frame["entry_policy"], frame["source_snapshot_hash"] = entry_policy, source_snapshot_hash
    frame["source_state_dataset_sha256"], frame["source_state_run_id"] = source_dataset_sha256, reference_run_id
    frame["episode_id"] = [stable_hash({"symbol": symbol, "entry_date": day})
        for symbol, day in zip(frame.symbol, frame.actual_entry_date)]
    contract = exit_contract(config, market_dates, entry_policy, candidate_policy=candidate_policy,
        behavior_policy=behavior_policy, continuation_policy_hash=continuation_policy_hash,
        source_snapshot_hash=source_snapshot_hash)
    frame["continuation_policy_hash"] = contract["continuation_policy_hash"]
    frame = _bind(frame, contract)
    frame.attrs.update(data_version=stable_hash({"source_dataset_sha256": source_dataset_sha256,
        "source_state_run_id": reference_run_id, "contract_hash": contract["contract_hash"],
        "wyckoff_schema_hash": WYCKOFF_SCHEMA_HASH}), actual_held_samples=len(frame),
        state_coverage={"behavior_policy": behavior_policy, "candidate_policy": candidate_policy,
            "episodes": int(frame.episode_id.nunique()), "actual_held_rows": len(frame),
            "mature_rows": int(frame.label_available.sum()), "ineligible_rows": int((~frame.input_eligible).sum()),
            "source": "immutable_audited_actual_broker_states"},
        label_reason_counts=dict(Counter(frame.label_reason)),
        source_state_dataset_sha256=source_dataset_sha256, source_state_run_id=reference_run_id)
    return frame


def make_union_decisions(features, base_decisions, *, entry_policy="risk", config=None,
                         candidate_policy="fresh_plus_wyckoff_test_lps_v1"):
    """Freeze a separate theoretical candidate policy, without editing Position.

    Native entries/exits keep priority. Confirmed Test/LPS may start a new
    episode only while the combined theoretical policy is flat. Their known
    support, bounded next-open price and risk budget are frozen in the plan;
    actual holdings, stops and execution still belong to BrokerSimulator.
    """
    from my_strategy.services.czsc_analysis import strategy_config
    from my_strategy.services.czsc_entry_plan import entry_plan_parameters, make_entry_plan
    if entry_policy not in {"fresh", "risk"} or len(features) != len(base_decisions):
        raise ValueError("union candidates require aligned features and a planned entry policy")
    settings = strategy_config(config)
    parameters = entry_plan_parameters()
    rows = features.to_dict("records")
    dates = [pd.Timestamp(row["date"]).date().isoformat() for row in rows]
    if dates != [row["date"] for row in base_decisions] or dates != sorted(set(dates)):
        raise ValueError("union candidate inputs must be date aligned")
    active, origin, consumed, result = None, None, set(), []
    for index, (feature, base) in enumerate(zip(rows, base_decisions)):
        row = copy.deepcopy(base)
        day, symbol = base["date"], str(feature["symbol"])
        close = float(base.get("reference_price", feature.get("close", np.nan)))
        position_start = base.get("position_start")
        if position_start is not None and day < position_start:
            row.update(target_weight=0., action="WAIT", position_intent="WAIT", entry_gate_passed=False,
                entry_plan=None, candidate_origin=None, candidate_policy=candidate_policy)
            result.append(row)
            continue
        native_plan = base.get("entry_plan") or {}
        native_buy = native_plan.get("status") == "active" and native_plan.get("entry_allowed") is True
        native_exit = (base.get("action") == "SELL" or base.get("rule_sell") is True
                       or bool(base.get("risk_exit_triggered")))
        w_exit = bool(feature.get("wyckoff_rule_sell", False))
        if active is not None:
            stop_hit = entry_policy == "risk" and active.get("stop_price") is not None and close <= active["stop_price"]
            if native_exit or w_exit or stop_hit:
                row.update(target_weight=0., action="SELL", position_intent="SELL", candidate_action="SELL",
                    entry_gate_passed=False, entry_plan={**active, "status": "exit", "entry_allowed": False},
                    candidate_origin=origin, candidate_policy=candidate_policy,
                    union_exit_reason="native_exit" if native_exit else "wyckoff_supply" if w_exit else "frozen_support_stop")
                active, origin = None, None
            else:
                row.update(target_weight=active["target_weight"], action="HOLD", position_intent="HOLD",
                    entry_plan={**active, "status": "holding", "entry_allowed": False},
                    entry_gate_passed=False, candidate_origin=origin, candidate_policy=candidate_policy)
            result.append(row)
            continue
        plan, selected = None, None
        if native_buy:
            plan = copy.deepcopy(native_plan)
            if plan.get("policy") != entry_policy:
                confirmed = pd.to_datetime(feature.get("point_confirmed_at"), errors="coerce")
                point_day = confirmed.date().isoformat() if pd.notna(confirmed) else None
                point_age = index - dates.index(point_day) if point_day in dates and point_day <= day else None
                plan = make_entry_plan(feature, symbol=symbol, signal_date=day, reference_price=close,
                    policy=entry_policy, max_weight=settings["execution"]["max_weight"], parameters=parameters,
                    eligible=bool(base.get("eligible")), rule_buy=True, rule_sell=native_exit,
                    ml_allowed=True, identity_consumed=consumed, point_age_bars=point_age)
            if plan.get("entry_allowed"):
                selected = "native_fresh"
        elif (bool(feature.get("wyckoff_rule_buy", False)) and _wyckoff_visible(feature, day)
                and bool(base.get("eligible")) and not native_exit and not w_exit):
            stop = float(feature.get("wyckoff_stop", np.nan))
            atr = close * float(feature.get("atr14_ratio", np.nan))
            if np.isfinite([close, stop, atr]).all() and 0 < stop < close and atr > 0:
                stop = stop - parameters["stop_atr_buffer"] * atr
                ceiling = close + atr
                plan_id = stable_hash({"symbol": symbol, "candidate_policy": candidate_policy,
                    "event": feature.get("wyckoff_event"), "available_at": feature.get("wyckoff_available_at"),
                    "anchor_at": feature.get("wyckoff_anchor_at"), "support": stop, "signal_date": day})
                fraction = parameters["risk_fraction"]
                weight = min(settings["execution"]["max_weight"], fraction * close / (close - stop)) if entry_policy == "risk" else settings["execution"]["max_weight"]
                plan = {"version": "wyckoff_union_entry_plan_v1", "plan_id": plan_id,
                    "symbol": symbol, "signal_date": day, "available_at": day + "T15:00:00+08:00",
                    "policy": entry_policy, "status": "active", "entry_allowed": plan_id not in consumed,
                    "valid_for_sessions": 1, "reference_price": close, "reference_price_source": "signal_day_close",
                    "price_floor": stop + .01 if entry_policy == "risk" else None,
                    "price_ceiling": ceiling if entry_policy == "risk" else None,
                    "stop_price": stop if entry_policy == "risk" else None,
                    "risk_fraction": fraction if entry_policy == "risk" else None,
                    "target_weight": weight, "wyckoff_support": stop,
                    "candidate_policy": candidate_policy, "parameters": {**parameters, "max_next_open_gap_atr": 1.},
                    "experimental": True, "reason_codes": []}
                if stop > 0 and plan["entry_allowed"]:
                    selected = "wyckoff_test_lps"
        if selected:
            active, origin = copy.deepcopy(plan), selected
            consumed.add(plan["plan_id"])
            row.update(target_weight=plan["target_weight"], action="BUY", position_intent="BUY", candidate_action="BUY",
                entry_plan=plan, entry_gate_passed=True, candidate_origin=origin, candidate_policy=candidate_policy)
        else:
            row.update(target_weight=0., action="WAIT", position_intent="WAIT", entry_gate_passed=False,
                entry_plan=None, candidate_origin=None, candidate_policy=candidate_policy)
        result.append(row)
    return result


def candidate_feature_row(feature, state, plan, *, initial_cash, target_weight):
    values = {key: feature.get(key, np.nan) for key in POLICY_FEATURE_COLUMNS}
    close = float(state["close"])
    risk = plan.get("policy") == "risk"
    values.update(candidate_cash_ratio=float(state["cash"]) / initial_cash,
        candidate_account_return=float(state["equity"]) / initial_cash - 1,
        candidate_stop_distance=close / float(plan["stop_price"]) - 1 if risk and plan.get("stop_price") else 0.,
        candidate_floor_distance=float(plan["price_floor"]) / close - 1 if risk and plan.get("price_floor") else 0.,
        candidate_ceiling_distance=float(plan["price_ceiling"]) / close - 1 if risk and plan.get("price_ceiling") else 0.,
        candidate_risk_fraction=float(plan.get("risk_fraction") or 0.),
        candidate_target_weight=float(target_weight), candidate_risk_plan=float(risk))
    eligible = (bool(feature.get("input_eligible", False)) and bool(feature.get("model_input_eligible", False))
        and _wyckoff_visible(feature, state["date"])
        and np.isfinite([values[key] for key in (*MA_TREND_COLUMNS, *POLICY_STATE_COLUMNS)]).all())
    return {**values, "input_eligible": bool(eligible), "symbol": feature["symbol"], "date": state["date"]}


def build_policy_dataset(*, features, raw, candidate_decisions, market_dates, start, initial_cash, config,
                         behavior_decisions=None, entry_policy="fresh", candidate_policy="fresh_v1",
                         behavior_policy="rules_fresh_v1", continuation_policy_hash=None,
                         source_snapshot_hash="explicit_fixture_source", frozen_exit_policy=None,
                         behavior_exit_policy=None, reference_callback=None, candidate_callback=None,
                         row_behavior_policy=None):
    """Label every frozen candidate using its actual flat cash/risk Broker clone.

    The reference behavior creates contemporaneous state, not a selected training
    sample. An entry rejected by the clone and an unclosed episode are retained
    with unavailable targets; neither becomes a negative. Independent accounts
    permit exact flat-state cloning without inventing cross-symbol liquidity.
    """
    behavior = behavior_decisions if behavior_decisions is not None else candidate_decisions
    _aligned(features, raw, candidate_decisions)
    _aligned(features, raw, behavior)
    contract = policy_contract(config, market_dates, entry_policy, candidate_policy=candidate_policy,
        behavior_policy=behavior_policy, continuation_policy_hash=continuation_policy_hash,
        source_snapshot_hash=source_snapshot_hash)
    account = execute_decisions(str(raw.iloc[0].symbol), raw, behavior, start, initial_cash, config,
        "wyckoff-policy-label-behavior", market_dates=market_dates, verified_sources=VERIFIED_SOURCES,
        entry_policy=entry_policy, exit_policy=behavior_exit_policy)
    if reference_callback:
        reference_callback(account)
    observed = {row["date"]: row for row in account["daily"]}
    records, clone_reasons = [], Counter()
    market = features.to_dict("records")
    candidates_seen = 0
    from my_strategy.adapters.czsc_adapter import legacy_fuyao_source
    valid_indices = np.flatnonzero(np.array([row.volume > 0 and row.amount > 0 and row.has_trade_price == 1
        and not legacy_fuyao_source(row.source) for row in raw.itertuples(index=False)]))
    seasoned_required = int(config["execution"]["seasoned_bars"])
    for index, intent in enumerate(candidate_decisions):
        plan = intent.get("entry_plan")
        if not isinstance(plan, dict) or plan.get("status") != "active" or not plan.get("entry_allowed"):
            continue
        day = intent["date"]
        if day < start:
            continue
        candidates_seen += 1
        state = observed[day]
        values = candidate_feature_row(market[index], state, plan, initial_cash=initial_cash,
            target_weight=intent["target_weight"])
        record = {**values, "label": np.nan, "label_available": False, "label_end": None,
            "label_reason": "candidate_not_decidable_existing_position", "net_return": np.nan,
            "entry_date": None, "exit_date": None, "entry_shares": 0, "fees": np.nan,
            "candidate_policy": candidate_policy, "behavior_policy": row_behavior_policy or behavior_policy,
            "continuation_policy_hash": contract["continuation_policy_hash"], "source_snapshot_hash": source_snapshot_hash,
            "entry_policy": entry_policy, "candidate_id": plan.get("plan_id"),
            "candidate_plan_hash": stable_hash(plan), "behavior_cash": float(state["cash"]),
            "behavior_shares": int(state["shares"]), "state_observed_at": day + "T15:00:00+08:00",
            "behavior_state_hash": stable_hash(state), "clone_ledger_hash": None,
            "entry_rejection_count": 0, "exit_rejection_count": 0,
            "entry_first_rejection": None, "clone_initial_cash": float(state["cash"])}
        if state["shares"] == 0:
            # Reconstruct the exact observed flat Broker state (cash and no cost
            # basis), retaining full price prehistory for seasoned/next-open guards.
            known_valid = valid_indices[valid_indices <= index]
            begin = int(known_valid[-seasoned_required - 1]) if len(known_valid) > seasoned_required else 0
            # Once seasoned, only the count gate (>= threshold) is used by
            # Broker; the bounded price prefix reproduces it exactly. Extend
            # the outcome slice if partial/rejected exits remain unclosed.
            native_end = next((offset for offset in range(index + 1, len(candidate_decisions))
                if candidate_decisions[offset]["target_weight"] == 0), len(candidate_decisions) - 1)
            finish = min(len(raw), native_end + 32)
            def cloned_exit(snapshot):
                # Cash is cloned from the observed account, but normalizing
                # capital remains that account's original initial capital.
                # This also protects future frozen callbacks using this field.
                snapshot["initial_cash"] = initial_cash
                return frozen_exit_policy(snapshot)
            while True:
                replay = []
                for offset in range(begin, finish):
                    decision = candidate_decisions[offset]
                    row = copy.deepcopy(decision)
                    if offset < index:
                        row.update(target_weight=0., action="WAIT", entry_plan=None)
                    elif offset > index:
                        row["entry_plan"] = None  # no reinvestment in this candidate episode
                    replay.append(row)
                clone = execute_decisions(str(raw.iloc[0].symbol), raw.iloc[begin:finish].reset_index(drop=True),
                    replay, day, float(state["cash"]), config, "wyckoff-policy-candidate-clone",
                    market_dates=market_dates, verified_sources=VERIFIED_SOURCES,
                    entry_policy=entry_policy, exit_policy=cloned_exit if frozen_exit_policy else None)
                if clone["metrics"]["open_shares"] == 0 or finish == len(raw):
                    break
                finish = min(len(raw), max(finish + 32, index + 2 * (finish - index)))
            entries = clone["ledger"]
            buys = [row for row in entries if row["action"] == "BUY"]
            buy_rejections = [row for row in clone["rejections"] if row["action"] == "BUY"
                and row.get("signal_date") == day]
            sell_rejections = [row for row in clone["rejections"] if row["action"] == "SELL"]
            record.update(clone_ledger_hash=stable_hash(entries), entry_rejection_count=len(buy_rejections),
                exit_rejection_count=len(sell_rejections),
                entry_first_rejection=buy_rejections[0]["reason"] if buy_rejections else None)
            if not buys:
                record["label_reason"] = "entry_rejected:" + buy_rejections[0]["reason"] if buy_rejections else "entry_unobserved_or_calendar_tail"
            else:
                buy = buys[0]
                record.update(entry_date=buy["date"], entry_shares=int(buy["shares"]), label_reason="episode_unclosed")
                # No other entry can be accepted: all later plans are absent.
                if len(buys) != 1:
                    raise ValueError("policy candidate clone illegally reinvested")
                if clone["metrics"]["open_shares"] == 0:
                    net = sum(float(row["cash_flow"]) for row in entries)
                    end = entries[-1]["date"]
                    record.update(label=float(net > 0), label_available=True, label_end=end,
                        label_reason="available", net_return=net / float(state["cash"]),
                        exit_date=end, fees=sum(float(row["fee"]) for row in entries))
            if candidate_callback:
                candidate_callback(record, clone)
        clone_reasons[record["label_reason"]] += 1
        records.append(record)
    extras = ("symbol", "date", "input_eligible", "label", "label_available", "label_end", "label_reason",
        "net_return", "entry_date", "exit_date", "entry_shares", "fees", "candidate_policy", "behavior_policy",
        "continuation_policy_hash", "source_snapshot_hash", "entry_policy", "candidate_id", "candidate_plan_hash",
        "behavior_cash", "behavior_shares", "state_observed_at", "behavior_state_hash", "clone_ledger_hash",
        "entry_rejection_count", "exit_rejection_count", "entry_first_rejection", "clone_initial_cash")
    frame = _bind(pd.DataFrame(records, columns=[*POLICY_FEATURE_COLUMNS, *extras]), contract, head="policy_entry")
    frame.attrs.update(data_version=stable_hash({"raw": raw.attrs.get("data_version"),
        "candidate_decisions": candidate_decisions, "behavior_decisions": behavior,
        "contract": contract["contract_hash"]}), candidate_counts={"all_frozen_candidates": candidates_seen,
        "decidable_flat_candidates": int((frame.behavior_shares == 0).sum()),
        "mature": int(frame.label_available.sum()), "ineligible": int((~frame.input_eligible).sum())},
        label_reason_counts=dict(clone_reasons), reference_account_metrics=account["metrics"],
        selection="all_frozen_candidates_before_final_model_selection")
    return frame


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _train(dataset, output_dir, train_end, validation_start, validation_end, *, head, device="auto",
           seed=42, epochs=20, hidden_sizes=(), **kwargs):
    from my_strategy.services.czsc_research_ml import train_model
    _check_contract(dataset.attrs.get("label_contract", {}), head=head)
    exit_head = head == "holding_exit"
    columns = EXIT_FEATURE_COLUMNS if exit_head else POLICY_FEATURE_COLUMNS
    version = EXIT_FEATURE_VERSION if exit_head else POLICY_FEATURE_VERSION
    schema = EXIT_FEATURE_SCHEMA_HASH if exit_head else POLICY_FEATURE_SCHEMA_HASH
    if (dataset.attrs.get("feature_version") != version or dataset.attrs.get("schema_hash") != schema
            or dataset.attrs.get("feature_columns") != list(columns)
            or dataset.attrs.get("label_version") != (EXIT_LABEL_VERSION if exit_head else POLICY_LABEL_VERSION)):
        raise ValueError("Wyckoff independent training feature order/version mismatch")
    if hidden_sizes:
        raise ValueError("first-stage Wyckoff independent heads require an immutable linear CPU export")
    manifest = train_model(dataset, columns, output_dir, train_end, validation_start, validation_end,
        device=device, seed=seed, epochs=epochs, hidden_sizes=(), **kwargs)
    _export_head(output_dir, head=head)
    return manifest


def train_exit_model(dataset, output_dir, train_end, validation_start, validation_end, **kwargs):
    return _train(dataset, output_dir, train_end, validation_start, validation_end, head="holding_exit", **kwargs)


def train_policy_model(dataset, output_dir, train_end, validation_start, validation_end, **kwargs):
    return _train(dataset, output_dir, train_end, validation_start, validation_end, head="policy_entry", **kwargs)


def _export_head(output_dir, *, head):
    import torch
    from my_strategy.services.czsc_research_ml import _verified_manifest
    directory = Path(output_dir)
    manifest, checkpoint = _verified_manifest(directory)
    _check_contract(manifest.get("label_contract", {}), head=head)
    if manifest["schema"]["hidden_sizes"]:
        raise ValueError("Wyckoff CPU export requires a linear head")
    parameters = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    exit_head = head == "holding_exit"
    columns = EXIT_FEATURE_COLUMNS if exit_head else POLICY_FEATURE_COLUMNS
    exported = {"format": "wyckoff_independent_linear_cpu_numpy_v1", "head": head,
        "columns": list(columns), "feature_version": manifest["feature_version"],
        "feature_schema_hash": manifest["feature_schema_hash"],
        "contract_hash": manifest["label_contract"]["contract_hash"],
        "weight": parameters["0.weight"].numpy().reshape(-1).tolist(),
        "bias": float(parameters["0.bias"].numpy()[0]), "preprocess": manifest["preprocess"],
        "temperature": manifest["calibration"]["temperature"]}
    export_path = directory / ("exit-linear.json" if exit_head else "policy-linear.json")
    serialized = json.dumps(exported, ensure_ascii=False, indent=2, allow_nan=False)
    if export_path.exists() and export_path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError("Wyckoff immutable head export already exists with another identity")
    export_path.write_text(serialized, encoding="utf-8")
    binding = {"head": head, "label_version": manifest["label_version"],
        "contract_hash": exported["contract_hash"], "feature_schema_hash": exported["feature_schema_hash"],
        "entry_policy": manifest["label_contract"]["entry_policy"],
        "candidate_policy": manifest["label_contract"]["candidate_policy"],
        "behavior_policy": manifest["label_contract"]["behavior_policy"],
        "continuation_policy_hash": manifest["label_contract"]["continuation_policy_hash"],
        "source_snapshot_hash": manifest["label_contract"]["source_snapshot_hash"],
        "model_manifest_sha256": manifest["manifest_sha256"], "checkpoint_sha256": manifest["checkpoint_sha256"],
        "available_at": manifest["available_at"], "export": export_path.name,
        "export_sha256": _file_hash(export_path), "runtime": "cpu_numpy", "actual_training_device": manifest["device"]}
    binding["binding_sha256"] = stable_hash(binding)
    path = directory / "head.binding.json"
    serialized = json.dumps(binding, ensure_ascii=False, indent=2, allow_nan=False)
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError("Wyckoff immutable head binding already exists with another identity")
    path.write_text(serialized, encoding="utf-8")
    return binding


def export_exit_head(output_dir):
    return _export_head(output_dir, head="holding_exit")


def export_policy_head(output_dir):
    return _export_head(output_dir, head="policy_entry")


class _LinearPredictor:
    head = "holding_exit"
    columns = EXIT_FEATURE_COLUMNS
    feature_version = EXIT_FEATURE_VERSION
    schema_hash = EXIT_FEATURE_SCHEMA_HASH
    label_version = EXIT_LABEL_VERSION

    def __init__(self, model_bundle, device=None, *, expected_contract=None):
        from my_strategy.services.czsc_research_ml import _verified_manifest
        bundle = model_bundle if isinstance(model_bundle, dict) else {"model_dir": str(model_bundle)}
        self.directory = Path(bundle["model_dir"])
        self.manifest, _ = _verified_manifest(self.directory)
        self.contract = self.manifest.get("label_contract", {})
        _check_contract(self.contract, head=self.head)
        if (self.manifest.get("label_version") != self.label_version
                or self.manifest.get("feature_version") != self.feature_version
                or self.manifest.get("feature_schema_hash") != self.schema_hash
                or self.manifest.get("schema", {}).get("columns") != list(self.columns)
                or bundle.get("contract_hash", self.contract["contract_hash"]) != self.contract["contract_hash"]
                or expected_contract and expected_contract["contract_hash"] != self.contract["contract_hash"]):
            raise ValueError("entry/exit weights or incompatible policy cannot be used as a Wyckoff independent head")
        self.threshold = float(bundle.get("threshold", .55))
        if not np.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("Wyckoff head threshold must be finite within (0, 1)")
        self.requested_device = device
        self._expected_binding_hash = bundle.get("head_binding_sha256") or bundle.get("binding_sha256")
        self.exported = self._load_export()
        self._counts = {"rows": 0, "input_rows": 0, "batches": 0, "model_loads": 1, "hash_checks": 0, "cache_hits": 0}
        self._seconds = 0.

    def _load_export(self):
        binding = json.loads((self.directory / "head.binding.json").read_text(encoding="utf-8"))
        export_name = "exit-linear.json" if self.head == "holding_exit" else "policy-linear.json"
        if (binding.get("binding_sha256") != stable_hash({k: v for k, v in binding.items() if k != "binding_sha256"})
                or self._expected_binding_hash and binding["binding_sha256"] != self._expected_binding_hash
                or binding.get("head") != self.head or binding.get("label_version") != self.label_version
                or binding.get("runtime") != "cpu_numpy" or binding.get("contract_hash") != self.contract["contract_hash"]
                or binding.get("feature_schema_hash") != self.schema_hash
                or binding.get("model_manifest_sha256") != self.manifest["manifest_sha256"]
                or binding.get("checkpoint_sha256") != self.manifest["checkpoint_sha256"]
                or binding.get("available_at") != self.manifest["available_at"]
                or binding.get("export") != export_name or self.manifest["schema"]["hidden_sizes"]):
            raise ValueError("Wyckoff independent CPU export binding integrity mismatch")
        for key in ("entry_policy", "candidate_policy", "behavior_policy", "continuation_policy_hash", "source_snapshot_hash"):
            if binding.get(key) != self.contract[key]:
                raise ValueError("Wyckoff independent CPU export policy/source binding mismatch")
        path = self.directory / export_name
        if _file_hash(path) != binding["export_sha256"]:
            raise ValueError("Wyckoff independent CPU export file hash mismatch")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value.get("format") != "wyckoff_independent_linear_cpu_numpy_v1" or value.get("head") != self.head
                or value.get("columns") != list(self.columns) or value.get("feature_version") != self.feature_version
                or value.get("feature_schema_hash") != self.schema_hash or value.get("contract_hash") != self.contract["contract_hash"]
                or value.get("preprocess") != self.manifest["preprocess"]
                or value.get("temperature") != self.manifest["calibration"]["temperature"]
                or len(value.get("weight", [])) != len(self.columns)
                or not np.isfinite([*value["weight"], value["bias"], value["temperature"]]).all()
                or value["temperature"] <= 0):
            raise ValueError("Wyckoff independent CPU export semantic identity mismatch")
        return value

    @property
    def diagnostics(self):
        return {**self._counts, "head": self.head, "device": "cpu", "devices": ["cpu"],
            "requested_device": self.requested_device, "actual_cpu_numpy_inference": self._counts["rows"] > 0,
            "actual_gpu_inference": False, "actual_mps_inference": False, "actual_cuda_inference": False,
            "actual_training_device": self.manifest["device"], "actual_mps_training": self.manifest.get("actual_mps_training", False),
            "contract_hash": self.contract["contract_hash"], "inference_seconds": self._seconds}

    def _verify_now(self):
        from my_strategy.services.czsc_research_ml import _verified_manifest
        current, _ = _verified_manifest(self.directory)
        if current["manifest_sha256"] != self.manifest["manifest_sha256"]:
            raise ValueError("Wyckoff independent model changed during replay")
        self.exported = self._load_export()
        self._counts["hash_checks"] += 1

    def predict(self, frame, *, as_of=None):
        _check_contract(frame.attrs.get("label_contract", {}), head=self.head)
        if (frame.attrs["label_contract"]["contract_hash"] != self.contract["contract_hash"]
                or frame.attrs.get("feature_version") != self.feature_version
                or frame.attrs.get("schema_hash") != self.schema_hash
                or frame.attrs.get("feature_columns") != list(self.columns)):
            raise ValueError("Wyckoff independent inference feature/target contract mismatch")
        from my_strategy.services.czsc_research_ml import _features, _transform
        started = time.perf_counter()
        self._verify_now()
        dates = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
        if ((dates < self.manifest["available_at"]).any()
                or as_of is not None and (pd.Timestamp(as_of).date().isoformat() < self.manifest["available_at"]
                    or (dates > pd.Timestamp(as_of).date().isoformat()).any())):
            raise ValueError("Wyckoff independent checkpoint unavailable at signal/as_of")
        eligible = frame.input_eligible.fillna(False).astype(bool).to_numpy()
        transformed = _transform(_features(frame, self.columns), self.exported["preprocess"])
        result = np.full(len(frame), np.nan)
        logits = (transformed[eligible] @ np.asarray(self.exported["weight"], dtype=np.float32)
            + np.float32(self.exported["bias"])) / np.float32(self.exported["temperature"])
        result[eligible] = 1. / (1. + np.exp(-np.clip(logits, -40, 40)))
        self._counts["rows"] += int(eligible.sum())
        self._counts["input_rows"] += len(frame)
        self._counts["batches"] += int(eligible.any())
        self._counts["cache_hits"] += 1
        self._seconds += time.perf_counter() - started
        return result

    def predict_one(self, values, *, as_of):
        # Live held/candidate states are already formed by this module's exact
        # schema. Avoid a pandas construction/type conversion for every close.
        started = time.perf_counter()
        self._verify_now()
        day = pd.Timestamp(values["date"]).date().isoformat()
        signal = pd.Timestamp(as_of).date().isoformat()
        if day < self.manifest["available_at"] or signal < self.manifest["available_at"] or day > signal:
            raise ValueError("Wyckoff independent checkpoint unavailable at signal/as_of")
        eligible = bool(values["input_eligible"])
        probability = np.nan
        if eligible:
            numbers = np.asarray([values[key] for key in self.columns], dtype=np.float64)
            preprocess = self.exported["preprocess"]
            filled = np.where(np.isfinite(numbers), numbers, np.asarray(preprocess["median"]))
            transformed = np.clip((filled - np.asarray(preprocess["mean"])) / np.asarray(preprocess["scale"]), -8, 8).astype(np.float32)
            logit = (transformed @ np.asarray(self.exported["weight"], dtype=np.float32)
                + np.float32(self.exported["bias"])) / np.float32(self.exported["temperature"])
            probability = float(1. / (1. + np.exp(-np.clip(logit, -40, 40))))
        self._counts["rows"] += int(eligible)
        self._counts["input_rows"] += 1
        self._counts["batches"] += int(eligible)
        self._counts["cache_hits"] += 1
        self._seconds += time.perf_counter() - started
        return probability


class ExitPredictor(_LinearPredictor):
    """CPU inference bound to independently trained actual-held Wyckoff target."""


class PolicyPredictor(_LinearPredictor):
    head = "policy_entry"
    columns = POLICY_FEATURE_COLUMNS
    feature_version = POLICY_FEATURE_VERSION
    schema_hash = POLICY_FEATURE_SCHEMA_HASH
    label_version = POLICY_LABEL_VERSION


def make_exit_policy(*, features, predictor=None, predictor_resolver=None, apply_exit=False,
                     candidate_policy=None, behavior_policy=None, continuation_policy_hash=None):
    market = {pd.Timestamp(row["date"]).date().isoformat(): row for row in features.to_dict("records")}
    def policy(snapshot):
        selected = predictor_resolver(snapshot["date"]) if predictor_resolver is not None else predictor
        response = {"head": "holding_exit", "probability": None, "request_exit": False, "apply": False,
            "research_arm": bool(apply_exit), "status": "wyckoff_exit_model_unavailable"}
        if selected is None:
            return response
        contract = selected.contract
        _check_contract(contract)
        if (contract["entry_policy"] != snapshot["entry_policy"]
                or candidate_policy and contract["candidate_policy"] != candidate_policy
                or behavior_policy and contract["behavior_policy"] != behavior_policy
                or continuation_policy_hash and contract["continuation_policy_hash"] != continuation_policy_hash):
            return {**response, "status": "wyckoff_exit_policy_contract_unavailable"}
        response.update(contract_hash=contract["contract_hash"], threshold=selected.threshold,
            available_at=selected.manifest["available_at"])
        if snapshot["date"] < selected.manifest["available_at"]:
            return {**response, "status": "wyckoff_exit_checkpoint_unavailable_at_signal"}
        feature = market.get(snapshot["date"])
        if feature is None:
            return {**response, "status": "wyckoff_exit_market_input_missing"}
        values = holding_feature_row(feature, snapshot)
        if not values["input_eligible"]:
            return {**response, "status": "wyckoff_exit_input_ineligible"}
        probability = selected.predict_one(values, as_of=snapshot["date"])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            return {**response, "status": "wyckoff_exit_probability_unavailable"}
        return {**response, "probability": probability, "request_exit": probability >= selected.threshold,
            "apply": bool(apply_exit), "status": "wyckoff_exit_research_applied" if apply_exit else "wyckoff_exit_shadow"}
    return policy


def make_entry_policy(*, features, predictor=None, predictor_resolver=None, apply_entry=False,
                      candidate_policy=None, continuation_policy_hash=None):
    """An actual-policy head sees this candidate's own live flat cash state."""
    market = {pd.Timestamp(row["date"]).date().isoformat(): row for row in features.to_dict("records")}
    def gate(snapshot):
        selected = predictor_resolver(snapshot["date"]) if predictor_resolver is not None else predictor
        response = {"head": "policy_entry", "probability": None, "request_entry": False,
            "apply": False, "research_arm": bool(apply_entry), "status": "wyckoff_policy_model_unavailable"}
        if selected is None:
            return response
        contract = selected.contract
        _check_contract(contract, head="policy_entry")
        if (snapshot["shares"] != 0 or contract["entry_policy"] != snapshot["entry_policy"]
                or candidate_policy and contract["candidate_policy"] != candidate_policy
                or continuation_policy_hash and contract["continuation_policy_hash"] != continuation_policy_hash):
            return {**response, "status": "wyckoff_policy_contract_unavailable"}
        response.update(contract_hash=contract["contract_hash"], threshold=selected.threshold,
            available_at=selected.manifest["available_at"])
        if snapshot["date"] < selected.manifest["available_at"]:
            return {**response, "status": "wyckoff_policy_checkpoint_unavailable_at_signal"}
        feature = market.get(snapshot["date"])
        if feature is None:
            return {**response, "status": "wyckoff_policy_market_input_missing"}
        values = candidate_feature_row(feature, snapshot, snapshot["entry_plan"],
            initial_cash=snapshot["initial_cash"], target_weight=snapshot["target_weight"])
        bar = snapshot["bar"]
        if (not values["input_eligible"] or bar["source"] not in VERIFIED_SOURCES or bar["has_trade_price"] != 1
                or bar["volume"] <= 0 or bar["amount"] <= 0):
            return {**response, "status": "wyckoff_policy_input_ineligible"}
        probability = selected.predict_one(values, as_of=snapshot["date"])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            return {**response, "status": "wyckoff_policy_probability_unavailable"}
        return {**response, "probability": probability, "request_entry": probability >= selected.threshold,
            "apply": bool(apply_entry), "status": "wyckoff_policy_research_applied" if apply_entry else "wyckoff_policy_shadow"}
    return gate


def make_runtime_exit_policy(runtime, features):
    """Use verified same-source checkpoints; ordinary Broker execution stays shadow."""
    from my_strategy.services.czsc_analysis import strategy_config
    if runtime.entry_policy == "legacy":
        return None
    if not hasattr(runtime, "exit_predictors"):
        runtime.exit_predictors = {}
    routes = {}
    def route_for(day):
        # The route family is explicit, never inferred from a file's name.
        routes[day] = runtime.resolver.resolve(day)
        return routes[day]
    def resolve(day):
        route = routes.get(day) or route_for(day)
        if route.get("bundle_family") == "dual":
            return None
        binding = route.get("exit_model")
        if not route.get("model_dir") or not binding or binding.get("unavailable") or not binding.get("model_dir"):
            return None
        if binding.get("entry_policy", "fresh") != runtime.entry_policy:
            return None
        directory = binding["model_dir"]
        if directory not in runtime.exit_predictors:
            predictor = ExitPredictor(binding, device="cpu")
            contract = predictor.contract
            if (contract["strategy_hash"] != stable_hash(strategy_config())
                    or contract["calendar_hash"] != stable_hash(runtime.market_dates)
                    or contract["candidate_policy"] != "fresh_v1"):
                return None
            route_source = route.get("source_snapshot_hash")
            if route_source and route_source != contract["source_snapshot_hash"]:
                return None
            runtime.exit_predictors[directory] = predictor
        return runtime.exit_predictors[directory]
    from my_strategy.services.czsc_dual_exit import (ExitPredictor as DualExitPredictor,
        exit_contract as dual_exit_contract, make_exit_policy as dual_exit_policy)
    def resolve_dual(day):
        route = routes.get(day) or route_for(day)
        if route.get("bundle_family") != "dual":
            return None
        binding = route.get("exit_model")
        if not route.get("model_dir") or not binding or binding.get("unavailable") or not binding.get("model_dir"):
            return None
        if binding.get("entry_policy", "fresh") != runtime.entry_policy:
            return None
        key = "dual:" + binding["model_dir"]
        if key not in runtime.exit_predictors:
            expected = dual_exit_contract(strategy_config(), runtime.market_dates, runtime.entry_policy)
            runtime.exit_predictors[key] = DualExitPredictor(binding, device="cpu", expected_contract=expected)
        return runtime.exit_predictors[key]
    w_callback = make_exit_policy(features=features, predictor_resolver=resolve,
        apply_exit=False, candidate_policy="fresh_v1")
    d_callback = dual_exit_policy(features=features, predictor_resolver=resolve_dual, apply_exit=False)
    def dispatch(snapshot):
        route = route_for(snapshot["date"])
        return d_callback(snapshot) if route.get("bundle_family") == "dual" else w_callback(snapshot)
    return dispatch


def make_runtime_entry_policy(runtime, features):
    """Actual-flat ordinary scoring of a different, hypothetical frozen exit.

    The ordinary account still executes native exits. The policy head's target
    is its sealed forward exit router, so the probability is explicitly
    counterfactual and never a qualification for this ordinary execution path.
    """
    from my_strategy.services.czsc_analysis import strategy_config
    if runtime.entry_policy == "legacy":
        return None
    if not hasattr(runtime, "policy_predictors"):
        runtime.policy_predictors = {}
    if not hasattr(runtime, "policy_reason_counts"):
        runtime.policy_reason_counts = Counter()
    reasons = {}
    def resolve(day):
        route = runtime.resolver.resolve(day)
        reasons[day] = "wyckoff_policy_model_unavailable"
        if route.get("bundle_family") == "dual":
            reasons[day] = "wyckoff_policy_dual_fallback_no_policy_head"
            return None
        # The ordinary candidate policy is fresh. Union risk heads score their
        # own explicit research arm and must not be relabelled as this policy.
        if runtime.entry_policy != "fresh":
            reasons[day] = "wyckoff_policy_candidate_contract_unavailable"
            return None
        binding = (route.get("policy_models") or {}).get("fresh")
        if not route.get("model_dir") or not binding or binding.get("unavailable") or not binding.get("model_dir"):
            return None
        directory = binding["model_dir"]
        if directory not in runtime.policy_predictors:
            predictor = PolicyPredictor(binding, device="cpu")
            contract = predictor.contract
            if (contract["strategy_hash"] != stable_hash(strategy_config())
                    or contract["calendar_hash"] != stable_hash(runtime.market_dates)
                    or contract["entry_policy"] != "fresh" or contract["candidate_policy"] != "fresh_v1"):
                reasons[day] = "wyckoff_policy_candidate_contract_unavailable"
                return None
            route_source = route.get("source_snapshot_hash")
            if route_source and route_source != contract["source_snapshot_hash"]:
                reasons[day] = "wyckoff_policy_source_contract_unavailable"
                return None
            runtime.policy_predictors[directory] = predictor
        return runtime.policy_predictors[directory]
    callback = make_entry_policy(features=features, predictor_resolver=resolve,
        apply_entry=False, candidate_policy="fresh_v1")
    def gate(snapshot):
        if not np.isclose(float(snapshot["initial_cash"]), 100000., rtol=0, atol=1e-6):
            response = {"head": "policy_entry", "probability": None, "request_entry": False,
                "apply": False, "research_arm": False, "status": "wyckoff_policy_initial_capital_contract_unavailable"}
        else:
            response = callback(snapshot)
        if response["probability"] is None and response["status"] == "wyckoff_policy_model_unavailable":
            response["status"] = reasons.get(snapshot["date"], response["status"])
        elif response["probability"] is not None:
            response["status"] = "wyckoff_policy_hypothetical_frozen_exit_shadow"
        response.update(target_scope="hypothetical_frozen_policy_roundtrip",
            target_initial_cash=100000., actual_account_initial_cash=float(snapshot["initial_cash"]),
            execution_policy_matches_target=False, ordinary_continuation_policy="native_rules",
            actual_account_context="actual_broker_flat_close", apply=False, applied=False,
            entry_policy_applied=False, exit_policy_applied=False)
        runtime.policy_reason_counts[response["status"]] += 1
        return response
    return gate


def evaluate_exit_arm(*, features, raw, base_decisions, market_dates, start, end, initial_cash,
                      config, model_bundle=None, device=None, apply_exit=False, entry_policy="fresh",
                      run_id="wyckoff-exit-arm", **contract_kwargs):
    _aligned(features, raw, base_decisions)
    contract = exit_contract(config, market_dates, entry_policy, **contract_kwargs)
    predictor = ExitPredictor(model_bundle, device, expected_contract=contract) if model_bundle is not None else None
    mask = pd.to_datetime(raw.date) <= pd.Timestamp(end)
    frame = raw.loc[mask].reset_index(drop=True)
    result = execute_decisions(str(frame.iloc[0].symbol), frame, base_decisions[:len(frame)], start, initial_cash,
        config, run_id, market_dates=market_dates, verified_sources=VERIFIED_SOURCES, entry_policy=entry_policy,
        exit_policy=make_exit_policy(features=features.loc[mask], predictor=predictor, apply_exit=apply_exit))
    result.update(exit_compute_info=predictor.diagnostics if predictor else {"head": "holding_exit", "rows": 0,
        "batches": 0, "actual_gpu_inference": False}, exit_contract=contract,
        exit_research_opt_in=bool(apply_exit), qualification="research_only_not_released")
    return result


def evaluate_policy_arm(*, features, raw, base_decisions, market_dates, start, end, initial_cash,
                        config, model_bundle=None, device=None, apply_entry=False, entry_policy="fresh",
                        exit_policy=None, predictor_resolver=None, run_id="wyckoff-policy-arm", **contract_kwargs):
    _aligned(features, raw, base_decisions)
    contract = policy_contract(config, market_dates, entry_policy, **contract_kwargs)
    predictor = PolicyPredictor(model_bundle, device, expected_contract=contract) if model_bundle is not None else None
    mask = pd.to_datetime(raw.date) <= pd.Timestamp(end)
    frame = raw.loc[mask].reset_index(drop=True)
    result = execute_decisions(str(frame.iloc[0].symbol), frame, base_decisions[:len(frame)], start, initial_cash,
        config, run_id, market_dates=market_dates, verified_sources=VERIFIED_SOURCES, entry_policy=entry_policy,
        exit_policy=exit_policy, entry_gate=make_entry_policy(features=features.loc[mask], predictor=predictor,
            predictor_resolver=predictor_resolver, apply_entry=apply_entry,
            candidate_policy=contract["candidate_policy"], continuation_policy_hash=contract["continuation_policy_hash"]))
    result.update(policy_compute_info=predictor.diagnostics if predictor else {"head": "policy_entry", "rows": 0,
        "batches": 0, "actual_gpu_inference": False}, policy_contract=contract,
        policy_research_opt_in=bool(apply_entry), qualification="research_only_not_released")
    return result
