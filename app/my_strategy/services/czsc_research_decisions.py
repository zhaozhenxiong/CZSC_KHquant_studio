"""One chronological native Position replay for research entry gates and exits."""
from __future__ import annotations

import copy
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from my_strategy.services.czsc_analysis import IntentSession, strategy_config
from my_strategy.services.czsc_entry_plan import ENTRY_POLICIES, entry_plan_parameters, make_entry_plan


def _rows(value: Any, length: int, name: str, *, dtype) -> np.ndarray:
    result = np.full(length, value, dtype=dtype) if np.isscalar(value) else np.asarray(value, dtype=dtype)
    if result.shape != (length,):
        raise ValueError(f"{name} must have one value per replay row")
    return result


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def replay_decisions(features: pd.DataFrame, raw: pd.DataFrame, *,
                     probabilities: Sequence[float] | None = None,
                     thresholds: float | Sequence[float] = .55,
                     apply_ml_mask: Sequence[bool] | None = None,
                     mode: str = "rules",
                     model_metadata: Sequence[dict[str, Any]] | None = None,
                     entry_policy: str = "legacy", entry_parameters: dict[str, Any] | None = None,
                     position_start: str | None = None,
                     config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Replay the existing research signals without replacing native risk exits.

    Shadow probabilities never alter entry gates unless the caller explicitly
    supplies a true mask. Account state is a theoretical Position initialized
    flat at the beginning of the supplied prefix, separate from personal holdings.
    """
    if mode not in {"rules", "price_volume", "ml", "czsc_price_volume"}:
        raise ValueError("unsupported research replay mode")
    if entry_policy not in ENTRY_POLICIES:
        raise ValueError("unsupported entry policy")
    parameters = entry_plan_parameters(entry_parameters)
    start = pd.Timestamp(position_start).date().isoformat() if position_start is not None and entry_policy != "legacy" else None
    account_context = "flat_at_start" if start else "theoretical_prefix_position"
    if len(features) != len(raw):
        raise ValueError("research features and raw bars must have matching lengths")
    count = len(raw)
    values = _rows(probabilities if probabilities is not None else np.nan, count, "probabilities", dtype=float)
    limits = _rows(thresholds, count, "thresholds", dtype=float)
    if not np.isfinite(limits).all() or ((limits < 0) | (limits > 1)).any():
        raise ValueError("probability thresholds must be finite and within [0, 1]")
    applied = _rows(apply_ml_mask if apply_ml_mask is not None else mode == "ml", count, "apply_ml_mask", dtype=bool)
    metadata = list(model_metadata) if model_metadata is not None else [{} for _ in range(count)]
    if len(metadata) != count:
        raise ValueError("model_metadata must have one value per replay row")
    if not count:
        return {"decisions": [], "events": [], "latest_decision": None, "agent_evidence": [],
                "entry_policy": entry_policy, "entry_parameters": parameters,
                "final_state": {"position": 0., "target_weight": 0., "account_context": account_context,
                                "position_start": start}}
    dates = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    if dates.duplicated().any() or not dates.is_monotonic_increasing or raw["symbol"].nunique() != 1:
        raise ValueError("research replay requires one symbol with unique ascending dates")
    if "date" in features and not dates.reset_index(drop=True).equals(pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d").reset_index(drop=True)):
        raise ValueError("research feature dates do not match raw bars")
    if "symbol" in features and not features["symbol"].eq(raw.iloc[0]["symbol"]).all():
        raise ValueError("research feature symbol does not match raw bars")
    settings = copy.deepcopy(strategy_config(config))
    settings["position"].update(name="CZSC量价研究",
        opens=[{"operate": "开多", "signals_all": ["日线_研究_开仓V1_满足_任意_任意_0"]}],
        exits=[{"operate": "平多", "signals_all": ["日线_研究_退出V1_满足_任意_任意_0"]}])
    session = IntentSession(str(raw.iloc[0]["symbol"]), settings)
    date_ids = {date: index for index, date in enumerate(dates)}
    consumed: set[str] = set()
    active_plan: dict[str, Any] | None = None
    for index, (feature, bar) in enumerate(zip(features.to_dict("records"), raw.itertuples(index=False), strict=True)):
        eligible = bool(feature["input_eligible"])
        rule_buy = bool(feature["rule_price_volume_buy"] if mode == "price_volume" else feature["rule_buy"])
        rule_sell = bool(feature["rule_price_volume_sell"] if mode == "price_volume" else feature["rule_sell"])
        probability = _finite(values[index])
        ml_passed = probability is not None and probability >= limits[index]
        ml_allowed = not applied[index] or ml_passed
        raw_buy = eligible and rule_buy and ml_allowed
        date = str(dates.iloc[index])
        available = date + "T15:00:00+08:00"
        before = float(session.position.pos)
        before_start = start is not None and date < start
        point_index = date_ids.get(str(feature.get("point_confirmed_at"))[:10])
        point_age = float(index - point_index) if point_index is not None and point_index <= index else None
        plan = copy.deepcopy(active_plan) if before > 0 and active_plan else make_entry_plan(
            feature, symbol=session.symbol, signal_date=date, reference_price=float(bar.close),
            policy=entry_policy, max_weight=float(settings["execution"]["max_weight"]), parameters=parameters,
            eligible=eligible, rule_buy=rule_buy, rule_sell=rule_sell, ml_allowed=ml_allowed,
            identity_consumed=consumed, before_start=before_start, point_age_bars=point_age)
        if before > 0:
            plan.update(status="holding", entry_allowed=False, reason_codes=["already_positioned"])
        buy = raw_buy if entry_policy == "legacy" else before == 0 and plan["entry_allowed"]
        structural_exit = bool(entry_policy == "risk" and before > 0 and active_plan
                               and active_plan.get("stop_price") is not None
                               and float(bar.close) <= float(active_plan["stop_price"]))
        signals = {"日线_研究_开仓V1": ("满足" if buy else "其他") + "_任意_任意_0",
                   "日线_研究_退出V1": ("满足" if rule_sell or structural_exit else "其他") + "_任意_任意_0"}
        if eligible and not before_start:
            session.position.update({"symbol": session.symbol, "dt": bar.dt, "id": int(bar.id), "close": float(bar.close), **signals})
        position = float(session.position.pos)
        changed = position != before
        action = ("BUY" if position > before else "SELL") if changed else ("HOLD" if position > 0 else "WAIT")
        candidate = "WAIT" if not eligible else "SELL" if rule_sell or structural_exit else "BUY" if rule_buy else "WAIT"
        if action == "BUY":
            plan.update(status="active", entry_allowed=True, reason_codes=[])
            if entry_policy == "legacy":
                plan["target_weight"] = float(settings["execution"]["max_weight"])
            active_plan = copy.deepcopy(plan)
            if entry_policy != "legacy":
                consumed.add(plan["plan_id"])
        elif action == "SELL":
            plan.update(status="exit", entry_allowed=False,
                        reason_codes=["structural_stop_triggered" if structural_exit else "position_exit"])
        target = position * float(active_plan["target_weight"] if active_plan else settings["execution"]["max_weight"])
        evidence = [
            {"role": "czsc", "judgment": "sell" if feature.get("rule_czsc_sell") else "support" if feature.get("rule_czsc_buy") else "observe",
             "point_type": feature.get("point_type"),
             "confirmed_at": feature.get("point_confirmed_at") if pd.notna(feature.get("point_confirmed_at")) else None,
             "input_hash": feature.get("input_hash")},
            {"role": "price_volume", "judgment": "sell" if feature.get("rule_price_volume_sell") else "support" if feature.get("rule_price_volume_buy") else "observe",
             "values": {key: _finite(feature.get(key)) for key in ("ma20_bias", "ma60_bias", "vol_ratio20")}},
            {"role": "ml", "judgment": ("support" if ml_passed else "veto") if applied[index] else "shadow" if probability is not None else "unavailable",
             "probability": probability, "threshold": float(limits[index]), "applied": bool(applied[index]), "model": dict(metadata[index])},
            {"role": "data_execution", "judgment": "eligible_signal" if eligible else "veto",
             "reasons": list(feature.get("reason_codes", [])), "source": feature.get("source"),
             "execution_checked": False, "execution_contract": "next_verified_open_broker_guards"}]
        decision = {"date": date, "available_at": available,
                    "target_weight": target, "eligible": eligible, "signals": signals,
                    "candidate_action": candidate, "position_intent": action, "action": action,
                    "entry_gate_passed": raw_buy if entry_policy == "legacy" else action == "BUY",
                    "raw_rule_buy": rule_buy, "rule_buy": rule_buy, "rule_sell": rule_sell,
                    "risk_exit_triggered": structural_exit, "entry_policy": entry_policy, "entry_plan": plan,
                    "probability": probability, "threshold": float(limits[index]),
                    "ml_filter_applied": bool(applied[index]) and eligible,
                    "model": dict(metadata[index]), "agent_evidence": evidence,
                    "reference_price": float(bar.close), "reference_price_date": date,
                    "reference_price_source": "signal_day_close", "account_context": account_context,
                    "position_start": start}
        session.decisions.append(decision)
        if changed:
            operation = session.position.operates[-1]
            session.events.append({"time": date, "available_at": available, "action": action,
                "reason": "结构失效价触发" if structural_exit else str(operation.get("op_desc", "研究组合结构事件")),
                "signal": "; ".join(f"{key}={value}" for key, value in signals.items()),
                "target_weight": decision["target_weight"], "reference_price": float(bar.close),
                "reference_price_date": date, "reference_price_source": "signal_day_close",
                "anchor_time": feature.get("point_anchor") if pd.notna(feature.get("point_anchor")) else None,
                "confirmed_at": feature.get("point_confirmed_at") if pd.notna(feature.get("point_confirmed_at")) else None,
                "model": dict(metadata[index]), "ml_filter_applied": decision["ml_filter_applied"],
                "entry_policy": entry_policy, "entry_plan": copy.deepcopy(plan),
                "account_context": account_context, "position_start": start})
        if action == "SELL":
            active_plan = None
    latest = session.decisions[-1]
    return {"decisions": session.decisions, "events": session.events, "latest_decision": latest,
            "agent_evidence": latest["agent_evidence"], "entry_policy": entry_policy, "entry_parameters": parameters,
            "final_state": {"position": float(session.position.pos), "target_weight": latest["target_weight"],
                            "position_intent": latest["position_intent"], "date": latest["date"],
                            "account_context": account_context, "position_start": start}}
