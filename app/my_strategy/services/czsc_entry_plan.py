"""Point-identity entry plans, separate from frozen features and native Position.

Fresh fixes event reuse and expiry semantics. Risk adds explicitly experimental
price/risk assumptions; none of these thresholds imply proven return improvement.
"""
from __future__ import annotations

import math
from typing import Any

from my_strategy.core.config_loader import load_config
from my_strategy.core.run_context import stable_hash


ENTRY_POLICIES = ("legacy", "fresh", "risk")
PLAN_VERSION = "czsc_entry_plan_v1"


def finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return None
    result = str(value)
    return result if result and result not in {"nan", "NaT", "None"} else None


def entry_plan_parameters(parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    values = dict(load_config("czsc_entry_plan")["risk"] if parameters is None else parameters)
    for key in ("max_ma20_atr", "risk_fraction", "stop_atr_buffer"):
        number = finite(values.get(key))
        if number is None or number < 0 or key != "stop_atr_buffer" and number == 0:
            raise ValueError(f"invalid entry plan parameter: {key}")
        values[key] = number
    age = finite(values.get("max_point_age_bars"))
    if age is None or age < 0 or age != int(age) or values["risk_fraction"] > 1:
        raise ValueError("invalid entry plan age or risk fraction")
    values["max_point_age_bars"] = int(age)
    return values


def make_entry_plan(feature: dict[str, Any], *, symbol: str, signal_date: str,
                    reference_price: float, policy: str, max_weight: float,
                    parameters: dict[str, Any], eligible: bool, rule_buy: bool,
                    rule_sell: bool, ml_allowed: bool, identity_consumed: set[str],
                    before_start: bool = False, point_age_bars: float | None = None) -> dict[str, Any]:
    """Construct one close-known plan; execution still requires the next session.

    The caller invokes this only when theoretically flat. Consumed identities
    are local to that replay/account, never shared with personal holdings.
    """
    if policy not in ENTRY_POLICIES:
        raise ValueError("unsupported entry policy")
    identity = {key: _text(feature.get(key)) for key in
                ("point_type", "point_anchor", "point_confirmed_at")}
    identified = all(identity.values()) and identity["point_type"] in {"二买", "三买", "二买辅助", "三买辅助"}
    plan_id = stable_hash({"symbol": symbol, "policy": policy, **identity}) if identified else None
    close = finite(reference_price)
    ma20, atr_ratio = finite(feature.get("ma20")), finite(feature.get("atr14_ratio"))
    atr = close * atr_ratio if close is not None and atr_ratio is not None and atr_ratio > 0 else None
    low, age = finite(feature.get("structure_low")), finite(point_age_bars)
    stop = low - parameters["stop_atr_buffer"] * atr if low is not None and atr is not None else None
    ceiling = ma20 + parameters["max_ma20_atr"] * atr if ma20 is not None and atr is not None else None
    distance = (close - ma20) / atr if close is not None and ma20 is not None and atr is not None else None
    reasons = []
    if before_start:
        reasons.append("pre_start_warmup")
    elif not eligible:
        reasons.append("input_ineligible")
    elif rule_sell:
        reasons.append("opposing_exit_signal")
    elif not rule_buy:
        reasons.append("rule_buy_not_met")
    elif not ml_allowed:
        reasons.append("ml_entry_veto")
    elif policy != "legacy" and not identified:
        reasons.append("buy_point_identity_unavailable")
    elif policy != "legacy" and plan_id in identity_consumed:
        reasons.append("buy_point_already_consumed")
    if not reasons and policy == "risk":
        if close is None or close <= 0 or ma20 is None or ma20 <= 0 or atr is None or low is None or age is None:
            reasons.append("risk_inputs_unavailable")
        else:
            if age > parameters["max_point_age_bars"]:
                reasons.append("point_age_exceeds_experimental_limit")
            if close > ceiling:
                reasons.append("ma20_distance_exceeds_experimental_limit")
            if stop <= 0 or stop >= close:
                reasons.append("invalid_structure_stop")
    allowed = not reasons
    weight = max_weight
    if policy == "risk" and allowed:
        weight = min(max_weight, parameters["risk_fraction"] * close / (close - stop))
    status = "active" if allowed else "none" if reasons == ["pre_start_warmup"] or reasons == ["rule_buy_not_met"] else "rejected"
    return {"version": PLAN_VERSION, "plan_id": plan_id, "symbol": symbol,
            "signal_date": signal_date, "available_at": signal_date + "T15:00:00+08:00",
            "created_at": signal_date + "T15:00:00+08:00",
            "identity": identity, "policy": policy, "status": status,
            "valid_for_sessions": 1, "entry_allowed": allowed, "reason_codes": reasons,
            "reference_price": close, "reference_price_source": "signal_day_close",
            "ma20": ma20, "atr14": atr, "structure_low": low,
            "price_floor": stop + .01 if policy == "risk" and stop is not None else None,
            "price_ceiling": ceiling if policy == "risk" else None,
            "stop_price": stop if policy == "risk" else None,
            "risk_fraction": parameters["risk_fraction"] if policy == "risk" else None,
            "target_weight": weight if allowed else 0., "parameters": dict(parameters),
            "experimental": policy == "risk", "diagnostics": {
                "point_age_bars": age, "point_age_source": "point_confirmed_at_observed_daily_bars",
                "finished_bi_age_bars": finite(feature.get("confirmed_age")), "ma20_distance_atr": distance,
                "structure_stop_candidate": stop,
                "signal_distance_to_structure_stop": close - stop if close is not None and stop is not None else None,
                "weekly_price_direction": finite(feature.get("weekly_direction")),
                "weekly_bi_direction": finite(feature.get("weekly_bi_direction"))}}
