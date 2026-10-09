"""Causal daily price/volume events; inferred supply/demand, not institutional truth.

Every range excludes the current bar. Pending events retain their original
boundary and cannot become buy evidence until a later observed test confirms it.
No completed-history chart or CZSC/MA signal is consumed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WYCKOFF_EVENT_VERSION = "wyckoff_causal_daily_events_v1"
DEFAULT_EVENT_CONFIG = {
    "range_bars": 60, "minimum_range_bars": 60,
    "range_max_width_atr": 20.0, "probe_atr": 0.15,
    "breakout_atr": 0.25, "breakout_relative_volume": 1.2,
    "test_volume_ratio": 0.8, "test_spread_ratio": 0.8,
    "test_max_bars": 10, "retest_max_bars": 15,
    "test_distance_atr": 1.0, "event_score_decay_bars": 20,
}


def event_config(config=None):
    value = dict(DEFAULT_EVENT_CONFIG)
    if config:
        incoming = config.get("events", config)
        value.update({key: incoming[key] for key in value if key in incoming})
    for key in ("range_bars", "minimum_range_bars", "test_max_bars", "retest_max_bars", "event_score_decay_bars"):
        if int(value[key]) != value[key] or value[key] < 1:
            raise ValueError("invalid Wyckoff event window: " + key)
        value[key] = int(value[key])
    if value["minimum_range_bars"] > value["range_bars"]:
        raise ValueError("minimum range exceeds trailing range")
    for key in set(value) - {"range_bars", "minimum_range_bars", "test_max_bars", "retest_max_bars", "event_score_decay_bars"}:
        if not np.isfinite(value[key]) or value[key] <= 0:
            raise ValueError("invalid Wyckoff event threshold: " + key)
    return value


def detect_wyckoff_events(raw, metrics, eligible, config=None):
    """Return same-index metadata and 10 event inputs from a forward state machine.

    ``metrics`` contains prior ATR, volume ratio, and prior rolling boundaries.
    ``eligible`` is independent raw quality eligibility; invalid bars clear
    pending state rather than inventing a test across a missing/unsafe interval.
    """
    settings = event_config(config)
    n = len(raw)
    arrays = {key: pd.to_numeric(raw[key], errors="coerce").to_numpy(dtype=float)
              for key in ("open", "high", "low", "close", "volume")}
    values = {key: np.asarray(metrics[key], dtype=float) for key in
              ("atr", "relative_volume", "range_low", "range_high", "range_count")}
    dates = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d").tolist()
    available = [day + "T15:00:00+08:00" for day in dates]
    columns = ["spring", "test", "sos", "lps", "upthrust", "sow", "lpsy"]
    flags = {"w_" + key: np.zeros(n, dtype=float) for key in columns}
    ages = np.full(n, np.nan)
    demand = np.zeros(n, dtype=float)
    supply = np.zeros(n, dtype=float)
    events, states, anchors, observations = [], [], [], []
    provenance, stops, buys, sells, evidence = [], np.full(n, np.nan), [], [], []
    range_low = np.full(n, np.nan); range_high = np.full(n, np.nan)
    pending = None
    last_event = None
    state = "unknown"
    for i in range(n):
        op, hi, lo, cl, vol = [arrays[key][i] for key in ("open", "high", "low", "close", "volume")]
        atr, rel, support, resistance, count = [values[key][i] for key in
            ("atr", "relative_volume", "range_low", "range_high", "range_count")]
        spread = hi - lo
        valid_range = (eligible[i] and np.isfinite([atr, support, resistance, rel]).all()
                       and atr > 0 and resistance > support and count >= settings["minimum_range_bars"]
                       and (resistance - support) / atr <= settings["range_max_width_atr"])
        event, anchor, observed, entry = "none", None, None, []
        buy, sell = False, False
        if not valid_range:
            pending, last_event, state = None, None, "unknown"
            event = "unknown"
        else:
            range_low[i], range_high[i] = support, resistance
            if pending and i - pending["index"] > pending["max_age"]:
                pending, state = None, "range"
            spring = lo < support - settings["probe_atr"] * atr and cl >= support
            upthrust = hi > resistance + settings["probe_atr"] * atr and cl <= resistance
            sos = (cl > resistance + settings["breakout_atr"] * atr and cl > op
                   and rel >= settings["breakout_relative_volume"])
            sow = (cl < support - settings["breakout_atr"] * atr and cl < op
                   and rel >= settings["breakout_relative_volume"])
            if spring and upthrust:
                pending, last_event, state, event = None, None, "conflict", "conflict"
            elif pending and i > pending["index"]:
                p = pending
                small_test = (vol <= p["volume"] * settings["test_volume_ratio"]
                              and spread <= p["spread"] * settings["test_spread_ratio"])
                if p["kind"] == "spring" and (lo < p["support"] - settings["probe_atr"] * atr or cl < p["support"]):
                    pending, state, event = None, "failed_demand", "spring_invalidated"
                elif p["kind"] == "spring" and small_test and (lo <= p["support"] + settings["test_distance_atr"] * atr
                                                              and lo >= p["support"] - settings["probe_atr"] * atr
                                                              and cl >= p["support"] + settings["probe_atr"] * atr):
                    event, state, buy = "Test", "demand_test_confirmed", True
                    anchor = p["anchor"]; stops[i] = min(p["low"], lo)
                    entry = ["spring_after_prior_range", "later_lower_volume_and_spread_test", "frozen_support_held"]
                    pending = None
                elif p["kind"] == "sos" and cl < p["support"] - settings["probe_atr"] * atr:
                    pending, state, event = None, "failed_demand", "sos_invalidated"
                elif p["kind"] == "sos" and small_test and (lo <= p["support"] + settings["test_distance_atr"] * atr
                                                           and cl >= p["support"] and cl > op):
                    event, state, buy = "LPS", "demand_retest_confirmed", True
                    anchor = p["anchor"]; stops[i] = min(p["support"], lo)
                    entry = ["earlier_observed_sos", "later_lower_volume_and_spread_pullback", "frozen_breakout_support_held"]
                    pending = None
                elif p["kind"] in ("sow", "upthrust") and cl > p["resistance"] + settings["probe_atr"] * atr:
                    pending, state, event = None, "failed_supply", "supply_invalidated"
                elif p["kind"] in ("sow", "upthrust") and small_test and (hi >= p["resistance"] - settings["test_distance_atr"] * atr
                                                                          and cl <= p["resistance"] and cl < op):
                    event, state, sell = "LPSY", "supply_retest_confirmed", True
                    anchor = p["anchor"]
                    entry = ["earlier_observed_supply", "later_weak_low_volume_rally", "frozen_resistance_not_recovered"]
                    pending = None
            # New evidence is evaluated only after a pending confirmation. A
            # confirmation wins over a newly moving rolling boundary.
            if event == "none":
                if spring:
                    event, state = "Spring", "spring_pending"
                    pending = {"kind": "spring", "index": i, "anchor": dates[i], "support": support,
                               "resistance": resistance, "volume": vol, "spread": spread, "low": lo,
                               "max_age": settings["test_max_bars"]}
                elif upthrust:
                    event, state, sell = "Upthrust", "upthrust_pending", True
                    pending = {"kind": "upthrust", "index": i, "anchor": dates[i], "support": support,
                               "resistance": resistance, "volume": vol, "spread": spread,
                               "max_age": settings["retest_max_bars"]}
                elif sos:
                    event, state = "SOS", "sos_pending"
                    pending = {"kind": "sos", "index": i, "anchor": dates[i], "support": resistance,
                               "resistance": resistance, "volume": vol, "spread": spread,
                               "max_age": settings["retest_max_bars"]}
                elif sow:
                    event, state, sell = "SOW", "sow_pending", True
                    pending = {"kind": "sow", "index": i, "anchor": dates[i], "support": support,
                               "resistance": support, "volume": vol, "spread": spread,
                               "max_age": settings["retest_max_bars"]}
                elif pending is None and state in ("unknown", "failed_demand", "failed_supply", "conflict"):
                    state = "range"
            event_key = event.lower()
            if event_key in columns:
                flags["w_" + event_key][i] = 1.0
                anchor = anchor or dates[i]
                observed = available[i]
                last_event = {"index": i, "kind": event, "anchor": anchor, "observed": observed}
            if last_event:
                ages[i] = i - last_event["index"]
                decay = max(0.0, 1.0 - ages[i] / settings["event_score_decay_bars"])
                weight = {"Spring": .5, "Test": 1., "SOS": .75, "LPS": 1.,
                          "Upthrust": .5, "SOW": .75, "LPSY": 1.}[last_event["kind"]]
                if last_event["kind"] in ("Spring", "Test", "SOS", "LPS"):
                    demand[i] = weight * decay
                else:
                    supply[i] = weight * decay
            if event.endswith("invalidated"):
                last_event = None; demand[i] = supply[i] = 0.; ages[i] = np.nan
            if event == "none" and pending:
                anchor = pending["anchor"]
        events.append(event); states.append(state); anchors.append(anchor)
        observations.append(observed)
        provenance.append(available[i - 1] if valid_range and i else None)
        buys.append(buy); sells.append(sell); evidence.append(entry)
    result = pd.DataFrame(flags, index=raw.index)
    result["w_event_age"] = ages
    result["w_demand_score"] = demand; result["w_supply_score"] = supply
    result["wyckoff_event"] = pd.Series(events, index=raw.index, dtype=object)
    result["wyckoff_state"] = pd.Series(states, index=raw.index, dtype=object)
    result["wyckoff_anchor_at"] = pd.Series(anchors, index=raw.index, dtype=object)
    result["wyckoff_observed_at"] = pd.Series(observations, index=raw.index, dtype=object)
    result["wyckoff_available_at"] = pd.Series(available, index=raw.index, dtype=object)
    result["wyckoff_range_low"] = range_low; result["wyckoff_range_high"] = range_high
    result["wyckoff_range_formed_at"] = pd.Series(provenance, index=raw.index, dtype=object)
    result["wyckoff_stop"] = stops
    result["wyckoff_rule_buy"] = np.asarray(buys, dtype=bool)
    result["wyckoff_rule_sell"] = np.asarray(sells, dtype=bool)
    result["wyckoff_entry_evidence"] = pd.Series(evidence, index=raw.index, dtype=object)
    result["wyckoff_event_version"] = WYCKOFF_EVENT_VERSION
    return result
