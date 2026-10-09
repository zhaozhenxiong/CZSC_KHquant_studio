"""Shared research replay must retain the former native Position chronology."""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.czsc_analysis import IntentSession, strategy_config
from my_strategy.services.czsc_research_decisions import replay_decisions


@pytest.fixture
def replay_input():
    dates = pd.bdate_range("2024-01-02", periods=150)
    raw = pd.DataFrame({"symbol": "000001.SZ", "date": dates,
        "dt": dates + pd.Timedelta(hours=15), "id": np.arange(len(dates)), "close": 10.})
    features = pd.DataFrame({"date": dates, "input_eligible": True,
        "rule_buy": False, "rule_sell": False,
        "rule_price_volume_buy": False, "rule_price_volume_sell": False})
    features.loc[[5, 30, 75], ["rule_buy", "rule_price_volume_buy"]] = True
    features.loc[[20, 65, 130], ["rule_sell", "rule_price_volume_sell"]] = True
    return features, raw


def _former_replay(features, raw, mode, probabilities, threshold):
    """Pre-extraction contract, independent of the new replay function."""
    settings = copy.deepcopy(strategy_config())
    settings["position"].update(name="CZSC量价研究",
        opens=[{"operate": "开多", "signals_all": ["日线_研究_开仓V1_满足_任意_任意_0"]}],
        exits=[{"operate": "平多", "signals_all": ["日线_研究_退出V1_满足_任意_任意_0"]}])
    session = IntentSession(str(raw.iloc[0]["symbol"]), settings)
    result = []
    for index, (row, bar) in enumerate(zip(features.itertuples(index=False), raw.itertuples(index=False), strict=True)):
        eligible = bool(row.input_eligible)
        buy = bool(row.rule_price_volume_buy if mode == "price_volume" else row.rule_buy) and eligible
        sell = bool(row.rule_price_volume_sell if mode == "price_volume" else row.rule_sell)
        if mode == "ml":
            buy = buy and math.isfinite(float(probabilities[index])) and probabilities[index] >= threshold
        signals = {"日线_研究_开仓V1": ("满足" if buy else "其他") + "_任意_任意_0",
                   "日线_研究_退出V1": ("满足" if sell else "其他") + "_任意_任意_0"}
        if eligible:
            session.position.update({"symbol": session.symbol, "dt": bar.dt, "id": int(bar.id), "close": float(bar.close), **signals})
        date = pd.Timestamp(bar.date).date().isoformat()
        result.append({"date": date, "available_at": date + "T15:00:00+08:00",
                       "target_weight": float(session.position.pos) * settings["execution"]["max_weight"],
                       "eligible": eligible, "signals": signals})
    return result


@pytest.mark.parametrize("mode", ["rules", "price_volume", "ml"])
def test_native_parity_with_former_replay(replay_input, mode):
    features, raw = replay_input
    raw.loc[10, "close"] = 8.  # native stop-loss, even without a rule exit
    raw.loc[31, "close"] = 7.
    features.loc[31, "input_eligible"] = False
    probabilities = np.linspace(.3, .9, len(raw))
    expected = _former_replay(features, raw, mode, probabilities, .55)
    actual = replay_decisions(features, raw, mode=mode, probabilities=probabilities)["decisions"]
    assert [{key: row[key] for key in expected[0]} for row in actual] == expected


def test_events_include_native_stop_and_day_close_reference(replay_input):
    features, raw = replay_input
    raw.loc[10, "close"] = 8.
    result = replay_decisions(features, raw)
    first, second = result["events"][:2]
    assert first["action"] == "BUY" and second["action"] == "SELL"
    assert second["time"] == raw.iloc[10]["date"].date().isoformat()
    assert second["reference_price"] == 8.
    assert "止损" in second["reason"]
    assert result["decisions"][6]["candidate_action"] == "WAIT"
    assert result["decisions"][6]["position_intent"] == "HOLD"
    assert result["final_state"]["position_intent"] == "WAIT"
    assert {entry["role"] for entry in result["agent_evidence"]} == {"czsc", "price_volume", "ml", "data_execution"}


def test_native_timeout_retained_without_rule_exit(replay_input):
    features, raw = replay_input
    features[["rule_buy", "rule_sell"]] = False
    features.loc[5, "rule_buy"] = True
    result = replay_decisions(features, raw)
    assert len(result["events"]) == 2
    assert "超时" in result["events"][-1]["reason"]


def test_shadow_does_not_filter_and_model_switch_does_not_reset_position(replay_input):
    features, raw = replay_input
    values = np.zeros(len(raw))
    shadow = replay_decisions(features, raw, probabilities=values, apply_ml_mask=np.zeros(len(raw), dtype=bool))
    assert shadow["decisions"][5]["position_intent"] == "BUY"
    active = replay_decisions(features, raw, mode="ml", probabilities=values)
    assert all(row["position_intent"] == "WAIT" for row in active["decisions"])
    mask = np.ones(len(raw), dtype=bool)
    mask[:6] = False
    switched = replay_decisions(features, raw, probabilities=values, apply_ml_mask=mask,
                                model_metadata=[{"checkpoint": "old" if index < 6 else "new"} for index in range(len(raw))])
    assert switched["decisions"][6]["target_weight"] == shadow["decisions"][6]["target_weight"]
    assert switched["decisions"][6]["position_intent"] == "HOLD"
    assert switched["events"][1]["action"] == "SELL"  # exit is never vetoed by entry classifier
    assert switched["events"][1]["model"]["checkpoint"] == "new"


def test_ineligible_price_cannot_trigger_risk_exit_and_future_append_is_invariant(replay_input):
    features, raw = replay_input
    raw.loc[10, "close"] = 7.
    features.loc[10, "input_eligible"] = False
    full = replay_decisions(features, raw)
    assert full["decisions"][10]["position_intent"] == "HOLD"
    assert not any(row["time"] == raw.iloc[10]["date"].date().isoformat() for row in full["events"])
    prefix = replay_decisions(features.iloc[:40], raw.iloc[:40])
    assert prefix["decisions"] == full["decisions"][:40]
    assert prefix["events"] == [row for row in full["events"] if row["time"] <= prefix["latest_decision"]["date"]]


def test_misaligned_dates_mask_or_future_duplicate_rejected(replay_input):
    features, raw = replay_input
    changed = features.copy()
    changed.loc[0, "date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="feature dates"):
        replay_decisions(changed, raw)
    changed = raw.copy()
    changed.loc[1, "date"] = changed.loc[0, "date"]
    with pytest.raises(ValueError, match="unique ascending"):
        replay_decisions(features, changed)
    with pytest.raises(ValueError, match="per replay row"):
        replay_decisions(features, raw, apply_ml_mask=[True])
    with pytest.raises(ValueError, match="thresholds"):
        replay_decisions(features, raw, thresholds=float("nan"))
