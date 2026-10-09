"""Entry-event freshness and experimental risk plans must be causal and bounded."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_research_decisions import replay_decisions


@pytest.fixture
def entry_input():
    dates = pd.bdate_range("2026-01-05", periods=8)
    raw = pd.DataFrame({"symbol": "600027.SH", "date": dates,
        "dt": dates + pd.Timedelta(hours=15), "id": np.arange(len(dates)), "close": 100.})
    features = pd.DataFrame({"date": dates, "symbol": "600027.SH", "input_eligible": True,
        "rule_buy": False, "rule_sell": False,
        "rule_price_volume_buy": False, "rule_price_volume_sell": False,
        "point_type": "二买辅助", "point_anchor": "2025-12-30",
        "point_confirmed_at": "2026-01-05T15:00:00+08:00",
        "confirmed_age": np.arange(len(dates)), "ma20": 98., "atr14_ratio": .02,
        "structure_low": 96., "weekly_direction": 1., "weekly_bi_direction": -1.})
    return features, raw


def test_fresh_holding_does_not_reset_native_stop_or_timeout(entry_input):
    features, raw = entry_input
    features.loc[[0, 1, 3], "rule_buy"] = True
    raw.loc[1, "close"] = 97.
    raw.loc[2:, "close"] = 91.
    legacy = replay_decisions(features, raw)
    fresh = replay_decisions(features, raw, entry_policy="fresh")
    assert legacy["decisions"][2]["action"] == "HOLD"  # prior LO reset the basis to 97
    assert fresh["decisions"][2]["action"] == "SELL"  # original 100 basis is retained
    held = fresh["decisions"][1]
    assert held["raw_rule_buy"] and held["candidate_action"] == "BUY"
    assert not held["entry_gate_passed"]
    assert held["signals"]["日线_研究_开仓V1"].startswith("其他")
    assert held["entry_plan"]["status"] == "holding"
    assert held["entry_plan"]["plan_id"] == fresh["events"][0]["entry_plan"]["plan_id"]
    assert fresh["events"][1]["entry_plan"]["status"] == "exit"
    assert fresh["decisions"][3]["entry_plan"]["reason_codes"] == ["buy_point_already_consumed"]
    assert fresh["decisions"][3]["action"] == "WAIT"


def test_new_identity_can_open_after_exit_but_conflicts_cannot(entry_input):
    features, raw = entry_input
    features.loc[[0, 2, 4, 6], "rule_buy"] = True
    features.loc[[1, 4, 5], "rule_sell"] = True
    features.loc[4:, "point_anchor"] = "2026-01-08"
    features.loc[4:, "point_confirmed_at"] = "2026-01-09T15:00:00+08:00"
    result = replay_decisions(features, raw, entry_policy="fresh")
    assert result["decisions"][2]["entry_plan"]["reason_codes"] == ["buy_point_already_consumed"]
    conflict = result["decisions"][4]
    assert conflict["candidate_action"] == "SELL" and conflict["action"] == "WAIT"
    assert conflict["entry_plan"]["reason_codes"] == ["opposing_exit_signal"]
    assert result["decisions"][6]["action"] == "BUY"
    assert result["events"][0]["entry_plan"]["plan_id"] != result["events"][-1]["entry_plan"]["plan_id"]


def test_fresh_reports_age_and_extension_without_experimental_veto(entry_input):
    features, raw = entry_input
    features.loc[0, "rule_buy"] = True
    features.loc[0, "confirmed_age"] = 22
    features.loc[0, "ma20"] = 80.
    fresh = replay_decisions(features, raw, entry_policy="fresh")
    plan = fresh["decisions"][0]["entry_plan"]
    assert plan["entry_allowed"] and plan["status"] == "active"
    assert plan["diagnostics"]["point_age_bars"] == 0
    assert plan["diagnostics"]["finished_bi_age_bars"] == 22
    assert plan["diagnostics"]["ma20_distance_atr"] == 10.
    assert plan["price_floor"] is None and plan["price_ceiling"] is None and plan["stop_price"] is None
    assert not plan["experimental"]
    assert plan["diagnostics"]["weekly_price_direction"] == 1.
    assert plan["diagnostics"]["weekly_bi_direction"] == -1.
    json.dumps(fresh, allow_nan=False)


@pytest.mark.parametrize("field,value,reason", [
    ("ma20", 90., "ma20_distance_exceeds_experimental_limit"),
    ("point_confirmed_at", None, "buy_point_identity_unavailable"),
    ("structure_low", np.nan, "risk_inputs_unavailable"),
    ("structure_low", 102., "invalid_structure_stop"),
])
def test_risk_hypotheses_reject_bad_plan_without_claiming_improvement(entry_input, field, value, reason):
    features, raw = entry_input
    features.loc[0, "rule_buy"] = True
    features.loc[0, field] = value
    result = replay_decisions(features, raw, entry_policy="risk")
    plan = result["decisions"][0]["entry_plan"]
    assert plan["experimental"] and not plan["entry_allowed"]
    assert reason in plan["reason_codes"]
    assert result["decisions"][0]["action"] == "WAIT"
    json.dumps(result, allow_nan=False)


def test_risk_target_stop_and_plan_identity_stay_frozen_while_holding(entry_input):
    features, raw = entry_input
    features.loc[[0, 1], "rule_buy"] = True
    features.loc[1:, "structure_low"] = 90.
    raw.loc[2:, "close"] = 95.
    result = replay_decisions(features, raw, entry_policy="risk")
    opened, held, exited = result["decisions"][:3]
    plan = opened["entry_plan"]
    assert plan["stop_price"] == 95.5 and plan["price_ceiling"] == 102.
    assert plan["risk_fraction"] == .01
    assert opened["target_weight"] == pytest.approx(.01 * 100 / (100 - 95.5))
    assert held["entry_plan"]["stop_price"] == plan["stop_price"]
    assert held["target_weight"] == opened["target_weight"]
    assert exited["action"] == "SELL" and exited["risk_exit_triggered"]
    assert exited["entry_plan"]["plan_id"] == plan["plan_id"]
    assert exited["entry_plan"]["reason_codes"] == ["structural_stop_triggered"]
    assert result["events"][-1]["reason"] == "结构失效价触发"


def test_fresh_starts_account_flat_without_consuming_warmup_identity(entry_input):
    features, raw = entry_input
    features.loc[[0, 3], "rule_buy"] = True
    start = raw.iloc[3]["date"].date().isoformat()
    fresh = replay_decisions(features, raw, entry_policy="fresh", position_start=start)
    assert all(row["action"] == "WAIT" and row["eligible"] for row in fresh["decisions"][:3])
    assert fresh["decisions"][0]["entry_plan"]["reason_codes"] == ["pre_start_warmup"]
    assert fresh["decisions"][3]["action"] == "BUY"
    assert fresh["decisions"][3]["account_context"] == "flat_at_start"
    assert fresh["decisions"][3]["position_start"] == start
    legacy = replay_decisions(features, raw, position_start=start)
    assert legacy["decisions"][0]["action"] == "BUY" and legacy["decisions"][3]["action"] == "HOLD"
    assert legacy["decisions"][0]["position_start"] is None


def test_fresh_future_append_does_not_change_prefix_and_uses_config(entry_input):
    features, raw = entry_input
    features.loc[0, "rule_buy"] = True
    settings = strategy_config()
    settings["position"]["timeout"] = 2
    full = replay_decisions(features, raw, entry_policy="fresh", config=settings)
    prefix = replay_decisions(features.iloc[:4], raw.iloc[:4], entry_policy="fresh", config=settings)
    assert prefix["decisions"] == full["decisions"][:4]
    assert "超时" in full["events"][1]["reason"]
    assert prefix["events"] == full["events"][:2]


def test_fresh_missing_buy_identity_fails_and_legacy_stays_compatible(entry_input):
    features, raw = entry_input
    features.loc[0, "rule_buy"] = True
    features.loc[0, "point_confirmed_at"] = None
    fresh = replay_decisions(features, raw, entry_policy="fresh")
    assert fresh["decisions"][0]["entry_plan"]["reason_codes"] == ["buy_point_identity_unavailable"]
    assert replay_decisions(features, raw)["decisions"][0]["action"] == "BUY"
    with pytest.raises(ValueError, match="entry policy"):
        replay_decisions(features, raw, entry_policy="unknown")
    with pytest.raises(ValueError, match="parameter"):
        replay_decisions(features, raw, entry_parameters={"risk_fraction": .01})


def test_legacy_entry_gate_keeps_raw_condition_while_fresh_means_new_open(entry_input):
    features, raw = entry_input
    features.loc[[0, 1], "rule_buy"] = True
    legacy = replay_decisions(features, raw)
    fresh = replay_decisions(features, raw, entry_policy="fresh")
    assert legacy["decisions"][1]["action"] == fresh["decisions"][1]["action"] == "HOLD"
    assert legacy["decisions"][1]["entry_gate_passed"]
    assert not fresh["decisions"][1]["entry_gate_passed"]


def test_risk_uses_buy_point_confirmation_age_not_latest_finished_pen_age(entry_input):
    features, raw = entry_input
    features.loc[6, "rule_buy"] = True
    features.loc[6, "confirmed_age"] = 0  # a different latest pen does not refresh this point
    parameters = {"max_ma20_atr": 2., "max_point_age_bars": 5,
                  "risk_fraction": .01, "stop_atr_buffer": .25}
    risk = replay_decisions(features, raw, entry_policy="risk", entry_parameters=parameters)
    plan = risk["decisions"][6]["entry_plan"]
    assert plan["diagnostics"]["point_age_bars"] == 6
    assert plan["diagnostics"]["finished_bi_age_bars"] == 0
    assert plan["reason_codes"] == ["point_age_exceeds_experimental_limit"]
    assert risk["decisions"][6]["action"] == "WAIT"
    fresh = replay_decisions(features, raw, entry_policy="fresh", entry_parameters=parameters)
    assert fresh["decisions"][6]["action"] == "BUY"
