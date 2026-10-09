"""Frozen entry plans change order eligibility, never broker accounting."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services.czsc_research_strategy_labels import build_strategy_labels


@pytest.fixture
def planned_input():
    dates = pd.bdate_range("2024-01-02", periods=100)
    raw = pd.DataFrame({"date": dates, "dt": dates + pd.Timedelta(hours=15),
        "id": np.arange(len(dates)), "symbol": "000001.SZ", "open": 10.,
        "high": 10.2, "low": 9.8, "close": 10., "volume": 1000000.,
        "amount": 10000000., "source": "tushare", "has_trade_price": 1})
    days = raw["date"].dt.strftime("%Y-%m-%d").tolist()
    plan = {"plan_id": "point-70", "symbol": "000001.SZ", "signal_date": days[70],
        "available_at": days[70] + "T15:00:00+08:00", "policy": "fresh", "status": "active",
        "entry_allowed": True, "valid_for_sessions": 1, "reference_price": 10.,
        "price_floor": None, "price_ceiling": None, "stop_price": None, "risk_fraction": None}
    decisions = [{"date": day, "eligible": True, "target_weight": .95 if 70 <= i < 80 else 0.,
        "action": "BUY" if i == 70 else "SELL" if i == 80 else "HOLD" if 70 < i < 80 else "WAIT"}
        for i, day in enumerate(days)]
    decisions[70]["entry_plan"] = plan
    for i in range(71, 80):
        decisions[i]["entry_plan"] = {**plan, "status": "holding", "entry_allowed": False}
    return raw, decisions, days


def _execute(raw, decisions, days, policy="fresh", start=None):
    return execute_decisions("000001.SZ", raw, decisions, start or days[0], 100000,
        strategy_config(), "planned-fixture", market_dates=days,
        verified_sources=VERIFIED_SOURCES, entry_policy=policy)


def test_legacy_is_unchanged_and_planned_fill_matches_same_broker(planned_input):
    raw, decisions, days = planned_input
    legacy = _execute(raw, decisions, days, "legacy")
    fresh = _execute(raw, decisions, days)
    for left, right in zip(legacy["ledger"], fresh["ledger"], strict=True):
        assert {k: v for k, v in left.items() if k != "reason"} == {k: v for k, v in right.items() if k != "reason"}
    assert legacy["metrics"] == fresh["metrics"]
    assert len(fresh["ledger"]) == 2
    assert fresh["entry_diagnostics"][0]["actual_cost_per_share"] > fresh["ledger"][0]["price"]


def test_warmup_hold_is_not_an_actual_account_entry(planned_input):
    raw, decisions, days = planned_input
    assert _execute(raw, decisions, days, "legacy", days[73])["ledger"]
    result = _execute(raw, decisions, days, start=days[73])
    assert result["ledger"] == []
    assert result["metrics"]["final_equity"] == 100000
    assert result["entry_diagnostics"][0]["reason"] == "entry_plan_not_active"


def test_rejected_buy_expires_and_theoretical_exit_has_no_fake_sale(planned_input):
    raw, decisions, days = planned_input
    raw.loc[71, "open"] = 10.6
    result = _execute(raw, decisions, days)
    assert result["ledger"] == []
    assert result["consumed_plan_ids"] == ["point-70"]
    assert any(x["reason"] == "conservative_upper_price_guard" for x in result["rejections"])
    assert _execute(raw, decisions, days, "legacy")["ledger"][0]["date"] == days[72]


def test_missing_next_session_never_fills_on_later_bar(planned_input):
    raw, decisions, days = planned_input
    missing = raw.drop(index=71)
    changed = decisions[:71] + decisions[72:]
    # Preserve the active plan as the immediately prior available signal.
    result = _execute(missing, changed, days)
    assert result["ledger"] == []
    assert any(x["reason"] == "entry_plan_expired" for x in result["rejections"])


def _risk(decisions):
    changed = copy.deepcopy(decisions)
    for row in changed:
        if "entry_plan" in row:
            row["entry_plan"].update(policy="risk", price_floor=9.01, price_ceiling=10.2,
                                     stop_price=9., risk_fraction=.01)
    return changed


@pytest.mark.parametrize("opening,reason", [(10.3, "entry_plan_price_above_ceiling"),
                                            (9., "entry_plan_price_below_floor")])
def test_risk_open_must_respect_frozen_price_interval(planned_input, opening, reason):
    raw, decisions, days = planned_input
    raw.loc[71, "open"] = opening
    result = _execute(raw, _risk(decisions), days, "risk")
    assert not result["ledger"]
    assert any(x["reason"] == reason for x in result["rejections"])


def test_actual_open_sizes_risk_and_no_position_additions(planned_input):
    raw, decisions, days = planned_input
    raw.loc[71, "open"] = 10.1
    result = _execute(raw, _risk(decisions), days, "risk")
    buy = result["ledger"][0]
    assert buy["shares"] == 900
    assert buy["shares"] * (buy["price"] - 9.) <= 100000 * .01
    assert [x["action"] for x in result["ledger"]] == ["BUY", "SELL"]


def test_actual_cost_stop_is_frozen_despite_new_holding_plan(planned_input):
    raw, decisions, days = planned_input
    decisions = _risk(decisions)
    decisions[74]["entry_plan"]["stop_price"] = 8.
    raw.loc[75, "close"] = 9.2
    raw.loc[76, "open"] = 9.3
    result = _execute(raw, decisions, days, "risk")
    assert result["ledger"][1]["date"] == days[76]
    assert result["ledger"][1]["reason"] == "actual_account_stop"
    assert result["daily"][74]["actual_stop_price"] == result["daily"][71]["actual_stop_price"]


def test_partial_exit_stays_pending_when_theoretical_target_reopens(planned_input):
    raw, decisions, days = planned_input
    raw.loc[80:81, "volume"] = 10000.  # 100-share prior-volume exit capacity.
    decisions[81]["target_weight"] = .95
    decisions[82]["target_weight"] = .95
    result = _execute(raw, decisions, days)
    sales = [x for x in result["ledger"] if x["action"] != "BUY"]
    assert sales[0]["action"] == "REDUCE" and sales[0]["date"] == days[81]
    assert sales[1]["date"] == days[82] and sales[-1]["position_after"] == 0
    assert sum(x["action"] == "BUY" for x in result["ledger"]) == 1


def test_planned_calendar_and_identity_are_fail_closed(planned_input):
    raw, decisions, days = planned_input
    with pytest.raises(ValueError, match="verified"):
        execute_decisions("000001.SZ", raw, decisions, days[0], 100000,
                          strategy_config(), "bad-calendar", entry_policy="fresh")
    decisions[70]["entry_plan"]["symbol"] = "600000.SH"
    assert any(x["reason"] == "entry_plan_identity_mismatch" for x in _execute(raw, decisions, days)["rejections"])


def _features(raw):
    result = pd.DataFrame({"date": raw["date"], "symbol": raw["symbol"], "input_eligible": True,
        "rule_buy": False, "rule_sell": False, "rule_price_volume_buy": False,
        "rule_price_volume_sell": False, "point_type": "二买辅助",
        "point_anchor": raw.iloc[60]["date"].date().isoformat(),
        "point_confirmed_at": raw.iloc[65]["date"].date().isoformat() + "T15:00:00+08:00"})
    result.loc[70, "rule_buy"] = True
    result.loc[80, "rule_sell"] = True
    return result


def test_strategy_labels_match_same_policy_cash_fees_and_actual_dates(planned_input):
    raw, decisions, days = planned_input
    labels = build_strategy_labels(_features(raw), raw, days, decisions=decisions)
    actual = _execute(raw, decisions, days)
    label = labels.iloc[70]
    paid, received = -actual["ledger"][0]["cash_flow"], actual["ledger"][1]["cash_flow"]
    assert label["label_available"]
    assert label["entry_date"] == actual["ledger"][0]["date"]
    assert label["label_end"] == label["exit_date"] == actual["ledger"][1]["date"]
    assert label["net_return"] == pytest.approx((received - paid) / 100000)
    assert label["invested_net_return"] == pytest.approx(received / paid - 1)
    assert label["fees"] == pytest.approx(sum(x["fee"] for x in actual["ledger"]))
    assert label["label"] == float(received > paid)
    assert labels.attrs["label_contract"]["maturity"] == "full_actual_exit_date"
    assert [{k: v for k, v in row.items() if k != "run_id"} for row in labels.attrs["execution"]["ledger"]] == [
        {k: v for k, v in row.items() if k != "run_id"} for row in actual["ledger"]]


def test_strategy_labels_use_one_complete_ledger_not_one_replay_per_candidate(planned_input, monkeypatch):
    from my_strategy.services import czsc_research_strategy_labels as module
    raw, decisions, days = planned_input
    features = _features(raw)
    features.loc[71:79, "rule_buy"] = True
    calls = []
    execute = module.execute_decisions
    monkeypatch.setattr(module, "execute_decisions", lambda *a, **kw: (calls.append(kw["entry_policy"]), execute(*a, **kw))[1])
    labels = build_strategy_labels(features, raw, days, decisions=decisions)
    assert calls == ["fresh"]
    assert labels["label_available"].sum() == 1
    assert set(labels.loc[71:79, "label_reason"]) == {"no_new_entry_plan"}


def test_unclosed_prefix_is_unavailable_and_future_append_keeps_mature_labels(planned_input):
    raw, decisions, days = planned_input
    features = _features(raw)
    unclosed = build_strategy_labels(features.iloc[:80], raw.iloc[:80], days, decisions=decisions[:80])
    assert unclosed.iloc[70]["label_reason"] == "position_unclosed"
    assert not unclosed.iloc[70]["label_available"]
    prefix = build_strategy_labels(features.iloc[:90], raw.iloc[:90], days, decisions=decisions[:90])
    raw.loc[95:, ["open", "close"]] = 100.
    full = build_strategy_labels(features, raw, days, decisions=decisions)
    mature = prefix["label_available"]
    pd.testing.assert_frame_equal(prefix.loc[mature], full.loc[prefix.index[mature]])


def test_partial_exit_matures_only_on_actual_final_sell(planned_input):
    raw, decisions, days = planned_input
    raw.loc[80:81, "volume"] = 10000.
    labels = build_strategy_labels(_features(raw), raw, days, decisions=decisions)
    ledger = labels.attrs["execution"]["ledger"]
    assert ledger[1]["action"] == "REDUCE"
    assert labels.iloc[70]["label_end"] == ledger[-1]["date"] == days[83]
    prefix = build_strategy_labels(_features(raw).iloc[:83], raw.iloc[:83], days, decisions=decisions[:83])
    assert prefix.iloc[70]["label_reason"] == "position_unclosed"


def test_unfilled_and_unavailable_candidates_are_retained(planned_input):
    raw, decisions, days = planned_input
    raw.loc[71, "open"] = 10.6
    labels = build_strategy_labels(_features(raw), raw, days, decisions=decisions)
    assert len(labels) == len(raw)
    assert labels.iloc[70]["label_reason"] == "entry_conservative_upper_price_guard"
    assert not labels.iloc[70]["label_available"]
    assert labels.attrs["execution"]["ledger"] == []
    unverified = build_strategy_labels(_features(raw), raw)
    assert set(unverified["label_reason"]) == {"calendar_unverified"}


def test_rule_exit_label_is_not_fixed_ten_day_exit(planned_input):
    raw, decisions, days = planned_input
    for row in decisions[75:]:
        row["target_weight"] = 0.
    raw.loc[76, "open"] = 10.3
    label = build_strategy_labels(_features(raw), raw, days, decisions=decisions).iloc[70]
    assert label["exit_date"] == label["label_end"] == days[76]
    assert label["label"] == 1


def test_strategy_labels_replay_real_shared_fresh_policy(planned_input):
    raw, _, days = planned_input
    labels = build_strategy_labels(_features(raw), raw, days)
    assert labels["label_available"].sum() == 1
    assert labels.iloc[70]["entry_date"] == days[71]
    assert labels.iloc[70]["exit_date"] == days[81]
    assert labels.attrs["label_contract"]["entry_parameters"]["risk_fraction"] == .01
    prefix = build_strategy_labels(_features(raw).iloc[:90], raw.iloc[:90], days)
    pd.testing.assert_frame_equal(prefix.loc[prefix["label_available"]],
                                  labels.loc[prefix.index[prefix["label_available"]]])


def test_strategy_labels_never_import_torch():
    result = subprocess.run([sys.executable, "-c", "import sys; from my_strategy.services.czsc_research_strategy_labels import build_strategy_labels; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_independent_label_audit_detects_cash_label_drift(planned_input, tmp_path):
    from my_strategy.scripts.audit_czsc_executable_entries import _labels
    from my_strategy.services.czsc_research_features import FEATURE_COLUMNS
    raw, decisions, days = planned_input
    features = _features(raw)
    labels = build_strategy_labels(features, raw, days, decisions=decisions)
    frozen = pd.DataFrame({"symbol": "000001.SZ", "date": days, "input_eligible": True, "close": 10.,
                           **{key: 0. for key in FEATURE_COLUMNS}})
    source = tmp_path / "frozen.parquet"
    frozen.to_parquet(source, index=False)
    record = {"symbol": "000001.SZ", "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "bars": len(raw), "data_end": days[-1]}
    (tmp_path / "labels").mkdir()
    (tmp_path / "label_audit/000001_SZ").mkdir(parents=True)
    execution = labels.attrs.pop("execution")
    labels.to_parquet(tmp_path / "labels/000001_SZ.parquet", index=False)
    audit_path = tmp_path / "label_audit/000001_SZ/execution.json"
    audit_path.write_text(json.dumps(execution), encoding="utf-8")
    assert not _labels(tmp_path, record, 100000, labels.attrs["label_contract"])["errors"]
    labels.loc[70, "net_return"] += .1
    labels.to_parquet(tmp_path / "labels/000001_SZ.parquet", index=False)
    errors = _labels(tmp_path, record, 100000, labels.attrs["label_contract"])["errors"]
    assert any(x["rule"] == "actual_label_ledger_value" and x["column"] == "net_return" for x in errors)
    execution["daily"][75]["cash"] += 100
    audit_path.write_text(json.dumps(execution), encoding="utf-8")
    errors = _labels(tmp_path, record, 100000, labels.attrs["label_contract"])["errors"]
    assert any(x["rule"] == "label_full_prefix_account_reconciliation" for x in errors)
