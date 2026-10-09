"""Wyckoff inputs are causal, raw-only, and separate from legacy rule identity."""
import copy

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.czsc_wyckoff_features import (
    W_INPUT_COLUMNS, WYCKOFF_SCHEMA_HASH, bind_wyckoff_profile,
    get_wyckoff_profile, prepare_wyckoff_inputs,
)


def raw_frame(n=220):
    phase = np.arange(n) % 4
    close = 10 + np.choose(phase, [-.4, .4, .2, -.2])
    frame = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=n), "symbol": "000001.SZ",
                          "open": close - .05, "high": 11., "low": 9., "close": close,
                          "volume": 1_000_000. + (np.arange(n) % 7) * 10000.}, index=np.arange(n) * 3 + 5)
    frame["amount"] = frame["volume"] * frame["close"]
    frame["source"] = "sina_unadjusted_v1"; frame["has_trade_price"] = 1
    frame["input_eligible"] = False; frame["rule_buy"] = False; frame["rule_sell"] = False
    frame["model_reason_codes"] = [[] for _ in range(n)]
    frame.attrs = {"volume_unit": "shares", "amount_unit": "CNY", "price_basis": "unadjusted",
                   "feature_version": "old_ma", "schema_hash": "old_schema", "feature_columns": ["old"]}
    return frame


def set_bar(frame, i, *, op, hi, lo, close, volume):
    label = frame.index[i]
    for key, value in {"open": op, "high": hi, "low": lo, "close": close, "volume": volume,
                       "amount": close * volume}.items():
        frame.loc[label, key] = value


def test_exact_32_schema_and_separate_identity():
    raw = raw_frame(); original = raw.copy(deep=True)
    result = prepare_wyckoff_inputs(raw)
    assert len(W_INPUT_COLUMNS) == len(set(W_INPUT_COLUMNS)) == 32
    assert get_wyckoff_profile()["schema_hash"] == WYCKOFF_SCHEMA_HASH
    pd.testing.assert_frame_equal(raw, original)
    pd.testing.assert_frame_equal(result[raw.columns], raw)
    assert result.attrs["feature_version"] == "old_ma"
    assert result.attrs["wyckoff_schema_hash"] == WYCKOFF_SCHEMA_HASH
    assert result["wyckoff_input_eligible"].iloc[119:].all()
    assert not result["wyckoff_input_eligible"].iloc[:119].any()
    assert not result["wyckoff_formal_qualified"].any()
    bound = bind_wyckoff_profile(result)
    assert bound.attrs["feature_columns"] == list(W_INPUT_COLUMNS)
    assert bound.attrs["feature_version"] == "wyckoff_pv_v1"


@pytest.mark.parametrize("end", [1, 60, 119, 120, 141, 142, 170, 219])
def test_future_append_never_changes_previous_rows(end):
    raw = raw_frame()
    set_bar(raw, 140, op=9.5, hi=10., lo=8.6, close=9.4, volume=2_000_000.)
    set_bar(raw, 141, op=9.2, hi=9.6, lo=9.05, close=9.5, volume=700_000.)
    full = prepare_wyckoff_inputs(raw)
    prefix = prepare_wyckoff_inputs(raw.iloc[:end].copy())
    new = [key for key in full if key not in raw]
    pd.testing.assert_frame_equal(full[new].iloc[:end], prefix[new], check_exact=True)


def test_spring_is_pending_then_later_test_is_buy_with_old_anchor():
    raw = raw_frame()
    set_bar(raw, 140, op=9.5, hi=10., lo=8.6, close=9.4, volume=2_000_000.)
    set_bar(raw, 141, op=9.2, hi=9.6, lo=9.05, close=9.5, volume=700_000.)
    result = prepare_wyckoff_inputs(raw)
    candidate, confirmed = result.iloc[140], result.iloc[141]
    assert candidate.wyckoff_event == "Spring"
    assert candidate.wyckoff_state == "spring_pending"
    assert not candidate.wyckoff_rule_buy
    assert confirmed.wyckoff_event == "Test" and confirmed.wyckoff_rule_buy
    assert confirmed.wyckoff_anchor_at == raw.iloc[140].date.strftime("%Y-%m-%d")
    assert confirmed.wyckoff_observed_at == raw.iloc[141].date.strftime("%Y-%m-%d") + "T15:00:00+08:00"
    assert confirmed.wyckoff_stop == 8.6
    assert "later_lower_volume_and_spread_test" in confirmed.wyckoff_entry_evidence


def test_range_excludes_current_probe():
    raw = raw_frame()
    set_bar(raw, 140, op=9.5, hi=10., lo=8.6, close=9.4, volume=2_000_000.)
    result = prepare_wyckoff_inputs(raw)
    assert result.iloc[140].wyckoff_range_low == 9.
    assert result.iloc[140].w_down_probe_atr > 0
    assert result.iloc[141].wyckoff_range_low == 8.6
    assert result.iloc[140].wyckoff_range_formed_at == result.iloc[139].wyckoff_available_at


def test_sos_then_lps_and_sow_then_lpsy():
    raw = raw_frame()
    set_bar(raw, 140, op=10.5, hi=11.8, lo=10.4, close=11.7, volume=2_000_000.)
    set_bar(raw, 141, op=11.1, hi=11.5, lo=11.0, close=11.4, volume=700_000.)
    r = prepare_wyckoff_inputs(raw)
    assert r.iloc[140].wyckoff_event == "SOS"
    assert not r.iloc[140].wyckoff_rule_buy
    assert r.iloc[141].wyckoff_event == "LPS" and r.iloc[141].wyckoff_rule_buy
    raw = raw_frame()
    set_bar(raw, 140, op=9.8, hi=9.9, lo=8.1, close=8.3, volume=2_000_000.)
    set_bar(raw, 141, op=9.0, hi=9.15, lo=8.6, close=8.7, volume=700_000.)
    r = prepare_wyckoff_inputs(raw)
    assert r.iloc[140].wyckoff_event == "SOW" and r.iloc[140].wyckoff_rule_sell
    assert r.iloc[141].wyckoff_event == "LPSY" and r.iloc[141].wyckoff_rule_sell


def test_conflicting_probe_has_no_buy_or_false_event():
    raw = raw_frame()
    set_bar(raw, 140, op=10., hi=11.6, lo=8.4, close=10., volume=2_000_000.)
    r = prepare_wyckoff_inputs(raw).iloc[140]
    assert r.wyckoff_event == r.wyckoff_state == "conflict"
    assert not r.wyckoff_rule_buy
    assert r.w_spring == r.w_upthrust == 0.


@pytest.mark.parametrize("kind,reason", [
    ("volume", "wyckoff_invalid_quantity_window"),
    ("source", "wyckoff_unverified_source_window"),
    ("verified", "wyckoff_trade_price_unverified_window"),
    ("amount", "wyckoff_volume_amount_unit_disagreement_window"),
    ("calendar", "wyckoff_calendar_gap_window"),
])
def test_unsafe_current_data_rejected_without_backfilling_previous(kind, reason):
    raw = raw_frame(); original = prepare_wyckoff_inputs(raw)
    label = raw.index[150]
    if kind == "volume": raw.loc[label, "volume"] = np.nan
    elif kind == "source": raw.loc[label, "source"] = "unknown"
    elif kind == "verified": raw.loc[label, "has_trade_price"] = 0
    elif kind == "amount": raw.loc[label, "amount"] *= 100
    else: raw.at[label, "model_reason_codes"] = ["calendar_missing_market_bar_window"]
    r = prepare_wyckoff_inputs(raw)
    assert not r.iloc[150].wyckoff_input_eligible
    assert reason in r.iloc[150].wyckoff_reason_codes
    pd.testing.assert_frame_equal(original[list(W_INPUT_COLUMNS)].iloc[:150], r[list(W_INPUT_COLUMNS)].iloc[:150])
    assert not r.iloc[150].wyckoff_rule_buy


def test_action_break_causes_only_causal_window_rejection():
    raw = raw_frame(300)
    raw.loc[raw.index[150]:, ["open", "high", "low", "close"]] *= .5
    raw["amount"] = raw["volume"] * raw["close"]
    r = prepare_wyckoff_inputs(raw)
    assert r.iloc[149].wyckoff_input_eligible
    assert r.iloc[150].wyckoff_corporate_action_suspected
    assert not r["wyckoff_input_eligible"].iloc[150:270].any()
    assert r.iloc[270].wyckoff_input_eligible
    assert "wyckoff_suspected_corporate_action_window" in r.iloc[150].wyckoff_reason_codes
    assert raw.iloc[150].close == r.iloc[150].close


def test_raw_only_independent_of_ma_and_structure_and_rules():
    raw = raw_frame(); first = prepare_wyckoff_inputs(raw)
    raw["ma20"] = 1e99; raw["bi1_direction"] = -1; raw["rule_buy"] = True
    raw["input_eligible"] = True
    second = prepare_wyckoff_inputs(raw)
    pd.testing.assert_frame_equal(first[list(W_INPUT_COLUMNS)], second[list(W_INPUT_COLUMNS)])
    assert first.wyckoff_input_eligible.equals(second.wyckoff_input_eligible)


@pytest.mark.parametrize("change", ["order", "duplicate", "symbols", "index", "date"])
def test_invalid_row_contract_fails_closed(change):
    raw = raw_frame(); frame = raw.copy()
    if change == "order": raw = raw.iloc[::-1]; frame = raw.copy()
    elif change == "duplicate": raw.loc[raw.index[1], "date"] = raw.iloc[0].date; frame = raw.copy()
    elif change == "symbols": raw.loc[raw.index[-1], "symbol"] = "600000.SH"
    elif change == "index": raw = raw.reset_index(drop=True)
    else: raw.loc[raw.index[-1], "date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError): prepare_wyckoff_inputs(frame, raw)


def test_units_not_declared_do_not_get_probabilities():
    raw = raw_frame(); raw.attrs.pop("volume_unit")
    r = prepare_wyckoff_inputs(raw)
    assert not r.wyckoff_input_eligible.any()
    assert "wyckoff_units_not_verified" in r.iloc[-1].wyckoff_reason_codes


def test_custom_configuration_is_different_identity_and_no_mutation():
    config = {"events": {"test_volume_ratio": .7}}; original = copy.deepcopy(config)
    assert get_wyckoff_profile(config)["schema_hash"] != WYCKOFF_SCHEMA_HASH
    assert prepare_wyckoff_inputs(raw_frame(), config=config).attrs["wyckoff_schema_hash"] == get_wyckoff_profile(config)["schema_hash"]
    assert config == original


def test_empty_preserves_route_metadata_schema():
    r = prepare_wyckoff_inputs(raw_frame().iloc[:0].copy())
    assert r.empty and len(W_INPUT_COLUMNS) == 32
    assert "wyckoff_rule_buy" in r and "wyckoff_available_at" in r
    assert r.attrs["wyckoff_schema_hash"] == WYCKOFF_SCHEMA_HASH
