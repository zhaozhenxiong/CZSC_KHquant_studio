"""Chronology, native confirmation and source-quality research contracts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from my_strategy.adapters.czsc_adapter import native_runtime
from my_strategy.services.czsc_analysis import CzscSession, strategy_config
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services.czsc_research_features import FEATURE_COLUMNS, FEATURE_SCHEMA_HASH, FEATURE_VERSION, build_features


@pytest.fixture(scope="module")
def clean_bars():
    count = 600
    random = np.random.default_rng(17)
    prices = 20 * np.exp(np.cumsum(random.normal(.0006, .025, count)))
    volume = random.uniform(600000, 2000000, count)
    dates = pd.bdate_range("2020-01-02", periods=count)
    frame = pd.DataFrame({"symbol": "000001.SZ", "date": dates, "open": prices * .998,
                          "high": prices * 1.009, "low": prices * .99, "close": prices,
                          "volume": volume, "amount": prices * volume,
                          "source": "tushare", "has_trade_price": 1})
    frame.attrs.update({"data_version": "clean-fixture", "price_basis": "unadjusted",
                        "volume_unit": "shares", "amount_unit": "CNY"})
    return frame


@pytest.fixture(scope="module")
def clean_features(clean_bars):
    return build_features(clean_bars)


def test_feature_schema_alignment_and_input_immutability(clean_bars, clean_features):
    original = clean_bars.copy(deep=True)
    result = build_features(clean_bars.iloc[:150].set_axis(np.arange(1000, 1150)))
    assert_frame_equal(clean_bars, original)
    assert len(FEATURE_COLUMNS) == 50 and len(set(FEATURE_COLUMNS)) == 50
    assert result.index.tolist() == list(range(1000, 1150))
    assert result["date"].tolist() == clean_bars["date"].iloc[:150].tolist()
    assert all(pd.api.types.is_float_dtype(result[column]) for column in FEATURE_COLUMNS)
    assert not np.isinf(result[list(FEATURE_COLUMNS)].to_numpy()).any()
    assert clean_features.attrs["feature_version"] == FEATURE_VERSION
    assert clean_features.attrs["schema_hash"] == FEATURE_SCHEMA_HASH
    assert clean_features.attrs["data_version"] == "clean-fixture"
    assert clean_features.attrs["higher_period_closure"] == "next_period_observed"
    assert clean_features.loc[119, "input_eligible"]
    assert clean_features.iloc[:119]["reason_codes"].map(lambda value: "quality_warmup" in value).all()
    assert clean_features["available_at"].iloc[119] == "2020-06-17T15:00:00+08:00"


@pytest.mark.parametrize("prefix", [80, 210, 411])
def test_prefix_replay_is_identical_to_full_history(clean_bars, clean_features, prefix):
    result = build_features(clean_bars.iloc[:prefix])
    assert_frame_equal(result, clean_features.iloc[:prefix], check_exact=True)


def test_future_price_volume_and_source_perturbation_cannot_change_past(clean_bars, clean_features):
    changed = clean_bars.copy()
    cutoff = 310
    changed.loc[cutoff:, ["open", "high", "low", "close"]] *= 9
    changed.loc[cutoff:, ["volume", "amount"]] *= 7
    changed.loc[cutoff:, "source"] = "fuyao"
    changed.loc[cutoff:, "has_trade_price"] = 0
    actual = build_features(changed)
    assert_frame_equal(actual.iloc[:cutoff], clean_features.iloc[:cutoff], check_exact=True)
    assert actual["input_hash"].iloc[cutoff] != clean_features["input_hash"].iloc[cutoff]


def test_higher_period_values_publish_only_after_next_bucket(clean_bars, clean_features):
    result = clean_features.set_index("date")
    assert result.loc["2020-01-03", "weekly_closed_bars"] == 0
    assert result.loc["2020-01-06", "weekly_closed_bars"] == 1
    assert result.loc["2020-01-06", "weekly_anchor"] == "2020-01-03"
    assert result.loc["2020-01-06", "weekly_available_at"] == "2020-01-06T15:00:00+08:00"
    assert pd.isna(result.loc["2020-01-10", "weekly_return"])
    prior_week = clean_bars.set_index("date")["close"].loc["2020-01-10"] / clean_bars.set_index("date")["close"].loc["2020-01-03"] - 1
    assert result.loc["2020-01-13", "weekly_return"] == pytest.approx(prior_week)
    assert result.loc["2020-01-31", "monthly_closed_bars"] == 0
    assert result.loc["2020-02-03", "monthly_closed_bars"] == 1
    assert result.loc["2020-02-03", "monthly_anchor"] == "2020-01-31"
    assert result.loc["2020-02-03", "monthly_available_at"] == "2020-02-03T15:00:00+08:00"
    assert result.loc["2020-03-02", "monthly_closed_bars"] == 2
    assert result.loc["2020-03-03", "monthly_available_at"] == "2020-03-02T15:00:00+08:00"


def test_native_signals_use_actual_finished_pen_offset(clean_bars, clean_features):
    native = native_runtime()
    settings = strategy_config()
    session = CzscSession("000001.SZ", settings, len(clean_bars), collect_confirmations=False, create_position=False)
    frame = clean_bars.copy()
    frame["dt"] = frame["date"] + pd.Timedelta(hours=15)
    frame["id"] = np.arange(len(frame))
    excluded_raw_buy = []
    confirmed_count = 0
    for index, row in enumerate(frame.itertuples(index=False)):
        session.advance_structure(row)
        analysis = session.kas["日线"]
        finished, all_pens = analysis.finished_bis, analysis.bi_list
        di = len(all_pens) - len(finished) + 1
        result = clean_features.iloc[index]
        for name, period, short in (("cxt_second_bs_V230320", 21, "bs2"), ("cxt_third_bs_V230318", 34, "bs3")):
            expected_raw = str(native._native.call_signal(name, analysis, {"di": 1, "ma_type": "SMA", "timeperiod": period})[0].v1)
            expected_confirmed = str(native._native.call_signal(name, analysis, {"di": di, "ma_type": "SMA", "timeperiod": period})[0].v1) if finished else "其他"
            assert result["raw_native_" + short] == expected_raw
            assert result["confirmed_native_" + short] == expected_confirmed
            if di > 1 and expected_raw in {"二买", "三买"} and expected_confirmed == "其他":
                excluded_raw_buy.append(index)
            if expected_confirmed != "其他":
                confirmed_count += 1
                assert result["point_confirmed"]
                assert result["point_anchor"] == pd.Timestamp(finished[-1].fx_b.dt).date().isoformat()
                assert result["point_confirmed_at"] <= result["available_at"]
                assert result["point_formed_at"] <= result["point_confirmed_at"]
                assert result["structure_confirmed_at"] <= result["available_at"]
    assert confirmed_count > 20 and len(excluded_raw_buy) > 5
    only_forming = clean_features.iloc[excluded_raw_buy]
    only_forming = only_forming.loc[(only_forming["confirmed_native_bs2"] == "其他") & (only_forming["confirmed_native_bs3"] == "其他")]
    assert len(only_forming) > 5
    assert not only_forming["point_confirmed"].any()
    assert not only_forming["rule_czsc_buy"].any()


def test_mean_and_volume_baselines_are_trailing_only(clean_bars, clean_features):
    row = clean_features.iloc[180]
    assert row["ma20"] == pytest.approx(clean_bars["close"].iloc[161:181].mean())
    assert row["ma20_slope5"] == pytest.approx(clean_bars["close"].iloc[161:181].mean() / clean_bars["close"].iloc[156:176].mean() - 1)
    assert row["vol_ratio20"] == pytest.approx(clean_bars["volume"].iloc[180] / clean_bars["volume"].iloc[160:180].mean())


@pytest.mark.parametrize("column,value,reason", [
    ("source", "fuyao", "legacy_fuyao_window"),
    ("source", "unknown_adjustment_vendor", "unverified_source_window"),
    ("source", None, "source_missing_window"),
    ("has_trade_price", 0, "trade_price_unverified_window"),
    ("amount", np.nan, "missing_values_window"),
    ("amount", 0, "volume_amount_mismatch_window"),
    ("volume", -1, "negative_volume_amount_window"),
])
def test_bad_input_window_is_retained_and_ineligible(clean_bars, column, value, reason):
    frame = clean_bars.iloc[:280].copy()
    frame.loc[150, column] = value
    result = build_features(frame)
    assert len(result) == len(frame)
    assert result["date"].tolist() == frame["date"].tolist()
    assert not result.iloc[150:270]["input_eligible"].any()
    assert not result.iloc[150:270]["model_input_eligible"].any()
    assert result.iloc[150:270]["model_reason_codes"].map(lambda reasons: reason in reasons).all()
    assert result.iloc[150:270]["reason_codes"].map(lambda reasons: reason in reasons).all()
    assert reason not in result.iloc[270]["reason_codes"]
    assert not result.iloc[150:270]["rule_buy"].any()
    if pd.isna(value) and column == "amount":
        assert result.loc[150, "engine_zero_quantity_substitution"]
        assert pd.isna(result.loc[150, "amount_ratio20"])


def test_legal_zero_volume_suspension_keeps_date_and_later_windows(clean_bars):
    frame = clean_bars.iloc[:200].copy()
    frame.loc[150, ["volume", "amount"]] = 0
    result = build_features(frame)
    assert len(result) == 200
    assert result.loc[150, "suspended"]
    assert "suspended_current_bar" in result.loc[150, "reason_codes"]
    assert result.loc[150, "vol_ratio20"] == 0
    assert not result.loc[150, "engine_zero_quantity_substitution"]
    assert not result.loc[150, "input_eligible"]
    assert result.loc[160, "input_eligible"]
    assert result.loc[160, "suspension_bars_window"] == 1
    assert "missing_values_window" not in result.loc[160, "reason_codes"]


def test_structural_dependencies_outside_quality_window_are_not_silently_clean(clean_bars):
    frame = clean_bars.copy()
    # A persistent broad oscillation creates a long zone whose first pens remain
    # relevant after a short rolling quality window has moved past their source.
    prices = 12 + np.arange(len(frame)) * .006 + 1.5 * np.sin(np.arange(len(frame)) / 13)
    frame["open"], frame["high"], frame["low"], frame["close"] = prices, prices + .15, prices - .15, prices + .04
    frame.loc[10, "source"] = "fuyao"
    result = build_features(frame)
    historical = result.loc[result["date"] > frame.loc[129, "date"]]
    assert historical["reason_codes"].map(lambda reasons: "legacy_fuyao_window" not in reasons).all()
    contaminated_structure = historical.loc[historical["reason_codes"].map(lambda reasons: "structural_input_unverified" in reasons)]
    assert len(contaminated_structure) > 0
    assert not contaminated_structure["input_eligible"].any()
    assert contaminated_structure["model_input_eligible"].all()
    assert contaminated_structure["model_reason_codes"].map(lambda reasons: "structural_input_unverified" not in reasons).all()


def test_ma_model_requires_120_quality_bars_and_60_active_observations(clean_bars):
    frame = clean_bars.iloc[:190].copy()
    frame.loc[:99, ["volume", "amount"]] = 0
    settings = {**strategy_config(), "research": {"quality_window_bars": 60}}
    result = build_features(frame, settings)
    assert result.attrs["model_quality_window_bars"] == 120
    assert not result.iloc[:159]["model_input_eligible"].any()
    assert "quality_warmup" in result.iloc[118]["model_reason_codes"]
    assert "insufficient_active_history" in result.iloc[158]["model_reason_codes"]
    assert result.iloc[159]["model_input_eligible"]
    assert result.iloc[159]["model_active_bars_window"] == 60


def test_unknown_nonempty_provider_is_rejected_in_current_window_and_structure(clean_bars):
    frame = clean_bars.copy()
    prices = 12 + np.arange(len(frame)) * .006 + 1.5 * np.sin(np.arange(len(frame)) / 13)
    frame["open"], frame["high"], frame["low"], frame["close"] = prices, prices + .15, prices - .15, prices + .04
    frame.loc[10, "source"] = "unknown_adjustment_vendor"
    frame.loc[150, "source"] = "unknown_adjustment_vendor"
    result = build_features(frame)
    assert "unverified_source_current_bar" in result.loc[150, "reason_codes"]
    assert "unverified_source_window" in result.loc[150, "reason_codes"]
    assert "unverified_source_current_bar" not in result.loc[151, "reason_codes"]
    assert result.iloc[150:270]["reason_codes"].map(lambda values: "unverified_source_window" in values).all()
    assert not result.iloc[150:270]["input_eligible"].any()
    historical = result.iloc[270:]
    assert historical["reason_codes"].map(lambda values: "unverified_source_window" not in values).all()
    contaminated = historical.loc[historical["reason_codes"].map(lambda values: "unverified_source_structural" in values)]
    assert len(contaminated) > 0 and not contaminated["input_eligible"].any()


@pytest.mark.parametrize("source", sorted(VERIFIED_SOURCES))
def test_shared_verified_source_contract_remains_eligible(clean_bars, source):
    frame = clean_bars.iloc[:150].copy()
    frame["source"] = source
    result = build_features(frame)
    assert result.iloc[-1]["input_eligible"]
    assert result.iloc[-1]["reason_codes"] == []
    assert result.attrs["verified_sources"] == sorted(VERIFIED_SOURCES)
    assert not result["source"].ne(source).any()


@pytest.mark.parametrize("defect,match", [("mixed", "不同股票"), ("duplicate", "重复"), ("unordered", "乱序"),
                                        ("missing_ohlc", "OHLC"), ("invalid_ohlc", "范围"), ("empty", "非空")])
def test_invalid_input_fails_closed_without_cross_stock_rolling(clean_bars, defect, match):
    frame = clean_bars.iloc[:150].copy()
    if defect == "mixed":
        frame.loc[140:, "symbol"] = "600519.SH"
    elif defect == "duplicate":
        frame.loc[100, "date"] = frame.loc[99, "date"]
    elif defect == "unordered":
        frame = frame.iloc[::-1]
    elif defect == "missing_ohlc":
        frame.loc[100, "close"] = np.nan
    elif defect == "invalid_ohlc":
        frame.loc[100, "high"] = frame.loc[100, "low"] * .5
    elif defect == "empty":
        frame = frame.iloc[:0]
    with pytest.raises(ValueError, match=match):
        build_features(frame)


def test_missing_quality_metadata_cannot_become_eligible(clean_bars):
    result = build_features(clean_bars.iloc[:150].drop(columns=["source", "has_trade_price"]))
    assert not result["input_eligible"].any()
    assert "source_missing_window" in result.iloc[-1]["reason_codes"]
    assert "trade_price_unverified_window" in result.iloc[-1]["reason_codes"]
