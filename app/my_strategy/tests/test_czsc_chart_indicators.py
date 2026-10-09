"""Numerical oracles and history boundaries for display-only chart overlays."""
import json
import math

import pandas as pd
import pytest

from my_strategy.services.czsc_chart_indicators import chart_indicators, daily_volume_profile


def bars(count):
    return [{"time": date.date().isoformat(), "open": float(i), "high": float(i), "low": float(i),
             "close": float(i), "volume": 1.0, "is_closed": True}
            for i, date in enumerate(pd.bdate_range("2025-01-01", periods=count), 1)]


def calculate(rows, start=None):
    return chart_indicators(rows, [row["time"] + "T15:00:00+08:00" for row in rows],
                            daily_volume_profile(rows), start, daily_bars=rows)


def test_known_ma_and_population_bollinger_values():
    result = calculate(bars(20))["rows"]
    assert result[3]["ma5"] is None
    assert result[4]["ma5"] == 3
    assert result[-1]["ma5"] == 18
    assert result[-1]["ma10"] == 15.5
    assert result[-1]["ma20"] == result[-1]["boll_mid"] == 10.5
    assert result[-1]["ma60"] is None
    assert result[-1]["boll_upper"] == pytest.approx(10.5 + 2 * math.sqrt(33.25))
    assert result[-1]["boll_lower"] == pytest.approx(10.5 - 2 * math.sqrt(33.25))
    json.dumps(result, allow_nan=False)


def test_volume_weighted_quantiles_not_price_percentages():
    rows = bars(120)
    result = calculate(rows)["rows"]
    assert result[-2]["chip70_low"] is None
    assert [result[-1][key] for key in ("chip70_low", "chip50", "chip70_high")] == [18, 60, 102]
    rows[-1]["volume"] = 10000
    dominant = calculate(rows)["rows"][-1]
    assert [dominant[key] for key in ("chip70_low", "chip50", "chip70_high")] == [120, 120, 120]
    assert dominant["chip_as_of"] == rows[-1]["time"]


def test_no_volume_or_short_history_has_no_fabricated_chip_values():
    rows = bars(130)
    for row in rows:
        row["volume"] = 0
    result = calculate(rows)["rows"][-1]
    assert result["chip70_low"] is None and result["chip_as_of"] is None
    assert result["ma_alignment"] == "bullish"
    assert result["ma20_direction"] == result["ma60_direction"] == "rising"


def test_clipping_retains_warmup_and_future_rows_do_not_change_prefix():
    rows = bars(145)
    cutoff = rows[129]["time"]
    full = calculate(rows)["rows"]
    assert calculate(rows, cutoff)["rows"] == [row for row in full if row["time"] >= cutoff]
    assert calculate(rows[:130])["rows"] == full[:130]


def test_weekly_profile_does_not_read_later_daily_values():
    daily = bars(130)
    weekly = [{**daily[119], "is_closed": True}, {**daily[124], "is_closed": False}]
    availability = [daily[120]["time"] + "T15:00:00+08:00", daily[124]["time"] + "T15:00:00+08:00"]
    result = chart_indicators(weekly, availability, daily_volume_profile(daily))["rows"]
    assert [row["chip_as_of"] for row in result] == [daily[119]["time"], daily[124]["time"]]
    assert result[0]["chip70_low"] == 18 and result[1]["chip70_low"] == 23
    assert result[1]["is_closed"] is False and result[1]["ma5"] is None
    assert result[0]["available_at"] == availability[0]


def test_empty_frequency_still_exposes_method_and_no_rows():
    result = chart_indicators([], [], daily_volume_profile([]))
    assert result["rows"] == [] and result["chip_window"] == 120
    assert result["chip_profile"] is None


def test_chip_histogram_conserves_volume_and_places_prices_in_bins():
    rows = bars(145)
    for index, row in enumerate(rows):
        row["volume"] = index + 1
    result = calculate(rows)
    profile = result["chip_profile"]
    assert profile["status"] == "available" and profile["observed_bars"] == 120
    assert len(profile["bins"]) == profile["bin_count"] == 32
    expected = sum(row["volume"] for row in rows[-120:])
    assert profile["total_volume"] == sum(item["volume"] for item in profile["bins"]) == expected
    assert sum(item["mass_fraction"] for item in profile["bins"]) == pytest.approx(1)
    for index, item in enumerate(profile["bins"]):
        observations = [row for row in rows[-120:]
                        if item["price_low"] <= row["close"]
                        and (row["close"] < item["price_high"]
                             or index == 31 and row["close"] == item["price_high"])]
        assert item["volume"] == sum(row["volume"] for row in observations)
        assert item["price_mid"] == pytest.approx((item["price_low"] + item["price_high"]) / 2)
    assert profile["chip70_low"] == result["rows"][-1]["chip70_low"]
    assert all("bins" not in row for row in result["rows"])
    json.dumps(result, allow_nan=False)


def test_chip_histogram_uses_hlc3_and_handles_single_price():
    rows = bars(120)
    for row in rows:
        row.update(high=8.0, low=2.0, close=5.0)
    profile = calculate(rows)["chip_profile"]
    populated = [item for item in profile["bins"] if item["volume"] > 0]
    assert len(populated) == 1 and populated[0]["volume"] == 120
    assert populated[0]["price_low"] <= 5 < populated[0]["price_high"]
    assert profile["chip70_low"] == profile["chip50"] == profile["chip70_high"] == 5
    json.dumps(profile, allow_nan=False)


def test_chip_histogram_reports_short_history_and_missing_volume():
    short = calculate(bars(119))["chip_profile"]
    assert short["status"] == "insufficient_history" and short["observed_bars"] == 119
    assert short["bins"] == [] and short["total_volume"] is None
    rows = bars(120)
    for row in rows:
        row["volume"] = 0
    no_volume = calculate(rows)["chip_profile"]
    assert no_volume["status"] == "no_volume" and no_volume["total_volume"] == 0
    assert no_volume["bins"] == [] and no_volume["chip50"] is None


def test_chip_histogram_replay_and_higher_anchor_exclude_future_bars():
    daily = bars(145)
    prefix = daily[:125]
    expected = calculate(prefix)["chip_profile"]
    for row in daily[125:]:
        row.update(high=1_000_000.0, low=1_000_000.0, close=1_000_000.0, volume=1_000_000.0)
    weekly = [{**daily[119], "is_closed": True}, {**daily[124], "is_closed": False}]
    availability = [daily[120]["time"] + "T15:00:00+08:00", daily[124]["time"] + "T15:00:00+08:00"]
    result = chart_indicators(weekly, availability, daily_volume_profile(daily), daily_bars=daily)
    assert result["chip_profile"] == expected
    assert result["chip_profile"]["as_of"] == daily[124]["time"]
    assert calculate(daily[:125])["chip_profile"] == expected
    assert calculate(daily, start=daily[124]["time"])["chip_profile"] == calculate(daily)["chip_profile"]
