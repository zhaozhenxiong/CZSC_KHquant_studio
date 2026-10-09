"""Display-only overlays; each value uses its observable history prefix."""
from __future__ import annotations

import numpy as np
import pandas as pd

MA_PERIODS = (5, 10, 20, 60)
CHIP_WINDOW = 120
CHIP_BINS = 32


def daily_volume_profile(bars: list[dict]) -> pd.DataFrame:
    """Weighted HLC/3 quantiles, not a shareholder cost or turnover model."""
    rows = []
    for index, bar in enumerate(bars):
        row = {"time": bar["time"], "chip70_low": None, "chip50": None, "chip70_high": None,
               "chip_as_of": None}
        if index + 1 >= CHIP_WINDOW:
            window = bars[index + 1 - CHIP_WINDOW:index + 1]
            prices = np.array([(item["high"] + item["low"] + item["close"]) / 3 for item in window])
            weights = np.array([item["volume"] for item in window], dtype=float)
            active = weights > 0
            if active.any():
                order = np.argsort(prices[active], kind="stable")
                prices, weights = prices[active][order], weights[active][order]
                cumulative = np.cumsum(weights)
                for key, quantile in (("chip70_low", .15), ("chip50", .5), ("chip70_high", .85)):
                    row[key] = float(prices[np.searchsorted(cumulative, cumulative[-1] * quantile)])
                row["chip_as_of"] = bar["time"]
        rows.append(row)
    return pd.DataFrame(rows, columns=["time", "chip70_low", "chip50", "chip70_high", "chip_as_of"])


def _chip_profile(daily_bars: list[dict], row: dict) -> dict:
    """One histogram from the same daily prefix as the visible quantiles."""
    as_of = row["chip_as_of"] or row["time"]
    window = [bar for bar in daily_bars if bar["time"] <= as_of][-CHIP_WINDOW:]
    result = {"status": "insufficient_history", "as_of": as_of, "window_bars": CHIP_WINDOW,
              "observed_bars": len(window), "bin_count": CHIP_BINS, "total_volume": None,
              "chip70_low": row["chip70_low"], "chip50": row["chip50"],
              "chip70_high": row["chip70_high"], "bins": []}
    if len(window) < CHIP_WINDOW:
        return result
    prices = np.array([(bar["high"] + bar["low"] + bar["close"]) / 3 for bar in window])
    weights = np.array([bar["volume"] for bar in window], dtype=float)
    active = weights > 0
    result["total_volume"] = float(weights[active].sum())
    if not active.any():
        result["status"] = "no_volume"
        return result
    prices, weights = prices[active], weights[active]
    bounds = float(prices.min()), float(prices.max())
    if bounds[0] == bounds[1]:
        half_width = max(abs(bounds[0]) * .005, .005)
        bounds = bounds[0] - half_width, bounds[1] + half_width
    volume, edges = np.histogram(prices, bins=CHIP_BINS, range=bounds, weights=weights)
    result["status"] = "available"
    result["bins"] = [{"price_low": float(edges[index]), "price_high": float(edges[index + 1]),
                       "price_mid": float((edges[index] + edges[index + 1]) / 2),
                       "volume": float(value), "mass_fraction": float(value / result["total_volume"])}
                      for index, value in enumerate(volume)]
    return result


def chart_indicators(bars: list[dict], available_at: list[str], profile: pd.DataFrame,
                     start: str | None = None, *, daily_bars: list[dict] | None = None) -> dict:
    """Compute before display clipping, including a clearly provisional last bucket."""
    rows = []
    if bars:
        frame = pd.DataFrame(bars)
        close = frame["close"]
        for period in MA_PERIODS:
            frame[f"ma{period}"] = close.rolling(period, min_periods=period).mean()
        frame["boll_mid"] = frame["ma20"]
        std = close.rolling(20, min_periods=20).std(ddof=0)
        frame["boll_upper"], frame["boll_lower"] = frame["ma20"] + 2 * std, frame["ma20"] - 2 * std
        for period in (20, 60):
            change = frame[f"ma{period}"] - frame[f"ma{period}"].shift(5)
            frame[f"ma{period}_direction"] = ["insufficient" if pd.isna(value) else "rising" if value > 0 else "falling" if value < 0 else "flat" for value in change]
        # A weekly/monthly anchor uses daily observations no later than itself,
        # never the eventual complete higher bucket or a later daily estimate.
        dates = frame[["time"]].assign(time=pd.to_datetime(frame["time"]))
        mapped = pd.merge_asof(dates, profile.assign(time=pd.to_datetime(profile["time"])), on="time", direction="backward")
        keys = [f"ma{period}" for period in MA_PERIODS] + ["boll_mid", "boll_upper", "boll_lower"]
        for index, item in frame.iterrows():
            if start and item["time"] < start:
                continue
            row = {"time": item["time"], "is_closed": bool(item["is_closed"]), "available_at": available_at[index]}
            row.update({key: None if pd.isna(item[key]) else float(item[key]) for key in keys})
            profile_row = mapped.iloc[index]
            row.update({key: None if pd.isna(profile_row[key]) else float(profile_row[key]) for key in ("chip70_low", "chip50", "chip70_high")})
            row["chip_as_of"] = None if pd.isna(profile_row["chip_as_of"]) else profile_row["chip_as_of"]
            values = [row[f"ma{period}"] for period in MA_PERIODS]
            row["ma_alignment"] = "insufficient" if None in values else "bullish" if all(a > b for a, b in zip(values, values[1:])) else "bearish" if all(a < b for a, b in zip(values, values[1:])) else "mixed"
            row.update({f"ma{period}_direction": item[f"ma{period}_direction"] for period in (20, 60)})
            rows.append(row)
    return {"version": "czsc_chart_overlays_v1", "ma_periods": list(MA_PERIODS), "boll_window": 20,
            "boll_std_multiplier": 2, "chip_window": CHIP_WINDOW, "chip_mass": .7,
            "chip_method": "daily_hlc3_volume_weighted_quantiles", "rows": rows,
            "chip_profile": _chip_profile(daily_bars, rows[-1]) if rows and daily_bars is not None else None}
