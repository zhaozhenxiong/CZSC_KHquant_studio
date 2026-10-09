"""Trailing price/volume and point-in-time frozen CZSC research features.

Native BS2/BS3 are MA-assisted signals, not a claim of complete Chan buy points.
The raw native observation can use an extendible last pen; the confirmed view
selects only ``finished_bis``. Neither view is backfilled from final structures.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.adapters.czsc_adapter import SOURCE_SHA256, legacy_fuyao_source, normalize_symbol
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_analysis import CzscSession, _pen_key, _zone_key, strategy_config
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES

FEATURE_VERSION = "czsc_point_in_time_ma_volume_v2"
FEATURE_COLUMNS = (
    "ret_1", "ret_5", "ret_20",
    "ma5_bias", "ma10_bias", "ma20_bias", "ma60_bias",
    "ma5_slope5", "ma10_slope5", "ma20_slope5", "ma60_slope5",
    "ma_alignment", "close_above_ma20", "atr14_ratio", "volatility20",
    "drawdown20", "drawdown60", "vol_ratio5", "vol_ratio20", "vol_ma5_to20",
    "amount_ratio20", "amount_ma5_to20", "up_volume_ratio", "down_volume_ratio",
    "up_volume_expansion", "pullback_contraction",
    "bi1_direction", "bi1_return", "bi1_length", "bi1_volume_ratio",
    "bi2_direction", "bi2_return", "bi2_length",
    "bi3_direction", "bi3_return", "bi3_length", "bi1_power_ratio",
    "confirmed_age", "zone_low_distance", "zone_high_distance", "zone_width_ratio",
    "weekly_return", "weekly_direction", "monthly_return", "monthly_direction",
    "weekly_bi_direction", "monthly_bi_direction", "native_bs2_side", "native_bs3_side",
    "finished_bi_count",
)
FEATURE_SCHEMA_HASH = stable_hash({"version": FEATURE_VERSION, "columns": FEATURE_COLUMNS,
                                   "native_source_sha256": SOURCE_SHA256, "verified_sources": sorted(VERIFIED_SOURCES)})
_NATIVE_SIGNALS = (("bs2", "cxt_second_bs_V230320", 21), ("bs3", "cxt_third_bs_V230318", 34))
_SIDES = {"二买": 1, "三买": 1, "二卖": -1, "三卖": -1}


def _direction(pen: Any) -> int:
    return {"向上": 1, "向下": -1}.get(str(pen.direction), 0)


def _day(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close", "volume", "amount"}
    if frame.empty or not required <= set(frame.columns):
        raise ValueError("特征输入需要非空单股行情及完整 OHLCV/amount 字段")
    if frame["symbol"].isna().any() or frame["symbol"].astype(str).nunique() != 1:
        raise ValueError("特征不能混合不同股票行情")
    result = frame.copy()
    symbol = normalize_symbol(str(frame.iloc[0]["symbol"]))
    result["symbol"] = symbol
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].isna().any() or result["date"].dt.tz is not None:
        raise ValueError("行情日期必须是有效本地日线日期")
    if (result["date"] != result["date"].dt.normalize()).any():
        raise ValueError("行情日期不能含盘中时刻")
    if result["date"].duplicated().any() or not result["date"].is_monotonic_increasing:
        raise ValueError("行情日期重复或乱序")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    prices = result[["open", "high", "low", "close"]]
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
        raise ValueError("OHLC 价格缺失或不合法；不能跳过日期推进结构")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any() or (result["low"] > result[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("OHLC 范围不合法")
    # Local sequential ids preserve windows even when the caller retains a sliced
    # frame's old ids/index. Missing volume is disclosed and never filled in output.
    result["id"] = np.arange(len(result), dtype=int)
    result["dt"] = result["date"] + pd.Timedelta(hours=15)
    return result


def _trailing_features(frame: pd.DataFrame) -> pd.DataFrame:
    close, volume, amount = frame["close"], frame["volume"], frame["amount"]
    volume = volume.where(np.isfinite(volume) & (volume >= 0))
    amount = amount.where(np.isfinite(amount) & (amount >= 0))
    values = pd.DataFrame(index=frame.index)
    for period in (1, 5, 20):
        values[f"ret_{period}"] = close.pct_change(period, fill_method=None)
    for period in (5, 10, 20, 60):
        ma = close.rolling(period, min_periods=period).mean()
        values[f"ma{period}"] = ma
        values[f"ma{period}_bias"] = close / ma - 1
        values[f"ma{period}_slope5"] = ma / ma.shift(5) - 1
    values["ma_alignment"] = (np.sign(values["ma5"] - values["ma10"]) + np.sign(values["ma10"] - values["ma20"]) + np.sign(values["ma20"] - values["ma60"])) / 3
    values["close_above_ma20"] = np.sign(close - values["ma20"])
    previous = close.shift(1)
    true_range = pd.concat([frame["high"] - frame["low"], (frame["high"] - previous).abs(), (frame["low"] - previous).abs()], axis=1).max(axis=1)
    values["atr14_ratio"] = true_range.rolling(14, min_periods=14).mean() / close
    values["volatility20"] = values["ret_1"].rolling(20, min_periods=20).std(ddof=0)
    for period in (20, 60):
        values[f"drawdown{period}"] = close / close.rolling(period, min_periods=period).max() - 1
    for period in (5, 20):
        # The comparison baseline excludes today's volume/amount.
        baseline = volume.shift(1).rolling(period, min_periods=period).mean().replace(0, np.nan)
        values[f"vol_ratio{period}"] = volume / baseline
    vol5, vol20 = volume.rolling(5, min_periods=5).mean(), volume.rolling(20, min_periods=20).mean()
    amount5, amount20 = amount.rolling(5, min_periods=5).mean(), amount.rolling(20, min_periods=20).mean()
    values["vol_ma5_to20"] = vol5 / vol20.replace(0, np.nan)
    values["amount_ratio20"] = amount / amount.shift(1).rolling(20, min_periods=20).mean().replace(0, np.nan)
    values["amount_ma5_to20"] = amount5 / amount20.replace(0, np.nan)
    values["up_volume_ratio"] = values["vol_ratio20"].where(values["ret_1"] > 0, 0)
    values["down_volume_ratio"] = values["vol_ratio20"].where(values["ret_1"] < 0, 0)
    values["up_volume_expansion"] = ((values["ret_1"] > 0) & (values["vol_ratio20"] >= 1.2)).astype(float)
    values["pullback_contraction"] = ((values["ret_1"] < 0) & (values["vol_ratio20"] <= .8)).astype(float)
    return values.replace([np.inf, -np.inf], np.nan)


def build_features(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """One row per supplied daily bar, including suspension and ineligible rows.

    ``research.quality_window_bars`` defaults to 120 (minimum 60). The entire
    trailing window and every referenced pen/zone dependency must use a source
    in the shared verified retrieval contract, have ``has_trade_price == 1``
    and must not contain legacy Fuyao prices.
    Native structures use zero volume only to advance a missing-volume row;
    this substitution remains ineligible and does not alter input or features.
    Missing structure/high-period features remain NaN for training imputation.
    """
    settings = strategy_config(config)
    quality_window = int(settings.get("research", {}).get("quality_window_bars", 120))
    if quality_window < 60:
        raise ValueError("研究质量窗口至少60根行情")
    bars = _validate_input(frame)
    values = _trailing_features(bars)
    trailing_records = values.to_dict(orient="records")
    input_records = bars.to_dict(orient="records")
    daily_dates = bars["date"].to_numpy()
    symbol = str(bars.iloc[0]["symbol"])
    count = len(bars)
    source = bars["source"].fillna("").astype(str) if "source" in bars else pd.Series("", index=bars.index)
    missing_source = source.str.strip().eq("").to_numpy()
    unverified_source = ~source.isin(VERIFIED_SOURCES).to_numpy()
    legacy = source.map(legacy_fuyao_source).to_numpy(dtype=bool)
    verified = pd.to_numeric(bars["has_trade_price"], errors="coerce").eq(1).to_numpy() if "has_trade_price" in bars else np.zeros(count, dtype=bool)
    quantity = bars[["volume", "amount"]].to_numpy(dtype=float)
    missing_quantity = ~np.isfinite(quantity).all(axis=1)
    negative_quantity = (quantity < 0).any(axis=1)
    mismatch_quantity = ((quantity[:, 0] == 0) != (quantity[:, 1] == 0))
    suspended = (quantity == 0).all(axis=1) & ~missing_quantity
    price_quality_bad = unverified_source | legacy | ~verified
    bad_quantity = missing_quantity | negative_quantity | mismatch_quantity
    invalid = price_quality_bad | bad_quantity
    reason_masks = {"source_missing_window": missing_source, "unverified_source_window": unverified_source, "legacy_fuyao_window": legacy,
                    "trade_price_unverified_window": ~verified, "missing_values_window": missing_quantity,
                    "negative_volume_amount_window": negative_quantity, "volume_amount_mismatch_window": mismatch_quantity}
    totals = {key: np.r_[0, np.cumsum(mask)] for key, mask in reason_masks.items()}
    bad_totals = np.r_[0, np.cumsum(invalid)]
    suspension_totals = np.r_[0, np.cumsum(suspended)]
    native_bars = bars.copy()
    for column in ("volume", "amount"):
        native_bars[column] = native_bars[column].where(np.isfinite(native_bars[column]) & (native_bars[column] >= 0), 0.0)
    session = CzscSession(symbol, settings, count, collect_confirmations=False, create_position=False)
    pen_confirmations: dict[tuple, tuple[str, int]] = {}
    zone_confirmations: dict[tuple, str] = {}
    raw_formations: dict[tuple, str] = {}
    signal_confirmations: dict[tuple, str] = {}
    higher_closed_at: dict[str, str] = {}
    higher_counts = {"周线": 0, "月线": 0}
    input_digest = hashlib.sha256()
    rows = []
    for index, row in enumerate(native_bars.itertuples(index=False)):
        date = _day(row.date)
        available = date + "T15:00:00+08:00"
        session.advance_structure(row)
        # Reading bounded native pen collections is necessary; never copy payload
        # history or use the final chart to reconstruct earlier snapshots.
        daily = session.kas["日线"]
        finished = daily.finished_bis
        bi_list = daily.bi_list
        snapshot = trailing_records[index].copy()
        snapshot.update({key: np.nan for key in FEATURE_COLUMNS if key not in snapshot})
        for pen in finished[-5:]:
            pen_confirmations.setdefault(_pen_key(pen), (available, index))
        dependencies = [index]
        for offset, pen in enumerate(reversed(finished[-3:]), 1):
            snapshot[f"bi{offset}_direction"] = _direction(pen)
            snapshot[f"bi{offset}_return"] = float(pen.fx_b.fx) / float(pen.fx_a.fx) - 1
            snapshot[f"bi{offset}_length"] = float(pen.length)
        metadata: dict[str, Any] = {"symbol": symbol, "date": input_records[index]["date"], "available_at": available,
            "source": str(source.iloc[index]), "structure_anchor": None, "structure_confirmed_at": None,
            "structure_low": None, "structure_high": None, "zone_anchor": None, "zone_confirmed_at": None,
            "zone_low": None, "zone_high": None, "point_type": "其他", "point_confirmed": False,
            "point_anchor": None, "point_formed_at": None, "point_confirmed_at": None,
            "suspended": bool(suspended[index]), "suspension_bars_window": int(suspension_totals[index + 1] - suspension_totals[max(0, index + 1 - quality_window)]),
            "engine_zero_quantity_substitution": bool(missing_quantity[index] or negative_quantity[index])}
        if finished:
            last = finished[-1]
            pen_time, pen_index = pen_confirmations[_pen_key(last)]
            snapshot["confirmed_age"] = float(index - pen_index)
            snapshot["bi1_power_ratio"] = float(last.power_price) / float(finished[-3].power_price) if len(finished) >= 3 and finished[-3].power_price > 0 else np.nan
            raw_pen = last.raw_bars
            pen_ids = [int(item.id) for item in raw_pen]
            pen_volume = bars.iloc[pen_ids]["volume"].mean() if pen_ids else np.nan
            baseline = bars["volume"].iloc[max(0, index - 20):index].mean()
            snapshot["bi1_volume_ratio"] = float(pen_volume / baseline) if baseline > 0 else np.nan
            metadata.update({"structure_anchor": _day(last.fx_b.dt), "structure_confirmed_at": pen_time,
                             "structure_low": float(last.low), "structure_high": float(last.high)})
            dependencies.extend(int(item.id) for pen in finished[-5:] for item in pen.raw_bars)
        snapshot["finished_bi_count"] = float(len(finished))
        zones = daily.zs_list
        if zones:
            zone = zones[-1]
            if len(zone.bis) >= 3 and zone.is_valid():
                zone_time = zone_confirmations.setdefault(_zone_key(zone), available)
                snapshot.update({"zone_low_distance": float(row.close) / float(zone.zd) - 1,
                                 "zone_high_distance": float(row.close) / float(zone.zg) - 1,
                                 "zone_width_ratio": (float(zone.zg) - float(zone.zd)) / float(row.close)})
                metadata.update({"zone_anchor": _day(zone.edt), "zone_confirmed_at": zone_time,
                                 "zone_low": float(zone.zd), "zone_high": float(zone.zg)})
                # Native ZS bounds use its first three pens. Extended zones do
                # not change those historical price dependencies.
                dependencies.extend(int(item.id) for pen in zone.bis[:3] for item in pen.raw_bars)
        for frequency, prefix in (("周线", "weekly"), ("月线", "monthly")):
            closed = session.closed[frequency]
            if len(closed) != higher_counts[frequency]:
                higher_closed_at[frequency], higher_counts[frequency] = available, len(closed)
            metadata[prefix + "_anchor"] = _day(closed[-1].dt) if closed else None
            metadata[prefix + "_available_at"] = higher_closed_at.get(frequency)
            snapshot[prefix + "_closed_bars"] = float(len(closed))
            if len(closed) >= 2:
                period_return = float(closed[-1].close) / float(closed[-2].close) - 1
                snapshot[prefix + "_return"] = period_return
                snapshot[prefix + "_direction"] = float(np.sign(period_return))
            high_finished = session.kas[frequency].finished_bis if frequency in session.kas else []
            if high_finished:
                snapshot[prefix + "_bi_direction"] = float(_direction(high_finished[-1]))
                # Aggregated bar ids are per-frequency ids, not daily ids.
                # Cover the first pen's complete source bucket by calendar span.
                span = 7 if frequency == "周线" else 31
                lower = pd.Timestamp(high_finished[-1].sdt).normalize() - pd.Timedelta(days=span)
                dependencies.append(int(np.searchsorted(daily_dates, lower.to_datetime64())))
        confirmed_di = len(bi_list) - len(finished) + 1
        for short_name, native_name, period in _NATIVE_SIGNALS:
            parameters = {"ma_type": "SMA", "timeperiod": period, "di": 1}
            raw_signals = session.czsc._native.call_signal(native_name, daily, parameters)
            raw_value = str(raw_signals[0].v1)
            raw_anchor = _pen_key(bi_list[-1]) if bi_list else None
            raw_key = (short_name, raw_anchor, raw_value)
            formed = raw_formations.setdefault(raw_key, available) if raw_value in _SIDES else None
            confirmed_value, confirmed_at = "其他", None
            if finished:
                if confirmed_di == 1:
                    confirmed_value = raw_value
                else:
                    parameters["di"] = confirmed_di
                    confirmed_value = str(session.czsc._native.call_signal(native_name, daily, parameters)[0].v1)
                if confirmed_value in _SIDES:
                    key = (short_name, _pen_key(finished[-1]), confirmed_value)
                    confirmed_at = signal_confirmations.setdefault(key, available)
            metadata.update({f"raw_native_{short_name}": raw_value, f"raw_native_{short_name}_formed_at": formed,
                             f"raw_native_{short_name}_anchor": raw_anchor[1] if raw_anchor else None,
                             f"confirmed_native_{short_name}": confirmed_value,
                             f"confirmed_native_{short_name}_at": confirmed_at})
            snapshot[f"native_{short_name}_side"] = float(_SIDES.get(confirmed_value, 0))
            if confirmed_value in _SIDES:
                key = (short_name, _pen_key(finished[-1]), confirmed_value)
                metadata.update({"point_type": confirmed_value + "辅助", "point_confirmed": True,
                                 "point_anchor": _day(finished[-1].fx_b.dt),
                                 "point_formed_at": min(raw_formations.get(key, confirmed_at), confirmed_at), "point_confirmed_at": confirmed_at})
        window_start = max(0, index + 1 - quality_window)
        reasons = [key for key, total in totals.items() if total[index + 1] - total[window_start] > 0]
        if unverified_source[index]:
            reasons.append("unverified_source_current_bar")
        if index + 1 < max(int(settings["warmup_bars"]), quality_window):
            reasons.insert(0, "quality_warmup")
        # Include the MA lookback immediately before referenced structures. The
        # native BS3 MA uses 34 bars at a pen's fractal, including its right bar.
        dependency_start = max(0, min(dependencies) - 34)
        if bad_totals[index + 1] - bad_totals[dependency_start] > 0:
            reasons.append("structural_input_unverified")
        if totals["unverified_source_window"][index + 1] - totals["unverified_source_window"][dependency_start] > 0:
            reasons.append("unverified_source_structural")
        if suspended[index]:
            reasons.append("suspended_current_bar")
        elif snapshot["vol_ratio20"] != snapshot["vol_ratio20"]:
            reasons.append("volume_baseline_unavailable")
        eligible = not reasons
        czsc_buy = snapshot["native_bs2_side"] == 1 or snapshot["native_bs3_side"] == 1
        czsc_sell = snapshot["native_bs2_side"] == -1 or snapshot["native_bs3_side"] == -1
        price_buy = snapshot["ma20"] > snapshot["ma60"] and float(row.close) > snapshot["ma20"] and snapshot["ma20_slope5"] > 0 and snapshot["ma60_slope5"] > 0 and (snapshot["up_volume_expansion"] == 1 or snapshot["pullback_contraction"] == 1)
        price_sell = float(row.close) < snapshot["ma20"] and snapshot["ma20_slope5"] < 0
        metadata.update({"input_eligible": eligible, "reason_codes": list(dict.fromkeys(reasons)),
                         "rule_czsc_buy": bool(eligible and czsc_buy), "rule_czsc_sell": bool(eligible and czsc_sell),
                         "rule_price_volume_buy": bool(eligible and price_buy), "rule_price_volume_sell": bool(eligible and price_sell),
                         "rule_buy": bool(eligible and czsc_buy and price_buy and snapshot["weekly_direction"] == 1),
                         "rule_sell": bool(eligible and (czsc_sell or price_sell or snapshot["weekly_direction"] == -1))})
        current_input = input_records[index]
        digest_values = {key: (None if pd.isna(current_input[key]) else str(current_input[key])) for key in ("symbol", "date", "open", "high", "low", "close", "volume", "amount")}
        digest_values.update({"source": str(source.iloc[index]), "has_trade_price": str(current_input.get("has_trade_price", "missing"))})
        input_digest.update((json.dumps(digest_values, ensure_ascii=False, sort_keys=True) + "\n").encode())
        metadata["input_hash"] = input_digest.hexdigest()
        rows.append({**snapshot, **metadata})
    result = pd.DataFrame(rows, index=frame.index)
    numeric_columns = list(FEATURE_COLUMNS) + ["ma5", "ma10", "ma20", "ma60", "weekly_closed_bars", "monthly_closed_bars",
                                                "structure_low", "structure_high", "zone_low", "zone_high"]
    for column in numeric_columns:
        result[column] = result[column].astype(float).replace([np.inf, -np.inf], np.nan)
    string_columns = ["symbol", "source", "available_at", "structure_anchor", "structure_confirmed_at",
                      "zone_anchor", "zone_confirmed_at", "point_type", "point_anchor", "point_formed_at", "point_confirmed_at",
                      "weekly_anchor", "weekly_available_at", "monthly_anchor", "monthly_available_at", "input_hash"]
    string_columns.extend(f"{view}_native_{short}{suffix}" for short, _, _ in _NATIVE_SIGNALS
                          for view, suffixes in (("raw", ("", "_formed_at", "_anchor")), ("confirmed", ("", "_at")))
                          for suffix in suffixes)
    # Explicit object/None metadata prevents Pandas' all-empty prefix inference
    # from changing column types after the first future signal is formed.
    for column in string_columns:
        result[column] = result[column].astype(object).where(result[column].notna(), None)
    output_schema = {column: str(dtype) for column, dtype in result.dtypes.items()}
    result.attrs.update({"feature_version": FEATURE_VERSION, "feature_columns": list(FEATURE_COLUMNS),
                         "schema": output_schema, "output_schema_hash": stable_hash(output_schema),
                         "schema_hash": FEATURE_SCHEMA_HASH, "config_hash": stable_hash(settings),
                         "data_version": str(frame.attrs.get("data_version") or input_digest.hexdigest()),
                         "source_sha256": SOURCE_SHA256, "quality_window_bars": quality_window,
                         "verified_sources": sorted(VERIFIED_SOURCES),
                         "higher_period_closure": "next_period_observed", "price_basis": frame.attrs.get("price_basis", "unverified"),
                         "signal_semantics": "frozen_native_ma_assisted_bs2_bs3_finished_pen_view",
                         "input_eligibility_scope": "feature_source_quality_not_execution_or_board",
                         "calendar_gap_check": "requires_external_verified_calendar_no_dates_inserted_or_removed",
                         "missing_quantity_policy": "preserve_row_and_features_native_zero_only_ineligible",
                         "volume_unit": frame.attrs.get("volume_unit", "shares"), "amount_unit": frame.attrs.get("amount_unit", "CNY")})
    return result
