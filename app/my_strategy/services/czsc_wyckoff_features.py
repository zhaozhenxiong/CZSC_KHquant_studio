"""Independent 32-column point-in-time Wyckoff price/volume profile."""
from __future__ import annotations

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services.czsc_wyckoff_events import (
    DEFAULT_EVENT_CONFIG, WYCKOFF_EVENT_VERSION, detect_wyckoff_events, event_config,
)

WYCKOFF_FEATURE_PROFILE = "wyckoff_pv_v1"
WYCKOFF_FEATURE_VERSION = WYCKOFF_FEATURE_PROFILE
W_INPUT_COLUMNS = (
    "w_spread_atr", "w_body_atr", "w_close_location", "w_upper_shadow", "w_lower_shadow", "w_gap_atr",
    "w_relative_volume20", "w_volume_robust60", "w_volume5_to20", "w_relative_amount20",
    "w_up_down_volume20", "w_price_volume_divergence", "w_pullback_push_volume20", "w_volume_contraction5",
    "w_range_width_atr", "w_range_duration", "w_range_position", "w_support_distance_atr",
    "w_resistance_distance_atr", "w_compression20_60", "w_down_probe_atr", "w_up_probe_atr",
    "w_spring", "w_test", "w_sos", "w_lps", "w_upthrust", "w_sow", "w_lpsy",
    "w_event_age", "w_demand_score", "w_supply_score",
)
WYCKOFF_COLUMNS = W_INPUT_COLUMNS
DEFAULT_QUALITY_CONFIG = {
    "quality_window": 120, "minimum_active_bars": 60,
    "unit_price_tolerance": 0.03, "mainboard_action_gap": 0.15,
    "other_board_action_gap": 0.25,
}
WYCKOFF_SEMANTICS = {
    "version": WYCKOFF_FEATURE_VERSION, "columns": W_INPUT_COLUMNS,
    "event_version": WYCKOFF_EVENT_VERSION, "events": DEFAULT_EVENT_CONFIG,
    "raw_price_basis": "unadjusted", "volume_unit": "shares", "amount_unit": "CNY",
    "raw_only": True, "price_moving_averages_or_czsc_inputs": "excluded",
    "atr": "prior_14_bars_mean_true_range_excludes_current",
    "normalization": "volume_amount_baselines_exclude_current_20_60_bars",
    "robust_volume": "prior_60_median_and_median_absolute_deviation_scale_1.4826_floor_1_share",
    "ranges": "prior_60_extrema_exclude_current_frozen_pending_event_boundaries",
    "optional_missing": "event_age_no_prior_event_train_only_imputation",
    "eligibility": DEFAULT_QUALITY_CONFIG, "sources": sorted(VERIFIED_SOURCES),
    "action_gap": "causal_suspicion_only_current_and_trailing_120_rejected_no_repair",
    "historical_identity": "missing_ST_delisting_and_action_timeline_blocks_formal_qualification",
}
WYCKOFF_SCHEMA_HASH = stable_hash(WYCKOFF_SEMANTICS)


def get_wyckoff_profile(config=None):
    """Exact column order and threshold identity; custom thresholds get new hash."""
    settings = event_config(config)
    quality = _quality_config(config)
    semantics = {**WYCKOFF_SEMANTICS, "events": settings, "eligibility": quality}
    return {"name": WYCKOFF_FEATURE_PROFILE, "version": WYCKOFF_FEATURE_VERSION,
            "schema_hash": stable_hash(semantics), "columns": list(W_INPUT_COLUMNS),
            "strategy_version": "czsc_wyckoff_three_experts_v1", "semantics": semantics}


def _quality_config(config):
    settings = dict(DEFAULT_QUALITY_CONFIG)
    if config:
        incoming = config.get("quality", {})
        settings.update({key: incoming[key] for key in settings if key in incoming})
    for key in ("quality_window", "minimum_active_bars"):
        if int(settings[key]) != settings[key] or settings[key] < 1:
            raise ValueError("invalid Wyckoff quality window")
        settings[key] = int(settings[key])
    if settings["quality_window"] < 120 or settings["minimum_active_bars"] < 60:
        raise ValueError("Wyckoff quality requires at least 120 bars and 60 active bars")
    for key in ("unit_price_tolerance", "mainboard_action_gap", "other_board_action_gap"):
        if not np.isfinite(settings[key]) or not 0 < settings[key] < 1:
            raise ValueError("invalid Wyckoff quality threshold: " + key)
    return settings


def _rolling_any(value, window):
    return pd.Series(np.asarray(value, dtype=bool)).rolling(window, min_periods=1).max().to_numpy(dtype=bool)


def _prior_mad(series, window=60):
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    if len(values) > window:
        windows = np.lib.stride_tricks.sliding_window_view(values[:-1], window)
        # Chunking bounds temporary memory without changing the causal window.
        for start in range(0, len(windows), 4096):
            rows = windows[start:start + 4096]
            median = np.median(rows, axis=1)
            result[window + start:window + start + len(rows)] = np.median(np.abs(rows - median[:, None]), axis=1)
    return pd.Series(result, index=series.index)


def _raw_quality(raw, frame, config=None):
    settings = _quality_config(config); n = len(raw); window = settings["quality_window"]
    numeric = raw[["open", "high", "low", "close", "volume", "amount"]].apply(pd.to_numeric, errors="coerce")
    op, hi, lo, cl, vol, amount = [numeric[key].to_numpy(dtype=float) for key in numeric]
    dates = pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    prices = np.column_stack((op, hi, lo, cl))
    bad_price = (~np.isfinite(prices).all(axis=1) | (prices <= 0).any(axis=1)
                 | (hi < np.maximum(op, cl)) | (lo > np.minimum(op, cl)) | (hi < lo))
    bad_quantity = (~np.isfinite(np.column_stack((vol, amount))).all(axis=1)
                    | (vol < 0) | (amount < 0) | ((vol == 0) != (amount == 0)))
    suspended = (vol == 0) & (amount == 0) & ~bad_quantity
    source = raw["source"].fillna("").astype(str) if "source" in raw else pd.Series("", index=raw.index)
    source_bad = ~source.isin(VERIFIED_SOURCES).to_numpy()
    trade = pd.to_numeric(raw["has_trade_price"], errors="coerce").eq(1).to_numpy() if "has_trade_price" in raw else np.zeros(n, bool)
    attrs = {**frame.attrs, **raw.attrs}
    unit_verified = attrs.get("volume_unit") == "shares" and attrs.get("amount_unit") == "CNY"
    unadjusted = attrs.get("price_basis") == "unadjusted"
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = amount / vol
    tolerance = settings["unit_price_tolerance"]
    unit_disagreement = (vol > 0) & (amount > 0) & ((avg < lo * (1 - tolerance)) | (avg > hi * (1 + tolerance)))
    prev = np.r_[np.nan, cl[:-1]]
    symbol = str(raw["symbol"].iloc[0]) if n and "symbol" in raw else ""
    mainboard = (symbol.startswith("60") and symbol.endswith(".SH")) or (symbol.startswith("00") and symbol.endswith(".SZ"))
    cutoff = settings["mainboard_action_gap"] if mainboard else settings["other_board_action_gap"]
    with np.errstate(divide="ignore", invalid="ignore"):
        action_gap = np.abs(op / prev - 1) > cutoff
    # Merely flag a possible corporate action/data discontinuity. No eventual
    # downloaded adjustment factor is used to alter historical coordinates.
    calendar_bad = np.zeros(n, dtype=bool)
    calendar = (config or {}).get("market_dates")
    if calendar is not None:
        if calendar != sorted(set(calendar)):
            raise ValueError("Wyckoff calendar must be sorted and unique")
        following = dict(zip(calendar[:-1], calendar[1:])); known = set(calendar)
        days = dates.tolist()
        gaps = np.array([day not in known or (i > 0 and following.get(days[i - 1]) != day) for i, day in enumerate(days)])
        calendar_bad = _rolling_any(gaps, window)
    elif "model_reason_codes" in frame or "reason_codes" in frame:
        key = "model_reason_codes" if "model_reason_codes" in frame else "reason_codes"
        calendar_bad = np.array(["calendar_missing_market_bar_window" in (list(x) if isinstance(x, (list, tuple, np.ndarray)) else [])
                                 for x in frame[key]], dtype=bool)
    masks = {
        "wyckoff_quality_warmup": np.arange(n) + 1 < window,
        "wyckoff_invalid_price_window": _rolling_any(bad_price, window),
        "wyckoff_invalid_quantity_window": _rolling_any(bad_quantity, window),
        "wyckoff_unverified_source_window": _rolling_any(source_bad, window),
        "wyckoff_trade_price_unverified_window": _rolling_any(~trade, window),
        "wyckoff_volume_amount_unit_disagreement_window": _rolling_any(unit_disagreement, window),
        "wyckoff_suspected_corporate_action_window": _rolling_any(action_gap, window),
        "wyckoff_calendar_gap_window": calendar_bad,
        "wyckoff_units_not_verified": np.full(n, not unit_verified, dtype=bool),
        "wyckoff_price_basis_not_unadjusted": np.full(n, not unadjusted, dtype=bool),
        "wyckoff_suspended_current_bar": suspended,
    }
    invalid = bad_price | bad_quantity | source_bad | ~trade | unit_disagreement | suspended
    active = pd.Series(~invalid).rolling(window, min_periods=1).sum().to_numpy()
    masks["wyckoff_insufficient_active_history"] = active < settings["minimum_active_bars"]
    bad = np.logical_or.reduce(list(masks.values()))
    reasons = [[] for _ in range(n)]
    for key, mask in masks.items():
        for i in np.flatnonzero(mask):
            reasons[i].append(key)
    return ~bad, reasons, {"corporate_action_suspected": action_gap,
                           "volume_amount_unit_disagreement": unit_disagreement,
                           "suspended": suspended, "active_bars": active,
                           "invalid_price": bad_price, "invalid_quantity": bad_quantity}


def bind_wyckoff_profile(frame, config=None):
    profile = get_wyckoff_profile(config)
    if not set(W_INPUT_COLUMNS) <= set(frame):
        raise ValueError("Wyckoff profile input columns missing")
    result = frame.copy(); result.attrs = dict(frame.attrs)
    result.attrs.update(feature_version=profile["version"], schema_hash=profile["schema_hash"],
                        feature_columns=profile["columns"], feature_profile=profile["name"])
    return result


def prepare_wyckoff_inputs(frame, raw=None, config=None):
    """Add independent W32 and point-in-time metadata without changing old rules.

    One supplied daily row remains one output row, in its original index. Raw
    input cannot be silently sorted, deduplicated, forward filled or adjusted.
    Config changes are included in the separate profile schema identity.
    """
    raw = frame if raw is None else raw
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "amount"}
    if not required <= set(raw):
        raise ValueError("Wyckoff raw columns missing: " + str(sorted(required - set(raw))))
    if len(frame) != len(raw) or not frame.index.equals(raw.index):
        raise ValueError("Wyckoff raw and feature rows/index must match exactly")
    if "date" in frame and not pd.to_datetime(frame["date"]).equals(pd.to_datetime(raw["date"])):
        raise ValueError("Wyckoff raw and feature dates differ")
    dates = pd.to_datetime(raw["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Wyckoff dates must be valid unique and increasing")
    if raw["symbol"].nunique(dropna=False) > 1:
        raise ValueError("Wyckoff inputs require one symbol per frame")
    if len(raw) == 0:
        result = frame.copy()
        for key in W_INPUT_COLUMNS:
            result[key] = pd.Series(index=result.index, dtype=float)
        result["wyckoff_input_eligible"] = pd.Series(index=result.index, dtype=bool)
        result["wyckoff_reason_codes"] = pd.Series(index=result.index, dtype=object)
        for key in ("wyckoff_event", "wyckoff_state", "wyckoff_anchor_at", "wyckoff_observed_at",
                    "wyckoff_available_at", "wyckoff_range_formed_at", "wyckoff_entry_evidence",
                    "wyckoff_qualification_reasons", "wyckoff_event_version"):
            result[key] = pd.Series(index=result.index, dtype=object)
        for key in ("wyckoff_range_low", "wyckoff_range_high", "wyckoff_stop", "wyckoff_active_bars"):
            result[key] = pd.Series(index=result.index, dtype=float)
        for key in ("wyckoff_rule_buy", "wyckoff_rule_sell", "wyckoff_formal_qualified",
                    "wyckoff_corporate_action_suspected", "wyckoff_volume_amount_unit_disagreement", "wyckoff_suspended"):
            result[key] = pd.Series(index=result.index, dtype=bool)
        profile = get_wyckoff_profile(config)
        result.attrs = dict(frame.attrs)
        result.attrs.update(wyckoff_feature_version=profile["version"], wyckoff_schema_hash=profile["schema_hash"],
                            wyckoff_feature_columns=profile["columns"], wyckoff_event_version=WYCKOFF_EVENT_VERSION,
                            wyckoff_profile=profile, wyckoff_formal_qualified=False)
        return result
    settings = event_config(config); profile = get_wyckoff_profile(config)
    eligible, reasons, quality = _raw_quality(raw, frame, config)
    p = raw[["open", "high", "low", "close", "volume", "amount"]].apply(pd.to_numeric, errors="coerce")
    op, hi, lo, cl, vol, amount = [p[key] for key in p]
    previous = cl.shift(1); spread = hi - lo
    true_range = pd.concat([spread, (hi - previous).abs(), (lo - previous).abs()], axis=1).max(axis=1)
    atr = true_range.shift(1).rolling(14, min_periods=14).mean().where(lambda x: x > 0)
    mean20 = vol.shift(1).rolling(20, min_periods=20).mean().where(lambda x: x > 0)
    median60 = vol.shift(1).rolling(60, min_periods=60).median()
    # MAD is fitted separately to each prior window; never centered on today's
    # volume or a final-history statistic.
    mad60 = _prior_mad(vol)
    high60 = hi.shift(1).rolling(settings["range_bars"], min_periods=settings["minimum_range_bars"]).max()
    low60 = lo.shift(1).rolling(settings["range_bars"], min_periods=settings["minimum_range_bars"]).min()
    high20 = hi.shift(1).rolling(20, min_periods=20).max()
    low20 = lo.shift(1).rolling(20, min_periods=20).min()
    width = (high60 - low60).where(lambda x: x > 0)
    direction = np.sign(cl - previous)
    up = vol.where(direction > 0, 0.).shift(1).rolling(20, min_periods=20).sum()
    down = vol.where(direction < 0, 0.).shift(1).rolling(20, min_periods=20).sum()
    up_count = (direction > 0).astype(float).shift(1).rolling(20, min_periods=20).sum()
    down_count = (direction < 0).astype(float).shift(1).rolling(20, min_periods=20).sum()
    duration = np.zeros(len(p), dtype=float); length = 0
    ranges_valid = ((width / atr <= settings["range_max_width_atr"]) & (previous >= low60) & (previous <= high60)).to_numpy()
    for i, valid in enumerate(ranges_valid):
        length = min(120, length + 1) if valid else 0
        duration[i] = length
    with np.errstate(divide="ignore", invalid="ignore"):
        current = pd.DataFrame({
            "w_spread_atr": spread / atr, "w_body_atr": (cl - op) / atr,
            "w_close_location": (cl - lo) / spread.where(spread > 0),
            "w_upper_shadow": (hi - pd.concat([op, cl], axis=1).max(axis=1)) / spread.where(spread > 0),
            "w_lower_shadow": (pd.concat([op, cl], axis=1).min(axis=1) - lo) / spread.where(spread > 0),
            "w_gap_atr": (op - previous) / atr,
            "w_relative_volume20": vol / mean20,
            "w_volume_robust60": (vol - median60) / (1.4826 * mad60).clip(lower=1.),
            "w_volume5_to20": vol.rolling(5, min_periods=5).mean() / mean20,
            "w_relative_amount20": amount / amount.shift(1).rolling(20, min_periods=20).mean().where(lambda x: x > 0),
            "w_up_down_volume20": up / down.clip(lower=1.),
            "w_price_volume_divergence": ((cl - previous) / atr) / (vol / mean20).where(lambda x: x > 0),
            "w_pullback_push_volume20": (down / down_count.clip(lower=1.)) / (up / up_count.clip(lower=1.)).clip(lower=1.),
            "w_volume_contraction5": vol.rolling(5, min_periods=5).mean() / vol.shift(5).rolling(5, min_periods=5).mean().where(lambda x: x > 0),
            "w_range_width_atr": width / atr, "w_range_duration": duration,
            "w_range_position": (cl - low60) / width,
            "w_support_distance_atr": (cl - low60) / atr,
            "w_resistance_distance_atr": (high60 - cl) / atr,
            "w_compression20_60": (high20 - low20) / width,
            "w_down_probe_atr": (low60 - lo).clip(lower=0.) / atr,
            "w_up_probe_atr": (hi - high60).clip(lower=0.) / atr,
        }, index=raw.index).replace([np.inf, -np.inf], np.nan)
    # Zero-spread trade bars have a defined neutral close and no wick; a real
    # zero-volume suspension remains separately ineligible.
    zero_spread = spread.eq(0) & cl.gt(0)
    current.loc[zero_spread, "w_close_location"] = .5
    current.loc[zero_spread, ["w_upper_shadow", "w_lower_shadow"]] = 0.
    unavailable = ~np.isfinite(current.to_numpy(dtype=float)).all(axis=1)
    for i in np.flatnonzero(unavailable):
        reasons[i].append("wyckoff_numeric_context_unavailable")
    eligible &= ~unavailable
    metrics = {"atr": atr, "relative_volume": current["w_relative_volume20"],
               "range_low": low60, "range_high": high60,
               "range_count": lo.shift(1).rolling(settings["range_bars"], min_periods=1).count()}
    events = detect_wyckoff_events(raw, metrics, eligible, config)
    result = frame.copy()
    for key in current:
        result[key] = current[key].astype(float)
    for key in events:
        result[key] = events[key]
    result["wyckoff_input_eligible"] = eligible
    result["wyckoff_reason_codes"] = pd.Series(reasons, index=frame.index, dtype=object)
    for name in ("corporate_action_suspected", "volume_amount_unit_disagreement", "suspended", "active_bars"):
        result["wyckoff_" + name] = quality[name]
    result["wyckoff_formal_qualified"] = False
    result["wyckoff_qualification_reasons"] = pd.Series([
        ["unadjusted_no_verified_corporate_action_timeline", "historical_st_delisting_timeline_unavailable", "shadow_research_only"]
        for _ in range(len(result))], index=result.index, dtype=object)
    result.attrs = dict(frame.attrs)
    result.attrs.update(wyckoff_feature_version=profile["version"], wyckoff_schema_hash=profile["schema_hash"],
                        wyckoff_feature_columns=profile["columns"], wyckoff_event_version=WYCKOFF_EVENT_VERSION,
                        wyckoff_profile=profile, wyckoff_price_basis="unchanged_unadjusted",
                        wyckoff_units={"volume": "shares", "amount": "CNY"},
                        wyckoff_identity_scope="no_historical_ST_or_delisting_qualification", wyckoff_formal_qualified=False)
    return result


enrich_wyckoff_features = prepare_wyckoff_inputs
