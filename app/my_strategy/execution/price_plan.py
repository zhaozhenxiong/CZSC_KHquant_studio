"""Point-in-time price plans and daily-bar limit-order fill semantics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

PricePlanSide = Literal["BUY", "SELL"]

PRICE_PLAN_COLUMNS = (
    "planned_order_side",
    "planned_order_type",
    "planned_target_price",
    "planned_price_low",
    "planned_price_high",
    "planned_price_diff",
    "planned_price_diff_pct",
    "planned_atr_pct",
    "planned_effective_session",
    "execution_reminder",
)


def attach_price_plans(
    market: pd.DataFrame,
    decisions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Attach next-session price plans using only data available on each signal date.

    The estimator is deliberately simple and auditable: a rolling true-range
    percentage determines a bounded discount for buys or premium for sells.
    No next-day open/high/low/close is used to create the plan.
    """
    out = decisions.copy()
    if out.empty:
        return out

    price_cfg = (config.get("backtest", {}) or {}).get("price_plan", {}) or {}
    lookback = max(2, int(price_cfg.get("lookback_days", 20)))
    atr_offset_ratio = max(0.0, float(price_cfg.get("atr_offset_ratio", 0.35)))
    range_atr_ratio = max(atr_offset_ratio, float(price_cfg.get("range_atr_ratio", 0.65)))
    min_offset = max(0.0, float(price_cfg.get("min_offset_pct", 0.003)))
    max_offset = max(min_offset, float(price_cfg.get("max_offset_pct", 0.025)))
    max_range = max(max_offset, float(price_cfg.get("max_range_pct", 0.05)))
    fallback_atr = max(min_offset, float(price_cfg.get("fallback_atr_pct", 0.012)))

    bars = market.copy()
    bars["date"] = bars["date"].astype(str).str[:10]
    if "close" not in bars.columns:
        raise ValueError("market data requires close for price planning")
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    for column in ("high", "low"):
        source = bars[column] if column in bars.columns else bars["close"]
        bars[column] = pd.to_numeric(source, errors="coerce").fillna(bars["close"])
    previous_close = bars["close"].shift(1)
    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous_close).abs(),
            (bars["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_pct = (
        true_range.rolling(lookback, min_periods=2).mean()
        / bars["close"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(fallback_atr)
    bars = bars.assign(_plan_close=bars["close"], _plan_atr_pct=atr_pct)

    out["date"] = out["date"].astype(str).str[:10]
    out = out.merge(bars[["date", "_plan_close", "_plan_atr_pct"]], on="date", how="left")
    close = pd.to_numeric(out["_plan_close"], errors="coerce")
    atr = pd.to_numeric(out["_plan_atr_pct"], errors="coerce").fillna(fallback_atr).clip(lower=0.0)
    offset = (atr * atr_offset_ratio).clip(lower=min_offset, upper=max_offset)
    range_offset = (atr * range_atr_ratio).clip(lower=offset, upper=max_range)
    near_offset = (offset * 0.60).clip(lower=min_offset / 2.0, upper=max_offset)

    action = out.get("final_action", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    risk_action = out.get("risk_action", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    target_source = out["target_weight"] if "target_weight" in out.columns else pd.Series(0.0, index=out.index)
    target = pd.to_numeric(target_source, errors="coerce").fillna(0.0)
    target_change = target.diff().fillna(target)

    buy_mask = action.eq("buy") | (target_change > 1e-9)
    sell_mask = action.isin(["sell", "reduce"]) | (target_change < -1e-9)
    gate_action = out.get("entry_quality_gate_action", pd.Series("", index=out.index)).fillna("").astype(str)
    blocked_entry = gate_action.isin(["buy_blocked_hold", "buy_blocked_watch"])
    buy_mask &= ~blocked_entry
    sell_mask &= ~blocked_entry
    force_sell = risk_action.eq("force_sell")
    sell_mask |= force_sell
    buy_mask &= ~sell_mask

    side = pd.Series("", index=out.index, dtype=object)
    side.loc[buy_mask] = "BUY"
    side.loc[sell_mask] = "SELL"

    target_price = close.copy()
    target_price.loc[buy_mask] = close.loc[buy_mask] * (1.0 - offset.loc[buy_mask])
    target_price.loc[sell_mask & ~force_sell] = close.loc[sell_mask & ~force_sell] * (
        1.0 + offset.loc[sell_mask & ~force_sell]
    )

    price_low = target_price.copy()
    price_high = target_price.copy()
    price_low.loc[buy_mask] = close.loc[buy_mask] * (1.0 - range_offset.loc[buy_mask])
    price_high.loc[buy_mask] = close.loc[buy_mask] * (1.0 - near_offset.loc[buy_mask])
    price_low.loc[sell_mask & ~force_sell] = close.loc[sell_mask & ~force_sell] * (
        1.0 + near_offset.loc[sell_mask & ~force_sell]
    )
    price_high.loc[sell_mask & ~force_sell] = close.loc[sell_mask & ~force_sell] * (
        1.0 + range_offset.loc[sell_mask & ~force_sell]
    )

    no_plan = side.eq("") | close.isna() | close.le(0)
    order_type = pd.Series("NEXT_DAY_LIMIT", index=out.index, dtype=object)
    order_type.loc[force_sell] = "NEXT_DAY_MARKET"
    order_type.loc[no_plan] = ""
    for series in (target_price, price_low, price_high):
        series.loc[no_plan] = np.nan

    price_diff = target_price - close
    price_diff_pct = price_diff / close.replace(0, np.nan)
    reminders = pd.Series("", index=out.index, dtype=object)
    reminders.loc[buy_mask & ~no_plan] = [
        f"明日（下一交易日）盘中回落至 {value:.2f} 附近再执行买入；参考区间 {low:.2f}–{high:.2f}。"
        for value, low, high in zip(
            target_price.loc[buy_mask & ~no_plan],
            price_low.loc[buy_mask & ~no_plan],
            price_high.loc[buy_mask & ~no_plan],
        )
    ]
    regular_sell = sell_mask & ~force_sell & ~no_plan
    reminders.loc[regular_sell] = [
        f"明日（下一交易日）盘中反弹至 {value:.2f} 附近再执行卖出；参考区间 {low:.2f}–{high:.2f}。"
        for value, low, high in zip(
            target_price.loc[regular_sell],
            price_low.loc[regular_sell],
            price_high.loc[regular_sell],
        )
    ]
    reminders.loc[force_sell & ~no_plan] = "风险强制退出：明日（下一交易日）开盘优先卖出，不等待目标价。"

    out["planned_order_side"] = side
    out["planned_order_type"] = order_type
    out["planned_target_price"] = target_price.round(4)
    out["planned_price_low"] = price_low.round(4)
    out["planned_price_high"] = price_high.round(4)
    out["planned_price_diff"] = price_diff.round(4)
    out["planned_price_diff_pct"] = price_diff_pct.round(8)
    out["planned_atr_pct"] = atr.round(8)
    out["planned_effective_session"] = np.where(no_plan, "", "next_trading_day")
    out["execution_reminder"] = reminders
    return out.drop(columns=["_plan_close", "_plan_atr_pct"])


def resolve_daily_limit_fill(
    side: PricePlanSide,
    *,
    open_price: float,
    high_price: float,
    low_price: float,
    limit_price: float,
) -> float | None:
    """Resolve a conservative daily-bar fill without assuming an intraday path."""
    values = (open_price, high_price, low_price, limit_price)
    if not all(np.isfinite(value) and value > 0 for value in values):
        return None
    if side == "BUY":
        if low_price > limit_price:
            return None
        return min(open_price, limit_price)
    if high_price < limit_price:
        return None
    return max(open_price, limit_price)
