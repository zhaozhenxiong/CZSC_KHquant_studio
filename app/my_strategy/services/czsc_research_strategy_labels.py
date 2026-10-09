"""Strategy-aligned labels from one chronological, executable account ledger.

Candidate selection uses contemporaneous rules/plans. Future fills are outcomes,
never a reason to remove an account from the evaluation population.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES

LABEL_VERSION = "czsc_strategy_ledger_v1"


def build_strategy_labels(features: pd.DataFrame, raw: pd.DataFrame,
                          market_dates: Sequence[Any] | None = None, *,
                          entry_policy: str = "fresh", initial_cash: float = 100000,
                          config: dict[str, Any] | None = None,
                          entry_parameters: dict[str, Any] | None = None,
                          decisions: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """Map actual complete round trips to their observed BUY signal row once.

    Labels mature on the final actual SELL date. Partial exits remain open;
    cancelled/rejected/held candidates remain unavailable. ``net_return`` uses
    fixed initial account capital; ``invested_net_return`` uses actual entry
    cash debit including fees. Training must purge label_end after its cutoff.
    Explicit decisions support independent execution/label parity checks.
    """
    if entry_policy not in {"legacy", "fresh", "risk"}:
        raise ValueError("unsupported entry policy")
    if len(features) != len(raw) or raw.empty:
        raise ValueError("strategy labels require matching nonempty feature/raw prefixes")
    if not np.isfinite(float(initial_cash)) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    dates = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d").tolist()
    symbols = raw["symbol"].astype(str)
    if dates != sorted(set(dates)) or symbols.nunique() != 1:
        raise ValueError("strategy labels require one symbol and ascending unique dates")
    if "date" in features and dates != pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d").tolist():
        raise ValueError("strategy label feature dates do not match raw bars")
    if "symbol" in features and not features["symbol"].astype(str).eq(symbols.iloc[0]).all():
        raise ValueError("strategy label feature symbol does not match raw bars")
    records = [{"date": day, "symbol": symbols.iloc[0], "label": np.nan, "label_end": None,
                "label_reason": "calendar_unverified", "net_return": np.nan,
                "invested_net_return": np.nan, "label_available": False,
                "entry_date": None, "exit_date": None, "entry_shares": 0,
                "invested_cash": np.nan, "net_pnl": np.nan, "fees": np.nan,
                "entry_plan_id": None, "entry_policy": entry_policy} for day in dates]
    result = pd.DataFrame(records, index=features.index)
    if market_dates is None:
        return result
    calendar = [pd.Timestamp(day).date().isoformat() for day in market_dates]
    if calendar != sorted(set(calendar)) or not calendar:
        raise ValueError("market_dates must be a verified ascending unique calendar")
    settings = strategy_config(config)
    if decisions is None:
        from my_strategy.services.czsc_research_decisions import replay_decisions
        replay_raw = raw.copy()
        replay_raw["dt"] = pd.to_datetime(replay_raw["date"]) + pd.Timedelta(hours=15)
        replay_raw["id"] = np.arange(len(replay_raw))
        replay = replay_decisions(features, replay_raw, entry_policy=entry_policy,
                                  entry_parameters=entry_parameters, config=settings)
        decisions, entry_parameters = replay["decisions"], replay["entry_parameters"]
    if len(decisions) != len(raw) or dates != [row["date"] for row in decisions]:
        raise ValueError("strategy label decisions do not match raw prefix")
    execution = execute_decisions(str(symbols.iloc[0]), raw, decisions, dates[0], initial_cash,
                                  settings, "strategy-label-ledger", market_dates=calendar,
                                  verified_sources=VERIFIED_SOURCES, entry_policy=entry_policy)
    lookup = {day: index for index, day in enumerate(dates)}
    next_market = dict(zip(calendar[:-1], calendar[1:]))
    for index, (feature, decision) in enumerate(zip(features.to_dict("records"), decisions, strict=True)):
        record = records[index]
        plan = decision.get("entry_plan") or {}
        record["entry_plan_id"] = plan.get("plan_id")
        if not bool(feature.get("input_eligible", True)):
            record["label_reason"] = "input_ineligible"
        elif plan.get("status") == "rejected":
            record["label_reason"] = "entry_plan_rejected:" + ",".join(plan.get("reason_codes", []))
        elif plan.get("status") == "active" and plan.get("entry_allowed"):
            record["label_reason"] = "entry_not_executed"
            record["entry_date"] = next_market.get(dates[index])
            if record["entry_date"] is None or record["entry_date"] > dates[-1]:
                record["label_reason"] = "entry_immature_calendar_tail"
        elif bool(feature.get("rule_buy", False)):
            record["label_reason"] = "no_new_entry_plan" if entry_policy != "legacy" else "entry_not_executed"
        else:
            record["label_reason"] = "not_entry_candidate"
    for rejection in execution["rejections"]:
        if rejection["action"] == "BUY" and rejection["signal_date"] in lookup:
            record = records[lookup[rejection["signal_date"]]]
            record["label_reason"] = "entry_" + rejection["reason"]
            record["entry_date"] = next_market.get(rejection["signal_date"])
    inventory, paid, received, fees, opened = 0, 0.0, 0.0, 0.0, None
    for entry in execution["ledger"]:
        if entry["action"] == "BUY":
            if inventory:
                raise ValueError("strategy labels do not support adding to an open position")
            opened = records[lookup[entry["signal_date"]]]
            inventory = int(entry["shares"])
            paid, received, fees = -float(entry["cash_flow"]), 0.0, float(entry["fee"])
            opened.update(entry_date=entry["date"], entry_shares=inventory, invested_cash=paid,
                          label_reason="position_unclosed")
        else:
            if opened is None:
                raise ValueError("strategy label ledger exits without an actual entry")
            inventory -= int(entry["shares"])
            received += float(entry["cash_flow"])
            fees += float(entry["fee"])
            if inventory == 0:
                pnl = received - paid
                opened.update(label=float(pnl > 0), label_end=entry["date"], exit_date=entry["date"],
                    net_return=pnl / initial_cash, invested_net_return=received / paid - 1,
                    net_pnl=pnl, fees=fees, label_available=True, label_reason="available")
                opened = None
    result = pd.DataFrame(records, index=features.index)
    calendar_hash = hashlib.sha256(json.dumps(calendar, separators=(",", ":")).encode()).hexdigest()
    result.attrs.update(label_version=LABEL_VERSION, entry_policy=entry_policy, initial_cash=initial_cash,
        label_contract={"label_version": LABEL_VERSION, "entry_policy": entry_policy,
            "entry_parameters": entry_parameters, "initial_cash": initial_cash,
            "calendar_sha256": calendar_hash, "strategy": settings,
            "maturity": "full_actual_exit_date", "net_return_denominator": "initial_cash",
            "invested_return_denominator": "actual_buy_cash_debit_including_fee"},
        execution=execution, unavailable_counts=result.loc[~result["label_available"], "label_reason"].value_counts().to_dict(),
        selection_boundary="All contemporaneous candidates retained. Unfilled/unclosed outcomes stay unavailable; account evaluation retains all accounts.")
    return result
