"""Cost pressure must replay fills and account state with frozen decisions."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from my_strategy.scripts.wyckoff_cost_stress import (
    COST_FIELDS, _aggregate, stressed_strategy, verify_base_account,
)
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions, _write_frame
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES


def fixture_account(settings):
    dates = pd.bdate_range("2024-01-02", periods=90)
    days = dates.strftime("%Y-%m-%d").tolist()
    raw = pd.DataFrame({"date": dates, "dt": dates + pd.Timedelta(hours=15),
        "id": np.arange(len(dates)), "symbol": "000001.SZ", "open": 10.,
        "high": 10.2, "low": 9.8, "close": 10., "volume": 1_000_000.,
        "amount": 10_000_000., "source": "tushare", "has_trade_price": 1})
    decisions = [{"date": day, "eligible": True, "target_weight": 1. if 70 <= index < 80 else 0.,
        "action": "BUY" if index == 70 else "SELL" if index == 80 else "HOLD",
        "reference_price": 10.} for index, day in enumerate(days)]
    plan = {"plan_id": "fixed-candidate", "symbol": "000001.SZ", "signal_date": days[70],
        "available_at": days[70] + "T15:00:00+08:00", "policy": "fresh", "status": "active",
        "entry_allowed": True, "valid_for_sessions": 1, "reference_price": 10., "target_weight": 1.,
        "price_floor": None, "price_ceiling": None, "stop_price": None, "risk_fraction": None}
    decisions[70]["entry_plan"] = plan
    for index in range(71, 80):
        decisions[index]["entry_plan"] = {**plan, "status": "holding", "entry_allowed": False}
    return execute_decisions("000001.SZ", raw, decisions, days[0], 100110., settings, "fixture",
        market_dates=days, verified_sources=VERIFIED_SOURCES, entry_policy="fresh")


def test_cost_multipliers_freeze_every_other_strategy_field():
    settings = strategy_config()
    before = copy.deepcopy(settings)
    stressed = stressed_strategy(settings, 1.5)
    assert settings == before
    for name in COST_FIELDS:
        assert stressed["execution"][name] == settings["execution"][name] * 1.5
        stressed["execution"][name] = settings["execution"][name]
    assert stressed == settings


@pytest.mark.parametrize("factor", [0., .99, -1., np.nan, np.inf])
def test_invalid_cost_pressure_rejected(factor):
    with pytest.raises(ValueError, match="multiplier"):
        stressed_strategy(strategy_config(), factor)


def test_real_pressure_changes_lots_cash_and_cost_stop():
    settings = strategy_config()
    settings["execution"].update(commission=0., stamp_tax=0., slippage=0., min_commission=100., max_weight=1.)
    base = fixture_account(settings)
    pressure = fixture_account(stressed_strategy(settings, 2.))
    assert base["ledger"][0]["shares"] == 10000
    assert pressure["ledger"][0]["shares"] == 9900
    assert base["ledger"][0]["fee"] == 100.
    assert pressure["ledger"][0]["fee"] == 200.
    assert base["ledger"][0]["cash_after"] == 10.
    assert pressure["ledger"][0]["cash_after"] == 910.
    original_fill = next(row for row in base["entry_diagnostics"] if row["status"] == "filled" and row["action"] == "BUY")
    pressure_fill = next(row for row in pressure["entry_diagnostics"] if row["status"] == "filled" and row["action"] == "BUY")
    assert pressure_fill["actual_cost_per_share"] > original_fill["actual_cost_per_share"]
    assert pressure_fill["stop_price"] > original_fill["stop_price"]
    assert pressure_fill["plan_id"] == original_fill["plan_id"] == "fixed-candidate"


def test_base_validation_allows_run_id_only_and_rejects_changed_state(tmp_path):
    account = fixture_account(strategy_config())
    for name in ("ledger", "daily", "rejections"):
        _write_frame(pd.DataFrame(account[name]), tmp_path / (name + ".csv"))
    different_run = copy.deepcopy(account)
    for name in ("ledger", "daily", "rejections"):
        for row in different_run[name]:
            if "run_id" in row:
                row["run_id"] = "another-run"
    verify_base_account(different_run, tmp_path)
    different_run["daily"][0]["equity"] += .01
    with pytest.raises(AssertionError):
        verify_base_account(different_run, tmp_path)


def test_compact_aggregate_keeps_identical_equity_cash_and_trade_statistics():
    account = fixture_account(strategy_config())
    account.update(cpu_rows=13, cpu_batches=11)
    compact = {**account, "daily": [{"date": row["date"], "equity": row["equity"]} for row in account["daily"]]}
    assert _aggregate([account], 100110., 3) == _aggregate([compact], 100110., 3)
