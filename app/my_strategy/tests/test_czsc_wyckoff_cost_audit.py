"""Cost audit rejects policy changes and actual base replay drift."""
import copy

import pandas as pd

from my_strategy.scripts.audit_wyckoff_cost_stress import COST_FIELDS, check_stress_settings, _base_equal


def settings():
    return {"position": {"timeout": 120}, "execution": {"commission": .0003, "stamp_tax": .0005,
        "min_commission": 5., "slippage": .0005, "lot_size": 100, "max_weight": .95}}


def test_stress_all_four_costs_with_frozen_policy_and_unmodified_base():
    base = settings(); previous = copy.deepcopy(base); stressed = copy.deepcopy(base)
    for name in COST_FIELDS: stressed["execution"][name] *= 1.5
    assert check_stress_settings(base, stressed, 1.5) == []
    assert base == previous
    stressed["position"]["timeout"] = 60
    assert check_stress_settings(base, stressed, 1.5)


def test_cannot_change_slippage_alone_or_skip_minimum_commission():
    base = settings(); stressed = copy.deepcopy(base); stressed["execution"]["slippage"] *= 2
    assert check_stress_settings(base, stressed, 2)
    for name in COST_FIELDS: stressed["execution"][name] = base["execution"][name] * 2
    stressed["execution"]["min_commission"] = 5
    assert check_stress_settings(base, stressed, 2)


def test_one_times_audit_ignores_run_identity_but_rejects_changed_cash(tmp_path):
    left, right = tmp_path / "stress", tmp_path / "original"; left.mkdir(); right.mkdir()
    for name in ("ledger", "daily", "rejections"):
        pd.DataFrame({"run_id": ["stress"], "cash": [100000.]}).to_csv(left / (name + ".csv"), index=False)
        pd.DataFrame({"run_id": ["original"], "cash": [100000.]}).to_csv(right / (name + ".csv"), index=False)
    assert _base_equal(left, right) == []
    pd.DataFrame({"run_id": ["stress"], "cash": [99995.]}).to_csv(left / "daily.csv", index=False)
    assert any(error["rule"] == "one_times_exact_original_account" for error in _base_equal(left, right))
