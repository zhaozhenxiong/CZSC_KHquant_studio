"""Independent audit catches leakage, censored negatives and invented cash."""
import copy
import zipfile

import numpy as np
import pandas as pd

from my_strategy.scripts.audit_czsc_wyckoff_research import (
    check_oof, check_policy_rows, check_linear_export_storage, check_market_inputs, MARKET_INPUT_COLUMNS,
)


def forward_rows():
    return pd.DataFrame({"date": ["2025-01-02", "2025-01-03"], "symbol": ["000001.SZ"] * 2,
        "label": [1., 0.], "label_end": ["2025-01-17", "2025-01-20"],
        "target_label_end": ["2025-01-17", "2025-01-20"], "oof_fold": ["2025Q1"] * 2,
        "expert_train_end": ["2024-09-30"] * 2, "expert_validation_end": ["2024-12-31"] * 2,
        "expert_available_at": ["2024-12-31"] * 2,
        "p_ma": [.4, .6], "p_structure": [.5, .7], "p_wyckoff": [.65, .25]})


def policy_rows():
    return pd.DataFrame({"date": ["2025-01-02", "2025-02-03"], "symbol": ["000001.SZ"] * 2,
        "candidate_id": ["plan1", "plan2"], "behavior_cash": [100000., 98230.], "behavior_shares": [0, 0],
        "clone_initial_cash": [100000., 98230.], "clone_ledger_hash": ["a", "b"],
        "label": [1., None], "label_available": [True, False], "label_end": ["2025-01-20", None],
        "net_return": [.03, None], "entry_date": ["2025-01-03", None], "exit_date": ["2025-01-20", None],
        "label_reason": ["available", "entry_rejected:gap"], "entry_shares": [3000, 0],
        "state_observed_at": ["2025-01-02T15:00:00+08:00", "2025-02-03T15:00:00+08:00"],
        "candidate_cash_ratio": [1., .9823], "candidate_account_return": [0., -.0177]})


def test_real_forward_mature_oof_and_cash_contract_pass():
    assert check_oof(forward_rows(), "2025-03-31") == []
    assert check_policy_rows(policy_rows()) == []


def test_in_sample_same_day_availability_and_immature_fusion_rejected():
    f = forward_rows(); f.loc[0, "expert_available_at"] = f.loc[0, "date"]
    f.loc[0, "expert_validation_end"] = f.loc[0, "date"]
    assert any(e["rule"] == "strict_forward_oof_mature_target" for e in check_oof(f, "2025-03-31"))
    assert any(e["rule"] == "strict_forward_oof_mature_target" for e in check_oof(forward_rows(), "2025-01-10"))


def test_missing_third_probability_is_not_zero_or_fake_evidence():
    f = forward_rows(); f.loc[0, "p_wyckoff"] = None
    assert any(e["rule"] == "oof_real_three_probabilities_same_binary_target" for e in check_oof(f, "2025-03-31"))


def test_rejected_and_unclosed_candidates_cannot_be_negative_labels():
    f = policy_rows(); f.loc[1, "label"] = 0.
    assert any(e["rule"] == "policy_rejected_censored_not_negative" for e in check_policy_rows(f))


def test_counterfactual_cash_cannot_reset_after_a_loss():
    f = policy_rows(); f.loc[1, "clone_initial_cash"] = 100000.
    assert any(e["rule"] == "policy_clone_exact_observed_cash" for e in check_policy_rows(f))


def test_existing_inventory_and_same_day_trade_fail_actual_policy_contract():
    f = policy_rows(); f.loc[0, "behavior_shares"] = 100
    assert any(e["rule"] == "policy_flat_actual_complete_fee_roundtrip_target" for e in check_policy_rows(f))
    f = policy_rows(); f.loc[0, "entry_date"] = f.loc[0, "date"]
    assert any(e["rule"] == "policy_flat_actual_complete_fee_roundtrip_target" for e in check_policy_rows(f))


def test_policy_target_sign_and_maturity_are_independent_of_entry_probability():
    f = policy_rows(); f.loc[0, "net_return"] = -.03
    assert any(e["rule"] == "policy_flat_actual_complete_fee_roundtrip_target" for e in check_policy_rows(f))


def test_self_consistent_json_export_cannot_replace_actual_trained_weights(tmp_path):
    checkpoint = tmp_path / "model.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("model/byteorder", "little")
        archive.writestr("model/data/0", np.asarray([.3, -.2, .1], dtype="<f4").tobytes())
        archive.writestr("model/data/1", np.asarray([.05], dtype="<f4").tobytes())
    export = {"columns": ["a", "b", "c"], "weight": [.3, -.2, .1], "bias": .05}
    assert check_linear_export_storage(checkpoint, export) == []
    export["weight"][0] = .4
    assert any(error["rule"] == "cpu_export_exact_original_linear_checkpoint_float_storage" for error in check_linear_export_storage(checkpoint, export))
    f = policy_rows(); f.loc[0, "label_end"] = "2025-01-03"
    assert any(e["rule"] == "policy_flat_actual_complete_fee_roundtrip_target" for e in check_policy_rows(f))


def test_exit_and_policy_cannot_use_later_or_other_symbol_market_inputs():
    source = pd.DataFrame({key: [1., 2.] for key in MARKET_INPUT_COLUMNS})
    source["date"] = ["2025-01-02", "2025-01-03"]
    source["symbol"] = "000001.SZ"
    samples = source.iloc[[0, 0]].copy()
    assert check_market_inputs(samples, source) == []  # two causal behavior paths may share one observation
    samples.iloc[0, samples.columns.get_loc("w_sos")] = 2.
    assert any(error["rule"] == "all_74_market_inputs_exact_frozen_observation" for error in check_market_inputs(samples, source))
    samples = source.iloc[[0]].copy(); samples["symbol"] = "600519.SH"
    assert any(error["rule"] == "market_input_source_identity" for error in check_market_inputs(samples, source))
