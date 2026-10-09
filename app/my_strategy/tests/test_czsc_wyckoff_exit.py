"""Independent Wyckoff targets use actual state and guarded Broker executions."""
from __future__ import annotations

import copy
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services.czsc_wyckoff_exit import (
    EXIT_FEATURE_COLUMNS, EXIT_FEATURE_SCHEMA_HASH, EXIT_LABEL_VERSION,
    POLICY_FEATURE_COLUMNS, POLICY_LABEL_VERSION,
    ExitPredictor, PolicyPredictor, _bind, bind_existing_exit_dataset,
    build_exit_dataset, build_policy_dataset, exit_contract, holding_feature_row,
    make_entry_policy, make_exit_policy, make_union_decisions, policy_contract,
    make_runtime_exit_policy, make_runtime_entry_policy,
    train_exit_model, train_policy_model,
)


@pytest.fixture
def sample():
    dates = pd.bdate_range("2024-01-02", periods=110)
    days = dates.strftime("%Y-%m-%d").tolist()
    raw = pd.DataFrame({"date": dates, "dt": dates + pd.Timedelta(hours=15), "id": np.arange(len(dates)),
        "symbol": "000001.SZ", "open": 10., "high": 10.2, "low": 9.8, "close": 10.,
        "volume": 1000000., "amount": 10000000., "source": "tushare", "has_trade_price": 1})
    raw.attrs["data_version"] = "frozen-fixture"
    decisions = [{"date": day, "eligible": True, "target_weight": .95 if 70 <= i < 80 else 0.,
        "action": "BUY" if i == 70 else "SELL" if i == 80 else "HOLD",
        "reference_price": 10.} for i, day in enumerate(days)]
    plan = {"plan_id": "candidate-70", "symbol": "000001.SZ", "signal_date": days[70],
        "available_at": days[70] + "T15:00:00+08:00", "policy": "fresh", "status": "active",
        "entry_allowed": True, "valid_for_sessions": 1, "reference_price": 10., "target_weight": .95,
        "price_floor": None, "price_ceiling": None, "stop_price": None, "risk_fraction": None}
    decisions[70]["entry_plan"] = plan
    for i in range(71, 80):
        decisions[i]["entry_plan"] = {**plan, "status": "holding", "entry_allowed": False}
    values = {key: .1 for key in {*EXIT_FEATURE_COLUMNS, *POLICY_FEATURE_COLUMNS}}
    features = pd.DataFrame(values, index=raw.index)
    features["date"], features["symbol"] = dates, raw.symbol
    features["input_eligible"] = features["model_input_eligible"] = features["wyckoff_input_eligible"] = True
    features["wyckoff_available_at"] = [day + "T15:00:00+08:00" for day in days]
    features["wyckoff_rule_buy"] = features["wyckoff_rule_sell"] = False
    features["wyckoff_stop"] = np.nan
    return raw, features, decisions, days


def account(sample, **kwargs):
    raw, _, decisions, days = sample
    return execute_decisions("000001.SZ", raw, decisions, days[0], 100000., strategy_config(), "fixture",
        market_dates=days, verified_sources=VERIFIED_SOURCES, entry_policy="fresh", **kwargs)


def exit_data(sample, **kwargs):
    raw, features, decisions, days = sample
    return build_exit_dataset(features=features, raw=raw, base_decisions=decisions, market_dates=days,
        start=days[0], initial_cash=100000., config=strategy_config(), **kwargs)


def policy_data(sample, **kwargs):
    raw, features, decisions, days = sample
    return build_policy_dataset(features=features, raw=raw, candidate_decisions=decisions,
        market_dates=days, start=days[0], initial_cash=100000., config=strategy_config(), **kwargs)


def entry_response(snapshot, *, apply=True, allowed=False, probability=.1):
    return {"head": "policy_entry", "contract_hash": "actual-policy-fixture", "probability": probability,
        "request_entry": allowed, "apply": apply, "research_arm": True}


def test_optional_entry_gate_none_and_shadow_preserve_every_prior_result(sample):
    original = account(sample)
    shadow = account(sample, entry_gate=lambda s: entry_response(s, apply=False))
    for key in original:
        assert shadow[key] == original[key]
    assert len(shadow["entry_gate_snapshots"]) == 1
    assert shadow["entry_gate_snapshots"][0]["date"] == sample[3][70]
    assert shadow["entry_gate_snapshots"][0]["cash"] == 100000.
    assert not shadow["entry_gate_diagnostics"][0]["applied"]


def test_entry_veto_consumes_one_plan_and_does_not_create_holding_or_retry(sample):
    result = account(sample, entry_gate=entry_response, exit_policy=lambda snapshot: None)
    assert not result["ledger"] and not result["holding_snapshots"]
    assert result["consumed_plan_ids"] == ["candidate-70"]
    assert result["rejections"][0]["date"] == sample[3][71]
    assert result["rejections"][0]["reason"] == "policy_entry_gate_veto"
    assert len(result["entry_gate_snapshots"]) == 1
    assert result["metrics"]["final_equity"] == 100000.


def test_entry_gate_uses_live_cash_of_its_own_continuous_ledger(sample):
    raw, _, decisions, days = sample
    raw.loc[81, "open"] = 10.1
    raw.loc[81, "high"] = 10.3
    plan = copy.deepcopy(decisions[70]["entry_plan"])
    plan.update(plan_id="candidate-90", signal_date=days[90], available_at=days[90] + "T15:00:00+08:00")
    for i in range(90, 96):
        decisions[i]["target_weight"] = .95
    decisions[90]["entry_plan"] = plan
    seen = []
    result = account(sample, entry_gate=lambda s: seen.append(s) or entry_response(s, allowed=True, probability=.9))
    assert len(seen) == 2 and seen[1]["date"] == days[90]
    assert seen[1]["ledger_entries_seen"] == 2
    assert seen[1]["cash"] == result["ledger"][1]["cash_after"] != 100000.
    assert result["ledger"][2]["date"] == days[91]


def test_entry_callback_cannot_mutate_cash_prices_or_plan(sample):
    def mutate(snapshot):
        snapshot.update(cash=0, target_weight=0)
        snapshot["entry_plan"]["status"] = "none"
        snapshot["bar"]["open"] = 0
        return entry_response(snapshot, apply=False)
    original, changed = account(sample), account(sample, entry_gate=mutate)
    for key in original:
        assert original[key] == changed[key]


@pytest.mark.parametrize("response,match", [
    ({"head": "holding_exit", "apply": True, "research_arm": True, "contract_hash": "x", "probability": .9, "request_entry": True}, "independent"),
    ({"head": "policy_entry", "apply": True, "research_arm": True, "contract_hash": "x", "request_entry": True}, "independent"),
    ({"probability": 1.1}, "probability"), ("veto", "mapping")])
def test_actual_entry_callback_contract_fails_closed(sample, response, match):
    with pytest.raises(ValueError, match=match):
        account(sample, entry_gate=lambda s: response)


def test_exit_labels_have_own_version_and_real_remaining_net_proceeds(sample):
    sample[0].loc[81, "open"] = 9.9
    refs = []
    frame = exit_data(sample, reference_callback=refs.append)
    row = frame.iloc[0]
    assert frame.attrs["label_version"] == EXIT_LABEL_VERSION
    assert frame.attrs["schema_hash"] == EXIT_FEATURE_SCHEMA_HASH
    assert row.actual_shares == refs[0]["ledger"][0]["shares"]
    assert row.continuation_net_proceeds == refs[0]["ledger"][1]["cash_flow"]
    assert row.forced_exit_date == sample[3][72]
    assert row.continuation_exit_date == row.label_end == sample[3][81]
    assert row.label == float(row.forced_net_proceeds > row.continuation_net_proceeds)
    assert frame.attrs["state_coverage"]["episodes"] == 1


def test_future_exit_prices_cannot_change_historical_w_or_holding_features(sample):
    before = exit_data(sample)
    sample[0].loc[81, "open"] = 10.1
    after = exit_data(sample)
    pd.testing.assert_frame_equal(before[list(EXIT_FEATURE_COLUMNS)], after[list(EXIT_FEATURE_COLUMNS)])
    assert np.all(after.continuation_net_proceeds > before.continuation_net_proceeds)


def test_exit_partial_liquidation_uses_only_remaining_ledger_credits(sample):
    sample[0].loc[80:81, "volume"] = 10000.
    refs = []
    frame = exit_data(sample, reference_callback=refs.append)
    state = frame.loc[frame.date == sample[3][81]].iloc[0]
    assert state.actual_shares == refs[0]["ledger"][1]["position_after"]
    assert state.continuation_net_proceeds == sum(row["cash_flow"] for row in refs[0]["ledger"][2:])
    assert state.label_end == refs[0]["ledger"][-1]["date"]


def test_missing_w_is_unavailable_but_optional_no_event_age_remains_eligible(sample):
    frame = exit_data(sample)
    assert frame.input_eligible.all()
    sample[1].loc[71, "w_event_age"] = np.nan
    assert exit_data(sample).iloc[0].input_eligible
    sample[1].loc[71, "w_relative_volume20"] = np.nan
    assert not exit_data(sample).iloc[0].input_eligible
    sample[1].loc[72, "wyckoff_available_at"] = sample[3][73] + "T15:00:00+08:00"
    assert not exit_data(sample).iloc[1].input_eligible


def test_final_or_future_behavior_states_cannot_train_exit_or_policy(sample):
    sample[2][70]["behavior_model_lineage"] = {"available_at": sample[3][71], "label_cutoff": sample[3][60]}
    with pytest.raises(ValueError, match="available"):
        exit_data(sample)
    sample[2][70]["behavior_model_lineage"] = {"available_at": sample[3][60], "label_cutoff": sample[3][70]}
    with pytest.raises(ValueError, match="overlap"):
        policy_data(sample)


def test_policy_labels_all_candidates_before_selection_clone_real_cash_and_fees(sample):
    _, _, decisions, _ = sample
    behavior = copy.deepcopy(decisions)
    for row in behavior:
        row.update(target_weight=0., entry_plan=None)
    clones = []
    frame = policy_data(sample, behavior_decisions=behavior, candidate_callback=lambda row, clone: clones.append(clone))
    assert len(frame) == frame.attrs["candidate_counts"]["all_frozen_candidates"] == 1
    row = frame.iloc[0]
    assert row.behavior_cash == row.clone_initial_cash == 100000.
    assert row.entry_date == sample[3][71] and row.label_end == sample[3][81]
    assert row.net_return == pytest.approx(sum(entry["cash_flow"] for entry in clones[0]["ledger"]) / row.behavior_cash)
    assert row.fees == sum(entry["fee"] for entry in clones[0]["ledger"])
    assert row.label == 0. and row.label_available
    assert frame.attrs["label_version"] == POLICY_LABEL_VERSION


def test_policy_cash_clone_keeps_original_capital_for_frozen_exit_callback(sample):
    raw, _, decisions, days = sample
    raw.loc[81, "open"] = 10.1
    raw.loc[81, "high"] = 10.3
    plan = copy.deepcopy(decisions[70]["entry_plan"])
    plan.update(plan_id="candidate-90", signal_date=days[90], available_at=days[90] + "T15:00:00+08:00")
    decisions[90]["entry_plan"] = plan
    for i in range(90, 96):
        decisions[i]["target_weight"] = .95
    seen = []
    frame = policy_data(sample, frozen_exit_policy=lambda snapshot: seen.append(snapshot) or None)
    second = frame.iloc[1]
    assert second.behavior_cash != 100000. and second.clone_initial_cash == second.behavior_cash
    assert all(snapshot["initial_cash"] == 100000. for snapshot in seen)
    assert second.candidate_cash_ratio == pytest.approx(second.behavior_cash / 100000.)


def test_policy_rejected_and_unclosed_candidates_remain_nan_not_losses(sample):
    sample[0].loc[71, "open"] = 10.6
    frame = policy_data(sample)
    assert len(frame) == 1 and not frame.iloc[0].label_available and pd.isna(frame.iloc[0].label)
    assert frame.iloc[0].label_reason == "entry_rejected:conservative_upper_price_guard"
    sample[0].loc[71, "open"] = 10.
    raw, features, decisions, days = sample
    frame = policy_data((raw.iloc[:80], features.iloc[:80], decisions[:80], days))
    assert frame.iloc[0].label_reason == "episode_unclosed"
    assert pd.isna(frame.iloc[0].label) and pd.isna(frame.iloc[0].label_end)


def test_policy_existing_actual_holdings_are_retained_as_not_decidable(sample):
    raw, features, decisions, days = sample
    second = copy.deepcopy(decisions[70]["entry_plan"])
    second.update(plan_id="candidate-75", signal_date=days[75], available_at=days[75] + "T15:00:00+08:00")
    decisions[75]["entry_plan"] = second
    frame = policy_data(sample)
    assert len(frame) == 2
    row = frame.iloc[1]
    assert row.behavior_shares > 0 and row.label_reason == "candidate_not_decidable_existing_position"
    assert not row.label_available and pd.isna(row.label)


def test_source_behavior_and_candidate_contracts_are_distinct(sample):
    cfg, days = strategy_config(), sample[3]
    first = exit_contract(cfg, days)
    assert first["contract_hash"] != exit_contract(cfg, days, candidate_policy="union_v1")["contract_hash"]
    assert first["contract_hash"] != exit_contract(cfg, days, behavior_policy="forward_three_fusion")["contract_hash"]
    assert first["contract_hash"] != exit_contract(cfg, days, source_snapshot_hash="new-source")["contract_hash"]
    assert first["contract_hash"] != policy_contract(cfg, days)["contract_hash"]


def test_reused_actual_states_preserve_both_paths_and_missing_w_is_not_invented(sample):
    old = exit_data(sample).drop(columns=[key for key in sample[1] if key.startswith("w_")])
    source = sample[1].copy()
    source = source.loc[source.index != 72]
    bound = bind_existing_exit_dataset(old, source, config=strategy_config(), market_dates=sample[3],
        source_dataset_sha256="a" * 64, reference_run_id="immutable-audited-reference")
    assert len(bound) == len(old)
    pd.testing.assert_series_equal(old.forced_net_proceeds, bound.forced_net_proceeds)
    pd.testing.assert_series_equal(old.continuation_net_proceeds, bound.continuation_net_proceeds)
    assert not bound.loc[bound.date == sample[3][72], "input_eligible"].iloc[0]
    assert bound.attrs["source_state_dataset_sha256"] == "a" * 64


def test_union_new_event_candidate_freezes_known_support_and_risk_plan(sample):
    _, features, decisions, days = sample
    for row in decisions:
        row.update(target_weight=0., entry_plan=None, action="WAIT")
    features.loc[70, ["wyckoff_rule_buy", "wyckoff_stop"]] = [True, 9.8]
    features.loc[75, "wyckoff_rule_sell"] = True
    union = make_union_decisions(features, decisions, entry_policy="risk")
    plan = union[70]["entry_plan"]
    assert union[70]["candidate_origin"] == "wyckoff_test_lps"
    assert plan["signal_date"] == days[70] and plan["stop_price"] < 9.8 < plan["reference_price"]
    assert plan["price_floor"] > plan["stop_price"] and plan["price_ceiling"] > plan["reference_price"]
    assert union[74]["target_weight"] == plan["target_weight"]
    assert union[75]["target_weight"] == 0. and union[75]["union_exit_reason"] == "wyckoff_supply"


def test_union_prestart_events_cannot_seed_an_unfunded_theoretical_holding(sample):
    _, features, decisions, days = sample
    for row in decisions:
        row.update(target_weight=0., entry_plan=None, action="WAIT", position_start=days[70])
    features.loc[65, ["wyckoff_rule_buy", "wyckoff_stop"]] = [True, 9.8]
    features.loc[70, ["wyckoff_rule_buy", "wyckoff_stop"]] = [True, 9.8]
    union = make_union_decisions(features, decisions, entry_policy="risk")
    assert all(row["target_weight"] == 0. for row in union[:70])
    assert union[70]["action"] == "BUY" and union[70]["entry_plan"]["signal_date"] == days[70]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    days = pd.bdate_range("2020-01-02", periods=300).strftime("%Y-%m-%d").tolist()
    index = np.arange(len(days))
    root = tmp_path_factory.mktemp("wyckoff-independent-heads")
    trained = {}
    for kind, columns, contract, train in (
        ("exit", EXIT_FEATURE_COLUMNS, exit_contract(strategy_config(), days), train_exit_model),
        ("policy", POLICY_FEATURE_COLUMNS, policy_contract(strategy_config(), days), train_policy_model)):
        values = {key: np.sin(index / (i % 5 + 1)) for i, key in enumerate(columns)}
        frame = _bind(pd.DataFrame({**values, "date": days, "symbol": "000001.SZ", "input_eligible": True,
            "label": (index % 2).astype(float), "label_end": pd.bdate_range("2020-01-08", periods=300).strftime("%Y-%m-%d"),
            "label_available": True}), contract, head="holding_exit" if kind == "exit" else "policy_entry")
        directory = root / kind
        manifest = train(frame, directory, days[149], days[160], days[209], device="cpu", epochs=1,
            batch_size=32, min_train_rows=30, min_validation_rows=10)
        trained[kind] = frame, directory, manifest
    return trained, days


@pytest.mark.parametrize("kind,predictor", [("exit", ExitPredictor), ("policy", PolicyPredictor)])
def test_real_heads_have_separate_targets_maturity_and_cpu_export_matches_torch(trained, kind, predictor):
    from my_strategy.services.czsc_research_ml import PredictorSession
    groups, days = trained
    frame, directory, manifest = groups[kind]
    model = predictor(directory, device="mps")
    values = model.predict(frame.iloc[210:230], as_of=days[229])
    baseline = PredictorSession(device="cpu").predict(frame.iloc[210:230], directory, as_of=days[229])
    np.testing.assert_allclose(values, baseline, rtol=0, atol=1.2e-7)
    assert manifest["train_label_end"] <= days[149] and manifest["validation_label_end"] <= days[209]
    assert manifest["training_batches"] > 0
    assert model.diagnostics["device"] == "cpu" and not model.diagnostics["actual_gpu_inference"]
    with pytest.raises(ValueError, match="unavailable"):
        model.predict(frame.iloc[150:151], as_of=days[150])


def test_independent_heads_cannot_swap_weights_or_targets(trained):
    groups, _ = trained
    with pytest.raises(ValueError, match="target contract"):
        ExitPredictor(groups["policy"][1])
    with pytest.raises(ValueError, match="target contract"):
        PolicyPredictor(groups["exit"][1])


@pytest.mark.parametrize("kind,predictor", [("exit", ExitPredictor), ("policy", PolicyPredictor)])
def test_daily_cpu_vector_matches_batch_and_torch_after_optional_imputation(trained, kind, predictor):
    from my_strategy.services.czsc_research_ml import PredictorSession
    groups, days = trained
    frame, directory, _ = groups[kind]
    selected = frame.iloc[210:230].copy()
    selected.loc[selected.index[0], predictor.columns[-1]] = np.nan
    daily_model = predictor(directory)
    daily = np.array([daily_model.predict_one(row, as_of=row["date"]) for row in selected.to_dict("records")])
    batch = predictor(directory).predict(selected, as_of=days[229])
    torch = PredictorSession(device="cpu").predict(selected, directory, as_of=days[229])
    np.testing.assert_allclose(daily, batch, rtol=0, atol=1.2e-7)
    np.testing.assert_allclose(daily, torch, rtol=0, atol=1.2e-7)
    assert daily_model.diagnostics["rows"] == daily_model.diagnostics["batches"] == 20


def test_cpu_export_worker_does_not_import_torch(trained):
    groups, days = trained
    directory = groups["exit"][1]
    program = "from my_strategy.services.czsc_wyckoff_exit import ExitPredictor; import sys; p=ExitPredictor(sys.argv[1]); assert 'torch' not in sys.modules; print(p.diagnostics['device'])"
    result = subprocess.run([sys.executable, "-c", program, str(directory)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "cpu"


def test_exit_and_entry_shadow_callbacks_preserve_existing_ledger(sample):
    class ExitFake:
        contract = exit_contract(strategy_config(), sample[3])
        threshold = .55
        manifest = {"available_at": sample[3][60]}
        def predict_one(self, values, as_of):
            return .99
    class PolicyFake:
        contract = policy_contract(strategy_config(), sample[3])
        threshold = .55
        manifest = {"available_at": sample[3][60]}
        def predict_one(self, values, as_of):
            return .01
    original = account(sample)
    shadow = account(sample, exit_policy=make_exit_policy(features=sample[1], predictor=ExitFake()),
        entry_gate=make_entry_policy(features=sample[1], predictor=PolicyFake()))
    for key in original:
        assert shadow[key] == original[key]
    applied = account(sample, entry_gate=make_entry_policy(features=sample[1], predictor=PolicyFake(), apply_entry=True))
    assert not applied["ledger"]


def test_missing_contract_or_future_checkpoint_callbacks_fall_back_without_scoring(sample):
    class Fake:
        contract = exit_contract(strategy_config(), sample[3], entry_policy="risk")
        threshold = .55
        manifest = {"available_at": sample[3][0]}
        def predict_one(self, values, as_of):
            raise AssertionError("mismatched exit contract must not score")
    result = account(sample, exit_policy=make_exit_policy(features=sample[1], predictor=Fake(), apply_exit=True))
    assert result["ledger"] == account(sample)["ledger"]
    assert result["exit_policy_diagnostics"][0]["status"] == "wyckoff_exit_policy_contract_unavailable"


def test_runtime_explicit_dual_fallback_dispatches_original_target_and_preserves_ledger(sample, monkeypatch):
    from types import SimpleNamespace
    from my_strategy.services.czsc_dual_exit import exit_contract as old_contract
    old_target = old_contract(strategy_config(), sample[3])
    new_target = exit_contract(strategy_config(), sample[3])
    class FakeDual:
        contract, threshold, manifest = old_target, .55, {"available_at": sample[3][60]}
        def __init__(self, binding, device=None, expected_contract=None):
            assert binding["model_dir"] == "old-dual-directory"
            assert expected_contract == old_target
        def predict_one(self, values, as_of):
            return .9
    class FakeW:
        contract, threshold, manifest = new_target, .55, {"available_at": sample[3][60]}
        def __init__(self, binding, device=None):
            assert binding["model_dir"] == "new-w-directory"
        def predict_one(self, values, as_of):
            return .7
    monkeypatch.setattr("my_strategy.services.czsc_dual_exit.ExitPredictor", FakeDual)
    monkeypatch.setattr("my_strategy.services.czsc_wyckoff_exit.ExitPredictor", FakeW)
    class Resolver:
        def resolve(self, day):
            family = "dual" if day <= sample[3][74] else "wyckoff"
            directory = "old-dual-directory" if family == "dual" else "new-w-directory"
            return {"model_dir": "bundle", "bundle_family": family,
                "exit_model": {"model_dir": directory, "entry_policy": "fresh"}}
    runtime = SimpleNamespace(entry_policy="fresh", market_dates=sample[3], resolver=Resolver())
    sample[1].loc[71, "w_relative_volume20"] = np.nan
    shadow = account(sample, exit_policy=make_runtime_exit_policy(runtime, sample[1]))
    assert shadow["ledger"] == account(sample)["ledger"]
    diagnostics = shadow["exit_policy_diagnostics"]
    assert diagnostics[0]["status"] == "exit_shadow" and diagnostics[0]["contract_hash"] == old_target["contract_hash"]
    assert diagnostics[-1]["status"] == "wyckoff_exit_shadow" and diagnostics[-1]["contract_hash"] == new_target["contract_hash"]
    assert not any(row["applied"] for row in diagnostics)
    assert set(runtime.exit_predictors) == {"dual:old-dual-directory", "new-w-directory"}


def test_runtime_actual_entry_is_hypothetical_frozen_exit_probability_and_shadow(sample, monkeypatch):
    from types import SimpleNamespace
    target = policy_contract(strategy_config(), sample[3], continuation_policy_hash="sealed-forward-exit-router")
    class Fake:
        contract, threshold, manifest = target, .55, {"available_at": sample[3][60]}
        def __init__(self, binding, device=None):
            assert binding["model_dir"] == "new-policy-directory"
            self.rows = 0
        def predict_one(self, values, as_of):
            assert values["candidate_cash_ratio"] == 1.
            self.rows += 1
            return .97
    monkeypatch.setattr("my_strategy.services.czsc_wyckoff_exit.PolicyPredictor", Fake)
    route = {"model_dir": "bundle", "bundle_family": "wyckoff", "source_snapshot_hash": target["source_snapshot_hash"],
        "policy_models": {"fresh": {"model_dir": "new-policy-directory"}}}
    runtime = SimpleNamespace(entry_policy="fresh", market_dates=sample[3],
        resolver=SimpleNamespace(resolve=lambda day: route))
    shadow = account(sample, entry_gate=make_runtime_entry_policy(runtime, sample[1]))
    for key, value in account(sample).items():
        assert shadow[key] == value
    response = shadow["entry_gate_diagnostics"][0]
    assert response["probability"] == .97
    assert response["target_scope"] == "hypothetical_frozen_policy_roundtrip"
    assert not response["execution_policy_matches_target"] and not response["applied"]
    assert not response["entry_policy_applied"] and not response["exit_policy_applied"]
    assert runtime.policy_predictors["new-policy-directory"].rows == 1


def test_runtime_missing_policy_head_or_typed_dual_fallback_does_not_fake_probability(sample):
    from types import SimpleNamespace
    runtime = SimpleNamespace(entry_policy="fresh", market_dates=sample[3],
        resolver=SimpleNamespace(resolve=lambda day: {"bundle_family": "dual", "model_dir": "old-bundle"}))
    shadow = account(sample, entry_gate=make_runtime_entry_policy(runtime, sample[1]))
    response = shadow["entry_gate_diagnostics"][0]
    assert response["probability"] is None and not response["applied"]
    assert response["status"] == "wyckoff_policy_dual_fallback_no_policy_head"
    assert runtime.policy_predictors == {}
    assert shadow["ledger"] == account(sample)["ledger"]


def test_runtime_policy_different_capital_scale_is_explicitly_unavailable(sample):
    from types import SimpleNamespace
    runtime = SimpleNamespace(entry_policy="fresh", market_dates=sample[3],
        resolver=SimpleNamespace(resolve=lambda day: (_ for _ in ()).throw(AssertionError("unsupported capital must not resolve/infer"))))
    raw, features, decisions, days = sample
    result = execute_decisions("000001.SZ", raw, decisions, days[0], 200000., strategy_config(), "fixture",
        market_dates=days, verified_sources=VERIFIED_SOURCES, entry_policy="fresh",
        entry_gate=make_runtime_entry_policy(runtime, features))
    response = result["entry_gate_diagnostics"][0]
    assert response["probability"] is None and not response["applied"]
    assert response["target_initial_cash"] == 100000. and response["actual_account_initial_cash"] == 200000.
    assert response["status"] == "wyckoff_policy_initial_capital_contract_unavailable"
