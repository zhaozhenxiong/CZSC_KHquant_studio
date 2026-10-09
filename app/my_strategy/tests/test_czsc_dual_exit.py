"""Actual-held exit targets and opt-in execution share the guarded Broker."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_dual_exit import (
    EXIT_FEATURE_COLUMNS, EXIT_FEATURE_SCHEMA_HASH, EXIT_FEATURE_VERSION, EXIT_LABEL_VERSION,
    ExitPredictor, _bind, build_exit_dataset, evaluate_exit_arm, exit_contract,
    holding_feature_row, make_exit_policy, train_exit_model,
)
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES


@pytest.fixture
def held_input():
    dates = pd.bdate_range("2024-01-02", periods=100)
    raw = pd.DataFrame({"date": dates, "dt": dates + pd.Timedelta(hours=15),
        "id": np.arange(len(dates)), "symbol": "000001.SZ", "open": 10., "high": 10.2,
        "low": 9.8, "close": 10., "volume": 1000000., "amount": 10000000.,
        "source": "tushare", "has_trade_price": 1})
    days = dates.strftime("%Y-%m-%d").tolist()
    plan = {"plan_id": "entry-70", "symbol": "000001.SZ", "signal_date": days[70],
        "available_at": days[70] + "T15:00:00+08:00", "policy": "fresh", "status": "active",
        "entry_allowed": True, "valid_for_sessions": 1, "reference_price": 10.,
        "price_floor": None, "price_ceiling": None, "stop_price": None, "risk_fraction": None}
    decisions = [{"date": day, "eligible": True, "target_weight": .95 if 70 <= i < 80 else 0.,
                  "action": "BUY" if i == 70 else "SELL" if i == 80 else "HOLD"}
                 for i, day in enumerate(days)]
    decisions[70]["entry_plan"] = plan
    for i in range(71, 80):
        decisions[i]["entry_plan"] = {**plan, "status": "holding", "entry_allowed": False}
    features = pd.DataFrame({key: .1 for key in EXIT_FEATURE_COLUMNS}, index=raw.index)
    features["date"], features["symbol"] = dates, raw.symbol
    features["input_eligible"] = features["model_input_eligible"] = True
    return raw, features, decisions, days


def execute(values, policy=None, **kwargs):
    raw, _, decisions, days = values
    return execute_decisions("000001.SZ", raw, decisions, days[0], 100000., strategy_config(),
        "exit-fixture", market_dates=days, verified_sources=VERIFIED_SOURCES,
        entry_policy=kwargs.pop("entry_policy", "fresh"), exit_policy=policy, **kwargs)


def labels(values, **kwargs):
    raw, features, decisions, days = values
    return build_exit_dataset(features=features, raw=raw, base_decisions=decisions,
        market_dates=days, start=days[0], initial_cash=100000., config=strategy_config(), **kwargs)


def request(snapshot, **kwargs):
    return {"head": "holding_exit", "contract_hash": "test-only-contract", "probability": .99,
            "request_exit": True, "apply": True, "research_arm": True, **kwargs}


def test_no_callback_and_shadow_preserve_existing_account(held_input):
    baseline = execute(held_input)
    shadow = execute(held_input, lambda s: request(s, apply=False))
    for key in baseline:
        assert shadow[key] == baseline[key]
    assert all(not row["applied"] for row in shadow["exit_policy_diagnostics"])
    assert "holding_snapshots" not in baseline


def test_shadow_preserves_prior_hard_stop_pending_when_native_exit_follows(held_input):
    held_input[0].loc[72, "open"] = 9.
    held_input[0].loc[72, "close"] = 9.5
    held_input[2][72]["target_weight"] = 0.
    baseline, shadow = execute(held_input), execute(held_input, lambda snapshot: None)
    assert baseline["ledger"][1]["reason"] == "actual_account_stop"
    assert baseline["ledger"][1]["date"] == held_input[3][73]
    assert shadow["ledger"] == baseline["ledger"]
    for key in baseline:
        assert shadow[key] == baseline[key]


def test_live_snapshots_only_after_actual_fill_include_buy_fee_and_prefix(held_input):
    result = execute(held_input, lambda s: None)
    first = result["holding_snapshots"][0]
    buy = result["ledger"][0]
    assert first["date"] == buy["date"] == held_input[3][71]
    assert first["holding_bars"] == 0 and first["ledger_entries_seen"] == 1
    assert first["actual_cost_per_share"] == -buy["cash_flow"] / buy["shares"] > buy["price"]
    assert first["last_ledger_entry"]["date"] <= first["date"]
    assert all(s["shares"] > 0 and s["date"] < held_input[3][81] for s in result["holding_snapshots"])


def test_callback_mutation_cannot_change_broker_state(held_input):
    def mutate(snapshot):
        snapshot.update(shares=0, cash=-100000., stop_price=1000.)
        snapshot["bar"]["close"] = 0.
        return None
    baseline, other = execute(held_input), execute(held_input, mutate)
    assert other["ledger"] == baseline["ledger"] and other["daily"] == baseline["daily"]
    assert other["holding_snapshots"][0]["shares"] > 0


def test_rejected_entry_does_not_create_actual_holding_samples(held_input):
    held_input[0].loc[71, "open"] = 10.6
    result = execute(held_input, request)
    assert result["holding_snapshots"] == [] and result["ledger"] == []
    assert labels(held_input).empty


def test_model_exit_is_next_session_and_T_plus_one(held_input):
    result = execute(held_input, request)
    buy, sale = result["ledger"]
    assert buy["date"] == held_input[3][71] and sale["date"] == held_input[3][72]
    assert sale["signal_date"] == held_input[3][71] and sale["reason"] == "research_exit_model"
    assert sale["date"] > buy["date"]


def test_partial_model_exit_remains_pending_and_signal_tracks_each_attempt(held_input):
    held_input[0].loc[71:73, "volume"] = 10000.
    result = execute(held_input, lambda s: request(s) if s["date"] == held_input[3][71] else None)
    sales = result["ledger"][1:]
    assert [entry["action"] for entry in sales] == ["REDUCE", "REDUCE", "REDUCE", "SELL"]
    assert all(entry["reason"] == "research_exit_model" for entry in sales)
    assert [entry["signal_date"] for entry in sales] == held_input[3][71:75]
    assert {row["model_request_date"] for row in result["entry_diagnostics"] if row.get("action") == "SELL"} == {held_input[3][71]}
    assert result["metrics"]["open_shares"] == 0


def test_model_rejected_sale_retries_without_claiming_same_day_fill(held_input):
    held_input[0].loc[72, "open"] = 9.5
    result = execute(held_input, lambda s: request(s) if s["date"] == held_input[3][71] else None)
    assert result["ledger"][1]["date"] == held_input[3][73]
    assert any(row["date"] == held_input[3][72] and row["reason"] == "conservative_lower_price_guard"
               for row in result["rejections"])


def test_missing_next_market_bar_is_rejection_not_a_substituted_open(held_input):
    raw, features, decisions, days = held_input
    selected = (raw.drop(index=72), features.drop(index=72), decisions[:72] + decisions[73:], days)
    result = execute(selected, request)
    assert any(row["date"] == days[73] and row["reason"] == "missing_next_market_bar" for row in result["rejections"])
    assert result["ledger"][1]["date"] == days[74]


@pytest.mark.parametrize("kind", ["native", "stop"])
def test_hard_exit_overrides_model_request_and_cannot_be_vetoed(held_input, kind):
    if kind == "native":
        held_input[2][71]["target_weight"] = 0.
        expected = "native_target_exit"
    else:
        held_input[0].loc[72, "open"] = 9.
        held_input[0].loc[72, "close"] = 9.5
        expected = "actual_account_stop"
    result = execute(held_input, request)
    assert result["ledger"][1]["reason"] == expected
    if kind == "native":
        assert result["exit_policy_diagnostics"][0]["hard_exit_priority"]
        assert not result["exit_policy_diagnostics"][0]["applied"]
    veto = execute(held_input, lambda s: request(s, probability=.01, request_exit=False))
    assert veto["ledger"][1]["reason"] == expected


@pytest.mark.parametrize("response,match", [
    ({"apply": True, "research_arm": True, "head": "entry", "contract_hash": "x"}, "independent"),
    ({"probability": float("nan")}, "probability"),
    ({"probability": 1.1}, "probability"),
    ("exit", "mapping"),
])
def test_callback_contract_and_probability_fail_closed(held_input, response, match):
    with pytest.raises(ValueError, match=match):
        execute(held_input, lambda s: response)


def test_legacy_cannot_silently_accept_exit_research(held_input):
    with pytest.raises(ValueError, match="fresh or risk"):
        execute(held_input, request, entry_policy="legacy")


def test_labels_compare_real_net_proceeds_and_later_full_exit(held_input):
    references = []
    held_input[0].loc[81, "open"] = 9.9
    dataset = labels(held_input, reference_callback=references.append)
    sample = dataset.iloc[0]
    reference = references[0]
    assert len(dataset) == 10 and sample["actual_shares"] == reference["ledger"][0]["shares"]
    assert sample["actual_cost_per_share"] == -reference["ledger"][0]["cash_flow"] / sample["actual_shares"]
    assert sample["forced_exit_date"] == held_input[3][72]
    assert sample["continuation_exit_date"] == sample["label_end"] == held_input[3][81]
    assert sample["continuation_net_proceeds"] == reference["ledger"][1]["cash_flow"]
    assert sample["label"] == float(sample["forced_net_proceeds"] > sample["continuation_net_proceeds"]) == 1.
    assert sample["exit_advantage"] == pytest.approx((sample["forced_net_proceeds"] - sample["continuation_net_proceeds"]) / 100000)
    assert dataset.attrs["label_version"] == EXIT_LABEL_VERSION
    assert dataset.attrs["label_contract"]["head"] == "holding_exit"
    assert dataset.attrs["feature_columns"] == list(EXIT_FEATURE_COLUMNS)


def test_identical_net_outcomes_are_not_profitable_exit_labels(held_input):
    dataset = labels(held_input)
    assert dataset.label_available.all() and set(dataset.label) == {0.}
    assert np.allclose(dataset.exit_advantage, 0.)


def test_future_changes_affect_outcome_not_historical_features(held_input):
    original = labels(held_input)
    held_input[0].loc[81, "open"] = 10.1
    changed = labels(held_input)
    pd.testing.assert_frame_equal(original[list(EXIT_FEATURE_COLUMNS)], changed[list(EXIT_FEATURE_COLUMNS)])
    assert np.all(changed.continuation_net_proceeds > original.continuation_net_proceeds)


def test_partial_continuation_uses_remaining_shares_and_full_exit_maturity(held_input):
    held_input[0].loc[80:81, "volume"] = 10000.
    references = []
    dataset = labels(held_input, reference_callback=references.append)
    sample = dataset.loc[dataset.date == held_input[3][81]].iloc[0]
    ledger = references[0]["ledger"]
    assert sample.actual_shares == ledger[1]["position_after"] < ledger[0]["shares"]
    assert sample.ledger_entries_seen == 2 and sample.remaining_fraction < 1
    assert sample.continuation_net_proceeds == sum(row["cash_flow"] for row in ledger[2:])
    assert sample.label_end == ledger[-1]["date"] == held_input[3][83]


def test_partial_forced_liquidation_and_rejection_are_retained(held_input):
    held_input[0].loc[71:73, "volume"] = 10000.
    held_input[0].loc[72, "open"] = 9.5
    sample = labels(held_input).iloc[0]
    assert sample.forced_rejection_count == 1
    assert sample.forced_first_rejection == "conservative_lower_price_guard"
    assert sample.forced_exit_date == held_input[3][75]
    assert sample.label_end == held_input[3][81]


def test_unclosed_positions_remain_samples_with_unavailable_targets(held_input):
    raw, features, decisions, days = held_input
    dataset = labels((raw.iloc[:80], features.iloc[:80], decisions[:80], days))
    assert len(dataset) == 9 and not dataset.label_available.any()
    assert set(dataset.label_reason) == {"continuation_unclosed"}
    assert dataset.label.isna().all() and dataset.label_end.isna().all()


def test_source_and_market_eligibility_cannot_be_replaced_by_imputation(held_input):
    result = execute(held_input, lambda s: None)
    snapshot = result["holding_snapshots"][0]
    feature = held_input[1].iloc[71].to_dict()
    assert holding_feature_row(feature, snapshot)["input_eligible"]
    for change in ({"input_eligible": False}, {"model_input_eligible": False}, {"ma5_bias": np.nan}):
        assert not holding_feature_row({**feature, **change}, snapshot)["input_eligible"]
    snapshot["bar"]["source"] = "unverified"
    assert not holding_feature_row(feature, snapshot)["input_eligible"]


@pytest.mark.parametrize("field", ["available_at", "structure_confirmed_at", "zone_confirmed_at", "weekly_available_at", "monthly_available_at"])
def test_future_structure_or_market_visibility_cannot_enter_exit_features(held_input, field):
    snapshot = execute(held_input, lambda s: None)["holding_snapshots"][0]
    feature = held_input[1].iloc[71].to_dict()
    assert not holding_feature_row({**feature, field: held_input[3][72] + "T15:00:00+08:00"}, snapshot)["input_eligible"]
    assert not holding_feature_row({**feature, field: "invalid-timestamp"}, snapshot)["input_eligible"]


def test_alignment_calendar_config_and_target_identity_are_bound(held_input):
    raw, features, decisions, days = held_input
    features.loc[72, "symbol"] = "600519.SH"
    with pytest.raises(ValueError, match="aligned"):
        labels(held_input)
    config = strategy_config()
    first = exit_contract(config, days)
    config["execution"]["commission"] *= 2
    assert first["contract_hash"] != exit_contract(config, days)["contract_hash"]
    assert first["contract_hash"] != exit_contract(strategy_config(), days[:-1])["contract_hash"]
    with pytest.raises(ValueError, match="calendar"):
        exit_contract(config, list(reversed(days)))


def test_evaluation_without_weights_is_explicitly_shadow(held_input):
    raw, features, decisions, days = held_input
    result = evaluate_exit_arm(features=features, raw=raw, base_decisions=decisions,
        market_dates=days, start=days[0], end=days[-1], initial_cash=100000, config=strategy_config())
    assert result["ledger"] == execute(held_input)["ledger"] or [
        {k: v for k, v in r.items() if k != "run_id"} for r in result["ledger"]] == [
        {k: v for k, v in r.items() if k != "run_id"} for r in execute(held_input)["ledger"]]
    assert result["exit_compute_info"]["rows"] == 0 and not result["exit_compute_info"]["actual_mps_inference"]
    assert not result["exit_research_opt_in"]


@pytest.fixture(scope="module")
def trained_exit(tmp_path_factory):
    dates = pd.bdate_range("2020-01-02", periods=300)
    days = dates.strftime("%Y-%m-%d").tolist()
    contract = exit_contract(strategy_config(), days)
    index = np.arange(len(days))
    values = {key: np.sin(index / (i % 5 + 1)) for i, key in enumerate(EXIT_FEATURE_COLUMNS)}
    frame = _bind(pd.DataFrame({**values, "date": days, "symbol": "000001.SZ", "input_eligible": True,
        "label": (index % 2).astype(float), "label_end": pd.bdate_range("2020-01-08", periods=300).strftime("%Y-%m-%d"),
        "label_available": True}), contract)
    directory = tmp_path_factory.mktemp("holding-exit-model")
    manifest = train_exit_model(frame, directory, days[149], days[160], days[209],
        device="cpu", epochs=1, batch_size=32, hidden_sizes=(), min_train_rows=30, min_validation_rows=10)
    return frame, directory, manifest, days


def test_real_exit_training_is_independent_mature_and_records_actual_cpu(trained_exit):
    frame, directory, manifest, days = trained_exit
    assert manifest["label_version"] == EXIT_LABEL_VERSION
    assert manifest["label_contract"]["head"] == "holding_exit"
    assert manifest["train_label_end"] <= days[149]
    assert manifest["validation_label_end"] <= days[209]
    assert manifest["sample_counts"]["train"] < 150 and manifest["sample_counts"]["validation"] < 50
    assert manifest["device"] == "cpu"
    assert not manifest["actual_gpu_training"] and not manifest["actual_mps_training"]
    assert manifest["training_batches"] > 0
    assert (directory / "model.pt").exists()


def test_real_exit_predictor_rejects_future_route_and_records_actual_work(trained_exit):
    frame, directory, manifest, days = trained_exit
    predictor = ExitPredictor(directory, device="cpu", expected_contract=frame.attrs["label_contract"])
    values = predictor.predict(frame.iloc[210:215], as_of=days[214])
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    assert predictor.diagnostics["rows"] == 5 and predictor.diagnostics["batches"] == 1
    assert not predictor.diagnostics["actual_mps_inference"]
    assert predictor.diagnostics["actual_cpu_numpy_inference"]
    from my_strategy.services.czsc_research_ml import PredictorSession
    torch_values = PredictorSession(device="cpu").predict(frame.iloc[210:215], directory, as_of=days[214])
    np.testing.assert_allclose(values, torch_values, rtol=0, atol=1.2e-7)
    with pytest.raises(ValueError, match="unavailable"):
        predictor.predict(frame.iloc[150:151], as_of=days[150])


def test_exit_weights_contract_and_feature_order_are_fail_closed(trained_exit):
    frame, directory, manifest, days = trained_exit
    with pytest.raises(ValueError, match="incompatible"):
        ExitPredictor({"model_dir": directory, "contract_hash": "entry-target"}, device="cpu")
    different = copy.deepcopy(frame.attrs["label_contract"])
    different["target"] = "one_minus_ten_day_entry_probability"
    different["contract_hash"] = stable_hash({k: v for k, v in different.items() if k != "contract_hash"})
    bad = frame.copy()
    bad.attrs["label_contract"] = different
    with pytest.raises(ValueError, match="target contract"):
        train_exit_model(bad, directory / "invalid", days[149], days[160], days[209], device="cpu")
    bad = frame.copy()
    bad.attrs["feature_columns"] = list(reversed(EXIT_FEATURE_COLUMNS))
    with pytest.raises(ValueError, match="order"):
        train_exit_model(bad, directory / "invalid", days[149], days[160], days[209], device="cpu")


def test_independent_exit_policy_never_infers_before_availability(held_input):
    class FakePredictor:
        contract = exit_contract(strategy_config(), held_input[3])
        threshold = .55
        manifest = {"available_at": held_input[3][72]}
        def predict(self, frame, as_of=None):
            assert frame.attrs["label_contract"]["head"] == "holding_exit"
            return np.array([.8])
    shadow = execute(held_input, make_exit_policy(features=held_input[1], predictor=FakePredictor()))
    assert shadow["exit_policy_diagnostics"][0]["status"] == "exit_checkpoint_unavailable_at_signal"
    assert shadow["ledger"] == execute(held_input)["ledger"]
    applied = execute(held_input, make_exit_policy(features=held_input[1], predictor=FakePredictor(), apply_exit=True))
    assert applied["ledger"][1]["date"] == held_input[3][73]


def test_day_resolver_only_receives_actual_held_dates_and_no_empty_account_inference(held_input):
    seen = []
    policy = make_exit_policy(features=held_input[1], predictor_resolver=lambda date: seen.append(date))
    result = execute(held_input, policy)
    assert seen == [row["date"] for row in result["holding_snapshots"]] == held_input[3][71:81]
    seen.clear()
    held_input[0].loc[71, "open"] = 10.6
    execute(held_input, policy)
    assert seen == []


def test_numpy_daily_and_batch_predictions_match_real_torch(trained_exit):
    frame, directory, manifest, days = trained_exit
    from my_strategy.services.czsc_research_ml import PredictorSession
    predictor = ExitPredictor(directory, device="mps")
    daily = np.array([predictor.predict_one(row, as_of=row["date"]) for row in frame.iloc[210:230].to_dict("records")])
    torch = PredictorSession(device="cpu").predict(frame.iloc[210:230], directory, as_of=days[229])
    np.testing.assert_allclose(daily, torch, rtol=0, atol=1.2e-7)
    assert predictor.diagnostics["device"] == "cpu" and predictor.diagnostics["requested_device"] == "mps"
    assert predictor.diagnostics["model_loads"] == 1 and predictor.diagnostics["rows"] == 20
    assert not predictor.diagnostics["actual_gpu_inference"]


def test_numpy_cpu_worker_does_not_import_torch(trained_exit):
    _, directory, _, _ = trained_exit
    script = "import sys; from my_strategy.services.czsc_dual_exit import ExitPredictor; p=ExitPredictor(sys.argv[1],device='mps'); assert 'torch' not in sys.modules; assert p.diagnostics['device']=='cpu'"
    result = subprocess.run([sys.executable, "-c", script, str(directory)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_cpu_export_has_independent_binding_and_rejects_bitrot(trained_exit, tmp_path):
    import shutil
    frame, directory, manifest, days = trained_exit
    copy_dir = tmp_path / "copied-exit"
    shutil.copytree(directory, copy_dir)
    binding = json.loads((copy_dir / "head.binding.json").read_text())
    assert binding["checkpoint_sha256"] == manifest["checkpoint_sha256"]
    assert binding["model_manifest_sha256"] == manifest["manifest_sha256"]
    with pytest.raises(ValueError, match="binding"):
        ExitPredictor({"model_dir": copy_dir, "head_binding_sha256": "different"}, device="cpu")
    predictor = ExitPredictor({"model_dir": copy_dir, "head_binding_sha256": binding["binding_sha256"]}, device="cpu")
    path = copy_dir / "exit-linear.json"
    exported = json.loads(path.read_text())
    exported["weight"][0] += 1.
    path.write_text(json.dumps(exported))
    with pytest.raises(ValueError, match="file hash"):
        predictor.predict_one(frame.iloc[210].to_dict(), as_of=days[210])


def test_missing_mature_or_single_class_exit_folds_fail_closed(trained_exit, tmp_path):
    frame, _, _, days = trained_exit
    with pytest.raises(ValueError, match="insufficient mature samples"):
        train_exit_model(frame.iloc[:5], tmp_path / "too-small", days[149], days[160], days[209], device="cpu")
    single = frame.copy()
    single["label"] = 0.
    with pytest.raises(ValueError, match="both label classes"):
        train_exit_model(single, tmp_path / "one-class", days[149], days[160], days[209], device="cpu")
    assert not (tmp_path / "too-small/model.pt").exists() and not (tmp_path / "one-class/model.pt").exists()


def test_fixed_ten_day_entry_checkpoint_cannot_be_reused_as_exit(trained_exit, tmp_path):
    import shutil
    from my_strategy.services.czsc_research_ml import LABEL_VERSION, _hash
    _, directory, _, _ = trained_exit
    copied = tmp_path / "entry-checkpoint"
    shutil.copytree(directory, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["label_version"] = LABEL_VERSION
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = _hash(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="entry weights"):
        ExitPredictor(copied, device="cpu")


def test_actual_mps_exit_training_exports_equivalent_cpu_head(trained_exit, tmp_path):
    import torch
    from my_strategy.services.czsc_research_ml import PredictorSession
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable on this test host")
    frame, _, _, days = trained_exit
    directory = tmp_path / "actual-mps-exit"
    manifest = train_exit_model(frame, directory, days[149], days[160], days[209],
        device="mps", epochs=1, batch_size=64, min_train_rows=30, min_validation_rows=10)
    assert manifest["actual_mps_training"] and manifest["training_batches"] > 0
    cpu = ExitPredictor(directory, device="mps")
    cpu_values = cpu.predict(frame.iloc[210:230], as_of=days[229])
    mps = PredictorSession(device="mps")
    mps_values = mps.predict(frame.iloc[210:230], directory, as_of=days[229])
    np.testing.assert_allclose(cpu_values, mps_values, rtol=0, atol=2e-7)
    assert mps.diagnostics["actual_mps_inference"] and mps.diagnostics["rows"] == 20
    assert cpu.diagnostics["actual_cpu_numpy_inference"] and not cpu.diagnostics["actual_mps_inference"]
