from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_research_ml import PredictorSession, build_labels, date_block_bootstrap, predict_model, train_model
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES


@pytest.fixture
def bars():
    dates = pd.bdate_range("2024-01-02", periods=140)
    price = 10 + np.arange(len(dates)) * 0.005 + 0.12 * np.sin(np.arange(len(dates)) / 5)
    return pd.DataFrame({"date": dates, "symbol": "000001.SZ", "open": price, "high": price + 0.2,
                         "low": price - 0.2, "close": price + 0.02, "volume": 1000000.0,
                         "amount": price * 1000000, "source": "tushare", "has_trade_price": 1})


def test_label_cpu_module_does_not_import_torch():
    result = subprocess.run([sys.executable, "-c", "import sys; from my_strategy.services.czsc_research_ml import build_labels; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_fixed_horizon_labels_match_actual_broker_ledger(bars):
    labels = build_labels(bars, bars["date"])
    signal_index, horizon = 70, 10
    decisions = [{"date": date.date().isoformat(), "target_weight": 0.95 if signal_index <= index < signal_index + horizon else 0.0}
                 for index, date in enumerate(bars["date"])]
    result = execute_decisions("000001.SZ", bars, decisions, "2024-01-02", 100000, strategy_config(), "label-parity",
                               market_dates=bars["date"].dt.strftime("%Y-%m-%d").tolist(), verified_sources=VERIFIED_SOURCES)
    label = labels.iloc[signal_index]
    assert label["label_available"]
    assert len(result["ledger"]) == 2
    assert label["entry_date"] == result["ledger"][0]["date"] == bars.iloc[71]["date"].date().isoformat()
    assert label["exit_date"] == result["ledger"][1]["date"] == bars.iloc[81]["date"].date().isoformat()
    assert label["entry_shares"] == result["ledger"][0]["shares"]
    assert label["fees"] == pytest.approx(sum(row["fee"] for row in result["ledger"]))
    assert label["net_return"] == pytest.approx(result["metrics"]["total_return"])


def test_calendar_is_required_and_missing_scheduled_bar_is_not_substituted(bars):
    assert set(build_labels(bars)["label_reason"]) == {"calendar_unverified"}
    missing_entry = bars.drop(index=71)
    labels = build_labels(missing_entry, bars["date"])
    assert labels.loc[70, "label_reason"] == "missing_entry_bar"
    assert np.isnan(labels.loc[70, "label"])
    assert labels.loc[70, "entry_date"] == bars.loc[71, "date"].date().isoformat()
    missing_middle = build_labels(bars.drop(index=75), bars["date"])
    assert missing_middle.loc[70, "label_reason"] == "gap_in_holding_window"


def test_label_rejections_partial_exit_and_prefix_maturity(bars):
    changed = bars.copy()
    changed.loc[71, "open"] = changed.loc[70, "close"] * 1.06
    assert build_labels(changed, bars["date"]).loc[70, "label_reason"] == "entry_conservative_upper_price_guard"
    changed = bars.copy()
    changed.loc[80, "volume"] = 100.0
    assert build_labels(changed, bars["date"]).loc[70, "label_reason"] == "exit_insufficient_prior_liquidity_for_full_exit"
    prefix = build_labels(bars.iloc[:90], bars["date"])
    full = build_labels(bars, bars["date"])
    mature = prefix["label_available"]
    pd.testing.assert_frame_equal(prefix.loc[mature], full.loc[prefix.index[mature]])
    assert prefix.loc[80, "label_reason"] == "label_immature"
    assert np.isnan(prefix.loc[80, "label"])
    assert full.loc[80, "label_available"]


def test_unverified_holding_prices_and_nonmainboard_remain_in_table(bars):
    changed = bars.copy()
    changed.loc[75, "source"] = "fuyao"
    result = build_labels(changed, bars["date"])
    assert len(result) == len(bars)
    assert result.loc[70, "label_reason"] == "unverified_research_source"
    changed["symbol"] = "300001.SZ"
    assert build_labels(changed, bars["date"]).loc[90, "label_reason"] == "entry_unsupported_board"


def test_future_execution_source_must_be_verified_even_when_signal_source_is_valid(bars):
    for index in (71, 75, 81):
        changed = bars.copy()
        changed.loc[index, "source"] = "provider_adjustment_unknown"
        labels = build_labels(changed, bars["date"])
        assert labels.loc[70, "label_reason"] == "unverified_research_source"
        assert not labels.loc[70, "label_available"] and np.isnan(labels.loc[70, "label"])


@pytest.fixture
def learning_data():
    dates = pd.bdate_range("2024-01-02", periods=130)
    x = np.sin(np.arange(len(dates)) / 3)
    return pd.DataFrame({"date": dates, "symbol": "000001.SZ", "x": x, "z": np.cos(np.arange(len(dates)) / 7),
                         "label": (x > 0).astype(float), "label_end": dates + pd.offsets.BDay(11),
                         "label_available": True, "input_eligible": True})


def _train(data, path):
    return train_model(data, ["x", "z"], path, "2024-03-29", "2024-04-01", "2024-05-31",
                       device="cpu", epochs=3, batch_size=32, hidden_sizes=(8, 4), min_train_rows=20, min_validation_rows=20)


def test_training_preprocess_is_train_only_and_purges_actual_end(learning_data, tmp_path):
    first = _train(learning_data, tmp_path / "first")
    changed = learning_data.copy()
    changed.loc[changed["date"] >= "2024-04-01", ["x", "z"]] = 1000000
    second = _train(changed, tmp_path / "second")
    assert first["preprocess"] == second["preprocess"]
    assert first["train_label_end"] <= "2024-03-29"
    assert first["validation_label_end"] <= "2024-05-31"
    train_rows = learning_data[(learning_data["date"] <= "2024-03-29") & (learning_data["label_end"] <= "2024-03-29")]
    assert first["sample_counts"]["train"] == len(train_rows)
    assert first["preprocess"]["mean"][0] == pytest.approx(train_rows["x"].mean())
    import torch
    left = torch.load(tmp_path / "first" / "model.pt", weights_only=True)["state_dict"]
    right = torch.load(tmp_path / "second" / "model.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(left[name], right[name]) for name in left)


def test_prediction_replay_schema_integrity_and_input_rejection(learning_data, tmp_path):
    _train(learning_data, tmp_path)
    test = learning_data.loc[learning_data["date"] > "2024-05-31"].copy()
    test.iloc[-1, test.columns.get_loc("input_eligible")] = False
    prediction = predict_model(test, tmp_path, device="cpu", as_of="2024-07-31")
    assert np.isfinite(prediction[:-1]).all() and np.isnan(prediction[-1])
    with pytest.raises(ValueError, match="unavailable"):
        predict_model(learning_data.iloc[:1], tmp_path, device="cpu")
    with pytest.raises(ValueError, match="schema"):
        predict_model(test.drop(columns=["z"]), tmp_path, device="cpu")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checkpoint hash"):
        predict_model(test, tmp_path, device="cpu")


def test_manifest_preprocess_mismatch_fails(learning_data, tmp_path):
    _train(learning_data, tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["preprocess"]["median"][0] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="integrity mismatch"):
        predict_model(learning_data.loc[learning_data["date"] > "2024-05-31"], tmp_path, device="cpu")


def test_same_columns_cannot_substitute_changed_feature_semantics(learning_data, tmp_path):
    learning_data.attrs.update(feature_version="causal_features_v1", schema_hash="locked-formulas-v1")
    manifest = _train(learning_data, tmp_path)
    assert manifest["feature_version"] == "causal_features_v1"
    assert manifest["feature_schema_hash"] == "locked-formulas-v1"
    assert manifest["feature_schema_binding"] == "bound"
    test = learning_data.loc[learning_data["date"] > "2024-05-31"].copy()
    assert np.isfinite(predict_model(test, tmp_path, device="cpu")).all()
    for attrs in ({}, {"feature_version": "causal_features_v2", "schema_hash": "locked-formulas-v1"},
                  {"feature_version": "causal_features_v1", "schema_hash": "changed-same-column-formulas"}):
        changed = test.copy()
        changed.attrs = attrs
        with pytest.raises(ValueError, match="semantic version/schema"):
            predict_model(changed, tmp_path, device="cpu")


def test_partial_feature_binding_fails_and_fixture_is_explicitly_unbound(learning_data, tmp_path):
    learning_data.attrs["feature_version"] = "partial"
    with pytest.raises(ValueError, match="both feature_version and schema_hash"):
        _train(learning_data, tmp_path / "partial")
    learning_data.attrs = {}
    assert _train(learning_data, tmp_path / "fixture")["feature_schema_binding"] == "unbound_research_fixture"


def test_explicit_cuda_failure_has_no_fallback(learning_data, tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        train_model(learning_data, ["x", "z"], tmp_path, "2024-03-29", "2024-04-01", "2024-05-31", device="cuda:1", min_train_rows=20, min_validation_rows=20)
    assert not (tmp_path / "model.pt").exists()


def test_training_epoch_progress_and_batch_cancel_leave_no_checkpoint(learning_data, tmp_path, monkeypatch):
    import torch
    completed = []
    train_model(learning_data, ["x", "z"], tmp_path / "complete", "2024-03-29", "2024-04-01", "2024-05-31",
                device="cpu", epochs=3, batch_size=16, min_train_rows=20, min_validation_rows=20,
                progress=lambda current, total: completed.append((current, total)))
    assert completed == [(1, 3), (2, 3), (3, 3)]
    forwards = []
    original = torch.nn.Sequential.forward
    def observed_forward(self, values):
        forwards.append(len(values))
        return original(self, values)
    monkeypatch.setattr(torch.nn.Sequential, "forward", observed_forward)
    def check_cancel():
        if forwards:
            raise RuntimeError("user cancelled training")
    cancelled_progress = []
    with pytest.raises(RuntimeError, match="user cancelled"):
        train_model(learning_data, ["x", "z"], tmp_path / "cancel", "2024-03-29", "2024-04-01", "2024-05-31",
                    device="cpu", epochs=3, batch_size=16, min_train_rows=20, min_validation_rows=20,
                    check_cancel=check_cancel, progress=lambda current, total: cancelled_progress.append((current, total)))
    assert len(forwards) == 1 and not cancelled_progress
    assert not (tmp_path / "cancel" / "model.pt").exists()
    assert not (tmp_path / "cancel" / "manifest.json").exists()


def test_bad_label_time_and_overlapping_splits_fail(learning_data, tmp_path):
    changed = learning_data.copy()
    changed.loc[0, "label_end"] = changed.loc[0, "date"]
    with pytest.raises(ValueError, match="later than"):
        _train(changed, tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        train_model(learning_data, ["x"], tmp_path, "2024-04-01", "2024-04-01", "2024-05-31", device="cpu")


def test_cuda_inference_cpu_parity_and_reproducible_training(learning_data, tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA parity requires an available GPU")
    selected = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    parameters = {"device": selected, "epochs": 3, "batch_size": 32, "hidden_sizes": (8, 4),
                  "min_train_rows": 20, "min_validation_rows": 20}
    first = train_model(learning_data, ["x", "z"], tmp_path / "first", "2024-03-29", "2024-04-01", "2024-05-31", **parameters)
    second = train_model(learning_data, ["x", "z"], tmp_path / "second", "2024-03-29", "2024-04-01", "2024-05-31", **parameters)
    assert first["actual_cuda_training"] and first["training_loss"] == second["training_loss"]
    test = learning_data.loc[learning_data["date"] > "2024-05-31"]
    cpu = predict_model(test, tmp_path / "first", device="cpu")
    gpu = predict_model(test, tmp_path / "first", device=selected)
    assert gpu == pytest.approx(cpu, abs=1e-6, rel=1e-6)


def test_mps_training_inference_and_portable_checkpoint_cpu_parity(learning_data, tmp_path):
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("MPS validation requires an available Mac GPU")
    manifest = train_model(learning_data, ["x", "z"], tmp_path, "2024-03-29", "2024-04-01", "2024-05-31",
                           device="mps", epochs=3, batch_size=32, hidden_sizes=(8, 4),
                           min_train_rows=20, min_validation_rows=20)
    assert manifest["actual_mps_training"] and manifest["actual_gpu_training"] and not manifest["actual_cuda_training"]
    checkpoint = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=True)
    assert all(tensor.device.type == "cpu" and tensor.dtype == torch.float32 for tensor in checkpoint["state_dict"].values())
    sample = learning_data.loc[learning_data["date"] > "2024-05-31"]
    cpu = predict_model(sample, tmp_path, device="cpu")
    session = PredictorSession(device="mps", batch_size=8)
    mps = session.predict(sample, tmp_path)
    np.testing.assert_allclose(mps, cpu, atol=1e-5, rtol=1e-5)
    assert session.diagnostics["actual_mps_inference"] and session.diagnostics["mps_rows"] == len(sample)


def test_date_blocks_preserve_cross_section_and_dependency():
    dates = pd.bdate_range("2024-01-01", periods=100)
    data = pd.DataFrame({"date": np.repeat(dates, 2), "paired_gain": np.tile([1.0, -1.0], 100)})
    result = date_block_bootstrap(data, "paired_gain", repetitions=100)
    assert result["mean"] == result["lower"] == result["upper"] == 0
    assert result["dates"] == 100 and result["rows"] == 200
    with pytest.raises(ValueError, match="holding dependence"):
        date_block_bootstrap(data, "paired_gain", block_length=5)
