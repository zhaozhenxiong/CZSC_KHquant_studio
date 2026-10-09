"""Task-local loaded models preserve hash, temporal and semantic checks."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.czsc_research_ml import PredictorSession, predict_model, train_model


@pytest.fixture
def trained(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=130)
    x = np.sin(np.arange(len(dates)) / 3)
    data = pd.DataFrame({"date": dates, "symbol": "000001.SZ", "x": x,
                        "z": np.cos(np.arange(len(dates)) / 7), "label": (x > 0).astype(float),
                        "label_end": dates + pd.offsets.BDay(11), "label_available": True, "input_eligible": True})
    data.attrs.update(feature_version="fixture_causal_v1", schema_hash="fixture_formulas_v1")
    manifest = train_model(data, ["x", "z"], tmp_path, "2024-03-29", "2024-04-01", "2024-05-31",
                device="cpu", epochs=2, batch_size=32, hidden_sizes=(8, 4), min_train_rows=20, min_validation_rows=20)
    assert manifest["training_batches"] > 0 and not manifest["actual_mps_training"] and not manifest["actual_cuda_training"]
    return tmp_path, data


def test_reuse_single_load_rechecks_hashes_and_preserves_batched_probabilities(trained, monkeypatch):
    import torch
    directory, data = trained
    test = data.loc[data["date"] > "2024-05-31"].copy()
    test.iloc[-1, test.columns.get_loc("input_eligible")] = False
    expected = predict_model(test, directory, "cpu", as_of="2024-07-31")
    load, calls = torch.load, []
    def tracked(*args, **kwargs):
        calls.append(args[0])
        return load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", tracked)
    session = PredictorSession("cpu", batch_size=8)
    actual = session.predict(test, directory, as_of="2024-07-31")
    again = session.predict(test.iloc[:3], directory, as_of="2024-07-31")
    assert actual == pytest.approx(expected, nan_ok=True, abs=1e-7)
    assert again == pytest.approx(expected[:3], abs=1e-7)
    assert len(calls) == session.diagnostics["model_loads"] == 1
    assert session.diagnostics["hash_checks"] == 2 and session.diagnostics["cache_hits"] == 1
    assert session.diagnostics["rows"] == len(test) - 1 + 3
    assert session.diagnostics["device"] == "cpu" and not session.diagnostics["actual_cuda_inference"]


def test_cached_model_rejects_weight_or_manifest_tamper(trained):
    directory, data = trained
    test = data.loc[data["date"] > "2024-05-31"]
    session = PredictorSession("cpu")
    session.predict(test, directory)
    checkpoint = directory / "model.pt"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="checkpoint hash"):
        session.predict(test, directory)
    checkpoint.write_bytes(original)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["preprocess"]["median"][0] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="integrity"):
        session.predict(test, directory)


def test_retrospective_entrypoint_explicit_and_future_or_schema_still_rejected(trained):
    directory, data = trained
    session = PredictorSession("cpu")
    prefix = data.iloc[:10].copy()
    with pytest.raises(ValueError, match="unavailable"):
        session.predict(prefix, directory, as_of="2024-01-31")
    retrospective = session.predict_retrospective(prefix, directory, as_of="2024-01-31")
    assert np.isfinite(retrospective).all()
    assert session.last_prediction["mode"] == "retrospective"
    assert session.last_prediction["feature_rows_before_available"] == 10
    assert not session.last_prediction["eligible_for_unbiased_certification"]
    with pytest.raises(ValueError, match="future feature"):
        session.predict_retrospective(prefix, directory, as_of="2024-01-02")
    changed = prefix.copy()
    changed.attrs["schema_hash"] = "modified_same_columns"
    with pytest.raises(ValueError, match="semantic version/schema"):
        session.predict_retrospective(changed, directory, as_of="2024-01-31")


@pytest.mark.parametrize("device", ["cuda:1", "mps"])
def test_explicit_accelerator_session_has_no_cpu_fallback(trained, monkeypatch, device):
    import torch
    directory, data = trained
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        PredictorSession(device).predict(data.loc[data["date"] > "2024-05-31"], directory)


def test_default_inference_uses_runtime_environment_instead_of_training_device(trained, monkeypatch):
    import torch
    from my_strategy.services.czsc_research_ml import _hash
    directory, data = trained
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("manifest_sha256")
    manifest["device"] = "cuda:99"  # Simulate portable CPU weights from another machine.
    manifest["manifest_sha256"] = _hash(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    original = path.read_bytes()
    monkeypatch.setenv("KHQUANT_COMPUTE_DEVICE", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    sample = data.loc[data["date"] > "2024-05-31"]
    session = PredictorSession()
    actual = session.predict(sample, directory)
    expected = PredictorSession("cpu").predict(sample, directory)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    assert session.diagnostics["device"] == "cpu"
    assert path.read_bytes() == original
