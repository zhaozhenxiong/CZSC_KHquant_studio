from __future__ import annotations

import json
import copy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.scripts.audit_czsc_research import _file_hash, _hash
from my_strategy.scripts.verify_czsc_research_reproducibility import (
    _fitting_hash, _frozen_device, _original_index, _probe, _probability_comparison, _semantics, _source, _tensor_comparison, verify_run,
)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "complete-production"
    (root / "models" / "production").mkdir(parents=True)
    (root / "reports").mkdir()
    checkpoint = root / "models" / "production" / "model.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    config = {"production": {"train_end": "2026-06-30", "validation_start": "2026-07-01", "validation_end": "2026-09-30"}, "seed": 42, "epochs": 20}
    schema = {"columns": ["feature"], "model_version": "fixture", "hidden_sizes": [32, 16]}
    preprocess = {"median": [0], "mean": [0], "scale": [1]}
    manifest = {**config["production"], "available_at": "2026-09-30", "seed": 42, "device": "cuda:1",
                "hyperparameters": {"epochs": 20, "batch_size": 2048, "learning_rate": .001},
                "schema": schema, "schema_sha256": _hash(schema), "preprocess": preprocess,
                "preprocess_sha256": _hash(preprocess), "checkpoint_sha256": _file_hash(checkpoint),
                "feature_schema_binding": "bound", "feature_version": "frozen-version", "feature_schema_hash": "frozen-schema"}
    manifest["manifest_sha256"] = _hash(manifest)
    research = {"run_id": root.name, "run_context": {"stocks": ["000001.SZ"]}, "config": config,
                "data_end": "2026-09-30", "model_manifest": manifest}
    metadata = {"status": "complete", "stocks": ["000001.SZ"], "config_hash": stable_hash(config), "research_config": config}
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "reports" / "research.json").write_text(json.dumps(research), encoding="utf-8")
    (checkpoint.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, root, research, manifest


def _resign_source(source):
    _, root, research, manifest = source
    manifest.pop("manifest_sha256", None)
    manifest["schema_sha256"] = _hash(manifest["schema"])
    manifest["manifest_sha256"] = _hash(manifest)
    research["model_manifest"] = manifest
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata.update(config_hash=stable_hash(research["config"]), research_config=research["config"])
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "reports" / "research.json").write_text(json.dumps(research), encoding="utf-8")
    (root / "models" / "production" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:0", "cuda:3"])
def test_source_accepts_actual_frozen_device_and_nondefault_configuration(source, device, monkeypatch):
    base, root, research, manifest = source
    research["config"].update(seed=17, epochs=7, batch_size=128, learning_rate=.003, hidden_sizes=[8, 4])
    manifest.update(seed=17, device=device, requested_device="auto")
    manifest["hyperparameters"].update(epochs=7, batch_size=128, learning_rate=.003)
    manifest["schema"]["hidden_sizes"] = [8, 4]
    _resign_source(source)
    monkeypatch.setenv("KHQUANT_COMPUTE_DEVICE", "cpu")
    assert _source(root.name, base)[2] == manifest
    assert _frozen_device(manifest) == device
    with pytest.raises(ValueError, match="frozen actual device"):
        _frozen_device(manifest, "auto")


@pytest.mark.parametrize("device", ["auto", "cuda", "mps:0", ""])
def test_source_rejects_unresolved_or_invalid_actual_device(source, device):
    base, root, _, manifest = source
    manifest["device"] = device
    _resign_source(source)
    with pytest.raises(ValueError, match="explicit frozen actual device"):
        _source(root.name, base)


@pytest.mark.parametrize("key,value", [("seed", 17), ("epochs", 7), ("batch_size", 128), ("learning_rate", .003), ("hidden_sizes", [8, 4])])
def test_source_rejects_training_configuration_mismatch_after_integrity_is_resigned(source, key, value):
    base, root, research, _ = source
    research["config"][key] = value
    _resign_source(source)
    with pytest.raises(ValueError, match="frozen configuration"):
        _source(root.name, base)


@pytest.fixture
def mps_reproduction(source, monkeypatch):
    from my_strategy.scripts import verify_czsc_research_reproducibility as verifier
    from my_strategy.services import czsc_research_ml as ml
    base, root, research, manifest = source
    research["config"].update(seed=17, epochs=7, feature_profile="ma_trend_v1")
    manifest.update(device="mps", requested_device="auto", seed=17, torch_version="fixture-torch", cuda_version=None,
                    train_signal_start="2026-06-01", data_version="frozen-data", label_contract={"horizon": 10},
                    device_name="Apple MPS", training_batches=2, actual_mps_training=True,
                    actual_cuda_training=False, actual_gpu_training=True,
                    calibration={"temperature": 1.0}, training_loss=[.4], sample_counts={"train": 1, "validation": 1},
                    train_metrics={"rows": 1}, validation_metrics={"rows": 1})
    manifest["hyperparameters"]["epochs"] = 7
    dataset = pd.DataFrame({"symbol": ["000001.SZ"] * 2, "date": ["2026-06-01", "2026-07-01"],
                            "label_end": ["2026-06-16", "2026-07-16"], "label": [0, 1],
                            "label_available": True, "input_eligible": True, "feature": [1., 2.]})
    dataset.attrs.update(feature_version=manifest["feature_version"], schema_hash=manifest["feature_schema_hash"],
                         data_version=manifest["data_version"], label_contract=manifest["label_contract"])
    (root / "dataset").mkdir()
    dataset.to_parquet(root / "dataset" / "training.parquet", index=False)
    manifest["dataset_sha256"] = _fitting_hash(dataset, manifest)
    research["training_rows"], research["dataset_records"] = len(dataset), []
    for index in range(20):
        symbol = f"{index + 1:06d}.SZ"
        feature = pd.DataFrame({"symbol": [symbol], "date": [research["data_end"]],
                                "input_eligible": False, "model_input_eligible": True, "feature": [float(index)]})
        feature.attrs.update(feature_version=manifest["feature_version"], schema_hash=manifest["feature_schema_hash"])
        path = root / "dataset" / f"{symbol}.parquet"
        feature.to_parquet(path, index=False)
        research["dataset_records"].append({"symbol": symbol, "path": str(path), "sha256": _file_hash(path)})
    _resign_source(source)
    calls, metadata = [], []
    fake_torch = SimpleNamespace(__version__="fixture-torch", version=SimpleNamespace(cuda=None),
                                 backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                                 cuda=SimpleNamespace(is_available=lambda: False),
                                 load=lambda *args, **kwargs: {"state_dict": {"weight": "same-frozen-tensor"}},
                                 get_num_threads=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    output_root = base / "reproduction"
    def subdir(*parts):
        path = output_root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path
    context = SimpleNamespace(run_id="reproduction", subdir=subdir, write_metadata=lambda value: metadata.append(value))
    def create_context(**kwargs):
        calls.append(("context", kwargs))
        return context
    def train(frame, columns, output, train_end, validation_start, validation_end, **kwargs):
        calls.append(("train", kwargs))
        assert frame["input_eligible"].all() and columns == manifest["schema"]["columns"]
        (output / "model.pt").write_bytes(b"retrained-fixture")
        return copy.deepcopy(manifest)
    def predict(frame, output, **kwargs):
        calls.append(("predict", kwargs))
        assert frame["input_eligible"].all() and frame["model_input_eligible"].all()
        return np.full(len(frame), .6)
    monkeypatch.setattr(verifier, "create_run_context", create_context)
    monkeypatch.setattr(verifier, "_tensor_comparison", lambda left, right: {"all_tensors_exact": left == right})
    monkeypatch.setattr(ml, "train_model", train)
    monkeypatch.setattr(ml, "predict_model", predict)
    return source, calls, metadata, fake_torch


def test_mps_reproduction_uses_frozen_parameters_and_model_only_probe_eligibility(mps_reproduction):
    source, calls, _, _ = mps_reproduction
    base, root, research, _ = source
    report = verify_run(root.name, runs_root=base)
    assert report["passed"] and report["actual_training_batches"] == 2
    trained = next(arguments for name, arguments in calls if name == "train")
    assert (trained["device"], trained["seed"], trained["epochs"]) == ("mps", 17, 7)
    assert trained["hidden_sizes"] == [32, 16]
    assert [arguments["device"] for name, arguments in calls if name == "predict"] == ["mps", "mps", "cpu"]
    assert all(record["eligibility_column"] == "model_input_eligible" for record in report["probe"]["real_records"])
    assert not report["source_artifacts_modified"]
    assert not pd.read_parquet(research["dataset_records"][0]["path"])["input_eligible"].any()


def test_unavailable_mps_is_rejected_before_creating_run_or_training(mps_reproduction):
    source, calls, metadata, fake_torch = mps_reproduction
    base, root, _, _ = source
    fake_torch.backends.mps.is_available = lambda: False
    with pytest.raises(RuntimeError, match="MPS.*no CPU fallback"):
        verify_run(root.name, runs_root=base)
    assert not calls and not metadata


def test_explicit_device_override_cannot_switch_frozen_mps_to_cpu(mps_reproduction):
    source, calls, metadata, _ = mps_reproduction
    base, root, _, _ = source
    with pytest.raises(ValueError, match="match the frozen actual device"):
        verify_run(root.name, runs_root=base, device="cpu")
    assert not calls and not metadata


def test_actual_cpu_fallback_during_mps_retrain_fails_without_inference(mps_reproduction, monkeypatch):
    from my_strategy.services import czsc_research_ml as ml
    source, calls, metadata, _ = mps_reproduction
    base, root, _, manifest = source
    monkeypatch.setattr(ml, "train_model", lambda *args, **kwargs: {**manifest, "device": "cpu"})
    with pytest.raises(RuntimeError, match="actual device differs.*no fallback"):
        verify_run(root.name, runs_root=base)
    assert not any(name == "predict" for name, _ in calls)
    assert metadata[-1]["status"] == "failed"


def test_complete_guard_runs_before_torch_import(tmp_path):
    root = tmp_path / "partial-production"
    root.mkdir()
    (root / "metadata.json").write_text('{"status":"running"}', encoding="utf-8")
    code = "\n".join([
        "import sys", "from pathlib import Path",
        "from my_strategy.scripts.verify_czsc_research_reproducibility import verify_run",
        "try:", "    verify_run('partial-production', runs_root=Path(sys.argv[1]))",
        "except ValueError as exc:", "    assert 'not complete' in str(exc)",
        "else:", "    raise AssertionError('partial source was accepted')",
        "assert 'torch' not in sys.modules",
    ])
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_source_model_hash_and_config_are_frozen(source):
    base, root, research, manifest = source
    assert _source(root.name, base)[2] == manifest
    path = root / "reports" / "research.json"
    research["config"]["epochs"] = 21
    path.write_text(json.dumps(research), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen configuration"):
        _source(root.name, base)
    research["config"]["epochs"] = 20
    path.write_text(json.dumps(research), encoding="utf-8")
    (root / "models" / "production" / "model.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint hash"):
        _source(root.name, base)
    with pytest.raises(ValueError, match="run identity"):
        _source("../complete-production", base)


def test_parquet_index_loss_is_detected_and_original_fingerprint_recovered(tmp_path):
    first = pd.DataFrame({"symbol": ["000002.SZ"] * 3, "date": ["2026-06-01", "2026-07-01", "2026-09-25"],
                          "label_end": ["2026-06-16", "2026-07-16", "2026-10-12"], "label": [0, 1, 0],
                          "label_available": True, "input_eligible": True, "feature": [1., 2., 3.]})
    second = first.assign(symbol="000001.SZ", feature=[4., 5., 6.])
    original = pd.concat([first, second], ignore_index=True).sort_values(["date", "symbol"])
    path = tmp_path / "training.parquet"
    original.to_parquet(path, index=False)
    read_back = pd.read_parquet(path)
    manifest = {"train_end": "2026-06-30", "validation_start": "2026-07-01", "validation_end": "2026-09-30"}
    records = [{"symbol": "000002.SZ"}, {"symbol": "000001.SZ"}]
    expected = _fitting_hash(original, manifest)
    assert _fitting_hash(read_back, manifest) != expected
    index = _original_index(read_back, records)
    assert list(index) == list(original.index)
    assert _fitting_hash(read_back, manifest, index) == expected
    changed = read_back.copy()
    changed.loc[0, "feature"] += .1
    assert _fitting_hash(changed, manifest, index) != expected
    # An unexpired end-of-period label is not part of the fitting fingerprint.
    changed = read_back.copy()
    changed.loc[changed["date"] == "2026-09-25", "feature"] += 100
    assert _fitting_hash(changed, manifest, index) == expected


def test_schema_binding_and_latest_real_rows(source):
    _, root, research, manifest = source
    research["dataset_records"] = []
    (root / "dataset").mkdir()
    for index in range(21):
        symbol = f"{index + 1:06d}.SZ"
        frame = pd.DataFrame({"symbol": [symbol] * 2, "date": ["2026-09-29", "2026-09-30"],
                              "input_eligible": [True, index != 0], "feature": [float(index), float(index + 1)]})
        frame.attrs.update(feature_version="frozen-version", schema_hash="frozen-schema")
        path = root / "dataset" / (symbol.replace(".", "_") + ".parquet")
        frame.to_parquet(path, index=False)
        research["dataset_records"].append({"symbol": symbol, "path": str(path), "sha256": _file_hash(path)})
    probe, evidence = _probe(root, research, manifest)
    assert len(probe) == len(evidence) == 20
    assert probe["date"].eq("2026-09-30").all()
    assert probe["input_eligible"].all()
    assert "000001.SZ" not in probe["symbol"].tolist()
    path = root / "dataset" / "000002_SZ.parquet"
    frame = pd.read_parquet(path)
    frame.attrs["schema_hash"] = "changed-semantic-schema"
    frame.to_parquet(path, index=False)
    research["dataset_records"][1]["sha256"] = _file_hash(path)
    with pytest.raises(ValueError, match="semantic mismatch"):
        _probe(root, research, manifest)
    missing = pd.DataFrame({"feature": [1.]})
    assert _semantics(missing, manifest) == ["feature_version", "schema_hash"]
    assert missing.attrs["schema_hash"] == manifest["feature_schema_hash"]


def test_exact_tensor_and_inference_tolerance_are_separate():
    import torch
    left = {"weight": torch.tensor([[1., 2.]]), "bias": torch.tensor([.5])}
    right = {name: value.clone() for name, value in left.items()}
    assert _tensor_comparison(left, right)["all_tensors_exact"]
    right["weight"][0, 0] += 1e-5
    comparison = _tensor_comparison(left, right)
    assert not comparison["all_tensors_exact"]
    assert comparison["tensors"][1]["max_absolute_difference"] > 1e-6
    right["weight"] = right["weight"].to(torch.float64)
    assert not _tensor_comparison(left, right)["tensors"][1]["compatible"]
    probabilities = np.array([.1, .6])
    assert _probability_comparison(probabilities, probabilities + 5e-7)["passed"]
    assert not _probability_comparison(probabilities, probabilities + 2e-6)["passed"]
    assert not _probability_comparison(probabilities, np.array([np.nan, .6]))["passed"]
