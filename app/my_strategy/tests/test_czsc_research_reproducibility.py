from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.scripts.audit_czsc_research import _file_hash, _hash
from my_strategy.scripts.verify_czsc_research_reproducibility import (
    _fitting_hash, _original_index, _probe, _probability_comparison, _semantics, _source, _tensor_comparison,
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
