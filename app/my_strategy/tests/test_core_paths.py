"""Path overrides and output containment are part of the runtime contract."""
import importlib
from pathlib import Path

import pytest

from my_strategy.core import paths


def test_paths_and_environment_override(monkeypatch, tmp_path):
    assert (paths.PROJECT_ROOT / "my_strategy" / "core" / "paths.py").is_file()
    with monkeypatch.context() as change:
        change.setenv("KHQUANT_DATA_ROOT", str(tmp_path / "data"))
        change.setenv("KHQUANT_ARTIFACT_ROOT", str(tmp_path / "outputs"))
        importlib.reload(paths)
        assert paths.RAW_DATA_ROOT == tmp_path / "data" / "raw"
        assert paths.ARTIFACT_RUNS_ROOT == tmp_path / "outputs" / "runs"
        assert paths.artifact_subdir("run-1", "reports").is_dir()
        assert not hasattr(paths, "MODEL_REGISTRY_ROOT")
    importlib.reload(paths)


@pytest.mark.parametrize("run_id", ["", "..", "../escape", "a/b", "C:\\temp"])
def test_run_path_rejects_traversal(run_id):
    with pytest.raises(ValueError):
        paths.artifact_run_dir(run_id)


def test_subdirectory_rejects_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACT_RUNS_ROOT", tmp_path)
    with pytest.raises(ValueError):
        paths.artifact_subdir("valid", "..", "..", "escape")


@pytest.mark.parametrize("transient_resolutions", [1, 2])
def test_subdirectory_retries_original_path_after_transient_resolution(tmp_path, monkeypatch, transient_resolutions):
    monkeypatch.setattr(paths, "ARTIFACT_RUNS_ROOT", tmp_path)
    original = tmp_path / "valid" / "evaluation" / "fold" / "stock"
    resolve = Path.resolve
    calls = 0

    def transient(path, *args, **kwargs):
        nonlocal calls
        if path == original:
            calls += 1
            if calls <= transient_resolutions:
                return tmp_path / "different-anchor" / "stock"
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", transient)
    destination = paths.artifact_subdir("valid", "evaluation", "fold", "stock")
    assert destination == original and destination.is_dir()
    assert calls == transient_resolutions + 1
    assert not (tmp_path / "different-anchor").exists()
