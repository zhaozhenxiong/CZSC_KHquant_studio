"""Cross-platform device contracts without pretending that a mock ran on MPS."""
from types import SimpleNamespace
import sys

import pytest

from my_strategy.core.device import requested_torch_device, select_torch_device
from my_strategy.scripts import docker_runtime_doctor as doctor
from my_strategy.services import czsc_compute as compute


def fake_torch(*, cuda_memory=(), mps=False, built=False):
    return SimpleNamespace(
        __version__="fixture", version=SimpleNamespace(cuda="fixture" if cuda_memory else None),
        cuda=SimpleNamespace(is_available=lambda: bool(cuda_memory), device_count=lambda: len(cuda_memory),
            mem_get_info=lambda index: (cuda_memory[index], cuda_memory[index]),
            get_device_properties=lambda index: SimpleNamespace(name=f"CUDA {index}", total_memory=cuda_memory[index], major=9, minor=0)),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps, is_built=lambda: built)))


def test_auto_selects_available_backend_and_preserves_explicit_requests(monkeypatch):
    monkeypatch.setenv("KHQUANT_COMPUTE_DEVICE", "cpu")
    assert requested_torch_device() == "cpu"
    assert select_torch_device(fake_torch(cuda_memory=(100, 200), mps=True), "auto") == "cuda:1"
    assert select_torch_device(fake_torch(mps=True), "auto") == "mps"
    assert select_torch_device(fake_torch(), "auto") == "cpu"
    assert select_torch_device(fake_torch(cuda_memory=(100, 200)), "cuda:0") == "cuda:0"
    assert select_torch_device(fake_torch(cuda_memory=(100,)), "cuda") == "cuda:0"
    assert select_torch_device(fake_torch(mps=True), "cpu") == "cpu"


@pytest.mark.parametrize("device", ["mps", "cuda", "cuda:99"])
def test_explicit_unavailable_accelerator_does_not_fall_back(device):
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        select_torch_device(fake_torch(), device)


@pytest.mark.parametrize("device", ["cuda:-1", "cuda:bogus", "mps:0", "tpu"])
def test_invalid_device_rejected_at_cli_api_boundary(device):
    from my_strategy.web_dashboard.api import AnalysisRequest, TaskSpec
    with pytest.raises(ValueError):
        requested_torch_device(device)
    with pytest.raises(ValueError):
        AnalysisRequest(symbol="000001.SZ", device=device)
    with pytest.raises(ValueError):
        TaskSpec(device=device)


def test_mps_selection_keeps_float64_structure_stage_on_cpu(monkeypatch):
    monkeypatch.setattr(compute, "_torch_runtime", lambda: (fake_torch(mps=True, built=True), None))
    status = compute.compute_status("auto")
    assert status["selected_device"] == "mps" and status["mps_available"] and status["mps_built"]
    stages = {stage["stage"]: stage for stage in status["stages"]}
    assert stages["model"]["device"] == "mps"
    assert stages["batched_signals"]["device"] == "cpu" and "float64" in stages["batched_signals"]["reason"]
    assert status["devices"][0]["device"] == "mps"
    assert compute.compute_status("cuda:99")["available"] is False


def test_cli_and_api_accept_mps_and_train_defaults_to_auto():
    from my_strategy.cli import build_parser
    from my_strategy.web_dashboard.api import AnalysisRequest, TaskSpec
    assert build_parser().parse_args(["research-train", "--end", "2026-09-30"]).device == "auto"
    assert AnalysisRequest(symbol="000001.SZ", device="mps").device == "mps"
    assert TaskSpec(device="mps").device == "mps"


def test_cpu_doctor_requires_torch_even_when_structure_capability_is_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    report = doctor.torch_report({"available": True, "selected_device": "cpu"})
    assert not report["ok"] and not report["training_work"] and "ModuleNotFoundError" in report["error"]


def test_cpu_doctor_runs_training_and_inference_and_restores_determinism():
    import torch
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    report = doctor.torch_report({"available": True, "selected_device": "cpu"})
    assert report["ok"], report["error"]
    assert report["training_work"] and report["inference_work"] and report["dtype"] == "float32"
    assert not report["actual_cuda_work"] and not report["actual_mps_work"]
    assert before == (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())


@pytest.mark.parametrize("device", ["cuda:0", "mps"])
def test_available_accelerator_doctor_runs_the_actual_training_operators(device):
    status = compute.compute_status(device)
    if not status["available"]:
        pytest.skip(f"Actual {device} doctor validation requires available hardware")
    report = doctor.torch_report(status)
    assert report["ok"], report["error"]
    assert report["training_work"] and report["inference_work"]
    assert report["actual_mps_work"] == (device == "mps")
    assert report["actual_cuda_work"] == device.startswith("cuda")
