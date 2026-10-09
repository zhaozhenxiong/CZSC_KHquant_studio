"""Shared Torch device selection without importing Torch in CPU workers."""
from __future__ import annotations

import os
import re
from typing import Any


def requested_torch_device(requested: str | None = None) -> str:
    value = str(requested if requested is not None else os.environ.get("KHQUANT_COMPUTE_DEVICE", "auto")).strip().lower()
    if not re.fullmatch(r"auto|cpu|mps|cuda(?::\d+)?", value):
        raise ValueError("计算设备必须为 auto、cpu、mps、cuda 或 cuda:编号")
    return value


def is_mps_available(torch: Any = None) -> bool:
    if torch is None:
        try:
            import torch
        except Exception:
            return False
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(backend is not None and backend.is_available())


def is_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def select_torch_device(torch: Any, requested: str | None = None) -> str:
    """Auto selects CUDA, MPS, CPU; explicit unavailable devices never fall back."""
    name = requested_torch_device(requested)
    if name == "cpu":
        return name
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if name == "auto":
        if cuda_count:
            free_memory = []
            for index in range(cuda_count):
                try:
                    free = torch.cuda.mem_get_info(index)[0]
                except Exception:
                    free = torch.cuda.get_device_properties(index).total_memory
                free_memory.append((free, index))
            return f"cuda:{max(free_memory)[1]}"
        return "mps" if is_mps_available(torch) else "cpu"
    if name == "mps":
        if not is_mps_available(torch):
            raise RuntimeError("explicit MPS device unavailable: mps; no CPU fallback")
        return name
    index = int(name.split(":")[1]) if ":" in name else 0
    if index >= cuda_count:
        raise RuntimeError(f"explicit CUDA device unavailable: {name}; no CPU fallback")
    return f"cuda:{index}"


def synchronize_torch_device(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def get_default_device(requested: str | None = None) -> str:
    import torch
    return select_torch_device(torch, requested)
