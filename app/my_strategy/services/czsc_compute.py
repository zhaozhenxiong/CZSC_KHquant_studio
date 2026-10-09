"""CPU structure preparation and real CUDA batch signals for native Position.

No Torch import occurs when this module is imported in a spawned CPU worker.
The chart path remains in czsc_analysis and retains its confirmation history.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.core.device import is_mps_available, requested_torch_device, select_torch_device
from my_strategy.services.czsc_analysis import CzscSession, IntentSession, strategy_config


@dataclass
class PreparedReplay:
    frame: pd.DataFrame
    symbol: str
    config_hash: str
    daily_directions: np.ndarray
    weekly_latest: np.ndarray
    weekly_previous: np.ndarray
    bar_ids: np.ndarray


def prepare_replay(frame: pd.DataFrame, config: dict[str, Any]) -> PreparedReplay:
    """Prepare only structures observable at each input row; no native Position."""
    settings = strategy_config(config)
    if frame.empty:
        raise ValueError("结构准备需要至少一根行情")
    symbol = str(frame.iloc[0]["symbol"])
    if frame["symbol"].astype(str).ne(symbol).any():
        raise ValueError("结构准备不能混合不同股票行情")
    session = CzscSession(symbol, settings, len(frame), collect_confirmations=False, create_position=False)
    directions = np.zeros(len(frame), dtype=np.int8)
    latest = np.full(len(frame), np.nan, dtype=np.float64)
    previous = np.full(len(frame), np.nan, dtype=np.float64)
    for index, row in enumerate(frame.itertuples(index=False)):
        direction, week_last, week_previous = session.advance_structure(row)
        directions[index] = {"向上": 1, "向下": -1}.get(direction, 0)
        if week_last is not None:
            latest[index], previous[index] = week_last, week_previous
    return PreparedReplay(frame, symbol, stable_hash(settings), directions, latest, previous,
                          frame["id"].to_numpy(dtype=np.int64, copy=True))


@lru_cache(maxsize=1)
def _torch_runtime():
    try:
        import torch
        return torch, None
    except Exception as error:
        return None, f"PyTorch 不可用：{type(error).__name__}: {error}"


def compute_status(device: str | None = None) -> dict[str, Any]:
    requested = requested_torch_device(device)
    # CPU replay is called in spawned workers; only the coordinator probes Torch.
    torch, import_error = (None, None) if requested == "cpu" else _torch_runtime()
    devices: list[dict[str, Any]] = []
    selected: str | None = "cpu"
    reason = import_error
    available = requested in {"auto", "cpu"}
    mps_available, mps_built = False, False
    if torch is not None:
        try:
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(index)
                    devices.append({"type": "cuda", "device": f"cuda:{index}", "index": index, "name": properties.name,
                                    "total_memory_bytes": properties.total_memory,
                                    "capability": [properties.major, properties.minor]})
            mps_available = is_mps_available(torch)
            backend = getattr(getattr(torch, "backends", None), "mps", None)
            mps_built = bool(backend is not None and backend.is_built())
            if mps_available:
                devices.append({"type": "mps", "device": "mps", "index": None, "name": "Apple MPS",
                                "total_memory_bytes": None, "capability": None})
            selected = select_torch_device(torch, requested)
            available = True
            reason = None
        except Exception as error:
            reason = f"计算设备检测失败：{type(error).__name__}: {error}"
            selected, available = ("cpu", True) if requested == "auto" else (None, False)
    elif requested not in {"auto", "cpu"}:
        selected = None
    signal_device = "cpu" if selected == "mps" else selected
    return {"available": available, "requested_device": requested, "selected_device": selected,
            "python_executable": sys.executable,
            "torch_available": torch is not None if requested != "cpu" else None, "torch_error": import_error,
            "torch_version": str(torch.__version__) if torch is not None else None,
            "cuda_version": str(torch.version.cuda) if torch is not None and torch.version.cuda else None,
            "mps_available": mps_available, "mps_built": mps_built,
            "devices": devices, "fallback_reason": reason,
            "stages": [{"stage": "structure", "device": "cpu", "description": "冻结 CZSC 逐根结构和高周期闭合"},
                       {"stage": "batched_signals", "device": signal_device, "description": "float64 闭合周价格比较与逐日预热资格",
                        "reason": "MPS 结构价格比较保留 CPU float64 精度" if selected == "mps" else None},
                       {"stage": "model", "device": selected, "description": "float32 MLP 训练和推理（有模型工作时）"},
                       {"stage": "position_and_broker", "device": "cpu", "description": "原生 Position 时序与实际 Broker/Ledger"}]}


def _numpy_signals(latest: np.ndarray, previous: np.ndarray, ids: np.ndarray, warmup: int):
    directions = np.where(latest > previous, 1, np.where(latest < previous, -1, 0)).astype(np.int8)
    return directions, ids + 1 >= warmup


def _cuda_signals(latest: np.ndarray, previous: np.ndarray, ids: np.ndarray, warmup: int, device: str):
    torch, _ = _torch_runtime()
    with torch.cuda.device(device):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        left = torch.as_tensor(latest, dtype=torch.float64, device=device)
        right = torch.as_tensor(previous, dtype=torch.float64, device=device)
        bar_ids = torch.as_tensor(ids, dtype=torch.int64, device=device)
        ones = torch.ones_like(bar_ids, dtype=torch.int8)
        directions = torch.where(left > right, ones, torch.where(left < right, -ones, torch.zeros_like(ones)))
        eligible = bar_ids + 1 >= warmup
        end.record()
        end.synchronize()
        elapsed = float(begin.elapsed_time(end))
        return directions.cpu().numpy(), eligible.cpu().numpy(), elapsed


def replay_prepared_batch(prepared_list: list[PreparedReplay], config: dict[str, Any],
                          device: str | None = None) -> tuple[list[IntentSession], dict[str, Any]]:
    """Batch signals on the chosen device, then feed them to native Position in order."""
    settings = strategy_config(config)
    config_hash = stable_hash(settings)
    for prepared in prepared_list:
        if prepared.config_hash != config_hash:
            raise ValueError("准备结果的配置哈希与当前策略不一致")
        if not len(prepared.frame) or any(len(values) != len(prepared.frame) for values in
                                        (prepared.daily_directions, prepared.weekly_latest, prepared.weekly_previous, prepared.bar_ids)):
            raise ValueError("准备结果的逐日信号与行情长度不一致")
        if prepared.frame["symbol"].astype(str).ne(prepared.symbol).any():
            raise ValueError("准备结果的股票与原始行情不一致")
        if not np.array_equal(prepared.bar_ids, prepared.frame["id"].to_numpy()):
            raise ValueError("准备结果的逐日编号与原始行情不一致")
    info = compute_status(device)
    if not info["available"]:
        raise RuntimeError(info["fallback_reason"])
    info["stages"] = [stage for stage in info["stages"] if stage.get("stage") != "model"]
    bars = sum(len(prepared.frame) for prepared in prepared_list)
    info.update(backend="numpy", actual_device="cpu", cuda_work=False, mps_work=False, gpu_work=False,
                cuda_batches=0, batch_size=len(prepared_list), bars=bars,
                signal_elements=bars * 2, tensor_operations=["closed_week_float64_comparison", "warmup_int64_comparison"],
                tensor_seconds=0.0, cuda_elapsed_ms=None, position_seconds=0.0)
    if not prepared_list:
        return [], info
    latest = np.concatenate([prepared.weekly_latest for prepared in prepared_list])
    previous = np.concatenate([prepared.weekly_previous for prepared in prepared_list])
    ids = np.concatenate([prepared.bar_ids for prepared in prepared_list])
    tensor_start = time.perf_counter()
    if str(info["selected_device"]).startswith("cuda:"):
        try:
            week_directions, eligible, elapsed = _cuda_signals(latest, previous, ids, int(settings["warmup_bars"]), info["selected_device"])
            info.update(backend="pytorch", actual_device=info["selected_device"], cuda_work=True, gpu_work=True,
                        cuda_batches=1, cuda_elapsed_ms=elapsed)
        except Exception as error:
            if info["requested_device"] != "auto":
                raise RuntimeError(f"CUDA 批量信号失败，未回退 CPU：{error}") from error
            info.update(selected_device="cpu", fallback_reason=f"CUDA 批量信号失败，auto 回退 CPU：{error}")
            info["stages"][1]["device"] = "cpu"
            week_directions, eligible = _numpy_signals(latest, previous, ids, int(settings["warmup_bars"]))
    else:
        week_directions, eligible = _numpy_signals(latest, previous, ids, int(settings["warmup_bars"]))
    info["tensor_seconds"] = time.perf_counter() - tensor_start
    position_start = time.perf_counter()
    sessions = []
    offset = 0
    direction_names = {-1: "向下", 0: "其他", 1: "向上"}
    for prepared in prepared_list:
        session = IntentSession(prepared.symbol, settings)
        for index, row in enumerate(prepared.frame.itertuples(index=False)):
            session.apply_signals(row, direction_names[int(prepared.daily_directions[index])],
                                  direction_names[int(week_directions[offset + index])], bool(eligible[offset + index]))
        sessions.append(session)
        offset += len(prepared.frame)
    info["position_seconds"] = time.perf_counter() - position_start
    return sessions, info
