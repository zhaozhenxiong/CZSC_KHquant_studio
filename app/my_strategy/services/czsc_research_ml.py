"""Isolated CZSC research labels, chronological MLPs and date-block uncertainty.

Importing this module does not import Torch: CPU preparation workers may build
labels; the coordinating parent alone trains and predicts on its requested GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from my_strategy.adapters.czsc_adapter import legacy_fuyao_source
from my_strategy.execution.cost_model import CostModel
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import _guard, _tick
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES


LABEL_VERSION = "czsc_fixed_horizon_broker_v1"
MODEL_VERSION = "czsc_research_mlp_v1"
ROLLING_WINDOWS = (
    {"train_end": "2025-09-30", "validation_start": "2025-10-01", "validation_end": "2025-12-31", "test_start": "2026-01-01", "test_end": "2026-03-31"},
    {"train_end": "2025-12-31", "validation_start": "2026-01-01", "validation_end": "2026-03-31", "test_start": "2026-04-01", "test_end": "2026-06-30"},
    {"train_end": "2026-03-31", "validation_start": "2026-04-01", "validation_end": "2026-06-30", "test_start": "2026-07-01", "test_end": "2026-09-30"},
)


def _date(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _execution(config: dict[str, Any] | None) -> tuple[dict[str, Any], CostModel]:
    settings = strategy_config(config)["execution"]
    if int(settings["lot_size"]) != 100 or not 0 < float(settings["max_weight"]) <= 1 or not 0 < float(settings["max_volume_participation"]) <= 1:
        raise ValueError("invalid lot, weight or prior-volume participation")
    if not 0 < float(settings["mainboard_conservative_limit"]) <= 0.05 or int(settings["seasoned_bars"]) < 60:
        raise ValueError("invalid conservative price or seasoned-bars guard")
    if any(not math.isfinite(float(settings[key])) or float(settings[key]) < 0 for key in ("commission", "stamp_tax", "min_commission", "slippage")) or float(settings["slippage"]) >= 1:
        raise ValueError("invalid fees or slippage")
    cost = CostModel(commission=float(settings["commission"]), stamp_tax=float(settings["stamp_tax"]),
                     min_commission=float(settings["min_commission"]), slippage=float(settings["slippage"]), version="flat_v1", impact_tiers=())
    return settings, cost


def build_labels(frame: pd.DataFrame, market_dates: Sequence[Any] | None = None,
                 horizon: int = 10, initial_cash: float = 100000,
                 config: dict[str, Any] | None = None) -> pd.DataFrame:
    """One independent fixed-capital round trip per signal, aligned to input rows.

    A verified common trading calendar is required. Signal t buys on t+1 and
    attempts full exit on t+horizon+1, using the same guards, ticks, fees and
    prior-volume lot sizing as execute_decisions. A missing scheduled bar,
    rejected entry, partial/unavailable exit or immature tail stays unavailable.
    No later observable bar is substituted and there is no retry in the label.
    """
    if not isinstance(horizon, int) or horizon < 1 or not math.isfinite(float(initial_cash)) or initial_cash <= 0:
        raise ValueError("horizon and initial_cash must be positive")
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "amount", "source", "has_trade_price"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"label input missing columns: {missing}")
    settings, cost = _execution(config)
    dates = frame["date"].map(_date)
    records = [{"date": day, "symbol": str(symbol), "label": np.nan, "label_end": None,
                "label_reason": "calendar_unverified", "net_return": np.nan, "label_available": False,
                "entry_date": None, "exit_date": None, "entry_shares": 0, "fees": np.nan}
               for day, symbol in zip(dates, frame["symbol"])]
    if market_dates is None or frame.empty:
        return pd.DataFrame(records, index=frame.index)
    calendar = [_date(day) for day in market_dates]
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("market_dates must be unique, ascending verified sessions")
    calendar_index = {day: index for index, day in enumerate(calendar)}
    as_of = max(dates)
    working = frame.copy()
    working["date"] = dates
    working["_position"] = np.arange(len(frame))
    lot = int(settings["lot_size"])
    for symbol, group in working.groupby("symbol", sort=False):
        if group["date"].duplicated().any() or not group["date"].is_monotonic_increasing:
            raise ValueError("each symbol must have unique ascending dates")
        rows = list(group.itertuples(index=False, name="LabelBar"))
        positions = group["_position"].to_numpy(dtype=int)
        lookup = {row.date: index for index, row in enumerate(rows)}
        valid_price = [getattr(row, "has_trade_price") == 1 and not legacy_fuyao_source(row.source)
                       and all(math.isfinite(float(getattr(row, key))) and float(getattr(row, key)) > 0 for key in ("open", "high", "low", "close"))
                       and all(math.isfinite(float(getattr(row, key))) and float(getattr(row, key)) >= 0 for key in ("volume", "amount")) for row in rows]
        verified_source = [str(row.source) in VERIFIED_SOURCES for row in rows]
        seasoned = np.r_[0, np.cumsum([valid and row.volume > 0 and row.amount > 0 for valid, row in zip(valid_price, rows)])]
        for row_index, row in enumerate(rows):
            record = records[positions[row_index]]
            session_index = calendar_index.get(row.date)
            if session_index is None:
                record["label_reason"] = "signal_not_in_calendar"
                continue
            if session_index + 1 < len(calendar):
                record["entry_date"] = calendar[session_index + 1]
            if session_index + horizon + 1 >= len(calendar):
                record["label_reason"] = "label_immature_calendar_tail"
                continue
            entry_day, exit_day = calendar[session_index + 1], calendar[session_index + horizon + 1]
            record.update({"entry_date": entry_day, "exit_date": exit_day, "label_end": exit_day})
            if exit_day > as_of:
                record["label_reason"] = "label_immature"
                continue
            if getattr(row, "input_eligible", True) is not True and getattr(row, "input_eligible", True) != 1:
                record["label_reason"] = "input_ineligible"
                continue
            if entry_day not in lookup or exit_day not in lookup:
                record["label_reason"] = "missing_entry_bar" if entry_day not in lookup else "missing_exit_bar"
                continue
            expected = calendar[session_index:session_index + horizon + 2]
            if any(day not in lookup for day in expected):
                record["label_reason"] = "gap_in_holding_window"
                continue
            interval = [lookup[day] for day in expected]
            if any(not verified_source[index] for index in interval):
                record["label_reason"] = "unverified_research_source"
                continue
            if any(not valid_price[index] for index in interval):
                record["label_reason"] = "unverified_holding_window"
                continue
            entry_index, exit_index = lookup[entry_day], lookup[exit_day]
            entry, exit_bar, prior_exit = rows[entry_index], rows[exit_index], rows[lookup[calendar[session_index + horizon]]]
            rejection = _guard(str(symbol), entry, row, "BUY", int(seasoned[entry_index]), settings)
            if rejection:
                record["label_reason"] = "entry_" + rejection
                continue
            entry_price = _tick(cost.execution_price("BUY", float(entry.open)))
            budget = float(initial_cash) * float(settings["max_weight"])
            capacity = int(float(row.volume) * float(settings["max_volume_participation"])) // lot * lot
            shares = min(int(budget / entry_price) // lot * lot, capacity)
            while shares > 0 and shares * entry_price + cost.fees("BUY", shares * entry_price) > budget:
                shares -= lot
            if shares <= 0:
                record["label_reason"] = "entry_insufficient_cash_or_prior_liquidity"
                continue
            record["entry_shares"] = shares
            rejection = _guard(str(symbol), exit_bar, prior_exit, "SELL", int(seasoned[exit_index]), settings)
            exit_capacity = int(float(prior_exit.volume) * float(settings["max_volume_participation"])) // lot * lot
            if rejection or exit_capacity < shares:
                record["label_reason"] = "exit_" + (rejection or "insufficient_prior_liquidity_for_full_exit")
                continue
            exit_price = _tick(cost.execution_price("SELL", float(exit_bar.open)))
            fees = cost.fees("BUY", shares * entry_price) + cost.fees("SELL", shares * exit_price)
            net_return = (shares * (exit_price - entry_price) - fees) / float(initial_cash)
            record.update({"label": float(net_return > 0), "net_return": net_return, "fees": fees,
                           "label_available": True, "label_reason": "available"})
    result = pd.DataFrame(records, index=frame.index)
    result.attrs.update({"label_version": LABEL_VERSION, "horizon": horizon, "initial_cash": initial_cash,
                         "calendar_sha256": _hash(calendar), "execution": settings, "cost_model": cost.as_config_dict(),
                         "verified_sources": sorted(VERIFIED_SOURCES),
                         "unavailable_counts": result.loc[~result["label_available"], "label_reason"].value_counts().to_dict(),
                         "selection_bias": "Training conditions on feasible entry and full scheduled exit; ledger evaluation must retain the complete frozen universe."})
    return result


def _device(name: str | None):
    import torch
    from my_strategy.core.device import select_torch_device, synchronize_torch_device
    selected = torch.device(select_torch_device(torch, name))
    # Actual float32 work checks the backend beyond hardware discovery.
    torch.ones(1, dtype=torch.float32, device=selected).add_(1)
    synchronize_torch_device(torch, selected)
    return selected


def _network(feature_count: int, hidden_sizes: Sequence[int]):
    import torch
    if len(hidden_sizes) == 0:
        return torch.nn.Sequential(torch.nn.Linear(feature_count, 1))
    if len(hidden_sizes) != 2 or any(not isinstance(size, int) or size < 1 for size in hidden_sizes):
        raise ValueError("MLP requires two positive hidden sizes")
    return torch.nn.Sequential(torch.nn.Linear(feature_count, hidden_sizes[0]), torch.nn.ReLU(),
                               torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]), torch.nn.ReLU(),
                               torch.nn.Linear(hidden_sizes[1], 1))


def _features(dataset: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns or len(set(columns)) != len(columns) or any(column not in dataset for column in columns):
        raise ValueError("feature schema is empty, duplicated or missing from dataset")
    values = dataset[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _fit_preprocess(values: np.ndarray) -> dict[str, Any]:
    all_missing = np.isnan(values).all(axis=0)
    medians = np.array([0.0 if absent else np.nanmedian(values[:, index]) for index, absent in enumerate(all_missing)])
    filled = np.where(np.isnan(values), medians, values)
    means, scales = filled.mean(axis=0), filled.std(axis=0)
    scales[scales < 1e-8] = 1.0
    return {"median": medians.tolist(), "mean": means.tolist(), "scale": scales.tolist(),
            "all_missing": all_missing.tolist(), "clip_standard_deviations": 8.0, "fit_scope": "purged_training_only"}


def _transform(values: np.ndarray, preprocess: dict[str, Any]) -> np.ndarray:
    filled = np.where(np.isnan(values), np.array(preprocess["median"]), values)
    return np.clip((filled - np.array(preprocess["mean"])) / np.array(preprocess["scale"]), -8, 8).astype(np.float32)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    p = np.clip(probabilities.astype(float), 1e-7, 1 - 1e-7)
    positive, negative = int((labels == 1).sum()), int((labels == 0).sum())
    ranks = pd.Series(p).rank(method="average").to_numpy()
    auc = float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)) if positive and negative else None
    bins = []
    for lower in np.arange(0.0, 1.0, 0.1):
        mask = (p >= lower) & (p < lower + 0.1 + (1e-8 if lower > 0.89 else 0))
        if mask.any():
            bins.append({"lower": round(float(lower), 1), "rows": int(mask.sum()), "predicted": float(p[mask].mean()), "observed": float(labels[mask].mean())})
    return {"rows": len(labels), "positive_rate": float(labels.mean()), "brier": float(np.mean((p - labels) ** 2)),
            "log_loss": float(-np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p))), "auc": auc, "calibration_bins": bins}


def train_model(dataset: pd.DataFrame, feature_columns: Sequence[str], output_dir: str | Path,
                train_end: str, validation_start: str, validation_end: str,
                device: str = "auto", seed: int = 42, *, epochs: int = 30,
                batch_size: int = 2048, learning_rate: float = 0.001,
                hidden_sizes: Sequence[int] = (32, 16), min_train_rows: int = 100,
                min_validation_rows: int = 30, progress: Callable[[int, int], None] | None = None,
                check_cancel: Callable[[], None] | None = None) -> dict[str, Any]:
    """Fixed architecture/epochs; training-only preprocessing, validation-only calibration.

    train_end is the last available training label date, not just signal date.
    Validation labels mature by validation_end. The checkpoint is consequently
    available at validation_end close and cannot predict earlier replay rows.
    Test rows are never used for fitting, calibration or hyperparameter choice.
    progress receives (completed_epochs, total_epochs); check_cancel may raise
    the caller's cancellation exception, which propagates without a checkpoint.
    """
    import torch
    from my_strategy.core.tz import local_now
    training_started_at = local_now().isoformat()
    check = check_cancel or (lambda: None)
    check()
    train_end, validation_start, validation_end = map(_date, (train_end, validation_start, validation_end))
    if not train_end < validation_start <= validation_end:
        raise ValueError("training and validation periods must be chronological and disjoint")
    if epochs < 1 or batch_size < 1 or not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise ValueError("invalid training hyperparameters")
    if not {"date", "label", "label_end", "label_available"}.issubset(dataset.columns):
        raise ValueError("dataset missing chronological label metadata")
    feature_version = dataset.attrs.get("feature_version")
    feature_schema_hash = dataset.attrs.get("schema_hash")
    if (feature_version is None) != (feature_schema_hash is None):
        raise ValueError("feature semantic binding requires both feature_version and schema_hash")
    values = _features(dataset, feature_columns)
    dates = pd.to_datetime(dataset["date"]).dt.strftime("%Y-%m-%d")
    label_ends = pd.to_datetime(dataset["label_end"], errors="coerce").dt.strftime("%Y-%m-%d")
    labels = pd.to_numeric(dataset["label"], errors="coerce").to_numpy(dtype=float)
    eligible = dataset.get("input_eligible", pd.Series(True, index=dataset.index)).fillna(False).astype(bool).to_numpy()
    available = dataset["label_available"].fillna(False).astype(bool).to_numpy() & eligible & np.isin(labels, [0, 1]) & label_ends.notna().to_numpy()
    if (available & (label_ends <= dates).to_numpy()).any():
        raise ValueError("available label end must be later than its signal date")
    train_mask = available & (dates <= train_end).to_numpy() & (label_ends <= train_end).to_numpy() & (label_ends < validation_start).to_numpy()
    validation_mask = available & (dates >= validation_start).to_numpy() & (dates <= validation_end).to_numpy() & (label_ends <= validation_end).to_numpy()
    if int(train_mask.sum()) < min_train_rows or int(validation_mask.sum()) < min_validation_rows:
        raise ValueError(f"insufficient mature samples: train={train_mask.sum()}, validation={validation_mask.sum()}")
    if len(np.unique(labels[train_mask])) < 2 or len(np.unique(labels[validation_mask])) < 2:
        raise ValueError("training and validation each require both label classes")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from my_strategy.core.device import requested_torch_device
    requested_device = requested_torch_device(device)
    selected = _device(device)
    torch.manual_seed(seed)
    if selected.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    # Small matrix GEMMs are reproducible without TF32 approximation.
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    preprocess = _fit_preprocess(values[train_mask])
    transformed = _transform(values, preprocess)
    train_x = torch.from_numpy(transformed[train_mask])
    train_y = torch.from_numpy(labels[train_mask].astype(np.float32)).reshape(-1, 1)
    model = _network(len(feature_columns), hidden_sizes).to(selected)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    training_batches = 0
    model.train()
    for epoch in range(epochs):
        check()
        permutation = torch.randperm(len(train_x), generator=generator)
        total = 0.0
        for start in range(0, len(train_x), batch_size):
            check()
            indices = permutation[start:start + batch_size]
            batch_x, batch_y = train_x[indices].to(selected), train_y[indices].to(selected)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            training_batches += 1
            total += float(loss.detach().item()) * len(indices)
        losses.append(total / len(train_x))
        if progress:
            progress(epoch + 1, epochs)
    model.eval()
    def logits(mask: np.ndarray) -> np.ndarray:
        chunks = []
        with torch.no_grad():
            selected_values = transformed[mask]
            for start in range(0, len(selected_values), batch_size):
                check()
                chunks.append(model(torch.from_numpy(selected_values[start:start + batch_size]).to(selected)).cpu().numpy().reshape(-1))
        return np.concatenate(chunks).astype(float)
    train_logits, validation_logits = logits(train_mask), logits(validation_mask)
    temperatures = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    def probability(z: np.ndarray, temperature: float) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z / temperature, -40, 40)))
    temperature_scores = []
    for temperature in temperatures:
        check()
        temperature_scores.append(_metrics(labels[validation_mask], probability(validation_logits, temperature))["log_loss"])
    temperature = temperatures[int(np.argmin(temperature_scores))]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists() or (destination / "model.pt").exists():
        raise FileExistsError("model output already exists; use an isolated run directory")
    checkpoint = destination / "model.pt"
    check()
    torch.save({"state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                "feature_columns": list(feature_columns), "hidden_sizes": list(hidden_sizes), "model_version": MODEL_VERSION,
                "feature_version": feature_version, "feature_schema_hash": feature_schema_hash}, checkpoint)
    schema = {"columns": list(feature_columns), "model_version": MODEL_VERSION, "hidden_sizes": list(hidden_sizes)}
    manifest = {"model_version": MODEL_VERSION, "architecture": "ma_logistic" if not hidden_sizes else "mlp",
                "label_version": dataset.attrs.get("label_version", LABEL_VERSION),
                "label_contract": dataset.attrs.get("label_contract", {}), "feature_version": feature_version,
                "feature_schema_hash": feature_schema_hash, "feature_schema_binding": "bound" if feature_version is not None else "unbound_research_fixture",
                "data_version": dataset.attrs.get("data_version"), "schema": schema, "schema_sha256": _hash(schema),
                "preprocess": preprocess, "preprocess_sha256": _hash(preprocess), "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _file_hash(checkpoint), "available_at": validation_end, "train_end": train_end,
                "validation_start": validation_start, "validation_end": validation_end,
                "train_signal_start": str(dates[train_mask].min()), "train_signal_end": str(dates[train_mask].max()),
                "train_label_end": str(label_ends[train_mask].max()), "validation_label_end": str(label_ends[validation_mask].max()),
                "seed": seed, "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda, "device": str(selected),
                "training_started_at": training_started_at, "training_completed_at": local_now().isoformat(),
                "device_name": torch.cuda.get_device_name(selected) if selected.type == "cuda" else "Apple MPS" if selected.type == "mps" else "CPU",
                "actual_cuda_training": selected.type == "cuda", "actual_mps_training": selected.type == "mps",
                "actual_gpu_training": selected.type in {"cuda", "mps"}, "requested_device": requested_device,
                "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate},
                "training_loss": losses, "training_batches": training_batches,
                "calibration": {"method": "validation_temperature_grid", "temperature": temperature,
                    "temperatures": list(temperatures), "validation_log_losses": temperature_scores, "fit_start": validation_start, "fit_end": validation_end},
                "train_metrics": _metrics(labels[train_mask], probability(train_logits, temperature)),
                "validation_metrics_uncalibrated": _metrics(labels[validation_mask], probability(validation_logits, 1)),
                "validation_metrics": _metrics(labels[validation_mask], probability(validation_logits, temperature)),
                "sample_counts": {"total": len(dataset), "unavailable_or_ineligible": int((~available).sum()), "train": int(train_mask.sum()), "validation": int(validation_mask.sum())},
                "dataset_sha256": hashlib.sha256(pd.util.hash_pandas_object(dataset.loc[train_mask | validation_mask], index=True).to_numpy().tobytes()).hexdigest(),
                "promotion_status": "shadow_pending_complete_universe_ledger_gate",
                "limitations": ["Validation calibration metrics are fitted evidence, not independent test results.",
                    "Feasible-label selection can bias training; evaluation must retain rejected trades and unexited positions.",
                    "No claim of improvement follows from classifier accuracy or model availability."]}
    manifest["manifest_sha256"] = _hash(manifest)
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def _verified_manifest(output_dir: str | Path) -> tuple[dict[str, Any], Path]:
    destination = Path(output_dir)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    integrity = dict(manifest)
    recorded = integrity.pop("manifest_sha256", None)
    if recorded != _hash(integrity) or manifest["schema_sha256"] != _hash(manifest["schema"]) or manifest["preprocess_sha256"] != _hash(manifest["preprocess"]):
        raise ValueError("model manifest/schema/preprocess integrity mismatch")
    checkpoint = destination / "model.pt"
    if _file_hash(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("model checkpoint hash mismatch")
    return manifest, checkpoint


class PredictorSession:
    """Task-local model reuse; every request rechecks immutable artifact hashes.

    The default method retains strict historical availability. Retrospective
    probabilities are exposed only by the separate, explicitly named method;
    they cannot be mistaken for an earlier available checkpoint by this API.
    Neither mode permits feature rows later than the requested information date.
    """
    def __init__(self, device: str | None = None, batch_size: int = 8192):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = device
        self.batch_size = batch_size
        self._models: dict[tuple[str, str, str], tuple[Any, Any]] = {}
        self._counts = {"rows": 0, "input_rows": 0, "batches": 0, "model_loads": 0,
                        "cache_hits": 0, "hash_checks": 0, "retrospective_rows": 0}
        self._devices: set[str] = set()
        self._cuda_rows = 0
        self._mps_rows = 0
        self._seconds = 0.
        self.last_prediction: dict[str, Any] = {}

    @property
    def diagnostics(self) -> dict[str, Any]:
        devices = sorted(self._devices)
        return {**self._counts, "device": devices[0] if len(devices) == 1 else None,
                "devices": devices, "actual_cuda_inference": self._cuda_rows > 0,
                "actual_mps_inference": self._mps_rows > 0, "actual_gpu_inference": self._cuda_rows + self._mps_rows > 0,
                "requested_device": self.device, "cuda_rows": self._cuda_rows, "mps_rows": self._mps_rows,
                "inference_seconds": self._seconds}

    def predict(self, dataset: pd.DataFrame, output_dir: str | Path, *, as_of: str | None = None) -> np.ndarray:
        return self._predict(dataset, output_dir, as_of=as_of, retrospective=False)

    def predict_retrospective(self, dataset: pd.DataFrame, output_dir: str | Path, *,
                              as_of: str | None = None) -> np.ndarray:
        """Explicit after-the-fact inspection, excluded from unbiased certification."""
        return self._predict(dataset, output_dir, as_of=as_of, retrospective=True)

    def _predict(self, dataset: pd.DataFrame, output_dir: str | Path, *,
                 as_of: str | None, retrospective: bool) -> np.ndarray:
        import torch
        started = time.perf_counter()
        manifest, checkpoint = _verified_manifest(output_dir)
        self._counts["hash_checks"] += 1
        if "date" not in dataset:
            raise ValueError("inference requires signal dates")
        dates = dataset["date"].map(_date)
        if as_of is not None and (dates > _date(as_of)).any():
            raise ValueError("model unavailable at requested replay/as_of date: future feature rows")
        if not retrospective and ((dates < manifest["available_at"]).any() or
                (as_of is not None and _date(as_of) < manifest["available_at"])):
            raise ValueError("model unavailable at requested replay/as_of date")
        columns = manifest["schema"]["columns"]
        if manifest.get("feature_schema_binding") == "bound":
            if dataset.attrs.get("feature_version") != manifest["feature_version"] or dataset.attrs.get("schema_hash") != manifest["feature_schema_hash"]:
                raise ValueError("feature semantic version/schema hash missing or mismatched")
        values = _transform(_features(dataset, columns), manifest["preprocess"])
        # Training hardware is immutable provenance, never the runtime default.
        from my_strategy.core.device import requested_torch_device
        requested_device = requested_torch_device(self.device)
        key = (str(Path(output_dir).resolve()), requested_device, manifest["manifest_sha256"])
        if key not in self._models:
            selected = _device(requested_device)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (payload["feature_columns"] != columns or payload["hidden_sizes"] != manifest["schema"]["hidden_sizes"] or payload["model_version"] != MODEL_VERSION
                    or payload.get("feature_version") != manifest.get("feature_version") or payload.get("feature_schema_hash") != manifest.get("feature_schema_hash")):
                raise ValueError("checkpoint feature schema mismatch")
            model = _network(len(columns), payload["hidden_sizes"]).to(selected)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            self._models[key] = (model, selected)
            self._counts["model_loads"] += 1
        else:
            self._counts["cache_hits"] += 1
        model, selected = self._models[key]
        self._devices.add(str(selected))
        result = np.full(len(dataset), np.nan, dtype=float)
        eligible = dataset.get("input_eligible", pd.Series(True, index=dataset.index)).fillna(False).astype(bool).to_numpy()
        indices = np.flatnonzero(eligible)
        with torch.inference_mode():
            for start in range(0, len(indices), self.batch_size):
                selected_indices = indices[start:start + self.batch_size]
                logits = model(torch.from_numpy(values[selected_indices]).to(selected)) / manifest["calibration"]["temperature"]
                result[selected_indices] = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                self._counts["batches"] += 1
        self._counts["rows"] += len(indices)
        self._counts["input_rows"] += len(dataset)
        self._counts["retrospective_rows"] += len(indices) if retrospective else 0
        self._cuda_rows += len(indices) if selected.type == "cuda" else 0
        self._mps_rows += len(indices) if selected.type == "mps" else 0
        seconds = time.perf_counter() - started
        self._seconds += seconds
        self.last_prediction = {"mode": "retrospective" if retrospective else "chronological",
            "available_at": manifest["available_at"], "device": str(selected), "rows": len(indices),
            "feature_rows_before_available": int((dates < manifest["available_at"]).sum()),
            "training_validation_overlap_rows": int(((dates >= manifest["train_signal_start"]) & (dates <= manifest["validation_end"])).sum()),
            "eligible_for_unbiased_certification": not retrospective, "inference_seconds": seconds}
        return result


def predict_model(dataset: pd.DataFrame, output_dir: str | Path, device: str | None = None,
                  *, as_of: str | None = None, batch_size: int = 8192) -> np.ndarray:
    """Verified checkpoint probabilities; unavailable inputs remain NaN."""
    return PredictorSession(device=device, batch_size=batch_size).predict(dataset, output_dir, as_of=as_of)


def predict_model_retrospective(dataset: pd.DataFrame, output_dir: str | Path, device: str | None = None,
                                *, as_of: str | None = None, batch_size: int = 8192) -> np.ndarray:
    """Explicit current-model historical inspection, never unbiased evaluation."""
    return PredictorSession(device=device, batch_size=batch_size).predict_retrospective(dataset, output_dir, as_of=as_of)


infer_model = predict_model


def date_block_bootstrap(frame: pd.DataFrame, value_column: str, *, date_column: str = "date",
                         block_length: int = 20, repetitions: int = 1000, seed: int = 42,
                         confidence: float = 0.95, dependence_horizon: int = 10) -> dict[str, Any]:
    """CI of the equal-date mean, keeping all same-date securities together.

    For a paired comparison pass per-stock/date differences, not separately
    resampled strategies. Contiguous moving date blocks preserve dependence.
    This utility does not calculate or replace actual ledger return/drawdown.
    """
    if block_length < dependence_horizon + 1 or repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("block must cover holding dependence; invalid repetitions/confidence")
    data = frame[[date_column, value_column]].copy()
    data[date_column] = data[date_column].map(_date)
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data.loc[~np.isfinite(data[value_column]), value_column] = np.nan
    daily = data.groupby(date_column)[value_column].mean().dropna().sort_index()
    values = daily.to_numpy(dtype=float)
    if len(values) < 2 * block_length:
        raise ValueError("insufficient dates for at least two full dependence blocks")
    generator = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for index in range(repetitions):
        starts = generator.integers(0, len(values) - block_length + 1, size=math.ceil(len(values) / block_length))
        sample = np.concatenate([values[start:start + block_length] for start in starts])[:len(values)]
        means[index] = sample.mean()
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(means, [tail, 1 - tail])
    return {"estimand": "equal-date mean of same-date cross-sectional means", "mean": float(values.mean()),
            "lower": float(lower), "upper": float(upper), "confidence": confidence, "dates": len(values), "rows": len(frame),
            "date_start": daily.index[0], "date_end": daily.index[-1], "block_length": block_length, "repetitions": repetitions, "seed": seed}


def bootstrap_sensitivity(frame: pd.DataFrame, value_column: str, *, block_lengths: Sequence[int] = (11, 20, 40),
                          **kwargs: Any) -> list[dict[str, Any]]:
    return [date_block_bootstrap(frame, value_column, block_length=length, **kwargs) for length in block_lengths]
