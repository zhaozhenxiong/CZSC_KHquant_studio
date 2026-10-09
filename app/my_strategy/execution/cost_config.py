"""Versioned, fail-closed loading for trading cost model configs.

Precedence in :func:`cost_model_from_backtest_section`:

1. inline ``backtest.cost_model`` mapping in the consumer YAML;
2. ``KHQUANT_COST_MODEL_CONFIG`` env var (path) — always wins over the
   consumer YAML's ``backtest.cost_model_config`` pointer;
3. ``backtest.cost_model_config`` path in the consumer YAML;
4. the canonical tiered default (:func:`default_cost_model_config`).

Legacy scalar keys (``commission``/``stamp_tax``/``slippage``/
``min_commission``) without a versioned resolution are rejected — they used
to imply ``flat_v1`` and silently keeping that would mis-price every run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from my_strategy.execution.cost_model import (
    AMOUNT_MA20_MIN_PERIODS,
    AMOUNT_MA20_WINDOW,
    FLAT_COST_VERSION,
    STRESSABLE_FIELDS,
    CostModel,
    ImpactTier,
)

COST_MODEL_CONFIG_SCHEMA_ID = "cost-model-config-v1"
COST_MODEL_CONFIG_ENV = "KHQUANT_COST_MODEL_CONFIG"

_MY_STRATEGY_ROOT = Path(__file__).resolve().parents[1]
_APP_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COST_MODEL_CONFIG_PATH = _MY_STRATEGY_ROOT / "configs" / "cost" / "tiered_cost_v1.yaml"
FLAT_COST_MODEL_CONFIG_PATH = _MY_STRATEGY_ROOT / "configs" / "cost" / "flat_v1.yaml"

_COST_MODEL_KEYS = {
    "version",
    "commission",
    "stamp_tax",
    "slippage",
    "min_commission",
    "missing_amount_policy",
    "impact_tiers",
}



def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = _APP_ROOT / path
    return path.resolve()


def cost_model_from_dict(raw: Any, *, source: str) -> CostModel:
    """Build a CostModel from a strict mapping (fail-closed on unknown/missing keys)."""
    if not isinstance(raw, dict):
        raise ValueError(f"cost model config must be a mapping: {source}")
    unknown = sorted(set(raw) - _COST_MODEL_KEYS)
    if unknown:
        raise ValueError(f"unknown cost model config keys in {source}: {unknown}")
    missing = sorted(_COST_MODEL_KEYS - set(raw))
    if missing:
        raise ValueError(f"missing cost model config keys in {source}: {missing}")
    tiers_raw = raw["impact_tiers"]
    if not isinstance(tiers_raw, (list, tuple)):
        raise ValueError(f"impact_tiers must be a list in {source}")
    tiers: list[ImpactTier] = []
    for index, tier_raw in enumerate(tiers_raw):
        if not isinstance(tier_raw, dict):
            raise ValueError(f"impact_tiers[{index}] must be a mapping in {source}")
        unknown_tier = sorted(set(tier_raw) - {"max_amount_ma20", "impact_bp"})
        if unknown_tier:
            raise ValueError(f"unknown impact_tiers[{index}] keys in {source}: {unknown_tier}")
        missing_tier = sorted({"max_amount_ma20", "impact_bp"} - set(tier_raw))
        if missing_tier:
            raise ValueError(f"missing impact_tiers[{index}] keys in {source}: {missing_tier}")
        max_amount = tier_raw["max_amount_ma20"]
        tiers.append(
            ImpactTier(
                max_amount_ma20=None if max_amount is None else float(max_amount),
                impact_bp=float(tier_raw["impact_bp"]),
            )
        )
    return CostModel(
        commission=float(raw["commission"]),
        stamp_tax=float(raw["stamp_tax"]),
        slippage=float(raw["slippage"]),
        min_commission=float(raw["min_commission"]),
        version=str(raw["version"]),
        impact_tiers=tuple(tiers),
        missing_amount_policy=str(raw["missing_amount_policy"]),
    )


def load_cost_model_config(path: str | Path) -> CostModel:
    """Load a schema-validated cost model config YAML (fail-closed schema_id)."""
    config_path = _resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"cost model config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"cost model config must be a mapping: {config_path}")
    if raw.get("schema_id") != COST_MODEL_CONFIG_SCHEMA_ID:
        raise ValueError(f"config schema_id must be {COST_MODEL_CONFIG_SCHEMA_ID!r}: {config_path}")
    payload = {key: value for key, value in raw.items() if key != "schema_id"}
    return cost_model_from_dict(payload, source=str(config_path))


def default_cost_model_config() -> CostModel:
    """Load the canonical tiered config; ``KHQUANT_COST_MODEL_CONFIG`` overrides with a path."""
    override = os.environ.get(COST_MODEL_CONFIG_ENV, "").strip()
    return load_cost_model_config(override or DEFAULT_COST_MODEL_CONFIG_PATH)


def cost_model_from_backtest_section(bt: dict[str, Any] | None) -> CostModel:
    """Resolve the effective CostModel for a consumer YAML's ``backtest`` section."""
    bt = bt or {}
    inline = bt.get("cost_model")
    if inline is not None:
        return cost_model_from_dict(inline, source="backtest.cost_model")
    override = os.environ.get(COST_MODEL_CONFIG_ENV, "").strip()
    if override:
        return load_cost_model_config(override)
    pointer = str(bt.get("cost_model_config") or "").strip()
    if pointer:
        return load_cost_model_config(pointer)
    legacy = sorted(key for key in ("commission", "stamp_tax", "slippage", "min_commission") if key in bt)
    if legacy:
        raise ValueError(
            f"legacy cost scalar keys {legacy} no longer resolve to flat_v1; "
            "migrate to an inline backtest.cost_model block or a "
            f"backtest.cost_model_config pointer (e.g. {DEFAULT_COST_MODEL_CONFIG_PATH.name}), "
            f"or pin {COST_MODEL_CONFIG_ENV}=.../flat_v1.yaml for historical reproduction"
        )
    return default_cost_model_config()
