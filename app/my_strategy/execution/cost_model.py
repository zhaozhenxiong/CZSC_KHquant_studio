"""Trading cost model.

Two versioned cost semantics:

- ``tiered_v1`` (default): the per-side slippage is resolved from liquidity
  impact tiers bucketed by ``amount_ma20`` (20-day mean of daily traded CNY
  amount) with absolute thresholds. Callers pass the bar's ``amount_ma20`` to
  :meth:`CostModel.execution_price`; tier resolution and the missing-data
  policy live inside this module so every caller shares one implementation.
- ``flat_v1``: fixed per-side ``slippage`` on price, identical to the
  historical model. Retained only as an explicit reproduction baseline
  (``version="flat_v1"`` or ``KHQUANT_COST_MODEL_CONFIG=.../flat_v1.yaml``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from my_strategy.execution.order import OrderSide

FLAT_COST_VERSION = "flat_v1"
TIERED_COST_VERSION = "tiered_v1"
COST_VERSIONS = (FLAT_COST_VERSION, TIERED_COST_VERSION)

MissingAmountPolicy = Literal["highest_tier", "flat"]
MISSING_AMOUNT_POLICIES = ("highest_tier", "flat")

STRESSABLE_FIELDS = ("commission", "slippage", "impact_tiers")

AMOUNT_MA20_WINDOW = 20
AMOUNT_MA20_MIN_PERIODS = 5


@dataclass(frozen=True)
class ImpactTier:
    """Per-side impact cost for one liquidity bucket.

    ``max_amount_ma20`` is the inclusive absolute-CNY upper bound of the
    bucket; ``None`` marks the open-ended most-liquid tier (last entry only).
    """

    max_amount_ma20: float | None
    impact_bp: float

    def __post_init__(self) -> None:
        if self.max_amount_ma20 is not None and (
            not math.isfinite(self.max_amount_ma20) or self.max_amount_ma20 <= 0
        ):
            raise ValueError("impact tier max_amount_ma20 must be a positive finite number or None")
        if not math.isfinite(self.impact_bp) or self.impact_bp <= 0:
            raise ValueError("impact tier impact_bp must be a positive finite number")


# Frozen tier table mirroring configs/cost/tiered_cost_v1.yaml.  The yaml
# remains the governance artifact; a lock test asserts the two stay identical.
DEFAULT_IMPACT_TIERS: tuple[ImpactTier, ...] = (
    ImpactTier(max_amount_ma20=20_000_000.0, impact_bp=25.0),
    ImpactTier(max_amount_ma20=50_000_000.0, impact_bp=15.0),
    ImpactTier(max_amount_ma20=200_000_000.0, impact_bp=8.0),
    ImpactTier(max_amount_ma20=1_000_000_000.0, impact_bp=5.0),
    ImpactTier(max_amount_ma20=None, impact_bp=3.0),
)


@dataclass(frozen=True)
class CostModel:
    commission: float = 0.0003
    stamp_tax: float = 0.0005
    slippage: float = 0.0005
    min_commission: float = 5.0
    version: str = TIERED_COST_VERSION
    impact_tiers: tuple[ImpactTier, ...] = DEFAULT_IMPACT_TIERS
    missing_amount_policy: str = "highest_tier"

    def __post_init__(self) -> None:
        object.__setattr__(self, "impact_tiers", tuple(self.impact_tiers))
        if self.version not in COST_VERSIONS:
            raise ValueError(f"unknown cost model version: {self.version!r}")
        if self.missing_amount_policy not in MISSING_AMOUNT_POLICIES:
            raise ValueError(
                f"missing_amount_policy must be one of {MISSING_AMOUNT_POLICIES}: {self.missing_amount_policy!r}"
            )
        if self.version == TIERED_COST_VERSION and not self.impact_tiers:
            raise ValueError("tiered_v1 requires a non-empty impact_tiers table")
        if self.version == FLAT_COST_VERSION and self.impact_tiers:
            raise ValueError("flat_v1 must not define impact_tiers")
        previous = 0.0
        for index, tier in enumerate(self.impact_tiers):
            if tier.max_amount_ma20 is None:
                if index != len(self.impact_tiers) - 1:
                    raise ValueError("only the last impact tier may be open-ended (max_amount_ma20=None)")
            else:
                if tier.max_amount_ma20 <= previous:
                    raise ValueError("impact tier thresholds must be strictly increasing")
                previous = tier.max_amount_ma20

    def resolve_tier_index(self, amount_ma20: float | None = None) -> int:
        """Return the impact tier index for ``amount_ma20`` (CNY), or -1 for flat slippage."""
        if self.version != TIERED_COST_VERSION:
            return -1
        if amount_ma20 is None or not math.isfinite(float(amount_ma20)) or float(amount_ma20) <= 0:
            # Missing/invalid liquidity data: charge the least-liquid tier
            # (conservative) or fall back to flat slippage, per policy.
            return 0 if self.missing_amount_policy == "highest_tier" else -1
        value = float(amount_ma20)
        for index, tier in enumerate(self.impact_tiers):
            if tier.max_amount_ma20 is None or value <= tier.max_amount_ma20:
                return index
        return len(self.impact_tiers) - 1

    def resolve_slippage(self, amount_ma20: float | None = None) -> float:
        """Per-side slippage fraction for a trade in a stock with this ``amount_ma20``."""
        tier_index = self.resolve_tier_index(amount_ma20)
        if tier_index < 0:
            return float(self.slippage)
        return float(self.impact_tiers[tier_index].impact_bp) / 10000.0

    def execution_price(
        self,
        side: OrderSide,
        reference_price: float,
        *,
        amount_ma20: float | None = None,
    ) -> float:
        slippage = self.resolve_slippage(amount_ma20)
        if side == "BUY":
            return float(reference_price) * (1 + slippage)
        return float(reference_price) * (1 - slippage)

    def fees(self, side: OrderSide, notional: float) -> float:
        commission_fee = max(float(notional) * self.commission, self.min_commission)
        tax = float(notional) * self.stamp_tax if side == "SELL" else 0.0
        return commission_fee + tax

    def scaled(self, multiplier: float, applies_to: tuple[str, ...] = STRESSABLE_FIELDS) -> "CostModel":
        """Return a copy with the stressable fields multiplied; stamp_tax is never scaled."""
        multiplier = float(multiplier)
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(f"cost stress multiplier must be a positive finite number: {multiplier}")
        unknown = sorted(set(applies_to) - set(STRESSABLE_FIELDS))
        if unknown:
            raise ValueError(f"unknown stress fields: {unknown} (allowed: {STRESSABLE_FIELDS})")
        tiers = self.impact_tiers
        if "impact_tiers" in applies_to and tiers:
            tiers = tuple(ImpactTier(tier.max_amount_ma20, tier.impact_bp * multiplier) for tier in tiers)
        return CostModel(
            commission=self.commission * multiplier if "commission" in applies_to else self.commission,
            stamp_tax=self.stamp_tax,
            slippage=self.slippage * multiplier if "slippage" in applies_to else self.slippage,
            min_commission=self.min_commission,
            version=self.version,
            impact_tiers=tiers,
            missing_amount_policy=self.missing_amount_policy,
        )

    def as_config_dict(self) -> dict[str, Any]:
        """Serializable resolved cost model (used for run/cell audit records)."""
        return {
            "version": self.version,
            "commission": float(self.commission),
            "stamp_tax": float(self.stamp_tax),
            "slippage": float(self.slippage),
            "min_commission": float(self.min_commission),
            "missing_amount_policy": self.missing_amount_policy,
            "impact_tiers": [
                {"max_amount_ma20": tier.max_amount_ma20, "impact_bp": float(tier.impact_bp)}
                for tier in self.impact_tiers
            ],
        }


def compute_amount_ma20(
    amount: pd.Series,
    window: int = AMOUNT_MA20_WINDOW,
    min_periods: int = AMOUNT_MA20_MIN_PERIODS,
) -> pd.Series:
    """Trailing mean of daily traded CNY amount through the current bar (unshifted).

    The value on bar T uses data through close T for T+1 executions.
    """
    if window < 1:
        raise ValueError("amount_ma20 window must be >= 1")
    if not 1 <= min_periods <= window:
        raise ValueError("amount_ma20 min_periods must be within [1, window]")
    series = pd.to_numeric(amount, errors="coerce")
    return series.rolling(window, min_periods=min_periods).mean()
