"""Order schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OrderSide = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class Order:
    date: str
    symbol: str
    side: OrderSide
    shares: int
    price: float
    reason: str = ""
    target_weight: float = 0.0
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: float | None = None
    signal_date: str = ""
    # Audit trail for the tiered cost model: the per-side slippage fraction
    # actually applied to ``price`` and the impact tier it came from (-1=flat).
    applied_slippage: float | None = None
    cost_tier: int = -1
