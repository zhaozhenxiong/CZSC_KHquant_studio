"""Trading ledger schema and DataFrame conversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass
class LedgerEntry:
    date: str
    symbol: str
    action: str
    price: float
    shares: int
    position_before: int
    position_after: int
    cash_flow: float
    fee: float
    cash_after: float
    equity_after: float
    target_weight: float = 0.0
    holding_cost_after: float = 0.0
    position_weight_before: float = 0.0
    reason: str = ""
    run_id: str = ""
    order_type: str = "MARKET"
    limit_price: float | None = None
    signal_date: str = ""
    applied_slippage: float = 0.0
    cost_tier: int = -1
    ledger_version: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Ledger:
    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def append(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([entry.to_dict() for entry in self.entries])
