"""Reusable broker simulator for backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from my_strategy.execution.cost_model import CostModel
from my_strategy.execution.ledger import Ledger, LedgerEntry
from my_strategy.execution.order import Order


@dataclass
class BrokerSimulator:
    initial_cash: float = 100000.0
    cost_model: CostModel = field(default_factory=CostModel)
    run_id: str = ""
    cash: float = field(init=False)
    positions: dict[str, int] = field(default_factory=dict)
    avg_cost: dict[str, float] = field(default_factory=dict)
    ledger: Ledger = field(default_factory=Ledger)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)

    def shares(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def market_value(self, symbol: str, price: float) -> float:
        return self.shares(symbol) * float(price)

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(self.shares(symbol) * float(price) for symbol, price in prices.items())

    def submit(
        self,
        order: Order,
        *,
        mark_price: float | None = None,
        equity_before: float | None = None,
        current_weight: float | None = None,
        portfolio_prices: dict[str, float] | None = None,
    ) -> LedgerEntry | None:
        if order.shares <= 0:
            return None
        symbol = order.symbol
        before = self.shares(symbol)
        if portfolio_prices is not None:
            required = {key for key, shares in self.positions.items() if shares > 0} | {symbol}
            if not required.issubset(portfolio_prices):
                raise ValueError("portfolio_prices must mark every held and traded symbol")
            if any(not math.isfinite(float(value)) or float(value) <= 0 for value in portfolio_prices.values()):
                raise ValueError("portfolio_prices must contain finite positive marks")
            if equity_before is None:
                equity_before = self.equity(portfolio_prices)
        notional = float(order.shares) * float(order.price)
        fee = self.cost_model.fees(order.side, notional)
        if order.side == "BUY":
            total = notional + fee
            if total > self.cash:
                return None
            self.cash -= total
            after = before + order.shares
            old_basis = self.avg_cost.get(symbol, 0.0) * before
            self.avg_cost[symbol] = (old_basis + total) / after if after else 0.0
            cash_flow = -total
            action = "BUY"
        else:
            sell_shares = min(order.shares, before)
            if sell_shares <= 0:
                return None
            notional = float(sell_shares) * float(order.price)
            fee = self.cost_model.fees(order.side, notional)
            net = notional - fee
            self.cash += net
            after = before - sell_shares
            if after == 0:
                self.avg_cost.pop(symbol, None)
            cash_flow = net
            action = "SELL" if after == 0 else "REDUCE"
        self.positions[symbol] = after
        price_for_equity = float(mark_price if mark_price is not None else order.price)
        before_equity = float(equity_before) if equity_before is not None else self.cash + before * price_for_equity
        weight_before = (
            float(current_weight)
            if current_weight is not None
            else (before * price_for_equity / before_equity if before_equity else 0.0)
        )
        entry = LedgerEntry(
            date=str(order.date),
            symbol=symbol,
            action=action,
            price=float(order.price),
            shares=int(order.shares if order.side == "BUY" else min(order.shares, before)),
            position_before=before,
            position_after=after,
            cash_flow=float(cash_flow),
            fee=float(fee),
            cash_after=float(self.cash),
            equity_after=float(self.equity(portfolio_prices) if portfolio_prices is not None else self.cash + after * price_for_equity),
            target_weight=float(order.target_weight),
            holding_cost_after=float(self.avg_cost.get(symbol, 0.0)),
            position_weight_before=float(weight_before),
            reason=order.reason,
            run_id=self.run_id,
            order_type=order.order_type,
            limit_price=order.limit_price,
            signal_date=order.signal_date,
            applied_slippage=float(order.applied_slippage or 0.0),
            cost_tier=int(order.cost_tier),
        )
        self.ledger.append(entry)
        return entry
