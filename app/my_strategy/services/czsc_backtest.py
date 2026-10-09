"""CZSC intentions executed next session through the shared A-share broker."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import copy
import math
from pathlib import Path
import re
from typing import Any, Callable

import pandas as pd

from my_strategy.adapters.czsc_adapter import SOURCE_SHA256, legacy_fuyao_source, load_bars, normalize_symbol
from my_strategy.core.run_context import RunContext, create_run_context, stable_hash
from my_strategy.execution.broker import BrokerSimulator
from my_strategy.execution.cost_model import CostModel
from my_strategy.execution.ledger import LedgerEntry
from my_strategy.execution.order import Order
from my_strategy.services.czsc_analysis import strategy_config


def _tick(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _guard(symbol: str, row: Any, previous: Any, side: str, seasoned: int,
           settings: dict[str, Any]) -> str | None:
    if not (symbol.endswith(".SH") and symbol.startswith("60") or symbol.endswith(".SZ") and symbol.startswith("00")):
        return "unsupported_board"
    if seasoned < int(settings["seasoned_bars"]):
        return "insufficient_seasoned_bars"
    if any(legacy_fuyao_source(getattr(bar, "source", "")) for bar in (row, previous)):
        return "legacy_fuyao_price_basis_unverified"
    if any(getattr(bar, "has_trade_price", None) != 1 for bar in (row, previous)):
        return "unverified_trade_price"
    if row.volume <= 0 or row.amount <= 0:
        return "suspended_or_zero_volume"
    band = float(settings["mainboard_conservative_limit"])
    upper, lower = _tick(float(previous.close) * (1 + band)), _tick(float(previous.close) * (1 - band))
    # A conservative 5% guard protects both ST and ordinary main-board samples;
    # these are not asserted to be the exchange's official limit prices.
    if side == "BUY" and float(row.open) >= upper:
        return "conservative_upper_price_guard"
    if side == "SELL" and float(row.open) <= lower:
        return "conservative_lower_price_guard"
    return None


def _planned_entry_rejection(plan: Any, intent: dict, symbol: str, date: str,
                             next_market: dict[str, str], policy: str, used: set[str]) -> str | None:
    if not isinstance(plan, dict) or plan.get("status") != "active" or not plan.get("entry_allowed"):
        return "entry_plan_not_active"
    if not plan.get("plan_id") or plan.get("plan_id") in used:
        return "entry_plan_consumed"
    if plan.get("symbol") != symbol or plan.get("policy") != policy:
        return "entry_plan_identity_mismatch"
    if plan.get("signal_date") != intent["date"] or plan.get("valid_for_sessions") != 1:
        return "entry_plan_invalid_time_contract"
    available = pd.Timestamp(plan.get("available_at"))
    if pd.isna(available) or available > pd.Timestamp(intent["date"] + "T15:00:00+08:00"):
        return "entry_plan_not_available"
    if next_market.get(plan["signal_date"]) != date:
        return "entry_plan_expired"
    return None


def execute_decisions(symbol: str, frame: pd.DataFrame, decisions: list[dict[str, Any]],
                      start: str, initial_cash: float, config: dict[str, Any], run_id: str,
                      *, market_dates: list[str] | None = None, verified_sources=None,
                      entry_policy: str = "legacy",
                      exit_policy: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
                      entry_gate: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None) -> dict[str, Any]:
    """Match only previously observed targets at the next bar's open.

    Sizing/liquidity use prior close data. Legacy targets permit retries. Fresh
    and risk policies require a frozen one-session plan and never add shares to
    an existing position. Planned account exits persist until actually flat.
    No forced final liquidation invents an execution after the requested end.
    An optional research callback observes actual close state only. Its explicit
    arm may request a persistent next-open exit, without vetoing hard exits.
    """
    settings = config["execution"]
    if entry_policy not in {"legacy", "fresh", "risk"}:
        raise ValueError("unsupported entry policy")
    planned = entry_policy != "legacy"
    if exit_policy is not None and not planned:
        raise ValueError("actual holding exit research requires fresh or risk entry policy")
    if entry_gate is not None and not planned:
        raise ValueError("actual candidate entry research requires fresh or risk entry policy")
    if planned:
        dates = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").tolist()
        if (not market_dates or list(market_dates) != sorted(set(market_dates))
                or len(decisions) != len(frame) or dates != [row["date"] for row in decisions]
                or dates != sorted(set(dates)) or set(frame["symbol"].astype(str)) != {symbol}):
            raise ValueError("planned entry requires aligned bars and a verified ascending calendar")
        if not math.isfinite(float(initial_cash)) or initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
    lot = int(settings["lot_size"])
    if lot != 100 or not 0 < float(settings["max_weight"]) <= 1 or not 0 < float(settings["max_volume_participation"]) <= 1:
        raise ValueError("主板整手、权重或成交量参与率配置不合法")
    if not 0 < float(settings["mainboard_conservative_limit"]) <= 0.05:
        raise ValueError("缺少历史 ST 元数据时价格保护不得宽于5%")
    if int(settings["seasoned_bars"]) < 60:
        raise ValueError("交易经历保护至少需要60根有效成交bar")
    if any(not math.isfinite(float(settings[key])) or float(settings[key]) < 0 for key in ("commission", "stamp_tax", "min_commission", "slippage")) or float(settings["slippage"]) >= 1:
        raise ValueError("费用必须为有限非负数，滑点必须小于100%")
    cost = CostModel(commission=float(settings["commission"]), stamp_tax=float(settings["stamp_tax"]),
                     min_commission=float(settings["min_commission"]), slippage=float(settings["slippage"]), version="flat_v1", impact_tiers=())
    broker = BrokerSimulator(initial_cash=initial_cash, cost_model=cost, run_id=run_id)
    rows = list(frame.itertuples(index=False))
    next_market = dict(zip(market_dates[:-1], market_dates[1:])) if market_dates is not None else None
    daily, rejected = [], []
    last_buy_date: str | None = None
    last_buy_index: int | None = None
    entry_diagnostics, used_plans, cancellations = [], set(), set()
    actual_position: dict[str, Any] | None = None
    sell_pending: str | None = None
    model_exit_request: dict[str, Any] | None = None
    holding_snapshots, exit_policy_diagnostics = [], []
    entry_gate_request: dict[str, Any] | None = None
    entry_gate_snapshots, entry_gate_diagnostics = [], []
    seasoned = 0
    for index, row in enumerate(rows):
        date = pd.Timestamp(row.date).date().isoformat()
        if date >= start:
            previous = rows[index - 1] if index else None
            intent = decisions[index - 1] if index else None
            if previous is not None and intent is not None:
                target = float(intent["target_weight"])
                held = broker.shares(symbol)
                side = "BUY" if target > 0 and held == 0 else "SELL" if target == 0 and held > 0 else None
                plan = intent.get("entry_plan")
                if planned and held > 0:
                    if target == 0:
                        sell_pending = "native_target_exit" if exit_policy is not None and sell_pending == "research_exit_model" else sell_pending or "native_target_exit"
                    stop = actual_position["stop_price"] if actual_position else None
                    if stop is not None and (float(previous.close) <= stop or float(row.open) <= stop):
                        if exit_policy is not None and sell_pending == "research_exit_model":
                            sell_pending = "actual_account_stop"
                        else:
                            sell_pending = sell_pending or "actual_account_stop"
                    if model_exit_request and not sell_pending:
                        sell_pending = "research_exit_model"
                    if sell_pending:
                        side, target = "SELL", 0.0
                if planned and side == "BUY":
                    reason = _planned_entry_rejection(plan, intent, symbol, date, next_market, entry_policy, used_plans)
                    if reason:
                        if reason == "entry_plan_expired" and isinstance(plan, dict) and plan.get("plan_id"):
                            used_plans.add(plan["plan_id"])
                        key = ((plan or {}).get("plan_id"), reason)
                        if key not in cancellations:
                            cancellations.add(key)
                            entry_diagnostics.append({"date": date, "signal_date": intent["date"], "symbol": symbol,
                                "plan_id": (plan or {}).get("plan_id"), "status": "cancelled", "reason": reason,
                                "entry_policy": entry_policy})
                            rejected.append({"date": date, "signal_date": intent["date"], "symbol": symbol,
                                "action": "BUY", "reason": reason, "target_weight": target, "reference_price": float(row.open)})
                        side = None
                    else:
                        used_plans.add(plan["plan_id"])
                        if (entry_gate_request is not None
                                and entry_gate_request["signal_date"] == intent["date"]
                                and entry_gate_request["plan_id"] == plan["plan_id"]
                                and entry_gate_request["request_entry"] is False):
                            reason = "policy_entry_gate_veto"
                            entry_diagnostics.append({"date": date, "signal_date": intent["date"], "symbol": symbol,
                                "plan_id": plan["plan_id"], "status": "cancelled", "reason": reason,
                                "entry_policy": entry_policy, "model_contract_hash": entry_gate_request["contract_hash"]})
                            rejected.append({"date": date, "signal_date": intent["date"], "symbol": symbol,
                                "action": "BUY", "reason": reason, "target_weight": target,
                                "reference_price": float(row.open)})
                            side = None
                if side is not None:
                    rejection = _guard(symbol, row, previous, side, seasoned, settings)
                    if side == "BUY" and intent.get("eligible") is False:
                        rejection = "research_signal_input_ineligible"
                    if next_market is not None and next_market.get(intent["date"]) != date:
                        rejection = "missing_next_market_bar"
                    if verified_sources is not None and any(getattr(bar, "source", "") not in verified_sources for bar in (row, previous)):
                        rejection = "unverified_research_source"
                    if side == "SELL" and last_buy_date == date:
                        rejection = "t_plus_one"
                    price = _tick(cost.execution_price(side, float(row.open)))
                    equity_before = broker.equity({symbol: float(row.open)})
                    capacity = int(float(previous.volume) * float(settings["max_volume_participation"])) // lot * lot
                    if side == "BUY":
                        budget = min(broker.cash, equity_before * target)
                        shares = int(budget / price) // lot * lot
                        shares = min(shares, capacity)
                        if planned and entry_policy == "risk":
                            floor, ceiling, stop, fraction = (plan.get(key) for key in ("price_floor", "price_ceiling", "stop_price", "risk_fraction"))
                            if (any(value is None or not math.isfinite(float(value)) for value in (floor, ceiling, stop, fraction))
                                    or not 0 < float(fraction) <= 1 or not 0 < float(stop) < float(ceiling)
                                    or float(floor) <= float(stop) or float(floor) > float(ceiling)):
                                rejection = "entry_plan_invalid_risk_contract"
                            elif price > float(ceiling):
                                rejection = "entry_plan_price_above_ceiling"
                            elif price <= float(stop) or price < float(floor):
                                rejection = "entry_plan_price_below_floor"
                            else:
                                risk_shares = int(equity_before * float(fraction) / (price - float(stop))) // lot * lot
                                shares = min(shares, risk_shares)
                        while shares > 0 and shares * price + cost.fees(side, shares * price) > budget:
                            shares -= lot
                    else:
                        shares = min(held, capacity)
                    if not rejection and shares <= 0:
                        rejection = "insufficient_cash_or_prior_liquidity"
                    if rejection:
                        rejected.append({"date": date, "signal_date": intent["date"], "symbol": symbol, "action": side,
                                         "reason": rejection, "target_weight": target, "reference_price": float(row.open)})
                        if planned:
                            entry_diagnostics.append({"date": date, "signal_date": intent["date"], "symbol": symbol,
                                "plan_id": (plan or {}).get("plan_id") if side == "BUY" else (actual_position or {}).get("plan_id"),
                                "action": side, "status": "rejected", "reason": rejection,
                                "entry_policy": entry_policy, "execution_price": price,
                                "stop_price": (plan or {}).get("stop_price") if side == "BUY" else (actual_position or {}).get("stop_price"),
                                "price_floor": (plan or {}).get("price_floor") if side == "BUY" else None,
                                "price_ceiling": (plan or {}).get("price_ceiling") if side == "BUY" else None,
                                "sell_pending": sell_pending})
                    else:
                        order = Order(date=date, symbol=symbol, side=side, shares=shares, price=price,
                                      target_weight=target, reason=("CZSC 确认结构目标；前日收盘信号、次日开盘执行" if not planned else
                                          "冻结入场计划；次日开盘执行" if side == "BUY" else sell_pending or "native_target_exit"),
                                      signal_date=intent["date"],
                                      applied_slippage=cost.slippage)
                        entry = broker.submit(order, mark_price=float(row.open), equity_before=equity_before,
                                              portfolio_prices={symbol: float(row.open)})
                        if entry is None:
                            rejected.append({"date": date, "signal_date": intent["date"], "symbol": symbol, "action": side, "reason": "broker_rejected"})
                            if planned:
                                entry_diagnostics.append({"date": date, "signal_date": intent["date"], "symbol": symbol,
                                    "plan_id": (plan or {}).get("plan_id"), "action": side, "status": "rejected",
                                    "reason": "broker_rejected", "entry_policy": entry_policy})
                        elif side == "BUY":
                            last_buy_date = date
                            last_buy_index = index
                            if planned:
                                actual_cost = float(entry.holding_cost_after)
                                cost_stop = actual_cost * (1 - float(config["position"]["stop_loss"]) / 10000)
                                frozen_stop = plan.get("stop_price") if entry_policy == "risk" else None
                                actual_position = {"plan_id": plan["plan_id"], "entry_date": date, "signal_date": intent["date"],
                                    "entry_price": price, "actual_cost_per_share": actual_cost,
                                    "entry_cash_debit": -float(entry.cash_flow), "entry_shares": shares,
                                    "frozen_structure_stop": frozen_stop, "stop_price": max(cost_stop, float(frozen_stop)) if frozen_stop is not None else cost_stop,
                                    "risk_fraction": plan.get("risk_fraction"), "entry_policy": entry_policy,
                                    "reference_price": plan.get("reference_price"), "price_floor": plan.get("price_floor"),
                                    "price_ceiling": plan.get("price_ceiling")}
                                entry_diagnostics.append({**actual_position, "date": date, "symbol": symbol, "action": "BUY",
                                    "status": "filled", "planned_stop_risk_cash": shares * (price - float(frozen_stop)) if frozen_stop is not None else None,
                                    "risk_budget_cash": equity_before * float(plan["risk_fraction"]) if frozen_stop is not None else None})
                        if planned and entry is not None and side == "SELL":
                            entry_diagnostics.append({"date": date, "signal_date": intent["date"], "symbol": symbol,
                                "plan_id": (actual_position or {}).get("plan_id"), "action": "SELL", "status": "filled",
                                "reason": sell_pending, "execution_price": price, "remaining_shares": broker.shares(symbol),
                                "stop_price": (actual_position or {}).get("stop_price"), "entry_policy": entry_policy})
                            if exit_policy is not None and model_exit_request is not None:
                                entry_diagnostics[-1].update(model_request_date=model_exit_request["signal_date"],
                                    model_contract_hash=model_exit_request["contract_hash"])
                            if broker.shares(symbol) == 0:
                                actual_position, sell_pending = None, None
                                model_exit_request = None
            daily.append({"date": date, "cash": float(broker.cash), "shares": broker.shares(symbol), "close": float(row.close),
                          "market_value": broker.market_value(symbol, float(row.close)), "equity": broker.equity({symbol: float(row.close)}),
                          "target_weight": float(decisions[index]["target_weight"]), "run_id": run_id})
            if planned:
                daily[-1].update(entry_policy=entry_policy, actual_entry_price=(actual_position or {}).get("entry_price"),
                    actual_stop_price=(actual_position or {}).get("stop_price"), actual_plan_id=(actual_position or {}).get("plan_id"),
                    sell_pending=sell_pending)
            if exit_policy is not None and broker.shares(symbol) > 0:
                position = actual_position or {}
                snapshot = {"symbol": symbol, "date": date, "available_at": date + "T15:00:00+08:00",
                    "bar_index": index, "shares": broker.shares(symbol), "cash": float(broker.cash),
                    "equity": float(daily[-1]["equity"]), "close": float(row.close),
                    "initial_cash": initial_cash,
                    "actual_cost_per_share": float(broker.avg_cost[symbol]), "entry_date": last_buy_date,
                    "holding_bars": index - last_buy_index if last_buy_index is not None else None,
                    "entry_shares": position.get("entry_shares"), "stop_price": position.get("stop_price"),
                    "entry_policy": entry_policy, "plan_id": position.get("plan_id"),
                    "sell_pending": sell_pending, "native_target_weight": float(decisions[index]["target_weight"]),
                    "ledger_entries_seen": len(broker.ledger.entries),
                    "last_ledger_entry": broker.ledger.entries[-1].to_dict(),
                    "next_market_session": next_market.get(date),
                    "bar": {key: getattr(row, key) for key in ("open", "high", "low", "close", "volume", "amount", "source", "has_trade_price")}}
                holding_snapshots.append(copy.deepcopy(snapshot))
                response = exit_policy(copy.deepcopy(snapshot)) or {}
                if not isinstance(response, dict):
                    raise ValueError("exit_policy must return a research decision mapping")
                probability = response.get("probability")
                if probability is not None and (not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1):
                    raise ValueError("exit probability must be finite within [0, 1] or unavailable")
                applied = response.get("apply") is True and response.get("research_arm") is True
                if applied and (response.get("head") != "holding_exit" or not response.get("contract_hash")):
                    raise ValueError("applied exit research requires an independent holding-exit contract")
                hard_exit = (snapshot["native_target_weight"] == 0 or
                             snapshot["stop_price"] is not None and float(row.close) <= float(snapshot["stop_price"]) or
                             sell_pending not in {None, "research_exit_model"})
                request = applied and bool(response.get("request_exit")) and probability is not None and not hard_exit
                if request and snapshot["next_market_session"] is not None and model_exit_request is None:
                    model_exit_request = {**copy.deepcopy(response), "signal_date": date,
                                          "execution_not_before": snapshot["next_market_session"]}
                exit_policy_diagnostics.append({**copy.deepcopy(response), "date": date, "shares": snapshot["shares"],
                    "ledger_entries_seen": snapshot["ledger_entries_seen"], "applied": bool(request and snapshot["next_market_session"]),
                    "hard_exit_priority": bool(hard_exit), "next_market_session": snapshot["next_market_session"],
                    "pending_signal_date": (model_exit_request or {}).get("signal_date")})
            if entry_gate is not None:
                # Decide at this close using the actual account, before observing
                # the next open. A veto consumes the existing one-session plan;
                # it never manufactures an actual holding or schedules retries.
                entry_gate_request = None
                candidate = decisions[index]
                candidate_plan = candidate.get("entry_plan")
                if (broker.shares(symbol) == 0 and float(candidate["target_weight"]) > 0
                        and isinstance(candidate_plan, dict) and candidate_plan.get("status") == "active"
                        and candidate_plan.get("entry_allowed") is True):
                    entry_snapshot = {"symbol": symbol, "date": date, "available_at": date + "T15:00:00+08:00",
                        "bar_index": index, "shares": 0, "cash": float(broker.cash), "equity": float(daily[-1]["equity"]),
                        "close": float(row.close), "entry_policy": entry_policy,
                        "entry_plan": copy.deepcopy(candidate_plan), "plan_id": candidate_plan.get("plan_id"),
                        "target_weight": float(candidate["target_weight"]), "initial_cash": initial_cash,
                        "ledger_entries_seen": len(broker.ledger.entries), "next_market_session": next_market.get(date),
                        "bar": {key: getattr(row, key) for key in ("open", "high", "low", "close", "volume", "amount", "source", "has_trade_price")}}
                    entry_gate_snapshots.append(copy.deepcopy(entry_snapshot))
                    response = entry_gate(copy.deepcopy(entry_snapshot)) or {}
                    if not isinstance(response, dict):
                        raise ValueError("entry_gate must return an independent research decision mapping")
                    probability = response.get("probability")
                    if probability is not None and (not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1):
                        raise ValueError("policy entry probability must be finite within [0, 1] or unavailable")
                    applied = response.get("apply") is True and response.get("research_arm") is True
                    if applied and (response.get("head") != "policy_entry" or not response.get("contract_hash")
                            or probability is None or not isinstance(response.get("request_entry"), bool)):
                        raise ValueError("applied candidate entry requires an independent actual-policy contract and probability")
                    if applied and entry_snapshot["next_market_session"] is not None:
                        entry_gate_request = {**copy.deepcopy(response), "signal_date": date,
                            "plan_id": candidate_plan["plan_id"], "execution_not_before": entry_snapshot["next_market_session"]}
                    entry_gate_diagnostics.append({**copy.deepcopy(response), "date": date,
                        "plan_id": candidate_plan.get("plan_id"), "cash": entry_snapshot["cash"],
                        "ledger_entries_seen": entry_snapshot["ledger_entries_seen"], "applied": bool(applied and entry_snapshot["next_market_session"]),
                        "next_market_session": entry_snapshot["next_market_session"]})
        if row.volume > 0 and row.amount > 0 and getattr(row, "has_trade_price", None) == 1 and not legacy_fuyao_source(getattr(row, "source", "")):
            seasoned += 1
    if not daily:
        raise ValueError("请求回测区间内无本地行情")
    equity = pd.Series([initial_cash, *[item["equity"] for item in daily]], dtype=float)
    returns = equity.pct_change().iloc[1:]
    total = float(equity.iloc[-1] / initial_cash - 1)
    volatility = float(returns.std(ddof=0) * math.sqrt(252))
    metrics = {"total_return": total, "max_drawdown": float((equity / equity.cummax() - 1).min()),
               "annualized_return": float((equity.iloc[-1] / initial_cash) ** (252 / len(daily)) - 1),
               "annualized_volatility": volatility, "sharpe": float(returns.mean() * 252 / volatility) if volatility else 0.0,
               "final_equity": float(equity.iloc[-1]), "trade_count": len(broker.ledger.entries), "rejection_count": len(rejected),
               "fees": float(sum(entry.fee for entry in broker.ledger.entries)), "open_shares": broker.shares(symbol), "days": len(daily)}
    ledger = broker.ledger.to_frame().to_dict(orient="records")
    result = {"ledger": ledger, "trades": ledger, "daily": daily, "rejections": rejected, "metrics": metrics,
              "cost_model": cost.as_config_dict()}
    if planned:
        result.update(entry_policy=entry_policy, entry_diagnostics=entry_diagnostics, actual_position=actual_position,
                      sell_pending=sell_pending, consumed_plan_ids=sorted(used_plans))
    if exit_policy is not None:
        result.update(holding_snapshots=holding_snapshots, exit_policy_diagnostics=exit_policy_diagnostics,
                      exit_policy_scope="actual_broker_close_state_next_verified_open_research_only")
    if entry_gate is not None:
        result.update(entry_gate_snapshots=entry_gate_snapshots, entry_gate_diagnostics=entry_gate_diagnostics,
                      entry_gate_scope="actual_flat_broker_close_state_one_session_plan_research_only")
    return result


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def backtest_stock(symbol: str, start: str, end: str, initial_cash: float = 100000,
                   *, db_path: str | Path | None = None, config: dict[str, Any] | None = None,
                   run_context: RunContext | None = None, run_id: str | None = None,
                   device: str | None = None, prepared: Any | None = None,
                   session: Any | None = None, compute_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist a reproducible ledger using all prior history for warmup."""
    symbol = normalize_symbol(symbol)
    start, end = pd.Timestamp(start).date().isoformat(), pd.Timestamp(end).date().isoformat()
    if start > end:
        raise ValueError("起始日期不得晚于结束日期")
    if not math.isfinite(float(initial_cash)) or initial_cash <= 0:
        raise ValueError("初始资金必须为正数")
    if run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", run_id):
        raise ValueError("无效运行标识")
    settings = strategy_config(config)
    if session is not None and prepared is None:
        raise ValueError("复用意图时必须提供其原始准备数据")
    if prepared is None:
        from my_strategy.services.czsc_compute import prepare_replay
        frame = load_bars(symbol, end=end, db_path=db_path)
        prepared = prepare_replay(frame, settings)
    else:
        frame = prepared.frame
        if prepared.symbol != symbol or prepared.config_hash != stable_hash(settings):
            raise ValueError("准备数据的股票或策略配置不匹配")
        if frame.empty or set(frame["symbol"].astype(str)) != {symbol}:
            raise ValueError("准备行情包含不匹配的股票代码")
        if pd.Timestamp(frame.iloc[-1]["date"]).date().isoformat() > end:
            raise ValueError("准备数据包含请求截止日之后的行情")
    if session is None:
        from my_strategy.services.czsc_compute import replay_prepared_batch
        sessions, compute_info = replay_prepared_batch([prepared], settings, device=device)
        session = sessions[0]
    if session.symbol != symbol:
        raise ValueError("意图会话股票与原始准备行情不一致")
    if len(session.decisions) != len(frame) or any(decision["date"] != pd.Timestamp(day).date().isoformat() for decision, day in zip(session.decisions, frame["date"])):
        raise ValueError("意图与原始准备行情日期不一致")
    context = run_context or create_run_context(task="czsc-backtest", as_of_date=end,
                                               config={"strategy": settings, "initial_cash": initial_cash}, data_version=frame.attrs["data_version"],
                                               run_id=run_id, scope="single", stocks=[symbol], source="api", start_date=start, end_date=end)
    result = execute_decisions(symbol, frame, session.decisions, start, float(initial_cash), settings, context.run_id)
    destination = context.subdir("backtests", symbol.replace(".", "_"))
    for name in ("ledger", "daily", "rejections"):
        output = pd.DataFrame(result[name])
        if output.empty:
            columns = list(LedgerEntry.__dataclass_fields__) if name == "ledger" else ["date", "signal_date", "symbol", "action", "reason", "target_weight", "reference_price"]
            output = pd.DataFrame(columns=columns)
        _write_frame(output, destination / f"{name}.csv")
    data_range = {"requested_start": start, "requested_end": end, "start": result["daily"][0]["date"], "end": result["daily"][-1]["date"],
                  "warmup_start": pd.Timestamp(frame.iloc[0]["date"]).date().isoformat(), "bars": len(frame)}
    result.update({"symbol": symbol, "run_id": context.run_id, "initial_cash": float(initial_cash), "data_end": data_range["end"], "data_range": data_range,
                   "data_version": frame.attrs["data_version"], "config_hash": stable_hash(settings), "strategy_version": settings["strategy_version"],
                   "compute_info": compute_info or {},
                   "price_basis": frame.attrs["price_basis"], "input_quality": {"legacy_fuyao_bars": frame.attrs["legacy_fuyao_bars"], "unverified_trade_price_bars": frame.attrs["unverified_trade_price_bars"]},
                   "source_sha256": SOURCE_SHA256, "context": context.to_dict(), "events": [event for event in session.events if event["time"] >= start],
                   "artifacts": {name: str(destination / f"{name}.csv") for name in ("ledger", "daily", "rejections")},
                   "limitations": ["仅支持沪深主板做多；创业板、科创板、北交所拒单。", "缺少历史 ST 身份与交易所限价：采用按0.01元四舍五入的保守5%开盘价格保护，可能拒绝普通股票可成交订单。",
                                    "60根历史有效成交bar仅证明样本已有交易经历，不等于IPO日期。", "旧fuyao来源可能为前复权价格，即使has_trade_price=1也未核验；当日或前日涉及该来源拒单，保留原bar供结构研究。", "Tushare原始价格未复权；未模拟现金分红、送转股。", "成交量参与上限基于前一交易bar的量，不使用当日完整成交量定仓。",
                                    "CZSC Position的止损与超时以理论信号参考价生成目标；实际成交价、费用和未成交以独立账本为准。",
                                    "在空仓与持仓状态转换时下单；建仓受成交量参与上限限制时不追补、不进行每日权重再平衡。",
                                    "佣金、最低佣金、卖出印花税与滑点采用配置固定费率，未追溯历史税率变化或单独模拟过户费。",
                                    "使用当前原始库快照逐根推进；未重建历史数据修订版本。",
                                    "周/月线在下一周期首根日线可观测后才闭合；无未来成交或期末强制卖出。"]})
    context.write_metadata({"config": {"strategy": settings, "initial_cash": initial_cash}, "data_range": data_range,
                            "data_version": frame.attrs["data_version"], "strategy_version": settings["strategy_version"], "artifacts": result["artifacts"], "compute_info": result["compute_info"]})
    return result
