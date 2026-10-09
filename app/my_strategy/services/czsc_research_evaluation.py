"""Spawn-safe CPU account evaluation of one frozen research stock.

The parent calculates CUDA probabilities before dispatch. Workers validate the
snapshot, replay native Position and the shared broker, and write only their
own fold/mode/symbol ledger files; aggregation and run metadata stay in parent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.adapters.czsc_adapter import load_bars, normalize_symbol
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions, _write_frame
from my_strategy.services.czsc_compute import prepare_replay, replay_prepared_batch
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES

EVALUATION_MODES = ("czsc", "price_volume", "czsc_price_volume", "ml")


def evaluate_record(record: dict[str, Any], fold: dict[str, Any], context: Any,
                    config: dict[str, Any], probabilities: np.ndarray,
                    market_dates: list[str] | None) -> dict[str, Any]:
    """Evaluate one account per mode with probabilities aligned to fold prefix.

    ``probabilities`` covers every frozen feature row up to ``test_end``,
    including warmup rows; the parent leaves unavailable-model rows as NaN.
    A missing test interval is explicit cash-retained evidence, not a dropped
    account. Hash or input differences fail closed before any ledger is written.
    This module and its CPU execution path never import or initialize Torch.
    """
    # Local import avoids a module cycle when the parent imports this worker.
    from my_strategy.services.czsc_research import _file_hash, rule_decisions

    symbol = normalize_symbol(str(record["symbol"]))
    snapshot_path = Path(record["path"])
    if _file_hash(snapshot_path) != record["sha256"]:
        raise ValueError("冻结特征文件哈希不匹配")
    features = pd.read_parquet(snapshot_path)
    dates = pd.to_datetime(features["date"], errors="raise").dt.strftime("%Y-%m-%d")
    features = features.loc[dates <= fold["test_end"]].copy()
    if features.empty or not (pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d") >= fold["test_start"]).any():
        return {"symbol": symbol, "accounts": {},
                "missing": {"symbol": symbol, "reason": "no_test_bars_cash_retained"}}
    features["date"] = pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d")
    raw = load_bars(symbol, end=fold["test_end"])
    if len(raw) != len(features):
        raise ValueError("回测输入与特征快照不对齐")
    if (raw["symbol"].astype(str).ne(symbol).any() or features["symbol"].astype(str).ne(symbol).any()
            or not np.array_equal(pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"), features["date"])):
        raise ValueError("回测日期或股票与特征快照不对齐")
    for column in ("open", "high", "low", "close", "volume", "amount", "source", "has_trade_price"):
        left, right = raw[column].to_numpy(), features[column].to_numpy()
        if not np.all((left == right) | (pd.isna(left) & pd.isna(right))):
            raise ValueError("行情已修订，必须重新生成冻结研究数据集")
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or len(values) != len(features):
        raise ValueError("父进程模型概率必须与冻结特征前缀逐行对齐")
    settings = strategy_config()
    baseline_sessions, _ = replay_prepared_batch([prepare_replay(raw, settings)], settings, device="cpu")
    accounts = {}
    for mode in EVALUATION_MODES:
        if mode == "czsc":
            decisions = baseline_sessions[0].decisions
        else:
            selected_mode = "ml" if mode == "ml" else "price_volume" if mode == "price_volume" else "rules"
            decisions = rule_decisions(features, raw, selected_mode, values, config["probability_threshold"])
        result = execute_decisions(symbol, raw, decisions, fold["test_start"], config["initial_cash"], settings, context.run_id,
                                   market_dates=market_dates, verified_sources=VERIFIED_SOURCES)
        destination = context.subdir("evaluation", fold["name"], mode, symbol.replace(".", "_"))
        for name in ("daily", "ledger", "rejections"):
            _write_frame(pd.DataFrame(result[name]), destination / (name + ".csv"))
        accounts[mode] = {**result, "symbol": symbol}
    return {"symbol": symbol, "accounts": accounts, "missing": None}
