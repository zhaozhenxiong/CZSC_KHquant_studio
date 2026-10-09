"""Shared date-routed research inputs, inference, Position and broker work."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pandas as pd

from my_strategy.adapters.czsc_adapter import load_bars, list_symbols, latest_market_date
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions, _write_frame


def _prepare_stock(symbol, end, market_dates, cache=None, db_path=None):
    from my_strategy.services.czsc_research import _calendar_quality, _file_hash
    from my_strategy.services.czsc_research_features import build_features, FEATURE_VERSION, FEATURE_SCHEMA_HASH
    from my_strategy.adapters.czsc_adapter import SOURCE_SHA256
    raw = load_bars(symbol, end=end, db_path=db_path)
    features, hit = None, False
    if cache:
        path = Path(cache["path"])
        if _file_hash(path) != cache["sha256"]:
            raise ValueError("冻结研究特征文件哈希不匹配")
        saved = pd.read_parquet(path)
        if (saved.attrs.get("feature_version") != FEATURE_VERSION or saved.attrs.get("schema_hash") != FEATURE_SCHEMA_HASH
                or saved.attrs.get("source_sha256") != SOURCE_SHA256
                or saved.attrs.get("config_hash") != stable_hash(strategy_config())):
            raise ValueError("缓存特征语义、源码或配置不匹配")
        saved = saved[pd.to_datetime(saved.date) <= pd.Timestamp(end)].reset_index(drop=True)
        columns = ["symbol", "open", "high", "low", "close", "volume", "amount", "source", "has_trade_price"]
        if (len(saved) == len(raw) and all(c in saved for c in columns)
                and pd.to_datetime(saved.date).tolist() == pd.to_datetime(raw.date).tolist()
                and saved[columns].equals(raw[columns].reset_index(drop=True))):
            features, hit = saved, True
    if features is None:
        features = build_features(raw)
    features = _calendar_quality(features, raw, market_dates)
    return {"symbol": symbol, "raw": raw, "features": features, "cache_hit": hit,
            "data_end": pd.Timestamp(raw.iloc[-1].date).date().isoformat(), "data_version": raw.attrs["data_version"]}


def _replay_stock(item, include_decisions=True):
    """Replay CPU-only Position state; inference stays in the parent process."""
    from my_strategy.services.czsc_research_decisions import replay_decisions
    routes = item["routes"]
    symbol = item["symbol"]
    supported = symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ")
    replay = replay_decisions(item["features"], item["raw"], probabilities=item["probabilities"],
                              thresholds=np.array([r.get("probability_threshold", .55) for r in routes]),
                              apply_ml_mask=np.array([supported and r.get("applied_to_entry", False) for r in routes]),
                              model_metadata=routes, entry_policy=item.get("entry_policy", "legacy"),
                              position_start=item.get("position_start"), entry_parameters=item.get("entry_parameters"))
    if not include_decisions:
        replay.pop("decisions")
    return replay


def route_segments(routes, *, start=None):
    """Compress model provenance without hiding a date's change of eligibility."""
    result = []
    for route in routes:
        day = route["date"][:10]
        if start and day < start:
            continue
        identity = {k: v for k, v in route.items() if k not in {"date", "model_dir"}}
        if result and result[-1]["identity"] == identity:
            result[-1]["end"] = day
            result[-1]["days"] += 1
        else:
            result.append({"start": day, "end": day, "days": 1, "identity": identity})
    return result


class ResearchRuntime:
    def __init__(self, *, end, usage_mode="historical", model_policy="auto", model_run_id=None,
                 model_fold=None, calendar_run_id=None, device=None, db_path=None,
                 entry_policy="legacy", position_start=None):
        from my_strategy.services.czsc_research import _calendar, research_status, run_path
        from my_strategy.services.czsc_research_models import ModelResolver
        from my_strategy.services.czsc_research_ml import PredictorSession
        from my_strategy.services.czsc_entry_plan import entry_plan_parameters, PLAN_VERSION
        if entry_policy not in {"legacy", "fresh", "risk"}:
            raise ValueError("invalid entry policy")
        if entry_policy == "risk" and usage_mode == "production":
            raise ValueError("价格与风险入场尚为实验，请使用历史研究模式")
        self.entry_policy = entry_policy
        self.position_start = position_start if entry_policy != "legacy" else None
        self.entry_parameters = entry_plan_parameters() if entry_policy != "legacy" else None
        self.entry_plan_config_hash = stable_hash({"version": PLAN_VERSION, "entry_policy": entry_policy, "parameters": self.entry_parameters})
        from my_strategy.core.device import requested_torch_device
        self.requested_device = requested_torch_device(device)
        from my_strategy.services.czsc_compute import compute_status
        capability = compute_status(device)
        if not capability["available"]:
            raise RuntimeError(capability["fallback_reason"] or "研究设备不可用")
        device = capability["selected_device"]
        self.selected_device = device
        self.end, self.db_path = end, db_path
        status = research_status()
        if not calendar_run_id and model_run_id:
            training = json.loads((run_path(model_run_id) / "reports/research.json").read_text(encoding="utf-8"))
            calendar_run_id = training["calendar"]["run_id"]
        calendar_run_id = calendar_run_id or status.get("calendar_run_id")
        if not calendar_run_id:
            raise ValueError("组合研究需要已核验交易日历")
        self.market_dates, self.calendar = _calendar(calendar_run_id)
        self.mode, self.policy = usage_mode, model_policy
        self.resolver = ModelResolver(usage_mode=usage_mode, model_policy=model_policy,
                                      model_run_id=model_run_id, checkpoint=model_fold,
                                      calendar_run_id=calendar_run_id, strategy_version="czsc_price_volume_mlp_v1")
        self.predictor = PredictorSession(device=device)
        self.cache = {}
        runs = [model_run_id] if model_run_id else [m["run_id"] for m in status["models"]]
        for run_id in runs:
            training = json.loads((run_path(run_id) / "reports/research.json").read_text(encoding="utf-8"))
            for record in training.get("dataset_records", []):
                self.cache.setdefault(record["symbol"], record)
        self.timings = {"preparation_wall_seconds": 0., "inference_wall_seconds": 0., "decision_wall_seconds": 0.}
        self.cache_hits = self.cache_misses = 0
        self._replay_pool = None
        self.cpu_replay_executor = "parent_serial"

    def pinned_check(self, day):
        if self.policy != "pinned" or self.mode == "retrospective":
            return
        route = self.resolver.resolve(day)
        if not route.get("model_dir"):
            raise ValueError(f"固定检查点在 {day} 不可用：{route.get('reason', route.get('status'))}；请选择自动按日期或更早检查点")

    def prepare_batches(self, symbols, *, workers=1, batch_size=32, progress=None, check_cancel=None):
        check = check_cancel or (lambda: None)
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) if workers > 1 else None
        self._replay_pool = pool
        completed, failed, finished = 0, 0, False
        try:
            for offset in range(0, len(symbols), batch_size):
                check()
                batch = symbols[offset:offset + batch_size]
                items, errors = {}, {}
                started = time.perf_counter()
                def consume(index, future=None):
                    nonlocal completed, failed
                    symbol = batch[index]
                    try:
                        items[index] = future.result() if future else _prepare_stock(symbol, self.end, self.market_dates, self.cache.get(symbol), self.db_path)
                    except (ValueError, FileNotFoundError, RuntimeError) as exc:
                        errors[index] = {"symbol": symbol, "error": str(exc)}
                        failed += 1
                    completed += 1
                    if progress:
                        progress("CPU并行研究特征准备" if pool else "CPU研究特征准备", completed, len(symbols), failed)
                if pool:
                    pending = {pool.submit(_prepare_stock, s, self.end, self.market_dates, self.cache.get(s), self.db_path): i for i, s in enumerate(batch)}
                    while pending:
                        check()
                        done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                        for future in done:
                            consume(pending.pop(future), future)
                else:
                    for index in range(len(batch)):
                        check()
                        consume(index)
                self.timings["preparation_wall_seconds"] += time.perf_counter() - started
                prepared = [items[i] for i in sorted(items)]
                self.cache_hits += sum(item["cache_hit"] for item in prepared)
                self.cache_misses += sum(not item["cache_hit"] for item in prepared)
                yield prepared, [errors[i] for i in sorted(errors)]
            check()
            finished = True
        finally:
            self._replay_pool = None
            if pool:
                pool.shutdown(wait=finished, cancel_futures=not finished)

    def replay_batch(self, items, *, progress=None, check_cancel=None, include_decisions=True):
        check = check_cancel or (lambda: None)
        grouped = {}
        started = time.perf_counter()
        for item in items:
            features = item["features"]
            routes = self.resolver.resolve_dates(features.date.tolist())
            if self.entry_policy != "legacy":
                for route in routes:
                    if route.get("model_dir"):
                        route.update(applied_to_entry=False, release_id=None, status="entry_contract_shadow",
                                     reason="固定10日模型与本次实际退出入场契约不同，仅作影子观察",
                                     reason_codes=["entry_contract_mismatch", "model_shadow"])
            item.update(entry_policy=self.entry_policy, position_start=self.position_start, entry_parameters=self.entry_parameters)
            item["routes"] = routes
            item["probabilities"] = np.full(len(features), np.nan)
            mainboard = item["symbol"].startswith("60") and item["symbol"].endswith(".SH") or item["symbol"].startswith("00") and item["symbol"].endswith(".SZ")
            indices = {}
            if mainboard:
                for index, (route, eligible) in enumerate(zip(routes, features.input_eligible.to_numpy(), strict=True)):
                    if route.get("model_dir") and bool(eligible):
                        indices.setdefault(route["model_dir"], []).append(index)
            for directory, selected in indices.items():
                grouped.setdefault(directory, []).append((item, selected))
        for directory, locations in grouped.items():
            check()
            frames = [item["features"].iloc[indices] for item, indices in locations]
            attrs = frames[0].attrs
            semantics = (attrs.get("feature_version"), attrs.get("schema_hash"))
            if any((frame.attrs.get("feature_version"), frame.attrs.get("schema_hash")) != semantics for frame in frames):
                raise ValueError("批量推理特征语义版本/schema不一致")
            joined = pd.concat(frames, ignore_index=True)
            joined.attrs = dict(attrs)
            predict = self.predictor.predict_retrospective if self.mode == "retrospective" else self.predictor.predict
            values = predict(joined, directory, as_of=self.end)
            cursor = 0
            for item, indices in locations:
                item["probabilities"][indices] = values[cursor:cursor + len(indices)]
                cursor += len(indices)
            if cursor != len(values):
                raise ValueError("批量模型概率行数与输入不一致")
        self.timings["inference_wall_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        replays = {}
        if self._replay_pool and items:
            self.cpu_replay_executor = "spawn_process_pool"
            pending = {}
            for index, item in enumerate(items):
                check()
                pending[self._replay_pool.submit(_replay_stock, item, include_decisions)] = index
            while pending:
                check()
                done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in done:
                    replays[pending.pop(future)] = future.result()
        else:
            for index, item in enumerate(items):
                check()
                replays[index] = _replay_stock(item, include_decisions)
        for index, item in enumerate(items):
            check()
            routes = item["routes"]
            replay = replays[index]
            item["replay"] = replay
            latest = replay["latest_decision"]
            latest.update(model_resolution={**routes[-1], "probability": float(item["probabilities"][-1]) if math.isfinite(item["probabilities"][-1]) else None},
                          signal_date=latest["date"], reference_close=latest["reference_price"],
                          position_scope=latest.get("account_context", "theoretical_prefix_position"),
                          basis={"BUY": "组合研究 Position 最新收盘产生开仓意图。", "SELL": "组合研究 Position 最新收盘产生退出意图。", "HOLD": "组合研究理论持仓继续持有，本日无新转仓。", "WAIT": "组合研究理论空仓等待，本日无新开仓。"}[latest["action"]],
                          next_market_session=next((d for d in self.market_dates if d > item["data_end"]), None),
                          calendar=self.calendar, model_used=bool(latest["ml_filter_applied"] and latest["probability"] is not None),
                          scope="组合研究 Position 理论持仓，非我的实盘持仓",
                          execution_basis="收盘意图；下一核验交易日开盘经 Broker 校验执行，参考价不是成交价。")
        self.timings["decision_wall_seconds"] += time.perf_counter() - started
        return items

    def compute_info(self, workers=1, batch_size=32):
        info = dict(self.predictor.diagnostics)
        info.update(selected_device=info.get("device") or self.selected_device, model_inference_rows=info.get("rows", 0),
                    model_inference_batches=info.get("batches", 0), cuda_work=info.get("actual_cuda_inference", False),
                    mps_work=info.get("actual_mps_inference", False),
                    gpu_work=info.get("actual_gpu_inference", info.get("actual_cuda_inference", False)),
                    actual_device=info.get("device") if info.get("rows", 0) else None,
                    requested_device=self.requested_device)
        info.update(self.timings, research=True, usage_mode=self.mode, model_policy=self.policy, entry_policy=self.entry_policy,
                    entry_parameters=self.entry_parameters, entry_plan_config_hash=self.entry_plan_config_hash,
                    cpu_workers=workers, configured_batch_size=batch_size,
                    cpu_preparation_executor="spawn_process_pool" if workers > 1 else "parent_serial",
                    cpu_replay_executor=self.cpu_replay_executor,
                    model_cache_hits=info.get("cache_hits", 0), cache_hits=self.cache_hits, cache_misses=self.cache_misses)
        return info


def analyze_research_frame(raw, *, model_run_id=None, model_fold=None, usage_mode="historical", model_policy="auto",
                           calendar_run_id=None, device=None, db_path=None, entry_policy="legacy", position_start=None):
    end = pd.Timestamp(raw.iloc[-1].date).date().isoformat()
    runtime = ResearchRuntime(end=end, model_run_id=model_run_id, model_fold=model_fold, usage_mode=usage_mode,
                              model_policy=model_policy, calendar_run_id=calendar_run_id, device=device, db_path=db_path,
                              entry_policy=entry_policy, position_start=position_start)
    runtime.pinned_check(end)
    started = time.perf_counter()
    item = _prepare_stock(str(raw.iloc[-1].symbol), end, runtime.market_dates, runtime.cache.get(str(raw.iloc[-1].symbol)), db_path)
    runtime.timings["preparation_wall_seconds"] += time.perf_counter() - started
    runtime.cache_hits, runtime.cache_misses = int(item["cache_hit"]), int(not item["cache_hit"])
    if item["data_version"] != raw.attrs["data_version"]:
        raise ValueError("分析结构与研究特征输入快照不一致")
    item = runtime.replay_batch([item])[0]
    return item, runtime.compute_info()


def run_research(*, kind, symbols=None, start=None, end, initial_cash=100000, model_run_id=None,
                 model_fold=None, usage_mode="historical", model_policy="auto", calendar_run_id=None,
                 device=None, cpu_workers=None, batch_size=None, held_symbols=None,
                 progress=None, check_cancel=None, run_callback=None, entry_policy="legacy"):
    from my_strategy.services.czsc_batch import batch_settings
    from my_strategy.services.czsc_research import _save, _json
    from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
    from my_strategy.web_dashboard.tasks import aggregate_accounts
    if usage_mode == "retrospective":
        raise ValueError("事后模型研究仅用于结构分析，不用于策略扫描或回测认证")
    symbols = list(dict.fromkeys(symbols or [x["symbol"] for x in list_symbols()]))
    if not symbols:
        raise ValueError("本地行情没有有效股票")
    workers, size = batch_settings(cpu_workers, batch_size)
    runtime = ResearchRuntime(end=end, model_run_id=model_run_id, model_fold=model_fold, usage_mode=usage_mode,
                              model_policy=model_policy, calendar_run_id=calendar_run_id, device=device,
                              entry_policy=entry_policy, position_start=start)
    runtime.pinned_check(start if kind == "backtest" else end)
    request = {"research": True, "usage_mode": usage_mode, "model_policy": model_policy,
               "model_run_id": model_run_id, "model_fold": model_fold, "calendar_run_id": runtime.calendar["run_id"],
               "start": start, "end": end, "device": device, "initial_cash": initial_cash, "symbols": symbols}
    if entry_policy != "legacy":
        request["entry_policy"] = entry_policy
        request["entry_parameters"] = runtime.entry_parameters
        request["entry_plan_config_hash"] = runtime.entry_plan_config_hash
    context = create_run_context(task="czsc-research-" + kind, as_of_date=end, config=request,
                                 stocks=symbols, scope="full" if kind == "scan" else "batch", start_date=start, end_date=end)
    if run_callback:
        run_callback(context.run_id)
    check = check_cancel or (lambda: None)
    rows, accounts, failures, routes, summaries = [], [], [], [], Counter()
    input_versions = {}
    ml_applied = False
    for items, errors in runtime.prepare_batches(symbols, workers=workers, batch_size=size, progress=progress, check_cancel=check):
        failures.extend(errors)
        check()
        runtime.replay_batch(items, check_cancel=check, include_decisions=kind == "backtest")
        if progress:
            info = runtime.compute_info()
            progress("GPU批量ML推理" if info.get("gpu_work") else "CPU研究决策", len(rows or accounts) + len(items) + len(failures), len(symbols), len(failures))
        for item in items:
            symbol, replay = item["symbol"], item["replay"]
            input_versions[symbol] = item["data_version"]
            latest, feature = replay["latest_decision"], item["features"].iloc[-1].to_dict()
            if kind == "scan":
                eligible = bool(feature["input_eligible"]) and item["data_end"] == end and (symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ"))
                candidate = latest.get("candidate_action", "WAIT") if eligible else "WAIT"
                action = latest.get("action", "WAIT") if eligible else "WAIT"
                personal_exit = "SELL" if eligible and symbol in (held_symbols or set()) and feature.get("rule_sell") else None
                category = "excluded" if not eligible else "exit" if action == "SELL" or personal_exit else "buy" if action == "BUY" else "watch"
                reasons = list(feature["reason_codes"])
                if item["data_end"] != end:
                    reasons.append("missing_target_date")
                if not (symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ")):
                    reasons.append("unsupported_board")
                if not eligible:
                    latest = {**latest, "eligible": False, "entry_gate_passed": False, "target_weight": 0.,
                              "model_used": False, "ml_filter_applied": False, "probability": None,
                              "model_resolution": {**latest["model_resolution"], "probability": None,
                                                   "applied_to_entry": False, "status": "input_veto",
                                                   "reason": "; ".join(reasons), "reason_codes": reasons},
                              "agent_evidence": [dict(evidence) for evidence in latest["agent_evidence"]],
                              "basis": "本次目标日输入不满足研究条件：" + "; ".join(reasons)}
                    latest["agent_evidence"][2].update(judgment="unavailable", probability=None, applied=False,
                                                        model=latest["model_resolution"])
                    latest["agent_evidence"][3].update(judgment="veto", reasons=reasons)
                    if latest.get("entry_plan"):
                        latest["entry_plan"] = {**latest["entry_plan"], "entry_allowed": False,
                                                "status": "rejected", "reason_codes": reasons}
                ml_applied = ml_applied or bool(eligible and latest["model_used"])
                rows.append(_json({**latest, "symbol": symbol, "category": category, "action": action,
                                  "candidate_action": candidate, "position_intent": action,
                                  "personal_exit_review": personal_exit, "data_end": item["data_end"], "data_version": item["data_version"],
                                  "model_probability": latest["model_resolution"]["probability"] if eligible else None, "model_run_id": latest["model_resolution"].get("model_run_id"),
                                  "model_validated": latest["model_used"], "combined_rule_buy": bool(feature["rule_buy"]),
                                  "combined_rule_sell": bool(feature["rule_sell"]), "reference_close": float(item["raw"].iloc[-1].close),
                                  "reference_levels": {k: feature.get(k) for k in ("ma20", "ma60", "structure_low", "structure_high", "zone_low", "zone_high", "vol_ratio20")},
                                  "reason": latest.get("basis", latest.get("reason", "组合研究判断")),
                                  "reason_codes": reasons,
                                  "probability_target": "10交易日固定期限、可完成入出场标签的费用后净收益为正概率",
                                  "warnings": ["理论 Position 意图与手动持仓退出复核分开；参考价非成交价。", "未复权及历史ST、公司行动边界保留。"]}))
            else:
                check()
                try:
                    account = execute_decisions(symbol, item["raw"], replay["decisions"], start, initial_cash / len(symbols),
                                                strategy_config(), context.run_id, market_dates=runtime.market_dates,
                                                verified_sources=VERIFIED_SOURCES, entry_policy=entry_policy)
                    destination = context.subdir("backtests", symbol.replace(".", "_"))
                    for name in ("daily", "ledger", "rejections"):
                        _write_frame(pd.DataFrame(account[name]), destination / (name + ".csv"))
                    _save(destination / "decisions.json", replay["decisions"])
                    accounts.append({**account, "symbol": symbol, "data_version": item["data_version"], "events": replay["events"]})
                except (ValueError, FileNotFoundError) as exc:
                    failures.append({"symbol": symbol, "error": str(exc)})
                for route, decision in zip(item["routes"], replay["decisions"], strict=True):
                    if route["date"][:10] >= start:
                        applied = bool(decision["ml_filter_applied"] and decision["probability"] is not None)
                        ml_applied = ml_applied or applied
                        summaries["ml_applied_days" if applied else "shadow_days" if route.get("model_dir") else "no_model_days"] += 1
                routes.extend({**s, "symbol": symbol} for s in route_segments(item["routes"], start=start))
        if progress:
            progress("多角色证据扫描" if kind == "scan" else "研究策略实际账本", len(rows if kind == "scan" else accounts) + len(failures), len(symbols), len(failures))
    check()
    if not (rows if kind == "scan" else accounts):
        _save(context.subdir("reports") / "failures.json", failures)
        context.write_metadata({"status": "failed", "failures": failures, "coverage": {"requested": len(symbols), "success": 0, "failed": len(failures)}})
        raise ValueError("研究任务全部失败：" + str(failures[:3]))
    data_version = stable_hash(input_versions)
    from my_strategy.adapters.czsc_adapter import SOURCE_SHA256
    common = {"run_id": context.run_id, "run_context": {**context.to_dict(), "data_version": data_version}, "request": request,
              "strategy_version": "czsc_price_volume_mlp_v1", "failures": failures, "model_run_id": model_run_id,
              "data_version": data_version, "input_versions": input_versions,
              "source_sha256": SOURCE_SHA256, "config_hash": stable_hash({"request": request, "strategy": strategy_config()}),
              "model_fold": model_fold, "usage_mode": usage_mode, "model_policy": model_policy,
              "entry_policy": entry_policy, "position_start": runtime.position_start,
              "entry_parameters": runtime.entry_parameters, "entry_plan_config_hash": runtime.entry_plan_config_hash,
              "calendar": runtime.calendar, "data_end": latest_market_date(end),
              "data_range": {"requested_start": start, "requested_end": end, "end": latest_market_date(end)},
              "coverage": {"requested": len(symbols), "success": len(rows if kind == "scan" else accounts), "failed": len(failures)},
              "model_gate": {"passed": ml_applied, "qualification": "dated_release_only"},
              "compute_info": runtime.compute_info(workers, size)}
    if kind == "scan":
        rows.sort(key=lambda r: ({"buy": 0, "exit": 1, "watch": 2, "excluded": 3}[r["category"]], r["symbol"]))
        result = {**common, "rows": rows, "category_counts": dict(Counter(r["category"] for r in rows))}
        _write_frame(pd.DataFrame([{k: v for k, v in r.items() if k not in {"agent_evidence", "model_resolution", "signals"}} for r in rows]), context.subdir("reports") / "screen_all.csv")
    else:
        result = {**common, **aggregate_accounts(accounts, initial_cash, len(symbols)), "accounts": accounts,
                  "allocation": "固定等额独立账户；失败账户现金保留；无跨股票资金再分配",
                  "model_routes": routes, "model_summary": {**{key: summaries[key] for key in ("shadow_days", "ml_applied_days", "no_model_days")}, "usage_mode": usage_mode, "model_policy": model_policy}}
    _save(context.subdir("reports") / "research-runtime.json", result)
    context.write_metadata({"status": "complete", "request": request, "data_version": data_version,
                            "input_versions": input_versions, "coverage": result["coverage"], "compute_info": result["compute_info"]})
    return _json(result)
