"""Point-in-time structures and long-only events from the real CZSC runtime."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from my_strategy.adapters.czsc_adapter import SOURCE_SHA256, load_bars, native_runtime, normalize_symbol, to_raw_bar
from my_strategy.core.config_loader import apply_env_overrides, load_config
from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_chart_indicators import chart_indicators, daily_volume_profile

FREQUENCIES = ("日线", "周线", "月线")


def strategy_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    result = dict(config) if config is not None else apply_env_overrides(load_config("czsc_strategy"))
    if result.get("czsc_version") != "1.0.1" or result.get("source_sha256") != SOURCE_SHA256 or result.get("higher_period_closure") != "next_period_observed":
        raise ValueError("CZSC 版本或高周期闭合协议不受支持")
    if tuple(result.get("frequencies", [])) != FREQUENCIES:
        raise ValueError("只支持日线、周线、月线")
    if int(result.get("warmup_bars", 0)) < 60 or int(result.get("min_bi_len", 0)) < 3:
        raise ValueError("预热至少需要60根行情，笔最小长度至少3")
    if result["position"].get("t0", False):
        raise ValueError("A 股策略禁止 T+0")
    if any(event.get("operate") != "开多" for event in result["position"]["opens"]) or any(event.get("operate") != "平多" for event in result["position"]["exits"]):
        raise ValueError("只允许开多与平多事件")
    return result


def _day(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _pen_key(pen: Any) -> tuple[Any, ...]:
    return (_day(pen.fx_a.dt), _day(pen.fx_b.dt), float(pen.fx_a.fx), float(pen.fx_b.fx))


def _zone_key(zone: Any) -> tuple[Any, ...]:
    return (_day(zone.sdt), _day(zone.edt), float(zone.zd), float(zone.zg))


class IntentSession:
    """Chronological native Position replay; never supplies chart confirmations."""
    def __init__(self, symbol: str, config: dict[str, Any], *, create_position: bool = True):
        self.czsc = native_runtime()
        self.config = config
        self.symbol = symbol
        self.signals: dict[str, list[dict[str, Any]]] = {frequency: [] for frequency in FREQUENCIES}
        self.events: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.position = self.new_position() if create_position else None

    def new_position(self):
        settings = self.config["position"]
        return self.czsc.Position(symbol=self.symbol,
                                  opens=[self.czsc.Event.load(event) for event in settings["opens"]],
                                  exits=[self.czsc.Event.load(event) for event in settings["exits"]],
                                  name=settings["name"], interval=int(settings["interval"]),
                                  timeout=int(settings["timeout"]), stop_loss=float(settings["stop_loss"]), t0=False)

    def apply_signals(self, row: Any, direction: str, week_direction: str, eligible: bool) -> None:
        if self.position is None:
            raise RuntimeError("结构准备阶段不得执行 Position")
        date = _day(row.date)
        available = date + "T15:00:00+08:00"
        signal_values = {"日线_确认笔_方向V1": direction + "_任意_任意_0", "周线_闭合价格_方向V1": week_direction + "_任意_任意_0"}
        for frequency, key in (("日线", "日线_确认笔_方向V1"), ("周线", "周线_闭合价格_方向V1")):
            self.signals[frequency].append({"time": date, "available_at": available, "key": key, "value": signal_values[key], "eligible": eligible})
        if eligible:
            before = float(self.position.pos)
            self.position.update({"symbol": self.symbol, "dt": row.dt, "id": int(row.id), "close": float(row.close), **signal_values})
            target = float(self.position.pos)
            if target != before:
                operation = self.position.operates[-1]
                self.events.append({"time": date, "available_at": available, "signal": "; ".join(f"{key}={value}" for key, value in signal_values.items()),
                                    "action": "BUY" if target > 0 else "SELL", "reason": str(operation.get("op_desc", "确认结构事件")), "target_weight": target * float(self.config["execution"]["max_weight"]),
                                    "reference_price": float(row.close), "reference_price_date": date, "reference_price_source": "signal_day_close"})
        else:
            target = 0.0
        self.decisions.append({"date": date, "available_at": available, "target_weight": target * float(self.config["execution"]["max_weight"]),
                               "eligible": eligible, "signals": signal_values})

    def payload(self, start: str | None = None) -> dict[str, Any]:
        raise RuntimeError("轻量决策会话没有结构确认时间，不能用于图表")


class CzscSession(IntentSession):
    """Higher periods are published only after their next bucket is observed."""
    def __init__(self, symbol: str, config: dict[str, Any], count: int, *,
                 collect_confirmations: bool = True, create_position: bool = True):
        super().__init__(symbol, config, create_position=create_position)
        self.collect_confirmations = collect_confirmations
        self.bg = self.czsc.BarGenerator(base_freq="日线", freqs=list(FREQUENCIES), max_count=max(count + 5, 1000), market="A股")
        self.kas: dict[str, Any] = {}
        self.closed: dict[str, list[Any]] = {frequency: [] for frequency in FREQUENCIES}
        self.closed_available: dict[str, list[str]] = {frequency: [] for frequency in FREQUENCIES}
        self.divergences: dict[str, list[dict[str, Any]]] = {frequency: [] for frequency in FREQUENCIES}
        self.divergence_seen: dict[str, set[tuple]] = {frequency: set() for frequency in FREQUENCIES}
        self.pending: dict[str, Any] = {}
        self.confirmations: dict[str, dict[str, dict[tuple, str]]] = {
            frequency: {"fractals": {}, "pens": {}, "zones": {}} for frequency in FREQUENCIES}

    def _advance(self, frequency: str, bar: Any, available: str) -> None:
        self.closed[frequency].append(bar)
        if frequency not in self.kas:
            self.kas[frequency] = self.czsc.CZSC([bar], min_bi_len=int(self.config["min_bi_len"]), max_bi_num=int(self.config["max_bi_num"]))
        else:
            self.kas[frequency].update(bar)
        if not self.collect_confirmations:
            return
        self.closed_available[frequency].append(available)
        analysis = self.kas[frequency]
        confirmed = self.confirmations[frequency]
        # Confirm a fractal when the third inclusion-adjusted bar is present.
        for fractal in analysis.fx_list:
            confirmed["fractals"].setdefault((_day(fractal.dt), float(fractal.fx), str(fractal.mark)), available)
        for pen in analysis.finished_bis:
            confirmed["pens"].setdefault(_pen_key(pen), available)
        for zone in analysis.zs_list:
            if len(zone.bis) >= 3 and zone.is_valid():
                confirmed["zones"].setdefault(_zone_key(zone), available)
        finished = analysis.finished_bis
        if finished:
            last = finished[-1]
            key = _pen_key(last)
            if key not in self.divergence_seen[frequency]:
                self.divergence_seen[frequency].add(key)
                di = len(analysis.bi_list) - len(finished) + 1
                values = self.czsc._native.call_signal("cxt_five_bi_V230619", analysis, {"di": di})
                label = str(values[0].v1)
                if "底背驰" in label or "顶背驰" in label:
                    self.divergences[frequency].append({"time": key[1], "price": key[3],
                        "kind": "bottom" if "底背驰" in label else "top", "label": label,
                        "available_at": available, "confirmed_at": confirmed["pens"][key],
                        "is_confirmed": True, "signal_name": "cxt_five_bi_V230619", "di": di,
                        "pen_anchors": [list(_pen_key(pen)) for pen in finished[-5:]],
                        "scope": "原生五笔形态参考，非买卖确认；不改变策略"})

    def advance_structure(self, row: Any) -> tuple[str, float | None, float | None]:
        bar = to_raw_bar(row)
        date = _day(row.date)
        available = date + "T15:00:00+08:00"
        self.bg.update(bar)
        self._advance("日线", bar, available)
        higher_bars = self.bg.bars
        for frequency in ("周线", "月线"):
            latest = higher_bars[frequency][-1]
            previous = self.pending.get(frequency)
            if previous is not None and previous.dt != latest.dt:
                self._advance(frequency, previous, available)
            self.pending[frequency] = latest
        daily = self.kas["日线"].finished_bis
        direction = str(daily[-1].direction) if daily else "其他"
        weekly = self.closed["周线"]
        return direction, float(weekly[-1].close) if len(weekly) >= 2 else None, float(weekly[-2].close) if len(weekly) >= 2 else None

    def update(self, row: Any) -> None:
        direction, latest, previous = self.advance_structure(row)
        week_direction = "其他" if latest is None else "向上" if latest > previous else "向下" if latest < previous else "其他"
        eligible = int(row.id) + 1 >= int(self.config["warmup_bars"])
        self.apply_signals(row, direction, week_direction, eligible)

    def payload(self, start: str | None = None) -> dict[str, Any]:
        if not self.collect_confirmations:
            raise RuntimeError("轻量决策会话没有结构确认时间，不能用于图表")
        result: dict[str, Any] = {}
        daily_bars = [{"time": _day(bar.dt), "high": float(bar.high), "low": float(bar.low),
                       "close": float(bar.close), "volume": float(bar.vol)} for bar in self.closed["日线"]]
        profile = daily_volume_profile(daily_bars)
        for frequency in FREQUENCIES:
            analysis = self.kas.get(frequency)
            bars = [{"time": _day(bar.dt), "open": float(bar.open), "high": float(bar.high), "low": float(bar.low),
                     "close": float(bar.close), "volume": float(bar.vol), "is_closed": True} for bar in self.closed[frequency]]
            pending = self.pending.get(frequency)
            availability = list(self.closed_available[frequency])
            if pending is not None and self.closed["日线"] and not any(bar.dt == pending.dt for bar in self.closed[frequency]):
                observed_day = _day(self.closed["日线"][-1].dt)
                # Display the observed OHLC snapshot only; pending is never advanced
                # into the higher-frequency structure or Position before closure.
                bars.append({"time": observed_day, "bucket_time": _day(pending.dt),
                             "open": float(pending.open), "high": float(pending.high), "low": float(pending.low),
                             "close": float(pending.close), "volume": float(pending.vol), "is_closed": False,
                             "observed_at": observed_day + "T15:00:00+08:00"})
                availability.append(observed_day + "T15:00:00+08:00")
            fractals, pens, zones = [], [], []
            if analysis is not None:
                confirmed = self.confirmations[frequency]
                for fractal in analysis.fx_list:
                    key = (_day(fractal.dt), float(fractal.fx), str(fractal.mark))
                    fractals.append({"time": key[0], "price": key[1], "kind": "top" if str(fractal.mark) == "顶分型" else "bottom",
                                     "confirmed_at": confirmed["fractals"].get(key), "is_confirmed": key in confirmed["fractals"]})
                finished = {_pen_key(pen) for pen in analysis.finished_bis}
                for pen in analysis.bi_list:
                    key = _pen_key(pen)
                    pens.append({"start_time": key[0], "end_time": key[1], "start_price": key[2], "end_price": key[3],
                                 "confirmed_at": confirmed["pens"].get(key) if key in finished else None, "is_confirmed": key in finished})
                for zone in analysis.zs_list:
                    if len(zone.bis) < 3 or not zone.is_valid():
                        continue
                    key = _zone_key(zone)
                    zones.append({"start_time": key[0], "end_time": key[1], "low": key[2], "high": key[3],
                                  "confirmed_at": confirmed["zones"].get(key), "is_confirmed": True})
            zones.sort(key=lambda item: (item["start_time"], item["end_time"]))
            previous = None
            for zone in zones:
                zone["direction"] = "up" if previous and zone["low"] > previous["low"] and zone["high"] > previous["high"] else "down" if previous and zone["low"] < previous["low"] and zone["high"] < previous["high"] else "range"
                previous = zone
            indicators = chart_indicators(bars, availability, profile, start, daily_bars=daily_bars)
            if start:
                bars = [item for item in bars if item["time"] >= start]
                fractals = [item for item in fractals if item["time"] >= start]
                pens = [item for item in pens if item["end_time"] >= start]
                zones = [item for item in zones if item["end_time"] >= start]
            result[frequency] = {"bars": bars, "fractals": fractals, "pens": pens, "zones": zones,
                                 "indicators": indicators,
                                 "divergences": [item for item in self.divergences[frequency] if not start or item["time"] >= start],
                                 "signals": [item for item in self.signals[frequency] if not start or item["time"] >= start]}
        return result


def run_session(frame: pd.DataFrame, config: dict[str, Any]) -> CzscSession:
    session = CzscSession(str(frame.iloc[0]["symbol"]), config, len(frame))
    for row in frame.itertuples(index=False):
        session.update(row)
    return session


def _verified_next_session(day: str) -> tuple[str | None, dict[str, Any]]:
    """Read the existing independent calendar without loading models or prices."""
    paths = sorted(ARTIFACT_RUNS_ROOT.glob("*/reports/calendar.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    unmatched = None
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                continue
            dates = value.get("dates", [])
            if value.get("verified") is not True or not value.get("source") or not isinstance(dates, list) or not dates:
                continue
            if dates != sorted(set(dates)) or any(_day(date) != date for date in dates):
                continue
        except (OSError, ValueError, TypeError):
            continue
        calendar = {"run_id": path.parent.parent.name, "source": value["source"], "hash": stable_hash(value)}
        if day not in dates:
            unmatched = unmatched or {**calendar, "status": "date_not_in_verified_calendar"}
            continue
        index = dates.index(day)
        following = dates[index + 1] if index + 1 < len(dates) else None
        return following, {**calendar, "status": "verified_next_session" if following else "calendar_exhausted"}
    return None, unmatched or {"run_id": None, "source": None, "hash": None, "status": "unavailable"}


def _next_session_decision(frame: pd.DataFrame, session: CzscSession) -> dict[str, Any]:
    """Summarize only the final native decision, never recycle an old entry."""
    row, decision = frame.iloc[-1], session.decisions[-1]
    day = decision["date"]
    event = session.events[-1] if session.events and session.events[-1]["time"] == day else None
    target = float(decision["target_weight"])
    action = event["action"] if event else "HOLD" if target > 0 else "WAIT"
    if not decision["eligible"]:
        action, basis = "WAIT", "预热不足，尚不执行策略决策。"
    elif event:
        basis = f"最新收盘触发原生开多事件，策略模拟目标仓位 {target:.0%}。" if action == "BUY" else "最新收盘触发原生平多事件，策略模拟目标清仓。"
    else:
        basis = "策略模拟持仓继续持有，最新收盘未触发新转仓意图。" if target > 0 else "策略模拟为空仓，最新收盘未触发新买入意图。"
    following, calendar = _verified_next_session(day)
    return {"action": action, "signal_date": day, "available_at": decision["available_at"],
            "reference_price": float(row["close"]), "reference_close": float(row["close"]),
            "reference_price_date": day, "reference_price_source": "signal_day_close", "target_weight": target,
            "eligible": bool(decision["eligible"]), "basis": basis, "signals": decision["signals"],
            "next_market_session": following, "calendar": calendar, "model_used": False,
            "scope": "原生 Position 理论模拟持仓，非我的实盘持仓",
            "execution_basis": "收盘意图；下一已核验交易日开盘经 Broker 校验执行，参考价不是成交价。"}


def analyze_stock(symbol: str, start: str | None = None, end: str | None = None, as_of: str | None = None,
                  *, db_path: str | Path | None = None, config: dict[str, Any] | None = None,
                  research: bool = False, usage_mode: str = "historical", model_policy: str = "auto",
                  model_run_id: str | None = None, model_fold: str | None = None,
                  calendar_run_id: str | None = None, device: str | None = None,
                  entry_policy: str = "legacy") -> dict[str, Any]:
    """Rebuild observable bars; planned research starts its flat Position at start."""
    settings = strategy_config(config)
    frame = load_bars(symbol, end=end, as_of=as_of, db_path=db_path)
    session = run_session(frame, settings)
    lower = pd.Timestamp(start).date().isoformat() if start else None
    if lower and lower > _day(frame.iloc[-1]["date"]):
        raise ValueError("起始日期晚于可用行情截止日")
    result = {"symbol": normalize_symbol(symbol), "data_end": _day(frame.iloc[-1]["date"]), "as_of": as_of or end or _day(frame.iloc[-1]["date"]),
            "config_hash": stable_hash(settings), "strategy_version": settings["strategy_version"], "czsc_version": "1.0.1", "source_sha256": SOURCE_SHA256,
            "data_version": frame.attrs["data_version"], "data_range": {"start": _day(frame.iloc[0]["date"]), "end": _day(frame.iloc[-1]["date"]), "bars": len(frame)},
            "price_basis": frame.attrs["price_basis"], "input_quality": {"legacy_fuyao_bars": frame.attrs["legacy_fuyao_bars"], "unverified_trade_price_bars": frame.attrs["unverified_trade_price_bars"]},
            "frequencies": session.payload(lower), "events": [event for event in session.events if not lower or event["time"] >= lower],
            "next_session_decision": _next_session_decision(frame, session),
            "warnings": ["周/月线在下一周期首次观测后才确认闭合，尚未闭合的末桶不用于规则。",
                         "包含旧fuyao来源，复权口径未核验（has_trade_price=1不足以证明未复权）；保留原bar，相关成交拒单。" if frame.attrs["legacy_fuyao_bars"] else "使用未复权价格；除权除息跳空会影响结构。",
                         f"未核验真实交易价格的bar：{frame.attrs['unverified_trade_price_bars']}。", "as_of截断当前行情库快照；未重建当时的数据修订版本。", "事件为下一交易时点的研究意图，真实成交以回测账本为准。"],
            "warmup": {"required": settings["warmup_bars"], "available": len(frame), "eligible": len(frame) >= settings["warmup_bars"]}}
    if research:
        from my_strategy.services.czsc_research_runtime import analyze_research_frame, route_segments
        item, compute = analyze_research_frame(frame, usage_mode=usage_mode, model_policy=model_policy,
                                               model_run_id=model_run_id, model_fold=model_fold,
                                               calendar_run_id=calendar_run_id, device=device, db_path=db_path,
                                               entry_policy=entry_policy, position_start=lower)
        replay = item["replay"]
        result["structure_baseline"] = {"events": result["events"], "next_session_decision": result["next_session_decision"]}
        result.update(strategy_version="czsc_price_volume_mlp_v1",
                      events=[e for e in replay["events"] if not lower or e["time"] >= lower],
                      next_session_decision=replay["latest_decision"], compute_info=compute,
                      research={"usage_mode": usage_mode, "model_policy": model_policy,
                                "entry_policy": entry_policy, "position_start": lower if entry_policy != "legacy" else None,
                                "entry_parameters": compute.get("entry_parameters"), "entry_plan_config_hash": compute.get("entry_plan_config_hash"),
                                "model_routes": route_segments(item["routes"], start=lower),
                                "final_state": replay["final_state"], "agent_evidence": replay["agent_evidence"]})
        if entry_policy != "legacy":
            result["config_hash"] = stable_hash({"strategy": settings, "entry_plan_config_hash": compute["entry_plan_config_hash"]})
        if usage_mode == "retrospective":
            result["warnings"].append("当前模型事后研究历史特征，可能与训练/验证重叠；不计入样本外认证，ML不改变生产意图。")
        from my_strategy.services.czsc_research import _json
        result = _json(result)
    return result
