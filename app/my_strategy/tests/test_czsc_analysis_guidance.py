from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_analysis as analysis


def calendar_file(root, dates, *, verified=True, name="verified-calendar"):
    path = root / name / "reports" / "calendar.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"verified": verified, "source": "independent:test_calendar", "dates": dates}
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


@pytest.fixture
def calendar_root(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ARTIFACT_RUNS_ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize("action,target", [("BUY", .95), ("SELL", 0)])
def test_latest_native_operation_has_signal_close_not_next_open(calendar_root, action, target):
    value = calendar_file(calendar_root, ["2026-09-30", "2026-10-09"])
    day = "2026-09-30"
    frame = pd.DataFrame([{"date": pd.Timestamp(day), "close": 4.95}])
    session = SimpleNamespace(decisions=[{"date": day, "available_at": day + "T15:00:00+08:00",
                                         "target_weight": target, "eligible": True, "signals": {"rule": "native"}}],
                              events=[{"time": day, "action": action, "reason": "原生操作"}])
    result = analysis._next_session_decision(frame, session)
    assert result["action"] == action and result["reference_price"] == 4.95
    assert result["reference_close"] == 4.95 and result["reference_price_date"] == day
    assert result["reference_price_source"] == "signal_day_close"
    assert result["next_market_session"] == "2026-10-09"
    assert result["calendar"]["hash"] == stable_hash(value)
    assert result["target_weight"] == target and result["model_used"] is False
    assert "理论模拟持仓" in result["scope"] and "不是成交价" in result["execution_basis"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("target,eligible,expected", [(.95, True, "HOLD"), (0, True, "WAIT"), (0, False, "WAIT")])
def test_previous_operation_is_not_recycled_as_next_day_entry(calendar_root, target, eligible, expected):
    calendar_file(calendar_root, ["2026-09-29", "2026-09-30", "2026-10-09"])
    frame = pd.DataFrame([{"date": pd.Timestamp("2026-09-30"), "close": 4.95}])
    session = SimpleNamespace(decisions=[{"date": "2026-09-30", "available_at": "2026-09-30T15:00:00+08:00",
                                         "target_weight": target, "eligible": eligible, "signals": {}}],
                              events=[{"time": "2026-09-29", "action": "BUY", "reason": "历史买入"}])
    result = analysis._next_session_decision(frame, session)
    assert result["action"] == expected and result["basis"] != "历史买入"
    assert result["signal_date"] == "2026-09-30"


def test_next_session_never_guesses_weekdays_or_raw_observation_union(calendar_root):
    assert analysis._verified_next_session("2026-09-30") == (None, {"run_id": None, "source": None, "hash": None, "status": "unavailable"})
    calendar_file(calendar_root, ["2026-09-30", "2026-10-01"], verified=False)
    assert analysis._verified_next_session("2026-09-30")[0] is None
    calendar_file(calendar_root, ["2026-09-29", "2026-09-30"])
    following, provenance = analysis._verified_next_session("2026-09-30")
    assert following is None and provenance["status"] == "calendar_exhausted"
    following, provenance = analysis._verified_next_session("2026-09-28")
    assert following is None and provenance["status"] == "date_not_in_verified_calendar"


def test_invalid_calendar_cannot_claim_next_session(calendar_root):
    calendar_file(calendar_root, ["2026-09-30", "2026-09-29"])
    assert analysis._verified_next_session("2026-09-30")[0] is None
    calendar_file(calendar_root, ["2026-09-30", "2026-09-30", "2026-10-09"])
    assert analysis._verified_next_session("2026-09-30")[0] is None
