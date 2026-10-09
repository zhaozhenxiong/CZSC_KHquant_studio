from __future__ import annotations

import copy
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from my_strategy.adapters.czsc_adapter import load_bars, native_runtime, to_raw_bar
from my_strategy.services.czsc_analysis import analyze_stock, run_session, strategy_config
from my_strategy.services.czsc_analysis import _next_session_decision
from my_strategy.services.czsc_backtest import execute_decisions


@pytest.fixture
def raw_db(tmp_path):
    path = tmp_path / "raw.db"
    dates = pd.bdate_range("2020-01-02", periods=420)
    prices = 12 + np.arange(len(dates)) * 0.006 + 1.5 * np.sin(np.arange(len(dates)) / 13)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE stock_daily_normalized (stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,has_trade_price INTEGER,source TEXT)")
        connection.execute("CREATE TABLE securities (stock TEXT,name TEXT,first_date TEXT,last_date TEXT,raw_rows INTEGER)")
        connection.executemany("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)",
                               [("000001.SZ", date.date().isoformat(), float(price), float(price + 0.15), float(price - 0.15),
                                 float(price + 0.04), 1000000.0, float(price * 1000000), 1, "test") for date, price in zip(dates, prices)])
        connection.execute("INSERT INTO securities VALUES (?,?,?,?,?)", ("000001.SZ", "测试股", dates[0].date().isoformat(), dates[-1].date().isoformat(), len(dates)))
    return path


def test_raw_input_preserves_units_and_cutoff(raw_db):
    frame = load_bars("000001", as_of="2020-01-03T14:59:00+08:00", db_path=raw_db)
    assert len(frame) == 1
    assert frame.iloc[0]["volume"] == 1000000
    assert frame.attrs["volume_unit"] == "shares"
    assert frame.attrs["price_basis"] == "unadjusted"
    assert load_bars("000001", as_of="2020-01-03", db_path=raw_db).iloc[-1]["date"] == pd.Timestamp("2020-01-03")


def test_duplicate_dates_fail_closed(raw_db):
    with sqlite3.connect(raw_db) as connection:
        connection.execute("INSERT INTO stock_daily_normalized SELECT * FROM stock_daily_normalized LIMIT 1")
    with pytest.raises(ValueError, match="重复"):
        load_bars("000001", db_path=raw_db)


def test_future_prices_cannot_change_replay(raw_db):
    cutoff = "2020-11-02"
    before = analyze_stock("000001", as_of=cutoff, db_path=raw_db)
    with sqlite3.connect(raw_db) as connection:
        connection.execute("UPDATE stock_daily_normalized SET open=open*7, high=high*7, low=low*7, close=close*7 WHERE date > ?", (cutoff,))
    after = analyze_stock("000001", as_of=cutoff, db_path=raw_db)
    assert before == after
    json.dumps(after, allow_nan=False)
    for payload in after["frequencies"].values():
        assert all(item["time"] <= cutoff for item in payload["bars"])
        assert all(item["confirmed_at"][:10] <= cutoff for item in payload["pens"] if item["is_confirmed"])


def test_research_cached_missing_evidence_can_be_saved_as_strict_json(raw_db, tmp_path, monkeypatch):
    from my_strategy.services import czsc_research_runtime
    from my_strategy.storage.czsc_results import ResultStore
    def research_frame(raw, **kwargs):
        decision = {"action": "WAIT", "agent_evidence": [{"role": "czsc", "confirmed_at": np.nan}],
                    "probability": np.float32(.5), "eligible": np.bool_(True)}
        return {"replay": {"events": [{"time": "2020-11-02", "anchor_time": np.nan}],
                           "latest_decision": decision, "final_state": {}, "agent_evidence": decision["agent_evidence"]},
                "routes": [{"date": "2020-11-02", "applied_to_entry": False}]}, {"rows": 1}
    monkeypatch.setattr(czsc_research_runtime, "analyze_research_frame", research_frame)
    result = analyze_stock("000001", as_of="2020-11-02", db_path=raw_db, research=True)
    json.dumps(result, allow_nan=False)
    assert result["events"][0]["anchor_time"] is None
    assert result["next_session_decision"]["agent_evidence"][0]["confirmed_at"] is None
    result["run_id"] = "strict-json-research-analysis"
    store = ResultStore(tmp_path / "results.db", tmp_path / "runs")
    store.save("analysis", result)
    assert store.get(result["run_id"])["next_session_decision"] == result["next_session_decision"]


def test_unclosed_higher_periods_are_displayed_without_structure_confirmation(raw_db):
    result = analyze_stock("000001", as_of="2020-02-05", db_path=raw_db)
    for frequency, lower in (("周线", "2020-02-03"), ("月线", "2020-02-01")):
        bars = result["frequencies"][frequency]["bars"]
        assert bars[-2]["time"] < lower and bars[-2]["is_closed"]
        assert bars[-1]["time"] == "2020-02-05" and not bars[-1]["is_closed"]
        assert bars[-1]["bucket_time"] >= bars[-1]["time"]
        assert bars[-1]["observed_at"] == "2020-02-05T15:00:00+08:00"


@pytest.mark.parametrize("cutoff", ["2020-02-05", "2020-02-28", "2020-03-02"])
def test_forming_bar_payload_cannot_change_decisions_or_confirmed_structures(raw_db, cutoff):
    frame = load_bars("000001", as_of=cutoff, db_path=raw_db)
    session = run_session(frame, strategy_config())
    decisions, events = copy.deepcopy(session.decisions), copy.deepcopy(session.events)
    confirmations = copy.deepcopy(session.confirmations)
    closed_counts = {frequency: len(bars) for frequency, bars in session.closed.items()}
    payload = session.payload(start=cutoff)
    assert session.decisions == decisions and session.events == events
    assert session.confirmations == confirmations
    assert {frequency: len(bars) for frequency, bars in session.closed.items()} == closed_counts
    for frequency in ("日线", "周线", "月线"):
        assert payload[frequency]["bars"][-1]["time"] == cutoff
        assert payload[frequency]["bars"][-1]["is_closed"] == (frequency == "日线")
        assert len(payload[frequency]["bars"]) == 1
    # Both high-frequency CZSC analyzers still end at their closed bucket.
    for frequency in ("周线", "月线"):
        if frequency in session.kas:
            assert session.kas[frequency].bars_raw[-1].dt == session.closed[frequency][-1].dt
            assert session.kas[frequency].bars_raw[-1].dt != session.pending[frequency].dt


def test_daily_structures_are_native_czsc(raw_db):
    frame = load_bars("000001", db_path=raw_db)
    config = strategy_config()
    session = run_session(frame, config)
    czsc = native_runtime()
    direct = czsc.CZSC([to_raw_bar(row) for row in frame.itertuples(index=False)], min_bi_len=config["min_bi_len"], max_bi_num=config["max_bi_num"])
    assert len(session.kas["日线"].finished_bis) > 0
    assert [(pen.sdt, pen.edt, pen.high, pen.low) for pen in session.kas["日线"].finished_bis] == [(pen.sdt, pen.edt, pen.high, pen.low) for pen in direct.finished_bis]


def test_chart_overlays_keep_prestart_history_and_closed_availability(raw_db):
    start, cutoff = "2020-10-01", "2020-11-04"
    full = analyze_stock("000001", as_of=cutoff, db_path=raw_db)
    clipped = analyze_stock("000001", start=start, as_of=cutoff, db_path=raw_db)
    for frequency in ("日线", "周线", "月线"):
        indicators = clipped["frequencies"][frequency]["indicators"]["rows"]
        assert indicators == [row for row in full["frequencies"][frequency]["indicators"]["rows"] if row["time"] >= start]
        assert [row["time"] for row in indicators] == [row["time"] for row in clipped["frequencies"][frequency]["bars"]]
        assert all(row["available_at"][:10] <= cutoff for row in indicators)
        assert all(row["chip_as_of"] is None or row["chip_as_of"] <= row["time"] for row in indicators)
        for marker in clipped["frequencies"][frequency]["divergences"]:
            assert marker["time"] <= marker["confirmed_at"][:10] <= marker["available_at"][:10] <= cutoff
            assert len(marker["pen_anchors"]) == 5 and marker["signal_name"] == "cxt_five_bi_V230619"
    assert clipped["frequencies"]["日线"]["indicators"]["rows"][0]["ma60"] is not None


def test_default_events_are_long_only_and_have_trades(raw_db):
    frame = load_bars("000001", db_path=raw_db)
    config = strategy_config()
    session = run_session(frame, config)
    assert {event["action"] for event in session.events} == {"BUY", "SELL"}
    prices = {_day.date().isoformat(): float(close) for _day, close in zip(frame["date"], frame["close"])}
    for event in session.events:
        assert event["reference_price"] == prices[event["time"]]
        assert event["reference_price_date"] == event["time"]
        assert event["reference_price_source"] == "signal_day_close"
    result = execute_decisions("000001.SZ", frame, session.decisions, "2020-04-01", 100000, config, "native-test")
    assert len(result["ledger"]) > 0
    assert all(trade["signal_date"] < trade["date"] for trade in result["ledger"])
    for row in result["daily"]:
        assert row["cash"] + row["shares"] * row["close"] == pytest.approx(row["equity"])
    assert sum(trade["cash_flow"] for trade in result["ledger"]) + 100000 == pytest.approx(result["daily"][-1]["cash"])
    assert all(trade["shares"] % 100 == 0 and trade["fee"] > 0 for trade in result["ledger"])


def test_guidance_matches_native_history_recomputed_at_each_operation(raw_db):
    frame = load_bars("000001", db_path=raw_db)
    config = strategy_config()
    full = run_session(frame, config)
    for event in full.events:
        prefix = load_bars("000001", as_of=event["time"], db_path=raw_db)
        replay = run_session(prefix, config)
        visible = SimpleNamespace(decisions=[item for item in full.decisions if item["date"] <= event["time"]],
                                  events=[item for item in full.events if item["time"] <= event["time"]])
        assert replay.decisions == visible.decisions and replay.events == visible.events
        assert _next_session_decision(prefix, replay) == _next_session_decision(prefix, visible)
        guidance = _next_session_decision(prefix, replay)
        assert guidance["action"] == event["action"]
        assert guidance["reference_price"] == event["reference_price"]


def test_broker_rejects_price_guard_and_nonmainboard(raw_db):
    frame = load_bars("000001", db_path=raw_db).iloc[:65].copy()
    config = strategy_config()
    decisions = [{"date": row.date().isoformat(), "target_weight": 0 if index < 60 else 0.95} for index, row in enumerate(frame["date"])]
    frame.loc[61, "open"] = frame.loc[60, "close"] * 1.06
    result = execute_decisions("000001.SZ", frame, decisions, "2020-01-02", 100000, config, "guard-test")
    assert result["rejections"][0]["reason"] == "conservative_upper_price_guard"
    nonmainboard = execute_decisions("300001.SZ", frame, decisions, "2020-01-02", 100000, config, "board-test")
    assert not nonmainboard["ledger"]
    assert all(item["reason"] == "unsupported_board" for item in nonmainboard["rejections"])


def test_shorting_config_fails_closed():
    config = copy.deepcopy(strategy_config())
    config["position"]["opens"][0]["operate"] = "开空"
    with pytest.raises(ValueError, match="只允许"):
        strategy_config(config)


def test_legacy_fuyao_is_preserved_but_cannot_fill(raw_db):
    with sqlite3.connect(raw_db) as connection:
        connection.execute("UPDATE stock_daily_normalized SET source='fuyao' WHERE date='2020-03-27'")
    frame = load_bars("000001", db_path=raw_db).iloc[:65].copy()
    assert len(frame) == 65
    assert frame.attrs["price_basis"] == "mixed_legacy_adjustment_unverified"
    assert frame.attrs["legacy_fuyao_bars"] == 1
    dates = frame["date"]
    legacy_index = int(frame.index[frame["source"] == "fuyao"][0])
    decisions = [{"date": date.date().isoformat(), "target_weight": 0 if index < legacy_index - 1 else 0.95} for index, date in enumerate(dates)]
    result = execute_decisions("000001.SZ", frame, decisions, "2020-01-02", 100000, strategy_config(), "legacy-test")
    rejected_dates = {item["date"] for item in result["rejections"] if item["reason"] == "legacy_fuyao_price_basis_unverified"}
    assert {dates.iloc[legacy_index].date().isoformat(), dates.iloc[legacy_index + 1].date().isoformat()} <= rejected_dates
    assert all(item["date"] not in rejected_dates for item in result["ledger"])
