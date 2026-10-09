"""Original Tencent STAR history contracts with offline source responses."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from my_strategy.data_manager import market_data_provider as module
from my_strategy.data_manager.market_data_provider import TencentStarUnadjustedProvider
from my_strategy.services.czsc_research_data import audit_data


def _bar(day, close="10.50"):
    return [day, "10.00", close, "11.00", "9.00", "12345.00", {}, "1.20", "12.34", "0.00", "0.00"]


def _payload(rows, symbol="sh688089"):
    return {"code": 0, "msg": "", "data": {symbol: {"day": rows, "qt": {symbol: ["unusable realtime quote"]}}}}


def _curl(monkeypatch, responses):
    calls = []
    responses = iter(responses)
    monkeypatch.setattr(module, "_curl_executable", lambda: "curl")

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 25 and command[command.index("--max-time") + 1] == "20"
        assert command[command.index("--noproxy") + 1] == "*"
        response = next(responses)
        if isinstance(response, subprocess.CompletedProcess):
            return response
        text = response if isinstance(response, str) else json.dumps(response)
        return subprocess.CompletedProcess(command, 0, text, "")

    monkeypatch.setattr(module.subprocess, "run", run)
    return calls


def test_original_day_share_units_amount_precision_and_actual_missing_date(monkeypatch):
    calls = _curl(monkeypatch, [_payload([_bar("2024-11-04"), _bar("2024-11-12")])])
    result = TencentStarUnadjustedProvider().fetch_daily("688089.SH", "2024-11-01", "2024-11-15")
    assert result.ok, result.error
    assert parse_qs(urlsplit(calls[0][-1]).query)["param"] == ["sh688089,day,2024-01-01,2024-11-15,640,"]
    frame = result.data
    assert frame.date.dt.strftime("%Y-%m-%d").tolist() == ["2024-11-04", "2024-11-12"]
    assert frame.volume.tolist() == [12345, 12345]
    assert frame.amount.tolist() == [123400, 123400]
    assert frame.turnover.tolist() == [1.2, 1.2]
    assert frame.attrs["amount_precision_cny"] == 100
    assert set(frame.source) == {"tencent_star_unadjusted_v1"}
    for column in ("open", "close", "high", "low"):
        assert frame[column].equals(frame[f"trade_{column}"])


def test_every_year_is_requested_and_identical_overlap_is_not_duplicated(monkeypatch):
    calls = _curl(monkeypatch, [
        {"code": 0, "data": []},
        _payload([_bar("2020-12-31")]),
        _payload([_bar("2020-12-31"), _bar("2021-01-04")]),
    ])
    result = TencentStarUnadjustedProvider().fetch_daily("688089.SH", "2019-01-01", "2021-01-05")
    assert result.ok, result.error
    assert [parse_qs(urlsplit(call[-1]).query)["param"][0] for call in calls] == [
        "sh688089,day,2019-01-01,2019-12-31,640,",
        "sh688089,day,2020-01-01,2020-12-31,640,",
        "sh688089,day,2021-01-01,2021-01-05,640,",
    ]
    assert result.data.date.dt.strftime("%Y-%m-%d").tolist() == ["2020-12-31", "2021-01-04"]


def test_overlap_with_revised_history_rejects_entire_fetch(monkeypatch):
    _curl(monkeypatch, [
        _payload([_bar("2020-12-31")]),
        _payload([_bar("2020-12-31", close="10.60"), _bar("2021-01-04")]),
    ])
    result = TencentStarUnadjustedProvider().fetch_daily("688089.SH", "2020-01-01", "2021-01-05")
    assert not result.ok and result.data is None and "conflicting" in result.error


@pytest.mark.parametrize("code", ["600519.SH", "000001.SZ", "920000.BJ", "688089.SZ", "689009.SH"])
def test_unverified_other_market_units_are_rejected_without_network(monkeypatch, code):
    calls = _curl(monkeypatch, [])
    result = TencentStarUnadjustedProvider().fetch_daily(code, "2015-01-01", "2026-10-08")
    assert not result.ok and "SH688 stocks only" in result.error and not calls


@pytest.mark.parametrize("change", [
    "bad_json", "api_error", "wrong_symbol", "quote_only", "adjusted", "short_bar",
    "invalid_date", "duplicate_date", "unordered_dates", "zero_price", "nan_volume", "negative_amount", "amount_overflow", "future_date",
])
def test_bad_original_source_response_cannot_be_replaced_by_quotes(monkeypatch, change):
    payload = _payload([_bar("2024-11-04"), _bar("2024-11-12")])
    data = payload["data"]["sh688089"]
    if change == "bad_json":
        payload = "not json"
    elif change == "api_error":
        payload["code"] = 1
    elif change == "wrong_symbol":
        payload["data"]["sh688143"] = payload["data"].pop("sh688089")
    elif change == "quote_only":
        del data["day"]
    elif change == "adjusted":
        data["qfqday"] = data.pop("day")
    elif change == "short_bar":
        data["day"][0] = data["day"][0][:6]
    elif change == "invalid_date":
        data["day"][0][0] = "2024-02-30"
    elif change == "duplicate_date":
        data["day"][1][0] = data["day"][0][0]
    elif change == "unordered_dates":
        data["day"].reverse()
    elif change == "zero_price":
        data["day"][0][1] = "0"
    elif change == "nan_volume":
        data["day"][0][5] = "nan"
    elif change == "negative_amount":
        data["day"][0][8] = "-1"
    elif change == "amount_overflow":
        data["day"][0][8] = "1e308"
    elif change == "future_date":
        data["day"][1][0] = "2025-01-01"
    _curl(monkeypatch, [payload])
    result = TencentStarUnadjustedProvider().fetch_daily("688089.SH", "2024-01-01", "2024-12-31")
    assert not result.ok and result.data is None and result.error


def test_failed_year_does_not_return_only_successful_prior_years(monkeypatch):
    _curl(monkeypatch, [_payload([_bar("2020-12-31")]), subprocess.CompletedProcess([], 52, "", "Empty reply")])
    result = TencentStarUnadjustedProvider().fetch_daily("688089.SH", "2020-01-01", "2021-01-05")
    assert not result.ok and result.data is None and "2021" in result.error and "curl exit=52" in result.error


def test_reduced_server_cap_cannot_silently_truncate_year(monkeypatch):
    rows = [_bar(day.date().isoformat()) for day in pd.date_range("2024-02-16", periods=320)]
    _curl(monkeypatch, [_payload(rows)])
    result = TencentStarUnadjustedProvider().fetch_daily("688089.SH", "2024-01-01", "2024-12-31")
    assert not result.ok and result.data is None and "truncated" in result.error


def test_tencent_verified_source_does_not_bypass_research_market_guard(tmp_path, monkeypatch):
    _curl(monkeypatch, [_payload([_bar("2024-11-12")])])
    result = TencentStarUnadjustedProvider().fetch_daily("688089.SH", "2024-11-12", "2024-11-12")
    assert result.ok, result.error
    row = result.data.iloc[0]
    path = tmp_path / "raw.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE securities(stock TEXT PRIMARY KEY);
            CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,
                close REAL,volume REAL,amount REAL,source TEXT,has_trade_price INTEGER);
        """)
        connection.execute("INSERT INTO securities VALUES (?)", (row.code,))
        connection.execute("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)", (
            row.code, "2024-11-12", float(row.open), float(row.high), float(row.low), float(row.close),
            float(row.volume), float(row.amount), row.source, 1,
        ))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    audited = audit_data("2024-11-12", min_history=1, db_path=path)
    assert audited["eligible_symbols"] == ["688089.SH"]
    assert audited["tradable_symbols"] == []  # Existing main-board execution scope remains.
    assert "unverified_source" not in audited["records"][0]["issues"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_default_source_factory_registers_exact_restricted_contract():
    provider = next(provider for provider in module._default_providers({"data_sources": {}})
                    if provider.name == "tencent_star_unadjusted_v1")
    assert isinstance(provider, TencentStarUnadjustedProvider)
