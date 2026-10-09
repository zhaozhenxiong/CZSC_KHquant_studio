"""Offline contract tests; no live requests or business database writes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from my_strategy.data_manager import market_data_provider as module
from my_strategy.data_manager.market_data_provider import (
    AkShareProvider, EastmoneyProvider, SinaUnadjustedProvider, ProviderResult,
)
from my_strategy.services.czsc_research_data import audit_data


def _curl(monkeypatch, payload, returncode=0):
    calls = []
    monkeypatch.setattr(module, "_curl_executable", lambda: "curl")

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 25
        assert "--noproxy" in command and "--max-time" in command
        assert command[command.index("--max-time") + 1] == "20"
        return subprocess.CompletedProcess(command, returncode, payload, "failure" if returncode else "")

    monkeypatch.setattr(module.subprocess, "run", run)
    return calls


def _eastmoney(code="600519", lines=None):
    return json.dumps({"rc": 0, "data": {"code": code, "klines": lines if lines is not None else [
        "2026-09-30,10,10.5,11,9,123,123456,20,5,0.5,1.2",
    ]}})


def _sina_rows():
    # Equal consecutive prices are legitimate bars, and a missing session must
    # remain missing even when a stock-capital table would contain that date.
    return [{"date": day + "T00:00:00.000Z", "open": 10, "high": 11, "low": 9,
             "close": 10.5, "volume": 12345, "amount": 123456}
            for day in ["2026-09-28", "2026-09-30", "2026-10-08"]]


def _bse_config(tmp_path, symbol="920000.BJ", raw_date="20201223", boundary="2021-11-15"):
    source = tmp_path / "bse_raw_api.json"
    source.write_text("[]", encoding="utf-8")
    manifest = tmp_path / "bse_boundaries.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "source_snapshot": {"file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "records": [{"symbol": symbol, "raw_fxssrq": raw_date, "bse_listing_boundary": boundary}],
    }), encoding="utf-8")
    return {"sina_unadjusted_v1": {"listing_boundaries_file": str(manifest)}}


def test_eastmoney_fixed_historical_contract_and_units(monkeypatch):
    calls = _curl(monkeypatch, _eastmoney())
    result = EastmoneyProvider().fetch_daily("600519.SH", "2026-09-01", "2026-10-08")
    assert result.ok, result.error
    query = parse_qs(urlsplit(calls[0][-1]).query)
    assert query["fqt"] == ["0"] and query["klt"] == ["101"]
    assert query["secid"] == ["1.600519"]
    assert query["beg"] == ["20260901"] and query["end"] == ["20261008"]
    assert "%2C" not in calls[0][-1]
    row = result.data.iloc[0]
    assert row.volume == 12300 and row.amount == 123456
    assert row.pct_chg == 5 and row.turnover == 1.2
    assert row.source == result.source == "eastmoney_unadjusted_v1"
    for column in ("open", "high", "low", "close"):
        assert row[f"trade_{column}"] == row[column]


@pytest.mark.parametrize("payload", [
    _eastmoney(code="000001"),
    _eastmoney(lines=[]),
    _eastmoney(lines=["2026-09-30,10,10"]),
    _eastmoney(lines=["invalid-date,10,10.5,11,9,123,123456,20,5,0.5,1.2"]),
    _eastmoney(lines=["2026-09-30,10,10.5,8,9,123,123456,20,5,0.5,1.2"]),
    _eastmoney(lines=["2026-09-30,10,10.5,11,9,-1,123456,20,5,0.5,1.2"]),
    _eastmoney(lines=["2026-09-30,10,10.5,11,9,123,nan,20,5,0.5,1.2"]),
    _eastmoney(lines=["2026-09-30,10,10.5,11,9,123,123456,20,5,0.5,1.2"] * 2),
    '{"rc":1,"data":null}', "not json",
])
def test_eastmoney_rejects_bad_source_payload(monkeypatch, payload):
    _curl(monkeypatch, payload)
    result = EastmoneyProvider().fetch_daily("600519.SH", "2026-09-01", "2026-10-08")
    assert not result.ok and result.data is None and result.error


def test_eastmoney_clips_requested_dates_without_inventing_rows(monkeypatch):
    _curl(monkeypatch, _eastmoney(lines=[
        f"{day},10,10.5,11,9,123,123456,20,5,0.5,1.2"
        for day in ["2026-09-28", "2026-09-30", "2026-10-08"]
    ]))
    result = EastmoneyProvider().fetch_daily("600519.SH", "2026-09-29", "2026-10-01")
    assert result.ok
    assert result.data.date.dt.strftime("%Y-%m-%d").tolist() == ["2026-09-30"]


@pytest.mark.parametrize("provider", [EastmoneyProvider, SinaUnadjustedProvider])
def test_adjusted_request_is_rejected_before_network(monkeypatch, provider):
    calls = _curl(monkeypatch, "")
    result = provider().fetch_daily("600519.SH", "2026-09-01", "2026-10-08", adjust="qfq")
    assert not result.ok and not calls


@pytest.mark.parametrize("provider", [EastmoneyProvider, SinaUnadjustedProvider])
def test_curl_failure_is_not_replaced_with_quotes(monkeypatch, provider):
    calls = _curl(monkeypatch, "", returncode=52)
    result = provider().fetch_daily("600519.SH", "2026-09-01", "2026-10-08")
    assert not result.ok and "curl exit=52" in result.error
    assert len(calls) == 1


@pytest.mark.parametrize(("code", "prefix"), [("600519.SH", "sh"), ("000001.SZ", "sz"), ("920000.BJ", "bj")])
def test_sina_uses_actual_prefix_original_dates_units_and_prices(monkeypatch, tmp_path, code, prefix):
    calls = _curl(monkeypatch, "compressed history")
    seen = []
    monkeypatch.setattr(SinaUnadjustedProvider, "_decode_history", staticmethod(
        lambda text, symbol: seen.append(symbol) or _sina_rows()))
    config = _bse_config(tmp_path) if prefix == "bj" else {}
    result = SinaUnadjustedProvider(config).fetch_daily(code, "2026-09-01", "2026-09-30")
    assert result.ok, result.error
    assert seen == [prefix + code.split(".")[0]]
    assert calls[0][-1].endswith(f"/{seen[0]}/hisdata_klc2/klc_kl.js")
    frame = result.data
    assert frame.date.dt.strftime("%Y-%m-%d").tolist() == ["2026-09-28", "2026-09-30"]
    assert frame.volume.tolist() == [12345, 12345]
    assert frame.amount.tolist() == [123456, 123456]
    assert set(frame.source) == {"sina_unadjusted_v1"}
    for column in ("open", "high", "low", "close"):
        assert frame[f"trade_{column}"].equals(frame[column])


@pytest.mark.parametrize("change", ["zero_ohlc", "missing_amount", "duplicate_date", "nan_volume", "invalid_date"])
def test_sina_rejects_invalid_rows_instead_of_filling_or_discarding(monkeypatch, change):
    rows = _sina_rows()
    if change == "zero_ohlc":
        rows[0].update({column: 0 for column in ("open", "high", "low", "close")})
    elif change == "missing_amount":
        del rows[0]["amount"]
    elif change == "duplicate_date":
        rows[1]["date"] = rows[0]["date"]
    elif change == "nan_volume":
        rows[0]["volume"] = float("nan")
    else:
        rows[0]["date"] = "invalid-date"
    _curl(monkeypatch, "compressed history")
    monkeypatch.setattr(SinaUnadjustedProvider, "_decode_history", staticmethod(lambda text, symbol: rows))
    result = SinaUnadjustedProvider().fetch_daily("600519.SH", "2026-09-01", "2026-10-08")
    assert not result.ok and result.data is None and result.error


@pytest.mark.parametrize(
    ("symbol", "raw_date", "boundary", "request_start", "expected"),
    [
        ("920000.BJ", "20201223", "2021-11-15", "2015-01-01", ["2021-11-15", "2022-12-27", "2026-10-08"]),
        ("920000.BJ", "20201223", "2021-11-15", "2021-11-16", ["2022-12-27", "2026-10-08"]),
        ("920001.BJ", "20221227", "2022-12-27", "2015-01-01", ["2022-12-27", "2026-10-08"]),
    ],
)
def test_sina_bse_history_starts_at_listing_or_later_request(monkeypatch, tmp_path, symbol, raw_date, boundary, request_start, expected):
    rows = [dict(_sina_rows()[0], date=day + "T00:00:00.000Z")
            for day in ["2019-05-23", "2019-07-12", "2021-11-12", "2021-11-15", "2022-12-27", "2026-10-08"]]
    for row in rows[:2]:
        row.update({column: 0 for column in ("open", "high", "low", "close")})
    _curl(monkeypatch, "compressed history")
    monkeypatch.setattr(SinaUnadjustedProvider, "_decode_history", staticmethod(lambda text, symbol: rows))
    config = _bse_config(tmp_path, symbol, raw_date, boundary)
    result = SinaUnadjustedProvider(config).fetch_daily(symbol, request_start, "2026-10-08")
    assert result.ok, result.error
    assert result.data.date.dt.strftime("%Y-%m-%d").tolist() == expected
    assert len(rows) == 6 and rows[0]["open"] == 0  # Original evidence remains unchanged.
    assert result.data.volume.tolist() == [12345] * len(expected)
    assert result.data.amount.tolist() == [123456] * len(expected)


def test_sina_bse_bad_price_inside_listing_range_is_rejected(monkeypatch, tmp_path):
    rows = _sina_rows()
    rows[0]["close"] = 0
    _curl(monkeypatch, "compressed history")
    monkeypatch.setattr(SinaUnadjustedProvider, "_decode_history", staticmethod(lambda text, symbol: rows))
    result = SinaUnadjustedProvider(_bse_config(tmp_path)).fetch_daily("920000.BJ", "2015-01-01", "2026-10-08")
    assert not result.ok and result.data is None and "OHLC" in result.error


@pytest.mark.parametrize("change", ["bad_json", "empty_records", "bad_date", "before_floor", "duplicate", "snapshot_hash", "missing_symbol"])
def test_sina_bse_bad_or_missing_metadata_fails_before_network(monkeypatch, tmp_path, change):
    config = _bse_config(tmp_path)
    path = tmp_path / "bse_boundaries.json"
    manifest = json.loads(path.read_text())
    if change == "bad_json":
        path.write_text("not json")
    else:
        if change == "empty_records":
            manifest["records"] = []
        elif change == "bad_date":
            manifest["records"][0]["bse_listing_boundary"] = "2021-02-30"
        elif change == "before_floor":
            manifest["records"][0]["bse_listing_boundary"] = "2020-12-23"
        elif change == "duplicate":
            manifest["records"] *= 2
        elif change == "snapshot_hash":
            (tmp_path / "bse_raw_api.json").write_text("modified")
        elif change == "missing_symbol":
            manifest["records"][0]["symbol"] = "920001.BJ"
        path.write_text(json.dumps(manifest))
    calls = _curl(monkeypatch, "")
    result = SinaUnadjustedProvider(config).fetch_daily("920000.BJ", "2015-01-01", "2026-10-08")
    assert not result.ok and result.data is None and result.error and not calls


@pytest.mark.parametrize("config", [{}, {"sina_unadjusted_v1": {"listing_boundaries_file": "missing_bse_boundaries.json"}}])
def test_sina_bse_missing_boundary_configuration_fails_closed(monkeypatch, config):
    calls = _curl(monkeypatch, "")
    result = SinaUnadjustedProvider(config).fetch_daily("920000.BJ", "2015-01-01", "2026-10-08")
    assert not result.ok and result.data is None and result.error and not calls


@pytest.mark.parametrize(("raw_date", "boundary", "request_end"), [
    ("20201223", "2021-11-15", "2021-11-12"),
    ("20270101", "2027-01-01", "2026-10-08"),
])
def test_sina_pre_bse_request_does_not_fetch_or_invent_sessions(monkeypatch, tmp_path, raw_date, boundary, request_end):
    calls = _curl(monkeypatch, "")
    config = _bse_config(tmp_path, raw_date=raw_date, boundary=boundary)
    result = SinaUnadjustedProvider(config).fetch_daily("920000.BJ", "2015-01-01", request_end)
    assert not result.ok and "no BSE-listed sessions" in result.error and not calls


@pytest.mark.parametrize("symbol", ["600519.SH", "000001.SZ"])
def test_sina_mainland_markets_do_not_require_bse_metadata(monkeypatch, symbol):
    _curl(monkeypatch, "compressed history")
    monkeypatch.setattr(SinaUnadjustedProvider, "_decode_history", staticmethod(lambda text, symbol: _sina_rows()))
    config = {"sina_unadjusted_v1": {"listing_boundaries_file": "missing_bse_boundaries.json"}}
    result = SinaUnadjustedProvider(config).fetch_daily(symbol, "2026-09-01", "2026-10-08")
    assert result.ok, result.error
    assert result.data.date.dt.strftime("%Y-%m-%d").tolist() == ["2026-09-28", "2026-09-30", "2026-10-08"]


def test_sina_factory_uses_explicit_listing_configuration(tmp_path):
    config = _bse_config(tmp_path)
    sina = next(provider for provider in module._default_providers(config) if isinstance(provider, SinaUnadjustedProvider))
    assert sina.config is config
    assert sina._bse_listing_boundary("920000.BJ") == pd.Timestamp("2021-11-15")


def test_sina_decoder_rejects_response_for_another_security():
    with pytest.raises(ValueError, match="code does not match"):
        SinaUnadjustedProvider._decode_history('var KLC_K2_sz000001="encoded";', "sh600519")


def test_sina_decoder_only_evaluates_installed_decoder(monkeypatch):
    from akshare.stock.cons import hk_js_decode
    import py_mini_racer

    calls = []

    class Decoder:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def eval(self, script):
            assert script == hk_js_decode

        def call(self, function, argument):
            calls.append((function, argument))
            return _sina_rows()

    monkeypatch.setattr(py_mini_racer, "MiniRacer", Decoder)
    rows = SinaUnadjustedProvider._decode_history(
        'var KLC_K2_sh600519="encoded"; /* upstream trailing comment */', "sh600519")
    assert calls == [("d", "encoded")] and rows == _sina_rows()


@pytest.mark.parametrize("provider", [EastmoneyProvider, SinaUnadjustedProvider])
def test_basic_delegation_preserves_source_identity(monkeypatch, provider):
    expected = ProviderResult(True, pd.DataFrame({"source": ["akshare:stock_info_a_code_name"]}), "akshare", "")
    monkeypatch.setattr(AkShareProvider, "fetch_basic", lambda self, **kwargs: expected)
    assert provider().fetch_basic(min_count=1) is expected


@pytest.mark.parametrize("provider", [EastmoneyProvider, SinaUnadjustedProvider])
def test_approved_contracts_pass_readonly_research_audit(monkeypatch, tmp_path, provider):
    if provider is EastmoneyProvider:
        _curl(monkeypatch, _eastmoney())
    else:
        _curl(monkeypatch, "compressed history")
        monkeypatch.setattr(SinaUnadjustedProvider, "_decode_history", staticmethod(lambda text, symbol: _sina_rows()))
    result = provider().fetch_daily("600519.SH", "2026-09-30", "2026-09-30")
    assert result.ok
    row = result.data.iloc[0]
    database = tmp_path / "isolated_raw.db"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE securities(stock TEXT PRIMARY KEY);
            CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,
                close REAL,volume REAL,amount REAL,source TEXT,has_trade_price INTEGER);
        """)
        connection.execute("INSERT INTO securities VALUES (?)", (row.code,))
        connection.execute("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)",
            (row.code, "2026-09-30", float(row.trade_open), float(row.trade_high), float(row.trade_low),
             float(row.trade_close), float(row.volume), float(row.amount), row.source, 1))
    audited = audit_data("2026-09-30", min_history=1, db_path=database)
    assert audited["tradable_symbols"] == ["600519.SH"]
