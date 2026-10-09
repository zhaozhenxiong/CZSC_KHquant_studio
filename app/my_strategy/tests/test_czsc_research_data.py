import hashlib
import sqlite3

import pandas as pd
import pytest

from my_strategy.services.czsc_research_data import audit_data
from my_strategy.scripts.repair_czsc_research_data import normalize_tushare_daily


def _database(tmp_path):
    path = tmp_path / "raw.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE securities(stock TEXT PRIMARY KEY);
            CREATE TABLE stock_daily_normalized(
                stock TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
                volume REAL, amount REAL, source TEXT, has_trade_price INTEGER,
                PRIMARY KEY(stock,date));
        """)
        for symbol, source, volume, amount, flag in [
            ("600000.SH", "tushare:daily", 100, 1000, 1),
            ("300001.SZ", "tushare", 100, 1000, 1),
            ("000001.SZ", "tushare", 0, 0, 1),
            ("000002.SZ", "tencent_realtime", 100, 0, 1),
            ("000003.SZ", "fuyao", 100, 1000, 1),
            ("000004.SZ", "tushare", 100, None, 0),
        ]:
            connection.execute("INSERT INTO securities VALUES (?)", (symbol,))
            for day in ("2026-09-29", "2026-09-30"):
                connection.execute("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)",
                                   (symbol, day, 10, 11, 9, 10, volume, amount, source, flag))
        connection.execute("INSERT INTO securities VALUES ('000005.SZ')")
    return path


def test_audit_is_read_only_and_preserves_missing_suspension_and_sources(tmp_path):
    path = _database(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = audit_data("2026-09-30", 2, path)
    rows = {r["symbol"]: r for r in result["records"]}
    assert rows["600000.SH"]["tradable"]
    assert rows["300001.SZ"]["eligible"] and not rows["300001.SZ"]["tradable"]
    assert rows["000001.SZ"]["eligible"] and rows["000001.SZ"]["zero_activity_count"] == 2
    assert not rows["000001.SZ"]["target_active"]
    assert rows["000002.SZ"]["issues"]["inconsistent_volume_amount"] == 2
    assert rows["000003.SZ"]["source_issues"] == {"fuyao": 2}
    assert rows["000004.SZ"]["issues"]["missing_or_nonfinite_ohlcv"] == 2
    assert not rows["000005.SZ"]["target_covered"] and not rows["000005.SZ"]["history_available"]
    assert result["counts"] == {"symbols": 7, "target_covered": 6, "history_available": 6,
                                "eligible": 3, "tradable": 1, "zero_activity_symbols": 1}
    assert result["market_dates_basis"] == "observed_raw_union_not_verified_calendar"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_future_bars_do_not_change_past_audit(tmp_path):
    path = _database(tmp_path)
    before = audit_data("2026-09-30", 2, path)
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)",
                           ("600000.SH", "2026-10-08", 100, 110, 90, 100, 100, 10000, "fuyao", 0))
    assert audit_data("2026-09-30", 2, path) == before


def test_missing_market_observation_is_not_fabricated_as_zero_volume(tmp_path):
    path = _database(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM stock_daily_normalized WHERE stock='600000.SH' AND date='2026-09-29'")
        connection.execute("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)",
                           ("600000.SH", "2026-09-28", 10, 11, 9, 10, 100, 1000, "tushare", 1))
    record = next(r for r in audit_data("2026-09-30", 2, path)["records"] if r["symbol"] == "600000.SH")
    assert record["window_dates"] == ["2026-09-28", "2026-09-30"]
    assert record["missing_observed_dates"] == ["2026-09-29"]
    assert record["zero_activity_count"] == 0


def test_historical_normalization_converts_units_without_replacing_prices():
    raw = pd.DataFrame([{"ts_code": "600000.SH", "trade_date": "20260930", "open": 10,
                         "high": 11, "low": 9, "close": 10.5, "vol": 123, "amount": 100,
                         "pct_chg": 1}])
    frame = normalize_tushare_daily(raw, "2026-09-30")
    assert frame.volume.tolist() == [12300]
    assert frame.amount.tolist() == [100000]
    assert frame.trade_open.tolist() == [10]
    assert frame.source.tolist() == ["tushare:daily"]
    with pytest.raises(ValueError, match="Unexpected"):
        normalize_tushare_daily(raw, "2026-09-29")
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_tushare_daily(pd.concat([raw, raw]), "2026-09-30")
    raw.loc[0, "amount"] = 0
    with pytest.raises(ValueError, match="Inconsistent"):
        normalize_tushare_daily(raw, "2026-09-30")
