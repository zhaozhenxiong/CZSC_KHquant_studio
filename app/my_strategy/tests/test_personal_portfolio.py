"""Manual portfolio persistence, selected quote valuation and HTTP validation."""
from __future__ import annotations

from contextlib import closing
import hashlib
import sqlite3

from fastapi.testclient import TestClient
import pytest

from my_strategy.storage.czsc_results import ResultStore
from my_strategy.storage.personal_portfolio import PersonalStore
from my_strategy.web_dashboard.app import create_app
from my_strategy.web_dashboard.tasks import TaskManager


@pytest.fixture
def store(tmp_path):
    raw = tmp_path / "raw.db"
    with closing(sqlite3.connect(raw)) as conn:
        conn.execute("CREATE TABLE securities (stock TEXT PRIMARY KEY,name TEXT)")
        conn.executemany("INSERT INTO securities VALUES (?,?)", [("000001.SZ", "平安银行"), ("600036.SH", "招商银行"), ("999999.SZ", "无行情测试")])
        conn.execute("CREATE TABLE stock_daily_normalized (stock TEXT,date TEXT,close REAL,source TEXT,has_trade_price INTEGER,PRIMARY KEY(stock,date))")
        conn.executemany("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?)", [
            ("000001.SZ", "2026-09-25", 8, "test", 1),
            ("000001.SZ", "2026-09-28", 12, "tushare", 1),
            ("600036.SH", "2026-09-25", 20, "fuyao", 1),
        ])
        conn.commit()
    return PersonalStore(tmp_path / "personal.db", raw)


def test_persisted_watchlist_is_additive_and_idempotent(store):
    store.add_watchlist(["000001.SZ", "000001.SZ"])
    created = store.watchlist()["items"][0]["added_at"]
    reopened = PersonalStore(store.db_path, store.raw_db)
    result = reopened.add_watchlist(["000001.SZ", "600036.SH"])
    assert [row["symbol"] for row in result["items"]] == ["000001.SZ", "600036.SH"]
    assert result["items"][0]["added_at"] == created
    assert reopened.remove_watchlist("000001.SZ")["coverage"]["total"] == 1
    assert store.watchlist()["items"][0]["symbol"] == "600036.SH"


def test_manual_holdings_persist_edit_and_remove_with_odd_lots(store):
    store.save_holding("000001.SZ", 137, 10)
    reopened = PersonalStore(store.db_path, store.raw_db)
    row = reopened.holdings()["items"][0]
    assert row["shares"] == 137
    assert row["cost_basis"] == 1370
    assert row["market_value"] == 1644
    assert row["unrealized_profit"] == 274
    assert row["unrealized_return_pct"] == pytest.approx(20)
    reopened.save_holding("000001.SZ", 151, 11)
    assert store.holdings()["coverage"]["total"] == 1
    assert store.holdings()["items"][0]["shares"] == 151
    assert reopened.remove_holding("000001.SZ")["items"] == []


def test_latest_local_quote_has_actual_date_source_and_readonly_raw(store):
    original = hashlib.sha256(store.raw_db.read_bytes()).hexdigest()
    row = store.add_watchlist(["000001.SZ", "600036.SH"])["items"][0]
    assert (row["name"], row["close"], row["price_date"], row["source"]) == ("平安银行", 12, "2026-09-28", "tushare")
    legacy = store.watchlist()["items"][1]
    assert legacy["price_date"] == "2026-09-25"
    assert legacy["price_basis"] == "mixed_legacy_adjustment_unverified"
    assert "Fuyao" in legacy["warnings"][0]
    assert hashlib.sha256(store.raw_db.read_bytes()).hexdigest() == original


def test_missing_prices_do_not_turn_partial_valuation_into_total(store):
    store.save_holding("000001.SZ", 100, 10)
    result = store.save_holding("999999.SZ", 200, 5)
    assert result["coverage"] == {"total": 2, "priced": 1, "missing": 1}
    assert result["totals"] == {"cost_basis": 2000, "priced_cost_basis": 1000, "market_value": 1200,
                                "unrealized_profit": 200, "unrealized_return_pct": 20, "complete": False}
    missing = result["items"][1]
    assert missing["name"] == "无行情测试"
    assert missing["close"] is None and missing["market_value"] is None and missing["unrealized_profit"] is None
    assert missing["warnings"]
    store.remove_holding("000001.SZ")
    assert store.holdings()["totals"]["market_value"] is None
    assert store.holdings()["totals"]["unrealized_profit"] is None


def test_missing_raw_db_keeps_manual_records_available(store):
    store.save_holding("000001.SZ", 100, 10)
    result = PersonalStore(store.db_path, store.raw_db.parent / "absent.db").holdings()
    assert result["items"][0]["shares"] == 100
    assert result["totals"]["market_value"] is None
    assert result["coverage"]["missing"] == 1
    assert "行情库不可用" in result["items"][0]["warnings"][0]


@pytest.mark.parametrize("price", [None, 0, -1, float("inf"), float("nan")])
def test_invalid_local_price_is_not_valued(store, price):
    with closing(sqlite3.connect(store.raw_db)) as conn:
        conn.execute("UPDATE stock_daily_normalized SET close=? WHERE stock=? AND date=?", (price, "000001.SZ", "2026-09-28"))
        conn.commit()
    row = store.save_holding("000001.SZ", 100, 10)["items"][0]
    assert not row["quote_available"]
    assert row["market_value"] is None


def test_storage_rejects_invalid_values_before_persisting(store):
    for shares, cost in [(0, 10), (-1, 10), (1.5, 10), (True, 10), (100, 0), (100, float("nan")), (100, float("inf")), (100, True)]:
        with pytest.raises(ValueError):
            store.save_holding("000001.SZ", shares, cost)
    with pytest.raises(ValueError):
        store.add_watchlist(["000001.SZ", "../escape"])
    assert store.holdings()["items"] == []
    assert store.watchlist()["items"] == []


def test_personal_http_crud_and_input_validation(store, tmp_path):
    manager = TaskManager(tmp_path / "jobs.db", ResultStore(tmp_path / "results.db", tmp_path / "runs"), workers=1)
    with TestClient(create_app(manager, store)) as client:
        assert client.get("/watchlist").status_code == 200
        assert client.get("/holdings").status_code == 200
        response = client.post("/api/watchlist", json={"symbols": ["000001.SZ", "600036.SH", "000001.SZ"]})
        assert response.status_code == 200
        assert response.json()["coverage"]["total"] == 2
        assert client.post("/api/watchlist", json={"symbols": ["bad"]}).status_code == 422
        assert client.post("/api/watchlist", json={"symbols": []}).status_code == 422
        assert client.delete("/api/watchlist/000001.SZ").json()["coverage"]["total"] == 1
        assert client.delete("/api/watchlist/000001.SZ").status_code == 200
        for shares in [0, -1, 1.5, "100", True, 1_000_000_001]:
            assert client.put("/api/holdings/000001.SZ", json={"shares": shares, "average_cost": 10}).status_code == 422
        for cost in [0, -1, "NaN", "Infinity", True, 1_000_000_001]:
            assert client.put("/api/holdings/000001.SZ", json={"shares": 100, "average_cost": cost}).status_code == 422
        assert client.put("/api/holdings/bad", json={"shares": 100, "average_cost": 10}).status_code == 400
        response = client.put("/api/holdings/000001.SZ", json={"shares": 137, "average_cost": 10})
        assert response.status_code == 200
        assert response.json()["items"][0]["market_value"] == 1644
        assert client.get("/api/holdings").json()["totals"]["unrealized_profit"] == 274
        assert client.delete("/api/holdings/000001.SZ").json()["items"] == []
        assert client.put("/api/holdings/000001.SZ", json={"shares": 100, "average_cost": 10}, headers={"Origin": "https://other.example"}).status_code == 403


def test_personal_records_survive_new_app_instance(store, tmp_path):
    results = ResultStore(tmp_path / "results.db", tmp_path / "runs")
    first = TaskManager(tmp_path / "jobs.db", results, workers=1)
    with TestClient(create_app(first, store)) as client:
        assert client.post("/api/watchlist", json={"symbols": ["000001.SZ"]}).status_code == 200
        assert client.put("/api/holdings/000001.SZ", json={"shares": 137, "average_cost": 10}).status_code == 200
    reopened = PersonalStore(store.db_path, store.raw_db)
    second = TaskManager(tmp_path / "jobs.db", results, workers=1)
    with TestClient(create_app(second, reopened)) as client:
        assert client.get("/api/watchlist").json()["items"][0]["symbol"] == "000001.SZ"
        assert client.get("/api/holdings").json()["items"][0]["shares"] == 137


def test_personal_endpoints_apply_global_auth(store, tmp_path, monkeypatch):
    monkeypatch.setenv("KHQUANT_API_TOKEN", "test-personal-secret")
    manager = TaskManager(tmp_path / "jobs.db", ResultStore(tmp_path / "results.db", tmp_path / "runs"), workers=1)
    with TestClient(create_app(manager, store)) as client:
        assert client.get("/api/watchlist").status_code == 401
        assert client.get("/api/holdings").status_code == 401
        assert client.post("/api/watchlist", json={"symbols": ["000001.SZ"]}).status_code == 401
        headers = {"Authorization": "Bearer test-personal-secret"}
        response = client.get("/api/holdings", headers=headers)
        assert response.status_code == 200 and "test-personal-secret" not in response.text
        assert client.put("/api/holdings/000001.SZ", json={"shares": 137, "average_cost": 10}, headers=headers).status_code == 200
