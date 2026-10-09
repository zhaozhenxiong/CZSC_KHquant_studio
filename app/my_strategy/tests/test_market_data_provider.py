"""Tests for the unified market data provider layer."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from my_strategy.data_manager import market_data_provider as provider_module
from my_strategy.data_manager.market_data_provider import (
    AkShareProvider,
    BaostockProvider,
    FuyaoProvider,
    LocalCSVProvider,
    MarketDataProvider,
    ProviderResult,
    RateLimiter,
    TencentRealtimeProvider,
    TushareProvider,
    UnifiedDataProvider,
    _normalize_minute_frame,
    _normalize_stock_code,
)
from my_strategy.data_manager.source_status import load_source_status
from my_strategy.data_manager.stock_pool import normalize_stock_code


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class DummyProvider(MarketDataProvider):
    _default_name = "dummy"

    def __init__(self, name: str = "dummy", data: dict[tuple[str, str, str], pd.DataFrame] | None = None):
        self._name = name
        self.data = data or {}
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def healthcheck(self) -> ProviderResult:
        return ProviderResult(True, pd.DataFrame(), self.name, "")

    def fetch_daily(self, code: str, start_date: str, end_date: str, **kwargs: Any) -> ProviderResult:
        self.call_count += 1
        key = (code, start_date, end_date)
        if key in self.data:
            df = self.data[key]
            return ProviderResult(not df.empty, df, self.name, "")
        return ProviderResult(False, None, self.name, "no data")

    def fetch_minute(self, code: str, period: str, start: str, end: str, **kwargs: Any) -> ProviderResult:
        return ProviderResult(False, None, self.name, "not implemented")

    def fetch_basic(self, **kwargs: Any) -> ProviderResult:
        return ProviderResult(False, None, self.name, "not implemented")


@pytest.mark.parametrize("normalize", [_normalize_stock_code, normalize_stock_code])
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("000001", "000001.SZ"),
        ("600000", "600000.SH"),
        (" 300750 ", "300750.SZ"),
        ("430047", "430047.BJ"),
        ("830799", "830799.BJ"),
        ("920005", "920005.BJ"),
        ("600000.SH", "600000.SH"),
        ("000001.SH", "000001.SH"),
        ("920005.BJ", "920005.BJ"),
        (" 830799.bj ", "830799.BJ"),
        ("600000.SZ", "600000.SZ"),
    ],
)
def test_normalize_stock_code(normalize, code, expected):
    assert normalize(code) == expected


def test_fuyao_provider_normalizes_daily_response(monkeypatch):
    payload = b'{"code":0,"data":{"item":[{"date_ms":1725206400000,"open_price":10,"high_price":11,"low_price":9.5,"close_price":10.5,"volume":1000,"turnover":10500}]}}'
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key")

    def fake_urlopen(request, timeout):
        assert request.get_header("X-api-key") == "test-key"
        assert "thscode=000001.SZ" in request.full_url
        assert "adjust=none" in request.full_url
        return _FakeResponse(payload)

    monkeypatch.setattr("my_strategy.data_manager.market_data_provider.urlopen", fake_urlopen)
    result = FuyaoProvider().fetch_daily("000001.SZ", "2024-09-02", "2024-09-02")

    assert result.ok
    assert result.source == "fuyao_unadjusted_v1"
    assert result.data["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-09-02"]
    assert result.data["source"].tolist() == ["fuyao_unadjusted_v1"]
    assert result.data["amount"].tolist() == [10500]


def test_unified_provider_uses_source_specific_rate_limits():
    provider = DummyProvider(name="fuyao")
    unified = UnifiedDataProvider(
        config={
            "data_sources": {"primary": "fuyao"},
            "provider_rate_limits": {"fuyao": {"calls": 15, "per_seconds": 1}},
        },
        providers=[provider],
    )

    limiter = unified._limiter_for("fuyao")

    assert limiter.calls == 15
    assert limiter.per_seconds == 1


def test_provider_result_dataclass():
    result = ProviderResult(True, pd.DataFrame({"x": [1]}), "dummy", "")
    assert result.ok is True
    assert result.source == "dummy"


def test_rate_limiter_enforces_delay():
    limiter = RateLimiter(calls=10.0, per_seconds=1.0)
    start = pd.Timestamp.now()
    for _ in range(3):
        limiter.acquire()
    elapsed = (pd.Timestamp.now() - start).total_seconds()
    assert elapsed < 0.5


def test_unified_provider_picks_first_successful_source():
    fail = DummyProvider(name="fail")
    ok = DummyProvider(
        name="ok",
        data={
            ("000001.SZ", "2024-01-01", "2024-01-05"): pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=5),
                    "open": [1.0] * 5,
                    "high": [1.1] * 5,
                    "low": [0.9] * 5,
                    "close": [1.0] * 5,
                    "volume": [100] * 5,
                }
            )
        },
    )
    fail2 = DummyProvider(name="fail2")
    unified = UnifiedDataProvider(
        config={"data_sources": {"primary": "fail", "fallback": ["fail2", "ok"]}},
        providers=[fail, fail2, ok],
    )
    result = unified.fetch_daily("000001.SZ", "2024-01-01", "2024-01-05")
    assert result.ok
    assert result.source == "ok"
    assert len(result.data) == 5


def test_unified_provider_returns_failure_when_all_sources_fail():
    fail = DummyProvider(name="fail")
    unified = UnifiedDataProvider(
        config={"data_sources": {"primary": "fail"}},
        providers=[fail],
    )
    result = unified.fetch_daily("000001.SZ", "2024-01-01", "2024-01-05")
    assert not result.ok
    assert "no data" in result.error


def test_default_provider_without_tushare_credentials_reaches_fallback_without_network(monkeypatch, tmp_path):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(TushareProvider, "_global_failed", False)
    config = {
        "tushare": {"token": ""},
        "data_sources": {"primary": "tushare", "fallback": ["akshare"]},
        "data_root": str(tmp_path),
    }
    monkeypatch.setattr(provider_module, "load_data_config", lambda: config)
    monkeypatch.setenv("KHQUANT_SOURCE_STATUS_PATH", str(tmp_path / "source-status.json"))

    def unexpected_network(*args, **kwargs):
        pytest.fail("Default provider initialization must not access the network")

    monkeypatch.setattr("socket.socket.connect", unexpected_network)
    expected = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "close": [10.0]})
    calls = []

    def fake_daily(self, code, start_date, end_date, **kwargs):
        calls.append((code, start_date, end_date))
        return ProviderResult(True, expected, self.name, "")

    monkeypatch.setattr(AkShareProvider, "fetch_daily", fake_daily)
    unified = UnifiedDataProvider()

    assert "tushare" not in unified._providers
    assert {"akshare", "eastmoney_unadjusted_v1", "sina_unadjusted_v1", "tencent_star_unadjusted_v1", "fuyao", "baostock", "tencent_realtime", "local_csv"} == set(unified._providers)
    assert not TushareProvider._global_failed
    result = unified.fetch_daily("000001.SZ", "2024-01-02", "2024-01-02")

    assert result.ok
    assert result.source == "akshare"
    pd.testing.assert_frame_equal(result.data, expected)
    assert calls == [("000001.SZ", "2024-01-02", "2024-01-02")]
    status = load_source_status(tmp_path / "source-status.json")[-1]
    assert status["source"] == "akshare:fetch_daily"
    assert status["fallback_used"]
    assert status["fallback_source"] == "tushare"


@pytest.mark.parametrize("credential_source", ["config", "environment"])
def test_default_provider_keeps_configured_tushare(monkeypatch, credential_source):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    config = {"tushare": {"token": ""}}
    token = f"{credential_source}-token"
    if credential_source == "config":
        config["tushare"]["token"] = token
    else:
        monkeypatch.setenv("TUSHARE_TOKEN", token)

    unified = UnifiedDataProvider(config=config)

    assert isinstance(unified._providers["tushare"], TushareProvider)
    assert unified._providers["tushare"]._token == token


@pytest.mark.parametrize("method,kwargs", [
    ("fetch_daily", {"code": "000001.SZ", "start_date": "2024-01-02", "end_date": "2024-01-02"}),
    ("fetch_daily_batch", {"codes": ["000001.SZ"], "start_date": "2024-01-02", "end_date": "2024-01-02"}),
    ("fetch_minute", {"code": "000001.SZ", "period": "5", "start": "2024-01-02", "end": "2024-01-02"}),
    ("fetch_basic", {}),
])
def test_explicit_unified_tushare_without_credentials_fails_without_fallback(monkeypatch, method, kwargs):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(provider_module, "load_data_config", lambda: {"tushare": {"token": ""}})

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("Explicit Tushare requests must not call another source")

    monkeypatch.setattr(AkShareProvider, method if method != "fetch_daily_batch" else "fetch_daily", unexpected_fallback)
    unified = UnifiedDataProvider()
    result = getattr(unified, method)(sources=["tushare"], **kwargs)

    assert not result.ok
    assert result.data is None
    assert "Tushare token not configured" in result.error


def test_explicit_tushare_without_credentials_keeps_clear_error(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(provider_module, "load_data_config", lambda: {"tushare": {"token": ""}})
    monkeypatch.setattr(TushareProvider, "_global_failed", False)

    with pytest.raises(RuntimeError, match="Tushare token not configured"):
        TushareProvider()._pro_api()


def test_unified_provider_fetch_basic_routes_to_provider():
    class BasicProvider(DummyProvider):
        def __init__(self):
            super().__init__(name="basic")

        def fetch_basic(self, **kwargs: Any) -> ProviderResult:
            self.call_count += 1
            return ProviderResult(True, pd.DataFrame({"code": ["000001.SZ"], "name": ["平安银行"]}), self.name, "")

    provider = BasicProvider()
    unified = UnifiedDataProvider(
        config={"data_sources": {"primary": "basic"}},
        providers=[provider],
    )
    result = unified.fetch_basic()
    assert result.ok
    assert result.source == "basic"
    assert provider.call_count == 1


def test_local_csv_provider_reads_existing_parquet(tmp_path: Path):
    data_root = tmp_path / "data"
    stock_dir = data_root / "raw" / "stock_daily"
    stock_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "open": [1.0] * 3,
            "high": [1.1] * 3,
            "low": [0.9] * 3,
            "close": [1.0] * 3,
            "volume": [100] * 3,
        }
    )
    df.to_parquet(stock_dir / "000001.SZ.parquet", index=False)
    provider = LocalCSVProvider(config={"data_root": str(data_root)})
    result = provider.fetch_daily("000001.SZ", "2024-01-01", "2024-01-03")
    assert result.ok
    assert len(result.data) == 3


def test_local_csv_provider_respects_date_range(tmp_path: Path):
    data_root = tmp_path / "data"
    stock_dir = data_root / "raw" / "stock_daily"
    stock_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "open": [1.0] * 5,
            "high": [1.1] * 5,
            "low": [0.9] * 5,
            "close": [1.0] * 5,
            "volume": [100] * 5,
        }
    )
    df.to_parquet(stock_dir / "000001.SZ.parquet", index=False)
    provider = LocalCSVProvider(config={"data_root": str(data_root)})
    result = provider.fetch_daily("000001.SZ", "2024-01-02", "2024-01-04")
    assert result.ok
    assert len(result.data) == 3


def test_normalize_minute_frame_basic():
    df = pd.DataFrame(
        {
            "datetime": ["2024-01-01 09:31:00", "2024-01-01 09:32:00"],
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [100, 200],
        }
    )
    out = _normalize_minute_frame(df, "000001.SZ", "5")
    assert len(out) == 2
    assert set(out["period"]) == {"5"}


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("akshare") is None,
    reason="akshare not installed",
)
def test_akshare_provider_healthcheck_smoke():
    provider = AkShareProvider()
    result = provider.healthcheck()
    assert isinstance(result, ProviderResult)


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("baostock") is None,
    reason="baostock not installed",
)
def test_baostock_provider_healthcheck_smoke():
    provider = BaostockProvider()
    result = provider.healthcheck()
    assert isinstance(result, ProviderResult)
