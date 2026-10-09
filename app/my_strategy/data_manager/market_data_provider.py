"""Unified market data provider abstraction with failover and audit.

This module defines a single interface for fetching A-share market data from
multiple upstream sources.  The ``UnifiedDataProvider`` reads the project data
configuration, applies rate limiting and retries, falls back across sources, and
records every fetch outcome into ``source_status.py`` for observability.

The layer is deliberately thin: providers only fetch and normalize.  Local cache
writes remain the responsibility of ``LocalCache`` and the storage layer.
"""

from __future__ import annotations

import abc
import contextlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .config import load_data_config, project_path
from .source_status import DataSourceStatus, append_source_status
from .validator import DataValidator

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def _akshare_clean_env():
    """Temporarily remove system proxy variables while akshare talks to Eastmoney.

    The system HTTP(S) proxy (commonly 127.0.0.1:7897) is unstable for
    push2his.eastmoney.com and leads to ``ProxyError`` / ``Remote end closed
    connection``.  curl reaches the same host fine without the proxy, so we
    clear the variables inside the AkShare call boundary.

    Important: ``requests`` reads proxy settings not only from environment
    variables but also from macOS system preferences via ``getproxies()``.
    Setting ``NO_PROXY=*`` forces ``requests`` to bypass the proxy for the
    duration of the call.
    """
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy")
    saved: dict[str, str | None] = {}
    for key in proxy_keys:
        saved[key] = os.environ.get(key)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)


def _akshare_call(func, *args, **kwargs):
    """Invoke an akshare function with proxy variables removed."""
    with _akshare_clean_env():
        return func(*args, **kwargs)


DEFAULT_DAILY_COLUMNS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "pct_chg",
    "trade_open",
    "trade_high",
    "trade_low",
    "trade_close",
    "source",
    "updated_at",
]


@dataclass
class ProviderResult:
    ok: bool
    data: pd.DataFrame | None = None
    source: str = ""
    error: str = ""


class RateLimiter:
    """Thread-safe token-bucket style rate limiter for provider calls."""

    def __init__(self, calls: float = 1.0, per_seconds: float = 1.0):
        self.calls = max(float(calls), 1.0)
        self.per_seconds = max(float(per_seconds), 1e-6)
        # Start with a single token: a full bucket (``calls`` tokens upfront)
        # lets callers burst past the configured cap in the first window, which
        # trips tushare's strict 50-calls/minute rolling limit.  One token keeps
        # the first call immediate while the sustained rate never exceeds
        # calls/per_seconds.
        self._tokens = min(float(self.calls), 1.0)
        self._last = time.monotonic()
        self._mutex = threading.Lock()

    def acquire(self) -> None:
        # Loop-based token bucket.  The old implementation slept while holding a
        # released-lock dance and unconditionally decremented tokens after the
        # sleep, so under multi-thread contention every sleeper drove _tokens
        # negative and overwrote _last, starving the refill to ~1 call/min (with
        # a 50/60 config, 16 workers took hours).  Here a thread that cannot get
        # a token simply sleeps outside the lock and re-checks; _last always
        # reflects real wall-clock elapsed, so the sustained rate stays at
        # calls/per_seconds regardless of worker count.
        while True:
            with self._mutex:
                now = time.monotonic()
                self._tokens = min(self.calls, self._tokens + (now - self._last) * self.calls / self.per_seconds)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) * self.per_seconds / self.calls
            time.sleep(wait)


class MarketDataProvider(abc.ABC):
    """Abstract A-share market data provider."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable provider identifier."""

    @abc.abstractmethod
    def healthcheck(self) -> ProviderResult:
        """Return a lightweight health probe result."""

    @abc.abstractmethod
    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> ProviderResult:
        """Fetch daily OHLCV bars for ``code`` in the inclusive date range."""

    @abc.abstractmethod
    def fetch_minute(
        self,
        code: str,
        period: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> ProviderResult:
        """Fetch minute bars for ``code`` between ``start`` and ``end``."""

    @abc.abstractmethod
    def fetch_basic(self, **kwargs: Any) -> ProviderResult:
        """Fetch the current all-A stock basic information table."""


class AkShareProvider(MarketDataProvider):
    """AkShare-based daily, minute, and basic data provider."""

    name = "akshare"

    def healthcheck(self) -> ProviderResult:
        try:
            import akshare as ak

            spot = _akshare_call(ak.stock_zh_a_spot_em)
            if spot is None or spot.empty:
                return ProviderResult(False, None, self.name, "empty spot response")
            return ProviderResult(True, spot.head(1), self.name, "")
        except Exception as exc:
            return ProviderResult(False, None, self.name, str(exc))

    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> ProviderResult:
        try:
            return self._fetch_daily(code, start_date, end_date)
        except Exception as exc:
            return ProviderResult(False, None, self.name, str(exc))

    def fetch_minute(
        self,
        code: str,
        period: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> ProviderResult:
        try:
            return self._fetch_minute(code, period, start, end)
        except Exception as exc:
            return ProviderResult(False, None, self.name, str(exc))

    def fetch_basic(self, **kwargs: Any) -> ProviderResult:
        import akshare as ak

        min_count = int(kwargs.get("min_count", 3000))

        def _from_spot() -> pd.DataFrame:
            spot = _akshare_call(ak.stock_zh_a_spot_em)
            rename = {
                "代码": "code",
                "名称": "name",
                "最新价": "latest_price",
                "总市值": "total_market_cap",
                "成交额": "turnover",
            }
            out = spot.rename(columns={k: v for k, v in rename.items() if k in spot.columns}).copy()
            if "code" not in out.columns or "name" not in out.columns:
                raise ValueError(f"missing code/name columns: {out.columns.tolist()}")
            return out

        def _from_code_name() -> pd.DataFrame:
            # Fallback when Eastmoney spot endpoint is blocked on the current network.
            info = _akshare_call(ak.stock_info_a_code_name)
            if "code" not in info.columns or "name" not in info.columns:
                raise ValueError(f"missing code/name columns: {info.columns.tolist()}")
            return info[["code", "name"]].copy()

        try:
            try:
                out = _from_spot()
                source_tag = "akshare:stock_zh_a_spot_em"
            except Exception:
                logger.warning("AkShare spot basic failed; falling back to stock_info_a_code_name")
                out = _from_code_name()
                source_tag = "akshare:stock_info_a_code_name"

            out["code"] = out["code"].map(_normalize_stock_code)
            for col in ["total_market_cap", "turnover", "latest_price"]:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce")
            out["source"] = source_tag
            out["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            if len(out) < min_count:
                return ProviderResult(False, None, self.name, f"too few all-A rows: {len(out)} < {min_count}")
            return ProviderResult(True, out.reset_index(drop=True), self.name, "")
        except Exception as exc:
            return ProviderResult(False, None, self.name, str(exc))

    @staticmethod
    def _fetch_daily(code: str, start_date: str, end_date: str) -> ProviderResult:
        import akshare as ak

        symbol = code.split(".")[0]
        start_arg = start_date.replace("-", "")
        end_arg = end_date.replace("-", "")

        def _try_hist() -> ProviderResult:
            """Primary Eastmoney k-line endpoint (adjusted + raw trade prices)."""
            adjusted = _akshare_call(
                ak.stock_zh_a_hist,
                symbol=symbol,
                period="daily",
                start_date=start_arg,
                end_date=end_arg,
                adjust="qfq",
            )
            raw = _akshare_call(
                ak.stock_zh_a_hist,
                symbol=symbol,
                period="daily",
                start_date=start_arg,
                end_date=end_arg,
                adjust="",
            )
            adjusted = adjusted.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "volume",
                    "成交额": "amount",
                    "涨跌幅": "pct_chg",
                    "换手率": "turnover",
                }
            )
            raw = raw.rename(
                columns={
                    "日期": "date",
                    "开盘": "trade_open",
                    "最高": "trade_high",
                    "最低": "trade_low",
                    "收盘": "trade_close",
                }
            )
            merged = _merge_adjusted_with_trade_price(adjusted, raw)
            merged = DataValidator.clean_stock_daily(merged, code)
            return ProviderResult(not merged.empty, merged, "akshare:stock_zh_a_hist", "" if not merged.empty else "empty data")

        def _try_daily() -> ProviderResult:
            """Sina daily endpoint with outstanding_share/turnover, no raw trade prices."""
            prefix = "sh" if code.endswith(".SH") else "sz"
            df = _akshare_call(
                ak.stock_zh_a_daily,
                symbol=f"{prefix}{symbol}",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            df = df.rename(
                columns={
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                    "amount": "amount",
                    "turnover": "turnover",
                }
            )
            # pct_chg is not provided; derive it from close.
            df["pct_chg"] = df["close"].pct_change() * 100
            df = DataValidator.clean_stock_daily(df, code)
            return ProviderResult(not df.empty, df, "akshare:stock_zh_a_daily", "" if not df.empty else "empty data")

        def _try_hist_tx() -> ProviderResult:
            """Tencent historical endpoint (last resort, includes trade prices)."""
            df = _akshare_call(
                ak.stock_zh_a_hist_tx,
                symbol=symbol,
                start_date=start_arg,
                end_date=end_arg,
            )
            df = df.rename(
                columns={
                    "date": "date",
                    "open": "open",
                    "close": "close",
                    "high": "high",
                    "low": "low",
                    "volume": "volume",
                    "turnover": "turnover",
                    "amount": "amount",
                }
            )
            # Reorder to close-standard and derive pct_chg.
            df = df[["date", "open", "high", "low", "close", "volume", "amount", "turnover"]]
            df["pct_chg"] = df["close"].pct_change() * 100
            # stock_zh_a_hist_tx volume is reported in shares matching Eastmoney style;
            # keep as-is so downstream validation/cache stays consistent.
            df = DataValidator.clean_stock_daily(df, code)
            return ProviderResult(not df.empty, df, "akshare:stock_zh_a_hist_tx", "" if not df.empty else "empty data")

        errors: list[str] = []
        for name, attempt in (
            ("stock_zh_a_hist", _try_hist),
            ("stock_zh_a_daily", _try_daily),
            ("stock_zh_a_hist_tx", _try_hist_tx),
        ):
            try:
                result = attempt()
                if result.ok:
                    if name != "stock_zh_a_hist":
                        logger.warning("AkShare fallback used for %s: primary=stock_zh_a_hist resolved by %s", code, name)
                    return result
                errors.append(f"{name}: {result.error}")
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        return ProviderResult(False, None, "akshare", " | ".join(errors))

    @staticmethod
    def _fetch_minute(code: str, period: str, start: str, end: str) -> ProviderResult:
        import akshare as ak

        symbol = code.split(".")[0]
        df = _akshare_call(
            ak.stock_zh_a_hist_min_em,
            symbol=symbol,
            start_date=_normalize_minute_datetime(start),
            end_date=_normalize_minute_datetime(end, end_of_day=True),
            period=period,
            adjust="",
        )
        rename = {
            "时间": "datetime",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df = _normalize_minute_frame(df, code, period)
        return ProviderResult(not df.empty, df, "akshare", "" if not df.empty else "empty minute data")


class TushareProvider(MarketDataProvider):
    """Tushare Pro API daily and basic data provider.

    Once a global auth/rate-limit failure is detected, the provider short-
    circuits all subsequent calls in the same process so that
    ``UnifiedDataProvider`` can fall back to AkShare instead of retrying
    Tushare for every stock.
    """

    name = "tushare"
    _global_failed = False
    _FATAL_ERRORS = (
        "token",
        "登录",
        "login",
        "频率",
        "frequency",
        "超限",
        "limit",
        "权限",
        "permission",
        "抱歉",
        "unauthorized",
        "auth",
    )

    def __init__(self, config: dict | None = None):
        self.config = config or load_data_config()
        self._pro = None
        self._token = self._resolve_token()

    @classmethod
    def _is_global_failed(cls) -> bool:
        return cls._global_failed

    @classmethod
    def _mark_global_failed(cls, error: str) -> None:
        if cls._global_failed:
            return
        lowered = str(error).lower()
        if any(kw in lowered for kw in cls._FATAL_ERRORS):
            cls._global_failed = True
            logger.warning("Tushare marked globally failed (%s); falling back to AkShare for this process.", error)

    def _resolve_token(self) -> str:
        tushare_cfg = self.config.get("tushare", {})
        token = tushare_cfg.get("token") or os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            self._mark_global_failed("Tushare token not configured")
            raise RuntimeError("Tushare token not configured")
        return str(token)

    def _pro_api(self):
        if self._global_failed:
            raise RuntimeError("Tushare globally failed; skipping")
        if self._pro is None:
            import tushare as ts

            self._pro = ts.pro_api(self._token)
        return self._pro

    def _wrap_result(self, result: ProviderResult) -> ProviderResult:
        if not result.ok:
            self._mark_global_failed(result.error)
        return result

    def healthcheck(self) -> ProviderResult:
        try:
            df = self._pro_api().query("stock_basic", exchange="", list_status="L", limit=1)
            if df is None or df.empty:
                return self._wrap_result(ProviderResult(False, None, self.name, "empty stock_basic response"))
            return ProviderResult(True, df.head(1), self.name, "")
        except Exception as exc:
            self._mark_global_failed(str(exc))
            return ProviderResult(False, None, self.name, str(exc))

    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> ProviderResult:
        try:
            return self._wrap_result(self._fetch_daily(code, start_date, end_date))
        except Exception as exc:
            self._mark_global_failed(str(exc))
            return ProviderResult(False, None, self.name, str(exc))

    def fetch_minute(
        self,
        code: str,
        period: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> ProviderResult:
        return ProviderResult(False, None, self.name, "tushare minute data not implemented")

    def fetch_basic(self, **kwargs: Any) -> ProviderResult:
        try:
            min_count = int(kwargs.get("min_count", 3000))
            df = self._pro_api().query(
                "stock_basic",
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,list_date",
            )
            if df is None or df.empty:
                return self._wrap_result(ProviderResult(False, None, self.name, "empty stock_basic response"))
            df = df.rename(columns={"ts_code": "code", "symbol": "short_code"})
            df["code"] = df["code"].map(_normalize_stock_code)
            df["source"] = "tushare:stock_basic"
            df["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            df = df.dropna(subset=["code", "name"])
            if len(df) < min_count:
                return self._wrap_result(ProviderResult(False, None, self.name, f"too few all-A rows: {len(df)} < {min_count}"))
            return ProviderResult(True, df.reset_index(drop=True), self.name, "")
        except Exception as exc:
            self._mark_global_failed(str(exc))
            return ProviderResult(False, None, self.name, str(exc))

    def _fetch_daily(self, code: str, start_date: str, end_date: str) -> ProviderResult:
        ts_code = code.split(".")[0]
        suffix = "SZ" if code.endswith(".SZ") else "SH"
        ts_code = f"{ts_code}.{suffix}"
        df = self._pro_api().query(
            "daily",
            ts_code=ts_code,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
        if df is None or df.empty:
            return self._wrap_result(ProviderResult(False, None, self.name, "empty data"))
        df = df.rename(
            columns={
                "trade_date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume",
                "amount": "amount",
                "pct_chg": "pct_chg",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        # tushare vol is in shares (hand * 100); align with akshare style.
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1000
        df["turnover"] = 0.0  # not provided by daily endpoint
        # tushare does not provide raw trade prices; use close as trade_close.
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["trade_open"] = df["open"]
        df["trade_high"] = df["high"]
        df["trade_low"] = df["low"]
        df["trade_close"] = df["close"]
        df["code"] = code
        df = DataValidator.clean_stock_daily(df, code)
        return ProviderResult(not df.empty, df, "tushare:daily", "" if not df.empty else "empty data")

    def fetch_daily_batch(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> ProviderResult:
        try:
            return self._wrap_result(self._fetch_daily_batch(codes, start_date, end_date))
        except Exception as exc:
            self._mark_global_failed(str(exc))
            return ProviderResult(False, None, self.name, str(exc))

    def _fetch_daily_batch(self, codes: list[str], start_date: str, end_date: str) -> ProviderResult:
        if self._global_failed:
            return ProviderResult(False, None, self.name, "Tushare globally failed; skipping")
        ts_codes = []
        for code in codes:
            raw = str(code).split(".")[0]
            suffix = "SZ" if str(code).endswith(".SZ") else "SH"
            ts_codes.append(f"{raw}.{suffix}")
        df = self._pro_api().query(
            "daily",
            ts_code=",".join(ts_codes),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
        if df is None or df.empty:
            return ProviderResult(False, None, self.name, "empty data")
        df = df.rename(
            columns={
                "trade_date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume",
                "amount": "amount",
                "pct_chg": "pct_chg",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1000
        df["turnover"] = 0.0
        for col in ["open", "high", "low", "close", "pct_chg"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["trade_open"] = df["open"]
        df["trade_high"] = df["high"]
        df["trade_low"] = df["low"]
        df["trade_close"] = df["close"]
        df["code"] = df["ts_code"].map(_normalize_stock_code)
        frames = []
        for code, group in df.groupby("code"):
            cleaned = DataValidator.clean_stock_daily(group, code)
            if not cleaned.empty:
                frames.append(cleaned)
        if not frames:
            return ProviderResult(False, None, self.name, "empty data after cleaning")
        return ProviderResult(True, pd.concat(frames, ignore_index=True), "tushare:daily", "")


class FuyaoProvider(MarketDataProvider):
    """Tonghuashun Fuyao API provider for A-share daily prices.

    The API key is read only from ``HITHINK_FINANCE_API_KEY`` in the process
    environment. It is deliberately not accepted from project configuration
    so generated configuration snapshots and Git history cannot contain it.
    """

    name = "fuyao"
    _base_url = "https://fuyao.aicubes.cn"
    _max_history_years = 10

    def __init__(self, config: dict | None = None):
        self.config = config or load_data_config()
        provider_cfg = self.config.get("fuyao", {})
        self.base_url = str(provider_cfg.get("base_url", self._base_url)).rstrip("/")
        self.timeout_seconds = float(provider_cfg.get("timeout_seconds", 20.0))
        self.adjust = str(provider_cfg.get("adjust", "none")).lower()

    def healthcheck(self) -> ProviderResult:
        result = self._request("/api/a-share/prices/snapshot", {"thscodes": "600519.SH"})
        if not result.ok:
            return result
        rows = result.data.get("item", []) if isinstance(result.data, dict) else []
        data = pd.DataFrame(rows[:1]) if rows else None
        return ProviderResult(bool(rows), data, self.name, "" if rows else "empty snapshot response")

    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> ProviderResult:
        try:
            start = pd.Timestamp(start_date).normalize()
            end = pd.Timestamp(end_date).normalize()
        except Exception as exc:
            return ProviderResult(False, None, self.name, f"invalid date range: {exc}")
        if end < start:
            return ProviderResult(False, None, self.name, "end date is before start date")
        if self.adjust not in {"none", "forward", "backward"}:
            return ProviderResult(False, None, self.name, f"unsupported adjustment: {self.adjust}")

        frames: list[pd.DataFrame] = []
        for window_start, window_end in self._history_windows(start, end):
            result = self._request(
                "/api/a-share/prices/historical",
                {
                    "thscode": _normalize_stock_code(code),
                    "interval": "1d",
                    "start": _shanghai_epoch_ms(window_start),
                    "end": _shanghai_epoch_ms(window_end),
                    "adjust": self.adjust,
                },
            )
            if not result.ok:
                return result
            items = result.data.get("item", []) if isinstance(result.data, dict) else []
            if items:
                frames.append(pd.DataFrame(items))

        if not frames:
            return ProviderResult(False, None, self.name, "empty historical response")
        raw = pd.concat(frames, ignore_index=True)
        required = {"date_ms", "open_price", "high_price", "low_price", "close_price", "volume", "turnover"}
        missing = sorted(required.difference(raw.columns))
        if missing:
            return ProviderResult(False, None, self.name, f"historical response missing fields: {missing}")
        out = raw.rename(
            columns={
                "date_ms": "date",
                "open_price": "open",
                "high_price": "high",
                "low_price": "low",
                "close_price": "close",
                "turnover": "amount",
            }
        )
        out["date"] = pd.to_datetime(out["date"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        for column in ("open", "high", "low", "close", "volume", "amount"):
            out[column] = pd.to_numeric(out[column], errors="coerce")
        out = out.sort_values("date").drop_duplicates("date", keep="last")
        out["pct_chg"] = out["close"].pct_change().fillna(0.0) * 100.0
        out["turnover"] = 0.0
        out["trade_open"] = out["open"]
        out["trade_high"] = out["high"]
        out["trade_low"] = out["low"]
        out["trade_close"] = out["close"]
        out["code"] = _normalize_stock_code(code)
        source = "fuyao_unadjusted_v1" if self.adjust == "none" else "fuyao"
        out["source"] = source
        out["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        out = DataValidator.clean_stock_daily(out, _normalize_stock_code(code))
        return ProviderResult(not out.empty, out, source, "" if not out.empty else "empty data after cleaning")

    def fetch_minute(
        self,
        code: str,
        period: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> ProviderResult:
        return ProviderResult(False, None, self.name, "Fuyao public API does not provide minute history")

    def fetch_basic(self, **kwargs: Any) -> ProviderResult:
        return ProviderResult(False, None, self.name, "Fuyao public API does not provide stock basic data")

    def _request(self, path: str, params: dict[str, Any]) -> ProviderResult:
        api_key = os.environ.get("HITHINK_FINANCE_API_KEY", "").strip()
        if not api_key:
            return ProviderResult(False, None, self.name, "HITHINK_FINANCE_API_KEY is not configured")
        request = Request(
            f"{self.base_url}{path}?{urlencode(params)}",
            headers={"X-api-key": api_key, "Accept": "application/json", "User-Agent": "KHQuant/1.0"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return ProviderResult(False, None, self.name, f"Fuyao request failed: {exc}")
        if not isinstance(payload, dict):
            return ProviderResult(False, None, self.name, "invalid Fuyao response")
        if payload.get("code") != 0:
            return ProviderResult(False, None, self.name, f"Fuyao API error {payload.get('code')}: {payload.get('message', 'unknown error')}")
        return ProviderResult(True, payload.get("data", {}), self.name, "")

    def _history_windows(self, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        cursor = start
        while cursor <= end:
            window_end = min(cursor + pd.DateOffset(years=self._max_history_years) - pd.Timedelta(days=1), end)
            windows.append((cursor, window_end))
            cursor = window_end + pd.Timedelta(days=1)
        return windows


class BaostockProvider(MarketDataProvider):
    """Baostock-based daily and minute data provider.

    Once a global login failure is detected, the provider short-circuits all
    subsequent calls in the same process to avoid spamming ``bs.login()``.
    """

    name = "baostock"
    _global_failed = False

    @classmethod
    def _is_global_failed(cls) -> bool:
        return cls._global_failed

    @classmethod
    def _mark_global_failed(cls, error: str) -> None:
        if cls._global_failed:
            return
        cls._global_failed = True
        logger.warning("Baostock marked globally failed (%s); skipping for this process.", error)

    def healthcheck(self) -> ProviderResult:
        try:
            if self._global_failed:
                return ProviderResult(False, None, self.name, "baostock globally failed; skipping")
            import baostock as bs

            lg = bs.login()
            try:
                if getattr(lg, "error_code", "0") != "0":
                    error = getattr(lg, "error_msg", "login failed")
                    self._mark_global_failed(error)
                    return ProviderResult(False, None, self.name, error)
                return ProviderResult(True, pd.DataFrame(), self.name, "")
            finally:
                bs.logout()
        except Exception as exc:
            self._mark_global_failed(str(exc))
            return ProviderResult(False, None, self.name, str(exc))

    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> ProviderResult:
        if self._global_failed:
            return ProviderResult(False, None, self.name, "baostock globally failed; skipping")
        try:
            return self._fetch_daily(code, start_date, end_date)
        except Exception as exc:
            self._mark_global_failed(str(exc))
            return ProviderResult(False, None, self.name, str(exc))

    def fetch_minute(
        self,
        code: str,
        period: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> ProviderResult:
        if self._global_failed:
            return ProviderResult(False, None, self.name, "baostock globally failed; skipping")
        try:
            return self._fetch_minute(code, period, start, end)
        except Exception as exc:
            self._mark_global_failed(str(exc))
            return ProviderResult(False, None, self.name, str(exc))

    def fetch_basic(self, **kwargs: Any) -> ProviderResult:
        return ProviderResult(False, None, self.name, "baostock does not provide basic stock table")

    @staticmethod
    def _fetch_daily(code: str, start_date: str, end_date: str) -> ProviderResult:
        import baostock as bs

        lg = bs.login()
        if getattr(lg, "error_code", "0") != "0":
            raise RuntimeError(getattr(lg, "error_msg", "baostock login failed"))
        try:
            market = "sh" if code.endswith(".SH") else "sz"
            bs_code = f"{market}.{code.split('.')[0]}"
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if getattr(rs, "error_code", "0") != "0":
                raise RuntimeError(getattr(rs, "error_msg", "baostock query failed"))
            adjusted = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])

            raw_rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )
            raw_rows = []
            while raw_rs.next():
                raw_rows.append(raw_rs.get_row_data())
            if getattr(raw_rs, "error_code", "0") != "0":
                raise RuntimeError(getattr(raw_rs, "error_msg", "baostock raw query failed"))
            raw = pd.DataFrame(raw_rows, columns=["date", "trade_open", "trade_high", "trade_low", "trade_close"])
            merged = _merge_adjusted_with_trade_price(adjusted, raw)
            merged = DataValidator.clean_stock_daily(merged, code)
            return ProviderResult(not merged.empty, merged, "baostock", "" if not merged.empty else "empty data")
        finally:
            bs.logout()

    @staticmethod
    def _fetch_minute(code: str, period: str, start: str, end: str) -> ProviderResult:
        if period == "1":
            return ProviderResult(False, None, "baostock", "baostock does not support 1-minute bars")
        import baostock as bs

        lg = bs.login()
        if getattr(lg, "error_code", "0") != "0":
            raise RuntimeError(getattr(lg, "error_msg", "baostock login failed"))
        try:
            market = "sh" if code.endswith(".SH") else "sz"
            bs_code = f"{market}.{code.split('.')[0]}"
            start_date = str(pd.to_datetime(start).date())
            end_date = str(pd.to_datetime(end).date())
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,time,code,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency=period,
                adjustflag="3",
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if getattr(rs, "error_code", "0") != "0":
                raise RuntimeError(getattr(rs, "error_msg", "baostock minute query failed"))
            df = pd.DataFrame(rows, columns=["date", "time", "bs_code", "open", "high", "low", "close", "volume", "amount"])
            if df.empty:
                return ProviderResult(False, None, "baostock", "empty minute data")
            df = _normalize_minute_frame(df, code, period)
            return ProviderResult(not df.empty, df, "baostock", "" if not df.empty else "empty minute data")
        finally:
            bs.logout()


class TencentRealtimeProvider(MarketDataProvider):
    """Tencent realtime quote provider for the current/latest trading day.

    This provider only returns data for the most recent session; historical ranges
    beyond today are not supported.
    """

    name = "tencent_realtime"

    def __init__(self, config: dict | None = None):
        self.config = config or load_data_config()

    def healthcheck(self) -> ProviderResult:
        result = self._fetch_batch(["000001.SH"], target_date=pd.Timestamp.now().strftime("%Y-%m-%d"))
        if result.ok:
            return ProviderResult(True, result.data.head(1) if result.data is not None else pd.DataFrame(), self.name, "")
        return ProviderResult(False, None, self.name, result.error)

    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> ProviderResult:
        target_date = kwargs.get("target_date") or end_date
        result = self._fetch_batch([code], target_date=target_date)
        if not result.ok or result.data is None:
            return result
        df = result.data.copy()
        if not df.empty:
            df = DataValidator.clean_stock_daily(df, code)
        return ProviderResult(not df.empty, df, self.name, "" if not df.empty else "no realtime row")

    def fetch_minute(
        self,
        code: str,
        period: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> ProviderResult:
        return ProviderResult(False, None, self.name, "tencent realtime does not provide minute history")

    def fetch_basic(self, **kwargs: Any) -> ProviderResult:
        return ProviderResult(False, None, self.name, "tencent realtime does not provide stock basic table")

    def _quote_symbol(self, code: str) -> str:
        raw = str(code).split(".")[0]
        return ("sh" if str(code).upper().endswith(".SH") else "sz") + raw

    def _fetch_batch(self, codes: list[str], target_date: str | None = None) -> ProviderResult:
        target_date = target_date or pd.Timestamp.now().strftime("%Y-%m-%d")
        target_stamp = str(target_date).replace("-", "")
        if not codes:
            return ProviderResult(False, None, self.name, "empty code list")
        url = "https://qt.gtimg.cn/q=" + ",".join(self._quote_symbol(code) for code in codes)
        try:
            result = subprocess.run(
                [_curl_executable(), "-s", "--noproxy", "*", "--max-time", "20", url],
                capture_output=True,
                timeout=25,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if result.returncode != 0:
                return ProviderResult(False, None, self.name, f"curl exit={result.returncode}: {result.stderr[-300:]}")
            text = result.stdout.encode("utf-8", errors="ignore").decode("gbk", errors="replace")
        except Exception as exc:
            return ProviderResult(False, None, self.name, str(exc))

        now_text = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            if "=" not in line or "~" not in line:
                continue
            parts = line.split("~")
            if len(parts) < 35:
                continue
            raw = parts[2].strip()
            if not raw.isdigit():
                continue
            code = raw + (".SH" if raw.startswith("6") else ".SZ")
            if code not in codes:
                continue
            stamp = parts[30].strip() if len(parts) > 30 else ""
            if not stamp.startswith(target_stamp):
                continue
            try:
                close = float(parts[3]) if parts[3] else 0.0
                open_price = float(parts[5]) if parts[5] else 0.0
                high = float(parts[33]) if parts[33] else 0.0
                low = float(parts[34]) if parts[34] else 0.0
                volume = float(parts[6].replace(",", "")) if parts[6] else 0.0
                pct_chg = float(parts[32]) if len(parts) > 32 and parts[32] else 0.0
            except Exception:
                continue
            if min(open_price, high, low, close) <= 0 or high < low or volume <= 0:
                continue
            rows.append(
                {
                    "date": pd.to_datetime(target_date),
                    "code": code,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "trade_open": open_price,
                    "trade_high": high,
                    "trade_low": low,
                    "trade_close": close,
                    "volume": volume * 100,
                    "amount": 0.0,
                    "pct_chg": pct_chg,
                    "turnover": 0.0,
                    "source": self.name,
                    "updated_at": now_text,
                }
            )
        if not rows:
            return ProviderResult(False, None, self.name, f"no valid realtime quotes for {codes}")
        return ProviderResult(True, pd.DataFrame(rows), self.name, "")


class LocalCSVProvider(MarketDataProvider):
    """Provider that reads from local CSV/Parquet files.

    This is the offline fallback used when network sources are unavailable.
    """

    name = "local_csv"

    def __init__(self, config: dict | None = None):
        self.config = config or load_data_config()

    @property
    def data_root(self) -> Path:
        return project_path(self.config.get("data_root", "my_strategy/data"))

    def healthcheck(self) -> ProviderResult:
        candidates = [
            self.data_root / "raw" / "stock_daily",
            self.data_root / "raw" / "custom_pool",
        ]
        for path in candidates:
            if path.exists() and any(path.iterdir()):
                return ProviderResult(True, pd.DataFrame(), self.name, "")
        return ProviderResult(False, None, self.name, "no local csv data found")

    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> ProviderResult:
        try:
            code = _normalize_stock_code(code)
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            candidates = [
                self.data_root / "raw" / "stock_daily" / f"{code}.csv",
                self.data_root / "raw" / "stock_daily" / f"{code}.parquet",
                self.data_root / "raw" / "custom_pool" / f"{code}.csv",
                self.data_root / "raw" / "custom_pool" / f"{code}.parquet",
            ]
            for path in candidates:
                if not path.exists():
                    continue
                if path.suffix.lower() == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                if "date" not in df.columns:
                    continue
                df["date"] = pd.to_datetime(df["date"])
                df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
                df = DataValidator.clean_stock_daily(df, code)
                if not df.empty:
                    return ProviderResult(True, df, self.name, "")
            return ProviderResult(False, None, self.name, f"no local csv data for {code}")
        except Exception as exc:
            return ProviderResult(False, None, self.name, str(exc))

    def fetch_minute(
        self,
        code: str,
        period: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> ProviderResult:
        return ProviderResult(False, None, self.name, "local csv minute data not implemented")

    def fetch_basic(self, **kwargs: Any) -> ProviderResult:
        path = self.data_root / "raw" / "stock_basic" / "all_a_stock_list.parquet"
        try:
            if path.exists():
                return ProviderResult(True, pd.read_parquet(path), self.name, "")
            return ProviderResult(False, None, self.name, f"local basic cache not found: {path}")
        except Exception as exc:
            return ProviderResult(False, None, self.name, str(exc))


@dataclass
class UnifiedProviderConfig:
    """Runtime tuning for the unified provider."""

    daily_sources: list[str] = field(default_factory=lambda: ["akshare", "baostock", "local_csv"])
    minute_sources: list[str] = field(default_factory=lambda: ["akshare", "baostock"])
    max_retry: int = 3
    base_sleep: float = 0.8
    rate_calls: float = 1.0
    rate_per_seconds: float = 1.0
    timeout_seconds: float = 60.0
    provider_rate_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    run_id: str = ""

    @classmethod
    def from_project_config(cls, config: dict | None = None) -> UnifiedProviderConfig:
        cfg = config or load_data_config()
        ds_cfg = cfg.get("data_sources", {})
        ms_cfg = cfg.get("minute_data_sources", {})
        update_cfg = cfg.get("update", {})
        daily = [ds_cfg.get("primary", "akshare"), *ds_cfg.get("fallback", ["baostock", "local_csv"])]
        minute = [ms_cfg.get("primary", "akshare"), *ms_cfg.get("fallback", ["baostock"])]
        default_limit = cfg.get("rate_limit", {})
        provider_limits: dict[str, tuple[float, float]] = {}
        for name, limit in (cfg.get("provider_rate_limits", {}) or {}).items():
            if isinstance(limit, dict):
                provider_limits[str(name).lower()] = (
                    float(limit.get("calls", default_limit.get("calls", 1.0)) or 1.0),
                    float(limit.get("per_seconds", default_limit.get("per_seconds", 1.0)) or 1.0),
                )
        return cls(
            daily_sources=_dedupe_sources(daily),
            minute_sources=_dedupe_sources(minute),
            max_retry=int(update_cfg.get("max_retry", 3)),
            base_sleep=float(update_cfg.get("sleep_seconds", 0.8)),
            rate_calls=float(default_limit.get("calls", 1.0) or 1.0),
            rate_per_seconds=float(default_limit.get("per_seconds", 1.0) or 1.0),
            timeout_seconds=float(default_limit.get("timeout_seconds", 60.0) or 60.0),
            provider_rate_limits=provider_limits,
        )


class UnifiedDataProvider:
    """Aggregate provider with automatic failover, retry, rate limiting and audit.

    Usage:
        provider = UnifiedDataProvider()
        result = provider.fetch_daily("000001.SZ", "2024-01-01", "2024-12-31")
        if result.ok:
            df = result.data
    """

    def __init__(
        self,
        config: dict | None = None,
        providers: Iterable[MarketDataProvider] | None = None,
        run_id: str = "",
    ):
        self.config = config or load_data_config()
        self.runtime = UnifiedProviderConfig.from_project_config(self.config)
        self.runtime.run_id = run_id or self.runtime.run_id
        self._providers: dict[str, MarketDataProvider] = {}
        for provider in providers or _default_providers(self.config):
            self._providers[provider.name] = provider
        self._limiters: dict[str, RateLimiter] = {}
        env_path = os.environ.get("KHQUANT_SOURCE_STATUS_PATH", "").strip()
        if env_path:
            self._audit_path: Path | None = Path(env_path)
        else:
            self._audit_path = None

    def healthcheck(self) -> ProviderResult:
        results: list[ProviderResult] = []
        for name in self.runtime.daily_sources:
            provider = self._providers.get(name)
            if provider is None:
                continue
            self._limiter_for(provider.name).acquire()
            result = provider.healthcheck()
            self._audit(provider.name, result, operation="healthcheck")
            results.append(result)
        ok_results = [r for r in results if r.ok]
        if ok_results:
            return ProviderResult(True, ok_results[0].data, ok_results[0].source, "")
        errors = " | ".join(f"{r.source}: {r.error}" for r in results if not r.ok)
        return ProviderResult(False, None, "", errors or "all sources failed healthcheck")

    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        sources: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> ProviderResult:
        source_list = list(sources) if sources is not None else self.runtime.daily_sources
        return self._fetch_with_fallback(
            "fetch_daily",
            source_list,
            code=_normalize_stock_code(code),
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )

    def fetch_daily_batch(
        self,
        codes: Iterable[str],
        start_date: str,
        end_date: str,
        sources: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> ProviderResult:
        """Fetch daily bars for many codes in as few upstream calls as possible.

        Tushare is invoked with comma-separated ``ts_code`` (chunked to respect
        response size).  When Tushare fails or is not the primary, the unified
        provider falls back to per-stock fetches through the next configured
        source (usually AkShare), still honoring rate limits and retries.
        """
        source_list = list(sources) if sources is not None else self.runtime.daily_sources
        codes = [_normalize_stock_code(c) for c in codes]
        if not codes:
            return ProviderResult(False, None, "", "empty code list")
        errors: list[str] = []
        primary = source_list[0] if source_list else ""
        for index, name in enumerate(source_list):
            provider = self._providers.get(name)
            if provider is None:
                if name == "tushare" and not (self.config.get("tushare", {}).get("token") or os.environ.get("TUSHARE_TOKEN")):
                    errors.append("tushare: Tushare token not configured")
                else:
                    errors.append(f"{name}: unsupported source")
                continue
            if index == 0:
                primary = name
            if name == "tushare" and hasattr(provider, "fetch_daily_batch"):
                # Chunk adaptively by the date range actually requested.  A short
                # incremental window (a few rows per code) can ride in large bulk
                # batches (``bulk_batch_size``) while staying far under tushare's
                # 6000-row response cap; a long/full-history window must use small
                # batches (``batch_size``) so rows are never truncated.  Without
                # this, ``update_batch``'s 200-code bulk batches were re-split
                # here into 100 two-code tushare calls per batch, multiplying the
                # call count ~100x and starving live progress.
                small = int(self.config.get("tushare", {}).get("batch_size", 2))
                bulk = int(self.config.get("tushare", {}).get("bulk_batch_size", 200))
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                est_rows = max(1, int((end_dt - start_dt).days * 0.8))
                chunk_size = bulk if est_rows * bulk <= 4000 else small
                chunks = [codes[i : i + chunk_size] for i in range(0, len(codes), chunk_size)]
                chunk_frames: list[pd.DataFrame] = []
                chunk_errors: list[str] = []
                for chunk in chunks:
                    result = self._call_with_retry(
                        provider,
                        "fetch_daily_batch",
                        codes=chunk,
                        start_date=start_date,
                        end_date=end_date,
                        **kwargs,
                    )
                    if result.ok and result.data is not None:
                        chunk_frames.append(result.data)
                    else:
                        chunk_errors.append(result.error or "unknown")
                if chunk_frames:
                    combined = pd.concat(chunk_frames, ignore_index=True)
                    self._audit(
                        name,
                        ProviderResult(True, combined, name, ""),
                        operation="fetch_daily_batch",
                        primary=primary,
                        status="success",
                        code_count=len(codes),
                    )
                    if index > 0:
                        logger.warning("Fallback used: primary=%s resolved by %s", primary, name)
                    return ProviderResult(True, combined, name, "")
                errors.append(f"{name}: {' | '.join(chunk_errors)}")
                continue
            # Fallback path: per-stock fetch for sources without native batch support.
            result = self._fetch_daily_batch_per_stock(
                provider, codes, start_date, end_date, primary=primary, **kwargs
            )
            status = "success" if result.ok else "failed"
            self._audit(name, result, operation="fetch_daily_batch", primary=primary, status=status, code_count=len(codes))
            if result.ok:
                if index > 0:
                    logger.warning("Fallback used: primary=%s resolved by %s", primary, name)
                return ProviderResult(True, result.data, name, "")
            errors.append(f"{name}: {result.error}")
        return ProviderResult(False, None, "", " | ".join(errors))

    def _fetch_daily_batch_per_stock(
        self,
        provider: MarketDataProvider,
        codes: list[str],
        start_date: str,
        end_date: str,
        primary: str,
        **kwargs: Any,
    ) -> ProviderResult:
        frames: list[pd.DataFrame] = []
        failed: list[str] = []
        for code in codes:
            result = self._call_with_retry(
                provider,
                "fetch_daily",
                code=code,
                start_date=start_date,
                end_date=end_date,
                **kwargs,
            )
            if result.ok and result.data is not None and not result.data.empty:
                frames.append(result.data)
            else:
                failed.append(f"{code}: {result.error or 'empty'}")
        if frames:
            return ProviderResult(True, pd.concat(frames, ignore_index=True), provider.name, "")
        tail = "; ".join(failed[:5])
        if len(failed) > 5:
            tail += "..."
        return ProviderResult(False, None, provider.name, tail)

    def fetch_minute(
        self,
        code: str,
        period: str,
        start: str,
        end: str,
        sources: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> ProviderResult:
        source_list = list(sources) if sources is not None else self.runtime.minute_sources
        return self._fetch_with_fallback(
            "fetch_minute",
            source_list,
            code=_normalize_stock_code(code),
            period=str(period).lower().replace("m", ""),
            start=start,
            end=end,
            **kwargs,
        )

    def fetch_basic(
        self,
        sources: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> ProviderResult:
        source_list = list(sources) if sources is not None else self.runtime.daily_sources
        return self._fetch_with_fallback("fetch_basic", source_list, **kwargs)

    def _fetch_with_fallback(
        self,
        method: str,
        source_list: list[str],
        **kwargs: Any,
    ) -> ProviderResult:
        errors: list[str] = []
        primary = source_list[0] if source_list else ""
        for index, name in enumerate(source_list):
            provider = self._providers.get(name)
            if provider is None:
                if name == "tushare" and not (self.config.get("tushare", {}).get("token") or os.environ.get("TUSHARE_TOKEN")):
                    errors.append("tushare: Tushare token not configured")
                else:
                    errors.append(f"{name}: unsupported source")
                continue
            if index == 0:
                primary = name
            result = self._call_with_retry(provider, method, **kwargs)
            status = "success" if result.ok else "failed"
            self._audit(name, result, operation=method, primary=primary, status=status, **kwargs)
            if result.ok:
                if index > 0:
                    logger.warning("Fallback used: primary=%s resolved by %s", primary, name)
                return ProviderResult(True, result.data, name, "")
            errors.append(f"{name}: {result.error}")
        return ProviderResult(False, None, "", " | ".join(errors))

    def _limiter_for(self, provider_name: str) -> RateLimiter:
        name = str(provider_name).lower()
        if name not in self._limiters:
            calls, per_seconds = self.runtime.provider_rate_limits.get(
                name, (self.runtime.rate_calls, self.runtime.rate_per_seconds)
            )
            self._limiters[name] = RateLimiter(calls=calls, per_seconds=per_seconds)
        return self._limiters[name]

    def _call_with_retry(
        self,
        provider: MarketDataProvider,
        method: str,
        **kwargs: Any,
    ) -> ProviderResult:
        last_result = ProviderResult(False, None, provider.name, "")
        for attempt in range(max(1, self.runtime.max_retry)):
            self._limiter_for(provider.name).acquire()
            try:
                last_result = getattr(provider, method)(**kwargs)
                if last_result.ok:
                    return last_result
            except Exception as exc:
                last_result = ProviderResult(False, None, provider.name, str(exc))
            if not last_result.ok and attempt < self.runtime.max_retry - 1:
                sleep = self.runtime.base_sleep * (2 ** attempt)
                logger.debug("Provider %s %s failed (attempt %d): %s; retry in %.2fs", provider.name, method, attempt + 1, last_result.error, sleep)
                time.sleep(sleep)
        return last_result

    def _audit(
        self,
        source: str,
        result: ProviderResult,
        operation: str,
        primary: str = "",
        status: str = "",
        **kwargs: Any,
    ) -> None:
        try:
            fallback_used = bool(primary and source != primary)
            data = result.data
            row_count = 0 if data is None else len(data)
            date_range = {}
            if data is not None and not data.empty and "date" in data.columns:
                dates = pd.to_datetime(data["date"], errors="coerce")
                if not dates.empty:
                    date_range = {"min": dates.min().strftime("%Y-%m-%d"), "max": dates.max().strftime("%Y-%m-%d")}
            status_obj = DataSourceStatus(
                source=f"{source}:{operation}",
                status=status or ("success" if result.ok else "failed"),
                fallback_used=fallback_used,
                fallback_source=primary if fallback_used else "",
                error=result.error,
                row_count=row_count,
                date_range=date_range,
                run_id=self.runtime.run_id,
            )
            append_source_status(status_obj, path=self._audit_path)
        except Exception as exc:
            logger.warning("Failed to audit source status: %s", exc)


def _default_providers(config: dict | None) -> list[MarketDataProvider]:
    cfg = config or load_data_config()
    providers: list[MarketDataProvider] = [AkShareProvider(), FuyaoProvider(cfg)]
    if cfg.get("tushare", {}).get("token") or os.environ.get("TUSHARE_TOKEN"):
        providers.append(TushareProvider(cfg))
    providers.extend([
        BaostockProvider(),
        TencentRealtimeProvider(cfg),
        LocalCSVProvider(cfg),
    ])
    return providers


def _dedupe_sources(sources: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in sources:
        name = str(name).lower()
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _shanghai_epoch_ms(value: pd.Timestamp) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Shanghai")
    else:
        timestamp = timestamp.tz_convert("Asia/Shanghai")
    return int(timestamp.value // 1_000_000)


def _normalize_stock_code(code: str) -> str:
    code = str(code).strip().split(".")[0].zfill(6)
    suffix = "SH" if code.startswith("6") else "SZ"
    return f"{code}.{suffix}"


def _merge_adjusted_with_trade_price(adjusted: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    if adjusted is None or adjusted.empty:
        return adjusted
    adjusted = adjusted.copy()
    adjusted["date"] = pd.to_datetime(adjusted["date"])
    if raw is None or raw.empty:
        return adjusted
    raw = raw[["date", "trade_open", "trade_high", "trade_low", "trade_close"]].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    return adjusted.merge(raw, on="date", how="left")


def _normalize_minute_datetime(value: str, end_of_day: bool = False) -> str:
    text = str(value)
    if len(text) <= 10:
        text = f"{text} {'15:00:00' if end_of_day else '09:30:00'}"
    return pd.to_datetime(text).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_minute_frame(df: pd.DataFrame, code: str, period: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "datetime" not in out.columns and {"date", "time"}.issubset(out.columns):
        raw_time = out["time"].astype(str)
        out["datetime"] = raw_time.where(raw_time.str.len() >= 14, out["date"].astype(str) + raw_time.str.slice(0, 6))
    if "datetime" in out.columns:
        out["datetime"] = _parse_minute_datetimes(out["datetime"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["code"] = code
    out["period"] = period
    if "datetime" in out.columns:
        out["date"] = out["datetime"].dt.strftime("%Y-%m-%d")
        out["time"] = out["datetime"].dt.strftime("%H:%M:%S")
    if "source" not in out.columns:
        out["source"] = ""
    if "updated_at" not in out.columns:
        out["updated_at"] = ""
    out = out.dropna(subset=["datetime" if "datetime" in out.columns else "date", "open", "high", "low", "close"])
    if "close" in out.columns:
        out = out[out["close"] > 0]
    if "high" in out.columns and "low" in out.columns:
        out = out[out["high"] >= out["low"]]
    out = out.drop_duplicates(subset=["datetime" if "datetime" in out.columns else "date"], keep="last")
    if "datetime" in out.columns:
        out = out.sort_values("datetime").reset_index(drop=True)
    return out


def _parse_minute_datetimes(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    compact = text.str.replace(r"\D", "", regex=True)
    parsed = pd.to_datetime(text, errors="coerce")
    need_compact = parsed.isna() & compact.str.len().ge(14)
    if need_compact.any():
        parsed.loc[need_compact] = pd.to_datetime(compact.loc[need_compact].str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    return parsed


def _curl_executable() -> str:
    for candidate in ("curl.exe", "curl"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("curl executable not found")


def get_unified_provider(config: dict | None = None, run_id: str = "") -> UnifiedDataProvider:
    return UnifiedDataProvider(config=config, run_id=run_id)
