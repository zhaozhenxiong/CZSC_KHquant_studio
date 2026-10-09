"""External market cache and point-in-time sentiment feature helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from .config import load_data_config, project_path


DEFAULT_INSTRUMENTS: list[dict[str, Any]] = [
    {"id": "csi300", "symbol": "000300.SH", "secid": "1.000300", "market": "CN", "group": "index", "weight": 0.75, "timezone": "Asia/Shanghai", "close_time": "15:00"},
    {"id": "sp500", "symbol": "标普500", "market": "US", "group": "index", "weight": 0.06, "timezone": "America/New_York", "close_time": "16:00"},
    {"id": "nasdaq", "symbol": "纳斯达克", "market": "US", "group": "sector", "weight": 0.06, "timezone": "America/New_York", "close_time": "16:00"},
    {"id": "dow", "symbol": "道琼斯", "market": "US", "group": "index", "weight": 0.02, "timezone": "America/New_York", "close_time": "16:00"},
    {"id": "nikkei", "symbol": "日经225", "market": "JP", "group": "index", "weight": 0.04, "timezone": "Asia/Tokyo", "close_time": "15:00"},
    {"id": "kospi", "symbol": "韩国KOSPI", "market": "KR", "group": "index", "weight": 0.04, "timezone": "Asia/Seoul", "close_time": "15:30"},
    {"id": "crb", "symbol": "路透CRB商品指数", "market": "US", "group": "sector", "weight": 0.03, "timezone": "America/New_York", "close_time": "16:00"},
]

DEFAULT_TRAINING_BENCHMARK: dict[str, Any] = {
    "id": "csi300",
    "symbol": "000300.SH",
    "secid": "1.000300",
    "market": "CN",
    "group": "benchmark",
    "timezone": "Asia/Shanghai",
    "close_time": "15:00",
}

FRED_SERIES_BY_INSTRUMENT: dict[str, str] = {
    "sp500": "SP500",
    "nasdaq": "NASDAQCOM",
    "dow": "DJIA",
    "nikkei": "NIKKEI225",
}

REQUIRED_COLUMNS = ["date", "instrument", "open", "high", "low", "close", "available_at"]


@dataclass(frozen=True)
class ExternalRefreshResult:
    instrument: str
    symbol: str
    status: str
    rows: int = 0
    error: str = ""


def configured_instruments(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = (config or {}).get("external_sentiment", {})
    raw = cfg.get("instruments") or DEFAULT_INSTRUMENTS
    instruments: list[dict[str, Any]] = []
    for item in raw:
        merged = {
            "market": "",
            "group": "index",
            "weight": 1.0,
            "timezone": "Asia/Shanghai",
            "close_time": "15:00",
            **dict(item),
        }
        if not merged.get("id") or not merged.get("symbol"):
            raise ValueError("external sentiment instruments require id and symbol")
        instruments.append(merged)
    return instruments


class ExternalMarketCache:
    """Own parquet files for point-in-time external and benchmark market history."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_data_config()
        self.data_root = project_path(self.config.get("data_root", "my_strategy/data"))
        self.cache_dir = self.data_root / "raw" / "external_market"
        self.metadata_dir = self.data_root / "metadata"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, instrument: str) -> Path:
        return self.cache_dir / f"{_safe_name(instrument)}.parquet"

    def read(self, instrument: str) -> pd.DataFrame:
        path = self.path_for(instrument)
        if not path.exists():
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        frame = pd.read_parquet(path)
        return _clean_history(frame, instrument)

    def read_many(self, instruments: Iterable[dict[str, Any]]) -> pd.DataFrame:
        frames = []
        for item in instruments:
            frame = self.read(str(item["id"]))
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        return pd.concat(frames, ignore_index=True).sort_values(["instrument", "date"]).reset_index(drop=True)

    def save(self, instrument: dict[str, Any], frame: pd.DataFrame, source: str) -> pd.DataFrame:
        clean = _clean_history(frame, str(instrument["id"]), instrument)
        if clean.empty:
            raise ValueError(f"refuse to save empty external market cache: {instrument['id']}")
        existing = self.read(str(instrument["id"]))
        merged = pd.concat([existing, clean], ignore_index=True)
        merged = _clean_history(merged, str(instrument["id"]), instrument)
        merged["source"] = source
        merged["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _atomic_write_parquet(merged, self.path_for(str(instrument["id"])))
        return merged

    def record_status(self, result: ExternalRefreshResult) -> None:
        path = self.metadata_dir / "external_market_status.csv"
        columns = ["time", "instrument", "symbol", "status", "rows", "error"]
        row = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "instrument": result.instrument,
            "symbol": result.symbol,
            "status": result.status,
            "rows": result.rows,
            "error": result.error,
        }
        old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
        old = old[old["instrument"].astype(str) != result.instrument]
        out = pd.concat([old, pd.DataFrame([row])], ignore_index=True)[columns]
        tmp = path.with_name(path.name + ".tmp")
        out.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)

    def refresh(self, instruments: Iterable[dict[str, Any]], source: str = "akshare_global_index") -> list[ExternalRefreshResult]:
        results: list[ExternalRefreshResult] = []
        for instrument in instruments:
            item = dict(instrument)
            try:
                if str(item.get("market", "")).upper() == "CN" and item.get("secid"):
                    history = download_eastmoney_history(str(item["secid"]))
                    resolved_source = "eastmoney_cn_index"
                else:
                    history, resolved_source = download_global_index_history(
                        str(item["symbol"]),
                        instrument_id=str(item["id"]),
                    )
                saved = self.save(item, history, resolved_source or source)
                result = ExternalRefreshResult(str(item["id"]), str(item["symbol"]), "success", len(saved))
            except Exception as exc:
                result = ExternalRefreshResult(str(item["id"]), str(item["symbol"]), "failed", 0, str(exc))
            self.record_status(result)
            results.append(result)
        return results

    def refresh_training_benchmark(self, benchmark: dict[str, Any] | None = None) -> ExternalRefreshResult:
        item = {**DEFAULT_TRAINING_BENCHMARK, **(benchmark or {})}
        try:
            try:
                history = download_eastmoney_history(str(item["secid"]))
                source = "eastmoney_cn_index"
            except Exception:
                history = download_tencent_index_history(str(item["symbol"]))
                source = "tencent_cn_index"
            saved = self.save(item, history, source)
            result = ExternalRefreshResult(str(item["id"]), str(item["symbol"]), "success", len(saved))
        except Exception as exc:
            result = ExternalRefreshResult(str(item["id"]), str(item["symbol"]), "failed", 0, str(exc))
        self.record_status(result)
        return result


def download_global_index_history(symbol: str, *, instrument_id: str = "") -> tuple[pd.DataFrame, str]:
    """Download one configured global index without writing local state.

    Eastmoney remains the primary source.  The S&P 500 has a FRED fallback so
    an intermittent Eastmoney TLS/proxy failure does not unnecessarily degrade
    the external-sentiment coverage.
    """
    try:
        from akshare.index.index_global_em import index_global_em_symbol_map

        item = index_global_em_symbol_map[symbol]
        return download_eastmoney_history(f"{item['market']}.{item['code']}"), "eastmoney_global_index"
    except Exception as direct_error:
        try:
            import akshare as ak

            return ak.index_global_hist_em(symbol=symbol), "akshare_global_index"
        except Exception as fallback_error:
            fred_series = FRED_SERIES_BY_INSTRUMENT.get(instrument_id)
            if fred_series:
                try:
                    return download_fred_history(fred_series), f"fred_{fred_series.lower()}"
                except Exception as fred_error:
                    raise RuntimeError(
                        f"direct={direct_error}; akshare={fallback_error}; fred={fred_error}"
                    ) from fred_error
            raise RuntimeError(f"direct={direct_error}; akshare={fallback_error}") from fallback_error


def download_fred_history(series_id: str) -> pd.DataFrame:
    """Fetch a daily FRED series as OHLC-compatible close-only history."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    proxy = _external_proxy()
    response = requests.get(
        url,
        timeout=30,
        proxies={"http": proxy, "https": proxy} if proxy else None,
    )
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text))
    if "observation_date" not in raw.columns or series_id not in raw.columns:
        raise RuntimeError(f"unexpected FRED columns for {series_id}: {list(raw.columns)}")
    close = pd.to_numeric(raw[series_id], errors="coerce")
    frame = pd.DataFrame({"date": raw["observation_date"], "close": close})
    frame["open"] = frame["close"]
    frame["high"] = frame["close"]
    frame["low"] = frame["close"]
    return frame[["date", "open", "high", "low", "close"]].dropna(subset=["close"])


def download_eastmoney_history(secid: str) -> pd.DataFrame:
    """Fetch an Eastmoney daily history series using configured proxy settings."""
    curl = _curl_executable()
    params = (
        f"secid={secid}&klt=101&fqt=1&lmt=50000&end=20500000&iscca=1&"
        "fields1=f1,f2,f3,f4,f5,f6,f7,f8&"
        "fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64&"
        "ut=f057cbcbce2a86e2866ab8877db1d059&forcect=1"
    )
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?{params}"
    result = subprocess.run(
        _curl_command(curl, url),
        capture_output=True,
        text=True,
        timeout=35,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl exit={result.returncode}: {result.stderr[-300:]}")
    try:
        payload = json.loads(result.stdout)
        data = payload.get("data") or {}
        lines = data.get("klines") or []
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid Eastmoney JSON: {result.stdout[:200]}") from exc
    if not lines:
        raise RuntimeError(f"empty Eastmoney history for {secid}")
    rows = [str(line).split(",") for line in lines]
    frame = pd.DataFrame(rows)
    if frame.shape[1] < 6:
        raise RuntimeError(f"unexpected Eastmoney columns for {secid}: {frame.shape[1]}")
    frame = frame.iloc[:, :8].copy()
    frame.columns = ["date", "open", "close", "high", "low", "volume", "amount", "amplitude"]
    return frame


def download_tencent_index_history(
    symbol: str,
    *,
    start_date: str = "2017-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch A-share index history from Tencent in bounded, auditable windows.

    Tencent returns at most 320 daily observations per request.  A 365-day
    calendar window stays below that limit for the CSI 300 and avoids hidden
    truncation when it is used as the Eastmoney fallback.
    """
    normalized = str(symbol).upper().split(".")[0]
    market_symbol = f"sh{normalized}" if str(symbol).upper().endswith(".SH") else f"sz{normalized}"
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize() if end_date else pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    if end < start:
        raise ValueError("Tencent index history end_date precedes start_date")

    curl = _curl_executable()
    windows: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + pd.Timedelta(days=364), end)
        url = (
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
            f"param={market_symbol},day,{cursor:%Y-%m-%d},{window_end:%Y-%m-%d},320,qfq"
        )
        result = subprocess.run(
            _curl_command(curl, url),
            capture_output=True,
            text=True,
            timeout=35,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Tencent curl exit={result.returncode}: {result.stderr[-300:]}")
        try:
            payload = json.loads(result.stdout)
            rows = ((payload.get("data") or {}).get(market_symbol) or {}).get("day") or []
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid Tencent index JSON: {result.stdout[:200]}") from exc
        if rows:
            part = pd.DataFrame(rows)
            if part.shape[1] < 6:
                raise RuntimeError(f"unexpected Tencent index columns for {symbol}: {part.shape[1]}")
            part = part.iloc[:, :6].copy()
            part.columns = ["date", "close", "open", "high", "low", "volume"]
            windows.append(part)
        cursor = window_end + pd.Timedelta(days=1)
    if not windows:
        raise RuntimeError(f"empty Tencent index history for {symbol}")
    return pd.concat(windows, ignore_index=True).drop_duplicates("date", keep="last")


def _curl_executable() -> str:
    for candidate in ("curl.exe", "curl"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("curl executable not found")


def _external_proxy() -> str:
    """Return an operator-supplied proxy without persisting credentials."""
    for name in ("KHQUANT_EXTERNAL_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _curl_command(curl: str, url: str) -> list[str]:
    command = [curl, "-s", "--max-time", "30"]
    proxy = _external_proxy()
    if proxy:
        command.extend(["--proxy", proxy])
    else:
        command.extend(["--noproxy", "*"])
    command.append(url)
    return command


def enrich_with_external_sentiment(
    factors: pd.DataFrame,
    config: dict[str, Any],
    cache: ExternalMarketCache | None = None,
) -> pd.DataFrame:
    """Attach only external observations available before each A-share open."""
    out = factors.copy()
    cfg = config.get("external_sentiment", {})
    if out.empty:
        return out
    instruments = configured_instruments(config)
    for column, default in {
        "external_sentiment_score": 0.0,
        "external_sentiment_confidence": 0.0,
        "external_sentiment_index_score": 0.0,
        "external_sentiment_sector_score": 0.0,
        "external_sentiment_coverage": 0.0,
        "external_sentiment_asof": "",
        "external_sentiment_status": "unavailable",
        "external_sentiment_reason": "external market cache unavailable",
        "external_sentiment_components": "{}",
    }.items():
        out[column] = default
    for item in instruments:
        out[f"external_sentiment_{item['id']}_score"] = 0.0
    if not bool(cfg.get("enabled", True)):
        out["external_sentiment_status"] = "disabled"
        out["external_sentiment_reason"] = "external sentiment disabled"
        return out

    market = (cache or ExternalMarketCache()).read_many(instruments)
    if market.empty:
        return out

    daily_scale = max(float(cfg.get("daily_move_scale", 0.025)), 1e-6)
    trend_scale = max(float(cfg.get("trend_move_scale", 0.07)), 1e-6)
    max_stale_days = int(cfg.get("max_stale_days", 5))
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])
    market["available_at"] = pd.to_datetime(market["available_at"], utc=True, errors="coerce")
    market = market.dropna(subset=["available_at"]).sort_values(["instrument", "available_at"])
    market["return_1d"] = market.groupby("instrument")["close"].pct_change()
    market["return_5d"] = market.groupby("instrument")["close"].pct_change(5)
    market["instrument_score"] = (
        0.65 * (market["return_1d"] / daily_scale).clip(-1.0, 1.0)
        + 0.35 * (market["return_5d"] / trend_scale).clip(-1.0, 1.0)
    ).clip(-1.0, 1.0)

    asof = pd.to_datetime(out["date"]).dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=25)
    left = pd.DataFrame({"row_id": out.index, "asof": asof.dt.tz_convert("UTC")}).sort_values("asof")
    pieces = []
    for item in instruments:
        instrument = str(item["id"])
        right = market[market["instrument"] == instrument][["available_at", "date", "instrument_score"]].dropna(subset=["instrument_score"])
        if right.empty:
            continue
        merged = pd.merge_asof(left, right.sort_values("available_at"), left_on="asof", right_on="available_at", direction="backward")
        merged["instrument"] = instrument
        merged["group"] = str(item.get("group", "index"))
        merged["weight"] = max(float(item.get("weight", 1.0)), 0.0)
        pieces.append(merged)
    if not pieces:
        return out

    joined = pd.concat(pieces, ignore_index=True)
    joined["age_days"] = (joined["asof"].dt.tz_localize(None).dt.normalize() - pd.to_datetime(joined["date"], errors="coerce")).dt.days
    joined["valid"] = joined["instrument_score"].notna() & joined["age_days"].between(0, max_stale_days)
    rows: dict[Any, dict[str, Any]] = {}
    total_weight = sum(max(float(item.get("weight", 1.0)), 0.0) for item in instruments) or 1.0
    for row_id, part in joined.groupby("row_id", sort=False):
        valid = part[part["valid"]].copy()
        coverage = float(valid["weight"].sum() / total_weight)
        score = float((valid["instrument_score"] * valid["weight"]).sum() / total_weight) if not valid.empty else 0.0
        index_part = valid[valid["group"] == "index"]
        sector_part = valid[valid["group"] == "sector"]
        index_score = _weighted_mean(index_part)
        sector_score = _weighted_mean(sector_part)
        newest = "" if valid.empty else pd.to_datetime(valid["date"]).max().strftime("%Y-%m-%d")
        status = "ok" if coverage >= 0.85 else "partial" if coverage > 0 else "stale"
        components = {
            str(item.instrument): {
                "score": round(float(item.instrument_score), 6),
                "date": pd.to_datetime(item.date).strftime("%Y-%m-%d"),
                "age_days": int(item.age_days),
                "group": str(item.group),
            }
            for item in valid.itertuples()
        }
        rows[row_id] = {
            "external_sentiment_score": round(float(np.clip(score, -1.0, 1.0)), 6),
            "external_sentiment_confidence": round(float(np.clip(coverage, 0.0, 1.0)), 6),
            "external_sentiment_index_score": round(index_score, 6),
            "external_sentiment_sector_score": round(sector_score, 6),
            "external_sentiment_coverage": round(coverage, 6),
            "external_sentiment_asof": newest,
            "external_sentiment_status": status,
            "external_sentiment_reason": f"coverage={coverage:.2f}; asof={newest or '-'}; markets={len(valid)}",
            "external_sentiment_components": json.dumps(components, ensure_ascii=False, sort_keys=True),
        }
        for item in instruments:
            instrument = str(item["id"])
            matching = valid[valid["instrument"] == instrument]
            rows[row_id][f"external_sentiment_{instrument}_score"] = (
                round(float(matching.iloc[-1]["instrument_score"]), 6) if not matching.empty else 0.0
            )
    for column in [name for name in out.columns if name.startswith("external_sentiment_")]:
        out.loc[list(rows), column] = [rows[index][column] for index in rows]
    out = _apply_regime_sentiment_blend(out, config)
    return out


def enrich_with_external_sentiment_many(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    cache: ExternalMarketCache | None = None,
) -> dict[str, pd.DataFrame]:
    """Attach the date-only external signal to many stocks with one calculation.

    External market observations do not
    depend on the A-share stock code.  Building the same point-in-time joins for
    every stock was therefore pure duplicate work.  This function creates one
    calendar over the union of requested dates, then maps those values back while
    preserving each input frame's index, order and non-external columns.
    """

    if not frames:
        return {}
    date_parts = [pd.to_datetime(frame["date"], errors="coerce") for frame in frames.values() if not frame.empty]
    if not date_parts:
        return {stock: enrich_with_external_sentiment(frame, config, cache) for stock, frame in frames.items()}
    dates = pd.concat(date_parts, ignore_index=True).dropna().drop_duplicates().sort_values()
    calendar = enrich_with_external_sentiment(pd.DataFrame({"date": dates}), config, cache)
    external_columns = [column for column in calendar.columns if column.startswith("external_sentiment_")]
    calendar = calendar.assign(_date_key=pd.to_datetime(calendar["date"], errors="coerce"))
    calendar = calendar.drop_duplicates("_date_key", keep="last").set_index("_date_key")

    enriched: dict[str, pd.DataFrame] = {}
    for stock, frame in frames.items():
        out = frame.copy()
        if out.empty:
            enriched[stock] = enrich_with_external_sentiment(out, config, cache)
            continue
        keys = pd.to_datetime(out["date"], errors="coerce")
        for column in external_columns:
            out[column] = keys.map(calendar[column]).to_numpy()
        enriched[stock] = out
    return enriched


def _apply_regime_sentiment_blend(out: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Blend the PIT-clean regime sentiment series into the external score.

    Enabled only via ``external_sentiment.regime_blend.enabled``; the disabled
    path is byte-identical to the pre-blend behaviour.  Dates without a regime
    observation (including z-score warm-up NaNs) keep the pure external score.
    See ADR_20260905_Regime_Sentiment_Blend.
    """

    blend = config.get("external_sentiment", {}).get("regime_blend", {})
    if not isinstance(blend, dict) or not bool(blend.get("enabled", False)):
        return out
    weight = min(max(float(blend.get("weight", 0.3)), 0.0), 1.0)
    source = str(blend.get("source") or "").strip()
    if source:
        path = Path(source)
    else:
        from my_strategy.core.paths import PROCESSED_DATA_ROOT

        path = PROCESSED_DATA_ROOT / "regime_sentiment" / "regime_sentiment_daily.parquet"

    out["external_sentiment_regime_score"] = 0.0
    if not path.exists():
        out["external_sentiment_reason"] = (
            out["external_sentiment_reason"].astype(str) + f"; regime_blend unavailable ({path.name})"
        )
        return out

    series = pd.read_parquet(path)
    series["date"] = pd.to_datetime(series["date"], errors="coerce")
    series = series.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date")
    score_map = pd.to_numeric(series["regime_sentiment_score"], errors="coerce")
    if "regime_sentiment_confidence" in series.columns:
        conf_map = pd.to_numeric(series["regime_sentiment_confidence"], errors="coerce")
    else:
        conf_map = pd.Series(dtype=float)

    keys = pd.to_datetime(out["date"], errors="coerce")
    regime_score = keys.map(score_map)
    regime_conf = keys.map(conf_map) if not conf_map.empty else pd.Series(np.nan, index=out.index)
    matched = regime_score.notna()
    if not bool(matched.any()):
        out["external_sentiment_reason"] = (
            out["external_sentiment_reason"].astype(str) + "; regime_blend no overlapping dates"
        )
        return out

    out.loc[matched, "external_sentiment_regime_score"] = regime_score[matched].round(6)
    blended_score = (
        (1.0 - weight) * pd.to_numeric(out["external_sentiment_score"], errors="coerce").fillna(0.0)
        + weight * regime_score.fillna(0.0)
    ).clip(-1.0, 1.0)
    blended_conf = (
        (1.0 - weight) * pd.to_numeric(out["external_sentiment_confidence"], errors="coerce").fillna(0.0)
        + weight * regime_conf.fillna(0.0)
    ).clip(0.0, 1.0)
    out.loc[matched, "external_sentiment_score"] = blended_score[matched].round(6)
    out.loc[matched, "external_sentiment_confidence"] = blended_conf[matched].round(6)
    note = f"; regime_blend w={weight:.2f}"
    out.loc[matched, "external_sentiment_reason"] = (
        out.loc[matched, "external_sentiment_reason"].astype(str) + note
    )
    return out


def _weighted_mean(frame: pd.DataFrame) -> float:
    if frame.empty or float(frame["weight"].sum()) <= 0:
        return 0.0
    return float((frame["instrument_score"] * frame["weight"]).sum() / frame["weight"].sum())


def _clean_history(frame: pd.DataFrame, instrument: str, definition: dict[str, Any] | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    item = definition or {}
    aliases = {
        "date": ["date", "日期"],
        "open": ["open", "今开", "开盘"],
        "high": ["high", "最高"],
        "low": ["low", "最低"],
        "close": ["close", "最新价", "收盘"],
    }
    out = pd.DataFrame()
    for target, names in aliases.items():
        source = next((name for name in names if name in frame.columns), None)
        out[target] = frame[source] if source else np.nan
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ["open", "high", "low", "close"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["instrument"] = instrument
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1) & (out["high"] >= out["low"])]
    if definition is None and "available_at" in frame.columns:
        out["available_at"] = frame["available_at"]
    else:
        out["available_at"] = [_availability_time(value, item) for value in out["date"]]
    extra_columns = ["source", "updated_at"]
    for column in extra_columns:
        if column in frame.columns:
            out[column] = frame[column]
    return out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _availability_time(value: pd.Timestamp, definition: dict[str, Any]) -> str:
    timezone = str(definition.get("timezone", "Asia/Shanghai"))
    close_time = str(definition.get("close_time", "15:00"))
    hour, minute = (int(part) for part in close_time.split(":", 1))
    local = pd.Timestamp(value).normalize() + pd.Timedelta(hours=hour, minutes=minute)
    return local.tz_localize(timezone).tz_convert("Asia/Shanghai").isoformat()


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
