"""Read-only raw-price input boundary for the frozen CZSC engine."""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_ROOT, PROJECT_ROOT, RAW_DATA_ROOT
from my_strategy.runtime_env import runtime_resource_root

CZSC_VERSION = "1.0.1"
SOURCE_SHA256 = "ab0e52644fe8936e0641372cb00a15e4b9d0bd93bb2c8d36402299e0d1af18c3"


def legacy_fuyao_source(value: Any) -> bool:
    source = str(value).lower()
    return "fuyao" in source and source != "fuyao_unadjusted_v1"


def normalize_symbol(symbol: str) -> str:
    value = str(symbol).strip().upper()
    if re.fullmatch(r"\d{6}", value):
        value += ".SH" if value.startswith("6") else ".BJ" if value.startswith(("4", "8", "9")) else ".SZ"
    if not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", value):
        raise ValueError("股票代码必须为六位数字及 SH/SZ/BJ 后缀")
    return value


def raw_path(db_path: str | Path | None = None) -> Path:
    return Path(db_path or os.environ.get("KHQUANT_RAW_DB") or RAW_DATA_ROOT / "khquant_raw.db").resolve()


@contextmanager
def _connection(db_path: str | Path | None):
    path = raw_path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"原始行情库不存在：{path}；请设置 KHQUANT_RAW_DB")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        yield connection
    finally:
        connection.close()


def date_cutoff(value: str | None) -> pd.Timestamp | None:
    """Date-only means that day's close; timestamps preserve the requested instant."""
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("无效日期")
    if stamp.tzinfo:
        stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{8}", str(value)):
        stamp += pd.Timedelta(hours=23, minutes=59, seconds=59)
    return stamp


def load_bars(symbol: str, start: str | None = None, end: str | None = None,
              as_of: str | None = None, *, db_path: str | Path | None = None) -> pd.DataFrame:
    """Read stored daily prices; volume is shares and amount is CNY.

    No factor columns or later quotes are substituted. Legacy Fuyao adjustment
    is disclosed, not relabelled. Invalid OHLC and duplicate dates fail closed.
    """
    symbol = normalize_symbol(symbol)
    cutoffs = [x for x in (date_cutoff(end), date_cutoff(as_of)) if x is not None]
    cutoff = min(cutoffs) if cutoffs else None
    clauses, params = ["stock = ?"], [symbol]
    if start:
        clauses.append("date >= ?")
        params.append(pd.Timestamp(start).date().isoformat())
    if cutoff is not None:
        clauses.append("date <= ?")
        params.append(cutoff.date().isoformat())
    with _connection(db_path) as connection:
        columns = {r[1] for r in connection.execute("PRAGMA table_info(stock_daily_normalized)")}
        required = {"stock", "date", "open", "high", "low", "close", "volume", "amount"}
        if not required <= columns:
            raise ValueError("原始行情库 schema 缺少字段：" + ",".join(sorted(required - columns)))
        extras = [x for x in ("source", "has_trade_price", "pct_chg") if x in columns]
        selected = "stock AS symbol,date,open,high,low,close,volume,amount"
        if extras:
            selected += "," + ",".join(extras)
        frame = pd.read_sql_query(f"SELECT {selected} FROM stock_daily_normalized WHERE {' AND '.join(clauses)} ORDER BY date", connection, params=params)
    if frame.empty:
        raise ValueError(f"{symbol} 在请求时点无本地行情")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["dt"] = frame["date"] + pd.Timedelta(hours=15)
    if cutoff is not None:
        frame = frame.loc[frame["dt"] <= cutoff].copy()
    if frame.empty:
        raise ValueError(f"{symbol} 在请求时点尚无已收盘行情")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError(f"{symbol} 行情日期重复或乱序")
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{symbol} 行情包含非有限数值")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame[["volume", "amount"]] < 0).any().any():
        raise ValueError(f"{symbol} 行情价格或量额不合法")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any() or (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError(f"{symbol} OHLC 范围不合法")
    frame["vol"] = frame["volume"]
    frame["id"] = np.arange(len(frame), dtype=int)
    legacy_count = int(frame["source"].map(legacy_fuyao_source).sum()) if "source" in frame else 0
    unverified_count = int((frame["has_trade_price"] != 1).sum()) if "has_trade_price" in frame else len(frame)
    frame.attrs.update({"price_basis": "mixed_legacy_adjustment_unverified" if legacy_count else "unadjusted", "legacy_fuyao_bars": legacy_count,
                        "unverified_trade_price_bars": unverified_count, "volume_unit": "shares", "amount_unit": "CNY", "raw_db": str(raw_path(db_path)),
                        "data_version": hashlib.sha256(frame.to_csv(index=False, float_format="%.10g").encode()).hexdigest()})
    return frame.reset_index(drop=True)


def list_symbols(query: str | None = None, limit: int | None = None, *, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    with _connection(db_path) as connection:
        sql = "SELECT stock AS symbol,name,first_date,last_date,raw_rows FROM securities"
        params: list[Any] = []
        if query:
            sql += " WHERE stock LIKE ? OR name LIKE ?"
            params = [f"%{query}%", f"%{query}%"]
        sql += " ORDER BY stock"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, min(int(limit), 10000)))
        return [dict(row) for row in connection.execute(sql, params)]


def latest_market_date(end: str | None = None, *, db_path: str | Path | None = None) -> str | None:
    """Return the latest stored market date, optionally bounded by a date."""
    with _connection(db_path) as connection:
        if end is None:
            return connection.execute("SELECT MAX(date) FROM stock_daily_normalized").fetchone()[0]
        cutoff = date_cutoff(end)
        if cutoff is None:
            raise ValueError("无效截止日期")
        return connection.execute("SELECT MAX(date) FROM stock_daily_normalized WHERE date <= ?", (cutoff.date().isoformat(),)).fetchone()[0]


def data_status(*, db_path: str | Path | None = None) -> dict[str, Any]:
    path = raw_path(db_path)
    if not path.is_file():
        return {"available": False, "raw_db": str(path), "reason": "请设置 KHQUANT_RAW_DB 到已有原始行情库"}
    with _connection(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(stock_daily_normalized)")}
        legacy = "SUM(CASE WHEN source LIKE '%fuyao%' AND source <> 'fuyao_unadjusted_v1' THEN 1 ELSE 0 END)" if "source" in columns else "0"
        unverified = "SUM(CASE WHEN has_trade_price IS NULL OR has_trade_price <> 1 THEN 1 ELSE 0 END)" if "has_trade_price" in columns else "COUNT(*)"
        summary = dict(connection.execute(f"SELECT MIN(date) AS start,MAX(date) AS end,COUNT(*) AS rows,COUNT(DISTINCT stock) AS symbols,{legacy} AS legacy_fuyao_bars,{unverified} AS unverified_trade_price_bars FROM stock_daily_normalized").fetchone())
        latest = connection.execute("SELECT COUNT(*) FROM stock_daily_normalized WHERE date=?", (summary["end"],)).fetchone()[0]
    return {"available": True, "raw_db": str(path), **summary, "data_end": summary["end"], "latest_date": summary["end"], "stocks": summary["symbols"],
            "source": "本地 SQLite 原始行情", "latest_symbols": latest,
            "price_basis": "mixed_legacy_adjustment_unverified" if summary["legacy_fuyao_bars"] else "unadjusted", "volume_unit": "shares", "amount_unit": "CNY", "frequencies": ["日线", "周线", "月线"]}


@lru_cache(maxsize=1)
def native_runtime():
    """Import the real PyO3 runtime; no fallback structure implementation exists."""
    try:
        distribution = importlib.metadata.distribution("czsc")
        version = distribution.version
    except ImportError as error:
        raise RuntimeError("CZSC 原生运行时未安装；请使用冻结的 CZSC 任务环境") from error
    if version != CZSC_VERSION:
        raise RuntimeError(f"需要 CZSC {CZSC_VERSION}，当前为 {version}")
    resources = runtime_resource_root()
    manifest_path = resources / "czsc-source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    source_install = origin.get("url", "").replace("\\", "/").rstrip("/").endswith("/vendor/czsc")
    wheel_hash = origin.get("archive_info", {}).get("hashes", {}).get("sha256")
    allowed_hashes = {manifest.get("built_wheel_sha256")}
    index_path = resources / "vendor" / "wheels" / "wheel-index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        source_hash = hashlib.sha256("\n".join(f"{entry['path']} {entry['sha256']}" for entry in manifest["files"]).encode()).hexdigest()
        if index.get("source_tree_sha256") != source_hash or index.get("package") != "czsc" or index.get("version") != CZSC_VERSION:
            raise RuntimeError("CZSC 平台 wheel 来源索引与冻结源码不匹配")
        allowed_hashes.update(entry["sha256"] for entry in index["wheels"])
    if not source_install and (not wheel_hash or wheel_hash not in allowed_hashes):
        raise RuntimeError("CZSC 安装来源不是冻结附件构建；同版本 PyPI 核心与附件不一致")
    if manifest.get("archive_sha256") != SOURCE_SHA256:
        raise RuntimeError("CZSC 源码冻结标识不匹配")
    source_root = resources / "vendor" / "czsc"
    for entry in manifest["files"]:
        path = source_root / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"CZSC 冻结源码校验失败：{entry['path']}")
    os.environ.setdefault("CZSC_HOME", str(ARTIFACT_ROOT / "czsc-cache"))
    import czsc
    return czsc


def to_raw_bar(row: Any):
    czsc = native_runtime()
    return czsc.RawBar(symbol=str(row.symbol), dt=row.dt, freq=czsc.Freq.D, id=int(row.id),
                        open=float(row.open), high=float(row.high), low=float(row.low), close=float(row.close),
                        vol=float(row.volume), amount=float(row.amount))
