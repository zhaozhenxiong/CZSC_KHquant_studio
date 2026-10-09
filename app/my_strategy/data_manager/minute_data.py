"""Minute-bar data center for intraday backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os

import pandas as pd

from .config import load_data_config, project_path
from .stock_pool import normalize_stock_code


SUPPORTED_MINUTE_PERIODS = {"1", "5", "15", "30", "60"}
MINUTE_COLUMNS = [
    "datetime",
    "date",
    "time",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "period",
    "source",
    "updated_at",
]


@dataclass
class MinuteDownloadResult:
    ok: bool
    data: pd.DataFrame | None = None
    source: str = ""
    error: str = ""


class MinuteLocalCache:
    def __init__(self, config: dict | None = None):
        self.config = config or load_data_config()
        self.data_root = project_path(self.config.get("data_root", "my_strategy/data"))
        self.minute_root = self.data_root / "raw" / "stock_minute"
        self.metadata_dir = self.data_root / "metadata"
        self.minute_root.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def minute_dir(self, period: str) -> Path:
        period = normalize_period(period)
        path = self.minute_root / f"{period}m"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def minute_path(self, code: str, period: str) -> Path:
        return self.minute_dir(period) / f"{normalize_stock_code(code)}.parquet"

    def read(self, code: str, period: str) -> pd.DataFrame:
        path = self.minute_path(code, period)
        if not path.exists():
            return pd.DataFrame(columns=MINUTE_COLUMNS)
        df = pd.read_parquet(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)

    def save(self, code: str, period: str, df: pd.DataFrame, source: str) -> None:
        clean = clean_minute_bars(df, normalize_stock_code(code), normalize_period(period))
        if clean.empty:
            raise ValueError("refuse to save empty minute cache")
        clean["source"] = source
        clean["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _atomic_write_parquet(clean, self.minute_path(code, period))

    def merge(self, code: str, period: str, new_df: pd.DataFrame, source: str) -> pd.DataFrame:
        old = self.read(code, period)
        frames = [df for df in [old, new_df] if df is not None and not df.empty]
        if not frames:
            return pd.DataFrame(columns=MINUTE_COLUMNS)
        merged = pd.concat(frames, ignore_index=True)
        clean = clean_minute_bars(merged, normalize_stock_code(code), normalize_period(period))
        self.save(code, period, clean, source=source)
        return clean

    def slice(self, code: str, period: str, start: str, end: str) -> pd.DataFrame:
        df = self.read(code, period)
        if df.empty:
            return df
        start_dt, end_dt = parse_minute_range(start, end)
        mask = (df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)
        return df.loc[mask].reset_index(drop=True)

    def has_range(self, code: str, period: str, start: str, end: str) -> bool:
        df = self.read(code, period)
        if df.empty:
            return False
        start_dt, end_dt = parse_minute_range(start, end)
        dts = pd.to_datetime(df["datetime"])
        return dts.min() <= start_dt and dts.max() >= end_dt


class MinuteDataDownloader:
    def __init__(self, config: dict | None = None):
        self.config = config or load_data_config()

    def download(self, code: str, period: str, start: str, end: str) -> MinuteDownloadResult:
        code = normalize_stock_code(code)
        period = normalize_period(period)
        sources = self.config.get("minute_data_sources", {})
        source_list = [sources.get("primary", "akshare"), *sources.get("fallback", ["baostock"])]
        errors: list[str] = []
        for source in source_list:
            source = str(source).lower()
            try:
                if source == "akshare":
                    df = self._download_akshare(code, period, start, end)
                elif source == "baostock":
                    df = self._download_baostock(code, period, start, end)
                else:
                    errors.append(f"{source}: unsupported source")
                    continue
                clean = clean_minute_bars(df, code, period)
                if clean.empty:
                    errors.append(f"{source}: empty data")
                    continue
                return MinuteDownloadResult(True, clean, source, "")
            except Exception as exc:
                errors.append(f"{source}: {exc}")
        return MinuteDownloadResult(False, None, "", " | ".join(errors))

    @staticmethod
    def _download_akshare(code: str, period: str, start: str, end: str) -> pd.DataFrame:
        import akshare as ak

        symbol = code.split(".")[0]
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol,
            start_date=normalize_minute_datetime(start),
            end_date=normalize_minute_datetime(end, end_of_day=True),
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
        return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    @staticmethod
    def _download_baostock(code: str, period: str, start: str, end: str) -> pd.DataFrame:
        if period == "1":
            raise ValueError("baostock does not support 1-minute bars")
        import baostock as bs

        lg = bs.login()
        if getattr(lg, "error_code", "0") != "0":
            raise RuntimeError(getattr(lg, "error_msg", "baostock login failed"))
        try:
            market = "sh" if code.endswith(".SH") else "sz"
            bs_code = f"{market}.{code.split('.')[0]}"
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,time,code,open,high,low,close,volume,amount",
                start_date=str(pd.to_datetime(start).date()),
                end_date=str(pd.to_datetime(end).date()),
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
                return df
            raw_time = df["time"].astype(str)
            df["datetime"] = raw_time.where(raw_time.str.len() >= 14, df["date"].astype(str) + raw_time.str.slice(0, 6))
            return df
        finally:
            bs.logout()


class MinuteDataCenter:
    def __init__(self, config: dict | None = None):
        self.config = config or load_data_config()
        self.cache = MinuteLocalCache(self.config)
        self.downloader = MinuteDataDownloader(self.config)

    def get_stock_minute(
        self,
        code: str,
        period: str = "5",
        start: str | None = None,
        end: str | None = None,
        auto_update: bool = True,
        allow_old_cache: bool | None = None,
    ) -> pd.DataFrame:
        code = normalize_stock_code(code)
        period = normalize_period(period)
        start = start or datetime.now().strftime("%Y-%m-%d 09:30:00")
        end = end or datetime.now().strftime("%Y-%m-%d 15:00:00")
        allow_old_cache = self.config.get("cache", {}).get("allow_use_old_cache", True) if allow_old_cache is None else allow_old_cache

        if self.cache.has_range(code, period, start, end):
            return self.cache.slice(code, period, start, end)

        if auto_update:
            result = self.downloader.download(code, period, start, end)
            if result.ok and result.data is not None and not result.data.empty:
                self.cache.merge(code, period, result.data, result.source)
                if self.cache.has_range(code, period, start, end):
                    return self.cache.slice(code, period, start, end)

        old = self.cache.slice(code, period, start, end)
        if allow_old_cache and not old.empty:
            return old
        raise RuntimeError(f"No usable minute data for {code} {period}m ({start} ~ {end})")


def get_stock_minute(code: str, period: str, start: str, end: str, auto_update: bool = True) -> pd.DataFrame:
    return MinuteDataCenter().get_stock_minute(code, period, start, end, auto_update=auto_update)


def clean_minute_bars(df: pd.DataFrame, code: str, period: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=MINUTE_COLUMNS)
    out = df.copy()
    if "datetime" not in out.columns and {"date", "time"}.issubset(out.columns):
        out["datetime"] = out["date"].astype(str) + out["time"].astype(str)
    out["datetime"] = parse_datetime_series(out["datetime"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = 0.0 if col in {"volume", "amount"} else pd.NA
    out["code"] = code
    out["period"] = period
    out["date"] = out["datetime"].dt.strftime("%Y-%m-%d")
    out["time"] = out["datetime"].dt.strftime("%H:%M:%S")
    if "source" not in out.columns:
        out["source"] = ""
    if "updated_at" not in out.columns:
        out["updated_at"] = ""
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    out = out[out["close"] > 0]
    out = out[out["high"] >= out["low"]]
    out = out.drop_duplicates("datetime", keep="last").sort_values("datetime").reset_index(drop=True)
    return out[[col for col in MINUTE_COLUMNS if col in out.columns]].copy()


def normalize_period(period: str | int) -> str:
    value = str(period).lower().replace("m", "").strip()
    if value not in SUPPORTED_MINUTE_PERIODS:
        raise ValueError(f"Unsupported minute period: {period}; expected one of {sorted(SUPPORTED_MINUTE_PERIODS)}")
    return value


def normalize_minute_datetime(value: str, end_of_day: bool = False) -> str:
    text = str(value)
    if len(text) <= 10:
        text = f"{text} {'15:00:00' if end_of_day else '09:30:00'}"
    return pd.to_datetime(text).strftime("%Y-%m-%d %H:%M:%S")


def parse_minute_range(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.to_datetime(normalize_minute_datetime(start)), pd.to_datetime(normalize_minute_datetime(end, end_of_day=True))


def parse_datetime_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    compact = text.str.replace(r"\D", "", regex=True)
    parsed = pd.to_datetime(text, errors="coerce")
    need_compact = parsed.isna() & compact.str.len().ge(14)
    if need_compact.any():
        parsed.loc[need_compact] = pd.to_datetime(compact.loc[need_compact].str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    return parsed


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
