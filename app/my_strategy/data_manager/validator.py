"""Data validators for KHQuant Local Data Center."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_DAILY_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
OPTIONAL_DAILY_COLUMNS = [
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
class ValidationResult:
    ok: bool
    error: str = ""


class DataValidator:
    """Validate and normalize market data before it can touch cache."""

    @staticmethod
    def clean_stock_daily(df: pd.DataFrame, code: str | None = None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "code", *REQUIRED_DAILY_COLUMNS[1:], *OPTIONAL_DAILY_COLUMNS])

        out = df.copy()
        if "date" not in out.columns:
            return out

        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for col in [
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
        ]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        if code:
            out["code"] = code
        elif "code" not in out.columns:
            out["code"] = ""

        keep = ["date", "code", "open", "high", "low", "close", "volume"]
        for col in OPTIONAL_DAILY_COLUMNS:
            if col in out.columns and col not in keep:
                keep.append(col)
        out = out[[col for col in keep if col in out.columns]].copy()
        out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        out = out.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"]).reset_index(drop=True)
        return out

    @staticmethod
    def validate_stock_daily(
        df: pd.DataFrame,
        min_rows: int = 100,
        require_min_rows: bool = True,
    ) -> ValidationResult:
        if df is None or df.empty:
            return ValidationResult(False, "empty dataframe")

        missing = [col for col in REQUIRED_DAILY_COLUMNS if col not in df.columns]
        if missing:
            return ValidationResult(False, f"missing columns: {missing}")

        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.isna().any():
            return ValidationResult(False, "invalid date values")
        if "code" in df.columns and df["code"].nunique(dropna=False) > 1:
            if df.duplicated(subset=["code", "date"]).any():
                return ValidationResult(False, "duplicated date values")
        else:
            if dates.duplicated().any():
                return ValidationResult(False, "duplicated date values")
        if require_min_rows and len(df) < min_rows:
            return ValidationResult(False, f"too few rows: {len(df)} < {min_rows}")
        if (pd.to_numeric(df["close"], errors="coerce") <= 0).any():
            return ValidationResult(False, "close must be positive")
        if (pd.to_numeric(df["high"], errors="coerce") < pd.to_numeric(df["low"], errors="coerce")).any():
            return ValidationResult(False, "high must be >= low")
        return ValidationResult(True, "")
