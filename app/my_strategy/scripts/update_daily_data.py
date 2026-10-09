#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Daily incremental update entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore[misc,assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.core.tz import local_now
from my_strategy.data_manager.config import load_data_config
from my_strategy.data_manager.local_cache import LocalCache
from my_strategy.data_manager.stock_pool import StockPoolManager
from my_strategy.data_manager.updater import DailyUpdater
from my_strategy.data_manager.industry_enricher import IndustryEnricher
from my_strategy.storage.access_layer import get_data_access
from my_strategy.validation.data_quality import DataQualityChecker

POOL_STOCK_TOKENS = {"all", "db", "local", "local_db", "*"}
QUALITY_CHECK_WINDOW_DAYS = 5


def _quote_symbol(code: str) -> str:
    raw = str(code).split(".")[0]
    return ("sh" if str(code).upper().endswith(".SH") else "sz") + raw


def _curl_executable() -> str:
    """Return the available curl executable for Windows hosts and Linux containers."""
    for candidate in ("curl.exe", "curl"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("curl executable not found")


def _default_for_column(df: pd.DataFrame, column: str):
    dtype = df[column].dtype
    if pd.api.types.is_integer_dtype(dtype):
        return 0
    if pd.api.types.is_float_dtype(dtype):
        return 0.0
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return pd.NaT
    return ""


def _cast_like(df: pd.DataFrame, column: str, value):
    dtype = df[column].dtype
    if pd.api.types.is_integer_dtype(dtype):
        return int(float(value)) if value not in ("", None) else 0
    if pd.api.types.is_float_dtype(dtype):
        return float(value) if value not in ("", None) else 0.0
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return pd.to_datetime(value)
    return "" if value is None else str(value)


def _write_realtime_row(cache: LocalCache, code: str, values: dict) -> bool:
    old = cache.read_stock_daily(code)
    if old.empty or "date" not in old.columns:
        return False
    old["date"] = pd.to_datetime(old["date"])
    row = {col: _default_for_column(old, col) for col in old.columns}
    for column, value in values.items():
        if column in row:
            row[column] = _cast_like(old, column, value)
    new_row = pd.DataFrame([row], columns=old.columns)
    out = pd.concat([old, new_row], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    cache.save_stock_daily(code, new_row, source="tencent_realtime", incremental=True)
    return True


def update_today_realtime(config: dict, stocks: list[str] | None, end_date: str | None, batch_size: int = 30, use_progress_bar: bool = True) -> pd.DataFrame:
    """Append today's valid realtime OHLC row to the DB-first local cache.

    The updater refuses quotes with zero OHLC or volume, so suspended/no-trade
    stocks keep their previous cache unchanged.
    """
    target_date = end_date or local_now().strftime("%Y-%m-%d")
    target_stamp = target_date.replace("-", "")
    cache = LocalCache(config)
    if stocks:
        pool = StockPoolManager(config).build_pool(stocks=stocks)
    else:
        db_stocks = get_data_access().list_stocks()
        if db_stocks:
            pool = pd.DataFrame({"stock": db_stocks})
        elif get_data_access().settings.write_files:
            paths = sorted(cache.stock_daily_dir.glob("*.parquet"))
            pool = pd.DataFrame({"stock": [path.stem for path in paths]})
        else:
            pool = pd.DataFrame({"stock": []})

    rows = []
    codes = pool["stock"].astype(str).drop_duplicates().tolist()
    total_batches = max(1, (len(codes) + batch_size - 1) // batch_size)
    progress_bar = None
    if use_progress_bar and tqdm is not None and sys.stdout.isatty():
        progress_bar = tqdm(
            total=total_batches,
            desc="Fetching realtime quotes",
            unit="batch",
            ncols=80,
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for batch_index, start in enumerate(range(0, len(codes), batch_size)):
        batch = codes[start : start + batch_size]
        url = "https://qt.gtimg.cn/q=" + ",".join(_quote_symbol(code) for code in batch)
        try:
            result = subprocess.run(
                [_curl_executable(), "-s", "--noproxy", "*", "--max-time", "20", url],
                capture_output=True,
                timeout=25,
            )
            text = result.stdout.decode("gbk", errors="replace")
        except Exception as exc:
            for code in batch:
                rows.append({"code": code, "last_date": "", "status": "failed", "error_msg": str(exc)})
            if progress_bar is not None:
                progress_bar.update(1)
            continue

        seen: set[str] = set()
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
            if code not in batch:
                continue
            seen.add(code)
            stamp = parts[30].strip() if len(parts) > 30 else ""
            if not stamp.startswith(target_stamp):
                rows.append({"code": code, "last_date": "", "status": "skipped", "error_msg": f"quote date {stamp[:8]} != {target_stamp}"})
                continue
            try:
                close = float(parts[3]) if parts[3] else 0.0
                open_price = float(parts[5]) if parts[5] else 0.0
                high = float(parts[33]) if parts[33] else 0.0
                low = float(parts[34]) if parts[34] else 0.0
                volume = float(parts[6].replace(",", "")) if parts[6] else 0.0
                pct_chg = float(parts[32]) if len(parts) > 32 and parts[32] else 0.0
            except Exception as exc:
                rows.append({"code": code, "last_date": "", "status": "failed", "error_msg": f"bad quote: {exc}"})
                continue
            if min(open_price, high, low, close) <= 0 or high < low or volume <= 0:
                rows.append({"code": code, "last_date": "", "status": "skipped", "error_msg": "zero OHLC or volume"})
                continue

            values = {
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
                "source": "tencent_realtime",
                "updated_at": now_text,
            }
            try:
                ok = _write_realtime_row(cache, code, values)
            except Exception as exc:
                rows.append({"code": code, "last_date": "", "status": "failed", "error_msg": str(exc)})
                continue
            rows.append({"code": code, "last_date": target_date if ok else "", "status": "success" if ok else "failed", "error_msg": "" if ok else "cache missing/invalid"})

        for code in batch:
            if code not in seen:
                rows.append({"code": code, "last_date": "", "status": "failed", "error_msg": "quote missing"})
        if progress_bar is not None:
            progress_bar.update(1)
        if start + batch_size < len(codes):
            time.sleep(0.2)

    if progress_bar is not None:
        progress_bar.close()
    return pd.DataFrame(rows)


def summarize_update(
    *,
    report: pd.DataFrame,
    update_id: str,
    started_at: str,
    mode: str,
    stocks_arg: str,
    end_date: str,
    quality_summary: dict[str, Any] | None = None,
) -> dict:
    storage = get_data_access()
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    status = report.get("status", pd.Series(dtype=str)).astype(str).str.lower() if not report.empty else pd.Series(dtype=str)
    summary = {
        "update_id": update_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "mode": mode,
        "stocks": stocks_arg,
        "end_date": end_date,
        "storage": storage.settings.__dict__,
        "total": int(len(report)),
        "success": int((status == "success").sum()),
        "failed": int((status == "failed").sum()),
        "skipped": int((status == "skipped").sum()),
        "quality_check": quality_summary or {},
    }
    if storage.settings.write_db:
        summary["db_update_run"] = storage.record_data_update(
            update_id=update_id,
            started_at=started_at,
            finished_at=finished_at,
            mode=mode,
            stocks_requested=stocks_arg,
            end_date=end_date,
            report=report,
        )
        summary["integrity"] = {
            "raw_db": {"path": str(storage.settings.raw_db), "exists": storage.settings.raw_db.exists()},
        }
    return summary


def _emit_stage(stage: str, message: str) -> None:
    """Report a post-update stage to dashboard subprocess consumers."""
    payload = {"event": "stage", "stage": stage, "message": message}
    sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stderr.flush()


def _metadata_refresh_status(metadata: pd.DataFrame, stocks: list[str], end_date: str) -> dict[str, Any]:
    """Tell whether a CSRC source snapshot has already been fetched for ``end_date``."""
    requested = sorted({str(stock).strip() for stock in stocks if str(stock).strip()})
    if not requested or metadata.empty or "stock" not in metadata.columns:
        return {"is_current": False, "requested_stocks": len(requested), "current_stocks": 0}
    frame = metadata.copy()
    frame["stock"] = frame["stock"].fillna("").astype(str).str.strip()
    frame = frame.drop_duplicates("stock", keep="last").set_index("stock")
    rows = frame.reindex(requested)
    fetched = pd.to_datetime(rows.get("industry_source_fetched_at"), errors="coerce")
    eligible = rows.get("classification_eligible", pd.Series("", index=rows.index)).fillna("").astype(str).str.lower()
    current = fetched.dt.date.ge(pd.Timestamp(end_date).date()) & eligible.isin({"true", "1", "yes"})
    snapshot_current = bool(fetched.dt.date.ge(pd.Timestamp(end_date).date()).any())
    return {
        "is_current": snapshot_current,
        "requested_stocks": len(requested),
        "current_stocks": int(current.sum()),
        "missing_or_ineligible_stocks": int((~current).sum()),
    }




def _run_quality_check(
    config: dict,
    report: pd.DataFrame,
    end_date: str,
    update_id: str,
) -> dict[str, Any]:
    """Run a conservative quality check on the stocks just updated.

    The check uses a small window around end_date so the daily update stays fast.
    Only error-level issues affect the exit status.
    """
    if report.empty or "code" not in report.columns:
        return {"skipped": True, "reason": "no stocks updated"}
    stocks = report.loc[report["status"].astype(str).str.lower() == "success", "code"].astype(str).unique().tolist()
    if not stocks:
        return {"skipped": True, "reason": "no successful updates"}
    end = pd.to_datetime(end_date)
    start = (end - pd.Timedelta(days=QUALITY_CHECK_WINDOW_DAYS)).strftime("%Y-%m-%d")
    check_end = (end + pd.Timedelta(days=QUALITY_CHECK_WINDOW_DAYS)).strftime("%Y-%m-%d")
    try:
        checker = DataQualityChecker(config=config)
        quality_report = checker.check(
            stocks=stocks,
            start_date=start,
            end_date=check_end,
            run_id=f"{update_id}_quality",
        )
        try:
            checker.save_report(quality_report)
        except Exception as exc:
            logging.getLogger(__name__).warning("Failed to persist quality report: %s", exc)
        summary = {
            "run_id": quality_report.run_id,
            "checked_at": quality_report.checked_at,
            "start_date": quality_report.start_date,
            "end_date": quality_report.end_date,
            "stocks_checked": len(quality_report.stocks),
            "total_issues": len(quality_report.issues),
            "errors": sum(1 for issue in quality_report.issues if issue.severity == "error"),
            "warnings": sum(1 for issue in quality_report.issues if issue.severity == "warning"),
            "has_errors": quality_report.has_errors(),
        }
        return summary
    except Exception as exc:
        logging.getLogger(__name__).warning("Quality check failed: %s", exc)
        return {"skipped": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Update KHQuant DB-first local daily data cache.")
    parser.add_argument("--config", default="my_strategy/configs/data_config.yaml")
    parser.add_argument("--mode", default=None, choices=["custom", "all_a", "db", "local", "local_db"], nargs="?")
    parser.add_argument("--stocks", default="", help="Comma-separated stock codes.")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--today-realtime", action="store_true", help="Append valid realtime OHLC for --end-date/today into the DB-first local cache.")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--max-stocks", type=int, default=0, help="Optional cap for pool-based updates; 0 means no cap.")
    parser.add_argument("--update-id", default="", help="Optional stable update id recorded in raw SQLite.")
    parser.add_argument("--json", action="store_true", help="Print a JSON payload with rows and DB summary.")
    parser.add_argument("--skip-quality-check", action="store_true", help="Skip the post-update data quality check.")
    parser.add_argument("--skip-metadata-refresh", action="store_true", help="Skip refreshing stock metadata after daily bars update.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent stock download workers (1 = serial).")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable the interactive progress bar.")
    args = parser.parse_args()

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    update_id = args.update_id or f"data_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = load_data_config(args.config)
    stocks_arg = args.stocks.strip()
    effective_mode = args.mode or config.get("stock_pool", {}).get("default_mode", "custom")
    if stocks_arg.lower() in POOL_STOCK_TOKENS:
        stocks = None
        if not args.mode:
            effective_mode = "db" if stocks_arg.lower() in {"all", "*"} else stocks_arg.lower()
    else:
        stocks = [item.strip() for item in args.stocks.split(",") if item.strip()] or None
    use_progress_bar = not args.no_progress_bar and not args.json
    if args.today_realtime:
        if stocks is None and args.max_stocks > 0:
            pool = StockPoolManager(config).build_pool(mode=effective_mode).head(args.max_stocks)
            stocks = pool["stock"].astype(str).tolist()
        report = update_today_realtime(config, stocks=stocks, end_date=args.end_date, batch_size=args.batch_size, use_progress_bar=use_progress_bar)
    else:
        progress_rows: list[dict[str, Any]] = []

        def _emit_progress(progress: dict[str, Any]) -> None:
            progress_rows.append(progress["row"])
            payload = {
                "event": "progress",
                "update_id": update_id,
                "current": progress["current"],
                "total": progress["total"],
                "row": progress["row"],
            }
            sys.stderr.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            sys.stderr.flush()

        updater = DailyUpdater(config, progress_callback=_emit_progress, workers=args.workers, use_progress_bar=use_progress_bar)
        report = updater.update_pool(mode=effective_mode, stocks=stocks, end_date=args.end_date, max_stocks=args.max_stocks or None)

    quality_summary = None
    if not args.skip_quality_check:
        _emit_stage("quality_check", "日线更新完成，正在校验数据质量…")
        quality_summary = _run_quality_check(
            config=config,
            report=report,
            end_date=args.end_date or local_now().strftime("%Y-%m-%d"),
            update_id=update_id,
        )
        _emit_stage("quality_check", "数据质量检查完成")

    metadata_summary: dict[str, Any] = {"skipped": True}
    if not args.skip_metadata_refresh:
        _emit_stage("metadata_refresh", "数据质量检查完成，正在刷新行业元数据…")
        try:
            enricher = IndustryEnricher(config=config)
            updated_codes = [
                str(c)
                for c in report.get("code", pd.Series(dtype=str)).dropna().astype(str).unique()
                if str(c)
            ]
            refresh_status = _metadata_refresh_status(
                enricher._read_metadata(), updated_codes, args.end_date or local_now().strftime("%Y-%m-%d")
            )
            if refresh_status["is_current"]:
                metadata_summary = {
                    "skipped": True,
                    "reason": "CSRC source snapshot already fetched today; retain its recorded coverage",
                    **refresh_status,
                }
            else:
                refreshed_df, industry_report = enricher.refresh_csrc_industry(
                    stocks=updated_codes,
                    run_id=f"{update_id}-industry",
                )
                metadata_summary = {
                    "skipped": False,
                    "stocks_checked": len(updated_codes),
                    "metadata_rows": len(refreshed_df),
                    "industry_classification": industry_report,
                    "refreshed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.warning("Metadata refresh failed: %s", exc)
            metadata_summary = {"skipped": False, "error": str(exc)}
        _emit_stage("metadata_refresh", "行业元数据刷新完成")

    summary = summarize_update(
        report=report,
        update_id=update_id,
        started_at=started_at,
        mode=effective_mode,
        stocks_arg=args.stocks,
        end_date=args.end_date or local_now().strftime("%Y-%m-%d"),
        quality_summary=quality_summary,
    )
    summary["metadata_refresh"] = metadata_summary
    exit_code = 2 if summary["failed"] or (quality_summary or {}).get("has_errors") else 0
    if args.json:
        output = {"summary": summary, "rows": report.to_dict("records")}
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
        sys.exit(exit_code)
    if report.empty:
        print("No stocks to update.")
    else:
        print(report.to_string(index=False))
    if quality_summary:
        print("\nData quality check:")
        print(json.dumps(quality_summary, ensure_ascii=False, indent=2, default=str))
    print("\nUpdate summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
