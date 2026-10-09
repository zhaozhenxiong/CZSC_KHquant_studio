"""Daily incremental update workflow."""

from __future__ import annotations

from datetime import datetime
import logging
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from my_strategy.core.tz import local_now

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore[misc,assignment]

from .config import load_data_config
from .downloader import DataDownloader, DownloadResult
from .local_cache import LocalCache
from .stock_pool import StockPoolManager, normalize_stock_code
from .validator import DataValidator


logger = logging.getLogger(__name__)


class DailyUpdater:
    def __init__(
        self,
        config: dict | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        workers: int = 1,
        use_progress_bar: bool = True,
    ):
        self.config = config or load_data_config()
        self.cache = LocalCache(self.config)
        self.downloader = DataDownloader(self.config)
        self.progress_callback = progress_callback
        self.workers = max(1, int(workers))
        self.use_progress_bar = bool(use_progress_bar)

    def update_one(self, code: str, start_date: str | None = None, end_date: str | None = None) -> dict:
        code = normalize_stock_code(code)
        update_cfg = self.config.get("update", {})
        start_date = start_date or update_cfg.get("start_date", "2015-01-01")
        end_date = end_date or local_now().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        old_df = self.cache.read_stock_daily(code)
        old_last_date = None if old_df.empty else pd.to_datetime(old_df["date"]).max()
        old_has_trade_prices = False
        if "trade_close" in old_df.columns:
            old_has_trade_prices = pd.to_numeric(old_df["trade_close"], errors="coerce").notna().any()

        download_start = pd.to_datetime(start_date)
        if update_cfg.get("auto_incremental", True) and old_last_date is not None and old_has_trade_prices:
            download_start = old_last_date + pd.Timedelta(days=1)

        if old_last_date is not None and old_last_date >= pd.to_datetime(end_date) and old_has_trade_prices:
            row = self._status_row(code, old_last_date, len(old_df), "local_cache", "success", "")
            self.cache.update_status(row)
            self.cache.append_log({**row, "time": now, "action": "skip", "start_date": "", "end_date": end_date})
            return row

        if download_start > pd.to_datetime(end_date):
            row = self._status_row(code, old_last_date, len(old_df), "local_cache", "success", "")
            self.cache.update_status(row)
            self.cache.append_log({**row, "time": now, "action": "skip", "start_date": "", "end_date": end_date})
            return row

        result = self.downloader.download_stock_daily(code, download_start.strftime("%Y-%m-%d"), end_date)
        if not result.ok or result.data is None or result.data.empty:
            # If the requested interval is ahead of the latest cached date (e.g.
            # today is a non-trading day or the market has not closed yet) and
            # the upstream returns empty, the local cache is already up to date.
            if old_last_date is not None and old_last_date >= download_start - pd.Timedelta(days=1):
                row = self._status_row(code, old_last_date, len(old_df), "local_cache", "success", "")
                self.cache.update_status(row)
                self.cache.append_log({**row, "time": now, "action": "skip", "start_date": download_start.strftime("%Y-%m-%d"), "end_date": end_date})
                return row
            row = self._status_row(code, old_last_date, len(old_df), "local_cache", "failed", result.error)
            self.cache.update_status(row)
            self.cache.append_log({**row, "time": now, "action": "update", "start_date": download_start.strftime("%Y-%m-%d"), "end_date": end_date})
            return row

        new_df = DataValidator.clean_stock_daily(result.data, code)
        new_valid = DataValidator.validate_stock_daily(new_df, min_rows=1, require_min_rows=False)
        if not new_valid.ok:
            row = self._status_row(code, old_last_date, len(old_df), result.source, "failed", new_valid.error)
            self.cache.update_status(row)
            self.cache.append_log({**row, "time": now, "action": "update", "start_date": download_start.strftime("%Y-%m-%d"), "end_date": end_date})
            return row

        merged = pd.concat([old_df, new_df], ignore_index=True) if not old_df.empty else new_df
        merged = DataValidator.clean_stock_daily(merged, code)
        min_rows_for_valid = int(self.config.get("cache", {}).get("min_rows_for_valid_stock", 100))
        # A short listing history is not malformed market data. Persist valid
        # OHLCV immediately and leave warm-up eligibility to model/factor
        # consumers, which already enforce their own minimum-history gates.
        merged_valid = DataValidator.validate_stock_daily(merged, require_min_rows=False)
        if not merged_valid.ok:
            row = self._status_row(code, old_last_date, len(old_df), result.source, "failed", merged_valid.error)
            self.cache.update_status(row)
            self.cache.append_log({**row, "time": now, "action": "update", "start_date": download_start.strftime("%Y-%m-%d"), "end_date": end_date})
            return row

        # Validation uses the complete local history, but persistence only
        # writes the newly downloaded/corrected dates into SQLite.
        self.cache.save_stock_daily(code, new_df, source=result.source, incremental=True)
        last_date = pd.to_datetime(merged["date"]).max()
        row = self._status_row(
            code,
            last_date,
            len(merged),
            result.source,
            "success",
            self._short_history_warning(len(merged), min_rows_for_valid),
        )
        self.cache.update_status(row)
        self.cache.append_log({**row, "time": now, "action": "update", "start_date": download_start.strftime("%Y-%m-%d"), "end_date": end_date})
        return row

    def update_batch(
        self,
        stocks: list[str],
        end_date: str | None = None,
        *,
        batch_cb: Callable[[list[dict], int], None] | None = None,
    ) -> pd.DataFrame:
        """Update many stocks with parallel sub-batch downloads.

        Sub-batch sizing is adaptive.  Codes that need only a short incremental
        window ride together in large batches (``tushare.bulk_batch_size``,
        default 200), keeping each tushare response far under the 6000-row cap;
        codes that need a long/full history go in small batches sized by
        ``tushare.batch_size`` (default 2) so their rows are never truncated.
        Each completed batch is validated and persisted immediately, and
        ``batch_cb`` is invoked with its status rows so callers can surface live
        progress instead of waiting for the whole universe to finish.
        """
        codes = [normalize_stock_code(code) for code in stocks]
        end_date = end_date or local_now().strftime("%Y-%m-%d")
        end_dt = pd.to_datetime(end_date)
        update_cfg = self.config.get("update", {})
        global_start = update_cfg.get("start_date", "2015-01-01")
        auto_incremental = update_cfg.get("auto_incremental", True)
        min_rows_for_valid = int(self.config.get("cache", {}).get("min_rows_for_valid_stock", 100))
        batch_size = int(self.config.get("tushare", {}).get("batch_size", 300))
        bulk_batch = int(self.config.get("tushare", {}).get("bulk_batch_size", 200))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        old_frames: dict[str, pd.DataFrame] = {}
        download_start_by_code: dict[str, str] = {}
        skip_codes: list[str] = []

        for code in codes:
            old_df = self.cache.read_stock_daily(code)
            old_last_date = None if old_df.empty else pd.to_datetime(old_df["date"]).max()
            old_has_trade_prices = False
            if "trade_close" in old_df.columns:
                old_has_trade_prices = pd.to_numeric(old_df["trade_close"], errors="coerce").notna().any()

            download_start = pd.to_datetime(global_start)
            if auto_incremental and old_last_date is not None and old_has_trade_prices:
                download_start = old_last_date + pd.Timedelta(days=1)

            if old_last_date is not None and old_last_date >= end_dt and old_has_trade_prices:
                skip_codes.append(code)
                old_frames[code] = old_df
                continue
            if download_start > end_dt:
                skip_codes.append(code)
                old_frames[code] = old_df
                continue

            old_frames[code] = old_df
            download_start_by_code[code] = download_start.strftime("%Y-%m-%d")

        rows: list[dict] = []
        for code in skip_codes:
            old_df = old_frames[code]
            last_date = None if old_df.empty else pd.to_datetime(old_df["date"]).max()
            rows.append(self._status_row(code, last_date, len(old_df), "local_cache", "success", ""))

        # Surface already-up-to-date stocks immediately so live counters show
        # skips before any download begins.
        if batch_cb is not None and rows:
            batch_cb(rows, len(rows))

        if not download_start_by_code:
            self.cache.update_status_many(rows)
            self.cache.append_log_many(
                [{**row, "time": now, "action": "skip", "start_date": "", "end_date": end_date} for row in rows]
            )
            return pd.DataFrame(rows)

        # Adaptive batching by data need: short-window codes batch large, codes
        # needing long/full history batch small (never truncating at 6000 rows).
        download_codes = list(download_start_by_code.keys())

        def _est_rows(code: str) -> int:
            start = pd.to_datetime(download_start_by_code[code])
            return max(1, int((end_dt - start).days * 0.8))

        bulk_codes: list[str] = []
        small_codes: list[str] = []
        for code in download_codes:
            if _est_rows(code) * bulk_batch <= 4000:
                bulk_codes.append(code)
            else:
                small_codes.append(code)
        sub_batches: list[list[str]] = []
        for i in range(0, len(bulk_codes), bulk_batch):
            sub_batches.append(bulk_codes[i : i + bulk_batch])
        for i in range(0, len(small_codes), batch_size):
            sub_batches.append(small_codes[i : i + batch_size])

        def _download_sub_batch(sub_batch: list[str]) -> tuple[list[str], DownloadResult]:
            sub_start = min(download_start_by_code[code] for code in sub_batch)
            try:
                result = self.downloader.download_stocks_daily(sub_batch, sub_start, end_date)
            except Exception as exc:
                result = DownloadResult(False, None, "", str(exc))
            return sub_batch, result

        def _resolve(sub_batch: list[str], result: DownloadResult) -> tuple[list[dict], dict[str, pd.DataFrame], dict[str, str]]:
            """Validate one downloaded sub-batch into status rows + frames to persist."""
            batch_rows: list[dict] = []
            frames: dict[str, pd.DataFrame] = {}
            sources: dict[str, str] = {}
            frame = result.data if result.ok and result.data is not None else pd.DataFrame()
            grouped = frame.groupby("code") if "code" in frame.columns else None
            source = result.source if result.ok else ""
            for code in sub_batch:
                old_df = old_frames[code]
                download_start_dt = pd.to_datetime(download_start_by_code[code])
                group = grouped.get_group(code) if grouped is not None and code in grouped.groups else pd.DataFrame()
                if not group.empty:
                    group = group[group["date"] >= download_start_dt].copy()

                if group.empty:
                    old_last_date = None if old_df.empty else pd.to_datetime(old_df["date"]).max()
                    if old_last_date is not None and old_last_date >= download_start_dt - pd.Timedelta(days=1):
                        batch_rows.append(self._status_row(code, old_last_date, len(old_df), "local_cache", "success", ""))
                    else:
                        batch_rows.append(self._status_row(code, None, len(old_df), source, "failed", f"no data returned for {code}"))
                    continue

                new_df = DataValidator.clean_stock_daily(group, code)
                new_valid = DataValidator.validate_stock_daily(new_df, min_rows=1, require_min_rows=False)
                if not new_valid.ok:
                    old_last_date = None if old_df.empty else pd.to_datetime(old_df["date"]).max()
                    batch_rows.append(self._status_row(code, old_last_date, len(old_df), source, "failed", new_valid.error))
                    continue

                merged = pd.concat([old_df, new_df], ignore_index=True) if not old_df.empty else new_df
                merged = DataValidator.clean_stock_daily(merged, code)
                # Newly listed stocks legitimately have fewer than the normal
                # model warm-up rows. Structural/price validation remains
                # strict, but history length is advisory for daily ingestion.
                merged_valid = DataValidator.validate_stock_daily(merged, require_min_rows=False)
                if not merged_valid.ok:
                    old_last_date = None if old_df.empty else pd.to_datetime(old_df["date"]).max()
                    batch_rows.append(self._status_row(code, old_last_date, len(old_df), source, "failed", merged_valid.error))
                    continue

                frames[code] = new_df
                sources[code] = source or "unknown"
                last_date = pd.to_datetime(merged["date"]).max()
                batch_rows.append(
                    self._status_row(
                        code,
                        last_date,
                        len(merged),
                        source,
                        "success",
                        self._short_history_warning(len(merged), min_rows_for_valid),
                    )
                )
            return batch_rows, frames, sources

        processed = len(rows)  # skip rows already counted
        if self.workers <= 1 or len(sub_batches) == 1:
            for sub_batch in sub_batches:
                _, result = _download_sub_batch(sub_batch)
                batch_rows, frames, sources = _resolve(sub_batch, result)
                rows.extend(batch_rows)
                if frames:
                    self.cache.save_stocks_daily(frames, source=sources, incremental=True)
                processed += len(sub_batch)
                if batch_cb is not None:
                    batch_cb(batch_rows, processed)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_batch = {executor.submit(_download_sub_batch, sb): sb for sb in sub_batches}
                for future in as_completed(future_to_batch):
                    sub_batch, result = future.result()
                    batch_rows, frames, sources = _resolve(sub_batch, result)
                    rows.extend(batch_rows)
                    if frames:
                        self.cache.save_stocks_daily(frames, source=sources, incremental=True)
                    processed += len(sub_batch)
                    if batch_cb is not None:
                        batch_cb(batch_rows, processed)

        log_rows = []
        for row in rows:
            code = row["code"]
            if code in skip_codes:
                log_rows.append({**row, "time": now, "action": "skip", "start_date": "", "end_date": end_date})
            else:
                log_rows.append(
                    {**row, "time": now, "action": "update", "start_date": download_start_by_code.get(code, ""), "end_date": end_date}
                )
        self.cache.update_status_many(rows)
        self.cache.append_log_many(log_rows)
        return pd.DataFrame(rows)


    def _update_pool_batch(self, stock_list: list[str], end_date: str | None = None) -> pd.DataFrame:
        total = len(stock_list)
        rows: list[dict] = [None] * total
        code_to_idx = {code: idx for idx, code in enumerate(stock_list)}
        progress_bar = None
        if self.use_progress_bar and tqdm is not None and sys.stdout.isatty():
            progress_bar = tqdm(
                total=total,
                desc="Updating daily bars (batch)",
                unit="stock",
                ncols=80,
                file=sys.stdout,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )

        def _on_batch(batch_rows: list[dict], processed: int) -> None:
            if progress_bar is not None:
                progress_bar.update(len(batch_rows))
                last = batch_rows[-1] if batch_rows else {}
                status = str(last.get("status", "")).lower()
                progress_bar.set_postfix_str(f"{'ok' if status == 'success' else 'fail' if status == 'failed' else 'skip'} {last.get('code', '')}")
            if self.progress_callback is not None:
                for row in batch_rows:
                    try:
                        self.progress_callback({
                            "current": processed,
                            "total": total,
                            "row": row,
                        })
                    except Exception:
                        logger.warning(
                            "Progress callback failed for %s (%s/%s)",
                            row.get("code", ""), processed, total, exc_info=True,
                        )

        report = self.update_batch(stock_list, end_date=end_date, batch_cb=_on_batch)
        for _, row in report.iterrows():
            code = str(row.get("code", ""))
            idx = code_to_idx.get(code)
            if idx is not None:
                rows[idx] = row.to_dict()

        if progress_bar is not None:
            progress_bar.close()
        return pd.DataFrame([r for r in rows if r is not None])

    def update_pool(
        self,
        mode: str | None = None,
        stocks: list[str] | None = None,
        end_date: str | None = None,
        max_stocks: int | None = None,
    ) -> pd.DataFrame:
        pool = StockPoolManager(self.config).build_pool(mode=mode, stocks=stocks)
        if max_stocks and max_stocks > 0:
            pool = pool.head(max_stocks).copy()
        total = len(pool)
        if total == 0:
            return pd.DataFrame()

        stock_list = pool["stock"].tolist()
        use_batch = self.config.get("update", {}).get("use_batch", True)
        if use_batch and total > 1:
            return self._update_pool_batch(stock_list, end_date=end_date)

        sleep_seconds = float(self.config.get("update", {}).get("sleep_seconds", 0.8))
        rows: list[dict] = [None] * total
        completed_count = 0
        progress_bar = None
        if self.use_progress_bar and tqdm is not None and sys.stdout.isatty():
            progress_bar = tqdm(
                total=total,
                desc="Updating daily bars",
                unit="stock",
                ncols=80,
                file=sys.stdout,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )

        def _notify(row: dict) -> None:
            nonlocal completed_count
            completed_count += 1
            if progress_bar is not None:
                status = str(row.get("status", "")).lower()
                if status == "success":
                    progress_bar.set_postfix_str(f"ok {row.get('code', '')}")
                elif status == "failed":
                    progress_bar.set_postfix_str(f"fail {row.get('code', '')}")
                progress_bar.update(1)
            if self.progress_callback is not None:
                try:
                    self.progress_callback({
                        "current": completed_count,
                        "total": total,
                        "row": row,
                        "rows": [r for r in rows if r is not None],
                    })
                except Exception:
                    logger.warning(
                        "Progress callback failed for %s (%s/%s)",
                        row.get("code", ""), completed_count, total, exc_info=True,
                    )

        def _update_single(idx_code: tuple[int, str]) -> tuple[int, dict]:
            idx, code = idx_code
            # Per-stock sleep remains effective for the serial path and as a
            # small throttle when workers > 1.
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return idx, self.update_one(code, end_date=end_date)

        if self.workers <= 1:
            for idx, code in enumerate(stock_list):
                _, row = _update_single((idx, code))
                rows[idx] = row
                _notify(row)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_idx = {
                    executor.submit(_update_single, (idx, code)): idx
                    for idx, code in enumerate(stock_list)
                }
                for future in as_completed(future_to_idx):
                    idx, row = future.result()
                    rows[idx] = row
                    _notify(row)

        if progress_bar is not None:
            progress_bar.close()
        return pd.DataFrame([r for r in rows if r is not None])

    @staticmethod
    def _short_history_warning(rows: int, minimum: int) -> str:
        if rows >= minimum:
            return ""
        return f"short history: {rows} < {minimum}; stored but not warm-up eligible"

    @staticmethod
    def _status_row(code: str, last_date, rows: int, source: str, status: str, error_msg: str) -> dict:
        if pd.isna(last_date):
            last_date_text = ""
        elif last_date is None:
            last_date_text = ""
        else:
            last_date_text = pd.to_datetime(last_date).strftime("%Y-%m-%d")
        return {
            "code": code,
            "last_date": last_date_text,
            "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows": rows,
            "source": source,
            "status": status,
            "error_msg": error_msg,
        }
