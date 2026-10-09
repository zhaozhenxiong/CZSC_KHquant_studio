"""CZSC-only persistent jobs with cancellation and exact run results."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

import pandas as pd
import psutil

from my_strategy.core.paths import METADATA_ROOT, PROJECT_ROOT
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.tz import local_now
from my_strategy.storage.czsc_results import ResultStore

ACTIVE = ("queued", "running", "cancelling")


class Cancelled(RuntimeError):
    pass


class TaskManager:
    def __init__(self, db_path: Path | None = None, results: ResultStore | None = None, workers: int = 2) -> None:
        self.db_path = db_path or METADATA_ROOT / "czsc_tasks.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.results = results or ResultStore()
        self._lock = threading.RLock()
        self._cancel: dict[str, threading.Event] = {}
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="czsc-job")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS czsc_tasks (job_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL)")
            conn.commit()
        with closing(sqlite3.connect(self.db_path)) as conn:
            active = conn.execute("SELECT payload FROM czsc_tasks WHERE status IN ('queued','running','cancelling')").fetchall()
        for payload, in active:
            job = json.loads(payload)
            try:
                owner = psutil.Process(job.get("owner_pid", -1))
                live = abs(owner.create_time() - job.get("owner_started", 0)) < .001
            except (psutil.Error, ValueError):
                live = False
            if not live:
                self._patch(job["job_id"], status="failed", error="服务重启中断了任务，请重新提交。")

    def _write(self, job: dict[str, Any]) -> None:
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            payload = {key: value for key, value in job.items() if key != "result"}
            conn.execute("INSERT INTO czsc_tasks VALUES (?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,payload=excluded.payload", (job["job_id"], stable_hash({"kind": job["kind"], "spec": job["spec"]}), job["status"], json.dumps(payload, ensure_ascii=False, allow_nan=False)))
            conn.commit()

    def get(self, job_id: str) -> dict[str, Any]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute("SELECT payload FROM czsc_tasks WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        job = json.loads(row[0])
        job["result"] = self.results.get(job["run_id"]) if job["status"] == "succeeded" else None
        return job

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute("SELECT payload FROM czsc_tasks ORDER BY rowid DESC LIMIT ?", (min(max(limit, 1), 200),)).fetchall()
        return [{**json.loads(row[0]), "result": None} for row in rows]

    def _patch(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            job.update(updates)
            job["updated_at"] = local_now().isoformat()
            self._write(job)
            return job

    def submit(self, kind: str, spec: dict[str, Any]) -> dict[str, Any]:
        if kind not in {"scan", "backtest", "update", "research_train", "dual_research_train", "wyckoff_research_train"}:
            raise ValueError("仅支持结构扫描、CZSC回测和行情更新")
        with self._lock:
            if self._closed:
                raise RuntimeError("任务服务正在关闭")
            fingerprint = stable_hash({"kind": kind, "spec": spec})
            with closing(sqlite3.connect(self.db_path)) as conn:
                row = conn.execute("SELECT payload FROM czsc_tasks WHERE fingerprint=? AND status IN ('queued','running','cancelling')", (fingerprint,)).fetchone()
            if row:
                return json.loads(row[0])
            job_id = uuid.uuid4().hex
            job = dict(job_id=job_id, kind=kind, spec=spec, status="queued", progress=0, progress_detail={"stage": "等待", "current": 0, "total": 0, "failed": 0}, error=None, result=None, run_id=None, created_at=local_now().isoformat())
            job.update(owner_pid=os.getpid(), owner_started=psutil.Process().create_time())
            self._write(job)
            self._cancel[job_id] = threading.Event()
            self._executor.submit(self._execute, job_id)
            return job

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            if job["status"] not in ACTIVE:
                return job
            event = self._cancel.get(job_id)
            if event:
                event.set()
            return self._patch(job_id, status="cancelling")

    def _check(self, job_id: str) -> None:
        if self._cancel[job_id].is_set() or self.get(job_id)["status"] == "cancelling":
            raise Cancelled("用户取消任务")

    def _execute(self, job_id: str) -> None:
        try:
            self._check(job_id)
            self._patch(job_id, status="running")
            job = self.get(job_id)
            result = self._run(job_id, job["kind"], job["spec"])
            self._check(job_id)
            self.results.save(job["kind"], result)
            updates = {"status": "succeeded", "progress": 100, "result": result, "run_id": result["run_id"]}
            if "compute_info" in result:
                updates["compute_info"] = result["compute_info"]
            self._patch(job_id, **updates)
        except Cancelled as exc:
            job = self._patch(job_id, status="cancelled", error=str(exc))
            self._finish_research_run(job)
        except Exception as exc:
            job = self._patch(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            self._finish_research_run(job)
        finally:
            with self._lock:
                self._cancel.pop(job_id, None)

    def _finish_research_run(self, job: dict[str, Any]) -> None:
        if not job.get("run_id") or not (job["kind"] in {"research_train", "dual_research_train", "wyckoff_research_train"} or job["spec"].get("research")):
            return
        path = (self.results.runs_root / job["run_id"] / "metadata.json").resolve()
        if not path.is_relative_to(self.results.runs_root.resolve()):
            raise ValueError("Research metadata outside run storage")
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata.update(status=job["status"], error=job["error"], finished_at=job["updated_at"])
            temporary = path.with_suffix(f".{job['job_id']}.tmp")
            temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)

    def _run(self, job_id: str, kind: str, spec: dict[str, Any]) -> dict[str, Any]:
        from my_strategy.adapters.czsc_adapter import latest_market_date, list_symbols
        from my_strategy.services.czsc_analysis import strategy_config
        if kind == "update":
            return self._update(job_id, spec)
        if kind == "wyckoff_research_train":
            from my_strategy.services.czsc_wyckoff_research import train_wyckoff_research
            def wyckoff_progress(stage, current, total, failed):
                self._check(job_id)
                self._patch(job_id, progress_detail={"stage": stage, "current": current, "total": total, "failed": failed})
            source = spec.get("source_training_run_id")
            if not source:
                raise ValueError("威科夫研究需要明确的冻结来源运行")
            return train_wyckoff_research(source_training_run_id=source, end=spec.get("end"),
                symbols=spec.get("symbols") or None, device=spec.get("device") or "auto",
                cpu_workers=spec.get("cpu_workers") or 4, progress=wyckoff_progress,
                check_cancel=lambda: self._check(job_id), run_callback=lambda run_id: self._patch(job_id, run_id=run_id))
        if kind == "dual_research_train":
            from my_strategy.services.czsc_dual_research import train_dual_research
            def dual_progress(stage, current, total, failed):
                self._check(job_id)
                self._patch(job_id, progress_detail={"stage": stage, "current": current, "total": total, "failed": failed})
            source = spec.get("source_training_run_id")
            if not source:
                raise ValueError("双专家研究需要明确的冻结来源运行")
            return train_dual_research(source_training_run_id=source, end=spec.get("end"),
                symbols=spec.get("symbols") or None, device=spec.get("device") or "auto",
                cpu_workers=spec.get("cpu_workers") or 4, progress=dual_progress,
                check_cancel=lambda: self._check(job_id), run_callback=lambda run_id: self._patch(job_id, run_id=run_id))
        if kind == "research_train" or spec.get("research"):
            from my_strategy.services.czsc_research import train_research, scan_research, backtest_research, research_status
            last_notice, last_progress, last_stage = 0.0, 0.0, None
            def research_progress(stage, current, total, failed):
                nonlocal last_notice, last_progress, last_stage
                self._check(job_id)
                if time.monotonic() - last_notice < .25 and current != total and stage == last_stage:
                    return
                last_notice = time.monotonic()
                last_stage = stage
                updates = {"progress_detail": {"stage": stage, "current": current, "total": total, "failed": failed}}
                if kind == "scan":
                    last_progress = max(last_progress, min(95, current / total * 95 if total else 0))
                    updates["progress"] = last_progress
                    if stage.startswith("CPU") or stage == "多角色证据扫描":
                        updates["progress_detail"]["device"] = "cpu"
                    elif stage.startswith("GPU"):
                        updates["progress_detail"]["device"] = capability["selected_device"]
                self._patch(job_id, **updates)
            from my_strategy.services.czsc_compute import compute_status
            capability = compute_status(spec.get("device"))
            if not capability["available"]:
                raise RuntimeError(capability["fallback_reason"] or "研究设备不可用")
            common = {"end": spec.get("end") or latest_market_date(), "symbols": spec.get("symbols") or [item["symbol"] for item in list_symbols()],
                      "device": spec.get("device"), "progress": research_progress, "check_cancel": lambda: self._check(job_id),
                      "run_callback": lambda run_id: self._patch(job_id, run_id=run_id)}
            if kind == "research_train":
                calendar_id = spec.get("calendar_run_id") or research_status()["calendar_run_id"]
                if not calendar_id:
                    raise ValueError("尚无已核验交易日历，先完成研究数据核验")
                return train_research(**common, calendar_run_id=calendar_id, cpu_workers=spec.get("cpu_workers") or 4)
            routed = {key: spec.get(key) for key in ("model_run_id", "model_fold", "calendar_run_id")}
            routed.update(usage_mode=spec.get("usage_mode", "historical"), model_policy=spec.get("model_policy", "auto"),
                          model_family=spec.get("model_family", "ma_trend"),
                          cpu_workers=spec.get("cpu_workers"), batch_size=spec.get("batch_size"),
                          entry_policy=spec.get("entry_policy", "legacy"))
            if kind == "scan":
                from my_strategy.storage.personal_portfolio import PersonalStore
                held = {item["symbol"] for item in PersonalStore().holdings()["items"]}
                return scan_research(**common, held_symbols=held, start=spec.get("start"), **routed)
            return backtest_research(**common, start=spec["start"], initial_cash=spec.get("initial_cash", 100000), **routed)
        symbols = spec.get("symbols") or [row["symbol"] for row in list_symbols()]
        if not symbols:
            raise ValueError("本地行情没有有效股票")
        latest = latest_market_date(spec.get("end"))
        end = spec.get("end") or latest
        scan_date = latest if kind == "scan" else end
        if not scan_date:
            raise ValueError("请求截止日之前没有市场行情")
        settings = strategy_config()
        context = create_run_context(task=f"czsc-{kind}", as_of_date=end, config={"request": spec, "strategy": settings}, data_version=str(latest), scope="full" if len(symbols) > 1 else "single", source=spec.get("source", "web"), stocks=symbols, start_date=spec.get("start"), end_date=end)
        self._patch(job_id, run_id=context.run_id)
        from my_strategy.services.czsc_batch import iter_replay_batches
        failures, rows, accounts, compute_batches = [], [], [], []
        last_progress, last_notice = 0.0, 0.0
        def preparation_progress(completed, total, failed):
            nonlocal last_progress, last_notice
            self._check(job_id)
            if time.monotonic() - last_notice < .25 and completed != total:
                return
            last_notice = time.monotonic()
            last_progress = max(last_progress, completed / total * 70 if total else 0)
            self._patch(job_id, progress=last_progress, progress_detail={"stage": "CPU并行结构预处理", "current": completed, "total": total, "failed": failed})
        for batch in iter_replay_batches(symbols, end=scan_date, config=settings, device=spec.get("device"),
                                        require_latest=kind == "scan", check_cancel=lambda: self._check(job_id),
                                        progress=preparation_progress, cpu_workers=spec.get("cpu_workers"), batch_size=spec.get("batch_size")):
            self._check(job_id)
            failures.extend(batch["failures"])
            compute_batches.append(batch["compute_info"])
            for prepared, session in zip(batch["prepared"], batch["sessions"], strict=True):
                self._check(job_id)
                symbol, frame = prepared.symbol, prepared.frame
                try:
                    if kind == "scan":
                        data_end = pd.Timestamp(frame.iloc[-1]["date"]).date().isoformat()
                        if data_end != scan_date:
                            raise ValueError(f"行情缺少扫描日 {scan_date}，标的最后日期为 {data_end}")
                        if spec.get("start") and spec["start"] > data_end:
                            raise ValueError("起始日期晚于可用行情截止日")
                        current = [event for event in session.events if str(event.get("available_at", event.get("time", "")))[:10] == data_end]
                        event = current[-1] if current else {}
                        warnings = ["周/月线在下一周期首次观测后才确认闭合，末桶不用于规则。", "事件为研究意图，真实成交以回测账本为准。"]
                        if frame.attrs.get("legacy_fuyao_bars"):
                            warnings.append("包含旧fuyao来源，复权口径未核验，相关成交拒单。")
                        rows.append({"symbol": symbol, "name": "", "signal": event.get("signal", "无已确认事件"), "action": event.get("action", "watch"), "available_at": event.get("available_at", data_end), "reason": event.get("reason", "当前交易日未触发规则"), "data_end": data_end, "data_version": frame.attrs["data_version"], "warnings": warnings})
                    else:
                        from my_strategy.services.czsc_backtest import backtest_stock
                        account = backtest_stock(symbol, start=spec["start"], end=end, initial_cash=float(spec.get("initial_cash", 100000)) / len(symbols), config=settings, run_context=context,
                                                 prepared=prepared, session=session, compute_info=batch["compute_info"])
                        accounts.append(account)
                except (ValueError, FileNotFoundError) as exc:
                    failures.append({"symbol": symbol, "error": str(exc)})
            completed = len(rows if kind == "scan" else accounts) + len(failures)
            last_progress = max(last_progress, completed / len(symbols) * 95)
            self._patch(job_id, progress=last_progress, compute_info=aggregate_compute(compute_batches), progress_detail={"stage": "批量信号与规则事件" if kind == "scan" else "实际成交回测", "current": completed, "total": len(symbols), "failed": len(failures)})
        self._check(job_id)
        self._patch(job_id, progress_detail={"stage": "保存结果", "current": len(symbols), "total": len(symbols), "failed": len(failures)})
        if len(failures) == len(symbols):
            raise ValueError(f"全部 {len(symbols)} 个标的失败：{failures[:3]}")
        input_versions = {item["symbol"]: item["data_version"] for item in (rows if kind == "scan" else accounts)}
        compute_info = aggregate_compute(compute_batches)
        context.write_metadata({"config": {"request": spec, "strategy": settings}, "input_versions": input_versions, "failures": failures, "compute_info": compute_info})
        common = {"run_id": context.run_id, "run_context": context.to_dict(), "data_version": stable_hash(input_versions), "input_versions": input_versions, "strategy_version": settings["strategy_version"], "source_sha256": settings["source_sha256"], "failures": failures, "coverage": {"requested": len(symbols), "success": len(symbols) - len(failures), "failed": len(failures)}, "data_range": {"start": spec.get("start"), "end": end}, "symbols": symbols, "compute_info": compute_info}
        if kind == "scan":
            return {**common, "rows": rows}
        return {**common, **aggregate_accounts(accounts, float(spec.get("initial_cash", 100000)), len(symbols)), "symbol": symbols[0] if len(symbols) == 1 else "", "allocation": "固定等额分仓；无跨股票资金再分配", "accounts": [{"symbol": account.get("symbol"), "run_id": account.get("run_id"), "metrics": account.get("metrics"), "data_range": account.get("data_range"), "data_version": account.get("data_version"), "config_hash": account.get("config_hash"), "artifacts": account.get("artifacts")} for account in accounts]}

    def _update(self, job_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        from my_strategy.adapters.czsc_adapter import data_status
        context = create_run_context(task="czsc-data-update", config=spec, scope="daily_update", source="web", as_of_date=spec.get("end"))
        log_path = context.subdir("logs") / "data-update.log"
        command = [sys.executable, "-u", str(PROJECT_ROOT / "my_strategy" / "scripts" / "update_daily_data.py"), "--mode", "local", "--update-id", context.run_id, "--json", "--no-progress-bar"]
        if spec.get("symbols"):
            command.extend(["--stocks", ",".join(spec["symbols"])])
        if spec.get("end"):
            command.extend(["--end-date", spec["end"]])
        self._patch(job_id, run_id=context.run_id, progress_detail={"stage": "更新行情", "current": 0, "total": 0, "failed": 0})
        offset, pending, failed = 0, "", 0
        with log_path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=output, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                while process.poll() is None:
                    if self.get(job_id)["status"] == "cancelling":
                        self._cancel[job_id].set()
                    if self._cancel[job_id].wait(.25):
                        process.terminate()
                        process.wait(timeout=15)
                        raise Cancelled("用户取消行情更新")
                    with log_path.open(encoding="utf-8", errors="replace") as progress_log:
                        progress_log.seek(offset)
                        pending += progress_log.read()
                        offset = progress_log.tell()
                    lines = pending.split("\n")
                    pending = lines.pop()
                    for line in lines:
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if event.get("event") == "progress":
                            current, total = event.get("current", 0), event.get("total", 0)
                            failed += int(event.get("row", {}).get("status") in {"failed", "error"})
                            self._patch(job_id, progress=current / total * 95 if total else 0, progress_detail={"stage": "更新行情", "current": current, "total": total, "failed": failed})
                if process.returncode:
                    raise RuntimeError(log_path.read_text(encoding="utf-8", errors="replace")[-3000:])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        status = data_status()
        log = log_path.read_text(encoding="utf-8", errors="replace")
        marker = log.rfind('{\n  "summary":')
        update_result = json.loads(log[marker:]) if marker >= 0 else {}
        return {"run_id": context.run_id, "run_context": context.to_dict(), "data_status": status, "update_summary": update_result.get("summary"), "log": log[-6000:]}

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for event in self._cancel.values():
                event.set()
        self._executor.shutdown(wait=True, cancel_futures=False)


def aggregate_compute(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep capability metadata separate from the measured work of each batch."""
    if not batches:
        return {"cuda_work": False, "mps_work": False, "gpu_work": False, "batches": 0}
    info = dict(batches[0])
    devices = list(dict.fromkeys(batch.get("selected_device") for batch in batches))
    actual_devices = list(dict.fromkeys(batch.get("actual_device", batch.get("selected_device")) for batch in batches))
    backends = list(dict.fromkeys(batch.get("backend") for batch in batches))
    reasons = list(dict.fromkeys(batch["fallback_reason"] for batch in batches if batch.get("fallback_reason")))
    info.update(batches=len(batches), cuda_work=any(batch.get("cuda_work", False) for batch in batches),
                mps_work=any(batch.get("mps_work", False) for batch in batches),
                gpu_work=any(batch.get("gpu_work", batch.get("cuda_work", False)) for batch in batches),
                actual_devices=actual_devices, actual_device=actual_devices[0] if len(actual_devices) == 1 else "mixed",
                batch_size=max(batch.get("batch_size", 0) for batch in batches), selected_devices=devices,
                selected_device=devices[0] if len(devices) == 1 else "mixed",
                backend=backends[0] if len(backends) == 1 else "mixed", fallback_reason="；".join(reasons) or None)
    for field in ("cuda_batches", "bars", "signal_elements", "tensor_seconds", "position_seconds", "preparation_wall_seconds"):
        info[field] = sum(batch.get(field) or 0 for batch in batches)
    info["tensor_operations"] = list(dict.fromkeys(operation for batch in batches for operation in batch.get("tensor_operations", [])))
    elapsed = [batch["cuda_elapsed_ms"] for batch in batches if batch.get("cuda_elapsed_ms") is not None and batch.get("cuda_work")]
    info["cuda_elapsed_ms"] = sum(elapsed) if elapsed else None
    info["stages"] = [{**stage, "device": info["actual_device"] if stage.get("stage") == "batched_signals" else stage.get("device")} for stage in info.get("stages", [])]
    info["batch_records"] = [{key: batch.get(key) for key in ("selected_device", "actual_device", "backend", "cuda_work", "mps_work", "gpu_work", "batch_size", "bars", "cuda_elapsed_ms", "fallback_reason")} for batch in batches]
    return info


def aggregate_accounts(accounts: list[dict[str, Any]], initial_cash: float, requested: int) -> dict[str, Any]:
    """Mark equal-budget accounts on a common date index; failed accounts stay cash."""
    series = []
    trades, rejections, limitations = [], [], []
    for account in accounts:
        frame = pd.DataFrame(account["daily"])
        if frame.empty:
            raise ValueError("回测未产生每日账本")
        date_key = "date" if "date" in frame else "dt"
        series.append(pd.Series(frame["equity"].to_numpy(), index=pd.to_datetime(frame[date_key])).sort_index())
        trades.extend(account.get("trades", []))
        rejections.extend(account.get("rejections", []))
        limitations.extend(account.get("limitations", []))
    daily = pd.concat(series, axis=1).sort_index().ffill().fillna(initial_cash / requested)
    equity = daily.sum(axis=1) + (requested - len(accounts)) * initial_cash / requested
    peak = equity.cummax().clip(lower=initial_cash)
    drawdown = equity / peak - 1
    net_return = float(equity.iloc[-1] / initial_cash - 1)
    values = [{"date": date.strftime("%Y-%m-%d"), "equity": float(value), "drawdown": float(drawdown.loc[date])} for date, value in equity.items()]
    return {"initial_cash": initial_cash, "daily": values, "trades": sorted(trades, key=lambda row: (row["date"], row["symbol"])), "rejections": rejections, "limitations": list(dict.fromkeys(limitations)), "metrics": {"net_return": net_return, "total_return": net_return, "max_drawdown": float(-drawdown.min()), "trades": len(trades), "trade_count": len(trades), "final_equity": float(equity.iloc[-1]), "fees": sum(float(row.get("fee", 0)) for row in trades)}}
