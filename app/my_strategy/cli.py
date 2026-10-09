"""Portable CLI for the CZSC-only research workbench."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from my_strategy.core.paths import PROJECT_ROOT
from my_strategy.runtime_env import configure_utf8_stdio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KHQuant CZSC 结构研究")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "scan", "backtest", "backtest-batch"):
        item = commands.add_parser(name)
        item.add_argument("--symbol", help="单股代码，例如 600519.SH")
        item.add_argument("--stocks", default="", help="逗号分隔代码；扫描/批量留空表示本地全部股票")
        item.add_argument("--start", default="2026-01-01", help="开始日期，默认2026-01-01；保留更早行情供预热")
        item.add_argument("--end", help="截止日期，默认最新本地行情日期")
        item.add_argument("--as-of", help="分析回放截止日或Asia/Shanghai时间")
        item.add_argument("--initial-cash", type=float, default=100000)
        item.add_argument("--device", help="计算设备：auto、cpu、mps、cuda 或 cuda:编号")
        item.add_argument("--research", action="store_true", default=True, help="统一缠论量价ML研究（默认）")
        item.add_argument("--model-family", choices=("ma_trend", "dual", "wyckoff"), default="ma_trend", help="均线单模型、双专家或威科夫三专家影子包")
        item.add_argument("--structure-only", dest="research", action="store_false", help="原生结构规则对照")
        item.add_argument("--usage-mode", choices=("production", "historical", "retrospective"), default="historical")
        item.add_argument("--model-policy", choices=("auto", "pinned"), default="auto")
        item.add_argument("--entry-policy", choices=("legacy", "fresh", "risk"), default="legacy",
                          help="入场契约：原目标仓位对照、有效买点研究、价格风险实验")
        item.add_argument("--model-run-id", help="研究模型运行；未认证时保持影子")
        item.add_argument("--model-fold", help="固定检查点名称")
        item.add_argument("--calendar-run-id", help="核验交易日历运行")
        if name != "analyze":
            item.add_argument("--cpu-workers", type=int, help="只读结构准备进程数（1到16，默认4）")
            item.add_argument("--batch-size", type=int, help="GPU 微批大小（1到128，默认32）")
        item.add_argument("--json", action="store_true")
    research = commands.add_parser("research-train")
    research.add_argument("--stocks", default="")
    research.add_argument("--end", required=True)
    research.add_argument("--calendar-run-id")
    research.add_argument("--device", default="auto")
    research.add_argument("--cpu-workers", type=int, default=4)
    research.add_argument("--json", action="store_true")
    dual = commands.add_parser("research-dual-train", help="双专家融合与实际持仓退出完整研究")
    dual.set_defaults(model_family="dual")
    dual.add_argument("--source-training-run-id", required=True, help="包含全列冻结特征与10日标签的已完成研究运行")
    dual.add_argument("--stocks", default="")
    dual.add_argument("--end", required=True)
    dual.add_argument("--device", default="auto")
    dual.add_argument("--cpu-workers", type=int, default=4)
    dual.add_argument("--json", action="store_true")
    wyckoff = commands.add_parser("research-wyckoff-train", help="威科夫量价三专家与真实持仓买卖契约研究")
    wyckoff.set_defaults(model_family="wyckoff")
    wyckoff.add_argument("--source-training-run-id", required=True)
    wyckoff.add_argument("--stocks", default="")
    wyckoff.add_argument("--end", required=True)
    wyckoff.add_argument("--device", default="auto")
    wyckoff.add_argument("--cpu-workers", type=int, default=4)
    wyckoff.add_argument("--json", action="store_true")
    for name in ("research-promote", "research-rollback", "research-status"):
        item = commands.add_parser(name)
        if name == "research-promote":
            item.add_argument("--model-run-id", required=True)
            item.add_argument("--model-fold", default="production")
        if name == "research-rollback":
            item.add_argument("--release-id", help="恢复已发布版本；留空则撤回活动模型")
        if name != "research-status":
            item.add_argument("--reason", required=True)
        else:
            item.add_argument("--model-family", choices=["ma_trend", "dual", "wyckoff"], default="ma_trend")
        item.add_argument("--json", action="store_true")
    for name in ("dashboard", "update-data", "package", "doctor"):
        item = commands.add_parser(name, add_help=False)
        item.add_argument("args", nargs=argparse.REMAINDER)
    for name in ("verify-db",):
        item = commands.add_parser(name)
        item.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    delegates = {"dashboard": ["-m", "my_strategy.web_dashboard.scripts.serve_dashboard"], "update-data": ["-m", "my_strategy.scripts.update_daily_data"], "package": ["-m", "my_strategy.scripts.package_project"], "doctor": ["-m", "my_strategy.scripts.docker_runtime_doctor"]}
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in delegates:
        return subprocess.call([sys.executable, *delegates[values[0]], *values[1:]], cwd=PROJECT_ROOT,
                               env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
    args = build_parser().parse_args(values)
    try:
        from my_strategy.adapters.czsc_adapter import data_status
        if args.command in {"doctor", "verify-db"}:
            import czsc
            status = data_status()
            print(json.dumps({"status": "ok", "czsc": czsc.__version__, "data": status}, ensure_ascii=False))
            return 0
        from my_strategy.web_dashboard.api import AnalysisRequest, TaskRequest
        from my_strategy.storage.czsc_results import ResultStore
        if args.command in {"research-promote", "research-rollback", "research-status"}:
            from my_strategy.storage.czsc_model_releases import ModelReleaseStore
            from my_strategy.services.czsc_research import research_status
            store = ModelReleaseStore()
            if args.command == "research-status":
                if args.model_family == "wyckoff":
                    from my_strategy.services.czsc_wyckoff_runtime import wyckoff_research_status
                    result = wyckoff_research_status()
                elif args.model_family == "dual":
                    from my_strategy.services.czsc_dual_runtime import dual_research_status
                    result = dual_research_status()
                else:
                    result = research_status()
            elif args.command == "research-promote":
                result = store.promote(args.model_run_id, args.model_fold, reason=args.reason)
            else:
                result = store.rollback(args.release_id, reason=args.reason)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0
        if args.command == "analyze":
            from my_strategy.services.czsc_analysis import analyze_stock
            from my_strategy.core.run_context import create_run_context
            if not args.symbol:
                raise ValueError("analyze 须指定 --symbol")
            spec = AnalysisRequest(symbol=args.symbol, start=args.start, end=args.end, as_of=args.as_of,
                                   model_family=args.model_family,
                                   research=args.research, usage_mode=args.usage_mode, model_policy=args.model_policy,
                                   entry_policy=args.entry_policy,
                                   model_run_id=args.model_run_id, model_fold=args.model_fold,
                                   calendar_run_id=args.calendar_run_id, device=args.device).model_dump(mode="json", exclude_none=True)
            result = analyze_stock(**spec)
            context = create_run_context(task="czsc-analysis", as_of_date=result["data_end"], config=spec, data_version=str(result.get("data_version", result["data_end"])), stocks=[args.symbol], source="cli", start_date=args.start, end_date=result["data_end"])
            result.update(run_id=context.run_id, run_context=context.to_dict())
            ResultStore().save("analysis", result)
        else:
            from my_strategy.web_dashboard.tasks import TaskManager
            stocks = ([args.symbol] if getattr(args, "symbol", None) else []) + [stock.strip() for stock in args.stocks.split(",") if stock.strip()]
            task_spec = {"symbols": stocks, "start": getattr(args, "start", None), "end": args.end, "initial_cash": getattr(args, "initial_cash", 100000)}
            task_spec.update({key: getattr(args, key) for key in ("device", "cpu_workers", "batch_size") if getattr(args, key, None) is not None})
            task_spec.update({key: getattr(args, key) for key in ("research", "model_family", "source_training_run_id", "model_run_id", "model_fold", "calendar_run_id", "usage_mode", "model_policy", "entry_policy") if getattr(args, key, None) is not None})
            if args.command == "research-train" and not task_spec.get("calendar_run_id"):
                from my_strategy.services.czsc_research import research_status
                task_spec["calendar_run_id"] = research_status()["calendar_run_id"]
            request = TaskRequest(kind="wyckoff_research_train" if args.command == "research-wyckoff-train" else "dual_research_train" if args.command == "research-dual-train" else "research_train" if args.command == "research-train" else "scan" if args.command == "scan" else "backtest", spec=task_spec)
            manager = TaskManager(workers=1)
            try:
                spec = request.spec.model_dump(mode="json", exclude_none=True)
                spec["source"] = "cli"
                job = manager.submit(request.kind, spec)
                last = None
                while job["status"] in {"queued", "running", "cancelling"}:
                    detail = job["progress_detail"]
                    current = (detail.get("stage"), detail.get("current"))
                    if current != last:
                        print(f'{detail.get("stage")} {detail.get("current", 0)}/{detail.get("total", 0)} failures={detail.get("failed", 0)}', file=sys.stderr)
                        last = current
                    time.sleep(.2)
                    job = manager.get(job["job_id"])
                if job["status"] != "succeeded":
                    raise RuntimeError(job["error"])
                result = job["result"]
            except KeyboardInterrupt:
                if "job" in locals():
                    manager.cancel(job["job_id"])
                return 130
            finally:
                manager.close()
        print(json.dumps(result if args.json else {key: result[key] for key in ("run_id", "symbol", "data_end", "coverage", "metrics") if key in result}, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
