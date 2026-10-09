#!/usr/bin/env python3
"""Append descriptive tail, duration, risk and funding evidence to a complete run.

This reader never changes training/evaluation reports, ledgers or qualification.
Old six-arm files are followed through their immutable reference manifests.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now

OLD_ARMS=("rules","ma_only","structure_only","fusion","rules_exit","fusion_exit")
MAIN_ARMS=(*OLD_ARMS,"wyckoff_rule","wyckoff_only","three_fusion","fusion_wyckoff_exit","three_fusion_exit","policy_fresh","union_rules","policy_union")


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""):digest.update(part)
    return digest.hexdigest()


def read_csv(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def close(left,right,message):
    if not math.isclose(float(left),float(right),rel_tol=1e-9,abs_tol=1e-6):raise ValueError(message)


def tail(values,mass=.05):
    if not values:return {"q05":None,"cvar05":None,"n":0}
    x=np.sort(np.asarray(values,dtype=float));weight=len(x)*mass;whole=int(math.floor(weight));fraction=weight-whole
    worst=float(x[:whole].sum())+(float(x[whole])*fraction if fraction and whole<len(x) else 0.)
    return {"q05":float(np.quantile(x,mass)),"cvar05":worst/weight,"n":len(x)}


def episodes_and_state(ledger,initial_cash,calendar,terminal_date):
    inventory=0;cash=float(initial_cash);basis=0.;paid=received=0.;entry=None;closed=[];day_states={};fees=notional=0.
    calendar_index={day:i for i,day in enumerate(calendar)}
    for row in ledger.to_dict("records"):
        day=str(row["date"])[:10];action=row["action"];shares=int(row["shares"])
        if day not in calendar_index or shares<=0:raise ValueError("invalid dated ledger quantity")
        close(row["position_before"],inventory,"ledger opening inventory mismatch")
        flow=float(row["cash_flow"]);fee=float(row["fee"]);fill=float(row["price"])
        if action=="BUY":
            close(flow,-fill*shares-fee,"BUY debit/fee mismatch")
            if inventory==0:entry=day;paid=received=0.
            basis+=-flow;inventory+=shares;paid+=-flow
        elif action in {"SELL","REDUCE"}:
            close(flow,fill*shares-fee,"SELL credit/fee mismatch")
            if shares>inventory:raise ValueError("sale exceeds inventory")
            basis-=basis/inventory*shares;inventory-=shares;received+=flow
            if inventory==0:
                closed.append({"net_return":received/paid-1.,"net_pnl":received-paid,
                    "entry_date":entry,"exit_date":day,"holding_sessions":calendar_index[day]-calendar_index[entry]})
                basis=0.;entry=None;paid=received=0.
        else:raise ValueError(f"unexpected ledger action {action!r} on {day}")
        cash+=flow;fees+=fee;notional+=fill*shares
        close(row["position_after"],inventory,"ledger closing inventory mismatch")
        close(row["cash_after"],cash,"ledger cash mismatch")
        if "holding_cost_after" in row:close(row["holding_cost_after"],basis/inventory if inventory else 0.,"ledger holding-cost mismatch")
        day_states[day]={"cash":cash,"shares":inventory,"cost_basis":basis}
    opened=None
    if inventory:
        opened={"entry_date":entry,"shares":inventory,"net_buy_debits":paid,"partial_sell_credits":received,
            "remaining_cost_basis":basis,"holding_sessions_to_window_end":calendar_index[terminal_date]-calendar_index[entry]+1}
    return closed,opened,day_states,fees,notional


def account_metrics(ledger,daily,calendar,initial_cash):
    if daily.empty:raise ValueError("source account has no daily evidence")
    daily=daily.copy();daily["date"]=pd.to_datetime(daily.date).dt.strftime("%Y-%m-%d")
    if daily.date.duplicated().any() or daily.date.tolist()!=sorted(daily.date.tolist()):raise ValueError("daily source ordering invalid")
    if not set(daily.date)<=set(calendar):raise ValueError("daily source escaped frozen window")
    closed,opened,states,fees,notional=episodes_and_state(ledger,initial_cash,calendar,calendar[-1])
    current={"cash":float(initial_cash),"shares":0,"cost_basis":0.};last_close=None
    cost_by_date={}
    for row in daily.to_dict("records"):
        day=row["date"]
        if day in states:current=states[day]
        close(row["cash"],current["cash"],"daily cash does not match actual ledger")
        close(row["shares"],current["shares"],"daily shares do not match actual ledger")
        market=float(row["shares"])*float(row["close"])
        close(row["market_value"],market,"daily marked value mismatch")
        close(row["equity"],float(row["cash"])+market,"daily equity mismatch")
        cost_by_date[day]=current["cost_basis"];last_close=float(row["close"])
    aligned=daily.set_index("date").reindex(calendar).ffill()
    for key,default in (("cash",initial_cash),("equity",initial_cash),("market_value",0.),("shares",0.)):
        aligned[key]=aligned[key].fillna(default)
    basis=pd.Series(cost_by_date).reindex(calendar).ffill().fillna(0.)
    if opened:
        opened.update(last_observed_date=str(daily.iloc[-1].date),last_known_market_value=float(aligned.iloc[-1].market_value),
            marked_net_pnl=float(opened["partial_sell_credits"]+aligned.iloc[-1].market_value-opened["net_buy_debits"]),
            future_liquidation_cost_included=False)
    return {"closed":closed,"open":opened,"fees":fees,"notional":notional,
        "equity":aligned.equity.to_numpy(dtype=float),"market_value":aligned.market_value.to_numpy(dtype=float),
        "cash":aligned.cash.to_numpy(dtype=float),"cost_basis":basis.to_numpy(dtype=float),
        "terminal_net_pnl":float(aligned.iloc[-1].equity-initial_cash)}


def _task(payload):
    run_root,fold,arm,symbols,calendar,initial_cash=payload
    root=Path(run_root);source_bindings=[];paths={};old=arm in OLD_ARMS
    if old:
        references=json.loads((root/"reports"/(fold["name"]+"-old-six-ledger-references.json")).read_text())[arm]
        for row in references["bindings"]:
            if row["kind"] in {"ledger","daily"}:paths.setdefault(row["symbol"],{})[row["kind"]]=(Path(row["path"]),row["sha256"])
    else:
        for symbol in symbols:
            directory=root/"evaluation"/fold["name"]/arm/symbol.replace(".","_")
            if directory.exists():paths[symbol]={key:(directory/(key+".csv"),None) for key in ("ledger","daily")}
    total_equity=np.full(len(calendar),len(symbols)*initial_cash,dtype=float);total_market=np.zeros(len(calendar));total_cash=total_equity.copy();total_basis=np.zeros(len(calendar))
    closed=[];opened=[];fees=notional=0.;account_pnls=[];source_rows=0
    for symbol in symbols:
        if symbol not in paths:continue
        frames={}
        for kind,(path,expected_sha) in paths[symbol].items():
            digest=file_sha(path)
            if expected_sha is not None and digest!=expected_sha:raise ValueError("immutable reference CSV hash changed")
            frame=read_csv(path);frames[kind]=frame;source_rows+=len(frame)
            source_bindings.append({"fold":fold["name"],"arm":arm,"symbol":symbol,"kind":kind,"path":str(path),"sha256":digest,"rows":len(frame)})
        values=account_metrics(frames["ledger"],frames["daily"],calendar,initial_cash)
        total_equity+=values["equity"]-initial_cash;total_cash+=values["cash"]-initial_cash;total_market+=values["market_value"];total_basis+=values["cost_basis"]
        closed.extend({"symbol":symbol,**x} for x in values["closed"])
        if values["open"]:opened.append({"symbol":symbol,**values["open"]})
        fees+=values["fees"];notional+=values["notional"];account_pnls.append((symbol,values["terminal_net_pnl"]))
    capital=len(symbols)*initial_cash;previous=np.r_[capital,total_equity[:-1]];daily_return=total_equity/previous-1.
    peak=np.maximum.accumulate(np.r_[capital,total_equity])[1:];drawdown=total_equity/peak-1.
    exposure=np.divide(total_market,total_equity,out=np.zeros_like(total_market),where=total_equity!=0)
    aggregate_path=root/"evaluation"/fold["name"]/arm/"aggregate_daily.csv"
    aggregate=read_csv(aggregate_path);agg=dict(zip(pd.to_datetime(aggregate.date).dt.strftime("%Y-%m-%d"),aggregate.equity))
    for day,value in zip(calendar,total_equity):
        if day not in agg:raise ValueError("frozen aggregate missing verified session")
        close(agg[day],value,"independent portfolio daily reaggregation mismatch")
    source_bindings.append({"fold":fold["name"],"arm":arm,"kind":"aggregate_daily","path":str(aggregate_path),"sha256":file_sha(aggregate_path),"rows":len(aggregate)})
    returns=[x["net_return"] for x in closed];durations=[x["holding_sessions"] for x in closed];abs_pnls=np.sort(np.abs([x[1] for x in account_pnls]));total_abs=float(abs_pnls.sum())
    result={"fold":fold["name"],"arm":arm,"qualification":"descriptive_observed_development_only","frozen_accounts":len(symbols),
        "accounts_with_source_daily":len(paths),"unavailable_accounts_cash_retained":len(symbols)-len(paths),"initial_cash":capital,
        "net_return":float(total_equity[-1]/capital-1.),"maximum_drawdown":float(-drawdown.min()),"annualized_daily_volatility":float(np.std(daily_return)*np.sqrt(252)),
        "annualized_downside_deviation":float(np.sqrt(np.mean(np.minimum(daily_return,0.)**2))*np.sqrt(252)),"worst_daily_return":float(daily_return.min()),
        "completed_round_trips":len(closed),"completed_round_trip_tail":tail(returns),"daily_return_tail":tail(daily_return.tolist()),
        "mean_closed_holding_sessions":float(np.mean(durations)) if durations else None,"median_closed_holding_sessions":float(np.median(durations)) if durations else None,
        "max_closed_holding_sessions":max(durations) if durations else None,"unclosed_episodes":len(opened),"open_episodes":opened,
        "mean_open_holding_sessions_to_window_end":float(np.mean([x["holding_sessions_to_window_end"] for x in opened])) if opened else None,
        "actual_fill_gross_notional":float(notional),"actual_fees":float(fees),"gross_two_sided_turnover":float(notional/total_equity.mean()),
        "annualized_gross_turnover":float(notional/total_equity.mean()*252/len(calendar)),"mean_market_exposure_fraction":float(exposure.mean()),
        "maximum_market_exposure_fraction":float(exposure.max()),"mean_fee_inclusive_capital_at_cost_fraction":float(total_basis.mean()/capital),
        "maximum_fee_inclusive_capital_at_cost_fraction":float(total_basis.max()/capital),"fee_inclusive_capital_CNY_sessions":float(total_basis.sum()),
        "largest_five_absolute_account_pnl_share":float(abs_pnls[-5:].sum()/total_abs) if total_abs else None,
        "source_rows_read":source_rows,"verified_window_sessions":len(calendar),"source_files_sha256":stable_hash(source_bindings)}
    return result,source_bindings


def summarize(run_id,*,cpu_workers=4,output_prefix="supplemental-metrics"):
    root=artifact_run_dir(run_id,create=False);report_path=root/"reports/wyckoff-research.json";report=json.loads(report_path.read_text())
    if report.get("status")!="complete" or report.get("model_family")!="wyckoff":raise ValueError("supplement requires a complete Wyckoff run")
    report_hash=file_sha(report_path);output=root/"reports"/(output_prefix+".json");source_path=root/"reports"/(output_prefix+"-source-files.jsonl");readable=root/"reports"/(output_prefix+".md")
    if any(path.exists() for path in (output,source_path,readable)):raise FileExistsError("supplement is append-only; choose a new explicit prefix")
    symbols=[x["symbol"] for x in report["dataset_records"] if x["symbol"].startswith("60") and x["symbol"].endswith(".SH") or x["symbol"].startswith("00") and x["symbol"].endswith(".SZ")]
    aux=report["config"].get("auxiliary_controls",["matched_dual_fixed_candidates"]);tasks=[]
    for fold in report["config"]["development_windows"]:
        days=[x for x in report["market_dates"] if fold["test_start"]<=x<=fold["test_end"]]
        for arm in (*MAIN_ARMS,*aux):tasks.append((str(root),fold,arm,symbols,days,report["config"]["initial_cash"]))
    metrics=[];bindings=[]
    if cpu_workers>1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            for result,sources in pool.map(_task,tasks,chunksize=1):metrics.append(result);bindings.extend(sources);print("descriptive supplemental metrics",len(metrics),"/",len(tasks),flush=True)
    else:
        for task in tasks:result,sources=_task(task);metrics.append(result);bindings.extend(sources)
    if file_sha(report_path)!=report_hash:raise ValueError("training report changed during supplemental read")
    value={"run_id":run_id,"created_at":local_now().isoformat(),"qualification":"descriptive_only_no_promotion_or_new_bootstrap_claim",
        "training_report_path":str(report_path),"training_report_sha256":report_hash,"source_files":len(bindings),"source_rows_read":sum(x.get("rows",0) for x in bindings),
        "source_manifest_path":str(source_path),"source_manifest_canonical_sha256":stable_hash(bindings),"actual_cpu_tasks":len(tasks),"cpu_workers":cpu_workers,
        "model_training_batches":0,"model_inference_rows":0,"metrics":metrics,
        "definitions":{"round_trip_tail":"complete actual inventory round trips only; net SELL/REDUCE credits / fee-inclusive BUY debits - 1",
            "cvar05":"lowest five percent empirical observation mass, fractional final observation; small n is descriptive only",
            "holding_sessions":"verified market session index at final SELL open minus initial BUY open; open age includes last window session",
            "unclosed":"partial SELL credits plus last known market value shown separately; future liquidation fee absent",
            "turnover":"sum actual filled price times shares for both BUY and SELL / average aggregate equity",
            "exposure":"end-of-day gross actual market value / aggregate equity, includes failed accounts retained as cash",
            "funding":"remaining fee-inclusive BUY cost basis / initial capital; FIFO-independent average-cost reconstruction",
            "daily_calendar":"verified frozen sessions; missing account dates forward-fill last known holdings and equity; pre-start/missing accounts cash",
            "scope":"independent fixed-budget accounts, not shared cash; all four historical windows observed; quarterly 160/240-block qualification remains unavailable"}}
    with source_path.open("x",encoding="utf-8") as stream:
        for row in bindings:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    value["source_manifest_file_sha256"]=file_sha(source_path)
    with output.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    lines=["# Wyckoff 实际账本补充指标","","仅对已观察开发窗口作描述；未平仓不计入完整往返尾部，季度日期区块不足仍不认证。","",f"来源完整训练运行：`{run_id}`。逐文件SHA清单：`{source_path.name}`。","","|窗口|组别|往返/未平仓|净收益%|回撤%|往返Q5%|往返CVaR5%|平均持仓交易日|双边换手|平均暴露%|平均成本资本占用%|","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(v,scale=1.):return "不可用" if v is None else f"{v*scale:.3f}"
    for item in metrics:
        t=item["completed_round_trip_tail"];lines.append(f"|{item['fold']}|{item['arm']}|{item['completed_round_trips']}/{item['unclosed_episodes']}|{fmt(item['net_return'],100)}|{fmt(item['maximum_drawdown'],100)}|{fmt(t['q05'],100)}|{fmt(t['cvar05'],100)}|{fmt(item['mean_closed_holding_sessions'])}|{fmt(item['gross_two_sided_turnover'])}|{fmt(item['mean_market_exposure_fraction'],100)}|{fmt(item['mean_fee_inclusive_capital_at_cost_fraction'],100)}|")
    with readable.open("x",encoding="utf-8") as stream:stream.write("\n".join(lines)+"\n")
    return {"output":str(output),"readable":str(readable),"metrics":len(metrics),"source_files":len(bindings),"source_manifest_canonical_sha256":value["source_manifest_canonical_sha256"],"source_manifest_file_sha256":value["source_manifest_file_sha256"]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run-id",required=True);parser.add_argument("--cpu-workers",type=int,default=4);parser.add_argument("--output-prefix",default="supplemental-metrics");args=parser.parse_args()
    if args.cpu_workers<1:raise ValueError("cpu workers must be positive")
    if not args.output_prefix.replace("-","").replace("_","").isalnum():raise ValueError("safe single output prefix required")
    print(json.dumps(summarize(args.run_id,cpu_workers=args.cpu_workers,output_prefix=args.output_prefix),ensure_ascii=False))

if __name__=="__main__":main()
