"""Fourteen frozen-account Wyckoff diagnostic arms; never a release API."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from my_strategy.core.config_loader import load_config
from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions, _write_frame
from my_strategy.services.czsc_dual_models import DualPredictorSession, verified_bundle as verified_dual_bundle, oof_splits
from my_strategy.services.czsc_dual_research import prepare_frozen_dataset, _raw, _mainboard, round_trip_statistics
from my_strategy.services.czsc_research import _file_hash, _json, _save
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services.czsc_research_decisions import replay_decisions
from my_strategy.services.czsc_research_ml import date_block_bootstrap
from my_strategy.services.czsc_wyckoff_features import prepare_wyckoff_inputs, W_INPUT_COLUMNS, get_wyckoff_profile
from my_strategy.services.czsc_wyckoff_models import (WyckoffPredictorSession, train_wyckoff_bundle,
    verified_bundle, attach_heads, get_bundle_profile)
from my_strategy.web_dashboard.tasks import aggregate_accounts

OLD_ARMS = ("rules", "ma_only", "structure_only", "fusion", "rules_exit", "fusion_exit")
NEW_ARMS = ("wyckoff_rule", "wyckoff_only", "three_fusion", "fusion_wyckoff_exit", "three_fusion_exit", "policy_fresh", "union_rules", "policy_union")
ALL_ARMS = (*OLD_ARMS, *NEW_ARMS)


def _prepare_record(payload):
    record, destination, cfg = payload
    if _file_hash(Path(record["path"])) != record["sha256"]: raise ValueError("source bytes changed during W preparation")
    frame = pd.read_parquet(record["path"])
    features = prepare_wyckoff_inputs(frame)
    if features.attrs.get("feature_version") != frame.attrs.get("feature_version") or features.attrs.get("schema_hash") != frame.attrs.get("schema_hash"):
        raise ValueError("W preparation must preserve old MA frame/cache identity")
    for key in ("date", "symbol", "label", "label_end", "label_available", "net_return"):
        if not features[key].equals(frame[key]): raise ValueError("W enrichment changed frozen source/y10 column " + key)
    path = Path(destination) / (record["symbol"].replace(".", "_") + ".parquet")
    features.to_parquet(path, index=False)
    projected = {**record, "path": str(path.resolve()), "sha256": _file_hash(path), "source_path": record["path"],
        "source_sha256": record["sha256"], "source_data_version": record["data_version"],
        "wyckoff_schema_hash": get_wyckoff_profile()["schema_hash"], "snapshot_mode": "immutable_new_wyckoff_projection"}
    sample = None
    if _mainboard(record["symbol"]):
        mask = (pd.to_datetime(features.date) >= pd.Timestamp(cfg["feature_start"])) & features.label_available.astype(bool)
        # Raw columns retained here permit independent input and execution audit.
        from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS, STRUCTURE_COLUMNS
        metadata = [key for key in ("date", "symbol", "available_at", "structure_confirmed_at", "zone_confirmed_at", "weekly_available_at", "monthly_available_at", "input_eligible", "reason_codes", "model_input_eligible", "model_reason_codes", "label", "label_end", "label_available", "net_return", "wyckoff_input_eligible", "wyckoff_reason_codes", "wyckoff_available_at") if key in features]
        sample = features.loc[mask, list(dict.fromkeys([*MA_TREND_COLUMNS,*STRUCTURE_COLUMNS,*W_INPUT_COLUMNS,*metadata]))].copy()
    return projected, sample, {"symbol": record["symbol"], "bars": len(features),
        "eligible_rows": int(features.wyckoff_input_eligible.sum()),
        "reason_counts": dict(Counter(reason for values in features.wyckoff_reason_codes for reason in values))}


def prepare_wyckoff_dataset(source_training_run_id, context, config, *, symbols=None, end=None,
                            cpu_workers=4, progress=None, check_cancel=None):
    old_data, source_records, snapshot, dates, calendar = prepare_frozen_dataset(source_training_run_id, context, config,
        symbols=symbols, end=end, progress=progress, check_cancel=check_cancel)
    del old_data
    directory = context.subdir("dataset", "wyckoff_features")
    tasks = [(record, str(directory), config) for record in source_records]
    check=check_cancel or (lambda:None); records, samples, quality=[],[],[]
    def collect(result,index):
        check(); record,sample,diagnostic=result; records.append(record); quality.append(diagnostic)
        if sample is not None and len(sample): samples.append(sample)
        if progress: progress("逐日威科夫事件与32项量价准备",index+1,len(tasks),0)
    if cpu_workers>1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            for i,result in enumerate(pool.map(_prepare_record,tasks,chunksize=1)): collect(result,i)
    else:
        for i,task in enumerate(tasks): collect(_prepare_record(task),i)
    if not samples: raise ValueError("no mature mainboard W research observations")
    data=pd.concat(samples,ignore_index=True).sort_values(["date","symbol"]).reset_index(drop=True)
    source=json.loads(Path(snapshot["source_report_path"]).read_text())
    data.attrs.update(data_version=stable_hash({"source_snapshot":snapshot["source_report_sha256"],
        "projection_hashes":{r["symbol"]:r["sha256"] for r in records},"profile":get_bundle_profile()}),
        label_version="czsc_fixed_horizon_broker_v1",label_contract=source["model_manifest"]["label_contract"],
        source_training_run_id=source_training_run_id,source_snapshot_hash=snapshot["source_report_sha256"])
    data.to_parquet(context.subdir("dataset")/"training.parquet",index=False)
    calendar["file_sha256"]=_file_hash(Path(calendar["path"]))
    _save(context.subdir("reports")/"wyckoff-input-quality.json",{"rows":quality,"formal_qualification_blocked":True,
        "unresolved":["corporate_action_effective_time_data_missing","historical_ST_and_delisted_pool_missing"],
        "unknown_critical_dates_preserved":True,"source_target_unchanged":True})
    return data,records,source_records,snapshot,dates,calendar


def _read_record(record,end=None):
    if _file_hash(Path(record["path"]))!=record["sha256"]: raise ValueError("new W feature source hash mismatch")
    frame=pd.read_parquet(record["path"])
    return frame if end is None else frame.loc[pd.to_datetime(frame.date)<=pd.Timestamp(end)].copy()


def bounded_replay_frame(frame,start):
    """Keep saturated real execution seasoning; structured inputs stay full-prefix.

    The original Position is explicitly flat before start. Its start and all
    forward bars are retained, with at least 60 actual source-qualified traded
    bars beforehand. No structural or Wyckoff feature is recomputed on this tail.
    """
    from my_strategy.services.czsc_backtest import legacy_fuyao_source
    after=np.flatnonzero((pd.to_datetime(frame.date)>=pd.Timestamp(start)).to_numpy())
    if not len(after):return frame,0
    begin=int(after[0]);prior=frame.iloc[:begin]
    valid=(prior.volume.gt(0)&prior.amount.gt(0)&prior.has_trade_price.eq(1)&~prior.source.map(legacy_fuyao_source)).to_numpy()
    seasoned=np.flatnonzero(valid);required=int(strategy_config()["execution"]["seasoned_bars"])
    offset=int(seasoned[-required]) if len(seasoned)>=required else 0
    return frame.iloc[offset:].copy(),offset


def frozen_candidate_filter(base_decisions, probabilities, thresholds, *, apply_mask=True):
    """Veto only pre-frozen original candidates; never generate new Position plans."""
    count=len(base_decisions);values=np.broadcast_to(np.asarray(probabilities,dtype=float),(count,))
    limits=np.broadcast_to(np.asarray(thresholds,dtype=float),(count,));applied=np.broadcast_to(np.asarray(apply_mask,dtype=bool),(count,))
    if not np.isfinite(limits).all() or ((limits<0)|(limits>1)).any(): raise ValueError("invalid frozen candidate threshold")
    output=[]
    for index,original in enumerate(base_decisions):
        row=copy.deepcopy(original);plan=row.get("entry_plan")
        candidate=isinstance(plan,dict) and plan.get("status")=="active" and plan.get("entry_allowed") is True
        probability=float(values[index]) if np.isfinite(values[index]) else None
        row.update(probability=probability,threshold=float(limits[index]),ml_filter_applied=bool(applied[index]),candidate_filter_contract="frozen_unfiltered_fresh_active_plans_v2")
        if candidate and applied[index] and (probability is None or probability<limits[index]):
            row["entry_plan"]={**plan,"status":"model_veto","entry_allowed":False,"reason_codes":["frozen_candidate_model_veto"]}
            row.update(entry_gate_passed=False,candidate_model_veto=True,model_gate_decision="VETO")
        output.append(row)
    return output


def _threshold_record(payload):
    record,fold,probabilities,grid,cash,dates,destination=payload
    frame=_read_record(record,fold["validation_end"]);frame,offset=bounded_replay_frame(frame,fold["validation_start"]);probabilities={name:values[offset:] for name,values in probabilities.items()}; raw=_raw(frame); settings=strategy_config()
    if not (pd.to_datetime(frame.date)>=pd.Timestamp(fold["validation_start"])).any(): return {"cash_reason":"no_validation_bars","symbol":record["symbol"],"scores":{}}
    base=replay_decisions(frame,raw,entry_policy="fresh",position_start=fold["validation_start"],config=settings)["decisions"]
    candidates=sum(bool(d.get("entry_plan") and d["entry_plan"].get("status")=="active" and d["entry_plan"].get("entry_allowed")) for d in base if d["date"]>=fold["validation_start"])
    result={}
    for expert,values in probabilities.items():
        result[expert]={}
        for threshold in grid:
            decisions=frozen_candidate_filter(base,values,threshold)
            account=execute_decisions(record["symbol"],raw,decisions,fold["validation_start"],cash,settings,
                "wyckoff-validation-threshold",market_dates=dates,verified_sources=VERIFIED_SOURCES,entry_policy="fresh")
            folder=Path(destination)/expert/str(threshold)/record["symbol"].replace(".","_");folder.mkdir(parents=True,exist_ok=True)
            for key in ("ledger","daily","rejections"): _write_frame(pd.DataFrame(account[key]),folder/(key+".csv"))
            accepted=sum(x["action"]=="BUY" for x in account["ledger"])
            result[expert][str(threshold)]={"account":account,"candidates":candidates,"accepted":accepted}
    return {"symbol":record["symbol"],"scores":result}


def validation_threshold_evaluator(records,context,cfg,dates,*,cpu_workers,progress,check_cancel):
    selected=[r for r in records if _mainboard(r["symbol"])];check=check_cancel or (lambda:None)
    def evaluate(predict_validation,fold):
        tasks=[];grid=cfg["threshold_grid"]
        for i,record in enumerate(selected):
            check();frame=_read_record(record,fold["validation_end"])
            values={name:np.full(len(frame),np.nan) for name in ("wyckoff","fusion")}
            mask=pd.to_datetime(frame.date)>=pd.Timestamp(fold["validation_start"])
            if mask.any():
                result=predict_validation(frame.loc[mask].copy())
                for name in values: values[name][mask.to_numpy()]=result[name]
            tasks.append((record,fold,values,grid,cfg["initial_cash"],dates,str(context.subdir("validation",fold["name"]))))
            if progress: progress("验证段固定候选实际概率",i+1,len(selected),0)
        gathered={name:{str(t):[] for t in grid} for name in ("wyckoff","fusion")};cash=[]
        def collect(result,i):
            check()
            if not result["scores"]: cash.append(result);return
            for name,by_threshold in result["scores"].items():
                for t,item in by_threshold.items():gathered[name][t].append(item)
            if progress:progress("验证阈值完整Broker账本",i+1,len(tasks),0)
        if cpu_workers>1:
            with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context("spawn")) as pool:
                for i,result in enumerate(pool.map(_threshold_record,tasks,chunksize=1)):collect(result,i)
        else:
            for i,task in enumerate(tasks):collect(_threshold_record(task),i)
        scores={}
        for name,by_threshold in gathered.items():
            scores[name]=[]
            for threshold,items in by_threshold.items():
                if not items:
                    scores[name].append({"threshold":float(threshold),"completed_round_trips":0,"candidate_coverage":0.,"net_return":0.,"max_drawdown":0.,"evidence":"complete_validation_broker_ledger"});continue
                accounts=[x["account"] for x in items];total=aggregate_accounts(accounts,len(selected)*cfg["initial_cash"],len(selected))
                metrics={**total["metrics"],**round_trip_statistics([a["ledger"] for a in accounts])}
                candidates=sum(x["candidates"] for x in items);accepted=sum(x["accepted"] for x in items)
                score={"threshold":float(threshold),"evidence":"complete_validation_broker_ledger",
                    **metrics,"fixed_candidate_count":candidates,"actual_buy_count":accepted,"candidate_coverage":accepted/candidates if candidates else 0.,
                    "failed_accounts_cash_retained":cash,"initial_cash":len(selected)*cfg["initial_cash"]}
                scores[name].append(score)
                _write_frame(pd.DataFrame(total["daily"]),context.subdir("validation",fold["name"],name,threshold)/"aggregate_daily.csv")
        _save(context.subdir("reports")/(fold["name"]+"-validation-thresholds.json"),scores)
        return scores
    return evaluate

def _forward_behavior(frame, bundles, start, predictor):
    """Continuous entry state from only checkpoints available by each close."""
    raw=_raw(frame);dates=pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d");values=np.full(len(frame),np.nan)
    thresholds=np.full(len(frame),.55);lineages=[{} for _ in range(len(frame))]
    ordered=sorted(bundles,key=lambda x:x["available_at"])
    for index,item in enumerate(ordered):
        upper=ordered[index+1]["available_at"] if index+1<len(ordered) else "9999-12-31"
        mask=(dates>item["available_at"])&(dates<=upper)
        if not mask.any():continue
        bundle=verified_bundle(item["path"])
        p=predictor.predict(frame.loc[mask].copy(),item["path"],as_of=str(dates[mask].max()))
        selected=bundle["thresholds"]["fusion"]["selected"]
        values[mask.to_numpy()]=p if selected else 0.
        thresholds[mask.to_numpy()]=bundle["probability_threshold"]
        for offset in np.flatnonzero(mask.to_numpy()):lineages[offset]={"available_at":item["available_at"],"label_cutoff":bundle["validation_end"],
            "bundle_sha256":bundle["bundle_sha256"],"checkpoint":item["name"],"retrospective":False}
    # Before first checkpoint, rules remain the frozen reference behavior.
    applied=np.isfinite(values)
    base=replay_decisions(frame,raw,mode="rules",entry_policy="fresh",position_start=start,config=strategy_config())["decisions"]
    replay={"decisions":frozen_candidate_filter(base,values,thresholds,apply_mask=applied)}
    for i,row in enumerate(replay["decisions"]):row["behavior_model_lineage"]=lineages[i]
    return replay["decisions"]


def _exit_state_record(payload):
    record,cfg,dates,destination,source_hash,bundles=payload
    from my_strategy.services.czsc_wyckoff_exit import build_exit_dataset,make_union_decisions
    frame=_read_record(record);frame,_=bounded_replay_frame(frame,cfg["feature_start"]);raw=_raw(frame);settings=strategy_config()
    base=replay_decisions(frame,raw,entry_policy="fresh",position_start=cfg["feature_start"],config=settings)["decisions"]
    union=make_union_decisions(frame,base,config=settings,candidate_policy="fresh_wyckoff_union_v1")
    outputs={}
    def callback(policy):
        def save(account):
            folder=Path(destination)/policy/record["symbol"].replace(".","_");folder.mkdir(parents=True,exist_ok=True)
            for key in ("ledger","daily","rejections","holding_snapshots"):_write_frame(pd.DataFrame(account.get(key,[])),folder/(key+".csv"))
        return save
    union_data=build_exit_dataset(features=frame,raw=raw,base_decisions=union,market_dates=dates,start=cfg["feature_start"],
        initial_cash=cfg["initial_cash"],config=settings,entry_policy="risk",candidate_policy="fresh_wyckoff_union_v1",
        behavior_policy="union_rules_risk_v1",source_snapshot_hash=source_hash,reference_callback=callback("union"))
    if len(union_data):outputs["union"]=union_data
    # New own-policy holdings are generated by strict chronological checkpoints,
    # not a final fitted model. They are pooled with original rule states under
    # the same frozen native continuation target and separately identified.
    predictor=WyckoffPredictorSession(device="cpu")
    forward=_forward_behavior(frame,bundles,cfg["feature_start"],predictor)
    fresh_data=build_exit_dataset(features=frame,raw=raw,base_decisions=forward,market_dates=dates,start=cfg["feature_start"],
        initial_cash=cfg["initial_cash"],config=settings,entry_policy="fresh",candidate_policy="fresh_v1",
        behavior_policy="rules_and_forward_three_fusion_fresh_v1",row_behavior_policy="forward_three_fusion_with_early_rules_fallback_v1",
        source_snapshot_hash=source_hash,reference_callback=callback("forward_three_fusion"))
    if len(fresh_data):outputs["fresh"]=fresh_data
    return outputs,predictor.diagnostics


def prepare_exit_training(records,context,cfg,dates,source_hash,baseline_report,bundles,*,cpu_workers,progress,check_cancel):
    from my_strategy.services.czsc_wyckoff_exit import bind_existing_exit_dataset
    selected=[r for r in records if _mainboard(r["symbol"])];selected_symbols={r["symbol"] for r in selected}
    baseline_root=Path(baseline_report["run_context"]["run_id"])
    old_path=artifact_run_dir(str(baseline_root),create=False)/"dataset/exit-training.parquet"
    original=pd.read_parquet(old_path);old_hash=_file_hash(old_path)
    original=original.loc[original.symbol.isin(selected_symbols)].copy()
    # Join the market inputs to actual audited snapshots without recreating them.
    market=[]
    for record in selected:
        frame=_read_record(record)
        wanted=pd.to_datetime(original.loc[original.symbol.eq(record["symbol"]),"date"]).dt.strftime("%Y-%m-%d")
        part=frame.loc[pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d").isin(wanted)].copy()
        market.append(part)
    bound=bind_existing_exit_dataset(original,pd.concat(market,ignore_index=True),config=strategy_config(),market_dates=dates,
        source_snapshot_hash=source_hash,source_dataset_sha256=old_hash,reference_run_id=baseline_report["run_id"],
        behavior_policy="rules_and_forward_three_fusion_fresh_v1")
    bound["behavior_policy"]="immutable_audited_rules_fresh_v1"
    parts={"fresh":[bound],"union":[]};diagnostics=[];check=check_cancel or (lambda:None)
    tasks=[(r,cfg,dates,str(context.subdir("evaluation","exit_reference")),source_hash,bundles) for r in selected]
    def collect(result,i):
        check();outputs,compute=result;diagnostics.append(compute)
        for policy,data in outputs.items():parts[policy].append(data)
        if progress:progress("严格前向自身持仓与候选扩展退出标签",i+1,len(tasks),0)
    if cpu_workers>1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            for i,result in enumerate(pool.map(_exit_state_record,tasks,chunksize=1)):collect(result,i)
    else:
        for i,task in enumerate(tasks):collect(_exit_state_record(task),i)
    output={}
    for policy,frames in parts.items():
        if not frames:output[policy]=pd.DataFrame();continue
        contracts={stable_hash(x.attrs["label_contract"]) for x in frames}
        if len(contracts)!=1:raise ValueError("pooled actual-state exit targets differ")
        data=pd.concat(frames,ignore_index=True).sort_values(["date","symbol"]).reset_index(drop=True);data.attrs=dict(frames[0].attrs)
        data.attrs.update(data_version=stable_hash({"source":source_hash,"policy":policy,"contract":data.attrs["label_contract"],
            "sources":{r["symbol"]:r["sha256"] for r in selected}}),actual_held_samples=len(data),
            label_reason_counts=dict(Counter(data.label_reason)),state_behavior_coverage=data.groupby("behavior_policy").size().to_dict())
        data.to_parquet(context.subdir("dataset")/(policy+"-exit-training.parquet"),index=False);output[policy]=data
    _save(context.subdir("reports")/"exit-state-coverage.json",{"reference_source_path":str(old_path),"reference_source_sha256":old_hash,
        "strict_forward_behavior_compute":{"rows":sum(x["rows"] for x in diagnostics),"batches":sum(x["batches"] for x in diagnostics),
            "device":"cpu","actual_mps_inference":False},
        "policies":{k:{"rows":len(v),"mature":int(v.label_available.sum()) if len(v) else 0,
            "behavior_coverage":v.attrs.get("state_behavior_coverage",{})} for k,v in output.items()}})
    return output


def _train_exit_heads(datasets,context,cfg,*,device,progress,check_cancel):
    from my_strategy.services.czsc_wyckoff_exit import train_exit_model
    catalog={"fresh":[],"union":[]};manifests=[];check=check_cancel or (lambda:None)
    splits=oof_splits(cfg["feature_start"],cfg["production"]["train_end"])
    for split in splits:
        for policy,data in datasets.items():
            check();directory=context.subdir("models","exit-oof",split["name"],policy)
            if data.empty:continue
            try:
                manifest=train_exit_model(data,directory,split["train_end"],split["validation_start"],split["validation_end"],
                    device=device,seed=cfg["seed"],epochs=cfg["epochs"],batch_size=cfg["batch_size"],
                    min_train_rows=cfg["min_train_rows"],min_validation_rows=cfg["min_validation_rows"])
            except ValueError as exc:
                if not any(t in str(exc) for t in ("insufficient mature samples","each require both label classes")):raise
                _save(directory/"unavailable.json",{"reason":str(exc),"split":split,"policy":policy});continue
            manifests.append(manifest);catalog[policy].append({"model_dir":str(directory),"available_at":manifest["available_at"],
                "contract_hash":manifest["label_contract"]["contract_hash"],"threshold":cfg["exit_threshold"],"manifest_sha256":manifest["manifest_sha256"],
                "label_cutoff":manifest["validation_end"],"candidate_policy":manifest["label_contract"]["candidate_policy"],
                "entry_policy":manifest["label_contract"]["entry_policy"],"checkpoint_sha256":manifest["checkpoint_sha256"]})
            if progress:progress("前向退出政策冻结 "+split["name"]+" "+policy,len(manifests),len(splits)*2,0)
    identity={policy:stable_hash({"version":"wyckoff_forward_exit_router_v1","models":items,"fallback":"native_rules_when_unavailable"}) for policy,items in catalog.items()}
    _save(context.subdir("reports")/"frozen-exit-routers.json",{"catalog":catalog,"identity":identity})
    return catalog,identity,manifests


def _exit_router(features,catalog,*,apply=True):
    from my_strategy.services.czsc_wyckoff_exit import ExitPredictor,make_exit_policy
    predictors={}
    def resolve(day):
        allowed=[x for x in catalog if x["available_at"]<=day and x["label_cutoff"]<day]
        if not allowed:return None
        binding=allowed[-1];key=binding["manifest_sha256"]
        if key not in predictors:predictors[key]=ExitPredictor(binding,device="cpu")
        return predictors[key]
    callback=make_exit_policy(features=features,predictor_resolver=resolve,apply_exit=apply)
    return callback,predictors


def _policy_record(payload):
    record,cfg,dates,source_hash,catalog,identities,destination=payload
    from my_strategy.services.czsc_wyckoff_exit import build_policy_dataset,make_union_decisions
    frame=_read_record(record);frame,_=bounded_replay_frame(frame,cfg["feature_start"]);raw=_raw(frame);settings=strategy_config()
    base=replay_decisions(frame,raw,entry_policy="fresh",position_start=cfg["feature_start"],config=settings)["decisions"]
    output={}
    for policy,entry,candidate,decisions in (("fresh","fresh","fresh_v1",base),
        ("union","risk","fresh_wyckoff_union_v1",make_union_decisions(frame,base,config=settings,candidate_policy="fresh_wyckoff_union_v1"))):
        exit_callback,predictors=_exit_router(frame,catalog[policy])
        def reference(account):
            folder=Path(destination).parent/"policy_behavior"/policy/record["symbol"].replace(".","_");folder.mkdir(parents=True,exist_ok=True)
            for key in ("ledger","daily","rejections","holding_snapshots"):_write_frame(pd.DataFrame(account.get(key,[])),folder/(key+".csv"))
        def callback(record,account):
            if record.get("clone_ledger_hash"):
                folder=Path(destination)/policy/record["symbol"].replace(".","_")/stable_hash({"candidate":record["candidate_id"],"date":record["date"]})[:20]
                folder.mkdir(parents=True,exist_ok=True)
                for key in ("ledger","daily","rejections"):_write_frame(pd.DataFrame(account[key]),folder/(key+".csv"))
                _save(folder/"record.json",record)
        data=build_policy_dataset(features=frame,raw=raw,candidate_decisions=decisions,behavior_decisions=decisions,market_dates=dates,
            start=cfg["feature_start"],initial_cash=cfg["initial_cash"],config=settings,entry_policy=entry,candidate_policy=candidate,
            behavior_policy="frozen_all_candidates_actual_cash_v1",continuation_policy_hash=identities[policy],source_snapshot_hash=source_hash,
            frozen_exit_policy=exit_callback,behavior_exit_policy=exit_callback,candidate_callback=callback,reference_callback=reference)
        if len(data):output[policy]=data
    return output


def prepare_policy_training(records,context,cfg,dates,source_hash,catalog,identities,*,cpu_workers,progress,check_cancel):
    selected=[r for r in records if _mainboard(r["symbol"])];parts={"fresh":[],"union":[]};check=check_cancel or (lambda:None)
    tasks=[(r,cfg,dates,source_hash,catalog,identities,str(context.subdir("evaluation","policy_candidate_clones"))) for r in selected]
    def collect(result,i):
        check()
        for policy,data in result.items():parts[policy].append(data)
        if progress:progress("全部冻结候选真实现金克隆与成熟策略标签",i+1,len(tasks),0)
    if cpu_workers>1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            for i,result in enumerate(pool.map(_policy_record,tasks,chunksize=1)):collect(result,i)
    else:
        for i,task in enumerate(tasks):collect(_policy_record(task),i)
    outputs={}
    for policy,frames in parts.items():
        if not frames:outputs[policy]=pd.DataFrame();continue
        if len({stable_hash(x.attrs["label_contract"]) for x in frames})!=1:raise ValueError("policy candidate contracts differ across frozen accounts")
        data=pd.concat(frames,ignore_index=True).sort_values(["date","symbol"]).reset_index(drop=True);data.attrs=dict(frames[0].attrs)
        data.attrs.update(data_version=stable_hash({"source":source_hash,"policy":policy,"router":identities[policy]}),
            label_reason_counts=dict(Counter(data.label_reason)),candidate_counts={"all_frozen_candidates":len(data),
            "decidable_flat_candidates":int(data.behavior_shares.eq(0).sum()),"mature":int(data.label_available.sum()),"ineligible":int((~data.input_eligible).sum())})
        data.to_parquet(context.subdir("dataset")/(policy+"-policy-training.parquet"),index=False);outputs[policy]=data
    return outputs

def _new_arm_record(payload):
    record,fold,probabilities,thresholds,baseline_threshold,heads,policy_heads,catalog,identities,cfg,dates,destination,source_hash=payload
    from my_strategy.services.czsc_wyckoff_exit import evaluate_exit_arm,evaluate_policy_arm,make_union_decisions
    frame=_read_record(record,fold["test_end"]);frame,offset=bounded_replay_frame(frame,fold["test_start"]);probabilities={name:values[offset:] for name,values in probabilities.items()};raw=_raw(frame);settings=strategy_config()
    if not (pd.to_datetime(frame.date)>=pd.Timestamp(fold["test_start"])).any():return {"symbol":record["symbol"],"accounts":{},"cash_reason":"no_test_bars"}
    base=replay_decisions(frame,raw,entry_policy="fresh",position_start=fold["test_start"],config=settings)["decisions"]
    decisions={};accounts={}
    for arm,expert in (("wyckoff_rule",None),("wyckoff_only","wyckoff"),("three_fusion","fusion"),("fusion_wyckoff_exit","dual_fusion"),("three_fusion_exit","fusion")):
        if arm=="wyckoff_rule":values=frame.wyckoff_rule_buy.fillna(False).to_numpy(dtype=float);threshold=.5
        else:
            values=probabilities[expert]
            threshold=baseline_threshold if expert=="dual_fusion" else thresholds[expert]["threshold"]
            if threshold is None:values=np.zeros(len(frame));threshold=.55
        replay={"decisions":frozen_candidate_filter(base,values,threshold)}
        decisions[arm]=replay["decisions"]
        head=heads.get("fresh")
        if arm.endswith("_exit") and head and not head.get("unavailable"):
            account=evaluate_exit_arm(features=frame,raw=raw,base_decisions=replay["decisions"],market_dates=dates,
                start=fold["test_start"],end=fold["test_end"],initial_cash=cfg["initial_cash"],config=settings,
                model_bundle=head,device="cpu",apply_exit=True,entry_policy="fresh",candidate_policy="fresh_v1",
                behavior_policy="rules_and_forward_three_fusion_fresh_v1",source_snapshot_hash=source_hash)
        else:
            account=execute_decisions(record["symbol"],raw,replay["decisions"],fold["test_start"],cfg["initial_cash"],settings,
                "wyckoff-new-arm",market_dates=dates,verified_sources=VERIFIED_SOURCES,entry_policy="fresh")
        accounts[arm]=account
    bridge_decisions=frozen_candidate_filter(base,probabilities["dual_fusion"],baseline_threshold)
    bridge=execute_decisions(record["symbol"],raw,bridge_decisions,fold["test_start"],cfg["initial_cash"],settings,"wyckoff-matched-dual-candidates",market_dates=dates,verified_sources=VERIFIED_SOURCES,entry_policy="fresh")
    union=make_union_decisions(frame,base,config=settings,candidate_policy="fresh_wyckoff_union_v1")
    accounts["union_rules"]=execute_decisions(record["symbol"],raw,union,fold["test_start"],cfg["initial_cash"],settings,
        "wyckoff-union-rules",market_dates=dates,verified_sources=VERIFIED_SOURCES,entry_policy="risk")
    for policy,arm,entry,candidate,intents in (("fresh","policy_fresh","fresh","fresh_v1",base),
        ("union","policy_union","risk","fresh_wyckoff_union_v1",union)):
        exit_callback,predictors=_exit_router(frame,catalog[policy])
        binding=policy_heads.get(policy)
        if binding and not binding.get("unavailable"):
            account=evaluate_policy_arm(features=frame,raw=raw,base_decisions=intents,market_dates=dates,
                start=fold["test_start"],end=fold["test_end"],initial_cash=cfg["initial_cash"],config=settings,
                model_bundle=binding,device="cpu",apply_entry=True,entry_policy=entry,exit_policy=exit_callback,
                candidate_policy=candidate,behavior_policy="frozen_all_candidates_actual_cash_v1",
                continuation_policy_hash=identities[policy],source_snapshot_hash=source_hash)
        else:
            account=execute_decisions(record["symbol"],raw,intents,fold["test_start"],cfg["initial_cash"],settings,
                "wyckoff-policy-unavailable-rules-fallback",market_dates=dates,verified_sources=VERIFIED_SOURCES,
                entry_policy=entry,exit_policy=exit_callback)
            account["policy_model_unavailable"]=binding
        account["rolling_exit_compute"]={"rows":sum(p.diagnostics["rows"] for p in predictors.values()),
            "batches":sum(p.diagnostics["batches"] for p in predictors.values()),"device":"cpu_numpy"}
        accounts[arm]=account
    auxiliary={"matched_dual_fixed_candidates":bridge}
    common=np.isfinite(probabilities["fusion"])
    for name,values,threshold in (("common_w_inputs_rules",np.where(common,1.,0.),.5),
        ("common_w_inputs_dual_gate",np.where(common,probabilities["dual_fusion"],np.nan),baseline_threshold)):
        intents=frozen_candidate_filter(base,values,threshold)
        auxiliary[name]=execute_decisions(record["symbol"],raw,intents,fold["test_start"],cfg["initial_cash"],settings,
            "wyckoff-common-input-auxiliary",market_dates=dates,verified_sources=VERIFIED_SOURCES,entry_policy="fresh")
    for arm,account in {**accounts,**auxiliary}.items():
        folder=Path(destination)/arm/record["symbol"].replace(".","_");folder.mkdir(parents=True,exist_ok=True)
        for key in ("ledger","daily","rejections","holding_snapshots","entry_diagnostics","exit_policy_diagnostics","entry_gate_diagnostics"):_write_frame(pd.DataFrame(account.get(key,[])),folder/(key+".csv"))
    return {"symbol":record["symbol"],"accounts":accounts,"auxiliary_accounts":auxiliary}


def _baseline_accounts(baseline_report,selected,fold,context,cfg):
    """Immutable old six-arm ledger references, paired on exact source and scope."""
    root=artifact_run_dir(baseline_report["run_id"],create=False);source_result=next(x for x in baseline_report["evaluation"] if x["fold"]["name"]==fold["name"])
    totals,variants,references={}, {},{}
    for arm in OLD_ARMS:
        accounts=[];bindings=[]
        for record in selected:
            directory=root/"evaluation"/fold["name"]/arm/record["symbol"].replace(".","_")
            if not directory.exists():continue
            data={}
            for key in ("ledger","daily","rejections"):
                path=directory/(key+".csv")
                try:frame=pd.read_csv(path)
                except pd.errors.EmptyDataError:frame=pd.DataFrame()
                data[key]=frame.where(pd.notna(frame),None).to_dict("records")
                bindings.append({"symbol":record["symbol"],"kind":key,"path":str(path),"sha256":_file_hash(path)})
            data["trades"]=data["ledger"]
            accounts.append(data)
        if not accounts:raise ValueError("immutable baseline ledger reference unavailable")
        total=aggregate_accounts(accounts,len(selected)*cfg["initial_cash"],len(selected));total["metrics"].update(round_trip_statistics([a["ledger"] for a in accounts]))
        if len(selected)==source_result["frozen_mainboard_accounts"]:
            for key,value in source_result["variants"][arm]["metrics"].items():
                actual=total["metrics"].get(key)
                if isinstance(value,(int,float)) and actual is not None and not np.isclose(actual,value,rtol=1e-9,atol=1e-9):raise ValueError("baseline full-ledger reaggregation mismatch "+arm+" "+key)
        totals[arm]=total;variants[arm]={"metrics":total["metrics"],"accounts":len(accounts),"immutable_reference":True}
        references[arm]={"run_id":baseline_report["run_id"],"bindings":bindings,"bindings_sha256":stable_hash(bindings)}
        _write_frame(pd.DataFrame(total["daily"]),context.subdir("evaluation",fold["name"],arm)/"aggregate_daily.csv")
    _save(context.subdir("reports")/(fold["name"]+"-old-six-ledger-references.json"),references)
    return totals,variants,references


def evaluate_wyckoff_fold(records,bundle_dir,fold,context,cfg,dates,baseline_report,catalog,identities,source_hash,
                           *,cpu_workers=4,device="mps",progress=None,check_cancel=None):
    selected=[r for r in records if _mainboard(r["symbol"])];check=check_cancel or (lambda:None)
    bundle=verified_bundle(bundle_dir);predictor=WyckoffPredictorSession(device=device);dual=DualPredictorSession(device=device)
    baseline=verified_dual_bundle(bundle["baseline_bundle"]["path"]);tasks=[]
    for i,record in enumerate(selected):
        check();frame=_read_record(record,fold["test_end"]);mask=pd.to_datetime(frame.date)>=pd.Timestamp(bundle["available_at"])
        values={name:np.full(len(frame),np.nan) for name in ("wyckoff","fusion","dual_fusion")}
        if mask.any():
            predictor.predict(frame.loc[mask].copy(),bundle_dir,as_of=fold["test_end"])
            for name in ("wyckoff","fusion"):values[name][mask.to_numpy()]=predictor.last_expert_values[name]
            values["dual_fusion"][mask.to_numpy()]=dual.predict(frame.loc[mask].copy(),bundle["baseline_bundle"]["path"],as_of=fold["test_end"])
        tasks.append((record,fold,values,bundle["thresholds"],baseline["probability_threshold"],bundle.get("exit_heads",{}),
            bundle.get("policy_heads",{}),catalog,identities,cfg,dates,str(context.subdir("evaluation",fold["name"])),source_hash))
        if progress:progress("三专家与旧双专家真实推理 "+fold["name"],i+1,len(selected),0)
    accounts={arm:[] for arm in NEW_ARMS};auxiliary_accounts={name:[] for name in ("matched_dual_fixed_candidates","common_w_inputs_dual_gate","common_w_inputs_rules")};cash=[]
    def collect(result,i):
        check()
        if result["accounts"]:
            for arm,account in result["accounts"].items():accounts[arm].append(account)
            for name,account in result["auxiliary_accounts"].items():auxiliary_accounts[name].append(account)
        else:cash.append({"symbol":result["symbol"],"reason":result["cash_reason"]})
        if progress:progress("十四臂独立账本 "+fold["name"],i+1,len(tasks),0)
    if cpu_workers>1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            for i,result in enumerate(pool.map(_new_arm_record,tasks,chunksize=1)):collect(result,i)
    else:
        for i,task in enumerate(tasks):collect(_new_arm_record(task),i)
    totals,variants,references=_baseline_accounts(baseline_report,selected,fold,context,cfg)
    for arm,items in accounts.items():
        if not items:raise ValueError("no new executable ledger accounts")
        total=aggregate_accounts(items,len(selected)*cfg["initial_cash"],len(selected));total["metrics"].update(round_trip_statistics([x["ledger"] for x in items]))
        totals[arm]=total;variants[arm]={"metrics":total["metrics"],"accounts":len(items),"qualification":"development_diagnostic"}
        if arm in {"wyckoff_only","three_fusion","three_fusion_exit"}:
            expert="wyckoff" if arm=="wyckoff_only" else "fusion";variants[arm]["entry_threshold_selection"]=bundle["thresholds"][expert]
        if arm in {"policy_fresh","policy_union"}:
            policy="fresh" if arm=="policy_fresh" else "union";variants[arm]["policy_head"]=bundle.get("policy_heads",{}).get(policy)
            variants[arm]["actual_policy_inference_rows"]=sum(x.get("policy_compute_info",{}).get("rows",0) for x in items)
            variants[arm]["fallback_accounts"]=sum(bool(x.get("policy_model_unavailable")) for x in items)
        _write_frame(pd.DataFrame(total["daily"]),context.subdir("evaluation",fold["name"],arm)/"aggregate_daily.csv")
    auxiliary={}
    for name,items in auxiliary_accounts.items():
        total=aggregate_accounts(items,len(selected)*cfg["initial_cash"],len(selected));total["metrics"].update(round_trip_statistics([x["ledger"] for x in items]))
        totals[name]=total;auxiliary[name]={"metrics":total["metrics"],"accounts":len(items),"purpose":"frozen_candidates_common_inputs_probability_vs_qualification_attribution"}
        _write_frame(pd.DataFrame(total["daily"]),context.subdir("evaluation",fold["name"],name)/"aggregate_daily.csv")
    bridge_total=totals["matched_dual_fixed_candidates"]
    initial=len(selected)*cfg["initial_cash"]
    def returns(arm):
        equity=pd.DataFrame(totals[arm]["daily"]).set_index("date")["equity"];prior=equity.shift(1);prior.iloc[0]=initial;return equity/prior-1
    comparisons={}
    for arm in NEW_ARMS:
        comparisons[arm]={}
        for baseline_arm in ("rules","fusion_exit","rules_exit","matched_dual_fixed_candidates","common_w_inputs_dual_gate","common_w_inputs_rules"):
            paired=pd.concat({"baseline":returns(baseline_arm),"new":returns(arm)},axis=1);paired["difference"]=paired["new"]-paired["baseline"]
            comparisons[arm][baseline_arm]={}
            for block in cfg["ledger_bootstrap_blocks"]:
                try:value=date_block_bootstrap(paired.reset_index(),"difference",block_length=block,dependence_horizon=cfg["ledger_dependence_bars"],seed=cfg["seed"])
                except ValueError as exc:value={"unavailable":str(exc)}
                comparisons[arm][baseline_arm][str(block)]=value
    result={"fold":fold,"qualification":"observed_historical_development_diagnostic","variants":variants,
        "frozen_mainboard_accounts":len(selected),"unavailable_accounts_cash_retained":cash,"old_six_reference_manifest_sha256":stable_hash(references),
        "paired_daily_return_uncertainty":comparisons,"candidate_contracts":{"same_pool":"frozen_unfiltered_fresh_active_plans_v2","expanded_pool":"fresh_wyckoff_union_v1"},
        "auxiliary_matched_dual_fixed_candidates":auxiliary["matched_dual_fixed_candidates"],"auxiliary_variants":auxiliary,
        "old_six_comparability":"immutable_diagnostics_old_Position_filter_can_change_candidates_not_pure_same_pool_increment",
        "compute_info":{"three_expert":predictor.diagnostics,"expert_inference":predictor.expert_diagnostics,"old_dual":dual.diagnostics,
            "exit_cpu_numpy_rows":sum(x.get("exit_compute_info",{}).get("rows",0)+x.get("rolling_exit_compute",{}).get("rows",0) for items in accounts.values() for x in items),
            "policy_cpu_numpy_rows":sum(x.get("policy_compute_info",{}).get("rows",0) for items in accounts.values() for x in items)}}
    _save(context.subdir("reports")/(fold["name"]+"-wyckoff.json"),result);return result


def _fit_head(data,directory,fold,trainer,cfg,device):
    if data.empty:return {"unavailable":"no_actual_mature_candidate_states"},None
    try:
        manifest=trainer(data,directory,fold["train_end"],fold["validation_start"],fold["validation_end"],device=device,
            seed=cfg["seed"],epochs=cfg["epochs"],batch_size=cfg["batch_size"],min_train_rows=cfg["min_train_rows"],min_validation_rows=cfg["min_validation_rows"])
    except ValueError as exc:
        if not any(t in str(exc) for t in ("insufficient mature samples","each require both label classes")):raise
        return {"unavailable":str(exc),"rules_fallback":True},None
    return {"model_dir":str(directory),"device":"cpu","contract_hash":manifest["label_contract"]["contract_hash"],
        "threshold":cfg["exit_threshold"] if manifest["label_contract"]["head"]=="holding_exit" else cfg["policy_threshold"],
        "available_at":manifest["available_at"],"manifest_sha256":manifest["manifest_sha256"],"checkpoint_sha256":manifest["checkpoint_sha256"],
        "label_cutoff":manifest["validation_end"]},manifest


def train_wyckoff_research(*,source_training_run_id,end=None,symbols=None,device="mps",cpu_workers=4,
                           progress=None,check_cancel=None,run_callback=None,config=None):
    cfg=copy.deepcopy(load_config("czsc_wyckoff_research") if config is None else config)
    if cfg["horizon"]!=10 or cfg["initial_cash"]!=100000 or cfg.get("publication_allowed") is not False or tuple(cfg["arms"])!=ALL_ARMS:
        raise ValueError("Wyckoff fixed target/shadow/fourteen-arm protocol immutable")
    source=json.loads((artifact_run_dir(source_training_run_id,create=False)/"reports/research.json").read_text())
    requested=[r["symbol"] for r in source["dataset_records"]] if symbols is None else list(symbols)
    context=create_run_context(task="czsc-wyckoff-research-train",as_of_date=end or source["data_end"],
        config={"research":cfg,"strategy":strategy_config()},seed=cfg["seed"],scope="train",stocks=requested,
        source="api",data_version=stable_hash({"source_run_id":source_training_run_id,"symbols":requested}),
        start_date=cfg["feature_start"],end_date=end or source["data_end"])
    if run_callback:run_callback(context.run_id)
    check=check_cancel or (lambda:None)
    try:
        data,records,source_records,snapshot,dates,calendar=prepare_wyckoff_dataset(source_training_run_id,context,cfg,
            symbols=symbols,end=end,cpu_workers=cpu_workers,progress=progress,check_cancel=check_cancel)
        baseline_path=artifact_run_dir(cfg["baseline_dual_run_id"],create=False)/"reports/dual-research.json"
        baseline_report=json.loads(baseline_path.read_text());baseline_hash=_file_hash(baseline_path)
        if baseline_report.get("status")!="complete" or baseline_report["source_snapshot"]["source_report_sha256"]!=snapshot["source_report_sha256"] or baseline_report["calendar"]["hash"]!=calendar["hash"]:
            raise ValueError("Wyckoff reused baselines require immutable exact same source/calendar")
        _save(context.subdir("config")/"czsc_wyckoff_research.json",cfg);_save(context.subdir("config")/"czsc_strategy.json",strategy_config())
        folds=[*cfg["development_windows"],{"name":"production",**cfg["production"]}];bundles=[]
        threshold_callback=validation_threshold_evaluator(records,context,cfg,dates,cpu_workers=cpu_workers,progress=progress,check_cancel=check_cancel)
        for fold in folds:
            check();old=next(x for x in baseline_report["model_bundles"] if x["name"]==fold["name"]);directory=context.subdir("models",fold["name"])
            bundle=train_wyckoff_bundle(data,directory,fold,baseline_bundle_dir=old["path"],device=device,config=cfg,
                threshold_evaluator=threshold_callback,progress=(lambda current,total:progress("量价与三专家OOF训练 "+fold["name"],current,total,0)) if progress else None,check_cancel=check_cancel)
            bundles.append({"name":fold["name"],"path":str(directory),"available_at":bundle["available_at"],"bundle_sha256":bundle["bundle_sha256"]})
        exit_data=prepare_exit_training(records,context,cfg,dates,snapshot["source_report_sha256"],baseline_report,bundles,
            cpu_workers=cpu_workers,progress=progress,check_cancel=check_cancel)
        catalog,identities,head_manifests=_train_exit_heads(exit_data,context,cfg,device=device,progress=progress,check_cancel=check_cancel)
        final_exits={}
        from my_strategy.services.czsc_wyckoff_exit import train_exit_model,train_policy_model
        for fold in folds:
            final_exits[fold["name"]]={}
            for policy,dataset in exit_data.items():
                check();binding,manifest=_fit_head(dataset,context.subdir("models",fold["name"],"exit-"+policy),fold,train_exit_model,cfg,device)
                final_exits[fold["name"]][policy]=binding
                if manifest:
                    head_manifests.append(manifest);catalog[policy].append(binding)
        for policy in catalog:
            catalog[policy].sort(key=lambda x:(x["available_at"],x["model_dir"]))
            identities[policy]=stable_hash({"version":"wyckoff_forward_exit_router_v1","models":catalog[policy],"fallback":"native_rules_when_unavailable"})
        _save(context.subdir("reports")/"frozen-exit-routers.json",{"catalog":catalog,"identity":identities,"chronological_only":True})
        policy_data=prepare_policy_training(records,context,cfg,dates,snapshot["source_report_sha256"],catalog,identities,
            cpu_workers=cpu_workers,progress=progress,check_cancel=check_cancel)
        for fold,item in zip(folds,bundles):
            policies={}
            for policy,dataset in policy_data.items():
                check();binding,manifest=_fit_head(dataset,context.subdir("models",fold["name"],"policy-"+policy),fold,train_policy_model,cfg,device)
                policies[policy]=binding
                if manifest:head_manifests.append(manifest)
            exits=final_exits[fold["name"]];bundle=attach_heads(item["path"],exit_head=exits["fresh"],exit_heads=exits,policy_heads=policies)
            item.update(bundle_sha256=bundle["bundle_sha256"],exit_model=bundle.get("exit_head"),policy_models=bundle.get("policy_heads"))
        evaluation=[]
        for fold,item in zip(folds,bundles):
            if "test_start" in fold:
                evaluation.append(evaluate_wyckoff_fold(records,item["path"],fold,context,cfg,dates,baseline_report,catalog,identities,
                    snapshot["source_report_sha256"],cpu_workers=cpu_workers,device=device,progress=progress,check_cancel=check_cancel))
        for old in snapshot["all_source_records"]:
            check()
            if _file_hash(Path(old["path"]))!=old["sha256"]:raise ValueError("original source changed during Wyckoff research")
        if _file_hash(Path(snapshot["source_report_path"]))!=snapshot["source_report_sha256"] or _file_hash(baseline_path)!=baseline_hash:raise ValueError("source/baseline report changed during new family training")
        manifests=[verified_bundle(x["path"],check_oof=True) for x in bundles];completed=local_now().isoformat()
        # Protocol dates are resolved only from data actually observed after the
        # complete freeze. No invented holiday calendar or backdated window.
        last_freeze=max([completed,*[x["training_completed_at"] for x in manifests]])
        verified_future=[day for day in dates if pd.Timestamp(day).date()>pd.Timestamp(last_freeze).date()]
        independent={**cfg["independent_window"],"actual_complete_freeze_at":last_freeze,
            "start":verified_future[0] if verified_future else None,"status":"future_unobserved" if verified_future else "awaiting_first_verified_session_after_actual_freeze"}
        compute={"trained_models":sum(x["compute_info"]["trained_models"] for x in manifests)+len(head_manifests),
            "training_batches":sum(x["compute_info"]["training_batches"] for x in manifests)+sum(x["training_batches"] for x in head_manifests),
            "reused_ma_structure_experts":10,"new_independent_heads":len(head_manifests),
            "actual_mps_training":any(x["compute_info"]["actual_mps_training"] for x in manifests) or any(x["actual_mps_training"] for x in head_manifests),
            "actual_cuda_training":any(x["compute_info"]["actual_cuda_training"] for x in manifests),
            "inference_rows":sum(e["compute_info"]["three_expert"]["rows"]+e["compute_info"]["old_dual"]["rows"] for e in evaluation),
            "inference_batches":sum(e["compute_info"]["three_expert"]["batches"]+e["compute_info"]["old_dual"]["batches"] for e in evaluation),
            "exit_cpu_numpy_rows":sum(e["compute_info"]["exit_cpu_numpy_rows"] for e in evaluation),
            "policy_cpu_numpy_rows":sum(e["compute_info"]["policy_cpu_numpy_rows"] for e in evaluation)}
        report={"run_id":context.run_id,"status":"complete","model_family":"wyckoff","strategy_version":cfg["version"],
            "publication_allowed":False,"promotion_status":"shadow_source_boundaries_and_independent_future_missing",
            "independent_window":independent,"observed_windows_diagnostic_only":True,"source_snapshot":snapshot,
            "source_artifacts_unchanged":True,"source_dataset_records":source_records,"dataset_records":records,"data_end":source["data_end"],
            "data_version":data.attrs["data_version"],"calendar":calendar,"market_dates":dates,"config":cfg,"run_context":context.to_dict(),
            "training_rows":len(data),"coverage":{"requested":len(requested),"success":len(records),"failed":0,
            "source_universe":snapshot["source_universe_count"],"full_universe":snapshot["selected_full_source_universe"]},
            "baseline_reference":{"run_id":baseline_report["run_id"],"report_path":str(baseline_path),"report_sha256":baseline_hash,"same_source_target":True},
            "model_bundles":bundles,"evaluation":evaluation,
            "policy_datasets":{policy:{"path":str(context.subdir("dataset")/(policy+"-policy-training.parquet")),
                "sha256":_file_hash(context.subdir("dataset")/(policy+"-policy-training.parquet")) if len(dataset) else None,
                "candidate_counts":dataset.attrs.get("candidate_counts",{}),"label_reason_counts":dataset.attrs.get("label_reason_counts",{}),
                "label_contract":dataset.attrs.get("label_contract",{})} for policy,dataset in policy_data.items()},
            "frozen_exit_routers":{"catalog":catalog,"identity":identities},
            "compute_info":compute,"model_gate":{"passed":False,"reason":"no_independent_future_and_corporate_actions_ST_identity_unresolved"},
            "conditional_dl":{"executed":False,"reason":"sequence_incremental_hypothesis_not_yet_supported_by_completed_ML_ledger_diagnostics"},
            "limitations":["Observed history is development only; all models and policy heads remain shadow.",
                "Current source pool survivorship, missing corporate actions and historical ST/delisting identity block formal qualification.",
                "Only SH60/SZ00 long-only Broker execution; other securities remain source coverage.",
                "Independent equal-capital accounts; no shared cash, forced terminal close or censored negative labels.",
                "Quarter windows lack required 160/240 date blocks; inference intervals unavailable."]}
        _save(context.subdir("reports")/"wyckoff-research.json",report)
        context.write_metadata({"status":"complete","model_family":"wyckoff","publication_allowed":False,"config":{"research":cfg,"strategy":strategy_config()},"coverage":report["coverage"],"compute_info":compute})
        return _json(report)
    except Exception as exc:
        context.write_metadata({"status":"failed","model_family":"wyckoff","publication_allowed":False,"error":str(exc),"config":{"research":cfg,"strategy":strategy_config()}});raise
