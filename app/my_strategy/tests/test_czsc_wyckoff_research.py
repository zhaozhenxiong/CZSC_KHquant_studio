from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from my_strategy.services.czsc_dual_models import train_dual_bundle
from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS,STRUCTURE_COLUMNS
from my_strategy.services.czsc_wyckoff_features import W_INPUT_COLUMNS,get_wyckoff_profile
from my_strategy.services.czsc_wyckoff_models import (FUSION_COLUMNS,WyckoffPredictorSession,WyckoffResolver,
    train_wyckoff_bundle,verified_bundle,freeze_broker_threshold,expert_inputs,get_bundle_profile)
from my_strategy.services.czsc_wyckoff_research import ALL_ARMS

@pytest.fixture
def sample():
    dates=pd.bdate_range("2023-01-02","2025-06-30");x=np.sin(np.arange(len(dates))/7)
    frame=pd.DataFrame({c:x+i*.01 for i,c in enumerate([*MA_TREND_COLUMNS,*STRUCTURE_COLUMNS,*W_INPUT_COLUMNS])})
    frame["finished_bi_count"]=6.;frame["date"]=dates.strftime("%Y-%m-%d");frame["symbol"]="000001.SZ"
    frame["available_at"]=[d.strftime("%Y-%m-%d")+"T15:00:00+08:00" for d in dates]
    frame["structure_confirmed_at"]=(dates-pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    frame["zone_confirmed_at"]=frame["weekly_available_at"]=frame["monthly_available_at"]=None
    frame["label"]=(x>0).astype(float);frame["net_return"]=np.where(x>0,.02,-.01)
    frame["label_end"]=(dates+pd.offsets.BDay(11)).strftime("%Y-%m-%d")
    for key in ("label_available","input_eligible","model_input_eligible","wyckoff_input_eligible"):frame[key]=True
    for key in ("reason_codes","model_reason_codes","wyckoff_reason_codes"):frame[key]=[[] for _ in dates]
    frame.attrs={"data_version":"fixture-frozen","label_version":"czsc_fixed_horizon_broker_v1",
        "label_contract":{"label_version":"czsc_fixed_horizon_broker_v1","horizon":10,"initial_cash":100000,"calendar_hash":"fixture"},
        "wyckoff_feature_version":get_wyckoff_profile()["version"],"wyckoff_schema_hash":get_wyckoff_profile()["schema_hash"],"wyckoff_feature_columns":list(W_INPUT_COLUMNS)}
    return frame


def test_separate_expert_qualifications_and_74_bound_inputs(sample):
    assert len(get_bundle_profile()["columns"])==74 and len(set(ALL_ARMS))==14
    frame=sample.iloc[:3].copy();frame.loc[0,"wyckoff_input_eligible"]=False;frame.loc[1,"model_input_eligible"]=False;frame.loc[2,"input_eligible"]=False
    assert expert_inputs(frame,"wyckoff").input_eligible.tolist()==[False,True,True]
    assert expert_inputs(frame,"ma").input_eligible.tolist()==[True,False,True]
    assert expert_inputs(frame,"structure").input_eligible.tolist()==[True,True,False]
    assert len(expert_inputs(frame,"wyckoff").attrs["feature_columns"])==32


def test_broker_threshold_never_selects_zero_trades_or_proxy():
    cfg={"threshold_grid":[.5,.55],"threshold_min_completed_round_trips":2,"threshold_min_candidate_coverage":.1}
    scores=[{"threshold":.5,"completed_round_trips":0,"candidate_coverage":0,"net_return":0,"max_drawdown":0,"evidence":"complete_validation_broker_ledger"},
        {"threshold":.55,"completed_round_trips":3,"candidate_coverage":.2,"net_return":-.01,"max_drawdown":.03,"evidence":"complete_validation_broker_ledger"}]
    chosen=freeze_broker_threshold(scores,cfg);assert chosen["threshold"]==.55 and chosen["scores"][0]["selectable"] is False
    scores[1]["completed_round_trips"]=0;assert freeze_broker_threshold(scores,cfg)["threshold"] is None
    scores[0]["evidence"]="all_feature_fixed_10_proxy"
    with pytest.raises(ValueError,match="Broker ledger"):freeze_broker_threshold(scores,cfg)


def test_resolver_empty_catalog_stays_rules_and_pin_guard(tmp_path):
    with pytest.raises(ValueError,match="explicit pin"):WyckoffResolver(usage_mode="retrospective",runs_root=tmp_path)
    assert WyckoffResolver(runs_root=tmp_path).resolve("2026-10-08")["applied_to_entry"] is False


def test_real_three_expert_forward_oof_reuse_and_future_guard(sample,tmp_path):
    fold={"train_end":"2024-12-31","validation_start":"2025-01-01","validation_end":"2025-03-31"}
    cfg={"feature_start":"2023-01-01","epochs":1,"batch_size":128,"min_train_rows":20,"min_validation_rows":20,"seed":42,
        "threshold_grid":[.5,.55],"threshold_min_completed_round_trips":2}
    oldpath=tmp_path/"old";old=train_dual_bundle(sample,oldpath,fold,device="cpu",config=cfg)
    oldbytes={str(p):p.read_bytes() for p in oldpath.rglob("*") if p.is_file()}
    def evidence(predictor,fold):
        validation=sample.loc[(sample.date>=fold["validation_start"])&(sample.date<=fold["validation_end"])]
        values=predictor(validation)
        assert set(values)=={"ma","structure","wyckoff","fusion"}
        return {name:[{"threshold":t,"evidence":"complete_validation_broker_ledger","completed_round_trips":3,
            "candidate_coverage":.3,"net_return":.02 if t==.55 else .01,"max_drawdown":.01} for t in cfg["threshold_grid"]] for name in ("wyckoff","fusion")}
    path=tmp_path/"new";bundle=train_wyckoff_bundle(sample,path,fold,baseline_bundle_dir=oldpath,device="cpu",config=cfg,threshold_evaluator=evidence)
    assert bundle["compute_info"]["trained_models"]==7 and bundle["compute_info"]["reused_experts"]==2
    oof=pd.read_parquet(bundle["oof"]["path"])
    assert (oof.expert_available_at<oof.date).all() and (oof.target_label_end<=fold["train_end"]).all()
    fusion=json.loads((path/"fusion/manifest.json").read_text())
    assert fusion["schema"]["columns"]==FUSION_COLUMNS and fusion["hyperparameters"]["weight_decay"]==.01
    session=WyckoffPredictorSession(device="cpu");test=sample.loc[sample.date>fold["validation_end"]].copy()
    assert np.isfinite(session.predict(test,path,as_of="2025-06-30")).all()
    assert set(session.last_expert_values)=={"ma","structure","wyckoff","fusion"}
    test.loc[test.index[0],"wyckoff_input_eligible"]=False
    result=session.predict(test,path,as_of="2025-06-30")
    assert np.isnan(result[0]) and np.isfinite(session.last_dual_fallback_values[0])
    assert session.last_fallback_model["family"]=="dual" and session.last_fallback_compute_info["rows"]==1
    with pytest.raises(ValueError,match="future feature"):session.predict(test,path,as_of="2025-04-01")
    assert verified_bundle(path,check_oof=True)["bundle_sha256"]==bundle["bundle_sha256"]
    assert all(Path(p).read_bytes()==value for p,value in oldbytes.items())
    # Only newly trained W and fusion are fitted on repeated outer folds.
    second=train_wyckoff_bundle(sample,tmp_path/"new2",fold,baseline_bundle_dir=oldpath,device="cpu",config=cfg,threshold_evaluator=evidence)
    assert second["compute_info"]["trained_models"]==2 and second["compute_info"]["reused_oof_blocks"]==5

def test_frozen_candidate_veto_never_creates_followup_plans():
    from my_strategy.services.czsc_wyckoff_research import frozen_candidate_filter
    base=[{"date":"2025-04-01","action":"BUY","target_weight":.9,"entry_gate_passed":True,"entry_plan":{"plan_id":"original","status":"active","entry_allowed":True}},
        {"date":"2025-04-02","action":"HOLD","target_weight":.9,"entry_plan":{"plan_id":"original","status":"holding","entry_allowed":False}},
        {"date":"2025-04-03","action":"SELL","target_weight":0.,"entry_plan":{"plan_id":"original","status":"exit","entry_allowed":False}}]
    veto=frozen_candidate_filter(base,[.1,.99,.99],.55)
    assert sum(bool(row.get("entry_plan",{}).get("entry_allowed")) for row in veto)==0
    assert veto[0]["entry_plan"]["plan_id"]=="original" and base[0]["entry_plan"]["entry_allowed"] is True
    assert veto[1]["entry_plan"]["status"]=="holding" and veto[2]["action"]=="SELL"
