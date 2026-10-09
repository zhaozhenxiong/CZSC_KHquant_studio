"""New research family exposes causal evidence and missing-expert provenance."""
from pathlib import Path
import json
import shutil
import subprocess

import pytest
from pydantic import ValidationError

from my_strategy.cli import build_parser
from my_strategy.web_dashboard.api import AnalysisRequest, TaskRequest


def test_wyckoff_task_source_identity_and_legacy_cli_default():
    for source in (None, "../foreign-run"):
        with pytest.raises(ValidationError):
            TaskRequest(kind="wyckoff_research_train", spec={"end": "2026-10-08", "source_training_run_id": source})
    task = TaskRequest(kind="wyckoff_research_train", spec={"end": "2026-10-08", "source_training_run_id": "frozen-run"})
    assert task.spec.model_family == "wyckoff"
    args = build_parser().parse_args(["research-wyckoff-train", "--source-training-run-id", "frozen-run", "--end", "2026-10-08"])
    assert args.model_family == "wyckoff"
    assert AnalysisRequest(symbol="000001.SZ", model_family="wyckoff").model_family == "wyckoff"
    assert build_parser().parse_args(["research-status"]).model_family == "ma_trend"


def test_catalog_never_exposes_incomplete_training_and_records_actual_clock(monkeypatch, tmp_path):
    from my_strategy.services import czsc_wyckoff_models, czsc_wyckoff_runtime
    root = tmp_path / "w-run"
    (root / "reports").mkdir(parents=True)
    metadata = root / "metadata.json"
    metadata.write_text(json.dumps({"status": "running"}))
    report = {"status": "complete", "model_family": "wyckoff", "model_bundles": [{"name": "early"}, {"name": "production"}],
              "source_snapshot": {"source_run_id": "frozen-run"}}
    (root / "reports/wyckoff-research.json").write_text(json.dumps(report))
    monkeypatch.setattr(czsc_wyckoff_runtime, "ARTIFACT_RUNS_ROOT", tmp_path)
    def verified(directory):
        return {"available_at": "2026-09-30", "training_completed_at":
            "2026-10-09T19:01:00+08:00" if directory.name == "production" else "2026-10-09T18:50:00+08:00"}
    monkeypatch.setattr(czsc_wyckoff_models, "verified_bundle", verified)
    assert czsc_wyckoff_runtime.wyckoff_research_status()["models"] == []
    metadata.write_text(json.dumps({"status": "complete"}))
    result = czsc_wyckoff_runtime.wyckoff_research_status()
    assert result["models"][0]["training_completed_at"] == "2026-10-09T19:01:00+08:00"
    assert result["models"][0]["source_training_run_id"] == "frozen-run"
    assert result["active_release"] is None


def test_three_expert_ui_separates_missing_fusion_fallback_and_actual_exit():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    source = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const elements=new Map();
function el(id) { if (!elements.has(id)) elements.set(id,{value:'',disabled:false,hidden:false,innerHTML:'',textContent:'',options:[],querySelector:()=>el(id+'-child')});return elements.get(id); }
global.document={getElementById:el};global.location={hash:''};
vm.runInThisContext(fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);','globalThis.testUI={state,researchSpec,populateResearchModels,modelResolutionText,dualFeatureEvidence};'));
const {state,researchSpec,populateResearchModels,modelResolutionText,dualFeatureEvidence}=testUI;
state.research={calendar_run_id:'verified',models:[{run_id:'ma'}],dual:{models:[{run_id:'dual'}]},wyckoff:{models:[{run_id:'w',model_family:'wyckoff',checkpoints:[{name:'2026Q3',available_at:'2026-06-30'}]}]}};
el('analysis-model-family').value='wyckoff';el('analysis-mode').value='research';el('analysis-usage-mode').value='historical';el('analysis-model-policy').value='auto';
populateResearchModels('analysis');assert(el('analysis-model').innerHTML.includes('value="w"'));assert(!el('analysis-model').innerHTML.includes('value="dual"'));
assert.equal(researchSpec('analysis').model_family,'wyckoff');
el('analysis-model-policy').value='pinned';el('analysis-model').value='w';el('analysis-checkpoint').value='2026Q3';el('analysis-end').value='2026-01-01';assert.throws(()=>researchSpec('analysis'),/不能用于/);
const text=modelResolutionText({model_family:'wyckoff',status:'dual_fallback_shadow',applied_to_entry:false,model_input_eligible:true,probability:.68,probability_source:'dual_fallback',expert_probabilities:{ma:.7,structure:.6,wyckoff:null,fusion:null},fallback_model:{model_run_id:'immutable-dual',checkpoint:'2026Q3'}}, {exit_model:{probability:null,reason:'真实持仓状态缺失'}});
assert(text.includes('威科夫量价专家 —'));assert(text.includes('融合 —'));assert(text.includes('回退概率来源'));assert(text.includes('immutable-dual'));assert(text.includes('三专家融合概率缺失'));assert(text.includes('ML 未参与入场'));assert(!text.includes('32%'));
const evidence=dualFeatureEvidence({expert_features:{ma:{values:{}},structure:{values:{}},wyckoff:{values:{w_demand_score:.6}}},wyckoff_evidence:{event:'Test',state:'demand_test_confirmed',anchor_at:'2026-01-01',observed_at:'2026-01-05T15:00:00+08:00',available_at:'2026-01-05T15:00:00+08:00',buy_candidate:true,input_eligible:true}});
assert(evidence.includes('74项'));assert(evidence.includes('32项输入'));assert(evidence.includes('原始锚点'));assert(evidence.includes('首次观测'));assert(evidence.includes('规则评分不等于盈利概率'));
"""
    result = subprocess.run([node, "-e", script, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_wyckoff_markers_use_confirmation_and_hide_future_and_unclosed_week():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    source = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/wyckoff_markers.js"
    script = r"""
const assert=require('node:assert/strict');require(process.argv[1]);
const event={event:'Test',anchor_at:'2026-01-01',observed_at:'2026-01-05T15:00:00+08:00',available_at:'2026-01-05T15:00:00+08:00',buy_candidate:true};
const args={events:[event,{...event,available_at:'2026-01-06T15:00:00+08:00'}],bars:[{time:'2026-01-01'},{time:'2026-01-05'},{time:'2026-01-06'}],frequency:'日线',asOf:'2026-01-05'};
const markers=KHQuantWyckoffMarkers.build(args);assert.equal(markers.length,1);assert.equal(markers[0].time,'2026-01-05');assert.equal(markers[0].shape,'square');assert(markers[0].text.includes('候选'));
assert.equal(KHQuantWyckoffMarkers.build({...args,frequency:'周线'}).length,0);
assert.equal(KHQuantWyckoffMarkers.build({...args,events:[{...event,observed_at:null}]}).length,0);
"""
    result = subprocess.run([node, "-e", script, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
