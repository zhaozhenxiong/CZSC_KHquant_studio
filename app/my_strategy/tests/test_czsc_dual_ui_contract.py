"""Dual research keeps identity, shadow status and real-position scope visible."""
from pathlib import Path
import json
import shutil
import subprocess

import pytest
from pydantic import ValidationError

from my_strategy.cli import build_parser
from my_strategy.web_dashboard.api import AnalysisRequest, TaskRequest


def test_dual_task_requires_frozen_source_and_rejects_unsafe_identity():
    with pytest.raises(ValidationError):
        TaskRequest(kind="dual_research_train", spec={"end": "2026-10-08"})
    with pytest.raises(ValidationError):
        TaskRequest(kind="dual_research_train", spec={"end": "2026-10-08", "source_training_run_id": "../other"})
    task = TaskRequest(kind="dual_research_train", spec={"end": "2026-10-08", "source_training_run_id": "frozen-run"})
    assert task.spec.source_training_run_id == "frozen-run"
    assert task.spec.model_family == "dual"
    args = build_parser().parse_args(["research-dual-train", "--source-training-run-id", "frozen-run", "--end", "2026-10-08"])
    assert args.source_training_run_id == "frozen-run"
    assert args.model_family == "dual"
    assert AnalysisRequest(symbol="000001.SZ", model_family="dual").model_family == "dual"
    assert build_parser().parse_args(["research-status", "--model-family", "dual"]).model_family == "dual"
    assert build_parser().parse_args(["research-status"]).model_family == "ma_trend"


def test_dual_status_reports_actual_completion_and_frozen_source(monkeypatch, tmp_path):
    from my_strategy.services import czsc_dual_models, czsc_dual_runtime
    root = tmp_path / "dual-run"
    (root / "reports").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({"status": "complete"}))
    (root / "reports/dual-research.json").write_text(json.dumps({
        "source_snapshot": {"source_run_id": "original-frozen-run"},
        "model_bundles": [{"name": "first"}, {"name": "production"}],
    }))
    def verified(directory):
        return {"available_at": "2026-09-30", "training_completed_at":
                "2026-10-09T15:53:15+08:00" if directory.name == "production" else "2026-10-09T15:50:00+08:00"}
    monkeypatch.setattr(czsc_dual_models, "verified_bundle", verified)
    monkeypatch.setattr(czsc_dual_runtime, "ARTIFACT_RUNS_ROOT", tmp_path)
    status = czsc_dual_runtime.dual_research_status()
    candidate = status["models"][0]
    assert candidate["source_training_run_id"] == "original-frozen-run"
    assert candidate["training_completed_at"] == "2026-10-09T15:53:15+08:00"
    assert candidate["checkpoints"][0]["training_completed_at"] != candidate["checkpoints"][0]["available_at"]
    assert status["active_release"] is None


def test_dual_ui_does_not_mix_candidate_families_or_invert_entry_probability():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    source = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const elements=new Map();
function el(id) { if (!elements.has(id)) elements.set(id,{value:'',disabled:false,hidden:false,innerHTML:'',textContent:'',options:[],querySelector:()=>el(id+'-child')});return elements.get(id); }
global.document={getElementById:el};global.location={hash:''};
vm.runInThisContext(fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);','globalThis.testUI={state,researchSpec,populateResearchModels,updateResearchControls,modelResolutionText};'));
const {state,researchSpec,populateResearchModels,updateResearchControls,modelResolutionText}=testUI;
state.research={calendar_run_id:'verified',models:[{run_id:'ma-run',feature_profile:'ma_trend_v1',checkpoints:[{name:'production',available_at:'2026-09-30'}]}],dual:{models:[{run_id:'dual-run',model_family:'dual',checkpoints:[{name:'2025Q4',available_at:'2025-09-30'}]}]}};
el('analysis-model-family').value='dual';el('analysis-mode').value='research';el('analysis-usage-mode').value='historical';el('analysis-model-policy').value='auto';
populateResearchModels('analysis');
assert(el('analysis-model').innerHTML.includes('dual-run'));
assert(!el('analysis-model').innerHTML.includes('ma-run'));
assert.equal(researchSpec('analysis').model_family,'dual');
el('analysis-model-policy').value='pinned';el('analysis-model').value='dual-run';el('analysis-checkpoint').value='2025Q4';el('analysis-end').value='2025-01-01';
updateResearchControls('analysis');assert.throws(()=>researchSpec('analysis'),/不能用于/);
const text=modelResolutionText({model_family:'dual',status:'historical_shadow',applied_to_entry:false,probability:.68,expert_probabilities:{ma:.7,structure:.6,fusion:.68}}, {exit_model:{probability:null,reason:'真实持仓状态缺失'}});
assert(text.includes('结构＋均线融合 ML'));assert(text.includes('均线专家 70%'));assert(text.includes('结构专家 60%'));assert(text.includes('融合 68%'));
assert(text.includes('ML 未参与入场'));assert(text.includes('真实持仓状态缺失'));assert(!text.includes('32%'));
"""
    result = subprocess.run([node, "-e", script, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
