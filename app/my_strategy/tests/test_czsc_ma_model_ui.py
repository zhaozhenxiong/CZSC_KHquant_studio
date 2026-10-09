"""Render actual shipped UI with distinct MA model/input/rule identities."""
from pathlib import Path
import shutil
import subprocess

import pytest


def _run_ui(script: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped JavaScript UI checks")
    path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    harness = r"""
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const elements=new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id,{value:'',hidden:false,disabled:false,innerHTML:'',textContent:'',options:[],
    querySelector:() => element(id+'-child'),replaceChildren() { this.innerHTML=''; }});
  return elements.get(id);
}
global.document={getElementById:element}; global.location={hash:''};
global.sessionStorage={getItem:() => null};
global.statusPayload={models:[],calendar_run_id:'verified'};
global.fetch=async () => ({ok:true,text:async () => JSON.stringify(statusPayload)});
const source=fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);',
  'globalThis.uiTest={state,modelResolutionText,researchComputeCaption,computeCaption,researchEvidence,scanActionCell,renderEvents,refreshResearch,updateResearchControls,showResult};');
vm.runInThisContext(source);
const {state,modelResolutionText,researchComputeCaption,computeCaption,researchEvidence,scanActionCell,renderEvents,refreshResearch,updateResearchControls,showResult}=uiTest;
"""
    result = subprocess.run([node, "-e", harness + script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_ma_missing_model_and_ineligible_input_never_render_placeholder_probability():
    _run_ui(r"""
const missing={feature_profile:'ma_trend_v1',status:'rules_no_model',model_run_id:null,checkpoint:null,
  probability:null,applied_to_entry:false,reason_codes:['model_unavailable'],model_input_eligible:true};
state.analysis={as_of:'2026-10-08',events:[],research:{feature_profile:'ma_trend_v1'},
  compute_info:{research:true,feature_profile:'ma_trend_v1',selected_device:'mps',actual_device:null,
    model_inference_rows:0,model_inference_batches:0,model_loads:0,inference_reason_counts:{model_unavailable:182}},
  next_session_decision:{action:'WAIT',candidate_action:'WAIT',entry_policy:'fresh',model_resolution:missing,
    model_input_eligible:true,agent_evidence:[{role:'ml',judgment:'unavailable',probability:null,model:missing}]}};
renderEvents();
assert(element('analysis-compute-status').textContent.includes('均线趋势 ML'));
assert(element('analysis-compute-status').textContent.includes('未执行 ML 推理（0 行）：无兼容检查点 182 行'));
const rendered=element('analysis-research-evidence').innerHTML;
assert(rendered.includes('均线趋势 ML：不可用'));
assert(rendered.includes('无兼容检查点') && rendered.includes('概率不可用'));
assert(!rendered.includes('概率 0%') && !rendered.includes('概率 50%'));
assert(!rendered.includes('未参与入场，仅作影子观察'));
const invalid={...missing,status:'input_veto',model_run_id:'ma-run',probability:.5,
  model_input_eligible:false,reason_codes:['model_input_ineligible']};
const row={category:'excluded',model_resolution:invalid,model_probability:.68,
  agent_evidence:[{role:'ml',judgment:'shadow',probability:.5,model:invalid}]};
const invalidHtml=scanActionCell(row)+researchEvidence(row);
assert(invalidHtml.includes('模型输入不可用') && invalidHtml.includes('均线趋势 ML：输入不可用'));
assert(!invalidHtml.includes('概率 50%') && !invalidHtml.includes('概率 68%'));
// Explicit null in a saved resolution must not pick up a stale row probability.
assert(!scanActionCell({category:'watch',model_resolution:missing,model_probability:.68}).includes('概率 68%'));
assert(!scanActionCell({category:'watch',model_probability:null,probability:.68}).includes('概率 68%'));
const savedNull={...missing,status:'historical_shadow',model_run_id:'ma-run',reason_codes:[]};
const stale={category:'watch',model_resolution:savedNull,agent_evidence:[
  {role:'ml',judgment:'shadow',probability:.68,model:{...savedNull,probability:.68}}]};
assert(!researchEvidence(stale).includes('概率 68%'));
assert(researchEvidence(stale).includes('概率不可用'));
assert(!modelResolutionText({...missing,probability:.5}).includes('概率 50%'));
for (const value of [NaN,Infinity,'',false]) {
  assert(modelResolutionText({...missing,probability:value}).includes('概率不可用'));
}
""")


def test_valid_ma_shadow_probability_is_independent_of_rule_and_entry_qualification():
    _run_ui(r"""
const route={feature_profile:'ma_trend_v1',status:'rule_input_veto_shadow',model_run_id:'ma-run',checkpoint:'2026Q1',
  model_input_eligible:true,probability:.68,probability_threshold:.55,applied_to_entry:false};
const row={category:'excluded',eligible:false,input_eligible:false,model_input_eligible:true,entry_policy:'fresh',
  model_resolution:route,agent_evidence:[{role:'ml',judgment:'support',probability:.68,threshold:.55,applied:false,model:route}]};
let html=scanActionCell(row)+researchEvidence(row);
assert(html.includes('规则资格：输入不可用'));
assert(html.includes('均线趋势 ML') && html.includes('10日研究概率 68%'));
assert(html.includes('影子观察，ML未参与入场'));
assert(html.includes('均线趋势 ML：影子观察') && !html.includes('均线趋势 ML：支持'));
assert(html.includes('假设新入场的10交易日费用后正收益概率 68%'));
assert(!html.includes('ML参与入场<') && !html.includes('ML 已参与入场过滤'));
for (const policy of ['fresh','risk']) {
  const active={...route,status:'production_active',applied_to_entry:true,release_id:'ma-release',reason_codes:[]};
  html=modelResolutionText(active,{entry_policy:policy,model_input_eligible:true});
  assert(html.includes('模型保持影子') && html.includes('ML 未参与入场'));
  assert(!html.includes('ML 已参与入场过滤'));
  assert(scanActionCell({category:'buy',candidate_action:'BUY',entry_policy:policy,model_resolution:active})
    .includes('规则资格：满足买入组合条件'));
}
assert(modelResolutionText({...route,status:'historical_released',applied_to_entry:true},
  {entry_policy:'legacy',model_input_eligible:true}).includes('ML 已参与入场过滤'));
assert(modelResolutionText({status:'rules_no_model',applied_to_entry:false}).includes('无兼容检查点'));
assert(!modelResolutionText({status:'rules_no_model'}).includes('均线趋势 ML'));
const malicious={...route,model_run_id:'<script>bad</script>'};
assert(!researchEvidence({...row,model_resolution:malicious}).includes('<script>'));
""")


def test_zero_row_reasons_and_actual_gpu_work_use_execution_records():
    _run_ui(r"""
const base={research:true,feature_profile:'ma_trend_v1',selected_device:'mps',actual_device:null,
  model_inference_rows:0,model_inference_batches:0,model_loads:0,mps_work:false};
let text=researchComputeCaption({...base,inference_reason_counts:{model_input_ineligible:4,unsupported_board:2}});
assert(text.includes('模型输入不可用 4 行') && text.includes('板块暂不支持 ML 推理 2 行'));
assert(!text.includes('无兼容检查点') && !text.includes('本次 MPS 模型推理'));
text=researchComputeCaption({...base,mps_work:true});
assert(text.includes('未执行 ML 推理（0 行）') && !text.includes('本次 MPS 模型推理'));
text=researchComputeCaption({...base,model_inference_rows:3,actual_device:'cpu'});
assert(text.includes('本次 CPU 模型推理') && !text.includes('本次 MPS 模型推理'));
text=researchComputeCaption({...base,model_inference_rows:3,mps_work:true,actual_device:'mps',model_loads:1});
assert(text.includes('本次 MPS 模型推理') && text.includes('3 行') && text.includes('模型加载 1 次'));
text=researchComputeCaption({...base,model_inference_rows:3,selected_device:'cpu'});
assert(text.includes('本次 CPU 模型推理'));
text=researchComputeCaption({...base,model_inference_rows:3});
assert(text.includes('实际设备未记录') && !text.includes('本次 MPS 模型推理'));
text=researchComputeCaption({model_inference_rows:0,selected_device:'mps'},
  {research:{model_routes:[{identity:{status:'rules_no_model'}}]}});
assert(text.includes('无兼容检查点') && !text.includes('均线趋势 ML'));
assert(researchComputeCaption({selected_device:'mps'}).includes('未保存本次 ML 推理统计'));
assert(computeCaption({available:true,selected_device:'mps'},true).includes('设备可用'));
""")


def test_model_selectors_status_and_training_variant_preserve_profile_identity():
    _run_ui(r"""
statusPayload={feature_profile:'ma_trend_v1',models:[],calendar_run_id:'verified'};
(async () => {
  await refreshResearch();
  assert(element('research-status').textContent.includes('均线趋势 ML · 0 个研究模型'));
  assert(element('research-status').textContent.includes('无兼容检查点'));
  assert(!element('research-status').textContent.includes('ML 保持影子观察'));
  const model={run_id:'ma-run',feature_profile:'ma_trend_v1',model_gate:{passed:false},
    checkpoints:[{name:'production',available_at:'2026-09-30'}]};
  statusPayload={feature_profile:'ma_trend_v1',models:[model],calendar_run_id:'verified',
    active_release:{release_id:'legacy-release',identity:{feature_profile:'czsc_price_volume_v1'}}};
  await refreshResearch();
  assert(element('analysis-model').innerHTML.includes('均线趋势 ML · ma-run'));
  const status=element('research-status').textContent;
  assert(status.includes('ML 活动生产发布 legacy-release'));
  assert(!status.includes('均线趋势 ML 活动生产发布'));
  assert(status.includes('本次模型入场资格仍按信号日期和入场规则核验'));
  element('analysis-mode').value='research';element('analysis-model').value='ma-run';
  element('analysis-model-policy').value='pinned';updateResearchControls('analysis');
  assert(element('analysis-checkpoint').innerHTML.includes('均线趋势 ML · 最终训练候选（production）'));
  const metrics={total_return:0,max_drawdown:0,completed_round_trips:0,trade_expectancy:0};
  showResult('research_train',{run_id:'ma-run',feature_profile:'ma_trend_v1',model_gate:{passed:false},
    coverage:{requested:1,success:1,failed:0},evaluation:[{fold:{name:'test'},variants:{ml:{metrics}}}]});
  assert(element('research-training-result').textContent.includes('均线趋势 ML · 研究运行 ma-run'));
  assert(element('research-evaluation').innerHTML.includes('组合规则＋均线趋势 ML'));
})().catch((error) => {console.error(error);process.exitCode=1;});
""")
