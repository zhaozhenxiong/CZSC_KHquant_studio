"""Exercise the shipped UI's dated model requests and actual participation labels."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_shipped_unified_research_requests_guidance_and_navigation():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped JavaScript UI checks")
    path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const elements=new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id,{value:'',hidden:false,disabled:false,innerHTML:'',textContent:'',
    querySelector:() => element(id+'-child'),replaceChildren() { this.innerHTML=''; }});
  return elements.get(id);
}
global.document={getElementById:element}; global.location={hash:''};
const source=fs.readFileSync(process.argv[1],'utf8')
  .replace('async function analyze(query, preserveDates = false, locate = null) {',
    'async function analyze(query,preserveDates=false,locate=null) { globalThis.navigationRequest={query,preserveDates,locate}; } async function unusedAnalyze(query,preserveDates=false,locate=null) {')
  .replace('init().catch(showError);',
    'globalThis.uiTest={state,researchSpec,updateResearchControls,nextSessionGuidance,renderEvents,scanActionCell,resultResearchContext,analyzeSymbol,renderBacktest};');
vm.runInThisContext(source);
const {state,researchSpec,updateResearchControls,nextSessionGuidance,renderEvents,scanActionCell,resultResearchContext,analyzeSymbol,renderBacktest}=uiTest;
state.research={calendar_run_id:'verified',models:[{run_id:'model-a',checkpoints:[
  {name:'production',available_at:'2026-09-30',train_label_end:'2026-06-30'},
  {name:'2025Q4',available_at:'2025-09-30',train_label_end:'2025-06-30'}]}]};
for (const prefix of ['analysis','scan','backtest']) {
  element(prefix+'-mode').value='research'; element(prefix+'-usage-mode').value='historical';
  element(prefix+'-model-policy').value='auto'; element(prefix+'-model').value='model-a';
  element(prefix+'-checkpoint').value='production'; element(prefix+'-start').value='2026-01-01';
  element(prefix+'-end').value='2026-09-30'; updateResearchControls(prefix);
  const spec=researchSpec(prefix);
  assert.equal(spec.research,true); assert.equal(spec.usage_mode,'historical'); assert.equal(spec.model_policy,'auto');
  assert.equal(spec.model_run_id,undefined); assert.equal(spec.model_fold,undefined);
  assert(element(prefix+'-model').disabled); assert(element(prefix+'-checkpoint').disabled);
  assert(element(prefix+'-checkpoint').innerHTML.includes('最终训练候选（production）'));
  assert(!element(prefix+'-checkpoint').innerHTML.includes('生产模型'));
}
element('backtest-model-policy').value='pinned'; updateResearchControls('backtest');
assert.throws(() => researchSpec('backtest'),/不能用于 2026-01-01/);
element('backtest-checkpoint').value='2025Q4';
assert.deepEqual(researchSpec('backtest'),{research:true,usage_mode:'historical',model_policy:'pinned',entry_policy:'fresh',model_run_id:'model-a',model_fold:'2025Q4',calendar_run_id:'verified'});
element('backtest-usage-mode').value='retrospective';
assert.throws(() => researchSpec('backtest'),/仅用于分析图/);
element('scan-usage-mode').value='retrospective'; assert.throws(() => researchSpec('scan'),/仅用于分析图/);
element('analysis-usage-mode').value='retrospective'; element('analysis-model-policy').value='pinned';
element('analysis-end').value='2026-01-01';
assert.equal(researchSpec('analysis').model_fold,'production');
element('analysis-mode').value='structure'; assert.deepEqual(researchSpec('analysis'),{research:false});

const resolution={status:'historical_shadow',model_run_id:'model-a',checkpoint:'2025Q4',available_at:'2025-09-30',
  applied_to_entry:false,probability:.68,probability_threshold:.55,reason_codes:['model_shadow','production_release_missing']};
const decision={action:'HOLD',candidate_action:'BUY',position_intent:'HOLD',basis:'持仓继续',reference_close:4.95,
  reference_price_date:'2026-09-30',next_market_session:'2026-10-08',position_scope:'theoretical_prefix_position',model_resolution:resolution,
  agent_evidence:[{role:'ml',judgment:'shadow',probability:.68,threshold:.55,applied:false}]};
state.query={usage_mode:'historical',as_of:'2026-09-30'};
const guidance=nextSessionGuidance(decision,'2026-09-30');
assert(guidance.includes('下一交易日 2026-10-08')); assert(guidance.includes('收盘参考价 4.95 元'));
assert(guidance.includes('10交易日费用后正收益研究概率 68%'));
assert(guidance.includes('ML 未参与入场')); assert(guidance.includes('model-a / 2025Q4'));
assert(guidance.includes('与手动持仓独立')); assert(!guidance.includes('生产已生效'));
const active=nextSessionGuidance({...decision,model_resolution:{...resolution,status:'production_active',applied_to_entry:true,release_id:'release-a',promoted_at:'2026-10-02T15:00:00+08:00',reason_codes:[]}},'2026-09-30');
assert(active.includes('生产已生效')); assert(active.includes('ML 已参与入场过滤')); assert(active.includes('发布 release-a'));
assert(nextSessionGuidance({...decision,usage_mode:'retrospective'},'2026-09-30').includes('不能作为历史可执行或认证成绩'));
state.analysis={next_session_decision:decision,research:{usage_mode:'historical',agent_evidence:decision.agent_evidence},events:[
  {time:'2026-09-29',action:'SELL',reference_price:4.86,reference_price_date:'2026-09-29',reason:'退出 <script>x</script>',available_at:'2026-09-29T15:00:00+08:00'}]};
renderEvents(); assert(!element('analysis-research-evidence').hidden);
assert(element('analysis-research-evidence').innerHTML.includes('机器学习：影子观察'));
assert(element('analysis-research-evidence').innerHTML.includes('组合条件：买入候选；仓位意图：持有'));
assert(element('event-list').innerHTML.includes('卖出参考价 4.86 元'));
assert(!element('event-list').innerHTML.includes('<script>'));
const cell=scanActionCell({...decision,category:'exit',personal_exit_review:'SELL'});
assert(cell.includes('仓位意图：持有')); assert(cell.includes('组合条件：买入候选'));
assert(cell.includes('手动持仓退出复核：卖出')); assert(cell.includes('影子观察，ML未参与入场'));
assert(!cell.includes('仓位意图：卖出'));

const request={research:true,usage_mode:'production',model_policy:'pinned',model_run_id:'model-a',model_fold:'2025Q4',calendar_run_id:'verified',device:'cuda:1',start:'2026-01-01',end:'2026-09-30',symbols:['600027.SH'],initial_cash:100000};
assert.deepEqual(resultResearchContext({request}),{research:true,usage_mode:'production',model_policy:'pinned',model_run_id:'model-a',model_fold:'2025Q4',calendar_run_id:'verified',device:'cuda:1',entry_policy:'legacy'});
assert.deepEqual(resultResearchContext({request:{research:false}}),{research:false});
analyzeSymbol('600027.SH','2026-09-29',{request}).then(() => {
  assert.equal(location.hash,'analysis'); assert.equal(navigationRequest.query.usage_mode,'production');
  assert.equal(navigationRequest.query.model_policy,'pinned'); assert.equal(navigationRequest.query.model_fold,'2025Q4');
  assert.equal(navigationRequest.query.as_of,'2026-09-29'); assert.equal(navigationRequest.query.start,'2026-01-01');
  assert.equal(navigationRequest.query.device,'cuda:1'); assert.equal(navigationRequest.query.initial_cash,undefined);
  assert.equal(navigationRequest.query.entry_policy,'legacy');
  state.backtest={request,model_summary:{usage_mode:'historical',model_policy:'auto',shadow_days:150,no_model_days:20,ml_applied_days:0},
    model_routes:[{symbol:'600027.SH',start:'2026-01-01',end:'2026-03-30',days:50,identity:resolution}],daily:[],trades:[]};
  renderBacktest(); const html=element('backtest-research-result').innerHTML;
  assert(html.includes('无模型规则执行 20 股票日')); assert(html.includes('影子规则执行 150 股票日'));
  assert(html.includes('ML未参与入场')); assert(html.includes('实际模型路由 1 段'));
  assert(html.includes('model-a / 2025Q4'));
}).catch((error) => { console.error(error); process.exitCode=1; });
"""
    result = subprocess.run([node, "-e", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_shell_defaults_are_combined_historical_auto_and_retrospective_is_analysis_only():
    shell = (Path(__file__).resolve().parents[1] / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    for prefix in ("analysis", "scan", "backtest"):
        assert f'id="{prefix}-mode"><option value="research">' in shell
        assert f'id="{prefix}-usage-mode"><option value="historical">' in shell
        assert f'id="{prefix}-model-policy"><option value="auto">' in shell
    assert shell.count('<option value="retrospective">') == 1
    assert 'id="analysis-research-evidence"' in shell
    assert 'id="chip-profile"' in shell and 'id="chart-range-start"' in shell
