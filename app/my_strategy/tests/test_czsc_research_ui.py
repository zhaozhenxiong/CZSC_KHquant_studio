"""Research additions preserve six routes, calendar validation and visible failure evidence."""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import time

from fastapi.testclient import TestClient
import pytest

from my_strategy.storage.czsc_results import ResultStore
from my_strategy.web_dashboard.app import create_app
from my_strategy.web_dashboard.tasks import TaskManager


@pytest.fixture
def manager(tmp_path):
    store = ResultStore(tmp_path / "runs.db", tmp_path / "runs")
    instance = TaskManager(tmp_path / "jobs.db", store, workers=1)
    yield instance
    instance.close()


@pytest.mark.parametrize("status", ["cancelled", "failed"])
def test_research_terminal_state_reaches_metadata_without_success_registration(manager, monkeypatch, status):
    from my_strategy.web_dashboard.tasks import Cancelled

    def run(job_id, kind, spec):
        run_id = "terminal-research-" + status
        path = manager.results.runs_root / run_id / "metadata.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"status": "running", "data_version": "frozen-input"}), encoding="utf-8")
        manager._patch(job_id, run_id=run_id)
        if status == "cancelled":
            raise Cancelled("cancelled during feature preparation")
        raise ValueError("invalid frozen snapshot")

    monkeypatch.setattr(manager, "_run", run)
    job = manager.submit("research_train", {"end": "2026-09-30", "calendar_run_id": "verified"})
    deadline = time.monotonic() + 3
    while manager.get(job["job_id"])["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(.01)
    # Closing waits for the terminal metadata write after the task status changes.
    manager._executor.shutdown(wait=True)
    final = manager.get(job["job_id"])
    metadata = json.loads((manager.results.runs_root / final["run_id"] / "metadata.json").read_text(encoding="utf-8"))
    assert final["status"] == metadata["status"] == status
    assert metadata["error"] == final["error"] and metadata["finished_at"]
    assert metadata["data_version"] == "frozen-input"
    assert manager.results.list("research_train") == []


def test_research_request_contract_requires_calendar_and_valid_run_ids(manager, monkeypatch):
    monkeypatch.setattr(manager, "submit", lambda kind, spec: {"kind": kind, "spec": spec})
    with TestClient(create_app(manager)) as client:
        base = {"kind": "research_train", "spec": {"end": "2026-09-30", "calendar_run_id": "calendar-20261001"}}
        response = client.post("/api/tasks", json=base)
        assert response.status_code == 202
        assert response.json()["kind"] == "research_train"
        assert "start" not in response.json()["spec"]
        for spec in ({"end": "2026-09-30"}, {"calendar_run_id": "calendar-20261001"}):
            assert client.post("/api/tasks", json={"kind": "research_train", "spec": spec}).status_code == 422
        for key in ("model_run_id", "calendar_run_id"):
            for value in ("../escape", "D:\\secret", ".", "", "a" * 181):
                assert client.post("/api/tasks", json={"kind": "scan", "spec": {key: value}}).status_code == 422
        accepted = client.post("/api/tasks", json={"kind": "scan", "spec": {"research": True, "model_run_id": "research-001", "end": "2026-09-30"}})
        assert accepted.status_code == 202 and accepted.json()["spec"]["research"]
        assert client.post("/api/tasks", json={"kind": "scan", "spec": {"research": "false"}}).status_code == 422


def test_checkpoint_request_is_restricted_to_explicit_research_scan_and_backtest(manager, monkeypatch):
    monkeypatch.setattr(manager, "submit", lambda kind, spec: {"kind": kind, "spec": spec})
    spec = {"research": True, "model_run_id": "research-001", "model_fold": "2025Q4",
            "start": "2026-01-01", "end": "2026-09-30"}
    with TestClient(create_app(manager)) as client:
        response = client.post("/api/tasks", json={"kind": "backtest", "spec": spec})
        assert response.status_code == 202 and response.json()["spec"]["model_fold"] == "2025Q4"
        assert response.json()["spec"]["model_policy"] == "pinned"
        scan = client.post("/api/tasks", json={"kind": "scan", "spec": spec})
        assert scan.status_code == 202
        assert scan.json()["spec"]["model_fold"] == "2025Q4"
        assert scan.json()["spec"]["model_policy"] == "pinned"
        for kind in ("research_train", "update"):
            assert client.post("/api/tasks", json={"kind": kind, "spec": spec}).status_code == 422
        for changed in ({"research": False}, {"model_run_id": None}, {"model_fold": "../2025Q4"}, {"model_fold": ""}):
            assert client.post("/api/tasks", json={"kind": "backtest", "spec": {**spec, **changed}}).status_code == 422


def test_task_dispatch_preserves_explicit_checkpoint(manager, monkeypatch):
    from my_strategy.services import czsc_compute, czsc_research
    monkeypatch.setattr(czsc_compute, "compute_status", lambda value: {"available": True, "selected_device": "cpu"})
    monkeypatch.setattr(czsc_research, "backtest_research", lambda **kwargs: kwargs)
    spec = {"research": True, "symbols": ["600027.SH"], "model_run_id": "research-001", "model_fold": "2025Q4",
            "start": "2026-01-01", "end": "2026-09-30", "device": "cpu"}
    result = manager._run("fixture", "backtest", spec)
    assert result["model_fold"] == "2025Q4" and result["model_run_id"] == "research-001"


def test_status_without_model_allows_research_observation(manager, monkeypatch):
    from my_strategy.services import czsc_research
    monkeypatch.setattr(czsc_research, "research_status", lambda: {"models": [], "calendar_run_id": None}, raising=False)
    monkeypatch.setattr(manager, "submit", lambda kind, spec: {"kind": kind, "spec": spec})
    with TestClient(create_app(manager)) as client:
        assert client.get("/api/research/status").json() == {"models": [], "calendar_run_id": None}
        response = client.post("/api/tasks", json={"kind": "scan", "spec": {"research": True, "end": "2026-09-30"}})
        assert response.status_code == 202 and "model_run_id" not in response.json()["spec"]
        assert client.get("/models").status_code == 404


def test_run_summary_and_report_preserve_categories_actual_dates_and_failure(manager):
    result = {"run_id": "research-screen-test", "model_gate": {"passed": False, "reason": "尚未证实改善"},
              "model_run_id": "research-model-test", "category_counts": {"watch": 1, "excluded": 1},
              "coverage": {"requested": 3, "success": 2, "failed": 1},
              "data_range": {"requested_end": "2026-09-30", "end": "2026-09-30"},
              "rows": [{"symbol": "000001.SZ", "category": "watch", "model_probability": .68, "model_validated": False, "data_end": "2026-09-30"},
                       {"symbol": "600519.SH", "category": "excluded", "model_probability": None, "data_end": "2026-09-28", "reason": "missing_target_date"}],
              "failures": [{"symbol": "000002.SZ", "error": "missing data"}]}
    manager.results.save("scan", result)
    row = manager.results.list()[0]
    assert row["summary"]["model_gate"]["passed"] is False
    assert row["summary"]["model_run_id"] == "research-model-test"
    assert row["summary"]["category_counts"] == {"watch": 1, "excluded": 1}
    with TestClient(create_app(manager)) as client:
        report = client.get("/api/runs/research-screen-test").json()
        assert report == result
        assert len(report["rows"]) + len(report["failures"]) == report["coverage"]["requested"]


def test_shipped_research_rows_filter_and_shadow_date_rendering():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped JavaScript rendering checks")
    path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {value:'',textContent:'',innerHTML:'',hidden:false,disabled:false,querySelector:() => element(id+'-child')});
  return elements.get(id);
}
global.document = {getElementById:element};
const source = fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);','globalThis.researchTest = {state,renderScan,researchSpec};');
vm.runInThisContext(source);
const {state,renderScan,researchSpec} = researchTest;
element('scan-filter').value = 'all';
state.scan = {data_end:'2026-09-30', data_range:{requested_end:'2026-09-30',end:'2026-09-30'},
  coverage:{requested:3,success:2,failed:1},category_counts:{watch:1,excluded:1},model_gate:{passed:false},
  rows:[{symbol:'000001.SZ',category:'watch',action:'watch',data_end:'2026-09-30',model_probability:.68,model_validated:false,point_confirmed_at:null,entry_trigger:'<script>bad</script>',agent_evidence:[{role:'ml',judgment:'shadow'}]},
        {symbol:'600519.SH',category:'excluded',action:'excluded',data_end:'2026-09-28',model_probability:null,reason:'missing_target_date'}],
  failures:[{symbol:'000002.SZ',error:'missing data'}]};
renderScan();
assert(element('scan-rows').innerHTML.includes('10日研究概率 68%'));
assert(element('scan-rows').innerHTML.includes('影子观察'));
assert(element('scan-rows').innerHTML.includes('行情 2026-09-28'));
assert(element('scan-rows').innerHTML.includes('结构确认 未确认'));
assert(!element('scan-rows').innerHTML.includes('<script>bad</script>'));
assert(element('scan-research-result').textContent.includes('尚未证实改善'));
assert(element('scan-coverage').textContent.includes('请求截止 2026-09-30'));
assert(element('scan-failures-child').innerHTML.includes('missing data'));
element('scan-filter').value='excluded'; renderScan();
assert(!element('scan-rows').innerHTML.includes('000001.SZ'));
assert(element('scan-rows').innerHTML.includes('600519.SH'));
element('scan-mode').value='research'; element('scan-model').value='';
state.research={calendar_run_id:'calendar-test'};
assert.deepStrictEqual(researchSpec('scan'),{research:true,usage_mode:'historical',model_policy:'auto',entry_policy:'fresh',calendar_run_id:'calendar-test'});
element('scan-model').value='research-test';
assert.strictEqual(researchSpec('scan').model_run_id,undefined);
element('scan-model-policy').value='pinned'; element('scan-checkpoint').value='2025Q4';
state.research.models=[{run_id:'research-test',checkpoints:[{name:'2025Q4',available_at:'2025-09-30'}]}];
assert.strictEqual(researchSpec('scan').model_run_id,'research-test');
assert.strictEqual(researchSpec('scan').model_fold,'2025Q4');
element('scan-mode').value='structure';
assert.deepStrictEqual(researchSpec('scan'),{research:false});
"""
    result = subprocess.run([node, "-e", script, str(path)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


def test_entry_policy_controls_and_plan_rendering_preserve_research_context():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped JavaScript rendering checks")
    root = Path(__file__).resolve().parents[1] / "web_dashboard/static"
    html = (root / "index.html").read_text(encoding="utf-8")
    for prefix in ("analysis", "scan", "backtest"):
        options = html.split(f'<select id="{prefix}-entry-policy">', 1)[1].split("</select>", 1)[0]
        assert options.index('value="fresh"') < options.index('value="legacy"') < options.index('value="risk"')
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id,{value:'',textContent:'',innerHTML:'',hidden:false,disabled:false,querySelector:() => element(id+'-child')});
  return elements.get(id);
}
global.document = {getElementById:element};
const source = fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);','globalThis.researchTest = {state,researchSpec,resultResearchContext,applyResearchContext,entryPlanHtml,modelResolutionText,renderScan,nextSessionGuidance};');
vm.runInThisContext(source);
const {state,researchSpec,resultResearchContext,applyResearchContext,entryPlanHtml,modelResolutionText,renderScan,nextSessionGuidance} = researchTest;
state.research = {calendar_run_id:'calendar-test',models:[]};
for (const prefix of ['analysis','scan','backtest']) {
  element(prefix+'-mode').value='research';
  assert.strictEqual(researchSpec(prefix).entry_policy,'fresh');
  element(prefix+'-entry-policy').value='risk'; element(prefix+'-usage-mode').value='production';
  assert.throws(() => researchSpec(prefix),/请选择历史滚动研究/);
  assert.strictEqual(element(prefix+'-usage-mode').value,'production');
  element(prefix+'-usage-mode').value='historical';
  assert.strictEqual(researchSpec(prefix).entry_policy,'risk');
}
const saved = {model_gate:{passed:false},request:{research:true,entry_policy:'risk',usage_mode:'historical',model_policy:'auto'}};
const savedContext = resultResearchContext(saved);
assert.strictEqual(savedContext.entry_policy,'risk');
applyResearchContext('analysis',savedContext);
applyResearchContext('backtest',researchSpec('analysis'));
assert.strictEqual(researchSpec('backtest').entry_policy,'risk');
const oldContext = resultResearchContext({model_gate:{passed:false},request:{research:true}});
assert.strictEqual(oldContext.entry_policy,'legacy');
applyResearchContext('scan',oldContext);
assert.strictEqual(researchSpec('scan').entry_policy,'legacy');
element('scan-mode').value='structure'; applyResearchContext('scan',{research:false});
assert(element('scan-entry-policy').disabled);
assert.deepStrictEqual(researchSpec('scan'),{research:false});
const plan = {policy:'risk',status:'active',entry_allowed:true,signal_date:'2026-09-30',reference_price:7.4,
  price_floor:7.25,price_ceiling:7.5,stop_price:6.9,risk_fraction:.01,target_weight:.3,valid_for_sessions:1,
  identity:{point_type:'二买辅助',point_confirmed_at:'2026-09-29T15:00:00+08:00'},reason_codes:[],
  diagnostics:{point_age_bars:1,weekly_price_direction:1,weekly_bi_direction:-1}};
let rendered = entryPlanHtml({entry_plan:plan});
assert(rendered.includes('待下一开盘核验') && rendered.includes('7.25～7.5 元'));
assert(rendered.includes('6.9 元') && rendered.includes('账户风险预算') && rendered.includes('1%') && rendered.includes('30%'));
assert(rendered.includes('闭合周价格：向上 · 已确认周笔：向下'));
assert(rendered.includes('未经独立认证') && rendered.includes('非成交价'));
rendered = entryPlanHtml({entry_plan:{...plan,status:'holding',entry_allowed:false,reason_codes:['already_positioned']}});
assert(rendered.includes('理论持仓 · 不新增买入') && !rendered.includes('入场计划仅对下一'));
assert(rendered.includes('入场时买点龄 1 根股票K线') && rendered.includes('入场时闭合周价格'));
assert(entryPlanHtml({entry_plan:{...plan,status:'rejected',entry_allowed:false,reason_codes:['buy_point_already_consumed']}}).includes('同一买点计划已使用'));
assert(entryPlanHtml({entry_plan:{...plan,status:'rejected',entry_allowed:false,reason_codes:['point_age_exceeds_experimental_limit']}}).includes('已过期（实验期限）'));
assert(entryPlanHtml({entry_plan:{...plan,status:'none',entry_allowed:false,reason_codes:['rule_buy_not_met']}}).includes('待触发'));
assert(entryPlanHtml({entry_plan:{...plan,status:'none',entry_allowed:false,reason_codes:['input_ineligible']}}).includes('无可执行计划'));
assert(entryPlanHtml({}).includes('按原目标仓位对照恢复'));
assert(entryPlanHtml({entry_plan:{...plan,policy:'fresh',price_floor:null,price_ceiling:null,stop_price:null,risk_fraction:null}}).includes('该规则未设结构失效价'));
rendered = entryPlanHtml({entry_plan:{...plan,identity:{point_type:'<script>bad</script>'},reason_codes:['<img src=x onerror=bad>']}});
assert(!rendered.includes('<script>') && !rendered.includes('<img'));
assert(modelResolutionText({status:'entry_contract_shadow',applied_to_entry:false}).includes('10日模型与本次实际退出契约不一致，仅影子'));
assert(nextSessionGuidance({action:'HOLD',candidate_action:'BUY',position_scope:'theoretical_prefix_position',reference_close:7.4,reference_price_date:'2026-09-30',next_market_session:'2026-10-08'}).includes('与手动持仓独立'));
assert(nextSessionGuidance({action:'HOLD',position_scope:'flat_at_start',position_start:'2026-01-01'}).includes('从研究开始日 2026-01-01 初始化为空仓'));
element('scan-filter').value='all';
state.scan={coverage:{requested:2,success:2,failed:0},failures:[],rows:[
  {symbol:'000001.SZ',category:'buy',position_intent:'BUY',action:'BUY',candidate_action:'BUY',entry_plan:plan},
  {symbol:'000002.SZ',category:'watch',position_intent:'HOLD',action:'HOLD',candidate_action:'BUY',entry_plan:{...plan,status:'holding',entry_allowed:false}}]};
renderScan(); assert(element('scan-rows').innerHTML.includes('理论仓位意图：持有'));
element('scan-filter').value='buy'; renderScan();
assert(element('scan-rows').innerHTML.includes('000001.SZ') && !element('scan-rows').innerHTML.includes('000002.SZ'));
"""
    result = subprocess.run([node, "-e", script, str(root / "js/workbench.js")], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
