"""Research scan progress reflects actual stages without claiming GPU availability is work."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from my_strategy.storage.czsc_results import ResultStore
from my_strategy.web_dashboard.tasks import TaskManager


@pytest.fixture
def manager(tmp_path):
    instance = TaskManager(tmp_path / "tasks.db", ResultStore(tmp_path / "results.db", tmp_path / "runs"), workers=1)
    yield instance
    instance.close()


def terminal(manager, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["status"] not in {"queued", "running", "cancelling"}:
            return job
        time.sleep(.01)
    raise AssertionError("research task did not finish")


def prepare_dispatch(monkeypatch):
    from my_strategy.services import czsc_compute
    from my_strategy.storage import personal_portfolio
    monkeypatch.setattr(czsc_compute, "compute_status", lambda device: {"available": True, "selected_device": device})
    monkeypatch.setattr(personal_portfolio, "PersonalStore", lambda: SimpleNamespace(holdings=lambda: {"items": []}))


def test_scan_progress_is_monotonic_and_actual_compute_persists(manager, monkeypatch):
    from my_strategy.services import czsc_research
    prepare_dispatch(monkeypatch)
    updates = []
    original_patch = manager._patch
    def capture(job_id, **fields):
        if "progress" in fields or "progress_detail" in fields:
            updates.append(fields.copy())
        return original_patch(job_id, **fields)
    monkeypatch.setattr(manager, "_patch", capture)
    actual_compute = {"research": True, "selected_device": "cuda:1", "cuda_work": True,
                      "model_inference_rows": 3, "model_inference_batches": 1, "cache_hits": 4, "cpu_workers": 5}
    def scan(**kwargs):
        assert kwargs["cpu_workers"] == 5 and kwargs["batch_size"] == 17
        assert kwargs["device"] == "cuda:1" and kwargs["model_run_id"] == "chosen-model"
        progress = kwargs["progress"]
        progress("CPU并行研究特征准备", 2, 4, 0)
        progress("GPU批量ML推理", 2, 4, 0)
        progress("CPU并行研究特征准备", 1, 4, 0)
        progress("多角色证据扫描", 4, 4, 1)
        return {"run_id": "scan-progress", "compute_info": actual_compute, "rows": [], "failures": []}
    monkeypatch.setattr(czsc_research, "scan_research", scan)
    submitted = manager.submit("scan", {"research": True, "end": "2026-09-30", "symbols": ["000001.SZ"],
                                        "device": "cuda:1", "cpu_workers": 5, "batch_size": 17, "model_run_id": "chosen-model"})
    result = terminal(manager, submitted["job_id"])
    assert result["status"] == "succeeded", result["error"]
    percentages = [fields["progress"] for fields in updates if "progress" in fields]
    assert percentages == sorted(percentages)
    assert percentages[:-1] == [47.5, 47.5, 47.5, 95] and percentages[-1] == 100
    devices = [fields["progress_detail"]["device"] for fields in updates if "progress_detail" in fields]
    assert devices == ["cpu", "cuda:1", "cpu", "cpu"]
    assert result["compute_info"] == actual_compute == result["result"]["compute_info"]
    assert manager.list()[0]["compute_info"] == actual_compute


def test_scan_cancel_still_interrupts_at_stage_boundary(manager, monkeypatch):
    from my_strategy.services import czsc_research
    prepare_dispatch(monkeypatch)
    entered, released = threading.Event(), threading.Event()
    def scan(**kwargs):
        entered.set()
        assert released.wait(3)
        kwargs["progress"]("CPU研究特征准备", 1, 2, 0)
        pytest.fail("cancelled scan reached inference")
    monkeypatch.setattr(czsc_research, "scan_research", scan)
    submitted = manager.submit("scan", {"research": True, "end": "2026-09-30", "symbols": ["000001.SZ"], "device": "cpu"})
    assert entered.wait(3)
    manager.cancel(submitted["job_id"])
    released.set()
    result = terminal(manager, submitted["job_id"])
    assert result["status"] == "cancelled" and result["result"] is None
    assert manager.results.list("scan") == []


def test_training_keeps_stage_local_counts_without_overall_percentage(manager, monkeypatch):
    from my_strategy.services import czsc_research
    prepare_dispatch(monkeypatch)
    observed = []
    def train(**kwargs):
        for stage, current, total in (("逐日特征与标签", 2, 2), ("训练 2025Q4", 1, 30), ("训练 2026Q2", 0, 1)):
            kwargs["progress"](stage, current, total, 0)
            observed.append(manager.list()[0])
        return {"run_id": "train-progress", "rows": []}
    monkeypatch.setattr(czsc_research, "train_research", train)
    submitted = manager.submit("research_train", {"end": "2026-09-30", "symbols": ["000001.SZ"],
                                                  "calendar_run_id": "verified", "device": "cpu"})
    result = terminal(manager, submitted["job_id"])
    assert result["status"] == "succeeded", result["error"]
    assert [job["progress"] for job in observed] == [0, 0, 0]
    assert [(job["progress_detail"]["current"], job["progress_detail"]["total"]) for job in observed] == [(2, 2), (1, 30), (0, 1)]


def test_shipped_ui_uses_actual_ml_work_and_preserves_explicit_selection():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped JavaScript checks")
    source_path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const elements=new Map();
function element(id) {
  if (!elements.has(id)) {
    const item={value:'',textContent:'',innerHTML:'',hidden:false,disabled:false};
    Object.defineProperty(item,'options',{get:()=>[...item.innerHTML.matchAll(/<option value="([^"]+)"/g)].map((match)=>({value:match[1]}))});
    elements.set(id,item);
  }
  return elements.get(id);
}
global.document={getElementById:element};
const source=fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);',
  'globalThis.progressTest={state,renderComputeResult,renderScanComputePlan,taskDetail,taskActions,refreshCompute};');
vm.runInThisContext(source);
const {state,renderComputeResult,renderScanComputePlan,taskDetail,taskActions,refreshCompute}=progressTest;
state.scan={};
renderComputeResult('scan',{research:true,selected_device:'cuda:1',cuda_work:true,model_inference_rows:64,
  model_inference_batches:2,cpu_workers:4,cache_hits:63});
let text=element('scan-compute-result').textContent;
assert(text.includes('CUDA MLP 推理') && text.includes('64 行') && text.includes('2 个推理批次'));
assert(text.includes('CPU 4 个准备进程') && text.includes('特征缓存命中 63'));
renderComputeResult('scan',{research:true,selected_device:'cuda:1',cuda_work:false,model_inference_rows:0});
text=element('scan-compute-result').textContent;
assert(text.includes('未执行 ML 推理（0 行）') && !text.includes('本次 CUDA'));
renderComputeResult('scan',{research:true,selected_device:'cpu',cuda_work:false,model_inference_rows:4});
assert(element('scan-compute-result').textContent.includes('CPU MLP 推理'));
renderComputeResult('scan',{research:true,selected_device:'mps',mps_work:true,model_inference_rows:4});
assert(element('scan-compute-result').textContent.includes('MPS MLP 推理'));
renderComputeResult('scan',{research:true,selected_device:'mps',mps_work:false,model_inference_rows:0});
assert(!element('scan-compute-result').textContent.includes('本次 MPS'));
renderComputeResult('scan',{selected_device:'mps',actual_device:'cpu',backend:'numpy',bars:4,mps_work:false});
assert(element('scan-compute-result').textContent.includes('CPU 信号计算'));
assert(element('scan-compute-result').textContent.includes('CPU float64 精度'));
renderComputeResult('scan',{research:true,training:true,selected_device:'mps',mps_work:true,model_training_models:2,model_training_batches:10});
assert(element('scan-compute-result').textContent.includes('MPS MLP 训练'));
assert(element('scan-compute-result').textContent.includes('10 个优化批次'));
renderComputeResult('scan',{research:true,selected_device:'cuda:1'});
assert(element('scan-compute-result').textContent.includes('未保存本次 ML 推理统计'));
element('scan-mode').value='research'; element('scan-model').value=''; element('scan-device').value='cuda:1';
element('scan-model-policy').value='auto';
renderScanComputePlan();
assert(element('scan-compute-plan').textContent.includes('按所选日期匹配模型并批量推理'));
assert(element('scan-compute-plan').textContent.includes('缺模型时保留规则判断'));
assert(element('scan-compute-plan').textContent.includes('设备 cuda:1'));
assert(!element('scan-compute-plan').textContent.includes('不执行 GPU ML 推理'));
assert.equal(element('scan-model').value,'');
element('scan-model-policy').value='pinned'; element('scan-model').value='explicit-model'; renderScanComputePlan();
assert.equal(element('scan-model').value,'explicit-model');
assert(element('scan-compute-plan').textContent.includes('CPU 准备结构与量价特征'));
assert(element('scan-compute-plan').textContent.includes('真实推理行数、批次和 GPU 工作以本次结果为准'));
element('scan-mode').value='structure'; renderScanComputePlan();
assert(element('scan-compute-plan').textContent.includes('批量信号运算'));
const running={kind:'scan',job_id:'fixture',status:'running',spec:{research:true,device:'cuda:1'},
  progress_detail:{stage:'CPU并行研究特征准备',current:32,total:64,device:'cpu'}};
text=taskDetail(running);
assert(text.includes('当前阶段 CPU') && text.includes('尚非 GPU 执行证明'));
assert(!text.includes('已执行 CUDA'));
assert(taskActions(running).includes('data-cancel="fixture"'));
running.status='succeeded'; running.compute_info={research:true,selected_device:'cuda:1',cuda_work:true,
  model_inference_rows:64,model_inference_batches:2};
text=taskDetail(running);
assert(text.includes('CUDA MLP 推理') && !text.includes('当前阶段'));
global.sessionStorage={getItem:()=>null};
global.fetch=async()=>({ok:true,status:200,text:async()=>JSON.stringify({available:true,requested_device:'mps',selected_device:'mps',
  devices:[{type:'mps',device:'mps',index:null,name:'Apple MPS',total_memory_bytes:null}]})});
refreshCompute('data','mps').then(()=>{
  for (const prefix of ['analysis','scan','backtest','data']) {
    assert.equal(element(`${prefix}-device`).value,'mps');
    assert(element(`${prefix}-device`).innerHTML.includes('value="mps"'));
    assert(!element(`${prefix}-device`).innerHTML.includes('value="cuda:0"'));
  }
  assert(element('compute-devices').textContent.includes('float32 模型训练与推理'));
}).catch((error)=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run([node, "-e", script, str(source_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
