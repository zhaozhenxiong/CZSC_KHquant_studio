"""Verify the actual shipped shell and point-in-time marker transformation."""
from pathlib import Path
import shutil
import subprocess

import pytest

from my_strategy.web_dashboard.scripts.validate_czsc_workbench import validate
from my_strategy.web_dashboard.api import AnalysisRequest


def test_new_analysis_default_and_explicit_history():
    assert str(AnalysisRequest(symbol="000001.SZ").start) == "2026-01-01"
    assert str(AnalysisRequest(symbol="000001.SZ", start="2024-01-01", end="2024-09-30").start) == "2024-01-01"
    assert AnalysisRequest(symbol="000001.SZ", start=None, as_of="2025-09-30").start is None


def test_shipped_shell_defaults_and_markers():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for the shipped JavaScript validation")
    assert validate(node)["status"] == "passed"
    script = Path(__file__).resolve().parents[1] / "web_dashboard/scripts/test_chart_markers.js"
    result = subprocess.run([node, str(script)], check=True, capture_output=True, text=True)
    assert '"status":"passed"' in result.stdout


def test_shipped_overlay_rendering_and_direction_colors():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped chart rendering checks")
    path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const elements = new Map(), liveLines = new Set();
function element(id) {
  if (!elements.has(id)) elements.set(id, {checked:true,hidden:false,clientWidth:800,clientHeight:450,
    _text:'',_html:'',children:[],attributes:{},
    set textContent(value) { this._text=value; this._html=''; }, get textContent() { return this._text; },
    set innerHTML(value) { this._html=value; }, get innerHTML() { return this._html; },
    addEventListener() {},querySelector:() => element(id+'-child'), replaceChildren() { this.children=[]; this._html=''; this._text=''; },
    setAttribute(key,value) { this.attributes[key]=value; }, append(child) { this.children.push(child); }});
  return elements.get(id);
}
global.document = {getElementById:element,documentElement:{},createElementNS:() => element('svg-'+elements.size)};
global.getComputedStyle = () => ({getPropertyValue:(key) => ({'--up':'red','--down':'green','--muted':'gray'}[key] || '#123456')});
global.ResizeObserver = class { observe() {} };
global.requestAnimationFrame = () => 1;
global.KHQuantMarkers = require(process.argv[2]);
require(require('path').join(require('path').dirname(process.argv[1]),'wyckoff_markers.js'));
let candles, crosshair;
function series() { return {data:[],markers:[],setData(rows) { this.data=rows; },setMarkers(rows) { this.markers=rows; },
  applyOptions() {},priceScale:() => ({applyOptions() {}}),priceToCoordinate:(value) => 150-value}; }
const chart = {addCandlestickSeries:() => (candles=series()),addHistogramSeries:series,
  addLineSeries(options) { const line=series(); line.options=options; liveLines.add(line); return line; },
  removeSeries:(line) => liveLines.delete(line),subscribeCrosshairMove:(fn) => { crosshair=fn; },subscribeClick() {},
  resize() {},timeScale:() => ({subscribeVisibleLogicalRangeChange() {},fitContent() {},getVisibleLogicalRange:() => ({from:0,to:1}),height:() => 20,timeToCoordinate:(time) => time.endsWith('02') ? 30 : 80}),
  priceScale:() => ({width:() => 50})};
global.LightweightCharts = {createChart:() => chart,LineStyle:{Solid:0,Dashed:2}};
global.window = {LightweightCharts};
const source = fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);',
  'globalThis.overlayTest = {state,renderAnalysis,renderIndicatorReadout,emptyChart};');
vm.runInThisContext(source);
const {state,renderAnalysis,renderIndicatorReadout,emptyChart} = overlayTest;
const first = {time:'2026-01-02',is_closed:true,ma5:null,ma10:null,ma20:null,ma60:null,boll_mid:null,boll_upper:null,boll_lower:null,
  chip70_low:null,chip70_high:null,chip50:null,chip_as_of:null,ma_alignment:'insufficient',ma20_direction:'insufficient',ma60_direction:'insufficient'};
const last = {...first,time:'2026-01-05',is_closed:false,ma5:10.25,ma10:10,ma20:9,ma60:8,
  boll_mid:9,boll_upper:11,boll_lower:7,chip70_low:8,chip70_high:12,chip50:10,chip_as_of:'2026-01-05',
  ma_alignment:'bullish',ma20_direction:'rising',ma60_direction:'falling'};
const bars = [first,last].map(({time,is_closed}) => ({time,is_closed,open:9,high:12,low:8,close:10,volume:100}));
const daily = {bars,indicators:{rows:[first,last]},pens:[],
  fractals:[{time:first.time,kind:'top',is_confirmed:true},{time:last.time,kind:'bottom',is_confirmed:false}],
  divergences:[{time:first.time,price:12,kind:'top',label:'类趋势顶背驰',confirmed_at:'2026-01-02T15:00:00+08:00',available_at:'2026-01-05T15:00:00+08:00',pen_anchors:[[first.time,last.time,8,12]]},
               {time:last.time,price:8,kind:'bottom',label:'aAb式底背驰',confirmed_at:'2026-01-05T15:00:00+08:00',available_at:'2026-01-05T15:00:00+08:00'}],
  zones:['up','down','range'].map((direction) => ({direction,start_time:first.time,end_time:last.time,low:8,high:10,is_confirmed:true}))};
state.query = {symbol:'000001.SZ',as_of:last.time};
state.analysis = {symbol:'000001.SZ',frequencies:{'日线':daily,'月线':{bars:[bars[0]],indicators:{rows:[first]}}},events:[]};
renderAnalysis(false);
assert.equal(liveLines.size,8);
for (const line of liveLines) { assert.equal(line.data[0].time,first.time); assert(!('value' in line.data[0])); }
assert(element('indicator-ma').innerHTML.includes('MA5 10.25'));
assert(element('indicator-ma').innerHTML.includes('多头排列 · MA20上行 · MA60下行'));
assert(element('indicator-context').textContent.includes('形成中，仅供展示'));
assert(element('indicator-chips').textContent.includes('中位价 10 · 日线截至 2026-01-05'));
assert.deepEqual(candles.markers.filter((marker) => !marker.text.includes('背驰')).map(({color,shape}) => [color,shape]),[['green','arrowDown'],['red','circle']]);
assert.deepEqual(candles.markers.filter((marker) => marker.text.includes('背驰')).map(({color,position,shape,text}) => [color,position,shape,text]),
  [['green','aboveBar','arrowDown','顶背驰参考'],['red','belowBar','arrowUp','底背驰参考']]);
assert(element('divergence-list').innerHTML.includes('类趋势顶背驰'));
assert(element('divergence-list').innerHTML.includes('末笔确认 2026-01-02T15:00:00+08:00'));
assert(element('divergence-list').innerHTML.includes('首次可见 2026-01-05T15:00:00+08:00'));
assert(element('divergence-list').innerHTML.includes('2026-01-02 → 2026-01-05'));
assert.deepEqual(element('zone-overlay').children[0].children.map((rect) => rect.attributes.stroke),['red','green','gray']);
element('layer-divergences').checked=false; renderAnalysis(false);
assert(!candles.markers.some((marker) => marker.text.includes('背驰')));
assert(element('divergence-details').hidden); assert.equal(element('divergence-list').innerHTML,'');
element('layer-divergences').checked=true;
crosshair({time:first.time,seriesData:new Map([[candles,bars[0]]])});
assert(element('indicator-ma').innerHTML.includes('MA5 —'));
assert(!element('indicator-ma').innerHTML.includes('10.25'));
assert(element('indicator-context').textContent.includes('所指K线 2026-01-02'));
crosshair({seriesData:new Map()});
assert(element('indicator-context').textContent.includes('最新K线 2026-01-05'));
element('layer-ma').checked=false; renderAnalysis(false);
assert.equal(liveLines.size,5); assert(element('indicator-ma').hidden);
assert([...liveLines].some((line) => line.data.at(-1).value === last.boll_mid));
for (const name of ['boll','chips']) element('layer-'+name).checked=false;
renderAnalysis(false); assert.equal(liveLines.size,0); assert(element('indicator-readout').hidden);
for (const name of ['ma','boll','chips']) element('layer-'+name).checked=true;
state.frequency='月线'; renderAnalysis(false);
assert.equal(liveLines.size,0); assert(element('indicator-ma').innerHTML.includes('MA60 —'));
assert(!element('indicator-chips').textContent.includes('2026-01-05'));
state.frequency='日线'; renderAnalysis(false); assert.equal(liveLines.size,8);
emptyChart('空结果','没有行情'); assert.equal(liveLines.size,0); assert(element('indicator-readout').hidden);
assert(element('divergence-details').hidden); assert.equal(element('divergence-list').innerHTML,'');
for (const name of ['context','ma','boll','chips']) assert.equal(element('indicator-'+name).textContent,'');
state.analysis=null; renderAnalysis(false); assert.equal(liveLines.size,0);
"""
    markers = path.with_name("chart_markers.js")
    result = subprocess.run([node, "-e", script, str(path), str(markers)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_research_checkpoint_preflight_and_failed_backtest_empty_state():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped backtest UI checks")
    path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id,{value:'',hidden:false,disabled:false,innerHTML:'',textContent:'',querySelector:(selector) => element(id+selector)});
  return elements.get(id);
}
global.document = {getElementById:element};
const source = fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);',
  'globalThis.backtestTest={state,updateBacktestCheckpoint,researchSpec,renderBacktest};');
vm.runInThisContext(source);
const {state,updateBacktestCheckpoint,researchSpec,renderBacktest}=backtestTest;
state.research={calendar_run_id:'verified',models:[{run_id:'train',checkpoints:[
  {name:'production',available_at:'2026-09-30',train_label_end:'2026-06-30'},
  {name:'2025Q4',available_at:'2025-09-30',train_label_end:'2025-06-30'}]}]};
element('backtest-mode').value='research'; element('backtest-model').value='train';
element('backtest-model-policy').value='pinned'; element('backtest-usage-mode').value='historical';
element('backtest-start').value='2026-01-01'; element('backtest-checkpoint').value='production';
updateBacktestCheckpoint();
assert(element('backtest-checkpoint').innerHTML.includes('2025Q4 历史检查点 · 可用 2025-09-30'));
assert(element('backtest-checkpoint-help').textContent.includes('当前开始日 2026-01-01 早于模型可用日'));
assert.throws(() => researchSpec('backtest'),/不能用于 2026-01-01/);
element('backtest-checkpoint').value='2025Q4';
assert.deepEqual(researchSpec('backtest'),{research:true,usage_mode:'historical',model_policy:'pinned',entry_policy:'fresh',model_run_id:'train',model_fold:'2025Q4',calendar_run_id:'verified'});
element('scan-mode').value='research'; element('scan-model').value='train';
assert.equal(researchSpec('scan').model_fold,undefined);
element('backtest-model').value=''; element('backtest-model-policy').value='auto'; updateBacktestCheckpoint();
assert(element('backtest-checkpoint').disabled);
assert.deepEqual(researchSpec('backtest'),{research:true,usage_mode:'historical',model_policy:'auto',entry_policy:'fresh',calendar_run_id:'verified'});
state.tracked.backtest='job'; state.tasks=[{job_id:'job',status:'failed',error:'模型 <too late>'}];
renderBacktest();
assert(element('backtest-metrics').innerHTML.includes('本次回测失败：模型 &lt;too late&gt;'));
assert(!element('backtest-metrics').innerHTML.includes('等待'));
assert.equal(element('equity-emptystrong').textContent,'回测失败');
assert(element('trade-rows').innerHTML.includes('本次未生成可展示成交账本'));
state.tasks[0].status='cancelled'; renderBacktest();
assert(element('backtest-metrics').innerHTML.includes('已取消'));
state.tasks[0].status='running'; renderBacktest();
assert(element('backtest-metrics').innerHTML.includes('回测正在运行'));
state.tasks=[]; renderBacktest();
assert(element('backtest-metrics').innerHTML.includes('尚未运行回测'));
"""
    result = subprocess.run([node, "-e", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
