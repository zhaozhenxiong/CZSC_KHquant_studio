import shutil
import subprocess
from pathlib import Path

import pytest


def test_chart_range_and_profile_stay_aligned_without_changing_information_cutoff():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for chart interaction checks')
    path = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js/workbench.js'
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const elements=new Map();
function newElement(tag='div') {
  return {tag,checked:true,hidden:false,disabled:false,clientWidth:145,clientHeight:450,value:'',textContent:'',innerHTML:'',children:[],attributes:{},
    listeners:{},addEventListener(key,callback){this.listeners[key]=callback;},querySelector:()=>newElement(),replaceChildren(){this.children=[];},append(child){this.children.push(child);},setAttribute(key,value){this.attributes[key]=value;}};
}
function element(id){if(!elements.has(id))elements.set(id,newElement());return elements.get(id);}
global.document={getElementById:element,documentElement:{},createElementNS:(_,tag)=>newElement(tag)};
global.getComputedStyle=()=>({getPropertyValue:()=> '#123456'});
global.ResizeObserver=class{observe(){}};
const frames=[];global.requestAnimationFrame=callback=>{frames.push(callback);return frames.length;};
global.KHQuantMarkers=require(process.argv[2]);
let range={from:0,to:9},rangeChanged,rangeWrites=0;
const scale={subscribeVisibleLogicalRangeChange(fn){rangeChanged=fn;},getVisibleLogicalRange:()=>range,
  setVisibleLogicalRange(value){range=value;rangeWrites++;rangeChanged();},fitContent(){this.setVisibleLogicalRange({from:0,to:9});},
  height:()=>30,timeToCoordinate:()=>50};
let coordinateShift=0;
function series(){return{setData(){},setMarkers(){},applyOptions(){},priceScale:()=>({applyOptions(){}}),priceToCoordinate:price=>350-(price-8)*60+coordinateShift};}
const chart={addCandlestickSeries:series,addHistogramSeries:series,addLineSeries:series,removeSeries(){},resize(){},
  subscribeCrosshairMove(){},subscribeClick(){},timeScale:()=>scale,priceScale:()=>({width:()=>50})};
global.LightweightCharts={createChart:()=>chart,LineStyle:{Solid:0,Dashed:2}};global.window={LightweightCharts};
const code=fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);',
  'globalThis.rangeTest={state,renderAnalysis,syncChartRange,applyChartRange,zoomChart,drawChipProfile,resizeCharts};');
vm.runInThisContext(code);
const{state,renderAnalysis,applyChartRange,zoomChart,drawChipProfile}=rangeTest;
const dates=['2026-01-02','2026-01-05','2026-01-06','2026-01-07','2026-01-08','2026-01-09','2026-01-12','2026-01-13','2026-01-14','2026-01-15'];
const bars=dates.map(time=>({time,open:10,high:12,low:9,close:11,volume:100,is_closed:true}));
const profile={status:'available',as_of:dates.at(-1),window_bars:120,chip70_low:9.5,chip70_high:11.5,chip50:10.5,
  bins:[{price_low:9,price_high:10,mass_fraction:.2},{price_low:10,price_high:11,mass_fraction:.8}]};
element('analysis-chart').clientWidth=800;
element('view-analysis').hidden=false;
state.query={symbol:'600027.SH',as_of:dates.at(-1)};
state.analysis={symbol:'600027.SH',frequencies:{'日线':{bars,indicators:{rows:[],chip_profile:profile}}},events:[]};
renderAnalysis(false);
assert.equal(element('chart-range-start').value,dates[0]);assert.equal(element('chart-range-end').value,dates.at(-1));
assert(element('chart-range-summary').textContent.includes('10 根'));
element('chart-range-start').value='2026-01-03';element('chart-range-end').value='2026-01-08';applyChartRange();
assert.deepEqual(range,{from:.5,to:4.5});assert.equal(element('chart-range-start').value,'2026-01-05');
assert.equal(element('chart-range-end').value,'2026-01-08');assert.equal(state.query.as_of,'2026-01-15');
element('chart-range-start').value='2026-01-09';element('chart-range-end').value='2026-01-05';
const writes=rangeWrites;applyChartRange();assert.equal(rangeWrites,writes);
assert(element('notice').textContent.includes('开始日期不能晚于'));
scale.setVisibleLogicalRange({from:-.5,to:9.5});zoomChart(.7);assert.equal(range.to-range.from,7);
zoomChart(100);assert.deepEqual(range,{from:-.5,to:9.5});
scale.setVisibleLogicalRange({from:6,to:12});zoomChart(.7);assert(range.from>=-.5 && range.to<=9.5);
const shapes=element('chip-profile-overlay').children.filter(item=>item.tag==='rect');assert.equal(shapes.length,2);
assert.equal(Number(shapes[0].attributes.y),230);assert.equal(Number(shapes[1].attributes.y),170);
assert.equal(Number(shapes[0].attributes.width)/Number(shapes[1].attributes.width),.25);
assert(shapes[0].children[0].textContent.includes('20%'));
assert(element('chip-profile-caption').textContent.includes('非真实持仓筹码'));
const before=element('chip-profile-overlay').children;
element('analysis-chart').listeners.pointermove();element('analysis-chart').listeners.pointerup();
assert.equal(frames.length,1);frames.shift()();assert.notEqual(element('chip-profile-overlay').children,before);
// Native chart geometry is recomputed during its scheduled paint after resize.
chart.resize=()=>{frames.push(()=>{coordinateShift=40;});};
rangeTest.resizeCharts();assert.equal(frames.length,2);frames.shift()();frames.shift()();
assert.equal(Number(element('chip-profile-overlay').children.find(item=>item.tag==='rect').attributes.y),270);
chart.resize=()=>{};
profile.status='insufficient_history';drawChipProfile();
assert(!element('chip-profile-overlay').children.some(item=>item.tag==='rect'));
assert(element('chip-profile-overlay').children.some(item=>item.textContent.includes('历史不足')));
element('layer-chips').checked=false;renderAnalysis(false);assert(element('chip-profile').hidden);
state.analysis=null;renderAnalysis(false);assert(element('chart-range-apply').disabled);
assert.equal(element('chart-range-start').value,'');assert.equal(element('chip-profile-overlay').children.length,0);
'''
    result = subprocess.run([node, '-e', script, str(path), str(path.with_name('chart_markers.js'))],
                            capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
