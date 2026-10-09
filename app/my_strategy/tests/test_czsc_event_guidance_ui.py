"""Check the shipped event prices and latest-session guidance without future inference."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_event_reference_prices_and_next_session_guidance():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped event guidance checks")
    path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id,{textContent:'',innerHTML:''});
  return elements.get(id);
}
global.document = {getElementById:element};
const source = fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);',
  'globalThis.eventTest={state,renderEvents,referencePriceText,nextSessionGuidance};');
vm.runInThisContext(source);
const {state,renderEvents,referencePriceText,nextSessionGuidance}=eventTest;
const buy = {time:'2026-07-29',available_at:'2026-07-29T15:00:00+08:00',action:'BUY',
  reference_price:4.73,reference_price_date:'2026-07-29',reference_price_source:'signal_day_close',
  reason:'开仓条件',signal:'确认笔向上'};
const sell = {...buy,time:'2026-09-29',available_at:'2026-09-29T15:00:00+08:00',action:'SELL',
  reference_price:4.98,reference_price_date:'2026-09-29',reason:'退出条件'};
state.query={symbol:'600027.SH',as_of:'2026-09-30'};
state.analysis={events:[buy,sell],data_end:'2026-09-30',
  frequencies:{'日线':{bars:[{time:'2026-09-30',close:99}]}},
  next_session_decision:{action:'WAIT',next_market_session:'2026-10-09',reference_close:4.95,
    reference_price_date:'2026-09-30',basis:'当前无新的开仓条件',
    scope:'原生 Position 理论模拟持仓，非我的实盘持仓'}};
renderEvents();
assert.equal(element('event-count').textContent,'2');
const events=element('event-list').innerHTML;
assert(events.includes('买入参考价 4.73 元（2026-07-29 日收盘，非成交价）'));
assert(events.includes('卖出参考价 4.98 元（2026-09-29 日收盘，非成交价）'));
assert(events.indexOf('卖出参考价') < events.indexOf('买入参考价'));
assert(!events.includes('99 元'));
assert(events.includes('可用 2026-09-29T15:00:00+08:00'));
const guidance=element('event-context').innerHTML;
assert(guidance.includes('下一交易日 2026-10-09'));
assert(guidance.includes('等待'));
assert(guidance.includes('收盘参考价 4.95 元（2026-09-30 日收盘，非成交价）'));
assert(guidance.includes('条件：当前无新的开仓条件'));
assert(guidance.includes('原生 Position 理论模拟持仓，非我的实盘持仓'));
assert(!guidance.includes('卖出参考价'));
assert(!guidance.includes('2026-09-29'));

const historicalDecision={...state.analysis.next_session_decision,action:'BUY',next_market_session:'2026-10-08'};
for (const cutoff of ['2026-10-08','2026-10-09T15:00:00+08:00']) {
  const text=nextSessionGuidance(historicalDecision,cutoff);
  assert(text.includes('当前判断暂不可用：行情仅截至 2026-09-30'));
  assert(text.includes('未覆盖请求截止 '+cutoff.slice(0,10)+'；下列为历史规则判断。'));
  assert(text.includes('历史规则判断（信号日 2026-09-30，对应交易日 2026-10-08）'));
  assert(text.includes('买入参考价 4.95 元'));
  assert(!text.includes('下一交易日 2026-10-08'));
}
assert(!nextSessionGuidance(historicalDecision,'2026-09-30').includes('历史规则判断'));
assert(!nextSessionGuidance(historicalDecision,'2026-10-07').includes('历史规则判断'));
assert(!nextSessionGuidance({...historicalDecision,reference_price_date:'2026-10-08'},'2026-10-08').includes('历史规则判断'));
state.query.as_of='2026-10-09'; state.analysis.next_session_decision=historicalDecision; renderEvents();
assert(element('event-context').innerHTML.includes('未覆盖请求截止 2026-10-09'));
state.query.as_of=null; state.analysis.as_of='2026-10-08'; renderEvents();
assert(element('event-context').innerHTML.includes('未覆盖请求截止 2026-10-08'));
state.query.as_of='2026-09-30'; state.analysis.as_of='2026-09-30';
state.analysis.next_session_decision={...historicalDecision,action:'WAIT',next_market_session:'2026-10-09'};

for (const [action,label,priceLabel] of [['BUY','买入','买入参考价'],['SELL','卖出','卖出参考价'],
                                      ['HOLD','持有','收盘参考价'],['WAIT','等待','收盘参考价']]) {
  const text=nextSessionGuidance({...state.analysis.next_session_decision,action});
  assert(text.includes(label)); assert(text.includes(priceLabel+' 4.95 元'));
}
const unverified=nextSessionGuidance({...state.analysis.next_session_decision,next_market_session:null});
assert(unverified.includes('下一交易日（日期待核验）'));
assert(!unverified.includes('2026-10-01'));
for (const value of [null,undefined,'',0,-1,NaN,Infinity,'not a price']) {
  assert.equal(referencePriceText(value,'2026-09-30','买入参考价'),'买入参考价暂不可用');
  const text=nextSessionGuidance({...state.analysis.next_session_decision,reference_close:value});
  assert(text.includes('收盘参考价暂不可用')); assert(!text.includes('0 元'));
}
assert.equal(referencePriceText(4.87,null,'卖出参考价'),'卖出参考价 4.87 元（信号日收盘，非成交价）');
state.analysis.events=[{...buy,reference_price:null}]; renderEvents();
assert(element('event-list').innerHTML.includes('买入参考价暂不可用'));
assert(!element('event-list').innerHTML.includes('4.95 元'));
state.analysis.next_session_decision=null; renderEvents();
assert.equal(element('event-context').innerHTML,'下一交易日判断暂不可用，请重新分析。');
assert(!element('event-context').innerHTML.includes('买入'));
assert.equal(nextSessionGuidance({action:'UNKNOWN'}),'下一交易日判断暂不可用，请重新分析。');
state.analysis.next_session_decision={action:'BUY',reference_close:4.95,
  next_market_session:'2026-10-09',basis:'条件 <script>alert(1)</script>',scope:'持仓 <b>x</b>'};
renderEvents();
assert(element('event-context').innerHTML.includes('&lt;script&gt;'));
assert(!element('event-context').innerHTML.includes('<script>'));
assert(element('event-context').innerHTML.includes('&lt;b&gt;x&lt;/b&gt;'));
state.analysis.events=[]; renderEvents();
assert(element('event-list').innerHTML.includes('该时点没有规则事件'));
assert(element('event-context').innerHTML.includes('买入参考价 4.95 元'));
"""
    result = subprocess.run([node, "-e", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_failed_analysis_clears_prior_buy_guidance():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped failed-analysis checks")
    path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id,{textContent:'',innerHTML:'',hidden:false,
    querySelector:(selector) => element(id+selector),replaceChildren() { this.innerHTML=''; }});
  return elements.get(id);
}
global.document={getElementById:element};
global.sessionStorage={getItem:() => null};
global.fetch=async () => { throw new Error('network unavailable'); };
global.KHQuantMarkers={describe:() => ''};
const source=fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);',
  'globalThis.failureTest={state,renderEvents,analyze};');
vm.runInThisContext(source);
const {state,renderEvents,analyze}=failureTest;
(async () => {
  state.analysis={events:[{action:'BUY',time:'2026-09-30',reference_price:4.95}],
    next_session_decision:{action:'BUY',reference_close:4.95,reference_price_date:'2026-09-30',
      next_market_session:'2026-10-08',basis:'最新收盘触发开多'}};
  renderEvents();
  assert(element('event-context').innerHTML.includes('买入参考价 4.95 元'));
  assert(element('event-list').innerHTML.includes('买入参考价 4.95 元'));
  await analyze({symbol:'600027.SH',as_of:'2026-09-30'});
  assert.equal(state.analysis,null);
  assert.equal(element('event-context').innerHTML,'下一交易日判断暂不可用，请重新分析。');
  assert(!element('event-context').innerHTML.includes('买入'));
  assert(!element('event-context').innerHTML.includes('4.95'));
  assert(element('event-list').innerHTML.includes('分析未完成'));
  assert(!element('event-list').innerHTML.includes('买入参考价'));
  assert.equal(element('event-count').textContent,'0');
  assert.equal(element('analysis-state').textContent,'分析失败');
  assert.equal(element('page-errorspan').textContent,'network unavailable');
  assert.equal(element('analysis-form[type=submit]').disabled,false);
})().catch((error) => { console.error(error); process.exitCode=1; });
"""
    result = subprocess.run([node, "-e", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
