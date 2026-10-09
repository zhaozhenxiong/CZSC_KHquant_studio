'use strict';
const assert = require('node:assert/strict');
const {build,periodKey,describe} = require('../static/js/chart_markers.js');
const dailyBars = ['2026-01-02','2026-01-05','2026-01-09','2026-01-12','2026-01-16','2026-01-19','2026-02-02'].map((time) => ({time}));
const events = [{time:'2026-01-05',available_at:'2026-01-05T15:00:00+08:00',action:'BUY'},
                {time:'2026-01-09',available_at:'2026-01-09T15:00:00+08:00',action:'SELL'},
                {time:'2025-12-31',available_at:'2025-12-31T15:00:00+08:00',action:'BUY'},
                {time:'2026-02-03',available_at:'2026-02-03T15:00:00+08:00',action:'BUY'}];
const trades = [{date:'2026-01-05',symbol:'000001.SZ',action:'BUY'},
                {date:'2026-01-09',symbol:'000001.SZ',action:'SELL'},
                {date:'2026-01-09',symbol:'600000.SH',action:'BUY'}];
const common = {dailyBars,events,trades,symbol:'000001.SZ',start:'2026-01-01',asOf:'2026-02-02',up:'red',down:'green'};
const daily = build({...common,bars:dailyBars,frequency:'日线'});
assert.equal(daily.intentCount,2); assert.equal(daily.tradeCount,2);
assert.deepEqual(daily.markers.map((marker) => marker.shape),['arrowUp','circle','arrowDown','square']);
assert.deepEqual(daily.markers.map((marker) => marker.color),['red','red','green','green']);
assert.deepEqual(daily.markers.map((marker) => marker.text),['意买','成买','意卖','成卖']);
const weekly = build({...common,bars:[{time:'2026-01-09'},{time:'2026-01-16'}],frequency:'周线'});
assert.equal(weekly.markers.length,4);
assert(weekly.mappings.every((mapping) => mapping.bar_date === '2026-01-09' && mapping.closed_at === '2026-01-12'));
assert.equal(weekly.markers[0].text,'意买');
assert(describe(weekly.mappings,'2026-01-09','周线').includes('买入意图 2026-01-05 · 周线闭合 2026-01-12'));
assert(describe(weekly.mappings,'2026-01-09','周线').includes('卖出成交 2026-01-09'));
assert(!describe(weekly.mappings,null,'周线').includes('2026-01-05'));
assert.equal(describe(weekly.mappings,'2026-01-16','周线'),'2026-01-16 · 该K线无买卖标记');
const monthly = build({...common,bars:[{time:'2026-01-30'}],frequency:'月线'});
assert.equal(monthly.markers.length,4);
assert(monthly.mappings.every((mapping) => mapping.closed_at === '2026-02-02'));
const unfinished = build({...common,dailyBars:dailyBars.filter((bar) => bar.time <= '2026-01-30'),asOf:'2026-01-30',bars:[{time:'2026-01-30'}],frequency:'月线'});
assert.equal(unfinished.markers.length,0); assert.equal(unfinished.deferredCount,4);
const visibleForming = build({...common,bars:[{time:'2026-01-30',is_closed:false,bucket_time:'2026-01-31'}],frequency:'月线'});
assert.equal(visibleForming.markers.length,0); assert.equal(visibleForming.deferredCount,4);
const closedAndForming = build({...common,bars:[{time:'2026-01-30',is_closed:true},{time:'2026-02-02',is_closed:false,bucket_time:'2026-02-28'}],frequency:'月线'});
assert.equal(closedAndForming.markers.length,4);
assert(closedAndForming.mappings.every((mapping) => mapping.bar_date === '2026-01-30'));
const replay = build({...common,dailyBars:dailyBars.filter((bar) => bar.time <= '2026-01-09'),asOf:'2026-01-09',bars:[{time:'2026-01-09'}],frequency:'周线'});
assert.equal(replay.markers.length,0);
assert.equal(periodKey('2025-12-31','周线'),'2026-01');
console.log(JSON.stringify({status:'passed',checks:'colors/shapes, source dates, completed buckets, replay, start cutoff, symbol isolation'}));
