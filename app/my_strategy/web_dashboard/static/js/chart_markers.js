/* Map dated intentions and actual executions only onto observable closed bars. */
((root) => {
  'use strict';
  const day = (value) => String(value || '').slice(0,10);
  function periodKey(value, frequency) {
    const text = day(value);
    if (frequency === '日线') return text;
    if (frequency === '月线') return text.slice(0,7);
    const date = new Date(`${text}T00:00:00Z`);
    date.setUTCDate(date.getUTCDate() + 4 - (date.getUTCDay() || 7));
    const year = date.getUTCFullYear();
    const week = Math.ceil(((date - Date.UTC(year,0,1)) / 86400000 + 1) / 7);
    return `${year}-${String(week).padStart(2,'0')}`;
  }
  function build({bars, dailyBars, frequency, events = [], trades = [], symbol, start = '', asOf, up, down}) {
    const markers = [], mappings = [];
    const byPeriod = new Map(bars.filter((bar) => bar.is_closed !== false).map((bar) => [periodKey(bar.time,frequency),bar.time]));
    let intentCount = 0, tradeCount = 0, deferredCount = 0;
    const add = (row, execution) => {
      const original = day(execution ? row.date : row.available_at || row.time);
      if (!original || original < start || original > asOf) return;
      const buying = /buy|open|买|开/i.test(row.action || '');
      const selling = /sell|close|卖|平/i.test(row.action || '');
      if (!buying && !selling) return;
      const key = periodKey(original,frequency), time = byPeriod.get(key);
      const closedAt = frequency === '日线' ? time : dailyBars.find((bar) => periodKey(bar.time,frequency) > key)?.time;
      if (!time || time < original || !closedAt || closedAt > asOf) { deferredCount++; return; }
      const label = `${execution ? '成' : '意'}${buying ? '买' : '卖'}`;
      markers.push({time, position:buying ? 'belowBar' : 'aboveBar', color:buying ? up : down,
                    shape:execution ? buying ? 'circle' : 'square' : buying ? 'arrowUp' : 'arrowDown',
                    text:label, size:execution ? 1.2 : 1});
      mappings.push({kind:execution ? 'execution' : 'intention', action:buying ? 'BUY' : 'SELL', original_date:original, bar_date:time, closed_at:closedAt});
      execution ? tradeCount++ : intentCount++;
    };
    events.forEach((event) => add(event,false));
    trades.filter((trade) => trade.symbol === symbol).forEach((trade) => add(trade,true));
    markers.sort((a,b) => a.time.localeCompare(b.time));
    return {markers,mappings,intentCount,tradeCount,deferredCount};
  }
  function describe(mappings, time, frequency) {
    if (!time) return '移到买卖标记所在K线，查看原始日期与周期闭合日期。';
    const items = mappings.filter((item) => item.bar_date === time);
    if (!items.length) return `${time} · 该K线无买卖标记`;
    return items.map((item) => `${item.action === 'BUY' ? '买入' : '卖出'}${item.kind === 'execution' ? '成交' : '意图'} ${item.original_date}${frequency === '日线' ? '' : ` · ${frequency}闭合 ${item.closed_at}`}`).join('；');
  }
  const api = {periodKey,build,describe};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.KHQuantMarkers = api;
})(globalThis);
