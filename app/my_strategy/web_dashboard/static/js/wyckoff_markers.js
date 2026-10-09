/* Daily causal price/volume evidence, independent of intentions and fills. */
(() => {
  'use strict';
  const build = ({events = [], bars = [], frequency = '日线', asOf = ''}) => {
    if (frequency !== '日线') return [];
    const days = new Set(bars.map((bar) => String(bar.time).slice(0, 10)));
    const names = new Set(['Spring','Test','SOS','LPS','Upthrust','SOW','LPSY']);
    return events.filter((event) => {
      const observed = String(event.observed_at || '').slice(0, 10);
      const available = String(event.available_at || '').slice(0, 10);
      return names.has(event.event) && observed && available && observed <= available
        && (!asOf || available <= asOf) && days.has(available);
    }).map((event) => {
      const supply = ['Upthrust','SOW','LPSY'].includes(event.event);
      return {time:String(event.available_at).slice(0,10), position:supply ? 'aboveBar' : 'belowBar',
        color:supply ? '#3b82f6' : '#d97706', shape:'square', size:.55,
        text:`W ${event.event}${event.buy_candidate ? '候选' : supply ? '供应' : '待测试'}`};
    });
  };
  globalThis.KHQuantWyckoffMarkers = {build};
})();
