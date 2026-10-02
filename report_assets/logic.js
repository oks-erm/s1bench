/* Pure offline calculations; tested against benchmark.py. No network APIs. */
const ReportLogic = (() => {
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const primary = r => Boolean(r.scored === undefined ? true : r.scored) && (r.phase === undefined ? 'primary' : r.phase) === 'primary';
  function percentile(values, p) {
    const a = values.filter(finite).sort((x,y) => x-y);
    if (!a.length) return null;
    const pos = (a.length-1)*p, lo = Math.floor(pos), hi = Math.ceil(pos);
    return a[lo] + (a[hi]-a[lo])*(pos-lo);
  }
  function fallback(records, summaries, dataset, task, baseline, labels) {
    const complete = new Map(summaries.filter(r => r.dataset===dataset && r.task===task && r.status==='complete').map(r=>[r.model,r]));
    const groups = new Map();
    for (const r of records.filter(r=>primary(r) && r.dataset===dataset && r.task===task)) {
      if (!groups.has(r.model)) groups.set(r.model,new Map());
      groups.get(r.model).set(r.id,r);
    }
    const reference = groups.get(baseline), results=[];
    if (!reference || !reference.size || !complete.has(baseline)) return results;
    for (const [alias, predictions] of groups) {
      if (alias===baseline || !complete.has(alias) || predictions.size!==reference.size ||
          [...predictions.keys()].some(id=>!reference.has(id)) || complete.get(alias).currency!==complete.get(baseline).currency) continue;
      let correct=0, critical=0, used=0; const costs=[], times=[];
      for (const [id,fast] of predictions) {
        const use = Boolean(fast.api_error) || !fast.valid || (fast.type==='choice' && labels.has(fast.decision));
        const final = use ? reference.get(id) : fast;
        correct += Number(Boolean(final.correct)); critical += Number(Boolean(final.critical) && !final.correct); used += Number(use);
        const first=fast._api_cost, second=use ? reference.get(id)._api_cost : 0;
        if (finite(first) && finite(second)) costs.push(first+second);
        const a=fast.latency_s, b=use ? reference.get(id).latency_s : 0;
        if (finite(a) && finite(b)) times.push(a+b);
      }
      const n=predictions.size;
      results.push({model:alias,cases:n,success_rate:correct/n,fallback_rate:used/n,critical_errors:critical,
        api_per_1k:costs.length ? costs.reduce((a,b)=>a+b,0)/costs.length*1000 : null,
        cost_coverage:costs.length/n,currency:complete.get(alias).currency,
        simulated_p95_ms:times.length===n ? percentile(times,.95)*1000 : null});
    }
    return results;
  }
  function csv(rows) {
    const fields=[...new Set(rows.flatMap(r=>Object.keys(r)))];
    function cell(v) {
      let s=v==null ? '' : typeof v==='object' ? JSON.stringify(v) : String(v);
      if (typeof v==='string' && /^[=+@\-\t\r]/.test(s)) s="'"+s;
      return '"'+s.replaceAll('"','""')+'"';
    }
    return '\ufeff'+[fields.map(cell).join(','),...rows.map(r=>fields.map(k=>cell(r[k])).join(','))].join('\r\n');
  }
  return {finite,primary,percentile,fallback,csv};
})();
if (typeof module !== 'undefined') module.exports=ReportLogic;
