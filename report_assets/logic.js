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
  function fallback(records, summaries, dataset, task, baseline, labels, acceptancePolicy=null) {
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
        const use = Boolean(fast.api_error) || !fast.valid || (fast.type==='choice' && labels.has(fast.decision)) || (acceptancePolicy!==null && acceptance(fast,acceptancePolicy)!=='accepted');
        const final = use ? reference.get(id) : fast;
        correct += Number(Boolean(final.correct)); critical += Number('critical_error_choices' in final ? final.valid && !final.api_error && final.critical_error_choices.includes(final.decision) : Boolean(final.critical) && !final.correct); used += Number(use);
        const first=fast._api_cost, second=use ? reference.get(id)._api_cost : 0;
        if (finite(first) && finite(second)) costs.push(first+second);
        const a=fast.latency_s, b=use ? reference.get(id).latency_s : 0;
        if (finite(a) && finite(b)) times.push(a+b);
      }
      const n=predictions.size;
      results.push({model:alias,cases:n,success_rate:correct/n,fallback_rate:used/n,critical_errors:critical,
        api_per_1k:costs.length===n ? costs.reduce((a,b)=>a+b,0)/costs.length*1000 : null,
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
  const reviewLabels=['NONE','CLARIFY','ESCALATE','ABSTAIN','HUMAN','HUMAN_REVIEW'];
  const probability=x=>finite(x)&&x>=0&&x<=1;
  function acceptance(r,p={},business=false){
    const mode=p.mode??'label';
    if(!['label','probability','confidence'].includes(mode) || ['threshold','negative_threshold','positive_threshold'].some(k=>k in p&&!probability(p[k])) || (p.negative_threshold??.1)>=(p.positive_threshold??.9))throw Error('Invalid acceptance policy');
    if(r.api_error||!r.valid)return 'failed';
    if(r.type==='score'||(r.type==='choice'&&(p.review_labels??reviewLabels).includes(r.decision)))return 'reviewed';
    if(business&&(p.business_approved!==true||(p.ineligible_labels??[]).includes(r.decision)))return 'reviewed';
    if(mode==='label')return r.type==='choice'?'accepted':'reviewed';
    if(p.accept_none)return 'reviewed';
    if(r.type==='noul')return probability(r.value)&&(r.value<=(p.negative_threshold??.1)||r.value>=(p.positive_threshold??.9))?'accepted':'reviewed';
    const probs=r.probabilities,validVector=probs&&typeof probs==='object'&&!Array.isArray(probs)&&Object.keys(probs).length>0&&Object.values(probs).every(probability)&&Math.abs(Object.values(probs).reduce((a,b)=>a+b,0)-1)<=1e-5;
    const value=mode==='probability'?(validVector?probs[r.decision]:null):r.confidence;
    return probability(value)&&value>=(p.threshold??.9)?'accepted':'reviewed';
  }
  function acceptanceSummary(rows,p={},planned=rows.length,business=false){
    const accepted=rows.filter(r=>acceptance(r,p,business)==='accepted'),errors=accepted.filter(r=>!r.correct).length;
    const directional=p.risk_policy==='directional-choice-v1'||rows.some(r=>'critical_error_choices' in r),exposure=r=>directional?Boolean(r.critical_error_choices?.length):Boolean(r.critical),error=r=>directional?Boolean(r.valid&&!r.api_error&&r.critical_error_choices?.includes(r.decision)):Boolean(r.critical&&!r.correct);
    const exposed=accepted.filter(exposure),critical=accepted.filter(error).length;
    return {planned,accepted:accepted.length,reviewed:rows.filter(r=>acceptance(r,p,business)==='reviewed').length,
      failed:rows.filter(r=>acceptance(r,p,business)==='failed').length,unattempted:Math.max(0,planned-rows.length),
      coverage:planned?accepted.length/planned:null,review_rate:planned?(planned-accepted.length)/planned:null,
      accepted_errors:errors,accepted_accuracy:accepted.length?1-errors/accepted.length:null,
      accepted_error_rate:accepted.length?errors/accepted.length:null,accepted_families:new Set(accepted.map(r=>r.cluster_id??r.id)).size,
      accepted_critical_errors:critical,accepted_risk_exposures:exposed.length,
      risk_policy:directional?'directional-choice-v1':'legacy-critical-case-errors',
      accepted_critical_rate:exposed.length?critical/exposed.length:null,
      critical_errors_all_attempts:rows.filter(error).length,
      risk_exposures_all_attempts:rows.filter(exposure).length};
  }
  function scenario(a,s,{api_per_1k=null,hosting_per_1k=null,api_currency='USD'}={}){
    for(const k of ['volume','manual_minutes','review_minutes','audit_fraction','audit_minutes','rework_minutes','labour_per_hour','setup_cost','recurring_cost','fx_api_to_labour'])if(s[k]!=null&&(!finite(s[k])||s[k]<0))throw Error(k+' must be nonnegative and finite');
    if((s.audit_fraction??0)>1)throw Error('Invalid audit fraction');
    if(s.fx_api_to_labour!=null&&s.fx_api_to_labour<=0)throw Error('Currency conversion must be positive when supplied');
    const n=a.planned??0,v=s.volume,usable=n>0&&finite(v)&&(a.unattempted??0)===0;
    const hours=(count,min)=>usable&&finite(count)&&finite(min)?count/n*v*min/60:null;
    const manual=hours(n,s.manual_minutes),review=hours(n-(a.accepted??0),s.review_minutes),audit=finite(s.audit_fraction)?hours((a.accepted??0)*s.audit_fraction,s.audit_minutes):null,rework=hours(a.accepted_errors??0,s.rework_minutes===undefined?0:s.rework_minutes);
    const effort=[review,audit,rework].every(finite)?review+audit+rework:null;
    const fx=api_currency===s.currency?1:s.fx_api_to_labour,labour=finite(effort)&&finite(s.labour_per_hour)?effort*s.labour_per_hour:null;
    const tech=usable&&[api_per_1k,hosting_per_1k,fx].every(finite)?(api_per_1k+hosting_per_1k)*v/1000*fx:null;
    const total=[labour,tech,s.recurring_cost].every(finite)?labour+tech+s.recurring_cost:null,baseline=finite(manual)&&finite(s.labour_per_hour)?manual*s.labour_per_hour:null,savings=finite(baseline)&&finite(total)?baseline-total:null;
    return {manual_hours:manual,review_hours:review,audit_hours:audit,estimated_rework_hours:rework,gross_hours_saved:finite(manual)&&finite(review)?manual-review:null,
      net_hours_saved:finite(manual)&&finite(effort)?manual-effort:null,manual_cost:baseline,monthly_total:total,total_per_1k:finite(total)&&v>0?total/v*1000:null,
      monthly_savings:savings,payback_months:finite(s.setup_cost)&&finite(savings)&&savings>0?s.setup_cost/savings:null,currency:s.currency??null,
      basis:'user assumptions; cohort error/prevalence extrapolation; capacity is not guaranteed cash savings'};
  }
  return {finite,primary,percentile,fallback,csv,acceptance,acceptanceSummary,scenario};
})();
if (typeof module !== 'undefined') module.exports=ReportLogic;
