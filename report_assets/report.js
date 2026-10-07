'use strict';
const data=JSON.parse(document.getElementById('report-data').textContent);
const L=ReportLogic, content=document.getElementById('content');
const casesById=new Map(data.cases.map(c=>[c.id,c]));
let baseline=data.baseline, dataset=data.default_dataset??(data.study_design?'shve_v2_evaluation':''), cohort='', activeTab='Quality', volume=null;
const fallbackLabels=new Map();
const labels={model:'Model',dataset:'Dataset',task:'Use case',planned:'Planned cases',attempted:'Attempted',families:'Families',
  success_rate:'Task success',success_ci_low:'Success CI low',success_ci_high:'Success CI high',delta_vs_baseline:'Gap vs reference',
  delta_ci_low:'Gap CI low',delta_ci_high:'Gap CI high',p50_ms:'Median (ms)',p95_ms:'p95 (ms)',api_per_1k:'API cost / 1k',
  active_hosting_per_1k:'Active hosting / 1k',active_total_per_1k:'Active total / 1k',always_on_30day_hosting:'Always-on hosting / 30 days',
  estimated_api_cost:'Estimated API cost',monthly_api:'Monthly API cost',monthly_api_savings_vs_baseline:'Monthly API savings vs reference',
  cost_per_correct:'Cost / correct answer',cost_coverage:'Cost coverage',rate_basis:'Price basis',input_per_million:'Input / 1M tokens',
  output_per_million:'Output / 1M tokens',critical_failures:'Critical failures',unsafe_decisions:'Unsafe decisions',risk_exposures:'Risk exposures',
  recommendation:'Next step',ci_method:'Interval method',macro_f1:'Macro F1',brier:'Brier score',mae:'MAE',valid_rate:'Valid answers',
  api_errors:'API errors',invalid_answers:'Invalid answers',none_precision:'NONE precision',none_recall:'NONE recall',
  non_none_on_none_rate:'Non-NONE on NONE',unsafe_upper95:'Unsafe upper 95%',auto_coverage:'Label-based automation coverage',auto_accuracy:'Label-based accepted accuracy',
  fallback_rate:'Fallback rate',simulated_p95_ms:'Simulated p95 (ms)',known_api_cost:'Known API cost',priced_requests:'Requests priced',
  label_based_coverage:'Non-review-label coverage',threshold_coverage:'Threshold-qualified coverage',business_eligible_coverage:'Business-policy-eligible coverage',
  coverage:'Accepted coverage',review_rate:'Review / deferral demand',accepted_accuracy:'Accepted-subset accuracy',accepted_error_rate:'Accepted-subset error',
  accepted_ci_low:'Accepted accuracy CI low',accepted_ci_high:'Accepted accuracy CI high',threshold:'Exploratory threshold',
  incremental_accuracy:'Model minus rules',model_api_per_1k:'Model API cost / 1k',model_p95_ms:'Model p95 (ms)',
  probability_basis:'Operating point',roc_auc:'ROC-AUC',average_precision:'Average precision (step AP)',log_loss:'Log loss',
  prediction_precision:'Precision',prediction_recall:'Recall (all planned positives)',prediction_f1:'Positive F1',prediction_specificity:'Specificity (all planned negatives)',
  probability_coverage:'Probability coverage (population weighted)',sample_positive:'Sample positives',population_positive:'Represented population positives',
  population_prevalence:'Represented population churn prevalence',represented_population:'Represented population',
  sample_success_rate:'Sample task success (oversampled diagnostic)',sample_macro_f1:'Sample macro F1 (oversampled diagnostic)',sample_brier:'Sample Brier (oversampled diagnostic)',
  quality_basis:'Quality interpretation',weighted_churn_roc_auc:'Weighted churn ROC-AUC',weighted_churn_ap:'Weighted churn average precision',
  weighted_churn_brier:'Weighted churn Brier',weighted_churn_log_loss:'Weighted churn log loss',weighted_churn_recall:'Weighted churn recall at 0.5',
  weighted_churn_precision:'Weighted churn precision at 0.5',weighted_churn_coverage:'Weighted churn probability coverage',
  observed_churn:'Observed outcome',estimated_no_churn:'Predicted no churn',estimated_churn:'Predicted churn',estimated_invalid:'Invalid / unattempted'};
const ratios=new Set(['success_rate','success_ci_low','success_ci_high','valid_rate','none_precision','none_recall','non_none_on_none_rate',
  'unsafe_upper95','auto_coverage','auto_accuracy','cost_coverage','fallback_rate','agreement','all_pairs_correct','all_runs_accuracy','valid_repeat_rate','repeat_agreement',
  'label_based_coverage','threshold_coverage','business_eligible_coverage','coverage','review_rate','accepted_accuracy','accepted_error_rate',
  'accepted_ci_low','accepted_ci_high','review_demand','rules_success_rate','accepted_critical_rate','precision','recall',
  'simulated_cascade_success','simulated_cascade_fallback_rate','prediction_precision','prediction_recall','prediction_f1','prediction_specificity','probability_coverage','population_prevalence',
  'weighted_churn_recall','weighted_churn_precision','weighted_churn_coverage','sample_success_rate']);
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function heading(text){content.append(el('h2',text));}
function note(text){content.append(el('p',text,'muted'));}
function format(v,key=''){
  if(v===null || v===undefined)return '—';
  if(typeof v==='boolean')return v ? 'Yes':'No';
  if(typeof v==='object')return JSON.stringify(v);
  if(L.finite(v)){
    if(ratios.has(key))return (v*100).toFixed(2)+'%';
    if(key.startsWith('delta_')||key==='incremental_accuracy')return (v*100).toFixed(2)+' pp';
    if(/cost|api|hosting|million|savings/.test(key) && !/coverage|errors/.test(key))return v.toLocaleString('en-US',{maximumFractionDigits:6});
    return v.toLocaleString('en-US',{maximumFractionDigits:4});
  }
  return String(v);
}
function table(rows,fields,parent=content,onRow=null){
  if(!rows.length){parent.append(el('p','No matching data.','empty'));return;}
  fields=fields||[...new Set(rows.flatMap(r=>Object.keys(r)))];
  const wrap=el('div',undefined,'table-wrap'), t=el('table'), head=el('thead'), tr=el('tr'), body=el('tbody');
  let current=[...rows], ascending=true;
  function draw(){body.replaceChildren();for(const row of current){const r=el('tr');for(const k of fields){const td=el('td');
    if(onRow && k===fields[0]){const b=el('button',format(row[k],k));b.onclick=()=>onRow(row);td.append(b);}
    else td.textContent=format(row[k],k);td.title=format(row[k],k);r.append(td);}body.append(r);}}
  for(const k of fields){const th=el('th'), b=el('button',labels[k]||k.replaceAll('_',' '));b.title='Sort by '+b.textContent;
    b.onclick=()=>{current.sort((a,c)=>{const x=a[k],y=c[k];if(x==null)return y==null?0:1;if(y==null)return -1;
      const cmp=typeof x==='number'&&typeof y==='number'?x-y:String(x).localeCompare(String(y));return ascending?cmp:-cmp;});ascending=!ascending;draw();};
    th.append(b);tr.append(th);}
  head.append(tr);t.append(head,body);wrap.append(t);parent.append(wrap);draw();return wrap;
}
function select(label,options,value,parent,onChange){const box=el('label',label,'control'), sel=el('select');sel.setAttribute('aria-label',label);
  for(const [v,text] of options){const option=el('option',text);option.value=v;sel.append(option);}sel.value=value;
  sel.onchange=()=>onChange(sel.value);box.append(sel);parent.append(box);return sel;
}
function detail(title,value,parent=content,open=false){const d=el('details');d.open=open;d.append(el('summary',title),el('pre',typeof value==='string'?value:JSON.stringify(value,null,2)));parent.append(d);}
function key(r){return JSON.stringify([r.dataset,r.task]);}
function summaries(){return data.summaries_by_reference[baseline];}
function selected(){return summaries().filter(r=>key(r)===cohort);}
function cohortParts(){return JSON.parse(cohort);}
function records(){const [d,t]=cohortParts();return data.records.filter(r=>r.dataset===d && r.task===t);}
function allVisible(){return summaries().filter(r=>!dataset || r.dataset===dataset);}
function churnFindings(){const [d,t]=cohortParts();return (data.decision_analysis?.churn_prediction||[]).filter(r=>r.dataset===d&&r.task===t);}
function predictiveRow(entry,quality,basis){return {model:entry.model,probability_basis:basis,threshold:quality.operating_point.threshold,
  ...quality.probability_metrics,prediction_precision:quality.operating_point.precision,prediction_recall:quality.operating_point.recall,
  prediction_f1:quality.operating_point.f1,prediction_specificity:quality.operating_point.specificity,
  probability_coverage:quality.coverage.weighted_probability_validity,planned:quality.sample.planned,
  sample_positive:quality.sample.positive,population_positive:quality.population.positive,
  represented_population:quality.population.represented,population_prevalence:quality.population.positive_prevalence};}
const predictiveFields=['model','probability_basis','threshold','roc_auc','average_precision','brier','log_loss','prediction_precision','prediction_recall','prediction_f1','prediction_specificity','probability_coverage','planned','sample_positive','population_positive','represented_population','population_prevalence'];
function color(alias,i){return data.colors[alias]||['#0072B2','#E69F00','#009E73','#8064C9','#D55E00','#56B4E9'][i%6];}
function chart(rows,fields,title,percent=false){
  if(!rows.length){note('Charts require complete cohorts. See coverage in the table.');return;}
  const wrap=el('div',undefined,'chart');wrap.append(el('h3',title));
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 960 330');svg.setAttribute('role','img');svg.setAttribute('aria-label',title);
  function shape(tag,attrs,text){const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.append(n);return n;}
  const maximum=percent?1:Math.max(1,...rows.flatMap(r=>fields.map(f=>L.finite(r[f])?r[f]:0)))*1.1;
  for(let i=0;i<=4;i++){const y=270-i*58;shape('line',{x1:80,y1:y,x2:940,y2:y,stroke:'#e4e8ef'});shape('text',{x:68,y:y+5,'text-anchor':'end',fill:'#657086','font-size':13},percent?(i*25)+'%':Math.round(maximum*i/4).toLocaleString());}
  const group=840/rows.length, bw=Math.min(85,(group-24)/fields.length);
  rows.forEach((r,i)=>{fields.forEach((f,j)=>{if(!L.finite(r[f]))return;const height=r[f]/maximum*232;
    const bar=shape('rect',{x:90+i*group+(group-bw*fields.length)/2+j*bw,y:270-height,width:bw-4,height,rx:3,fill:fields.length===1?color(r.model,i):j===0?'#0072B2':'#E69F00'});
    const titleNode=document.createElementNS(ns,'title');titleNode.textContent=r.model+' · '+(labels[f]||f)+': '+format(r[f],f);bar.append(titleNode);
    shape('text',{x:90+i*group+(group-bw*fields.length)/2+j*bw+(bw-4)/2,y:Math.max(24,263-height),'text-anchor':'middle','font-size':12,fill:'#263046'},format(r[f],f));});
    shape('text',{x:90+i*group+group/2,y:299,'text-anchor':'middle','font-size':14,fill:'#263046'},r.model);});
  wrap.append(svg);if(fields.length>1){const legend=el('div',undefined,'legend');fields.forEach((f,i)=>{const entry=el('span'),s=el('i',undefined,'swatch');s.style.background=i===0?'#0072B2':'#E69F00';entry.append(s,document.createTextNode(labels[f]||f));legend.append(entry);});wrap.append(legend);}content.append(wrap);
}
function renderFilters(){
  const parent=document.getElementById('filters');parent.replaceChildren();
  select('Reference model',Object.keys(data.summaries_by_reference).map(x=>[x,x]),baseline,parent,v=>{baseline=v;render();});
  const datasets=[...new Set(summaries().map(r=>r.dataset))].sort();
  const splitLabel=x=>data.study_design?({'shve_v2_development':'Threshold selection','shve_v2_evaluation':'Performance evaluation','shve_improved_development':'Threshold selection','shve_improved_evaluation':'Performance evaluation'}[x]||x):x;
  select(data.study_design?'Data split':'Dataset',[['',data.study_design?'Both splits (shown separately)':'All datasets'],...datasets.map(x=>[x,splitLabel(x)])],dataset,parent,v=>{dataset=v;cohort='';volume=null;render();});
  const cohorts=[...new Map(allVisible().map(r=>[key(r),r])).entries()].sort((a,b)=>a[0].localeCompare(b[0]));
  if(!cohorts.some(([k])=>k===cohort))cohort=cohorts[0]?.[0]||'';
  select('Inspect one use case',cohorts.map(([k,r])=>[k,splitLabel(r.dataset)+' / '+r.task]),cohort,parent,v=>{cohort=v;volume=null;render();});
}
function overview(){heading('Business decision matrix');note('All use cases in the selected dataset. Every tab and case is included in this file, regardless of the view selected when it was exported.');
  if(allVisible().some(r=>r.churn_predictive_fixed))note('Churn task success, reference gaps and acceptance accuracy are oversampled sample diagnostics. Use the weighted churn prediction fields for population estimates; probability metrics describe valid responses and must be read with coverage.');
  const interpretations=(data.decision_analysis?.baseline_interpretations||[]).filter(r=>!dataset||r.dataset===dataset);
  if(interpretations.length)table(interpretations,['dataset','task','rules_success_rate','interpretation']);
  table(allVisible().map(r=>({...r,quality_basis:r.quality_basis||'Matched sample task metrics',
    weighted_churn_roc_auc:r.churn_predictive_fixed?.probability_metrics.roc_auc,weighted_churn_ap:r.churn_predictive_fixed?.probability_metrics.average_precision,
    weighted_churn_brier:r.churn_predictive_fixed?.probability_metrics.brier,weighted_churn_log_loss:r.churn_predictive_fixed?.probability_metrics.log_loss,
    weighted_churn_recall:r.churn_predictive_fixed?.operating_point.recall,weighted_churn_precision:r.churn_predictive_fixed?.operating_point.precision,
    weighted_churn_coverage:r.churn_predictive_fixed?.coverage.weighted_probability_validity,
    scenario_net_hours_saved:r.workflow_scenario?.net_hours_saved,scenario_review_hours:r.workflow_scenario?.review_hours,scenario_total_per_1k:r.workflow_scenario?.total_per_1k,scenario_currency:r.workflow_scenario?.currency})),['dataset','task','model','requested_model','effort','status',...(data.decision_analysis?.churn_prediction?['quality_basis']:[]),'success_rate','delta_vs_baseline','delta_ci_low','delta_ci_high',
    ...(data.decision_analysis?.churn_prediction?['weighted_churn_roc_auc','weighted_churn_ap','weighted_churn_brier','weighted_churn_log_loss','weighted_churn_recall','weighted_churn_precision','weighted_churn_coverage']:[]),
    'evaluation_scope','deployment_eligibility','label_based_coverage','threshold_coverage','business_eligible_coverage','outstanding_targets','blocked_business_decisions','deployment_recommendation','p95_ms','api_per_1k','currency','cost_coverage','scenario_net_hours_saved','scenario_review_hours','scenario_total_per_1k','scenario_currency','critical_failures','unsafe_decisions','risk_exposures','attempted','families','recommendation']);}
function baselineInterpretation(){const [ds,t]=cohortParts();const finding=(data.decision_analysis?.baseline_interpretations||[]).find(r=>r.dataset===ds&&r.task===t);if(finding)note(finding.interpretation);}
function quality(){heading('Quality');baselineInterpretation();const churn=churnFindings();
  if(churn.length){content.append(el('h3','Weighted churn prediction'));
    if(data.study_design?.prediction_scope)note(data.study_design.prediction_scope);
    note('Sampling weights restore monthly label proportions after rare-churn oversampling. ROC-AUC, noninterpolated step average precision, Brier and log loss use valid probabilities only. Coverage uses all planned population weight; invalid and missing predictions reduce recall. These measures do not establish retention uplift.');
    const rows=churn.flatMap(entry=>[predictiveRow(entry,entry.fixed,'Fixed threshold 0.5'),...(entry.selected?[predictiveRow(entry,entry.selected,'Development-selected threshold')]:[])]);
    chart(churn.map(entry=>predictiveRow(entry,entry.fixed,'Fixed threshold 0.5')),['roc_auc','average_precision'],'Population-weighted discrimination on valid probabilities',true);table(rows,predictiveFields);
    for(const entry of churn){detail(entry.model+' · development-only threshold selection',entry.threshold_selection);detail(entry.model+' · fixed-threshold uncertainty',entry.fixed.uncertainty);if(entry.selected)detail(entry.model+' · selected-threshold uncertainty',entry.selected.uncertainty);}
    content.append(el('h3','Oversampled sample diagnostics'));
    note('The following task success, macro F1, Brier and reliability counts describe the oversampled sample; they are not population quality estimates.');
    table(selected().map(r=>({...r,sample_success_rate:r.success_rate,sample_macro_f1:r.macro_f1,sample_brier:r.brier})),['model','planned','attempted','families','sample_success_rate','sample_macro_f1','sample_brier','valid_rate','api_errors','invalid_answers']);
  }else{chart(selected().filter(r=>r.status==='complete'),['success_rate'],'Task success on complete matched cohorts',true);
  table(selected(),['model','planned','attempted','families','success_rate','success_ci_low','success_ci_high','ci_method','macro_f1','brier','mae','valid_rate','api_errors','invalid_answers',
    'none_precision','none_recall','non_none_on_none_rate','unsafe_decisions','risk_exposures','unsafe_upper95','risk_direction','auto_coverage','auto_accuracy']);
  }
  note('Critical failures are test failures, not measured production harm. Missing values (—) are unavailable, not zero.');
  const [d,t]=cohortParts();content.append(el('h3','Repeat consistency'));table(data.stability.filter(r=>r.dataset===d && r.task===t));
  content.append(el('h3','Paired robustness checks'));table(data.robustness.filter(r=>r.dataset===d&&r.task===t));}
function latency(){heading('Latency');chart(selected().filter(r=>r.status==='complete'),['p50_ms','p95_ms'],'Client latency, milliseconds');
  table(selected(),['model','p50_ms','p95_ms','attempt_p95_ms','timed_completed','api_errors']);note('Sequential client timings after warm-up; this is not a concurrent throughput test.');}
function cost(){heading('Cost and forecast');content.append(el('h3','Estimated API cost for the complete run'));
  table(data.cost_totals,['model','requests','priced_requests','estimated_api_cost','known_api_cost','currency','input_per_million','output_per_million','rate_basis']);
  note('Includes all use cases, warm-ups and repeats. Rate-based estimates, not provider invoices. Unknown charges stay unknown. Local API fees exclude hardware and electricity.');
  const [,task]=cohortParts(), policy={...data.business,...(data.business.use_cases||{})[task]};
  if(volume===null)volume=policy.monthly_volume??100000;
  const controls=el('div',undefined,'filters'),box=el('label','Monthly requests for this use case','control'),input=el('input');input.type='number';input.min='0';input.step='1';input.value=volume;box.append(input);controls.append(box);content.append(controls);
  const forecast=el('div');content.append(forecast);
  function draw(){forecast.replaceChildren();const rows=selected().map(r=>({...r,
      monthly_api:r.status==='complete' && r.cost_coverage===1 && L.finite(r.api_per_1k)?r.api_per_1k/1000*volume:null}));
    const ref=rows.find(r=>r.model===baseline);for(const r of rows)r.monthly_api_savings_vs_baseline=r.model!==baseline && ref && r.currency===ref.currency && L.finite(r.monthly_api) && L.finite(ref.monthly_api)?ref.monthly_api-r.monthly_api:null;
    table(rows,['model','api_per_1k','cost_per_correct','cost_coverage','active_hosting_per_1k','active_total_per_1k','monthly_api','monthly_api_savings_vs_baseline','always_on_30day_hosting','currency','rate_basis'],forecast);}
  input.oninput=()=>{const v=Number(input.value);if(input.value!=='' && Number.isFinite(v) && v>=0){volume=v;draw();}};draw();
  note('Prices and hosting rates are captured at export. Change volume to explore a forecast; this does not change historical spend or quality. Active hosting uses serial execution time. Always-on hosting assumes 720 hours. Blank local hosting means no hourly estimate was supplied.');}
function fallback(){heading('Fallback scenario');const [d,t]=cohortParts(), rs=records();
  const options=[...new Set(rs.filter(r=>L.primary(r)&&r.type==='choice'&&r.valid).map(r=>r.decision))].sort();
  if(!fallbackLabels.has(cohort))fallbackLabels.set(cohort,new Set(options.filter(x=>['NONE','CLARIFY','ESCALATE','HUMAN_REVIEW'].includes(x))));
  const chosen=fallbackLabels.get(cohort),controls=el('fieldset'),legend=el('legend','Also fall back on these explicit choice labels');controls.append(legend);
  const out=el('div'),usePolicy=el('input');usePolicy.type='checkbox';const policyLabel=el('label',' Also fall back on acceptance-policy deferral');policyLabel.prepend(usePolicy);controls.append(policyLabel);const draw=()=>{out.replaceChildren();const rows=usePolicy.checked?selected().flatMap(r=>L.fallback(data.records,summaries(),d,t,baseline,chosen,policyFor(r)).filter(x=>x.model===r.model)):L.fallback(data.records,summaries(),d,t,baseline,chosen);table(rows,null,out);};usePolicy.onchange=draw;
  for(const option of options){const label=el('label'),check=el('input');check.type='checkbox';check.checked=chosen.has(option);check.onchange=()=>{check.checked?chosen.add(option):chosen.delete(option);draw();};label.append(check,document.createTextNode(option));controls.append(label);}
  if(!options.length)controls.append(el('p','No categorical labels for this use case. Invalid answers and API errors still trigger fallback.'));
  content.append(controls,out);draw();note('Offline replay using the selected reference model. Invalid answers and API failures always fall back. Requires complete matched cases in the same currency. Costs and latencies add serially; no model calls are made.');}
function confusion(){heading('Confusion matrices');note('Rows are expected labels; columns are predictions. Primary attempts only. Missing requests are not counted.');const [d,t]=cohortParts();
  const churn=churnFindings(),matrices=data.confusion.filter(r=>r.dataset===d && r.task===t);if(!matrices.length){note(churn.length?'No primary attempts are available for an observed sample-count matrix.':'Score tasks use numeric errors instead of a categorical confusion matrix. See Quality.');if(!churn.length)return;}
  for(const matrix of matrices){content.append(el('h3',matrix.model));const counts=matrix.counts, names=[...new Set([...Object.keys(counts),...Object.values(counts).flatMap(Object.keys).filter(k=>k!=='(invalid/API error)')])].sort();
    const predicted=Object.values(counts).some(r=>'(invalid/API error)' in r)?[...names,'(invalid/API error)']:names;
    const maximum=Math.max(1,...Object.values(counts).flatMap(Object.values)),wrap=el('div',undefined,'table-wrap'),tableNode=el('table',undefined,'matrix'),head=el('tr');
    head.append(el('th','Expected ↓ / Predicted →'));for(const name of predicted)head.append(el('th',name));tableNode.append(head);
    for(const gold of names){const row=el('tr');row.append(el('th',gold));for(const pred of predicted){const value=counts[gold]?.[pred]||0,ratio=value/maximum,td=el('td',String(value));td.style.background=`rgb(${239-Math.round(190*ratio)},${246-Math.round(130*ratio)},${255-Math.round(66*ratio)})`;td.style.color=ratio>=.55?'white':'#172b4d';row.append(td);}tableNode.append(row);}wrap.append(tableNode);content.append(wrap);}
  if(churn.length){content.append(el('h3','Population-weighted estimated counts'));
    note('Supplementary operating-point matrices estimate represented population counts from sampling weights. Invalid includes failed, invalid and unattempted cases; original matrices above retain actual observed sample counts.');
    for(const entry of churn){for(const [quality,basis] of [[entry.fixed,'Fixed threshold 0.5'],...(entry.selected?[[entry.selected,'Development-selected threshold']]:[])]){
      content.append(el('h3',entry.model+' · '+basis+' · '+quality.operating_point.threshold));
      table(Object.entries(quality.operating_point.confusion_weighted).map(([actual,counts])=>({observed_churn:actual==='actual_true'?'Churn':'No churn',
        estimated_no_churn:counts.predicted_false,estimated_churn:counts.predicted_true,estimated_invalid:counts.invalid})),['observed_churn','estimated_no_churn','estimated_churn','estimated_invalid']);}}}}
function casesView(){heading('Cases and responses');note('Includes original inputs, frozen questions, normalized predictions and full saved responses. Case contents travel with this file.');
  const controls=el('div',undefined,'filters'),list=el('div'),inspection=el('div');content.append(controls,list,inspection);
  let model='',mode='failures',search='',page=0,all=false;
  select('Model',[['','All models'],...Object.keys(data.profiles).map(x=>[x,x])],model,controls,v=>{model=v;page=0;draw();});
  select('Cases',[['failures','Failures'],['primary','All primary cases'],['checks','Warm-up / connection checks'],['repeat','Repeated sample']],mode,controls,v=>{mode=v;page=0;draw();});
  const box=el('label','Search case ID or input','control'),input=el('input');input.type='search';input.oninput=()=>{search=input.value.toLowerCase();page=0;draw();};box.append(input);controls.append(box);
  const label=el('label'),check=el('input');check.type='checkbox';check.onchange=()=>{all=check.checked;page=0;draw();};label.append(check,document.createTextNode(' Include all use cases'));controls.append(label);
  function inspect(row){inspection.replaceChildren();detail('Original case and frozen question',casesById.get(row.id),inspection,true);detail('Normalized prediction',Object.fromEntries(Object.entries(row).filter(([k])=>k!=='raw')),inspection,true);detail('Full raw response and API error',{response:row.raw,error:row.error_detail},inspection,true);}
  function draw(){list.replaceChildren();inspection.replaceChildren();const source=all?data.records.filter(r=>!dataset||r.dataset===dataset):records();
    const filtered=source.filter(r=>(!model||r.model===model) && (mode==='failures'?L.primary(r)&&!r.correct:mode==='primary'?L.primary(r):mode==='repeat'?r.phase==='repeat':['warmup','connection_check'].includes(r.phase)) &&
      (!search||(r.id+' '+JSON.stringify(casesById.get(r.id))).toLowerCase().includes(search)));
    const pages=Math.max(1,Math.ceil(filtered.length/50));page=Math.min(page,pages-1);const nav=el('div',undefined,'actions'),prev=el('button','Previous'),next=el('button','Next');prev.disabled=page===0;next.disabled=page===pages-1;
    prev.onclick=()=>{page--;draw();};next.onclick=()=>{page++;draw();};nav.append(el('span',`${filtered.length.toLocaleString()} matching requests · Page ${page+1} of ${pages}`),prev,next);list.append(nav);
    const shown=filtered.slice(page*50,(page+1)*50);table(shown,['id','model','dataset','task','phase','expected','value','correct','valid','api_error','latency_s','input_tokens','output_tokens'],list,inspect);
    if(shown.length)inspect(shown[0]);}draw();}
function evidence(){heading('Profiles and evidence');note('Configured checkpoint provenance is not independent verification of server identity. The captured price basis and business settings apply throughout this report.');
  detail('Model profiles',data.profiles,content,true);detail('Applied prices and hosting assumptions',data.quotes);detail('Run evidence and analysis assumptions',data.manifest);}
function protocol(){heading('Protocol and limitations');content.append(el('div',data.protocol.replaceAll('**',''),'protocol'));
  if(data.decision_analysis?.churn_prediction)detail('Improved churn prediction scope and frozen protocol',{prediction_scope:data.study_design?.prediction_scope,churn_protocol:data.study_design?.churn_protocol,threshold_rule:data.study_design?.threshold_rule},content,true);
  detail('Captured protocol',data.manifest.protocol,content,true);content.append(el('h3','All repeat checks'));table(data.stability);content.append(el('h3','All paired robustness checks'));table(data.robustness);}
function guide(){heading('Metric guide');const wrap=table(data.metric_guide.map(([metric,meaning])=>({metric,meaning})));if(wrap)wrap.classList.add('guide');}
const acceptanceOverrides=new Map(), workflowSettings={...(data.decision_analysis?.workflow_cost||{})};
function policyFor(row){return acceptanceOverrides.get(cohort)||row.acceptance_policy||{};}
function numberControl(parent,label,value,onChange,max=null){const box=el('label',label,'control'),input=el('input');input.type='number';input.min='0';input.step='any';if(max!==null)input.max=String(max);input.value=value??'';input.setAttribute('aria-label',label);input.onchange=()=>{const v=input.value===''?null:Number(input.value);if(v===null||(L.finite(v)&&v>=0&&(max===null||v<=max)))onChange(v);else input.reportValidity();};box.append(input);parent.append(box);return input;}
function acceptanceView(){heading('Acceptance and risk–coverage');
  if(churnFindings().length)note('Acceptance, per-label and probability-quality fields in this view are oversampled sample diagnostics. Population-weighted churn prediction metrics and coverage are on Quality.');
  note('Label-based, threshold-qualified and business-policy-eligible coverage are different measures. Missing scores go to review. Score threshold automation is unsupported. These fixtures do not certify live automation.');
  const controls=el('div',undefined,'filters');content.append(controls);
  const initial=policyFor(selected()[0]);let p={...initial};
  select('Acceptance measure',[['saved','Saved policy per model'],['label','Label-based only'],['probability','Selected-option probability'],['confidence','Returned vendor confidence']],acceptanceOverrides.has(cohort)?p.mode:'saved',controls,v=>{
    if(v==='saved')acceptanceOverrides.delete(cohort);else acceptanceOverrides.set(cohort,{...p,accept_none:false,mode:v,selection_basis:'exploratory on displayed cohort'});render();});
  if(acceptanceOverrides.has(cohort)){
    const update=(key,value)=>{if(value===null)return;const next={...p,[key]:value};if((next.negative_threshold??.1)>=(next.positive_threshold??.9)){alert('Negative threshold must be below positive threshold.');return;}acceptanceOverrides.set(cohort,next);render();};
    numberControl(controls,'Choice threshold',p.threshold??.9,v=>update('threshold',v),1);
    numberControl(controls,'Noul negative at or below',p.negative_threshold??.1,v=>update('negative_threshold',v),1);
    numberControl(controls,'Noul positive at or above',p.positive_threshold??.9,v=>update('positive_threshold',v),1);
  }
  const values=selected().map(r=>{const rr=records().filter(x=>L.primary(x)&&x.model===r.model),q=policyFor(r),stats=acceptanceOverrides.has(cohort)?{...L.acceptanceSummary(rr,q,r.planned),accepted_ci_low:null,accepted_ci_high:null,ci_method:'Edited policy: interval not recomputed offline'}:r.acceptance;return {model:r.model,policy:q.selection_basis??'label-based default',measure:q.mode??'label',...stats,business_eligible_coverage:L.acceptanceSummary(rr,q,r.planned,true).coverage};});
  table(values);note('Accepted accuracy uses accepted cases; coverage uses all planned cases. Failed and unattempted cases cannot disappear from the denominator. Any edited policy is exploratory, not unbiased validation.');
  content.append(el('h3','Exploratory threshold curve: coverage versus accepted error'));
  const [ds,task]=cohortParts(),curves=(data.decision_analysis?.risk_curves||[]).filter(r=>r.dataset===ds&&r.task===task);
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 800 310');svg.setAttribute('role','img');svg.setAttribute('aria-label','Automation coverage versus accepted-subset error');
  function shape(tag,attrs,text){const x=document.createElementNS(svg.namespaceURI,tag);Object.entries(attrs).forEach(([k,v])=>x.setAttribute(k,v));if(text!==undefined)x.textContent=text;svg.append(x);return x;}
  for(let i=0;i<=4;i++){const v=i/4;shape('line',{x1:65,y1:260-v*220,x2:770,y2:260-v*220,stroke:'#dce3ee'});shape('text',{x:10,y:265-v*220,fill:'#334155'},Math.round(v*100)+'%');shape('text',{x:65+v*700,y:285,fill:'#334155'},Math.round(v*100)+'%');}
  for(const r of curves){if(!L.finite(r.coverage)||!L.finite(r.accepted_error_rate))continue;const dot=shape('circle',{cx:65+700*r.coverage,cy:260-220*r.accepted_error_rate,r:5,fill:color(r.model,0)});const title=document.createElementNS(svg.namespaceURI,'title');title.textContent=`${r.model}: threshold ${r.threshold}, accepted ${r.accepted}, families ${r.accepted_families}, errors ${r.accepted_errors}`;dot.append(title);}
  shape('text',{x:280,y:308,fill:'#334155'},'Automation coverage →');content.append(svg);table(curves,['model','threshold','coverage','accepted_error_rate','accepted','accepted_families','accepted_critical_errors','accepted_risk_exposures','review_rate','failed','accepted_ci_low','accepted_ci_high','ci_method','model_api_per_1k','model_p95_ms','cascade_reference','simulated_cascade_success','simulated_cascade_fallback_rate','simulated_cascade_api_per_1k','simulated_cascade_p95_ms']);
  note('Vertical axis: accepted-subset error. Saved curves use selected-option probability unless captured policy specifies vendor confidence. Boundary samples cannot establish zero future risk. Curves are exploratory even when the separate selected policy was frozen using development only.');
  for(const r of selected()){content.append(el('h3',r.model+' · directional errors and reliability'));table(r.error_directions||[]);table(r.per_label||[]);detail('Review recall / probability quality',{review_recall:r.review_recall,review_exposures:r.review_exposures,...r.probability_quality});if(r.adjudication?.flagged_cases)detail('Adjudication sensitivity',r.adjudication);}
}
function workflow(){heading('Workflow economics');note('Editable assumptions. Deferral includes review labels, failed calls and missing threshold scores. Audits apply only to accepted records. Estimated rework extrapolates cohort errors; constructed class proportions are not production prevalence.');
  const s=workflowSettings;if(s.volume==null)s.volume=data.business.monthly_volume??100000;if(!s.currency)s.currency='USD';
  const controls=el('div',undefined,'filters'),out=el('div');content.append(controls,out);
  function draw(){out.replaceChildren();try{const rows=selected().map(r=>{const a=L.acceptanceSummary(records().filter(x=>L.primary(x)&&x.model===r.model),policyFor(r),r.planned);return {model:r.model,...L.scenario(a,s,{api_per_1k:r.cost_coverage===1?r.api_per_1k:null,hosting_per_1k:r.active_hosting_per_1k,api_currency:r.currency})};});table(rows,null,out);}catch(error){out.append(el('p',error.message,'notice'));}}
  select('Labour / scenario currency',['USD','EUR','GBP'].map(v=>[v,v]),s.currency,controls,v=>{s.currency=v;draw();});
  for(const [k,label] of [['volume','Monthly workflow records'],['manual_minutes','Manual minutes / record'],['review_minutes','Review minutes / deferred record'],['audit_fraction','Audit fraction of accepted records (0–1)'],['audit_minutes','Audit minutes / sampled record'],['rework_minutes','Rework minutes / incorrect accepted record'],['labour_per_hour','Loaded labour cost / hour'],['setup_cost','One-off setup cost'],['recurring_cost','Other recurring cost / month'],['fx_api_to_labour','Scenario currency units per API currency unit']])numberControl(controls,label,s[k],v=>{s[k]=v;draw();},k==='audit_fraction'?1:null);
  draw();note('Blank inputs remain unknown. Local API charges of zero exclude hosting. Positive recurring savings are required for setup payback. Released staff capacity is not guaranteed cash savings. These scenarios use hypothetical acceptance, not organisational approval.');
}
function diagnosticsView(){heading('Dataset diagnostics');const d=data.decision_analysis?.diagnostics||{};detail('Counts, repeated templates and family overlap',Object.fromEntries(Object.entries(d).filter(([k])=>!['error_concentration','subgroups'].includes(k))),content,true);table((d.error_concentration||[]).filter(r=>!dataset||r.dataset===dataset));content.append(el('h3','Available subgroup results'));table((d.subgroups||[]).filter(r=>!dataset||r.dataset===dataset));note('Development/calibration/test family overlap needs review. Deliberate standard/stress variants may share families. Declared family IDs do not establish independent source customers. Small subgroups are descriptive.');}
function baselineView(){heading('Deterministic baselines');baselineInterpretation();note(data.decision_analysis?.baseline_basis||'No deterministic baseline was supplied for this dataset.');const [ds,t]=cohortParts();const churn=churnFindings();
 if(churn.length){content.append(el('h3','Historical probability baseline'));note('A constant historical churn probability is frozen before confirmation; this is a predictive reference, not a retention intervention or a semantic rule solver. Metrics are population weighted; discrimination and calibration use valid probabilities.');
   table(churn.map(entry=>predictiveRow({model:entry.model+' · historical prior'},entry.baseline,'Frozen historical probability')),predictiveFields);return;}
 const rules=(data.decision_analysis?.baseline_summaries||[]).filter(r=>r.dataset===ds&&r.task===t);table(rules,['model','dataset','task','success_rate','success_ci_low','success_ci_high','macro_f1','attempted','families','critical_failures','risk_exposures','p50_ms']);
 if(rules.length){table((data.decision_analysis?.baseline_comparisons||[]).filter(r=>r.dataset===ds&&r.task===t));}
 note('Per-task findings take priority. Pooled case accuracy weights cases; an unweighted task average weights each task equally. Rule-solvable tasks require demonstrated incremental benefit before adding model cost.');}

const views={'Acceptance and risk':acceptanceView,'Workflow economics':workflow,'Dataset diagnostics':diagnosticsView,'Deterministic baselines':baselineView,'Overview':overview,'Quality':quality,'Latency':latency,'Cost and forecast':cost,'Fallback scenario':fallback,'Confusion matrices':confusion,'Cases':casesView,'Profiles and evidence':evidence,'Protocol':protocol,'Metric guide':guide};
function render(){renderFilters();const tabs=document.getElementById('tabs');tabs.replaceChildren();Object.keys(views).forEach((name,i)=>{const b=el('button',name);b.id='tab-'+i;b.setAttribute('role','tab');b.setAttribute('aria-selected',String(name===activeTab));b.setAttribute('aria-controls','content');
    b.onclick=()=>{activeTab=name;render();document.getElementById('tab-'+i).focus();};b.onkeydown=e=>{if(['ArrowRight','ArrowLeft','Home','End'].includes(e.key)){e.preventDefault();const keys=Object.keys(views);const j=e.key==='Home'?0:e.key==='End'?keys.length-1:(i+(e.key==='ArrowRight'?1:-1)+keys.length)%keys.length;activeTab=keys[j];render();document.getElementById('tab-'+j).focus();}};tabs.append(b);});
  content.replaceChildren();content.setAttribute('aria-labelledby','tab-'+Object.keys(views).indexOf(activeTab));if(cohort)views[activeTab]();else note('No result cohorts are available.');
  for(const link of document.querySelectorAll('[data-download]'))link.href=downloadHref(link.dataset.download);}
function downloadHref(kind){return './download/'+kind+(kind==='comparison.csv'?'?'+new URLSearchParams({baseline,dataset}):'');}
function completeReportHtml(){const page=document.documentElement.cloneNode(true);
  for(const id of ['subtitle','notices','summary-cards','filters','tabs','content','downloads'])page.querySelector('#'+id).replaceChildren();
  return '<!doctype html>\n'+page.outerHTML;
}
function download(name,text,type){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=el('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
document.getElementById('subtitle').textContent=data.run_name+' · Exported '+data.exported_utc.slice(0,10)+' · Reference at export: '+data.baseline;
if(data.study_design){const design=el('details',undefined,'notice');design.append(el('summary','One dataset, two splits — what this run measures'));
  if(data.study_design.prediction_scope)design.append(el('p',data.study_design.prediction_scope));
  design.append(el('p','No model training occurs in this benchmark. Threshold selection: '+data.study_design.split_counts.development+' cases. '+data.study_design.development_purpose),el('p','Performance evaluation: '+data.study_design.split_counts.evaluation+' cases. '+data.study_design.evaluation_purpose));
  for(const [name,requirement] of Object.entries(data.study_design.downstream_requirements))design.append(el('p',name.replaceAll('_',' ')+': '+requirement));
  design.append(el('p',data.study_design.additional_evidence));document.getElementById('notices').append(design);}
if(data.warnings.length){const notices=el('details',undefined,'notice');notices.append(el('summary',data.warnings.length===1?'Evidence limit':data.warnings.length+' evidence limits and next validation steps'));
  for(const warning of data.warnings)notices.append(el('p',warning));document.getElementById('notices').append(notices);}
for(const [value,label] of [[Object.keys(data.profiles).length,'Models'],[data.cases.length,'Dataset cases'],[data.records.length,'Recorded requests'],[new Set(data.cases.map(c=>c.task)).size,'Use cases']]){const card=el('div',undefined,'card');card.append(el('strong',value.toLocaleString()),el('span',label));document.getElementById('summary-cards').append(card);}
const footer=document.getElementById('downloads'),actions=el('div',undefined,'actions');
const httpDownloads=document.documentElement.dataset.downloads==='http'&&['http:','https:'].includes(location.protocol);
for(const [name,kind,action] of [['Download complete HTML','report.html',()=>download(data.run_name+'_report.html',completeReportHtml(),'text/html;charset=utf-8')],['Download comparison CSV','comparison.csv',()=>download('benchmark_comparison.csv',L.csv(allVisible()),'text/csv;charset=utf-8')],['Download complete data JSON','data.json',()=>download('benchmark_report_data.json',JSON.stringify(data,null,2),'application/json')]]){
  const control=el(httpDownloads?'a':'button',name);
  if(httpDownloads){control.dataset.download=kind;control.href=downloadHref(kind);control.download=kind;control.className='download-link';}
  else control.onclick=action;
  actions.append(control);
}footer.append(actions,el('p','The HTML download includes every tab, both splits and all saved cases, regardless of the current filters. It opens offline with no installation or API keys. Prices and business targets are captured at export; volume and fallback controls are scenarios. New inference requires the local application.','muted'));
render();
