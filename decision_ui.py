"""Streamlit controls over the shared saved-response analysis."""
import copy
import pandas as pd
import streamlit as st
import plotly.express as px
import benchmark as b
import decision_analysis as d


def controls(cfg, report):
    path=report['folder']
    # Load separately frozen analysis settings without modifying run configuration.
    saved=path/'thresholds.analysis.json'
    if saved.exists(): cfg['frozen_acceptance']=b.parse(saved.read_text())
    with st.expander('Acceptance and workflow assumptions (saved-response analysis)'):
        st.caption('Editing thresholds here is exploratory on this dataset. Frozen development-only thresholds remain the default when available. No inference calls occur.')
        tasks=sorted({c['task'] for c in report['cases']})
        task=st.selectbox('Acceptance use case',tasks,key='accept_task')
        edit=st.checkbox('Explore a different acceptance policy',key='accept_edit')
        if edit:
            p=cfg.setdefault('tasks',{}).setdefault(task,{}).setdefault('acceptance',{})
            p['mode']=st.selectbox('Acceptance measure',['label','probability','confidence'],format_func=lambda x:{'label':'Label-based only','probability':'Selected-option probability','confidence':'Returned vendor confidence'}[x])
            p['threshold']=st.number_input('Choice acceptance threshold',0.,1.,float(p.get('threshold',.9)),.01)
            p['negative_threshold']=st.number_input('Noul: accept negative at or below',0.,1.,.1,.01)
            p['positive_threshold']=st.number_input('Noul: accept positive at or above',0.,1.,.9,.01)
            if p['negative_threshold']>=p['positive_threshold']:
                st.error('Negative threshold must be below positive threshold.');st.stop()
            p['selection_basis']='exploratory on displayed dataset; not frozen before evaluation'
            for per_model in cfg.get('frozen_acceptance',{}).values():per_model.pop(task,None)
        s=cfg.setdefault('workflow_cost',{})
        s['volume']=st.number_input('Monthly workflow records',0,1000000000,int(s.get('volume',cfg['business'].get('monthly_volume',100000))))
        s['currency']=st.selectbox('Labour / scenario currency',['USD','EUR','GBP'],index=['USD','EUR','GBP'].index(s.get('currency','USD')))
        for key,label in [('manual_minutes','Manual processing minutes / record'),('review_minutes','Human review minutes / deferred record'),
                          ('audit_fraction','Audit fraction of accepted records (0–1)'),('audit_minutes','Audit minutes / sampled record'),
                          ('rework_minutes','Rework minutes / incorrect accepted record'),('labour_per_hour','Loaded labour cost / hour'),
                          ('setup_cost','One-off setup cost'),('recurring_cost','Other recurring cost / month'),
                          ('fx_api_to_labour','Scenario currency units per API currency unit (if different)')]:
            text=st.text_input(label,value='' if s.get(key) is None else str(s[key]),key='workflow_'+key)
            try:s[key]=float(text) if text.strip() else None
            except ValueError:st.error(label+' must be numeric or blank.');st.stop()
        try:d.scenario({'planned':0},s)
        except ValueError as exc:st.error(str(exc));st.stop()
        st.caption('Blank inputs stay unknown. API and hosting amounts require an explicit conversion when currencies differ. Savings release capacity; they are not guaranteed cash savings.')


def render(scoped, records, cases, cfg, report):
    st.subheader('Acceptance, risks and business evidence')
    st.caption('Three distinct measures: non-review-label coverage, threshold-qualified coverage, and business-policy-eligible coverage. AI-reviewed SHVE policies are not approved for live automation.')
    st.dataframe(pd.DataFrame([{k:r[k] for k in ['model','evaluation_scope','label_based_coverage','threshold_coverage','business_eligible_coverage','deployment_eligibility','outstanding_targets','deployment_recommendation']} for r in scoped]),hide_index=True,width='stretch')
    st.dataframe(pd.DataFrame([{'model':r['model'],**r['acceptance']} for r in scoped]),hide_index=True,width='stretch')
    curves=[]
    for r in scoped:
        rows=[x for x in records if b.primary_row(x) and (x['model'],x['dataset'],x['task'])==(r['model'],r['dataset'],r['task'])]
        raw_curves=d.risk_curve(rows,r['acceptance_policy'],r['planned'])
        curves += [{'model':r['model'],**x} for x in d.cascade_curves(raw_curves,records,scoped,r,cfg['business']['baseline'])]
    if curves:
        frame=pd.DataFrame(curves)
        st.plotly_chart(px.line(frame,x='coverage',y='accepted_error_rate',color='model',markers=True,
                               hover_data=['threshold','accepted','accepted_families','accepted_critical_errors','review_rate'],
                               title='Exploratory risk–coverage trade-off',template='plotly_white'),width='stretch')
        st.dataframe(frame,width='stretch')
    st.caption('Threshold curves are exploratory. Accepted-subset intervals use declared families; boundary samples cannot establish zero future risk.')
    st.dataframe(pd.DataFrame([{'model':r['model'],**r['workflow_scenario']} for r in scoped]),width='stretch')
    for r in scoped:
        with st.expander(r['model']+' · label, risk and probability detail'):
            st.write('Per-label precision / recall');st.dataframe(pd.DataFrame(r['per_label']),width='stretch')
            st.write({'review_recall':r['review_recall'],'review_exposures':r['review_exposures'],'use_case_overrides':r['use_case_overrides'],
                      'blocked_business_decisions':r['blocked_business_decisions']})
            st.dataframe(pd.DataFrame(r['error_directions']),width='stretch')
            st.json(r['probability_quality'])
            if r['adjudication']['flagged_cases']:
                st.write('Adjudication sensitivity');st.json(r['adjudication'])
    with st.expander('Dataset diagnostics and family concentration'):
        st.json(d.diagnostics(cases,records))
    if cfg.get('shve_protocol'):
        from shve import baseline_records, BASELINE_VERSION
        rcfg=copy.deepcopy(cfg);rcfg['models']=[{'name':'rules','enabled':True,'display_name':'Deterministic rules','api':'rules','model':BASELINE_VERSION,'endpoint':'','deployment':'local'}]
        rules=b.summarize(baseline_records(cases,cfg),cases,rcfg,bootstrap=100)[0]
        scope={(r['dataset'],r['task']) for r in scoped}
        st.subheader('Deterministic baseline — separate from model deployments')
        st.dataframe(pd.DataFrame([{k:r[k] for k in ['dataset','task','success_rate','macro_f1','attempted','families','critical_failures','p50_ms']} for r in rules if (r['dataset'],r['task']) in scope]),width='stretch')
        comparisons=d.baseline_comparisons(scoped,records,rules,baseline_records(cases,cfg))
        st.dataframe(pd.DataFrame(comparisons),hide_index=True,width='stretch')
        st.caption('Input-only rules; vocabulary frozen from development. Timing is an in-process function measurement, not HTTP latency. No downstream predictive experiment was performed.')
