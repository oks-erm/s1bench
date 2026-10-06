import copy
import json
import shutil
import subprocess
import unittest
import benchmark as b

class DecisionTests(unittest.TestCase):
    def row(self, **kw):
        return dict(dict(id='a', cluster_id='f', type='choice', valid=True, api_error=None,
                         decision='YES', value='YES', expected='YES', correct=True,
                         probabilities={'YES': .8, 'CLARIFY': .2}, confidence=.4), **kw)

    def test_threshold_semantics_and_missing_scores(self):
        import decision_analysis as d
        r=self.row()
        self.assertEqual(d.acceptance(r, {'mode':'probability','threshold':.8}), 'accepted')
        self.assertEqual(d.acceptance(r, {'mode':'confidence','threshold':.8}), 'reviewed')
        for value in [None, float('nan'), -1, 2, True]:
            self.assertEqual(d.acceptance(self.row(confidence=value), {'mode':'confidence','threshold':.5}), 'reviewed')
        self.assertEqual(d.acceptance(self.row(decision='HUMAN_REVIEW'), {'mode':'label'}), 'reviewed')
        self.assertEqual(d.acceptance(self.row(valid=False), {'mode':'label'}), 'failed')
        self.assertEqual(d.acceptance(self.row(type='score'), {'mode':'probability','threshold':.5}), 'reviewed')

    def test_asymmetric_noul_and_policy_eligibility(self):
        import decision_analysis as d
        p={'mode':'probability','negative_threshold':.1,'positive_threshold':.8}
        for value, result in [(0.1,'accepted'),(.5,'reviewed'),(.8,'accepted')]:
            self.assertEqual(d.acceptance(self.row(type='noul',value=value),p),result)
        with self.assertRaises(ValueError): d.validate_policy(dict(p,negative_threshold=.9))
        self.assertEqual(d.acceptance(self.row(decision='MATCH'), {'mode':'label','ineligible_labels':['MATCH']}, business=True),'reviewed')
        self.assertEqual(d.acceptance(self.row(), {'mode':'label','business_approved':False}, business=True),'reviewed')

    def test_directional_risk_keeps_failures_separate(self):
        rows=[self.row(decision='CLARIFY',correct=False,critical_error_choices=['YES']),
              self.row(id='b', decision='YES',correct=False,critical_error_choices=['YES']),
              self.row(id='c',valid=False,correct=False,critical_error_choices=['YES'])]
        r=b.risk_metrics(rows,'entity_match','choice',{})
        self.assertEqual((r['unsafe_decisions'],r['risk_exposures']),(1,3))
        self.assertIsNone(r['unsafe_upper95'])
        self.assertIsNone(b.risk_metrics([self.row(critical_error_choices=[])],'x','choice',{})['unsafe_rate'])

    def test_legacy_critical_errors_are_not_hidden_by_new_acceptance_tables(self):
        import decision_analysis as d
        rows=[self.row(critical=True,correct=False)]
        result=d.acceptance_summary(rows,{'mode':'label'})
        self.assertEqual(result['accepted_critical_errors'],1)
        self.assertEqual(result['accepted_risk_exposures'],1)
        self.assertEqual(result['risk_policy'],'legacy-critical-case-errors')

    def test_directional_repeat_flip_differs_from_safe_abstention(self):
        from test_benchmark import case, config, prediction
        c=case(expected='billing');c['critical_error_choices']=['NONE']
        cfg=config();cfg['protocol']['repetitions']=3
        rows=[prediction(c,value='billing'),prediction(c,value='NONE',phase='repeat',repetition=2),
              prediction(c,value='billing',phase='repeat',repetition=3)]
        result=b.repeat_summary(rows,[c],cfg,[c['id']])[0]
        self.assertEqual(result['critical_flips'],1)
        c['critical_error_choices']=[]
        self.assertEqual(b.repeat_summary(rows,[c],cfg,[c['id']])[0]['critical_flips'],0)

    def test_scenario_audit_and_currency(self):
        import decision_analysis as d
        a={'accepted':80,'reviewed':15,'failed':5,'planned':100,'accepted_errors':8}
        s={'volume':1000,'manual_minutes':6,'review_minutes':3,'audit_fraction':.1,'audit_minutes':2,
           'rework_minutes':5,'labour_per_hour':30,'currency':'EUR','setup_cost':100,'recurring_cost':10}
        r=d.scenario(a,s,api_per_1k=2,hosting_per_1k=3,api_currency='USD')
        self.assertEqual(r['audit_hours'],80*2/60)
        self.assertIsNone(r['monthly_total'])
        r=d.scenario(a,dict(s,fx_api_to_labour=.9),api_per_1k=2,hosting_per_1k=3,api_currency='USD')
        self.assertAlmostEqual(r['monthly_total'],(10+80*2/60+80*5/60)*30+10+4.5)
        self.assertIsNotNone(r['payback_months'])
        with self.assertRaises(ValueError):d.scenario(a,dict(s,fx_api_to_labour=0),api_per_1k=2,hosting_per_1k=3,api_currency='USD')

    def test_summary_denominators_and_invalid_vector(self):
        import decision_analysis as d
        rows=[self.row(),self.row(id='b',decision='CLARIFY',correct=False),self.row(id='c',valid=False,correct=False)]
        s=d.acceptance_summary(rows,{'mode':'probability','threshold':.8},planned=4)
        self.assertEqual((s['accepted'],s['reviewed'],s['failed'],s['unattempted']),(1,1,1,1))
        self.assertEqual(s['coverage'],.25)
        self.assertIsNone(d.probability_vector(self.row(probabilities={'YES':1.2,'CLARIFY':-.2}),['YES','CLARIFY']))

    @unittest.skipUnless(shutil.which('node'),'Node unavailable')
    def test_browser_acceptance_and_scenario_parity(self):
        import decision_analysis as d
        rows=[self.row(),self.row(confidence=None),self.row(valid=False),self.row(type='noul',value=.9)]
        policies=[{'mode':'label'},{'mode':'probability','threshold':.8,'negative_threshold':.1,'positive_threshold':.8}]
        inp={'rows':rows,'policies':policies}
        code="const L=require('./report_assets/logic.js'),p=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(p.policies.map(q=>p.rows.map(r=>L.acceptance(r,q)))));"
        proc=subprocess.run(['node','-e',code],input=json.dumps(inp),text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(proc.stdout),[[d.acceptance(r,p) for r in rows] for p in policies])

    @unittest.skipUnless(shutil.which('node'),'Node unavailable')
    def test_browser_acceptance_counts_and_risk_denominators_match_python(self):
        import decision_analysis as d
        rows=[self.row(critical_error_choices=[]),self.row(id='b',decision='YES',correct=False,critical_error_choices=['YES']),
              self.row(id='c',valid=False,correct=False,critical_error_choices=['YES'])]
        p={'mode':'probability','threshold':.8}
        code="const L=require('./report_assets/logic.js'),p=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(L.acceptanceSummary(p.rows,p.policy,4)));"
        proc=subprocess.run(['node','-e',code],input=json.dumps({'rows':rows,'policy':p}),text=True,capture_output=True,check=True)
        python=d.acceptance_summary(rows,p,4);browser=json.loads(proc.stdout)
        self.assertEqual(browser,{k:python[k] for k in browser})

    def test_freeze_uses_development_only(self):
        import shve
        cases=[dict(id='a',task='sector_imputation',split='development',question={'criteria':{'YES':'','CLARIFY':''}}),
               dict(id='b',task='sector_imputation',split='evaluation',question={'criteria':{'YES':'','CLARIFY':''}})]
        rows=[self.row(model='jev',task='sector_imputation',phase='primary'),self.row(id='b',model='jev',task='sector_imputation',phase='primary')]
        cfg={'tasks':{}}
        before=shve.freeze_thresholds(rows,cases,cfg)
        rows[1].update(correct=False,confidence=0,probabilities=None)
        self.assertEqual(before,shve.freeze_thresholds(rows,cases,cfg))

    @unittest.skipUnless(shutil.which('node'),'Node unavailable')
    def test_browser_cost_scenarios_parity(self):
        import decision_analysis as d
        a={'accepted':80,'reviewed':15,'failed':5,'planned':100,'accepted_errors':8}
        s={'volume':1000,'manual_minutes':6,'review_minutes':3,'audit_fraction':.1,'audit_minutes':2,
           'rework_minutes':5,'labour_per_hour':30,'currency':'EUR','setup_cost':100,'recurring_cost':10}
        for fx in [None,.9]:
            s['fx_api_to_labour']=fx
            opts={'api_per_1k':2,'hosting_per_1k':3,'api_currency':'USD'}
            code="const L=require('./report_assets/logic.js'),p=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(L.scenario(...p)));"
            proc=subprocess.run(['node','-e',code],input=json.dumps([a,s,opts]),text=True,capture_output=True,check=True)
            self.assertEqual(json.loads(proc.stdout),d.scenario(a,s,**opts))

    def test_all_review_and_no_review(self):
        import decision_analysis as d
        rows=[self.row(),self.row(id='b')]
        self.assertEqual(d.acceptance_summary(rows,{'mode':'probability','threshold':1})['accepted'],0)
        self.assertEqual(d.acceptance_summary(rows,{'mode':'probability','threshold':0})['accepted'],2)

    def test_all_review_threshold_preserves_semantic_review_labels(self):
        import decision_analysis as d
        from test_benchmark import case, config, prediction, profile
        c=case(expected='billing');cfg=config(profile())
        cfg['frozen_acceptance']={'m':{c['task']:{'mode':'probability','threshold':1,'accept_none':True}}}
        result=b.summarize([prediction(c,value='billing')],[c],cfg)[0][0]
        self.assertEqual(result['label_based_coverage'],1)
        self.assertEqual(result['threshold_coverage'],0)
        self.assertIsNone(result['review_recall'])
        self.assertEqual(d.acceptance(self.row(),{'mode':'probability','threshold':0,'accept_none':True}),'reviewed')

    def test_invalid_probability_vector_defers_in_python_and_browser(self):
        import decision_analysis as d
        p={'mode':'probability','threshold':.8}
        rows=[self.row(probabilities={'YES':.9,'CLARIFY':-.1}),
              self.row(probabilities={'YES':.9,'CLARIFY':.9}),
              self.row(probabilities={'YES':.9,'CLARIFY':float('inf')})]
        self.assertEqual([d.acceptance(r,p) for r in rows],['reviewed']*3)
        if shutil.which('node'):
            code="const L=require('./report_assets/logic.js'),p=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(p.rows.map(r=>L.acceptance(r,p.policy))));"
            proc=subprocess.run(['node','-e',code],input=json.dumps({'rows':rows[:2],'policy':p}),text=True,capture_output=True,check=True)
            self.assertEqual(json.loads(proc.stdout),['reviewed']*2)

    def test_risk_curve_costs_and_latency_require_complete_evidence(self):
        import decision_analysis as d
        rows=[self.row(_api_cost=.01,latency_s=.2)]
        curve=d.risk_curve(rows,{'mode':'probability','accept_none':True})
        self.assertEqual(curve[0]['accepted'],1)
        self.assertEqual(curve[0]['model_api_per_1k'],10)
        self.assertEqual(curve[0]['model_p95_ms'],200)
        curve=d.risk_curve(rows,planned=2)
        self.assertIsNone(curve[0]['model_api_per_1k'])

    def test_clustered_diagnostics_and_probability_coverage(self):
        import decision_analysis as d
        cases=[{'id':'a','dataset':'dev','task':'x','split':'development','cluster_id':'shared','input':'a','question':{'type':'choice','criteria':{'YES':'','CLARIFY':''}}},
               {'id':'b','dataset':'eval','task':'x','split':'evaluation','cluster_id':'shared','input':'b','question':{'type':'choice','criteria':{'YES':'','CLARIFY':''}}}]
        rows=[self.row(dataset='dev',task='x',model='m'),self.row(id='b',dataset='eval',task='x',model='m',probabilities=None)]
        result=d.diagnostics(cases,rows)
        self.assertTrue(result['family_overlap'][0]['boundary_overlap'])
        self.assertEqual(d.probability_quality(rows,cases)['probability_coverage'],.5)
        accepted=d.acceptance_summary(rows,{'mode':'label'})
        self.assertIsNone(accepted['accepted_ci_high'])

    def test_dataset_roles_flag_development_boundaries_but_not_stress_pairs(self):
        import decision_analysis as d
        from test_benchmark import case
        cases=[case('a',dataset='standard',family='shared'),case('b',dataset='stress',family='shared')]
        cases[0]['dataset_role']='standard';cases[1]['dataset_role']='stress'
        self.assertFalse(d.diagnostics(cases,[])['family_overlap'][0]['boundary_overlap'])
        cases[0]['dataset_role']='development';cases[1]['dataset_role']='calibration'
        self.assertTrue(d.diagnostics(cases,[])['family_overlap'][0]['boundary_overlap'])
