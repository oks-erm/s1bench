import copy
import unittest
import tempfile
from pathlib import Path
import benchmark as b
import report_export as e
from test_benchmark import case, config, prediction, profile

class BusinessReportTests(unittest.TestCase):
    def test_policy_compliance_comment_is_scoped_to_rule_solvable_tasks(self):
        semantic=case('semantic',expected='VALID')
        semantic.update(task='semantic_validation',split='evaluation',dataset='shve_v2_evaluation',
                        input={'record':{'MinFillLevel':1,'MaxFillLevel':2,'MaxCapacity':3,
                                         'negative_ratio':0,'unit_metadata':dict.fromkeys(
                                             ['MinFillLevel','MaxFillLevel','MaxCapacity'],'litres')}})
        semantic['question']['criteria']={'VALID':'Checks pass','INVALID':'A check fails','CLARIFY':'Unknown'}
        sector=case('sector',expected='CLARIFY')
        sector.update(task='sector_imputation',split='evaluation',dataset='shve_v2_evaluation',
                      input={'record':{'activity_note':'unknown'}})
        sector['question']['criteria']={'Sector 1':'Industrial','CLARIFY':'Unknown'}
        cfg=config(profile());cfg['shve_protocol']={'dataset_revision':'test'}
        report={'folder':Path('test'),'config':cfg,'cases':[semantic,sector],
                'records':[prediction(semantic,value='VALID'),prediction(sector,value='CLARIFY')],
                'availability':{},'manifest':{}}
        interpretations=e.build_payload(report)['decision_analysis'].get('baseline_interpretations',[])
        self.assertEqual([(r['dataset'],r['task']) for r in interpretations],
                         [('shve_v2_evaluation','semantic_validation')])
        self.assertIn('policy compliance',interpretations[0]['interpretation'])
        # A failed rules cohort cannot justify this interpretation.
        semantic['expected']='INVALID'
        interpretations=e.build_payload(report)['decision_analysis'].get('baseline_interpretations',[])
        self.assertEqual(interpretations,[])

    def test_shve_report_explains_split_and_downstream_evidence(self):
        cases=[case('d',expected='CLARIFY'),case('e',expected='CLARIFY')]
        for c in cases:
            c.update(task='sector_imputation',input={'record':{'activity_note':'unknown'}})
            c['question']['criteria']={'Sector 1':'Industrial','CLARIFY':'Insufficient evidence'}
        cases[0].update(split='development',dataset='shve_v2_development')
        cases[1].update(split='evaluation',dataset='shve_v2_evaluation')
        cfg=config(profile());cfg['shve_protocol']={'dataset_revision':'test'}
        report={'folder':Path('test'),'config':cfg,'cases':cases,
                'records':[prediction(c,value='CLARIFY') for c in cases],
                'availability':{},'manifest':{}}
        payload=e.build_payload(report)
        self.assertEqual(payload['study_design']['split_counts'],{'development':1,'evaluation':1})
        self.assertIn('threshold',payload['study_design']['development_purpose'])
        self.assertIn('churn outcomes',payload['study_design']['downstream_requirements']['churn_uplift'])
        self.assertIn('need not be called',payload['study_design']['downstream_requirements']['routing_selection'])
        self.assertIn('intervention/control trial',payload['study_design']['downstream_requirements']['churn_uplift'])
        self.assertEqual(payload['warnings'],["Vision is policy classification only; downstream routing quality, savings and churn uplift were not measured."])
        self.assertIn('Download complete HTML',e.render_report(payload))

    def test_deployment_and_targets_separate_from_quality(self):
        c=case(expected='billing');c['critical_error_choices']=[]
        cfg=config(profile());rows=[prediction(c,value='billing')]
        summary=b.summarize(rows,[c],cfg)[0][0]
        self.assertEqual(summary['deployment_eligibility'],'unknown')
        self.assertEqual(len(summary['outstanding_targets']),3)
        self.assertEqual(summary['business_eligible_coverage'],0)
        self.assertEqual(summary['label_based_coverage'],1)

    def test_shve_human_review_is_not_automated_in_any_label_measure(self):
        c=case(expected='HUMAN_REVIEW');c['question']['criteria']['HUMAN_REVIEW']='Human decision required'
        cfg=config(profile());cfg['shve_protocol']={'dataset_revision':'test'}
        summary=b.summarize([prediction(c,value='HUMAN_REVIEW')],[c],cfg)[0][0]
        self.assertEqual(summary['auto_coverage'],0)
        self.assertEqual(summary['label_based_coverage'],0)

    def test_html_keeps_evidence_and_analysis_separate(self):
        c=case(expected='billing');cfg=config(profile());rows=[prediction(c,value='billing')]
        report={'folder':Path('test'),'config':cfg,'cases':[c],'records':rows,'availability':{},'manifest':{'analysis_costs':[{'original':True}]}}
        before=copy.deepcopy(report)
        payload=e.build_payload(report)
        self.assertEqual(payload['manifest']['original_run_costs'],[{'original':True}])
        self.assertIn('decision_analysis',payload)
        self.assertEqual(report,before)
        html=e.render_report(payload)
        for title in ['Acceptance and risk','Workflow economics','Dataset diagnostics','Deterministic baselines']:
            self.assertIn(title,html)
        self.assertNotIn('CLM embedding caches',html)

    def test_report_records_provider_reported_settings_without_inventing_defaults(self):
        c=case(expected='billing');cfg=config(profile());row=prediction(c,value='billing')
        row['raw']={'reasoning':{'effort':'medium'},'temperature':1,'authorization':'secret'}
        profiles=b.summarize([row],[c],cfg)[2]
        settings=profiles['m']['effective_settings']
        self.assertEqual(settings['explicit'],{})
        self.assertEqual(settings['provider_reported']['reasoning'],[{'effort':'medium'}])
        self.assertNotIn('authorization',settings['provider_reported'])
        self.assertNotIn('top_p',settings['provider_reported'])

    def test_adjudication_does_not_change_primary_score_or_gold(self):
        cases=[case('a',expected='billing',family='synthetic:retrieval:pattern-7'),case('b',expected='billing')]
        for c in cases:c['task']='retrieval'
        cfg=config(profile());rows=[prediction(cases[0],value='NONE'),prediction(cases[1],value='billing')]
        summary=b.summarize(rows,cases,cfg)[0][0]
        self.assertEqual(summary['success_rate'],.5)
        self.assertEqual(summary['adjudication']['success_without_flagged'],1)
        self.assertEqual(cases[0]['expected'],'billing')

    def test_rule_comparison_withholds_partial_model_difference(self):
        import decision_analysis as d
        c=case(expected='billing');cfg=config(profile());rows=[prediction(c,value='billing')]
        models=b.summarize(rows,[c],cfg)[0]
        rules=[dict(models[0],model='rules')]
        result=d.baseline_comparisons(models,rows,rules,[dict(rows[0],model='rules')])
        self.assertEqual(result[0]['incremental_accuracy'],0)
        self.assertIn('comparison_method',result[0])
        models[0]['status']='partial'
        result=d.baseline_comparisons(models,rows,rules,[dict(rows[0],model='rules')])
        self.assertIsNone(result[0]['incremental_accuracy'])
        self.assertEqual(result[0]['comparison_method'],'incomplete model cohort')

    def test_analysis_exports_artifacts_without_changing_frozen_evidence(self):
        import shve
        c=case(expected='CLARIFY');c.update(task='sector_imputation',split='evaluation',input={'record':{'activity_note':'unknown'}})
        c['question']['criteria']={'Sector 1':'Industrial','CLARIFY':'Insufficient evidence'}
        cfg=config(profile('jev'));cfg['shve_protocol']={'dataset_revision':'test'}
        rows=[prediction(c,alias='jev',value='CLARIFY')]
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            frozen={'raw.jsonl':b.jsonl_bytes(rows),'data.snapshot.jsonl':b.jsonl_bytes([c])}
            for name,content in frozen.items():(folder/name).write_bytes(content)
            b.write_json(folder/'config.snapshot.json',cfg)
            frozen['config.snapshot.json']=(folder/'config.snapshot.json').read_bytes()
            b.write_json(folder/'manifest.json',{'analysis_costs':[],'selected_models':['jev']})
            shve.export_analysis(folder)
            for name,content in frozen.items():self.assertEqual((folder/name).read_bytes(),content)
            for name in ['report.html','business_summary.md','business_decisions.csv','per_label.csv',
                         'risk_curves.csv','baseline.raw.jsonl','baseline_summary.csv','error_directions.csv']:
                self.assertTrue((folder/name).exists(),name)
