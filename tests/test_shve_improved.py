import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import benchmark as b
import shve_cases
import shve_improved as s
from churn_data import make_case


def cases():
    sample = shve_cases.build_cases()
    for row in sample:
        row['dataset'] = 'shve_improved_' + row['split']
    for month, label in [('2026-01', 0), ('2026-02', 1), ('2026-04', 0), ('2026-05', 1)]:
        sample.append(make_case({'ID_anon': month, 'snapshot_month': month,
                                 'ChurnNextMonth': label, 'tenure_months': 12},
                                population_count=100, sample_count=1, prior=.01, source_sha='a'*64))
    return sample


def config(sample):
    cfg = b.default_config()
    cfg['models'] = [{'name': alias, 'model': alias, 'api': 'openai' if alias=='gpt' else 'systemone',
                     'endpoint': 'https://api.openai.com/v1/responses' if alias=='gpt' else
                     ('https://api.typesafe.ai/v1/systemone' if alias=='jev' else 'http://127.0.0.1:8000/v1/systemone'),
                     'enabled': False, 'deployment': 'local' if alias in ('nimble','laya') else 'hosted',
                     'api_key': 'hidden-test-key' if alias in ('gpt','jev') else ''} for alias in s.MODELS]
    return s.run_config(cfg, sample)


class ImprovedRunnerTests(unittest.TestCase):
    def test_analysis_exports_unavailable_model_without_changing_evidence(self):
        # A startup failure has no valid-answer denominator; exporting it must
        # preserve unavailable values and the frozen zero-request evidence.
        sample=cases();cfg=config(sample)
        for model in cfg['models']:model['enabled']=model['name']=='laya'
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            b.save_report(folder,[],sample,cfg,{'laya':{'status':'unavailable','detail':'Startup timeout'}},
                          repeat_ids=[c['id'] for c in sample[:100]],
                          batches={c['id']:0 for c in sample},protocol=cfg['protocol'],
                          plan=b.request_plan(sample,cfg),models=[cfg['models'][-1]],started=b.utc())
            before={n:(folder/n).read_bytes() for n in
                    ('raw.jsonl','data.snapshot.jsonl','config.snapshot.json','manifest.json')}
            payload=s.export_analysis(folder)
            self.assertIn('| incomplete | unavailable | unavailable |',
                          (folder/'business_summary.md').read_text())
            self.assertEqual(payload['decision_analysis']['churn_prediction'][0]
                             ['fixed']['coverage']['weighted_probability_validity'],0)
            self.assertEqual(before,{n:(folder/n).read_bytes() for n in before})

    def test_config_preserves_profiles_freezes_counts_and_observed_contract(self):
        sample = cases(); cfg = config(sample)
        self.assertEqual([m['name'] for m in cfg['models']], list(s.MODELS))
        self.assertTrue(next(m for m in cfg['models'] if m['name']=='gpt')['structured_output'])
        self.assertEqual(cfg['tasks']['churn_prediction']['label_semantics'], 'observed_outcome')
        self.assertEqual(cfg['shve_protocol']['suite'], 'improved')
        self.assertEqual(b.request_plan(sample,cfg)['maximum_requests'], (len(sample)+202)*4)
        b.validate_data(sample,cfg)

    def test_baselines_use_inputs_and_churn_prior_not_expected(self):
        sample=cases(); cfg=config(sample)
        before=s.baseline_records(sample,cfg)
        altered=copy.deepcopy(sample)
        for c in altered:
            if c['task']=='churn_prediction':c['expected']=not c['expected']
        after=s.baseline_records(altered,cfg)
        self.assertEqual([r['value'] for r in before],[r['value'] for r in after])
        self.assertEqual([r['value'] for r in before if r['task']=='churn_prediction'],[.01]*4)

    def test_churn_threshold_uses_development_not_evaluation(self):
        sample=cases();cfg=config(sample);rows=s.baseline_records(sample,cfg)
        for model in cfg['models']:model['enabled']=model['name']=='gpt'
        for r in rows:r['model']='gpt'
        first=s.churn_summaries(rows,sample,cfg)
        changed=copy.deepcopy(rows)
        for r in changed:
            if r['dataset'].endswith('evaluation'):r['value']=.99
        second=s.churn_summaries(changed,sample,cfg)
        self.assertEqual(first[0]['threshold_selection'],second[0]['threshold_selection'])
        self.assertEqual(len(first),2)
        self.assertEqual(first[0]['fixed']['operating_point']['threshold'],.5)

    def test_churn_selected_model_with_no_attempts_keeps_planned_coverage(self):
        sample=cases();cfg=config(sample)
        for model in cfg['models']:model['enabled']=model['name']=='laya'
        result=s.churn_summaries([],sample,cfg)
        self.assertEqual(len(result),2)
        for cohort in result:
            self.assertEqual(cohort['model'],'laya')
            self.assertEqual(cohort['fixed']['sample']['unattempted'],2)
            self.assertEqual(cohort['fixed']['coverage']['weighted_probability_validity'],0)
            self.assertIsNone(cohort['selected'])

    def test_recovery_rejects_changed_inputs_and_complete_model(self):
        sample=cases();cfg=config(sample)
        report={'cases':sample,'config':b.scrub_config(cfg),'records':s.baseline_records(sample,cfg),
                'manifest':{'protocol':cfg['protocol'],'stability_sample_ids':[c['id'] for c in sample[:100]]}}
        for r in report['records']:r['model']='gpt'
        report['records'] += [{'model':'gpt','id':c['id'],'phase':'repeat','repetition':rep}
                              for rep in (2,3) for c in sample[:100]]
        report['records'] += [{'model':'gpt','id':c['id'],'phase':'warmup','repetition':1} for c in sample[:2]]
        with self.assertRaisesRegex(ValueError,'complete'):
            s.recovery_config(report,sample,cfg,['gpt'])
        changed=copy.deepcopy(sample);changed[0]['input']['context']='changed'
        with self.assertRaisesRegex(ValueError,'data differs'):
            s.recovery_config(report,changed,cfg,['laya'])

    def test_recovery_reenables_a_missing_model_without_changing_protocol(self):
        sample=cases();cfg=config(sample)
        for m in cfg['models']:m['enabled']=m['name']=='laya'
        report={'cases':sample,'config':b.scrub_config(cfg),'records':[],
                'manifest':{'protocol':cfg['protocol'],'stability_sample_ids':[c['id'] for c in sample[:100]]}}
        recovered=s.recovery_config(report,sample,cfg,['nimble'])
        self.assertEqual([m['name'] for m in recovered['models'] if m['enabled']],['nimble'])
        self.assertEqual(recovered['protocol'],cfg['protocol'])

    def test_load_rejects_a_self_consistent_hash_for_missing_categories(self):
        sample=cases();contents=b.jsonl_bytes(sample)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/'benchmark_data.jsonl').write_bytes(contents)
            b.write_json(folder/'manifest.json',{'revision':'shve-improved-v1-20261007',
                         'benchmark_data_sha256':hashlib.sha256(contents).hexdigest(),
                         'task_counts':dict(__import__('collections').Counter(c['task'] for c in sample))})
            with self.assertRaisesRegex(ValueError,'task counts'):
                s.load_suite(folder)


if __name__=='__main__':unittest.main()
