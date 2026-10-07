import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest

import report_export as export
from test_benchmark import case, config, prediction, profile


class ImprovedReportTests(unittest.TestCase):
    def fixture(self):
        cfg = config(profile('jev'), profile('gpt'))
        cfg['shve_protocol'] = {'suite': 'improved', 'dataset_revision': 'test',
                                'prediction_scope': 'Retrospective supplied-label discrimination; prospective feature availability was not verified.'}
        cases = []
        for split in ('development', 'evaluation'):
            for suffix, expected, weight in [('p', True, 2), ('n', False, 8)]:
                c = case(split + suffix, expected=expected, dataset='shve_improved_' + split)
                c.update(split=split, dataset_role=split, task='churn_prediction',
                         question={'type': 'noul', 'instructions': 'Estimate P(churn next month).'},
                         input={'months_active': 12, 'monthly_deliveries': 2 if expected else 5,
                                'historical_positive_rate': .3},
                         metadata={'sampling_weight': weight,
                                   'sampling_stratum': ('2026-01' if split == 'development' else '2026-04') + ':' + str(int(expected))})
                cases.append(c)
        # Real authored cases exercise the frozen lexical control without test doubles.
        import shve_cases
        for split in ('development', 'evaluation'):
            c = next(c for c in shve_cases.build_cases() if c['split'] == split and c['task'] == 'sector_imputation')
            c['dataset'] = 'shve_improved_' + split
            cases.append(c)
        records = [prediction(c, alias=model, value=(.8 if c['expected'] else .2)
                              if c['task'] == 'churn_prediction' else c['expected'])
                   for model in ('jev', 'gpt') for c in cases]
        return {'folder': Path('improved-test'), 'config': cfg, 'cases': cases,
                'records': records, 'manifest': {}, 'availability': {}}

    def test_improved_payload_exports_weighted_prediction_and_separate_thresholds(self):
        # Using the legacy baseline or omitting weights would make population evidence disappear.
        report = self.fixture()
        before = copy.deepcopy(report)
        payload = export.build_payload(report)
        self.assertEqual(payload['default_dataset'], 'shve_improved_evaluation')
        findings = payload['decision_analysis']['churn_prediction']
        evaluation = next(r for r in findings if r['model'] == 'jev' and r['dataset'] == 'shve_improved_evaluation')
        self.assertEqual(evaluation['fixed']['operating_point']['threshold'], .5)
        self.assertEqual(evaluation['fixed']['sample']['positive'], 1)
        self.assertEqual(evaluation['fixed']['population']['positive_prevalence'], .2)
        self.assertEqual(evaluation['fixed']['probability_metrics']['roc_auc'], 1)
        self.assertEqual(evaluation['fixed']['probability_metrics']['average_precision'], 1)
        self.assertIsNotNone(evaluation['selected'])
        self.assertEqual(evaluation['threshold_selection']['threshold'], .8)
        self.assertEqual(evaluation['baseline']['probability_metrics']['roc_auc'], .5)
        self.assertEqual(evaluation['baseline']['probability_metrics']['average_precision'], .2)
        for summaries in payload['summaries_by_reference'].values():
            summary = next(r for r in summaries if r['task'] == 'churn_prediction')
            self.assertIn('oversampled diagnostic', summary['quality_basis'])
            self.assertIn('churn_predictive_fixed', summary)
        self.assertTrue(all(r['task'] != 'churn_prediction' for r in payload['decision_analysis']['baseline_comparisons']))
        self.assertEqual(report, before)
        self.assertEqual(payload['study_design']['prediction_scope'], report['config']['shve_protocol']['prediction_scope'])
        json.dumps(payload, allow_nan=False)

    def test_improved_evidence_limit_and_control_scope_are_explicit(self):
        # A weak lexical control must not be presented as all possible code solutions.
        payload = export.build_payload(self.fixture())
        self.assertEqual(payload['warnings'], ['Model routing measures target selection; downstream answer quality, savings and churn uplift were not measured.'])
        self.assertIn('frozen simple control', payload['decision_analysis']['baseline_basis'])
        self.assertIn('general advantage over code', payload['decision_analysis']['baseline_basis'])

    @unittest.skipUnless(shutil.which('node'), 'Node unavailable')
    def test_no_valid_development_scores_or_evaluation_attempts_remain_visible(self):
        # A missing cohort must show abstentions rather than hiding its weighted matrix.
        report = self.fixture()
        report['records'] = [r for r in report['records']
                             if r['task'] != 'churn_prediction' or r['dataset'] != 'shve_improved_evaluation']
        for row in report['records']:
            if row['task'] == 'churn_prediction':
                row.update(value=None, valid=False, correct=False, decision=None)
        payload = export.build_payload(report)
        findings = [r for r in payload['decision_analysis']['churn_prediction']
                    if r['dataset'] == 'shve_improved_evaluation']
        self.assertEqual(len(findings), 2)
        for entry in findings:
            self.assertIsNone(entry['selected'])
            self.assertEqual(entry['fixed']['coverage']['weighted_probability_validity'], 0)
            self.assertEqual(entry['fixed']['operating_point']['recall'], 0)
            self.assertEqual(entry['fixed']['operating_point']['confusion_weighted']['actual_true']['invalid'], 2)
        proc = subprocess.run([shutil.which('node'), '-e', DOM_PROGRAM,
                               str(export.ASSETS / 'logic.js'), str(export.ASSETS / 'report.js')],
                              input=json.dumps(payload), text=True, capture_output=True, check=True)
        rendered = json.loads(proc.stdout)
        self.assertIn('Population-weighted estimated counts', rendered['Confusion matrices'])
        self.assertIn('No churn', rendered['Confusion matrices'])
        self.assertIn('Invalid / unattempted', rendered['Confusion matrices'])
        self.assertNotIn('Development-selected threshold', rendered['Quality'])

    @unittest.skipUnless(shutil.which('node'), 'Node unavailable')
    def test_browser_renders_weighted_quality_matrix_baseline_and_all_existing_tabs(self):
        # This executes the viewer, catching broken filters and misleading population displays.
        payload = export.build_payload(self.fixture())
        proc = subprocess.run([shutil.which('node'), '-e', DOM_PROGRAM,
                               str(export.ASSETS / 'logic.js'), str(export.ASSETS / 'report.js')],
                              input=json.dumps(payload), text=True, capture_output=True, check=True)
        result = json.loads(proc.stdout)
        self.assertEqual(result['dataset'], 'shve_improved_evaluation')
        self.assertEqual(len(result['tabs']), 14)
        self.assertIn('Cases', result['tabs'])
        self.assertIn('Profiles and evidence', result['tabs'])
        self.assertIn('Download complete HTML', result['downloads'])
        self.assertIn('Weighted churn prediction', result['Quality'])
        for metric in ['ROC-AUC', 'Average precision', 'Brier', 'Log loss', 'Precision', 'Recall', 'Probability coverage']:
            self.assertIn(metric, result['Quality'])
        self.assertIn('Fixed threshold 0.5', result['Quality'])
        self.assertIn('Development-selected threshold', result['Quality'])
        self.assertIn('oversampled diagnostic', result['Quality'])
        self.assertIn('Population-weighted estimated counts', result['Confusion matrices'])
        self.assertIn('Historical probability baseline', result['Deterministic baselines'])
        self.assertIn('Weighted churn ROC-AUC', result['Overview'])
        self.assertIn('prospective feature availability was not verified', result['Protocol'])


DOM_PROGRAM = r"""
const fs=require('fs'),vm=require('vm');
const payload=JSON.parse(fs.readFileSync(0,'utf8'));
class Element {
  constructor(tag,text=''){this.tagName=tag;this.children=[];this.ownText=text;this.style={};this.dataset={};this.attributes={};this.classList={add:()=>{}};}
  get textContent(){return this.ownText+this.children.map(c=>c.textContent).join(' ');}
  set textContent(value){this.ownText=String(value);this.children=[];}
  append(...children){this.children.push(...children);}
  prepend(...children){this.children.unshift(...children);}
  replaceChildren(...children){this.ownText='';this.children=children;}
  setAttribute(key,value){this.attributes[key]=value;}
  focus(){}
}
const ids=Object.fromEntries(['report-data','content','filters','tabs','subtitle','notices','summary-cards','downloads'].map(id=>[id,new Element('div')]));
ids['report-data'].textContent=JSON.stringify(payload);
const document={getElementById:id=>ids[id],createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag),createTextNode:text=>new Element('text',text),querySelectorAll:()=>[],documentElement:new Element('html'),body:new Element('body')};
const context=vm.createContext({document,ReportLogic:require(process.argv[1]),location:{protocol:'file:'},URLSearchParams});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const result={dataset:vm.runInContext('dataset',context),tabs:ids.tabs.children.map(e=>e.textContent),downloads:ids.downloads.textContent};
for(const tab of ['Quality','Confusion matrices','Deterministic baselines','Overview','Protocol']){
  vm.runInContext("cohort=JSON.stringify([dataset,'churn_prediction']);activeTab="+JSON.stringify(tab)+";render();",context);
  result[tab]=ids.content.textContent;
}
process.stdout.write(JSON.stringify(result));
"""


if __name__ == '__main__':
    unittest.main()
