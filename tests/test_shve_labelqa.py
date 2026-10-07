import copy
from contextlib import nullcontext
import importlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import benchmark as b
import shve_cases
import shve_improved as s
import shve_recovery
from test_shve_improved import config


class LabelQaRevisionTests(unittest.TestCase):
    def source(self, root):
        cases = shve_cases.build_cases()[:4] + [c for c in shve_cases.build_cases() if c['task']=='label_qa']
        for c in cases:
            c['dataset']='shve_improved_'+c['split']
            if c['task']=='label_qa':
                c['question']['instructions'] = shve_cases.SECTOR_POLICY + ' Audit the assigned sector: KEEP, CORRECT or CLARIFY.'
        cfg=config(cases)
        cfg['protocol'].update(repeat_cases=2,batches=1,max_requests=280)
        repeats=[cases[0]['id'],cases[4]['id']]
        batches={c['id']:1 for c in cases}
        rows=[]
        for model in cfg['models']:
            scheduled=[(c,'primary',1) for c in cases]
            scheduled += [(c,'warmup',1) for c in cases[:2]]
            scheduled += [(next(c for c in cases if c['id']==i),'repeat',n) for n in (2,3) for i in repeats]
            for c,phase,n in scheduled:
                row=s.baseline_records([c],cfg)[0]
                row.update(model=model['name'],phase=phase,scored=phase=='primary',repetition=n,batch=1,request_index=len(rows)+1)
                rows.append(row)
        source=root/'source'; source.mkdir()
        # Combined older reports have no run.plan.json; recover batches from
        # their agreed frozen primary records rather than modifying them.
        b.save_report(source,rows,cases,cfg,{},repeat_ids=repeats,batches=batches,protocol=cfg['protocol'],
                      plan=b.request_plan(cases,cfg),models=cfg['models'],started=b.utc())
        return b.read_report(source)

    def test_seed_freezes_only_corrected_task_and_preserves_inputs_gold_and_repeat_ids(self):
        mod=importlib.import_module('shve_labelqa')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=self.source(root)
            before={p.name:p.read_bytes() for p in source['folder'].iterdir() if p.is_file()}
            seed=mod.prepare_seed(source,output_root=root)
            saved=b.read_report(seed)
            self.assertEqual(len(saved['cases']),60)
            self.assertEqual({c['task'] for c in saved['cases']},{'label_qa'})
            original={c['id']:c for c in source['cases']}
            for c in saved['cases']:
                self.assertEqual(c['input'],original[c['id']]['input'])
                self.assertEqual(c['expected'],original[c['id']]['expected'])
                self.assertIn('Sector 3 = Hospitality/Catering;',c['question']['instructions'])
            self.assertEqual(saved['manifest']['stability_sample_ids'],[source['cases'][4]['id']])
            self.assertEqual(len(shve_recovery.remaining_calls(saved,'gpt')),64)
            self.assertEqual(b.audit_report(saved),([],[]))
            self.assertEqual(before,{p.name:p.read_bytes() for p in source['folder'].iterdir() if p.is_file()})

    def test_revision_requires_complete_correction_and_preserves_other_task_evidence(self):
        mod=importlib.import_module('shve_labelqa')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'results').mkdir(); source=self.source(root)
            before={p.name:p.read_bytes() for p in source['folder'].iterdir() if p.is_file()}
            seed=b.read_report(mod.prepare_seed(source,output_root=root))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                mod.revise_report(source,seed,output_root=root)
            def inference(model,c,cfg):
                row=s.baseline_records([c],cfg)[0];row['model']=model['name'];return row
            with patch.object(b,'BASE',root),patch.object(b,'call_model',side_effect=inference):
                result=shve_recovery.recover(seed,seed['config'],list(s.MODELS),lambda m:nullcontext(),threading.Event())
            corrected=b.read_report(result['folder'])
            revised=b.read_report(mod.revise_report(source,corrected,output_root=root))
            self.assertEqual(b.audit_report(revised),([],[]))
            self.assertEqual(shve_recovery.remaining_calls(revised,'gpt'),[])
            self.assertEqual(len(revised['records']),len(source['records'])+8)
            old=[r for r in source['records'] if r['task']!='label_qa']
            kept=[r for r in revised['records'] if r['task']!='label_qa']
            for a,z in zip(old,kept):
                self.assertEqual({k:v for k,v in a.items() if k!='request_index'},
                                 {k:v for k,v in z.items() if k not in {'request_index','source_run','source_request_index'}})
            self.assertEqual(len([r for r in revised['records'] if r['task']=='label_qa' and b.primary_row(r)]),240)
            self.assertEqual(revised['manifest']['stability_sample_ids'],source['manifest']['stability_sample_ids'])
            self.assertTrue((revised['folder']/'category_revision.json').exists())
            self.assertEqual(before,{p.name:p.read_bytes() for p in source['folder'].iterdir() if p.is_file()})
            tampered=copy.deepcopy(corrected);tampered['cases'][0]['expected']='CORRECT'
            with self.assertRaises(ValueError):mod.revise_report(source,tampered,output_root=root)

    def test_second_interruption_keeps_completed_model_and_new_partial_slots(self):
        mod=importlib.import_module('shve_labelqa')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'results').mkdir();source=self.source(root)
            seed=b.read_report(mod.prepare_seed(source,output_root=root))
            def inference(model,c,cfg):
                row=s.baseline_records([c],cfg)[0];row['model']=model['name'];return row
            with patch.object(b,'BASE',root),patch.object(b,'call_model',side_effect=inference):
                first=shve_recovery.recover(seed,seed['config'],['gpt'],lambda m:nullcontext(),threading.Event())
            previous=b.read_report(first['folder'])
            cfg,selected=mod.execution_config(source,previous,source['config'])
            cancel=threading.Event()
            def stop_after_one(model,c,cfg):
                row=inference(model,c,cfg);cancel.set();return row
            with patch.object(b,'BASE',root),patch.object(b,'call_model',side_effect=stop_after_one):
                second=shve_recovery.recover(previous,cfg,selected,lambda m:nullcontext(),cancel)
            merged=b.read_report(mod.merge_progress(previous,b.read_report(second['folder']),selected,output_root=root))
            self.assertEqual(b.audit_report(merged),([],[]))
            self.assertEqual(len([r for r in merged['records'] if r['model']=='gpt']),64)
            self.assertEqual(len([r for r in merged['records'] if r['model']=='jev']),1)
            _,pending=mod.execution_config(source,merged,source['config'])
            self.assertEqual(pending,['jev','nimble','laya'])
            self.assertEqual(len(shve_recovery.remaining_calls(merged,'jev')),63)

    def test_execution_rejects_unrelated_cases_before_calls_and_skips_complete_aliases(self):
        mod=importlib.import_module('shve_labelqa')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'results').mkdir();source=self.source(root)
            seed=b.read_report(mod.prepare_seed(source,output_root=root))
            bad=copy.deepcopy(seed);bad['cases'][0]['input']={'evidence':['different scope']}
            with self.assertRaises(ValueError):mod.execution_config(source,bad,source['config'])
            def inference(model,c,cfg):
                row=s.baseline_records([c],cfg)[0];row['model']=model['name'];return row
            with patch.object(b,'BASE',root),patch.object(b,'call_model',side_effect=inference):
                result=shve_recovery.recover(seed,seed['config'],['gpt'],lambda m:nullcontext(),threading.Event())
            partial=b.read_report(result['folder'])
            cfg,selected=mod.execution_config(source,partial,source['config'])
            self.assertEqual(selected,['jev','nimble','laya'])
            self.assertFalse(next(m for m in cfg['models'] if m['name']=='gpt')['enabled'])
            with patch.object(b,'BASE',root),patch.object(b,'call_model',side_effect=inference):
                rest=shve_recovery.recover(partial,cfg,selected,lambda m:nullcontext(),threading.Event())
            combined=b.combine_reports([(partial['folder'],['gpt']),(rest['folder'],selected)],output_root=root)
            revised=b.read_report(mod.revise_report(source,b.read_report(combined['folder']),output_root=root))
            self.assertEqual(b.audit_report(revised),([],[]))
