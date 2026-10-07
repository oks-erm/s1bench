import unittest
from contextlib import nullcontext
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import shve_recovery as recovery
import benchmark as b
import shve_improved as s
import shve_cases
from test_shve_improved import config


class RecoveryPlanTests(unittest.TestCase):
    def report(self):
        return {'cases':[{'id':'a'},{'id':'b'}],
                'manifest':{'protocol':{'warmup_calls':2,'repetitions':3},
                            'stability_sample_ids':['a']},
                'records':[{'model':'m','id':'a','phase':'warmup','repetition':1},
                           {'model':'m','id':'b','phase':'warmup','repetition':1},
                           {'model':'m','id':'a','phase':'primary','repetition':1,'scored':True},
                           {'model':'m','id':'b','phase':'primary','repetition':1,'scored':True},
                           {'model':'m','id':'a','phase':'repeat','repetition':2}]}

    def test_missing_repeat_only_does_not_rerun_completed_primary_calls(self):
        pending=recovery.remaining_calls(self.report(),'m')
        self.assertEqual([(c['id'],phase,repetition) for c,phase,repetition in pending],[('a','repeat',3)])

    def test_complete_full_protocol_has_no_pending_calls(self):
        report=self.report()
        report['records'].append({'model':'m','id':'a','phase':'repeat','repetition':3})
        self.assertEqual(recovery.remaining_calls(report,'m'),[])

    def test_missing_alias_gets_exact_original_protocol(self):
        pending=recovery.remaining_calls(self.report(),'missing')
        self.assertEqual(len(pending),6)
        self.assertEqual(sum(phase=='warmup' for _,phase,_ in pending),2)
        self.assertEqual(sum(phase=='primary' for _,phase,_ in pending),2)

    def test_duplicate_record_is_rejected_not_silently_overwritten(self):
        report=self.report();report['records'].append(report['records'][2].copy())
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            recovery.remaining_calls(report,'m')

    def test_real_saved_recovery_preserves_evidence_and_calls_only_missing_slot(self):
        cases=shve_cases.build_cases()[:4]
        for case in cases:case['dataset']='shve_improved_'+case['split']
        cfg=config(cases)
        for m in cfg['models']:m['enabled']=m['name']=='gpt'
        cfg['protocol'].update(repeat_cases=1,batches=1,bootstrap_samples=100,max_requests=8)
        full=s.baseline_records(cases,cfg)
        for i,row in enumerate(full):row.update(model='gpt',batch=1,request_index=i+1,repetition=1)
        for phase,case,repetition in [('warmup',cases[0],1),('warmup',cases[1],1),('repeat',cases[0],2)]:
            row=s.baseline_records([case],cfg)[0]
            row.update(model='gpt',phase=phase,scored=False,repetition=repetition,batch=1,request_index=len(full)+1)
            full.append(row)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'results/source';source.mkdir(parents=True)
            batches={c['id']:1 for c in cases};repeats=[cases[0]['id']]
            b.write_json(source/'run.plan.json',{'batches':batches,'stability_sample_ids':repeats,
                                              'protocol':cfg['protocol'],'request_serialization':'preserve_order'})
            b.save_report(source,full,cases,cfg,{'gpt':{'status':'stopped'}},repeat_ids=repeats,batches=batches,
                          protocol=cfg['protocol'],plan=b.request_plan(cases,cfg),models=[cfg['models'][1]],started=b.utc())
            before={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
            previous=b.read_report(source)
            called=[]
            def inference(model,case,cfg):
                called.append(case['id']);row=s.baseline_records([case],cfg)[0];row['model']=model['name'];return row
            with patch.object(b,'BASE',root),patch.object(b,'call_model',side_effect=inference):
                result=recovery.recover(previous,cfg,['gpt'],lambda m:nullcontext(),threading.Event())
            saved=b.read_report(result['folder'])
            self.assertEqual(called,[cases[0]['id']])
            self.assertEqual(b.audit_report(saved),([],[]))
            self.assertEqual(len(saved['records']),8)
            self.assertEqual(recovery.remaining_calls(saved,'gpt'),[])
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir() if p.is_file()})
            combined=b.combine_reports([(result['folder'],['gpt'])],output_root=root/'combined')
            self.assertEqual(b.audit_report(b.read_report(combined['folder'])),([],[]))


if __name__=='__main__':unittest.main()
