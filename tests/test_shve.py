import copy
import io
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import benchmark as b
from test_benchmark import case, config, profile

class SHVETests(unittest.TestCase):
    def test_risk_labels_validated(self):
        for risks in [['billing'],['not-a-choice'],[['nested']],['NONE','NONE']]:
            c=case(expected='billing');c['critical_error_choices']=risks
            with self.assertRaises(ValueError):b.validate_data([c],config())

    def test_input_only_adapter_requests(self):
        c=case();c.update(gold_reason='SECRET_REASON',split='SECRET_SPLIT',scenario='SECRET_SCENARIO',
                         critical_error_choices=['billing'],review_status='SECRET_REVIEW')
        class Response(io.BytesIO):
            headers={}
        for api in ['systemone','openai','ollama']:
            captured=[]
            def fetch(req, **kwargs):
                captured.append(json.loads(req.data))
                return Response(b'{"answers":{"answer":{"type":"choice","choice":"NONE"}}}')
            with patch('urllib.request.urlopen',side_effect=fetch):b.call_model(profile(api=api),c,config())
            body=json.dumps(captured)
            for marker in ['SECRET_REASON','SECRET_SPLIT','SECRET_SCENARIO','SECRET_REVIEW','critical_error_choices','cluster_id','expected']:
                self.assertNotIn(marker,body)

    def test_rules_are_input_only_and_directional(self):
        import shve
        q={'type':'choice','criteria':{'MATCH':'','NO_MATCH':'','CLARIFY':''},'instructions':'delivery point'}
        self.assertEqual(shve.baseline('entity_match',{'left':{'delivery_point_ref':'a'},'right':{'delivery_point_ref':'b'}},q),'NO_MATCH')
        self.assertEqual(shve.baseline('entity_match',{'left':{'delivery_point_ref':'a','site_token':'x'},'right':{'delivery_point_ref':'a','site_token':'y'}},q),'CLARIFY')
        self.assertEqual(shve.baseline('semantic_validation',{'record':{'MinFillLevel':2,'MaxFillLevel':1,'MaxCapacity':3,'negative_ratio':0,'unit_metadata':dict.fromkeys(['MinFillLevel','MaxFillLevel','MaxCapacity'],'litres')}},q),'INVALID')
        self.assertEqual(shve.baseline('sector_imputation',{'record':{'activity_note':'unsupported activity','expected':'Sector 1'}},q),'CLARIFY')

    def test_four_model_guard(self):
        import shve
        cfg=b.default_config()
        for m in cfg['models']:
            if m['name']=='gpt':m['model']='configured-id'
        result=shve.run_config(cfg)
        self.assertEqual([m['name'] for m in result['models']],['jev','gpt','nimble','laya'])
        self.assertEqual(result['protocol']['max_requests'],5608)
        self.assertEqual(next(m for m in result['models'] if m['name']=='gpt')['model'],'configured-id')
        with self.assertRaises(ValueError):shve.run_config(dict(cfg,models=cfg['models'][:-2]))

    def test_inputs_and_plan_are_frozen_before_first_request(self):
        from test_benchmark import prediction
        cfg=config();c=case()
        with tempfile.TemporaryDirectory() as tmp:
            def call(model,item,settings):
                folders=[p for p in Path(tmp).iterdir() if p.is_dir()]
                self.assertEqual(len(folders),1)
                for filename in ('config.initial.json','data.snapshot.jsonl','run.plan.json'):
                    self.assertTrue((folders[0]/filename).exists(),filename)
                return prediction(item)
            with patch.object(b,'call_model',side_effect=call),patch.object(b,'refresh_prices',return_value=[]):
                result=b.run(cfg,[c],output_root=tmp)
            self.assertIsNone(result['error'])

    def test_nimble_digest_is_exact_and_no_pull(self):
        import shve
        class Response(io.BytesIO):
            pass
        urls=[]
        def fetch(request, **kwargs):
            url=request if isinstance(request,str) else request.full_url
            urls.append(url)
            return Response(json.dumps({'models':[{'name':'nimble:latest','digest':'sha256:abc'}]} if url.endswith('/tags') else {'details':{'quantization_level':'Q8_0'}}).encode())
        m={'name':'nimble','model':'nimble','endpoint':'http://127.0.0.1:11435/v1/systemone'}
        with patch('urllib.request.urlopen',side_effect=fetch):shve.capture_identity(m)
        self.assertEqual(m['identity_evidence']['digest'],'sha256:abc')
        self.assertTrue(all(u.endswith(('/api/tags','/api/show')) for u in urls))
