import io
import json
import unittest
from unittest.mock import patch

import benchmark as b
from test_benchmark import case,config,profile


class ResponseContractTests(unittest.TestCase):
    def call(self,c,enabled):
        model=profile(api='openai',endpoint='https://api.openai.com/v1/responses')
        model['structured_output']=enabled
        captured=[]
        class Response(io.BytesIO):
            headers={}
        def fetch(request,**kwargs):
            captured.append(json.loads(request.data))
            return Response(b'{"output":[{"type":"message","content":[{"type":"output_text","text":"{\\"value\\":\\"NONE\\"}"}]}]}')
        with patch('urllib.request.urlopen',side_effect=fetch):
            result=b.call_model(model,c,config(model))
        return captured[0],result

    def test_strict_choice_response_contract_uses_visible_options_only(self):
        c=case();c['gold_reason']='HIDDEN_REASON'
        request,result=self.call(c,True)
        self.assertTrue(result['valid'])
        fmt=request.get('text',{}).get('format',{})
        self.assertEqual(fmt.get('type'),'json_schema')
        self.assertTrue(fmt['strict'])
        self.assertEqual(fmt['schema']['properties']['value']['enum'],['billing','NONE'])
        self.assertEqual(fmt['schema']['required'],['value'])
        self.assertFalse(fmt['schema']['additionalProperties'])
        self.assertNotIn('HIDDEN_REASON',json.dumps(request))

    def test_noul_schema_requests_probability_number_not_class_string(self):
        c=case(expected=False);c['question']={'type':'noul','instructions':'Estimate probability'}
        request,_=self.call(c,True)
        self.assertEqual(request.get('text',{}).get('format',{}).get('schema',{}).get('properties',{}).get('value',{}).get('type'),'number')

    def test_existing_prompt_only_profile_is_preserved(self):
        request,result=self.call(case(),False)
        self.assertNotIn('text',request)
        self.assertTrue(result['valid'])

    def test_chat_endpoint_does_not_claim_unrequested_schema_enforcement(self):
        model=profile(api='openai',endpoint='https://api.openai.com/v1/chat/completions')
        model['structured_output']=True
        self.assertEqual(b.captured_profile(model)['response_contract'],
                         'prompt-only JSON; validated after response')
