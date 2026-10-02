import copy
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import benchmark as b
from starter_data import generate
from data_prompt import preparation_prompt


def case(identity="a", expected="NONE", family=None, dataset="test"):
    return {"id":identity,"dataset":dataset,"dataset_role":"production_holdout","task":"intent",
            "input":"Request "+identity,"expected":expected,"cluster_id":family or identity,
            "review_status":"reviewed","label_source":"human",
            "question":{"type":"choice","instructions":"Route the request.",
                        "criteria":{"billing":"Charges","NONE":"Outside support"}}}

def profile(alias="m", api="systemone", endpoint="http://127.0.0.1:9999/v1/systemone"):
    return {"name":alias,"display_name":alias,"enabled":True,"api":api,"endpoint":endpoint,
            "model":"mock-1","deployment":"local","params":{}}

def config(*models):
    cfg = b.default_config()
    cfg["models"] = list(models) or [profile()]
    cfg["protocol"].update(warmup_calls=0,batches=2,repeat_cases=0,repetitions=1,bootstrap_samples=100)
    cfg["business"]["baseline"] = cfg["models"][0]["name"]
    return cfg

def prediction(c, alias="m", value="NONE", phase="primary", repetition=1):
    valid, correct, error, decision = b.grade(value,c,config())
    return {"model":alias,"id":c["id"],"dataset":c["dataset"],"task":c["task"],"type":c["question"]["type"],
            "cluster_id":c["cluster_id"],"expected":c["expected"],"critical":c.get("critical",False),
            "value":value,"decision":decision,"valid":valid,"correct":correct,"numeric_error":error,
            "api_error":None,"latency_s":.1,"phase":phase,"scored":phase=="primary",
            "repetition":repetition,"resolved_model":"mock-1","input_tokens":10,"output_tokens":1}


class CoreTests(unittest.TestCase):
    def test_starter_size_validation_family_grouping(self):
        cases = generate()
        self.assertEqual(len(cases),1400)
        self.assertEqual(len({c["task"] for c in cases}),12)
        b.validate_data(cases,config())
        self.assertLess(len({c["cluster_id"] for c in cases}),len(cases))
        self.assertTrue(any(c["expected"]=="NONE" for c in cases))
        for c in cases:
            kind = c["question"]["type"]
            gold_value = float(c["expected"]) if kind=="noul" else c["expected"]
            self.assertTrue(b.grade(gold_value,c,config())[1],c["id"])

    def test_strict_data_rejects_contradictions(self):
        a = case()
        z = copy.deepcopy(a)
        z.update(id="b",expected="billing")
        with self.assertRaisesRegex(ValueError,"contradictory"):
            b.validate_data([a,z],config())
        z["expected"] = "NONE"
        z["cluster_id"] = "other"
        with self.assertRaisesRegex(ValueError,"independent families"):
            b.validate_data([a,z],config())

    def test_json_duplicates_and_nan_rejected(self):
        for text in ['{"x":1,"x":2}','{"x":NaN}']:
            with self.assertRaises(ValueError):
                b.parse(text)

    def test_pluggable_data_and_manifest(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"a.jsonl").write_bytes(b.jsonl_bytes([case(dataset="A")]))
            (root/"b.jsonl").write_bytes(b.jsonl_bytes([case(dataset="B")]))
            b.write_json(root/"manifest.json",{"datasets":[
                {"name":"holdout_A","path":"a.jsonl","role":"production_holdout"},
                {"name":"holdout_B","path":"b.jsonl","role":"production_holdout"}]})
            cases = b.load_data(root/"manifest.json",cfg)
            self.assertEqual(len({c["id"] for c in cases}),2)
            self.assertEqual({c["dataset"] for c in cases},{"holdout_A","holdout_B"})

    def test_local_zero_and_paid_unknown_and_repricing(self):
        model = profile()
        self.assertEqual(b.api_cost({},model),0)
        hosted = {**model,"deployment":"hosted"}
        self.assertIsNone(b.api_cost({},hosted))
        hosted["pricing_per_million"]={"input":2,"output":8}
        self.assertAlmostEqual(b.api_cost({"input_tokens":100,"output_tokens":20},hosted),.00036)
        archived = config({**hosted,"pricing_per_million":None})
        current = config(hosted)
        c = case()
        r = prediction(c)
        summaries,*_ = b.summarize([r],[c],archived,current_prices=current)
        self.assertEqual(summaries[0]["cost_coverage"],1)
        self.assertAlmostEqual(summaries[0]["api_per_1k"],.028)
        current["models"][0]["model"]="different-id"
        summaries,*_ = b.summarize([r],[c],archived,current_prices=current)
        self.assertIsNone(summaries[0]["api_per_1k"])

    def test_inline_key_wins_and_snapshots_remove_it(self):
        cfg = config({**profile(),"api_key":"inline","api_key_env":"TEST_BENCH_KEY"})
        with patch.dict("os.environ",{"TEST_BENCH_KEY":"env"}):
            self.assertEqual(b.model_key(cfg["models"][0]),"inline")
        self.assertNotIn("api_key",b.scrub_config(cfg)["models"][0])
        self.assertEqual(b.model_key({"api_key_env":"TEST_BENCH_KEY"}),"")

    def test_family_batches_and_partial_comparison(self):
        cases = [case("a",family="shared"),case("b",family="shared"),case("c")]
        batches = b.assign_batches(cases,2,42)
        self.assertEqual(batches["a"],batches["b"])
        summaries,*_ = b.summarize([prediction(cases[0])],cases,config())
        self.assertNotEqual(summaries[0]["status"],"complete")
        self.assertEqual(summaries[0]["planned"],3)
        self.assertAlmostEqual(summaries[0]["success_rate"],1/3)

    def test_paired_intervals_and_boundary_caution(self):
        cases = [case(str(i)) for i in range(20)]
        rows = [prediction(c) for c in cases]
        lo,hi,method = b.paired_interval(rows,rows)
        self.assertLess(lo,0)
        self.assertGreater(hi,0)
        self.assertIsNone(b.paired_interval(rows,rows[:-1])[0])
        rows[1]["cluster_id"]=rows[0]["cluster_id"]
        self.assertIsNone(b.success_interval(rows)[0])

    def test_repeats_do_not_inflate_primary_and_invalid_groups_fail(self):
        c = case()
        cfg = config()
        cfg["protocol"]["repetitions"]=3
        rows = [prediction(c),prediction(c,phase="repeat",repetition=2),
                prediction(c,value="invalid",phase="repeat",repetition=3)]
        summaries,_,_,_,repeat,_ = b.summarize(rows,[c],cfg,repeat_ids=[c["id"]])
        self.assertEqual(summaries[0]["attempted"],1)
        self.assertEqual(repeat[0]["complete"],1)
        self.assertEqual(repeat[0]["agreement"],0)

    def test_cross_dataset_robustness_pairs(self):
        a,z = case("a",dataset="standard"),case("z",dataset="stress")
        for c in (a,z):
            c.update(pair_id="unique",pair_relation="distractor")
        result = b.robustness_summary([prediction(a),prediction(z)],[a,z],config())
        self.assertEqual(result[0]["complete_pairs"],1)
        self.assertEqual(result[0]["all_pairs_correct"],1)

    def test_binary_risk_direction(self):
        c = case()
        c.update(task="authorization",expected=False,question={"type":"noul","instructions":"Approved?"})
        r = prediction(c,value=1.)
        metric = b.risk_metrics([r],"authorization","noul",{})
        self.assertEqual(metric["risk_exposures"],1)
        self.assertEqual(metric["unsafe_decisions"],1)

    def test_prompt_and_budget(self):
        self.assertIn("JSONL",preparation_prompt())
        cases = generate()
        cfg = config(*[profile(str(i)) for i in range(5)])
        cfg["protocol"] = b.default_config()["protocol"]
        self.assertEqual(b.request_plan(cases,cfg)["maximum_requests"],8010)
        b.validate_config(cfg)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.requests=[]
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path,body,self.headers.get("Authorization")))
                if self.path=="/fail":
                    self.send_response(400)
                    result={"error":{"message":"Unsupported parameter: temperature"}}
                elif self.path.endswith("/systemone"):
                    self.send_response(200)
                    result={"model":"mock-1","answers":{"answer":{"type":"choice","choice":"NONE"}},
                            "usage":{"input_tokens":10,"output_tokens":1}}
                elif self.path.endswith("/responses"):
                    self.send_response(200)
                    result={"model":"mock-1","output":[{"type":"message","content":[
                        {"type":"output_text","text":'{"value":"NONE"}'}]}],
                        "usage":{"input_tokens":10,"output_tokens":1}}
                else:
                    self.send_response(200)
                    content='not JSON' if self.path=="/invalid" else '{"value":"NONE"}'
                    result={"model":"mock-1","choices":[{"message":{"content":content}}],
                            "message":{"content":content},"usage":{"prompt_tokens":10,"completion_tokens":1}}
                raw=json.dumps(result).encode()
                self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        self.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url=f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_adapters_native_responses_chat_ollama(self):
        for api,path in [("systemone","/v1/systemone"),("openai","/v1/responses"),
                         ("openai","/v1/chat/completions"),("ollama","/api/chat")]:
            model=profile(api,api,self.url+path)
            model["api_key"]="test-secret"
            row=b.call_model(model,case(),config(model))
            self.assertTrue(row["correct"],row)
            self.assertGreaterEqual(row["latency_s"],0)
            request_path,body,auth=self.requests[-1]
            self.assertEqual(auth,"Bearer test-secret")
            self.assertNotIn("temperature",body)
            if api=="systemone":
                self.assertIn("questions",body)
            if path.endswith("/responses"):
                self.assertIn("instructions",body)

    def test_http400_full_error_and_malformed_answer(self):
        failed=b.call_model(profile(endpoint=self.url+"/fail"),case(),config())
        self.assertEqual(failed["api_error"],"HTTP_400")
        self.assertIn("Unsupported parameter",failed["error_detail"])
        invalid=b.call_model(profile(api="openai",endpoint=self.url+"/invalid"),case(),config())
        self.assertFalse(invalid["valid"])
        self.assertIsNone(invalid["api_error"])

    def test_unavailable_model_does_not_prevent_good_run(self):
        good=profile("good",endpoint=self.url+"/v1/systemone")
        bad=profile("bad",endpoint=self.url+"/fail")
        cfg=config(good,bad)
        cfg["protocol"].update(warmup_calls=1)
        with tempfile.TemporaryDirectory() as root:
            result=b.run(cfg,[case("a"),case("z")],output_root=root)
            self.assertIsNone(result["error"])
            report=b.read_report(result["folder"])
            self.assertEqual(report["availability"]["bad"]["status"],"unavailable")
            primary=[r for r in report["records"] if b.primary_row(r)]
            self.assertEqual({r["model"] for r in primary},{"good"})
            self.assertEqual(len(primary),2)
            self.assertEqual(b.audit_report(report)[0],[])
            raw=Path(result["folder"])/"raw.jsonl"
            raw.write_text(raw.read_text()+"\n",encoding="utf-8")
            self.assertTrue(b.audit_report(report)[0])

    def test_stop_saves_partial_and_does_not_issue_next_request(self):
        cancel=threading.Event()
        def progress(event):
            if event.get("row"):
                cancel.set()
        cfg=config(profile(endpoint=self.url+"/v1/systemone"))
        with tempfile.TemporaryDirectory() as root:
            result=b.run(cfg,[case("a"),case("z")],cancel=cancel,progress=progress,output_root=root)
            self.assertTrue(result["interrupted"])
            report=b.read_report(result["folder"])
            self.assertEqual(len(report["records"]),1)
            self.assertEqual(report["manifest"]["issued_requests"],1)
            self.assertEqual(b.audit_report(report)[0],[])


if __name__=="__main__":
    unittest.main()
