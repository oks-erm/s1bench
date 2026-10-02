import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import local_clm_server as clm


class FakeEngine:
    def __init__(self):
        self.calls = []

    def answer(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {name: {"type": "noul", "noul": 0.9} for name in questions}}


class LocalCLMTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.server = clm.create_server(self.engine, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, body):
        request = Request(self.url + "/v1/systemone", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        return urlopen(request, timeout=5)

    def test_typed_request_reaches_engine_and_reports_exact_model(self):
        body = {"state": {"text": "Hello"}, "model": clm.MODEL_ID,
                "questions": {"answer": {"type": "noul", "instructions": "Greeting?"}}}
        with self.post(body) as response:
            result = json.load(response)
            self.assertIn("X-CLM-Latency-Ms", response.headers)
        self.assertEqual(result["model"], clm.MODEL_ID)
        self.assertEqual(result["answers"]["answer"]["noul"], 0.9)
        self.assertEqual(self.engine.calls, [(body["state"], body["questions"])])

    def test_invalid_requests_never_reach_model(self):
        for body in [[], {}, {"state": "hi", "questions": {}},
                     {"state": "hi", "model": "wrong", "questions": {"q": {"type": "noul"}}},
                     {"state": "hi", "questions": {"q": {"type": "choice", "criteria": []}}}]:
            with self.subTest(body=body), self.assertRaises(HTTPError) as caught:
                self.post(body)
            self.assertEqual(caught.exception.code, 400)
        self.assertFalse(self.engine.calls)

    def test_health_and_discovery_do_not_run_inference(self):
        with urlopen(self.url + "/health", timeout=5) as response:
            self.assertEqual(json.load(response)["status"], "ok")
        with urlopen(self.url + "/v1/models", timeout=5) as response:
            self.assertEqual(json.load(response)["data"][0]["id"], clm.MODEL_ID)
        self.assertFalse(self.engine.calls)

    def test_engine_failure_returns_error_not_success(self):
        def fail(*args):
            raise RuntimeError("test inference failure")
        self.engine.answer = fail
        with self.assertLogs(level="ERROR"), self.assertRaises(HTTPError) as caught:
            self.post({"state": "hi", "questions": {"q": {"type": "noul"}}})
        self.assertEqual(caught.exception.code, 500)
        self.assertNotIn("test inference failure", caught.exception.read().decode())


if __name__ == "__main__":
    unittest.main()
