import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import benchmark as b
import report_export as export
from test_benchmark import case, config, prediction, profile


class ReportExportTests(unittest.TestCase):
    def fixture(self):
        cfg=config(profile("fast"), profile("reference"))
        cfg["business"]["baseline"]="reference"
        cfg["models"][0].update(api_key="never-share-key",headers={"Authorization":"never-share-header"})
        cases=[case(str(i),expected="billing" if i%2 else "NONE",dataset="A" if i<4 else "B") for i in range(8)]
        rows=[]
        for model in ("fast","reference"):
            for c in cases:
                r=prediction(c,alias=model,value=c["expected"] if model=="reference" else "NONE")
                r.update(raw={"response":"saved answer"},input_tokens=10,output_tokens=2,latency_s=.1 if model=="fast" else .5)
                rows.append(r)
        rows.append(dict(rows[0],phase="repeat",repetition=2))
        return {"folder":Path("example-run"),"config":cfg,"cases":cases,"records":rows,
                "manifest":{"protocol":cfg["protocol"],"quotes":{"fast":cfg["models"][0]}},"availability":{}}

    def test_complete_payload_matches_python_analysis_for_all_references(self):
        report=self.fixture()
        payload=export.build_payload(report)
        self.assertEqual(len(payload["cases"]),8)
        self.assertEqual(len(payload["records"]),17)
        self.assertEqual({r["dataset"] for r in payload["records"]},{"A","B"})
        for baseline,rows in payload["summaries_by_reference"].items():
            cfg=copy.deepcopy(report["config"])
            cfg["business"]["baseline"]=baseline
            expected=b.summarize(report["records"],report["cases"],cfg)[0]
            self.assertEqual(rows,expected)
        self.assertEqual(payload["confusion"],b.confusion_tables(report["records"]))
        html=export.render_report(payload)
        self.assertNotIn("never-share-key",html)
        self.assertNotIn("never-share-header",html)
        self.assertNotRegex(html,r'<(?:script|link)[^>]+(?:src|href)=')
        self.assertIn("connect-src 'none'",html)
        for name in ("Cost and forecast","Fallback scenario","Confusion matrices","Cases","Profiles and evidence","Protocol","Metric guide"):
            self.assertIn(name,html)

    def test_untrusted_case_strings_cannot_escape_embedded_json(self):
        report=self.fixture()
        malicious='</script><img src=x onerror=alert(1)>'
        report["cases"][0]["input"]=malicious
        report["records"][0]["raw"]={"message":malicious}
        html=export.complete_html_report(report)
        self.assertNotIn(malicious,html)
        text=re.search(r"<script id='report-data' type='application/json'>(.*?)</script>",html,re.S).group(1)
        self.assertEqual(json.loads(text)["cases"][0]["input"],malicious)

    @unittest.skipUnless(shutil.which("node"),"Node is only needed to validate offline browser calculations")
    def test_browser_fallback_matches_python_on_errors_missing_costs_and_currencies(self):
        report=self.fixture()
        payload=export.build_payload(report)
        records=payload["records"]
        records[0].update(valid=False,correct=False,api_error="HTTP_500",_api_cost=None,latency_s=None)
        records[1]["_api_cost"]=.007
        program="""const fs=require('fs'),L=require(process.argv[1]);
          const p=JSON.parse(fs.readFileSync(0,'utf8'));
          const out=p.scenarios.map(s=>L.fallback(p.records,s.summaries,s.dataset,'intent',s.baseline,new Set(s.labels)));
          console.log(JSON.stringify(out));"""
        scenarios=[]
        for baseline,summaries in payload["summaries_by_reference"].items():
            for dataset in ["A","B"]:
                for labels in [[],["NONE"],["NONE","billing"]]:
                    scenarios.append(dict(baseline=baseline,summaries=summaries,dataset=dataset,labels=labels))
        mixed=copy.deepcopy(scenarios[0]);mixed["summaries"][0]["currency"]="EUR";scenarios.append(mixed)
        partial=copy.deepcopy(scenarios[0]);partial["summaries"][0]["status"]="partial";scenarios.append(partial)
        proc=subprocess.run([shutil.which("node"),"-e",program,str(export.ASSETS/"logic.js")],
                            input=json.dumps({"records":records,"scenarios":scenarios}),text=True,capture_output=True,check=True)
        actual=json.loads(proc.stdout)
        for scenario,result in zip(scenarios,actual):
            expected=b.fallback_replay(records,scenario["summaries"],scenario["dataset"],"intent",scenario["baseline"],set(scenario["labels"]))
            self.assertEqual(len(expected),len(result))
            for py,js in zip(expected,result):
                for key in py:
                    if isinstance(py[key],float):self.assertAlmostEqual(py[key],js[key],places=12)
                    else:self.assertEqual(py[key],js[key])


if __name__ == '__main__':
    unittest.main()
