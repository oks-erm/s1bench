import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark as b

class DashboardTests(unittest.TestCase):
    def test_dashboard_starts_and_accepts_model_and_pricing_edits_without_network(self):
        from streamlit.testing.v1 import AppTest
        source=(Path(__file__).resolve().parents[1]/"bench_ui.py").read_text("utf-8")
        with tempfile.TemporaryDirectory() as root:
            app=AppTest.from_string(source.replace(
                'BASE = Path(__file__).resolve().parent',
                'BASE = Path('+repr(root)+')'),default_timeout=30)
            with patch.object(b,"BASE",Path(root)),patch("urllib.request.urlopen",side_effect=AssertionError("No network on page load")):
                app.run()
                self.assertEqual(len(app.exception),0,[x.message for x in app.exception])
                self.assertEqual(len(app.tabs),6)
                self.assertTrue(any(x.label=="Start benchmark" for x in app.button))
                enabled=next(x for x in app.checkbox if x.label=="Include in runs")
                enabled.check().run()
                self.assertEqual(len(app.exception),0,[x.message for x in app.exception])
                price=next(x for x in app.text_input if x.label=="Input price per million tokens")
                price.set_value("2").run()
                out=next(x for x in app.text_input if x.label=="Output price per million tokens")
                out.set_value("5").run()
                self.assertEqual(len(app.exception),0,[x.message for x in app.exception])
                self.assertEqual(app.session_state["cfg"]["models"][0]["pricing_per_million"]["input"],2.)


    def test_saved_results_render_and_reprice_without_model_calls(self):
        from streamlit.testing.v1 import AppTest
        from test_benchmark import case, profile, prediction, config
        source=(Path(__file__).resolve().parents[1]/"bench_ui.py").read_text("utf-8")
        cfg=config(profile("fast"),profile("reference"))
        cfg["protocol"].update(repetitions=3,repeat_cases=2)
        cfg["business"]["baseline"]="reference"
        def fake_call(model,c,run_cfg):
            row=prediction(c,alias=model["name"])
            row.update(estimated_cost=0.,cost_currency="USD",raw={"mock":True})
            return row
        with tempfile.TemporaryDirectory() as root:
            with patch.object(b,"BASE",Path(root)),patch.object(b,"call_model",side_effect=fake_call):
                result=b.run(cfg,[case("a"),case("z")],output_root=Path(root)/"results")
                b.write_json(Path(root)/"config.json",cfg)
            app=AppTest.from_string(source.replace(
                'BASE = Path(__file__).resolve().parent','BASE = Path('+repr(root)+')'),default_timeout=30)
            app.session_state["report_folder"]=result["folder"]
            with patch.object(b,"BASE",Path(root)),patch("urllib.request.urlopen",side_effect=AssertionError("No network in saved results")):
                app.run()
                self.assertEqual(len(app.exception),0,[x.message for x in app.exception])
                self.assertFalse(list(app.error),[x.value for x in app.error])
                self.assertTrue(any(x.value=="Business decision matrix" for x in app.subheader))
                self.assertTrue(any(x.label=="Cost price basis" for x in app.selectbox))
                self.assertTrue(any(x.label=="Protocol" for x in app.expander))
