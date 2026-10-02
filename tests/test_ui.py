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
                self.assertFalse(any(x.label=="Include in runs" for x in app.checkbox))
                self.assertFalse(any(x.key and x.key.startswith("run_model_") for x in app.tabs[0].checkbox))
                self.assertTrue(any(x.label=="Run complete benchmark" for x in app.button))
                self.assertFalse(any(x.label in {"Add model","Remove model"} for x in app.button))
                self.assertFalse(any(x.label in {"Add a model","Model to remove","Model for one-case check"} for x in app.selectbox))
                enabled=next(x for x in app.tabs[2].checkbox if x.label=="Jev")
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
        cfg=config(*(profile(name) for name in ("jev","laya","nimble","gpt","clm")))
        cfg["protocol"].update(repetitions=3,repeat_cases=2)
        cfg["business"]["baseline"]="gpt"
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
            with (patch.object(b,"BASE",Path(root)),
                  patch("urllib.request.urlopen",side_effect=AssertionError("No network in saved results")),
                  patch.object(b,"html_report",wraps=b.html_report) as export):
                app.run()
                self.assertEqual(len(app.exception),0,[x.message for x in app.exception])
                figures=export.call_args.args[3]
                self.assertEqual(len(figures),2)
                colors={trace.name:trace.marker.color for trace in figures[0].data}
                self.assertEqual(set(colors),{"jev","laya","nimble","gpt","clm"})
                self.assertEqual(len(set(colors.values())),5)
                self.assertEqual(len({trace.marker.color for trace in figures[1].data}),2)
                # Test the actual download path after Streamlit has rendered the charts.
                standalone=export._mock_wraps(*export.call_args.args)
                for figure in figures:
                    self.assertNotRegex(figure.to_json(),r'#[0]{4}[0-9]{2}')
                    for trace in figure.data:
                        self.assertIn(trace.marker.color,standalone)
                self.assertFalse(list(app.error),[x.value for x in app.error])
                self.assertTrue(any(x.value=="Business decision matrix" for x in app.subheader))
                self.assertTrue(any(x.label=="Cost price basis" for x in app.selectbox))
                self.assertTrue(any(x.label=="Protocol" for x in app.expander))
                self.assertTrue(any("Loaded report: "+result["folder"] in x.value for x in app.caption))
                from types import SimpleNamespace
                import queue
                app.session_state["job"]=SimpleNamespace(
                    finished=True,notified=True,folder=result["folder"],error=None,
                    issued=54,maximum=8010,messages=[],events=queue.Queue(),connection_test=False)
                app.run()
                next(x for x in app.button if x.label=="Load selected run").click().run()
                self.assertEqual(len(app.exception),0,[x.message for x in app.exception])
                self.assertFalse(any("Requests issued" in x.value for x in app.markdown))
                self.assertNotIn("job",app.session_state)


    def test_run_tab_uses_checkboxes_and_full_data_or_all_selected_connection_test(self):
        from streamlit.testing.v1 import AppTest
        source=(Path(__file__).resolve().parents[1]/"bench_ui.py").read_text("utf-8")
        calls=[]
        def fake_run(cfg,cases,cancel=None,progress=None,selected=None,limit=None,output_root=None,
                     connection_test=False,model_session=None):
            calls.append({"models":[m["name"] for m in cfg["models"] if m.get("enabled")],
                          "selected":selected,"cases":len(cases),"limit":limit,
                          "connection_test":connection_test,"managed":model_session is not None})
            return {"folder":None,"error":None,"interrupted":False}
        with tempfile.TemporaryDirectory() as root:
            app=AppTest.from_string(source.replace(
                'BASE = Path(__file__).resolve().parent','BASE = Path('+repr(root)+')'),default_timeout=30)
            with patch("local_runtime.can_manage",return_value=True),patch.object(b,"BASE",Path(root)),patch.object(b,"run",side_effect=fake_run),patch(
                    "urllib.request.urlopen",side_effect=AssertionError("No network in GUI test")):
                app.run()
                self.assertEqual(len(app.exception),0)
                self.assertEqual({x.label for x in app.tabs[2].checkbox},
                                 {"Jev","Laya","Nimble","GPT Luna","CLM v0.1 8B"})
                self.assertEqual(len(app.tabs[2].selectbox),0)
                self.assertFalse(any("effort=n/a" in x.value for x in app.tabs[2].caption))
                full=next(x for x in app.button if x.label=="Run complete benchmark")
                short=next(x for x in app.button if x.label=="Test selected models")
                self.assertTrue(full.disabled)
                self.assertTrue(short.disabled)
                next(x for x in app.tabs[2].checkbox if x.label=="Laya").check().run()
                next(x for x in app.tabs[2].checkbox if x.label=="Nimble").check().run()
                next(x for x in app.button if x.label=="Run complete benchmark").click().run()
                app.session_state["job"].thread.join(2)
                self.assertEqual(len(calls),1)
                self.assertEqual(calls[0]["models"],["laya","nimble"])
                self.assertEqual(calls[0]["selected"],["laya","nimble"])
                self.assertEqual(calls[0]["cases"],1400)
                self.assertIsNone(calls[0]["limit"])
                self.assertFalse(calls[0]["connection_test"])
                self.assertTrue(calls[0]["managed"])
                app.run()
                next(x for x in app.button if x.label=="Test selected models").click().run()
                app.session_state["job"].thread.join(2)
                self.assertEqual(len(calls),2)
                self.assertEqual(calls[1]["selected"],["laya","nimble"])
                self.assertTrue(calls[1]["connection_test"])
                self.assertEqual(len(app.exception),0,[x.message for x in app.exception])

    def test_short_result_shows_connections_and_raw_errors_without_quality_matrix(self):
        from streamlit.testing.v1 import AppTest
        from test_benchmark import case, profile, prediction, config
        source=(Path(__file__).resolve().parents[1]/"bench_ui.py").read_text("utf-8")
        cfg=config(profile("working"),profile("broken"))
        def fake_call(model,c,run_cfg):
            row=prediction(c,alias=model["name"])
            row.update(estimated_cost=0.,cost_currency="USD",raw={"mock":True})
            if model["name"]=="broken":
                row.update(api_error="HTTP_400",error_detail="Unsupported parameter",
                           valid=False,correct=False,raw={"error":"Unsupported parameter"})
            return row
        with tempfile.TemporaryDirectory() as root:
            with patch.object(b,"BASE",Path(root)),patch.object(b,"call_model",side_effect=fake_call):
                result=b.run(cfg,[case("a"),case("z")],connection_test=True,output_root=Path(root)/"results")
                b.write_json(Path(root)/"config.json",cfg)
            app=AppTest.from_string(source.replace(
                'BASE = Path(__file__).resolve().parent','BASE = Path('+repr(root)+')'),default_timeout=30)
            app.session_state["report_folder"]=result["folder"]
            with patch.object(b,"BASE",Path(root)),patch("urllib.request.urlopen",side_effect=AssertionError("No network in saved results")):
                app.run()
                self.assertEqual(len(app.exception),0,[x.message for x in app.exception])
                self.assertFalse(any(x.value=="Business decision matrix" for x in app.subheader))
                self.assertTrue(any(x.value=="Connection test" for x in app.subheader))
                status_tables=[x.value for x in app.dataframe if "Status" in x.value.columns]
                self.assertTrue(status_tables)
                self.assertEqual(set(status_tables[0]["Status"]),{"Responding","Failed"})
                self.assertTrue(any(x.label=="Cases" and x.value=="Warmup / connection checks" for x in app.selectbox))
