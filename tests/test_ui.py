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
