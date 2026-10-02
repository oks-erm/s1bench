from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import benchmark as b
from test_benchmark import case, config, prediction, profile


class SequentialRunTests(unittest.TestCase):
    def test_duplicate_run_is_rejected_before_model_calls_and_lock_releases(self):
        with tempfile.TemporaryDirectory() as root, patch.object(b, "call_model",
                side_effect=lambda m, c, cfg: prediction(c, alias=m["name"])) as call:
            with b.run_lock(root):
                with self.assertRaisesRegex(ValueError, "already running"):
                    b.run(config(), [case()], output_root=root)
                call.assert_not_called()
            result = b.run(config(), [case()], output_root=root)
            self.assertIsNone(result["error"])
            call.assert_called_once()

    def test_five_models_share_cases_protocol_and_report_without_overlap(self):
        models = [profile(name) for name in ("laya", "nimble", "clm", "jev", "gpt")]
        cfg = config(*models)
        cfg["protocol"].update(warmup_calls=1, repeat_cases=2, repetitions=3)
        cases = [case(str(i)) for i in range(4)]
        active, events = [], []

        @contextmanager
        def session(model):
            self.assertFalse(active)
            active.append(model["name"])
            events.append(("start", model["name"]))
            try:
                yield
            finally:
                events.append(("stop", active.pop()))

        def call(model, item, cfg):
            self.assertEqual(active, [model["name"]])
            return prediction(item, alias=model["name"])

        with tempfile.TemporaryDirectory() as root, patch.object(b, "call_model", side_effect=call):
            result = b.run(cfg, cases, output_root=root, model_session=session)
            report = b.read_report(result["folder"])
            self.assertIsNone(result["error"])
            self.assertEqual(len(report["records"]), 45)
            self.assertEqual(report["manifest"]["execution_order"], "model_by_model")
            self.assertEqual(report["manifest"]["selected_models"], [m["name"] for m in models])
            for model in models:
                rows = [r for r in report["records"] if r["model"] == model["name"]]
                self.assertEqual({r["id"] for r in rows if b.primary_row(r)}, {c["id"] for c in cases})
                self.assertEqual(sum(r["phase"] == "repeat" for r in rows), 4)
            self.assertEqual(events, [(event, m["name"]) for m in models for event in ("start", "stop")])
            self.assertEqual(b.audit_report(report), ([], []))
            html = (Path(result["folder"]) / "report.html").read_text()
            self.assertTrue(all(m["name"] in html for m in models))

    def test_cancel_unloads_current_model_and_does_not_start_next(self):
        cancel, stopped, started = threading.Event(), [], []

        @contextmanager
        def session(model):
            started.append(model["name"])
            try:
                yield
            finally:
                stopped.append(model["name"])

        def call(model, item, cfg):
            cancel.set()
            return prediction(item, alias=model["name"])

        with tempfile.TemporaryDirectory() as root, patch.object(b, "call_model", side_effect=call):
            result = b.run(config(profile("a"), profile("b")), [case()], cancel=cancel,
                           output_root=root, model_session=session)
            self.assertTrue(result["interrupted"])
            self.assertEqual(started, ["a"])
            self.assertEqual(stopped, ["a"])
            self.assertEqual(b.audit_report(b.read_report(result["folder"])), ([], []))

    def test_startup_failure_is_visible_and_other_models_still_run(self):
        @contextmanager
        def session(model):
            if model["name"] == "broken":
                raise RuntimeError("Model startup failed")
            yield

        with tempfile.TemporaryDirectory() as root, patch.object(b, "call_model",
                side_effect=lambda m, c, cfg: prediction(c, alias=m["name"])):
            result = b.run(config(profile("broken"), profile("good")), [case()],
                           output_root=root, model_session=session)
            report = b.read_report(result["folder"])
            self.assertEqual(report["availability"]["broken"]["status"], "unavailable")
            self.assertEqual(report["availability"]["good"]["status"], "ready")
            summaries, *_ = b.summarize(report["records"], report["cases"], report["config"], report["availability"])
            self.assertEqual(next(r for r in summaries if r["model"] == "broken")["attempted"], 0)
            self.assertEqual(b.audit_report(report), ([], []))


if __name__ == "__main__":
    unittest.main()
