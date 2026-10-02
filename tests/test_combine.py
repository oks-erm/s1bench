from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark as b
from test_benchmark import case, config, prediction, profile


class CombineTests(unittest.TestCase):
    def make_runs(self, root, changed_data=False, changed_protocol=False):
        cfg = config(profile("a"), profile("b"), profile("c"))
        cfg["protocol"].update(warmup_calls=1, repeat_cases=2, repetitions=2)
        cases = [case("a"), case("b"), case("c")]
        first = b.run(cfg, cases, selected=["a", "b"], output_root=root)["folder"]
        if changed_data:
            cases[0]["expected"] = "billing"
        if changed_protocol:
            cfg["protocol"]["repetitions"] = 3
        cfg["models"][2]["notes"] = "Bounded allocator cache"
        second = b.run(cfg, cases, selected=["c"], output_root=root)["folder"]
        return first, second

    @staticmethod
    def call(model, item, cfg):
        return prediction(item, alias=model["name"])

    def test_combines_complete_models_without_calls_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as root, patch.object(b, "call_model", side_effect=self.call) as call:
            first, second = self.make_runs(root)
            # A source can have been stopped after the explicitly selected models finished.
            manifest = b.read_report(first)["manifest"]
            manifest["interrupted"] = True
            b.write_json(Path(first)/"manifest.json", manifest)
            b.seal(first)
            calls_before = call.call_count
            combined = b.combine_reports([(first, ["a", "b"]), (second, ["c"])], output_root=root)
            self.assertEqual(call.call_count, calls_before)
            report = b.read_report(combined["folder"])
            self.assertEqual(len(report["records"]), 18)
            self.assertEqual(b.audit_report(report), ([], []))
            self.assertEqual(report["manifest"]["selected_models"], ["a", "b", "c"])
            self.assertFalse(report["manifest"]["interrupted"])
            self.assertEqual(len(report["manifest"]["sources"]), 2)
            self.assertEqual(report["config"]["models"][2]["notes"], "Bounded allocator cache")
            self.assertEqual([r["request_index"] for r in report["records"]], list(range(1, 19)))
            self.assertTrue(all(r.get("source_run") for r in report["records"]))

    def test_rejects_mismatched_data_and_protocol(self):
        for field in ("changed_data", "changed_protocol"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root, patch.object(b, "call_model", side_effect=self.call):
                first, second = self.make_runs(root, **{field: True})
                with self.assertRaises(ValueError):
                    b.combine_reports([(first, ["a", "b"]), (second, ["c"])], output_root=root)

    def test_rejects_duplicate_and_incomplete_models(self):
        with tempfile.TemporaryDirectory() as root, patch.object(b, "call_model", side_effect=self.call):
            first, second = self.make_runs(root)
            for selections in ([(first, ["a"]), (first, ["a"])], [(first, ["c"]) ]):
                with self.assertRaises(ValueError):
                    b.combine_reports(selections, output_root=root)

    def test_rejects_tampered_source(self):
        with tempfile.TemporaryDirectory() as root, patch.object(b, "call_model", side_effect=self.call):
            first, second = self.make_runs(root)
            raw = Path(first)/"raw.jsonl"
            raw.write_text(raw.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "audit"):
                b.combine_reports([(first, ["a", "b"]), (second, ["c"])], output_root=root)

    def test_rejects_changed_option_order_or_serialization(self):
        for change in ("order", "serialization"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as root, patch.object(b, "call_model", side_effect=self.call):
                first, second = self.make_runs(root)
                report = b.read_report(second)
                if change == "order":
                    criteria = report["cases"][0]["question"]["criteria"]
                    report["cases"][0]["question"]["criteria"] = dict(reversed(list(criteria.items())))
                    (Path(second)/"data.snapshot.jsonl").write_bytes(b.jsonl_bytes(report["cases"]))
                else:
                    report["manifest"].pop("request_serialization")
                    b.write_json(Path(second)/"manifest.json", report["manifest"])
                b.seal(second)
                with self.assertRaisesRegex(ValueError, "datasets|serialization"):
                    b.combine_reports([(first, ["a", "b"]), (second, ["c"])], output_root=root)

    def test_legacy_combination_keeps_original_serialization(self):
        with tempfile.TemporaryDirectory() as root, patch.object(b, "call_model", side_effect=self.call):
            first, second = self.make_runs(root)
            for folder in (first, second):
                report = b.read_report(folder)
                report["manifest"].pop("request_serialization")
                b.write_json(Path(folder)/"manifest.json", report["manifest"])
                b.seal(folder)
            combined = b.combine_reports([(first, ["a", "b"]), (second, ["c"])], output_root=root)
            self.assertEqual(b.read_report(combined["folder"])["manifest"]["request_serialization"], "sorted_keys")


if __name__ == "__main__":
    unittest.main()
