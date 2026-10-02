import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark as b
from test_benchmark import case, config, prediction


class ConfusionTests(unittest.TestCase):
    def test_primary_counts_keep_errors_and_separate_models_cohorts_and_types(self):
        rows = [prediction(case("a")), prediction(case("b"),value="billing"),
                prediction(case("c"),value="not-an-option"),
                prediction(case("d"),phase="repeat",repetition=2),
                prediction(case("e"),phase="warmup"),
                prediction(case("f"),alias="other"),
                prediction(case("g",dataset="other-data"))]
        binary = dict(rows[0],type="noul",task="binary",expected=True,decision=False)
        score = dict(rows[0],type="score",task="score",expected=1,decision=1)
        rows.extend([binary,score])
        tables = b.confusion_tables(rows)
        self.assertEqual(len(tables),4)
        table = next(t for t in tables if (t["model"],t["dataset"],t["task"])==("m","test","intent"))
        self.assertEqual(table["counts"],{"NONE":{"NONE":1,"billing":1,"(invalid/API error)":1}})
        self.assertEqual(next(t for t in tables if t["task"]=="binary")["counts"],{"True":{"False":1}})

    def test_saved_report_includes_matrices_and_preserves_json_evidence(self):
        with tempfile.TemporaryDirectory() as root, patch.object(b,"call_model",side_effect=lambda m,c,cfg: prediction(c)):
            result = b.run(config(),[case()],output_root=root)
            folder = Path(result["folder"])
            self.assertIn("Confusion matrices",(folder/"report.html").read_text())
            self.assertEqual(b.parse((folder/"confusion.json").read_text()),{"m / test / intent":{"NONE":{"NONE":1}}})
            self.assertEqual(b.audit_report(b.read_report(folder)),([],[]))

    def test_html_escapes_labels_and_includes_zero_cells_and_error_column(self):
        tables = [{"model":"<model>","dataset":"test","task":"intent",
                   "counts":{"<gold>":{"<pred>":2,"(invalid/API error)":1}}}]
        html = b.html_report([],{}, {},confusion=tables)
        self.assertIn("&lt;model&gt;",html)
        self.assertIn("&lt;gold&gt;",html)
        self.assertNotIn("<pred>",html)
        self.assertIn("(invalid/API error)",html)
        self.assertIn(">0</td>",html)
        self.assertIn(">2</td>",html)
        self.assertIn("background:rgb(",html)


if __name__ == "__main__":
    unittest.main()
