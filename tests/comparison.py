"""Compatibility and ordering checks for saved validation runs."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.comparison import load_comparison
from scripts.evaluate import compute_metrics, fingerprint


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mytorch-comparison-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.validation = self.work / "validation.txt"
        self.validation.write_text("fixed validation fixture")
        self.metrics = compute_metrics(["Nothing", "Check", "Checkmate"],
                                       ["Nothing", "Check", "Checkmate"])
        self.metrics.update(prediction_seconds=1.0, artifacts={
            "dataset": fingerprint(self.validation), "analyzer": {"sha256": "a" * 64}})

    def report(self, name, metrics=None):
        directory = self.work / name
        directory.mkdir()
        (directory / "metrics.json").write_text(json.dumps(metrics or self.metrics))
        return directory

    def test_order_is_by_quality_not_elapsed_time(self):
        poorer = copy.deepcopy(self.metrics)
        poorer.update(compute_metrics(["Nothing", "Check", "Checkmate"],
                                      ["Nothing", "Nothing", "Nothing"]))
        poorer["prediction_seconds"] = 0.01
        slow = self.report("slow-correct")
        fast = self.report("fast-poor", poorer)
        reports = load_comparison([fast, slow], self.validation)
        self.assertEqual([report["run"] for report in reports], ["slow-correct", "fast-poor"])
        self.assertEqual(reports[0]["worst_recall"], 1)

    def test_rejects_wrong_dataset_even_with_same_class_counts(self):
        wrong = copy.deepcopy(self.metrics)
        wrong["artifacts"]["dataset"]["sha256"] = "b" * 64
        path = self.report("test-run", wrong)
        with self.assertRaisesRegex(ValueError, "declared validation"):
            load_comparison([path], self.validation)

    def test_rejects_mixed_builds_and_duplicate_runs(self):
        first = self.report("first")
        changed = copy.deepcopy(self.metrics)
        changed["artifacts"]["analyzer"]["sha256"] = "b" * 64
        second = self.report("second", changed)
        with self.assertRaisesRegex(ValueError, "same analyzer"):
            load_comparison([first, second], self.validation)
        with self.assertRaisesRegex(ValueError, "only once"):
            load_comparison([first, first], self.validation)

    def test_rejects_invalid_or_incompatible_reports(self):
        cases = []
        for key, value in (("labels", ["Check", "Nothing", "Checkmate"]),
                           ("rows", "predicted"), ("counts", [[1, 0, 0]] * 3)):
            invalid = copy.deepcopy(self.metrics)
            invalid["confusion_matrix"][key] = value
            cases.append(invalid)
        invalid = copy.deepcopy(self.metrics)
        invalid["macro_f1"] = float("nan")
        cases.append(invalid)
        invalid = copy.deepcopy(self.metrics)
        invalid["samples"] = 12
        cases.append(invalid)
        invalid = copy.deepcopy(self.metrics)
        del invalid["artifacts"]["analyzer"]
        cases.append(invalid)
        for index, invalid in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                load_comparison([self.report(f"invalid-{index}", invalid)], self.validation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
