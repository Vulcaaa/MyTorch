"""Metric reference cases and evaluation through the real analyzer."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts/evaluate.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_prediction_color_normalization(self):
        for state in ("Check", "Checkmate"):
            for suffix in ("", " White", " Black"):
                self.assertEqual(evaluation.prediction_class(state + suffix), state)
        self.assertEqual(evaluation.prediction_class("Nothing"), "Nothing")
        for invalid in ("", "Nothing White", "Check Blue", "Check White extra"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluation.prediction_class(invalid)

    def test_known_asymmetric_confusion_matrix(self):
        metrics = evaluation.compute_metrics(
            ["Nothing", "Nothing", "Check", "Check", "Check", "Checkmate"],
            ["Nothing", "Check", "Check", "Check", "Checkmate", "Nothing"],
        )
        self.assertEqual(metrics["confusion_matrix"]["counts"], [[1, 1, 0], [0, 2, 1], [1, 0, 0]])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["per_class"]["Nothing"],
                         dict(precision=0.5, recall=0.5, f1=0.5, support=2))
        self.assertAlmostEqual(metrics["per_class"]["Check"]["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], (0.5 + 2 / 3) / 3)

    def test_missing_classes_and_invalid_predictions(self):
        metrics = evaluation.compute_metrics(["Nothing"], ["Nothing"])
        self.assertEqual(metrics["per_class"]["Check"],
                         dict(precision=0.0, recall=0.0, f1=0.0, support=0))
        for actual, predicted in (([], []), (["Check"], []),
                                  (["Check"], ["Unknown"]), (["Check"], [""])):
            with self.subTest(predicted=predicted), self.assertRaises(ValueError):
                evaluation.compute_metrics(actual, predicted)

    def test_real_analyzer_reports_and_failures(self):
        with tempfile.TemporaryDirectory(prefix="mytorch-eval-") as directory:
            work = Path(directory)
            config = work / "constant.yml"
            config.write_text("layers: [777, 3]\nhidden_activation: relu\n"
                              "output_activation: softmax\ninitialization: he\nseed: 42\n")
            result = subprocess.run([str(ROOT / "my_torch_generator"), str(config), "1"],
                                    cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            model = work / "models/constant_1.nn"
            contents = json.loads(model.read_text())
            contents["layers"] = [{"weights": [[0.0] * 777 for _ in range(3)],
                                   "biases": [1.0, 0.0, 0.0]}]
            model.write_text(json.dumps(contents))
            original = model.read_bytes()
            dataset = work / "positions.txt"
            dataset.write_text("\n8/8/8/8/8/8/8/8 w - - 0 1 Nothing\n\n"
                               "8/8/8/8/8/8/8/8 b - - 0 1 Check Black\n")
            output = work / "run"
            command = [sys.executable, str(ROOT / "scripts/evaluate.py"), str(model),
                       str(dataset), "--output", str(output), "--predictions"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["accuracy"], 0.5)
            self.assertEqual(metrics["samples"], 2)
            self.assertGreaterEqual(metrics["prediction_seconds"], 0)
            self.assertEqual(len(metrics["artifacts"]["model"]["sha256"]), 64)
            with (output / "predictions.csv").open() as source:
                self.assertEqual(list(csv.reader(source)), [
                    ["dataset_line", "actual", "predicted"],
                    ["2", "Nothing", "Nothing"], ["4", "Check", "Nothing"]])
            self.assertTrue((output / "confusion-matrix.svg").exists())
            self.assertFalse((output / "confusion-matrix.csv").exists())
            default_output = work / "default-report"
            result_default = subprocess.run(command[:-2] + [str(default_output)], capture_output=True, text=True)
            self.assertEqual(result_default.returncode, 0, result_default.stderr)
            self.assertEqual(set(p.name for p in default_output.iterdir()),
                             {"README.md", "metrics.json", "confusion-matrix.svg"})
            self.assertEqual(model.read_bytes(), original)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output already exists", result.stderr)
            contents["layers"][0]["biases"] = [0.0, 1.0, 0.0]
            model.write_text(json.dumps(contents))
            colored_metrics, _, predictions = evaluation.predict_metrics(
                model, dataset, ROOT / "my_torch_analyzer")
            self.assertEqual(predictions, ["Check", "Check"])
            self.assertEqual(colored_metrics["accuracy"], 0.5)
            for invalid in ("", "8/8/8/8/8/8/8/8 w - - 0 1",  # Missing label.
                            "8/8/8/8/8/8/8/8 w - - 0 1 Nothing White",
                            "bad-fen w - - 0 1 Nothing"):
                dataset.write_text(invalid)
                with self.subTest(dataset=invalid):
                    result = subprocess.run(command[:-1] + [str(work / "failed")],
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse((work / "failed").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
