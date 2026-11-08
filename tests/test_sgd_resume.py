"""Check experiment isolation and recovery without performing neural training."""

import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import sgd_resume


class ResumeProtocolTest(unittest.TestCase):
    def test_three_independent_arms_and_recovery_without_retraining(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            (source / "data").mkdir(parents=True)
            (source / "tools").mkdir()
            (source / "tools/my_torch_analyzer").write_text("frozen executable")
            for name in ("train.fen", "validation.fen"):
                (source / "data" / name).write_text(name)
            metrics = {"accuracy": 0.88, "macro_f1": 0.87, "per_class": {}, "confusion_matrix": {}}
            original = {"config": {"seed": 46, "layers": [777, 128, 64, 3],
                         "hidden_activation": "sigmoid", "output_activation": "softmax",
                         "learning_rate": 0.08}, "layers": [{"weights": [[1.25]], "biases": [0.5]}]}
            for seed in (44, 45, 46):
                run = source / "runs" / f"seed-{seed}"
                run.mkdir(parents=True)
                model = copy.deepcopy(original)
                model["config"]["seed"] = seed
                (run / "model.nn").write_text(json.dumps(model))
                score = dict(metrics, macro_f1=0.87 if seed == 46 else 0.85)
                (run / "candidate-metrics.json").write_text(json.dumps(score))
                (run / "config.json").write_text("{}")
                (run / "history.csv").write_text("epoch,train_loss,validation_loss\n")
            calls = []

            def train(command, cwd, stdout, stderr, check):
                self.assertEqual(command[command.index("--epochs") + 1], "40")
                self.assertEqual(command[command.index("--patience") + 1], "20")
                loaded = json.loads(Path(command[-2]).read_text())
                rate = loaded["config"]["learning_rate"]
                expected = copy.deepcopy(original)
                expected["config"]["learning_rate"] = rate
                self.assertEqual(loaded, expected)
                self.assertNotIn("test.fen", " ".join(command))
                calls.append(rate)
                shutil.copy2(command[-2], Path(cwd) / "model.nn")
                (Path(cwd) / "history.csv").write_text("epoch,train_loss,validation_loss\n1,0.2,0.4\n")
                # The real analyzer emits progress on stderr, not stdout.
                stderr.write("Training complete: best_epoch=0 validation_loss=0.3\n")
                stderr.flush()
                return SimpleNamespace(returncode=0)

            predictions = 0

            def predict(*_):
                nonlocal predictions
                predictions += 1
                if predictions == 2:
                    raise RuntimeError("simulated inference interruption")
                return copy.deepcopy(metrics), [], []

            command = ["sgd_resume", "--source", str(source), "--output", str(output)]
            with patch.dict(os.environ), patch.object(sgd_resume.subprocess, "run", side_effect=train), \
                    patch.object(sgd_resume, "predict_metrics", side_effect=predict):
                with patch.object(sys, "argv", command):
                    with self.assertRaisesRegex(RuntimeError, "interruption"):
                        sgd_resume.main()
                self.assertEqual(calls, [0.08])
                with patch.object(sys, "argv", command + ["--resume"]):
                    sgd_resume.main()
            self.assertEqual(calls, [0.08, 0.02, 0.008])
            result = json.loads((output / "summary.json").read_text())
            self.assertEqual(result["protocol"]["status"], "completed")
            self.assertEqual([r["selected_epoch"] for r in result["protocol"]["runs"]], [0, 0, 0])
            self.assertEqual(json.loads((source / "runs/seed-46/model.nn").read_text()), original)


if __name__ == "__main__":
    unittest.main()
