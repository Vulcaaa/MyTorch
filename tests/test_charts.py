"""Chart exports use recorded values without rerunning learning or inference."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.charts import render_directory, loss_chart, pyplot
from scripts.evaluate import compute_metrics


class ChartTests(unittest.TestCase):
    def test_checkpoint_zero_and_continuation_use_total_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            history = Path(temporary) / "history.csv"
            history.write_text("epoch,train_loss,validation_loss\n1,0.5,0.8\n2,0.4,0.9\n")
            chart = loss_chart(history, {"best_epoch": 0}, "completed", 0.7, offset=100)
            content = chart.read_text()
            ET.fromstring(content)
            self.assertIn("Selected checkpoint: epoch 100", content)
            self.assertIn("Stopped: epoch 102", content)

    def test_backfill_preserves_sources_replaces_csv_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            metrics = compute_metrics(["Nothing", "Check"], ["Check", "Check"])
            source = json.dumps(metrics)
            (work / "metrics.json").write_text(source)
            (work / "confusion-matrix.csv").write_text("old export")
            (work / "README.md").write_text("My handwritten conclusion.\n")
            pyplot()  # Allow matplotlib to discover installed fonts before forbidding inference.
            with patch("subprocess.run", side_effect=AssertionError("must not run prediction")):
                paths = render_directory(work)
                render_directory(work)
            self.assertEqual(paths, [work / "confusion-matrix.svg"])
            ET.parse(paths[0])
            self.assertEqual((work / "metrics.json").read_text(), source)
            self.assertFalse((work / "confusion-matrix.csv").exists())
            readme = (work / "README.md").read_text()
            self.assertIn("My handwritten conclusion.", readme)
            self.assertEqual(readme.count("<!-- charts:start -->"), 1)

    def test_partial_campaign_compares_only_completed_evaluated_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            runs = []
            for index, status in enumerate(("completed", "completed", "failed")):
                directory = work / f"run-{index}"
                directory.mkdir()
                metrics = compute_metrics(["Nothing"], ["Nothing"])
                (directory / "metrics.json").write_text(json.dumps({"validation": metrics}))
                runs.append(dict(directory=directory.name, status=status, configuration=f"net-{index}",
                                 seed=42, training_seconds=10 + index))
            (work / "config.json").write_text(json.dumps(dict(status="failed", runs=runs)))
            render_directory(work)
            content = (work / "comparison.svg").read_text()
            self.assertIn("net-0 / seed 42", content)
            self.assertIn("net-1 / seed 42", content)
            self.assertNotIn("net-2 / seed 42", content)
            self.assertIn("campaign status: failed", content)

    def test_active_campaign_is_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "config.json").write_text(json.dumps(dict(status="preparing", runs=[])))
            self.assertEqual(render_directory(work), [])
            self.assertEqual([p.name for p in work.iterdir()], ["config.json"])
