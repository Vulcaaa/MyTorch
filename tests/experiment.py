"""Campaign orchestration with real C++ generation/prediction and a test-only trainer."""

import copy
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.comparison import load_experiment, load_history
from scripts.experiment import load_spec, run_experiment


# This fixture tests the adapter contract, not learning quality. Production code
# never supplies a training implementation or uses this artificial weight update.
TRAINER = '''import json
from pathlib import Path
import sys
initial, trained, settings, history = map(Path, sys.argv[1:])
request = json.loads(settings.read_text())
if request["settings"].get("fail_seed") == request["seed"]:
    sys.exit(7)
model = json.loads(initial.read_text())
model["layers"][-1]["biases"][0] += 1 + request["seed"] / 100
if len(model["layers"]) == 1:
    layer = model["layers"][0]
    layer["weights"] = [[0.0] * 777 for _ in range(3)]
    layer["biases"] = [1.0, 0.0, 0.0]
    if request["seed"] == 42:
        layer["weights"][1][768] = 4.0
        layer["weights"][2][769] = 5.0
    elif request["seed"] == 43:
        layer["weights"][1][768] = 4.0
trained.write_text(json.dumps(model))
history.write_text("epoch,train_loss,validation_loss\\n1,1.0,1.1\\n2,0.8,0.9\\n")
'''


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mytorch-experiment-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.network = self.work / "base.conf"
        self.network.write_text("layers=777,3\nhidden_activation=relu\n"
                                "output_activation=softmax\ninitialization=he\nseed=999\n")
        for role, counter in (("train", 1), ("validation", 2)):
            (self.work / f"{role}.txt").write_text(
                f"8/8/8/8/8/8/8/8 w - - {counter} 1 Nothing\n"
                f"8/8/8/8/8/8/8/8 b - - {counter} 1 Check\n"
                f"8/8/8/8/8/8/8/8 w - a6 {counter} 1 Checkmate\n")
        self.adapter = self.work / "trainer fixture.py"
        self.adapter.write_text(TRAINER)
        self.spec = {
            "version": 1, "seeds": [42, 43, 44],
            "datasets": {"train": "train.txt", "validation": "validation.txt"},
            "configurations": [{"name": "small", "network": "base.conf", "overrides": {}},
                               {"name": "large", "network": "base.conf", "overrides": {"layers": "777,2,3"}}],
            "training": {"command": None, "settings": None}}
        self.config = self.work / "experiment.json"
        self.output = self.work / "results with spaces"

    def write_spec(self):
        self.config.write_text(json.dumps(self.spec))

    def enable_fixture(self):
        self.spec["training"] = {
            "command": [sys.executable, str(self.adapter), "{initial_model}", "{trained_model}",
                        "{training_config}", "{history}"],
            "settings": {"epochs": 2, "checkpoint": "last"}}

    def run_campaign(self, prepare=False):
        self.write_spec()
        return run_experiment(self.config, self.output, prepare=prepare)

    def test_prepare_uses_central_seeds_without_creating_evaluations(self):
        summary = self.run_campaign(prepare=True)
        self.assertEqual(summary["status"], "prepared")
        self.assertFalse(summary["selection_ready"])
        self.assertEqual(summary["leading_configurations"], [])
        manifest = json.loads((self.output / "experiment.json").read_text())
        self.assertEqual(len(manifest["runs"]), 6)
        for run in manifest["runs"]:
            model = json.loads(Path(run["initial_model"]["path"]).read_text())
            self.assertEqual(model["config"]["seed"], run["seed"])
            self.assertFalse((self.output / run["directory"] / "evaluation").exists())
        for configuration in summary["configurations"]:
            self.assertEqual(configuration["missing_seeds"], [42, 43, 44])
            self.assertIsNone(configuration["macro_f1"]["mean"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_campaign(prepare=True)

    def test_unconfigured_training_fails_before_creating_output(self):
        with self.assertRaisesRegex(ValueError, "use --prepare"):
            self.run_campaign()
        self.assertFalse(self.output.exists())

    def test_full_adapter_flow_and_sample_standard_deviation(self):
        self.enable_fixture()
        summary = self.run_campaign()
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(summary["selection_ready"])
        self.assertEqual(len(summary["runs"]), 6)
        self.assertEqual(summary["leading_configurations"], ["small"])
        small = next(row for row in summary["configurations"] if row["configuration"] == "small")
        self.assertAlmostEqual(small["macro_f1"]["mean"], 31 / 54)
        self.assertGreater(small["macro_f1"]["std"], 0)
        for row in summary["configurations"]:
            entries = [report for report in summary["runs"] if report["configuration"] == row["configuration"]]
            scores = [report["metrics"]["macro_f1"] for report in entries]
            self.assertAlmostEqual(row["macro_f1"]["mean"], statistics.mean(scores))
            self.assertAlmostEqual(row["macro_f1"]["std"], statistics.stdev(scores))
            self.assertEqual(row["missing_seeds"], [])
            for report in entries:
                run_directory = Path(report["directory"]).parent
                self.assertEqual(len(load_history(run_directory / "history.csv")), 2)
                self.assertTrue((run_directory / "train.command.json").is_file())
        # Re-reading uses the frozen plan, even if the authoring config changes.
        self.config.write_text("{}")
        self.assertEqual(load_experiment(self.output)["expected_seeds"], [42, 43, 44])
        manifest_path = self.output / "experiment.json"
        original_manifest = json.loads(manifest_path.read_text())
        duplicate = copy.deepcopy(original_manifest)
        duplicate["runs"].append(duplicate["runs"][0])
        manifest_path.write_text(json.dumps(duplicate))
        with self.assertRaisesRegex(ValueError, "duplicate or unexpected"):
            load_experiment(self.output)
        incomplete = copy.deepcopy(original_manifest)
        incomplete["runs"].pop()
        manifest_path.write_text(json.dumps(incomplete))
        self.assertFalse(load_experiment(self.output)["selection_ready"])
        manifest_path.write_text(json.dumps(original_manifest))
        # Corrupting a delivered model invalidates the experiment summary.
        path = Path(summary["runs"][0]["directory"]).parent / "trained.nn"
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "trained model differs"):
            load_experiment(self.output)

    def test_one_seed_is_exploratory_even_if_all_configurations_finish(self):
        self.enable_fixture()
        self.spec["seeds"] = [42]
        summary = self.run_campaign()
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(all(row["complete"] for row in summary["configurations"]))
        self.assertFalse(summary["selection_ready"])
        self.assertEqual(summary["leading_configurations"], [])

    def test_backend_failure_retains_partial_results_without_selection(self):
        self.enable_fixture()
        self.spec["training"]["settings"]["fail_seed"] = 43
        with self.assertRaisesRegex(ValueError, "exited with 7"):
            self.run_campaign()
        summary = load_experiment(self.output)
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["selection_ready"])
        self.assertEqual(len(summary["runs"]), 1)
        partial = next(row for row in summary["configurations"] if row["configuration"] == "small")
        self.assertIsNone(partial["macro_f1"]["std"])
        self.assertEqual(partial["missing_seeds"], [43, 44])
        self.assertTrue((self.output / "summary.json").exists())
        manifest = json.loads((self.output / "experiment.json").read_text())
        self.assertEqual(manifest["runs"][1]["status"], "failed")

    def test_noop_backend_never_evaluates_initial_models(self):
        self.enable_fixture()
        self.adapter.write_text("# Backend returns success but produces no model.\n")
        with self.assertRaisesRegex(ValueError, "without writing trained.nn"):
            self.run_campaign()
        self.assertEqual(load_experiment(self.output)["runs"], [])
        self.assertEqual(list(self.output.rglob("metrics.json")), [])

    def test_copying_initial_model_is_not_training(self):
        self.enable_fixture()
        self.adapter.write_text("import shutil, sys\nshutil.copyfile(sys.argv[1], sys.argv[2])\n")
        with self.assertRaisesRegex(ValueError, "did not change any weights"):
            self.run_campaign()
        self.assertEqual(list(self.output.rglob("metrics.json")), [])

    def test_rejects_invalid_protocol_before_running_commands(self):
        original = copy.deepcopy(self.spec)
        mutations = [lambda spec: spec.update(seeds=[42, 42]),
                     lambda spec: spec["datasets"].update(test="validation.txt"),
                     lambda spec: spec["datasets"].update(validation="train.txt"),
                     lambda spec: spec["configurations"][0].update(name="../bad"),
                     lambda spec: spec["configurations"][0]["overrides"].update(seed=5),
                     lambda spec: spec["training"].update(command="shell command"),
                     lambda spec: spec["training"].update(command=["{unknown}"])]
        for mutation in mutations:
            self.spec = copy.deepcopy(original)
            mutation(self.spec)
            self.write_spec()
            with self.subTest(spec=self.spec), self.assertRaises(ValueError):
                load_spec(self.config)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
