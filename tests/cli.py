"""End-to-end checks for the generator and the analyzer's current CLI."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    "layers=3,2,1\n"
    "hidden_activation=relu\n"
    "output_activation=sigmoid\n"
    "initialization=he\n"
    "seed=42\n"
)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mytorch-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.models = self.work / "models"
        (self.work / "network.conf").write_text(CONFIG)

    def run_command(self, executable, *arguments):
        return subprocess.run(
            [str(ROOT / executable), *arguments],
            cwd=self.work, capture_output=True, text=True, timeout=10,
        )

    def generate(self, *arguments):
        result = self.run_command("my_torch_generator", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        return result

    def model(self, name):
        return json.loads((self.models / name).read_text())

    def test_help(self):
        for executable in ("my_torch_generator", "my_torch_analyzer"):
            with self.subTest(executable=executable):
                result = self.run_command(executable, "--help")
                self.assertEqual(result.returncode, 0)
                self.assertIn("USAGE", result.stdout)
                self.assertEqual(result.stderr, "")
        self.assertEqual(list(self.models.glob("*.nn")), [])

    def test_generate_multiple_configs_and_distinct_models(self):
        (self.work / "other.conf").write_text(CONFIG.replace("layers=3,2,1", "layers=2,1"))
        self.generate("network.conf", "3", "other.conf", "2")
        self.assertEqual(
            {path.name for path in self.models.glob("*.nn")},
            {"network_1.nn", "network_2.nn", "network_3.nn", "other_1.nn", "other_2.nn"},
        )
        self.assertEqual(list(self.work.glob("*.nn")), [])
        models = [self.model(f"network_{index}.nn") for index in range(1, 4)]
        self.assertEqual([model["config"]["seed"] for model in models], [42, 43, 44])
        self.assertNotEqual(models[0]["layers"], models[1]["layers"])
        self.assertEqual(self.model("other_1.nn")["config"]["layers"], [2, 1])

    def test_existing_paths_and_repeated_runs(self):
        self.models.mkdir()
        (self.models / "network_1.nn").write_text("do not overwrite")
        (self.models / "network_2.nn").mkdir()
        (self.models / "network_3.nn").symlink_to("missing-target")
        (self.models / "protected").write_text("protected target")
        (self.models / "network_4.nn").symlink_to("protected")
        self.generate("network.conf", "2")
        previous = {index: (self.models / f"network_{index}.nn").read_bytes() for index in (5, 6)}
        self.generate("network.conf", "1")
        self.assertTrue((self.models / "network_7.nn").is_file())
        self.assertEqual((self.models / "network_1.nn").read_text(), "do not overwrite")
        self.assertTrue((self.models / "network_2.nn").is_dir())
        self.assertTrue((self.models / "network_3.nn").is_symlink())
        self.assertTrue((self.models / "network_4.nn").is_symlink())
        self.assertEqual((self.models / "protected").read_text(), "protected target")
        for index, content in previous.items():
            self.assertEqual((self.models / f"network_{index}.nn").read_bytes(), content)

    def test_same_basename_in_different_directories(self):
        for folder in ("a", "b"):
            (self.work / folder).mkdir()
            (self.work / folder / "shared.conf").write_text(CONFIG)
        self.generate("a/shared.conf", "1", "b/shared.conf", "1")
        self.assertEqual({path.name for path in self.models.glob("*.nn")}, {"shared_1.nn", "shared_2.nn"})

    def test_config_name_with_spaces_and_multiple_dots(self):
        (self.work / "my network.v1.conf").write_text(CONFIG)
        self.generate("my network.v1.conf", "1")
        self.assertEqual(self.model("my network.v1_1.nn")["format"], "mytorch")

    def test_models_path_is_a_file(self):
        self.models.write_text("keep this file")
        result = self.run_command("my_torch_generator", "network.conf", "1")
        self.assertEqual(result.returncode, 84)
        self.assertTrue(result.stderr)
        self.assertEqual(self.models.read_text(), "keep this file")
        self.assertEqual(list(self.work.glob("*.nn")), [])

    def test_seed_wraparound(self):
        (self.work / "network.conf").write_text(CONFIG.replace("seed=42", "seed=4294967295"))
        self.generate("network.conf", "2")
        self.assertEqual(self.model("network_1.nn")["config"]["seed"], 4294967295)
        self.assertEqual(self.model("network_2.nn")["config"]["seed"], 0)

    def test_invalid_arguments(self):
        cases = [[], ["network.conf"], ["network.conf", "1", "network.conf"], ["--unknown"]]
        cases += [["network.conf", count] for count in (
            "0", "-1", "1.5", "1abc", "+1", "", " 1", "1 ", "2147483648",
        )]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self.run_command("my_torch_generator", *arguments)
                self.assertEqual(result.returncode, 84)
                self.assertTrue(result.stderr)
                self.assertEqual(list(self.models.glob("*.nn")), [])

    def test_missing_or_invalid_config(self):
        (self.work / "invalid.conf").write_text("layers=3,0,1\n")
        for config in ("missing.conf", "invalid.conf"):
            with self.subTest(config=config):
                result = self.run_command("my_torch_generator", config, "1")
                self.assertEqual(result.returncode, 84)
                self.assertIn(config, result.stderr)
                self.assertEqual(list(self.models.glob("*.nn")), [])

    def analyzer_model(self):
        path = self.work / "chess.conf"
        path.write_text(CONFIG.replace("layers=3,2,1", "layers=777,3")
                        .replace("output_activation=sigmoid", "output_activation=softmax"))
        self.generate(path.name, "1")
        return "models/chess_1.nn"

    def test_prediction_order_labels_and_model_preservation(self):
        model_path = self.work / self.analyzer_model()
        model = json.loads(model_path.read_text())
        weights = [[0.0] * 777 for _ in range(3)]
        weights[1][768] = 4.0  # Black to move selects Check.
        model["layers"] = [{"weights": weights, "biases": [1.0, -1.0, 0.0]}]
        model_path.write_text(json.dumps(model))
        original = model_path.read_bytes()
        positions = self.work / "positions.txt"
        positions.write_text(
            "8/8/8/8/8/8/8/8 b - - 0 1 Nothing\n"
            "8/8/8/8/8/8/8/8 w - - 0 1 Checkmate\n"
            "8/8/8/8/8/8/8/8 b - - 0 1\n"
        )
        result = self.run_command("my_torch_analyzer", "--predict",
                                  str(model_path), positions.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "Check\nNothing\nCheck\n")
        self.assertEqual(result.stderr, "")
        self.assertEqual(model_path.read_bytes(), original)

    def test_analyzer_rejects_incompatible_model_before_loading_dataset(self):
        model_path = self.work / self.analyzer_model()
        model = json.loads(model_path.read_text())
        for inputs, outputs in ((768, 3), (777, 2), (768, 2)):
            model["config"]["layers"] = [inputs, outputs]
            model["layers"] = [{"weights": [[0.0] * inputs for _ in range(outputs)],
                                "biases": [0.0] * outputs}]
            model_path.write_text(json.dumps(model))
            for mode in ("--predict", "--train"):
                with self.subTest(inputs=inputs, outputs=outputs, mode=mode):
                    result = self.run_command("my_torch_analyzer", mode,
                                              str(model_path), "missing.txt")
                    self.assertEqual(result.returncode, 84)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("requires 777 inputs and 3 outputs", result.stderr)
                    self.assertIn(f"got {inputs} inputs and {outputs} outputs", result.stderr)
                    self.assertNotIn("cannot open dataset", result.stderr)

    def test_analyzer_dataset_errors(self):
        model = self.analyzer_model()
        fen = "8/8/8/8/8/8/8/8 w - - 0 1"
        path = self.work / "chess positions.txt"
        cases = [
            (None, "cannot open dataset"),
            ("", "no positions"),
            (" \t\n\r\n", "no positions"),
            ("\n8/8/8 w - - 0 1 Nothing", ":2: FEN:"),
            ("\n" + fen + " Unknown", ":2: unknown label"),
            (fen + " Check White extra", ":1: unexpected data"),
        ]
        for mode in ("--predict", "--train"):
            for contents, diagnostic in cases:
                with self.subTest(mode=mode, contents=contents):
                    path.unlink(missing_ok=True)
                    if contents is not None:
                        path.write_text(contents)
                    result = self.run_command(
                        "my_torch_analyzer", mode, model, path.name,
                    )
                    self.assertEqual(result.returncode, 84)
                    self.assertEqual(result.stdout, "")
                    self.assertIn(path.name, result.stderr)
                    self.assertIn(diagnostic, result.stderr)

    def test_analyzer_requires_training_labels(self):
        model = self.analyzer_model()
        path = self.work / "positions.txt"
        path.write_text("\n8/8/8/8/8/8/8/8 w - - 0 1")
        result = self.run_command(
            "my_torch_analyzer", "--train", model, path.name,
        )
        self.assertEqual(result.returncode, 84)
        self.assertEqual(result.stdout, "")
        self.assertIn("positions.txt:2: missing training label", result.stderr)

    @unittest.skipUnless(os.name == "posix", "requires POSIX file permissions")
    def test_analyzer_unreadable_dataset(self):
        model = self.analyzer_model()
        path = self.work / "unreadable.txt"
        path.write_text("8/8/8/8/8/8/8/8 w - - 0 1 Nothing")
        path.chmod(0)
        try:
            if os.access(path, os.R_OK):
                self.skipTest("current user can bypass file permissions")
            result = self.run_command(
                "my_torch_analyzer", "--predict", model, path.name,
            )
            self.assertEqual(result.returncode, 84)
            self.assertEqual(result.stdout, "")
            self.assertIn("unreadable.txt: cannot open dataset", result.stderr)
        finally:
            path.chmod(0o600)

if __name__ == "__main__":
    unittest.main(verbosity=2)
