"""End-to-end checks for the generator and the analyzer's current CLI."""

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LOG_LINE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z \[(INFO|WARN|ERROR)\] (.*)")
CONFIG = (
    "layers: [3, 2, 1]\n"
    "hidden_activation: relu\n"
    "output_activation: sigmoid\n"
    "initialization: he\n"
    "seed: 42\n"
)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mytorch-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.models = self.work / "models"
        (self.work / "network.yml").write_text(CONFIG)

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
        (self.work / "other.yml").write_text(CONFIG.replace("layers: [3, 2, 1]", "layers: [2, 1]"))
        self.generate("network.yml", "3", "other.yml", "2")
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
        self.generate("network.yml", "2")
        previous = {index: (self.models / f"network_{index}.nn").read_bytes() for index in (5, 6)}
        self.generate("network.yml", "1")
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
            (self.work / folder / "shared.yml").write_text(CONFIG)
        self.generate("a/shared.yml", "1", "b/shared.yml", "1")
        self.assertEqual({path.name for path in self.models.glob("*.nn")}, {"shared_1.nn", "shared_2.nn"})

    def test_config_name_with_spaces_and_multiple_dots(self):
        (self.work / "my network.v1.yml").write_text(CONFIG)
        self.generate("my network.v1.yml", "1")
        self.assertEqual(self.model("my network.v1_1.nn")["format"], "mytorch")

    def test_models_path_is_a_file(self):
        self.models.write_text("keep this file")
        result = self.run_command("my_torch_generator", "network.yml", "1")
        self.assertEqual(result.returncode, 84)
        self.assertTrue(result.stderr)
        self.assertEqual(self.models.read_text(), "keep this file")
        self.assertEqual(list(self.work.glob("*.nn")), [])

    def test_seed_wraparound(self):
        (self.work / "network.yml").write_text(CONFIG.replace("seed: 42", "seed: 4294967295"))
        self.generate("network.yml", "2")
        self.assertEqual(self.model("network_1.nn")["config"]["seed"], 4294967295)
        self.assertEqual(self.model("network_2.nn")["config"]["seed"], 0)

    def test_invalid_arguments(self):
        cases = [[], ["network.yml"], ["network.yml", "1", "network.yml"], ["--unknown"]]
        cases += [["network.yml", count] for count in (
            "0", "-1", "1.5", "1abc", "+1", "", " 1", "1 ", "2147483648",
        )]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self.run_command("my_torch_generator", *arguments)
                self.assertEqual(result.returncode, 84)
                self.assertTrue(result.stderr)
                self.assertEqual(list(self.models.glob("*.nn")), [])

    def test_missing_or_invalid_config(self):
        (self.work / "invalid.yml").write_text("layers: [3, 0, 1]\n")
        for config in ("missing.yml", "invalid.yml"):
            with self.subTest(config=config):
                result = self.run_command("my_torch_generator", config, "1")
                self.assertEqual(result.returncode, 84)
                self.assertIn(config, result.stderr)
                self.assertEqual(list(self.models.glob("*.nn")), [])

    def logs(self, prefix):
        return sorted((self.work / "logs").glob(f"{prefix}_*.log"))

    def training_histories(self):
        """Automatic training history directories, which generation logs must not be mistaken for."""
        return [path for path in (self.work / "logs").glob("train_*") if path.is_dir()]

    def log_messages(self, path):
        """Return the "[LEVEL] message" part of each entry, checking every timestamp."""
        messages = []
        for line in path.read_text().splitlines():
            match = LOG_LINE.fullmatch(line)
            self.assertIsNotNone(match, f"malformed log line: {line!r}")
            messages.append(f"[{match[1]}] {match[2]}")
        return messages

    def test_generation_log(self):
        (self.work / "other.yml").write_text(CONFIG.replace("layers: [3, 2, 1]", "layers: [2, 1]"))
        result = self.generate("network.yml", "2", "other.yml", "1")
        self.assertEqual(result.stdout, "")
        [log] = self.logs("generation")
        self.assertRegex(log.name, r"^generation_\d{4}-\d\d-\d\d_\d{6}\.log$")
        messages = self.log_messages(log)
        settings = "hidden_activation=relu output_activation=sigmoid initialization=he learning_rate=0.01"
        self.assertEqual(messages[:-1], [
            "[INFO] Generation started",
            f"[INFO] Configuration network.yml: layers=3,2,1 {settings} seed=42",
            "[INFO] Generated models/network_1.nn (seed 42)",
            "[INFO] Generated models/network_2.nn (seed 43)",
            f"[INFO] Configuration other.yml: layers=2,1 {settings} seed=42",
            "[INFO] Generated models/other_1.nn (seed 42)",
        ])
        self.assertRegex(messages[-1], r"^\[INFO\] Generation complete: 3 models in \d+\.\d{3} s$")

    def test_each_generation_run_has_its_own_log(self):
        self.generate("network.yml", "1")
        self.generate("network.yml", "1")
        logs = self.logs("generation")
        self.assertEqual(len(logs), 2)
        self.assertIn("[INFO] Generated models/network_1.nn (seed 42)", self.log_messages(logs[0]))
        self.assertIn("[INFO] Generated models/network_2.nn (seed 42)", self.log_messages(logs[1]))

    def test_generation_logs_failures(self):
        result = self.run_command("my_torch_generator", "network.yml", "0")
        self.assertEqual(result.returncode, 84)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "my_torch_generator: network count must be a positive integer\n")
        [log] = self.logs("generation")
        messages = self.log_messages(log)
        self.assertEqual(messages[0], "[INFO] Generation started")
        self.assertRegex(messages[-1], r"^\[ERROR\] Generation failed after \d+\.\d{3} s: "
                         r"network count must be a positive integer$")

    def test_generation_continues_without_log(self):
        (self.work / "logs").write_text("not a directory")
        result = self.run_command("my_torch_generator", "network.yml", "1")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertRegex(result.stderr, r"^warning: logging disabled: [^\n]*logs[^\n]*\n$")
        self.assertTrue((self.models / "network_1.nn").is_file())
        self.assertEqual((self.work / "logs").read_text(), "not a directory")

    def test_help_does_not_log(self):
        self.run_command("my_torch_generator", "--help")
        self.assertFalse((self.work / "logs").exists())

    def analyzer_model(self):
        path = self.work / "chess.yml"
        path.write_text(CONFIG.replace("layers: [3, 2, 1]", "layers: [777, 3]")
                        .replace("output_activation: sigmoid", "output_activation: softmax"))
        self.generate(path.name, "1")
        return "models/chess_1.nn"

    def test_prediction_order_labels_and_model_preservation(self):
        model_path = self.work / self.analyzer_model()
        model = json.loads(model_path.read_text())
        weights = [[0.0] * 777 for _ in range(3)]
        weights[1][768] = 4.0  # Black to move selects Check.
        weights[2][769] = 5.0  # En passant on file a selects Checkmate.
        model["layers"] = [{"weights": weights, "biases": [1.0, -1.0, 0.0]}]
        model_path.write_text(json.dumps(model))
        original = model_path.read_bytes()
        positions = self.work / "positions.txt"
        positions.write_text(
            "8/8/8/8/8/8/8/8 b - - 0 1 Nothing\n"
            "8/8/8/8/8/8/8/8 w - - 0 1 Checkmate\n"
            "8/8/8/8/8/8/8/8 b - - 0 1\n"
            "8/8/8/8/8/8/8/8 w - a6 0 1\n"
        )
        result = self.run_command("my_torch_analyzer", "--predict",
                                  str(model_path), positions.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "Check\nNothing\nCheck\nCheckmate\n")
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
                                              *(["--validation", "validation.txt"] if mode == "--train" else []),
                                              str(model_path), "missing.txt")
                    self.assertEqual(result.returncode, 84)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("requires 777 inputs and 3 outputs", result.stderr)
                    self.assertIn(f"got {inputs} inputs and {outputs} outputs", result.stderr)
                    self.assertNotIn("cannot open dataset", result.stderr)

    def test_prediction_rejects_missing_or_invalid_model(self):
        model_path = self.work / self.analyzer_model()
        valid_model = json.loads(model_path.read_text())
        invalid_weights = json.loads(json.dumps(valid_model))
        invalid_weights["layers"][0]["weights"][0].pop()
        cases = (
            (None, "cannot open model for reading"),
            ("{invalid json", "invalid model JSON"),
            (json.dumps(invalid_weights), "incorrect number of weights per neuron"),
        )
        for contents, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                model_path.unlink(missing_ok=True)
                if contents is not None:
                    model_path.write_text(contents)
                result = self.run_command("my_torch_analyzer", "--predict",
                                          str(model_path), "missing.txt")
                self.assertEqual(result.returncode, 84)
                self.assertEqual(result.stdout, "")
                self.assertIn(str(model_path), result.stderr)
                self.assertIn(diagnostic, result.stderr)
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
                        "my_torch_analyzer", mode,
                        *(["--validation", "validation.txt"] if mode == "--train" else []),
                        model, path.name,
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
            "my_torch_analyzer", "--train", "--validation", "validation.txt", model, path.name,
        )
        self.assertEqual(result.returncode, 84)
        self.assertEqual(result.stdout, "")
        self.assertIn("positions.txt:2: missing training label", result.stderr)

    def test_training_argument_errors_precede_file_loading(self):
        cases = [
            (["--train", "--validation", "val.txt", "--epochs", "0"], "--epochs"),
            (["--train", "--validation", "val.txt", "--min-delta", "nan"], "--min-delta"),
            (["--train", "--validation", "val.txt", "--history", "a.csv", "--history", "b.csv"],
             "duplicate option: --history"),
            (["--predict", "--batch-size", "32"], "only valid with --train"),
        ]
        for options, diagnostic in cases:
            with self.subTest(options=options):
                result = self.run_command("my_torch_analyzer", *options, "missing.nn", "missing.txt")
                self.assertEqual(result.returncode, 84)
                self.assertEqual(result.stdout, "")
                self.assertIn(diagnostic, result.stderr)
                self.assertNotIn("cannot open model", result.stderr)
        self.assertFalse((self.work / "logs").exists())

    def test_training_saves_trained_model_and_history(self):
        model = self.analyzer_model()
        original = (self.work / model).read_bytes()
        (self.work / "train.txt").write_text("8/8/8/8/8/8/8/8 w - - 0 1 Nothing\n")
        (self.work / "val.txt").write_text("8/8/8/8/8/8/8/8 b - - 0 1 Nothing\n")
        history = self.work / "history.csv"
        for save_options, saved in (([], model), (["--save", "trained.nn"], "trained.nn")):
            with self.subTest(save_options=save_options):
                (self.work / model).write_bytes(original)
                history.unlink(missing_ok=True)
                result = self.run_command(
                    "my_torch_analyzer", "--train", "--validation", "val.txt",
                    "--epochs", "2", "--batch-size", "1", "--threads", "1", "--history", history.name,
                    *save_options, model, "train.txt",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn("Training complete", result.stderr)
                self.assertTrue(any(re.fullmatch(r"\[INFO\] CPU backend=(scalar|openblas|accelerate) threads=1", msg)
                                    for log in self.logs("train") for msg in self.log_messages(log)))
                rows = history.read_text().splitlines()
                self.assertEqual(rows[0], "epoch,train_loss,validation_loss")
                self.assertEqual([row.split(",")[0] for row in rows[1:]], ["1", "2"])
                # Learning "Nothing" lowers the validation loss, so the saved weights changed.
                trained = json.loads((self.work / saved).read_text())
                self.assertNotEqual(trained["layers"], json.loads(original)["layers"])
                if save_options:
                    self.assertEqual((self.work / model).read_bytes(), original)
                self.assertEqual(self.training_histories(), [])

    def test_training_minimal_syntax_uses_automatic_validation(self):
        model = self.analyzer_model()
        original = (self.work / model).read_bytes()
        data = ("8/8/8/8/8/8/8/8 w - - 0 1 Nothing\n"
                "8/8/8/8/8/8/8/8 b - - 0 1 Nothing\n")
        (self.work / "positions.txt").write_text(data)
        histories = []
        saved_models = []
        for save_options, saved in (([], model), (["--save", "trained.nn"], "trained.nn")):
            with self.subTest(save_options=save_options):
                (self.work / model).write_bytes(original)
                result = self.run_command(
                    "my_torch_analyzer", "--train", *save_options, model, "positions.txt",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn("Automatic validation split", result.stderr)
                self.assertIn("train=1 validation=1 seed=42", result.stderr)
                self.assertIn("Training complete", result.stderr)
                history_line = next(line for line in result.stderr.splitlines()
                                    if line.startswith("Training history: "))
                history = self.work / history_line.removeprefix("Training history: ")
                histories.append(history.read_text())
                saved_models.append((self.work / saved).read_bytes())
                self.assertNotEqual(json.loads(saved_models[-1])["layers"], json.loads(original)["layers"])
                if save_options:
                    self.assertEqual((self.work / model).read_bytes(), original)
                self.assertEqual((self.work / "positions.txt").read_text(), data)
        self.assertEqual(histories[0], histories[1])
        self.assertEqual(saved_models[0], saved_models[1])

    def test_automatic_validation_rejects_unsplittable_data(self):
        model = self.analyzer_model()
        original = (self.work / model).read_bytes()
        position = "8/8/8/8/8/8/8/8 w - - 0 1 Nothing\n"
        cases = [
            (position, "at least two distinct"),
            (position + "8/8/8/8/8/8/8/8 w K - 99 50 Nothing\n", "at least two distinct"),
            (position + "8/8/8/8/8/8/8/8 w - - 0 1 Check\n", "conflicting labels"),
        ]
        for data, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                (self.work / "positions.txt").write_text(data)
                result = self.run_command("my_torch_analyzer", "--train", "--save", "trained.nn",
                                          model, "positions.txt")
                self.assertEqual(result.returncode, 84)
                self.assertIn(diagnostic, result.stderr)
                self.assertEqual((self.work / model).read_bytes(), original)
                self.assertFalse((self.work / "trained.nn").exists())
                self.assertEqual(self.training_histories(), [])

    def test_training_logs_failures(self):
        model = self.analyzer_model()
        (self.work / "train.txt").write_text("8/8/8/8/8/8/8/8 w - - 0 1 Nothing\n")
        result = self.run_command(
            "my_torch_analyzer", "--train", "--validation", "missing.txt",
            "--save", "trained.nn", "--history", "history.csv", model, "train.txt",
        )
        self.assertEqual(result.returncode, 84)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "MyTorch: missing.txt: cannot open dataset\n")
        [log] = self.logs("train")
        self.assertRegex(log.name, r"^train_\d{4}-\d\d-\d\d_\d{6}\.log$")
        messages = self.log_messages(log)
        self.assertEqual(messages[:2], [
            "[INFO] Training started",
            "[INFO] Model models/chess_1.nn: layers=777,3 hidden_activation=relu "
            "output_activation=softmax initialization=he learning_rate=0.01 seed=42",
        ])
        self.assertEqual(len(messages), 3)
        self.assertRegex(messages[2], r"^\[ERROR\] Training failed after \d+\.\d{3} s: "
                         r"missing\.txt: cannot open dataset$")

    def check_training_log(self, expected_start, epochs, saved):
        """Check a successful training log; losses and durations vary, so they are matched."""
        [log] = self.logs("train")
        messages = self.log_messages(log)
        backend = [m for m in messages if m.startswith("[INFO] CPU backend=")]
        self.assertEqual(len(backend), 1)
        self.assertRegex(backend[0], r"^\[INFO\] CPU backend=(scalar|openblas|accelerate) threads=auto/environment$")
        messages.remove(backend[0])
        self.assertEqual(messages[:len(expected_start)], expected_start)
        loss = r"\d+(\.\d+)?(e-?\d+)?"
        rest = messages[len(expected_start):]
        self.assertEqual(len(rest), epochs + 4)
        self.assertRegex(rest[0], rf"^\[INFO\] Initial validation_loss={loss}$")
        self.assertRegex(rest[1], r"^\[INFO\] History: logs/train_[^/]+/history\.csv$")
        for epoch in range(1, epochs + 1):
            self.assertRegex(rest[1 + epoch], rf"^\[INFO\] Epoch {epoch}/{epochs}: "
                             rf"train_loss={loss} validation_loss={loss} \(\d+\.\d{{3}} s\)$")
        best = re.fullmatch(rf"\[INFO\] Saved model from epoch (\d+) \(validation_loss={loss}\) "
                            rf"to {re.escape(saved)}", rest[-2])
        self.assertIsNotNone(best, rest[-2])
        self.assertRegex(rest[-1], rf"^\[INFO\] Training complete in \d+\.\d{{3}} s: "
                         rf"epochs_completed={epochs} best_epoch={best[1]}$")

    def test_training_log(self):
        model = self.analyzer_model()
        (self.work / "train.txt").write_text("8/8/8/8/8/8/8/8 w - - 0 1 Nothing\n")
        (self.work / "val.txt").write_text("8/8/8/8/8/8/8/8 b - - 0 1 Nothing\n")
        result = self.run_command(
            "my_torch_analyzer", "--train", "--validation", "val.txt", "--epochs", "2",
            "--batch-size", "1", "--save", "trained.nn", model, "train.txt",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("warning", result.stderr)
        self.check_training_log([
            "[INFO] Training started",
            "[INFO] Model models/chess_1.nn: layers=777,3 hidden_activation=relu "
            "output_activation=softmax initialization=he learning_rate=0.01 seed=42",
            "[INFO] Train set train.txt: 1 position",
            "[INFO] Validation set val.txt: 1 position",
            "[INFO] Settings: epochs=2 batch_size=1 patience=10 min_delta=0.0001",
        ], epochs=2, saved="trained.nn")

    def test_training_log_with_automatic_validation(self):
        model = self.analyzer_model()
        (self.work / "positions.txt").write_text("8/8/8/8/8/8/8/8 w - - 0 1 Nothing\n"
                                                 "8/8/8/8/8/8/8/8 b - - 0 1 Nothing\n")
        result = self.run_command("my_torch_analyzer", "--train", "--epochs", "1", model, "positions.txt")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.check_training_log([
            "[INFO] Training started",
            "[INFO] Model models/chess_1.nn: layers=777,3 hidden_activation=relu "
            "output_activation=softmax initialization=he learning_rate=0.01 seed=42",
            "[INFO] Train set positions.txt: 1 position",
            "[INFO] Validation set: 1 position held out from positions.txt (90/10 per class, seed 42)",
            "[INFO] Settings: epochs=1 batch_size=32 patience=10 min_delta=0.0001",
        ], epochs=1, saved=model)

    def test_training_continues_without_log(self):
        model = self.analyzer_model()
        original = (self.work / model).read_bytes()
        (self.work / "train.txt").write_text("8/8/8/8/8/8/8/8 w - - 0 1 Nothing\n")
        (self.work / "val.txt").write_text("8/8/8/8/8/8/8/8 b - - 0 1 Nothing\n")
        for log in (self.work / "logs").iterdir():
            log.unlink()
        (self.work / "logs").rmdir()
        (self.work / "logs").write_text("not a directory")
        result = self.run_command(
            "my_torch_analyzer", "--train", "--validation", "val.txt", "--epochs", "1",
            "--history", "history.csv", "--save", "trained.nn", model, "train.txt",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        warnings = [line for line in result.stderr.splitlines() if "warning" in line]
        self.assertEqual(len(warnings), 1, result.stderr)
        self.assertTrue(warnings[0].startswith("warning: logging disabled: "))
        self.assertIn("Training complete", result.stderr)
        trained = json.loads((self.work / "trained.nn").read_text())
        self.assertNotEqual(trained["layers"], json.loads(original)["layers"])
        self.assertEqual((self.work / "logs").read_text(), "not a directory")

    def test_prediction_does_not_log(self):
        model = self.analyzer_model()
        (self.work / "positions.txt").write_text("8/8/8/8/8/8/8/8 w - - 0 1\n")
        result = self.run_command("my_torch_analyzer", "--predict", model, "positions.txt")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.logs("train"), [])

    def test_training_validates_independent_validation_set(self):
        model = self.analyzer_model()
        original = (self.work / model).read_bytes()
        fen = "8/8/8/8/8/8/8/8 w - - 0 1"
        (self.work / "train.txt").write_text(fen + " Nothing\n")
        validation = self.work / "validation.txt"
        cases = [
            (None, "cannot open dataset"),
            ("", "no positions"),
            (fen + "\n", "missing training label"),
            ("invalid\n", "expected six FEN fields"),
            ("8/8/8/8/8/8/8/8 w - - 99 50 Nothing\n", "share a model-visible input"),
        ]
        for contents, diagnostic in cases:
            with self.subTest(contents=contents):
                validation.unlink(missing_ok=True)
                if contents is not None:
                    validation.write_text(contents)
                result = self.run_command(
                    "my_torch_analyzer", "--train", "--validation", validation.name,
                    "--save", "trained.nn", "--history", "history.csv", model, "train.txt",
                )
                self.assertEqual(result.returncode, 84)
                self.assertEqual(result.stdout, "")
                self.assertIn(diagnostic, result.stderr)
                self.assertNotIn("not implemented", result.stderr)
                self.assertEqual((self.work / model).read_bytes(), original)
                self.assertFalse((self.work / "trained.nn").exists())
                self.assertFalse((self.work / "history.csv").exists())

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
