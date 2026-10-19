"""End-to-end checks for the generator and the analyzer's current CLI."""

import json
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
        return json.loads((self.work / name).read_text())

    def test_help(self):
        for executable in ("my_torch_generator", "my_torch_analyzer"):
            with self.subTest(executable=executable):
                result = self.run_command(executable, "--help")
                self.assertEqual(result.returncode, 0)
                self.assertIn("USAGE", result.stdout)
                self.assertEqual(result.stderr, "")
        self.assertEqual(list(self.work.glob("*.nn")), [])

    def test_generate_multiple_configs_and_distinct_models(self):
        (self.work / "other.conf").write_text(CONFIG.replace("layers=3,2,1", "layers=2,1"))
        self.generate("network.conf", "3", "other.conf", "2")
        self.assertEqual(
            {path.name for path in self.work.glob("*.nn")},
            {"network_1.nn", "network_2.nn", "network_3.nn", "other_1.nn", "other_2.nn"},
        )
        models = [self.model(f"network_{index}.nn") for index in range(1, 4)]
        self.assertEqual([model["config"]["seed"] for model in models], [42, 43, 44])
        self.assertNotEqual(models[0]["layers"], models[1]["layers"])
        self.assertEqual(self.model("other_1.nn")["config"]["layers"], [2, 1])

    def test_existing_paths_and_repeated_runs(self):
        (self.work / "network_1.nn").write_text("do not overwrite")
        (self.work / "network_2.nn").mkdir()
        (self.work / "network_3.nn").symlink_to("missing-target")
        (self.work / "protected").write_text("protected target")
        (self.work / "network_4.nn").symlink_to("protected")
        self.generate("network.conf", "2")
        previous = {index: (self.work / f"network_{index}.nn").read_bytes() for index in (5, 6)}
        self.generate("network.conf", "1")
        self.assertTrue((self.work / "network_7.nn").is_file())
        self.assertEqual((self.work / "network_1.nn").read_text(), "do not overwrite")
        self.assertTrue((self.work / "network_2.nn").is_dir())
        self.assertTrue((self.work / "network_3.nn").is_symlink())
        self.assertTrue((self.work / "network_4.nn").is_symlink())
        self.assertEqual((self.work / "protected").read_text(), "protected target")
        for index, content in previous.items():
            self.assertEqual((self.work / f"network_{index}.nn").read_bytes(), content)

    def test_same_basename_in_different_directories(self):
        for folder in ("a", "b"):
            (self.work / folder).mkdir()
            (self.work / folder / "shared.conf").write_text(CONFIG)
        self.generate("a/shared.conf", "1", "b/shared.conf", "1")
        self.assertEqual({path.name for path in self.work.glob("*.nn")}, {"shared_1.nn", "shared_2.nn"})

    def test_config_name_with_spaces_and_multiple_dots(self):
        (self.work / "my network.v1.conf").write_text(CONFIG)
        self.generate("my network.v1.conf", "1")
        self.assertEqual(self.model("my network.v1_1.nn")["format"], "mytorch")

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
                self.assertEqual(list(self.work.glob("*.nn")), [])

    def test_missing_or_invalid_config(self):
        (self.work / "invalid.conf").write_text("layers=3,0,1\n")
        for config in ("missing.conf", "invalid.conf"):
            with self.subTest(config=config):
                result = self.run_command("my_torch_generator", config, "1")
                self.assertEqual(result.returncode, 84)
                self.assertIn(config, result.stderr)
                self.assertEqual(list(self.work.glob("*.nn")), [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
