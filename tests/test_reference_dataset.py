"""Reference integrity failures and deterministic restoration on small fixtures."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiment import load_spec
from scripts.reference_dataset import ARTIFACTS, ROOT, TOOLS, fingerprint, prepare, recreate, verify
from test_split_dataset import row


class ReferenceDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mytorch-reference-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for relative in (*TOOLS, "configs/experiments/compare-network-sizes.yml", "configs/networks/basic-network.conf"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, path)
        source = self.root / "dataset/raw/examples.txt"
        source.parent.mkdir(parents=True)
        lines = [row(index, label) for index, label in enumerate(
            ["Nothing"] * 12 + ["Check"] * 9 + ["Checkmate"] * 6)]
        # Include duplicate and invalid rows, as in the real raw sources.
        source.write_text("".join(lines) + row(0, "Nothing", "5 12") + "invalid\n")
        self.manifest_path = prepare("chess-v1", root=self.root)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.storage = self.root / self.manifest["storage"]

    def artifact(self, role):
        return self.storage / ARTIFACTS[role]

    def refresh_hash(self, role):
        self.manifest["artifacts"][role].update(fingerprint(self.artifact(role)))

    def test_recreation_matches_every_hash_and_ignores_unselected_sources(self):
        # A later raw addition must never silently enter an existing reference.
        (self.root / "dataset/raw/later.txt").write_text(row(0, "Checkmate"))
        output = self.root / "recreated"
        recreate(self.manifest, root=self.root, output=output)
        for relative in ARTIFACTS.values():
            self.assertEqual((self.storage / relative).read_bytes(), (output / relative).read_bytes())
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.root.rglob("*") if path.is_file()}
        counts = verify(self.manifest, root=self.root)
        self.assertEqual(sum(counts["cleaned"].values()), 27)
        self.assertEqual(self.manifest["cleaning_summary"]["duplicates"], 1)
        self.assertEqual(self.manifest["cleaning_summary"]["rejected_records"], 1)
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before})

    def test_refuses_overwriting_and_restores_missing_local_reference(self):
        with self.assertRaisesRegex(ValueError, "new version"):
            prepare("chess-v1", root=self.root)
        with self.assertRaisesRegex(ValueError, "will not be overwritten"):
            recreate(self.manifest, root=self.root)
        shutil.rmtree(self.storage)
        recreate(self.manifest, root=self.root)
        verify(self.manifest, root=self.root)
        self.assertEqual(json.loads(self.manifest_path.read_text()), self.manifest)

    def test_missing_or_modified_files_fail(self):
        paths = [self.root / self.manifest["sources"][0]["path"], self.artifact("train"),
                 self.artifact("provenance")]
        for path in paths:
            original = path.read_bytes()
            for action in ("remove", "modify"):
                with self.subTest(path=path, action=action):
                    if action == "remove":
                        path.unlink()
                    else:
                        path.write_bytes(original + b"\n")
                    with self.assertRaisesRegex(ValueError, "missing file|modified file"):
                        verify(self.manifest, root=self.root)
                    path.write_bytes(original)

    def test_manifest_source_and_split_counts_are_checked(self):
        original = copy.deepcopy(self.manifest)
        self.manifest["sources"][0]["statistics"]["records"] += 1
        with self.assertRaisesRegex(ValueError, "source counts"):
            verify(self.manifest, root=self.root)
        self.manifest = copy.deepcopy(original)
        self.manifest["artifacts"]["train"]["statistics"]["records"] += 1
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            verify(self.manifest, root=self.root)

    def test_overlap_ignores_move_counters_even_with_updated_hash(self):
        fields = self.artifact("train").read_text().splitlines()[0].split()
        fields[4:6] = ["99", "123"]
        with self.artifact("validation").open("a") as stream:
            stream.write(" ".join(fields) + "\n")
        self.refresh_hash("validation")
        with self.assertRaisesRegex(ValueError, "overlapping model input"):
            verify(self.manifest, root=self.root)

    def test_missing_coverage_and_changed_labels_even_with_updated_hash(self):
        path = self.artifact("train")
        original = path.read_text()
        lines = original.splitlines(keepends=True)
        path.write_text("".join(lines[1:]))
        self.refresh_hash("train")
        with self.assertRaisesRegex(ValueError, "cover every cleaned input"):
            verify(self.manifest, root=self.root)
        fields = lines[0].split()
        fields[6] = "Check" if fields[6] == "Nothing" else "Nothing"
        path.write_text(" ".join(fields) + "\n" + "".join(lines[1:]))
        self.refresh_hash("train")
        with self.assertRaisesRegex(ValueError, "input/label absent"):
            verify(self.manifest, root=self.root)

    def test_metadata_and_provenance_counts_are_checked_independently(self):
        path = self.artifact("split_metadata")
        original = path.read_text()
        info = json.loads(original)
        info["seed"] = 99
        path.write_text(json.dumps(info))
        self.refresh_hash("split_metadata")
        with self.assertRaisesRegex(ValueError, "split metadata disagrees"):
            verify(self.manifest, root=self.root)
        path.write_text(original)
        self.refresh_hash("split_metadata")
        path = self.artifact("provenance")
        path.write_text("".join(path.read_text().splitlines(keepends=True)[1:]))
        self.refresh_hash("provenance")
        with self.assertRaisesRegex(ValueError, "provenance count"):
            verify(self.manifest, root=self.root)

    def test_recreation_fails_without_installing_different_outputs(self):
        output = self.root / "bad-recreation"
        self.manifest["artifacts"]["train"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "modified file"):
            recreate(self.manifest, root=self.root, output=output)
        self.assertFalse(output.exists())

    def test_recreation_accepts_python_and_tool_drift_when_artifacts_match(self):
        output = self.root / "recreated"
        with (self.root / TOOLS[0]).open("a") as stream:
            stream.write("\n# Comment-only change.\n")
        with patch("scripts.reference_dataset.platform.python_version", return_value="3.14.0"), \
                self.assertWarns(UserWarning) as caught:
            recreate(self.manifest, root=self.root, output=output)
        self.assertIn("reference was prepared with", str(caught.warnings[0].message))
        self.assertTrue(any("preparation tool differs" in str(warning.message)
                            for warning in caught.warnings))
        verify(self.manifest, root=self.root, storage=output)
        # Drift must never weaken the output integrity check or install bad artifacts.
        self.manifest["artifacts"]["train"]["sha256"] = "0" * 64
        rejected = self.root / "bad-recreation"
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "modified file"):
            recreate(self.manifest, root=self.root, output=rejected)
        self.assertFalse(rejected.exists())

    def test_default_experiment_resolves_frozen_artifacts_and_excludes_test(self):
        plan = load_spec(self.root / "configs/experiments/compare-network-sizes.yml")
        self.assertEqual(set(plan["datasets"]), {"train", "validation", "split_metadata"})
        for role, entry in plan["datasets"].items():
            self.assertEqual(Path(entry["path"]).resolve(), self.artifact(role).resolve())
            self.assertEqual(entry["sha256"], self.manifest["artifacts"][role]["sha256"])

    def test_verify_cli_reports_failure_without_traceback(self):
        missing = self.root / "missing.json"
        result = subprocess.run([sys.executable, "-m", "scripts.reference_dataset", "verify", str(missing)],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Reference error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
