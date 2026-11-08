import json
from pathlib import Path
import tempfile
import unittest

from scripts.freeze_dataset import freeze, verify
from scripts.reference_dataset import fingerprint
from test_split_dataset import row


class FrozenTrainingTests(unittest.TestCase):
    def fixture(self, root):
        storage = root / "dataset/references/base"
        storage.mkdir(parents=True)
        artifacts = {}
        for i, role in enumerate(("train", "validation", "test")):
            path = storage / f"{role}.fen"
            path.write_text(row(i, "Nothing"))
            artifacts[role] = dict(path=path.name, **fingerprint(path))
        manifest = root / "dataset/manifests/base.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps(dict(storage=str(storage.relative_to(root)), artifacts=artifacts)))
        return storage, manifest

    def test_freeze_survives_source_removal_and_detects_modified_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, manifest = self.fixture(root)
            train = root / "prepared.fen"
            train.write_text(row(0, "Nothing") + row(3, "Check") + row(4, "Checkmate"))
            extra = root / "origin.json"
            extra.write_text('{"seed":43}\n')
            frozen = freeze("derived", train, manifest, {"origin.json": extra}, root=root)
            data = json.loads(frozen.read_text())
            self.assertEqual(verify(data, root=root)["train"], dict(Nothing=1, Check=1, Checkmate=1))
            storage = root / data["storage"]
            self.assertEqual((storage / "splits/validation.txt").read_bytes(), (base / "validation.fen").read_bytes())
            with self.assertRaisesRegex(ValueError, "already exists"):
                freeze("derived", train, manifest, {}, root=root)
            train.unlink()
            extra.unlink()
            verify(data, root=root)
            (storage / "provenance/origin.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "modified file"):
                verify(data, root=root)

    def test_overlap_fails_without_installing_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, manifest = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "overlapping"):
                freeze("bad", base / "validation.fen", manifest, {}, root=root)
            self.assertFalse((root / "dataset/references/bad").exists())
            self.assertFalse((root / "dataset/manifests/bad.json").exists())


if __name__ == "__main__":
    unittest.main()
