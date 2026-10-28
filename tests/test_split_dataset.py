import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.clean_dataset import parse_labeled_fen
from scripts.split_dataset import split_dataset


PIECES = "PNBRQKpnbrqk"


def row(index, label, counters="0 1"):
    piece = PIECES[index % len(PIECES)]
    turn = "w" if index // len(PIECES) % 2 == 0 else "b"
    en_passant = ("-", "a6" if turn == "w" else "a3")[index // 24]
    return f"{piece}7/8/8/8/8/8/8/8 {turn} - {en_passant} {counters} {label}\n"


class SplitDatasetTests(unittest.TestCase):
    def make_input(self, root, sizes=(12, 9, 6), duplicate=False):
        path = root / "cleaned.txt"
        lines = [row(index, label)
                 for label, start, size in zip(("Nothing", "Check", "Checkmate"),
                                               (0, sizes[0], sizes[0] + sizes[1]), sizes)
                 for index in range(start, start + size)]
        if duplicate:
            lines.append(row(0, "Nothing", "7 42"))
        path.write_text("".join(lines), encoding="utf-8")
        return path

    def test_reproducibility_counts_and_no_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_input(root, duplicate=True)
            first = root / "first"
            second = root / "second"
            info = split_dataset(source, first, seed=17, ratios=(0.6, 0.2, 0.2))
            self.assertEqual(info, split_dataset(source, second, seed=17, ratios=(0.6, 0.2, 0.2)))
            for name in ("train", "validation", "test", "split_info"):
                suffix = ".json" if name == "split_info" else ".txt"
                self.assertEqual((first / f"{name}{suffix}").read_bytes(),
                                 (second / f"{name}{suffix}").read_bytes())
            keys = {}
            for name in ("train", "validation", "test"):
                lines = (first / f"{name}.txt").read_text().splitlines()
                keys[name] = {parse_labeled_fen(line)[2] for line in lines}
                counts = Counter(parse_labeled_fen(line)[1] for line in lines)
                self.assertEqual(dict(counts), {label: value for label, value in info["output"][name].items() if value})
            self.assertFalse(keys["train"] & keys["validation"])
            self.assertFalse(keys["train"] & keys["test"])
            self.assertFalse(keys["validation"] & keys["test"])
            labels = {parse_labeled_fen(line)[2]: parse_labeled_fen(line)[1]
                      for line in source.read_text().splitlines()}
            expected = {
                "train": {"Nothing": 7, "Check": 5, "Checkmate": 4},
                "validation": {"Nothing": 3, "Check": 2, "Checkmate": 1},
                "test": {"Nothing": 2, "Check": 2, "Checkmate": 1},
            }
            for name in keys:
                self.assertEqual(dict(Counter(labels[key] for key in keys[name])), expected[name])
            self.assertEqual(sum(len(part) for part in keys.values()), 27)
            self.assertEqual(sum(sum(values.values()) for values in info["output"].values()), 28)
            self.assertEqual(info["balance_training"], "none")

    def test_balancing_changes_training_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_input(root)
            plain = split_dataset(source, root / "plain", seed=5)
            balanced = split_dataset(source, root / "balanced", seed=5, balance_training=True)
            for name in ("validation", "test"):
                self.assertEqual(plain["output"][name], balanced["output"][name])
                self.assertEqual((root / "plain" / f"{name}.txt").read_bytes(),
                                 (root / "balanced" / f"{name}.txt").read_bytes())
            self.assertEqual(len(set(balanced["output"]["train"].values())), 1)
            self.assertEqual(plain["output"]["train"], balanced["before_balancing"]["train"])
            self.assertEqual(json.loads((root / "balanced" / "split_info.json").read_text()), balanced)

    def test_impossible_class_and_conflict_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_input(root, sizes=(4, 4, 2))
            with self.assertRaisesRegex(ValueError, "Checkmate: need at least 3 distinct positions"):
                split_dataset(source, root / "splits")
            self.assertFalse((root / "splits").exists())
            source = self.make_input(root, sizes=(4, 4, 4))
            source.write_text(source.read_text() + row(0, "Check"))
            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                split_dataset(source, root / "splits")
            with self.assertRaisesRegex(ValueError, "ratios must be"):
                split_dataset(source, root / "splits", ratios=(0.8, 0.3, -0.1))


if __name__ == "__main__":
    unittest.main()
