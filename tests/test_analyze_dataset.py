import tempfile
import unittest
from pathlib import Path

from scripts.analyze_dataset import (
    analyze_split, input_overlap, position_features, shortcut_baselines,
)


class DatasetAnalysisTests(unittest.TestCase):
    def test_fen_square_orientation_and_material(self):
        features = position_features("k7/8/8/8/8/8/8/7K b - - 0 1")
        self.assertEqual(features["squares"], [(0, "k"), (63, "K")])
        self.assertEqual(features["piece_count"], 2)
        self.assertEqual(features["shortcut"], "b:2")

    def test_model_equivalence_conflicts_and_anomalies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            path.write_text("k7/8/8/8/8/8/8/7K w - - 0 1 Nothing\n"
                            "k7/8/8/8/8/8/8/7K w KQ - 8 20 Nothing\n"
                            "k7/8/8/8/8/8/8/7K w - - 1 2 Check\n"
                            "8/8/8/8/8/8/8/8 b - - 0 1 Checkmate\n")
            summary, inputs = analyze_split(path)
            self.assertEqual(summary["rows"], 4)
            self.assertEqual(summary["unique_inputs"], 2)
            self.assertEqual(summary["repeated_input_rows"], 2)
            self.assertEqual(summary["conflicting_inputs"], 1)
            self.assertEqual(summary["classes"]["Nothing"]["square_piece_counts"]["k"][0], 2)
            self.assertEqual(summary["classes"]["Checkmate"]["king_count_anomalies"], 1)
            other = dict(inputs)
            other[next(iter(other))] = "Checkmate"
            self.assertEqual(input_overlap(inputs, other),
                             {"shared_inputs": 2, "different_labels": 1})

    def test_invalid_and_empty_files_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text("\ninvalid\n")
            with self.assertRaisesRegex(ValueError, r"bad.txt:2:"):
                analyze_split(path)
            path.write_text("\n")
            with self.assertRaisesRegex(ValueError, "no labeled positions"):
                analyze_split(path)

    def test_baseline_uses_training_labels_and_falls_back_for_unseen_groups(self):
        train = {"classes": {label: {"rows": count} for label, count in
                             [("Nothing", 3), ("Check", 2), ("Checkmate", 0)]},
                 "shortcut_groups": {"w:2": {"Nothing": 3}, "b:3": {"Check": 2}}}
        validation = {"shortcut_groups": {"w:2": {"Check": 2},
                                           "b:3": {"Check": 1},
                                           "w:4": {"Nothing": 1}}}
        scores = shortcut_baselines(train, validation)
        self.assertEqual(scores["training_majority_class"], "Nothing")
        self.assertEqual(scores["unseen_validation_group_rows"], 1)
        self.assertEqual(scores["train_majority"]["accuracy"], 0.25)
        self.assertEqual(scores["turn_and_piece_count"]["accuracy"], 0.5)
        self.assertAlmostEqual(scores["turn_and_piece_count"]["macro_f1"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
