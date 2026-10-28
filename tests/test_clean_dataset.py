import json
import tempfile
import unittest
from pathlib import Path

from scripts.clean_dataset import clean_dataset


WHITE = "8/8/8/8/8/8/8/8 w - - 0 1"
BLACK = "8/8/8/8/8/8/8/8 b - - 0 1"


class CleanDatasetTests(unittest.TestCase):
    def test_cleaning_across_files_and_source_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset"
            raw = dataset / "raw"
            raw.mkdir(parents=True)
            (raw / "a.txt").write_text(
                f"{WHITE} Check White\n"
                f"{BLACK} Nothing\n"
                "bad row\n"
                f"{WHITE} Checkmate Black\n"
                "\n",
                encoding="utf-8",
            )
            (raw / "b.txt").write_text(
                "8/8/8/8/8/8/8/8 w KQ - 15 20 Check Black\n"
                "8/8/8/8/8/8/8/8 b - - 12 99 Nothing\n",
                encoding="utf-8",
            )
            report = clean_dataset(dataset)
            self.assertEqual(report["records"], 6)
            self.assertEqual(report["accepted"], 1)
            self.assertEqual(report["duplicates"], 1)
            self.assertEqual(report["conflicting_records"], 3)
            self.assertEqual(report["by_class"], {"Nothing": 1, "Check": 0, "Checkmate": 0})
            self.assertEqual(report["by_source"]["raw/a.txt"], {"records": 4, "retained": 1, "rejected": 1})
            self.assertEqual(report["by_source"]["raw/b.txt"], {"records": 2, "retained": 1, "rejected": 0})
            self.assertEqual(report["rejected"][0]["file"], "raw/a.txt")
            self.assertEqual(report["rejected"][0]["line"], 3)
            self.assertEqual(
                [(ref["file"], ref["line"], ref["label"]) for ref in report["conflicts"][0]["sources"]],
                [("raw/a.txt", 1, "Check"), ("raw/a.txt", 4, "Checkmate"), ("raw/b.txt", 1, "Check")],
            )
            processed = dataset / "processed"
            self.assertEqual((processed / "cleaned.txt").read_text(), BLACK + " Nothing\n")
            self.assertEqual(len((processed / "provenance.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len(json.loads((processed / "provenance.jsonl").read_text().strip())), 2)
            self.assertEqual(clean_dataset(dataset), report)  # Output is never read as input.

    def test_labels_and_fen_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset"
            dataset.mkdir()
            (dataset / "source.txt").write_text(
                f"{WHITE} Check\n"
                f"{BLACK} Checkmate White\n"
                f"{WHITE} Nothing White\n"
                "8/8/8/8/8/8/8/8 w KK - 0 1 Nothing\n"
                "8/8/8/8/8/8/8/8 w - a3 0 1 Nothing\n"
                "8/8/8/8/8/8/8/8 w - - 0 0 Nothing\n"
                f"{WHITE}\n"
                f"{BLACK} Unknown\n",
                encoding="utf-8",
            )
            report = clean_dataset(dataset)
            self.assertEqual(report["accepted"], 2)
            self.assertEqual(len(report["rejected"]), 6)
            self.assertEqual(report["by_class"], {"Nothing": 0, "Check": 1, "Checkmate": 1})
            self.assertEqual([item["line"] for item in report["rejected"]], [3, 4, 5, 6, 7, 8])

    def test_en_passant_file_remains_part_of_the_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset"
            dataset.mkdir()
            (dataset / "source.txt").write_text(
                "8/8/8/8/8/8/8/8 w - a6 0 1 Check\n"
                "8/8/8/8/8/8/8/8 w - b6 0 1 Nothing\n"
                "8/8/8/8/8/8/8/8 w - a6 9 12 Checkmate\n",
                encoding="utf-8",
            )
            report = clean_dataset(dataset)
            self.assertEqual(report["accepted"], 1)
            self.assertEqual(report["conflicting_records"], 2)
            self.assertIn(" b6 ", (dataset / "processed" / "cleaned.txt").read_text())


if __name__ == "__main__":
    unittest.main()
