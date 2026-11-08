"""Augmentation contract, leak prevention and actual C++ input equivalence."""
import itertools
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

from scripts.augment_dataset import augment, input_key, mirror_fen, mirror_row


ROOT = Path(__file__).resolve().parents[1]
FEN = "r3k2r/ppp2ppp/2n5/3pP3/8/2N5/PPP2PPP/R3K2R w KQkq d6 17 29"


class AugmentationTests(unittest.TestCase):
    def test_explicit_asymmetric_board_and_all_metadata(self):
        expected = "r3k2r/ppp2ppp/2n5/8/3Pp3/2N5/PPP2PPP/R3K2R b KQkq d3 17 29"
        self.assertEqual(mirror_fen(FEN), expected)
        for suffix, swapped in (("", ""), (" White", " Black"), (" Black", " White")):
            for label in ("Check", "Checkmate"):
                row = FEN + " " + label + suffix
                self.assertEqual(mirror_row(row), expected + " " + label + swapped)
                self.assertEqual(mirror_row(mirror_row(row)), row)
        self.assertEqual(mirror_row(FEN + " Nothing"), expected + " Nothing")

    def test_every_castling_subset_order_and_en_passant_file(self):
        rights = ["-"] + ["".join(p) for n in range(1, 5)
                          for p in itertools.permutations("KQkq", n)]
        for castling, turn, file in itertools.product(rights, "wb", "-abcdefgh"):
            ep = "-" if file == "-" else file + ("6" if turn == "w" else "3")
            fen = f"8/8/8/8/8/8/8/8 {turn} {castling} {ep} 000 001"
            mirrored = mirror_fen(fen)
            expected_rights = "".join(r for r in "KQkq" if r.swapcase() in castling) or "-"
            self.assertEqual(mirrored.split()[2], expected_rights)
            canonical = "".join(r for r in "KQkq" if r in castling) or "-"
            self.assertEqual(mirror_fen(mirrored), fen.replace(f" {castling} ", f" {canonical} "))
            self.assertNotEqual(input_key(mirrored), input_key(fen))
            self.assertEqual(input_key(mirrored)[-1], input_key(fen)[-1])

    def test_encoded_key_ignores_only_unencoded_fields(self):
        self.assertEqual(input_key(FEN), input_key(FEN.replace("KQkq d6 17 29", "- d6 0 1")))
        for changed in (FEN.replace(" d6 ", " e6 "), FEN.replace(" d6 ", " - "),
                        FEN.replace("R3K2R", "r3K2R")):
            self.assertNotEqual(input_key(FEN), input_key(changed))

    def test_invalid_inputs_and_labels_fail(self):
        for bad in (FEN.replace("d6", "d3"), FEN.replace("KQkq", "KK"),
                    FEN.replace("r3k2r", "r4k2r"), FEN + " extra"):
            with self.assertRaises(ValueError):
                mirror_fen(bad)
        for label in ("Other", "Nothing White", "Check Green", "Check White extra"):
            with self.assertRaises(ValueError):
                mirror_row(FEN + " " + label)

    def fixture(self, root, train, validation, test):
        splits = root / "splits"
        splits.mkdir()
        for name, rows in (("train", train), ("validation", validation), ("test", test)):
            (splits / f"{name}.txt").write_text("\n".join(rows) + "\n")
        return [splits / f"{n}.txt" for n in ("train", "validation", "test")]

    def test_dedup_leak_exclusion_and_label_blind_heldouts(self):
        a = FEN
        b = FEN.replace("2n5", "1n6")
        c = FEN.replace("2n5", "n7")
        d = FEN.replace("2n5", "3n4")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.fixture(root, [a + " Check White", a.replace("17 29", "0 1") + " Check",
                                        mirror_fen(a) + " Check Black", b + " Nothing",
                                        c + " Checkmate", d + " Nothing"],
                                 [mirror_fen(b) + " LABEL IS OPAQUE"], [mirror_fen(c)])
            before = [p.read_bytes() for p in paths]
            report = augment(*paths, root / "derived")
            counts = report["counts"]
            self.assertEqual((counts["source_train_rows"], counts["original_unique"],
                              counts["original_duplicates_removed"]), (6, 5, 1))
            self.assertEqual(counts["excluded_original_duplicate"], 2)
            self.assertEqual(counts["excluded_validation_overlap"], 1)
            self.assertEqual(counts["excluded_test_overlap"], 1)
            self.assertEqual(counts["added"], 1)
            rows = (root / "derived/train.fen").read_text().splitlines()
            self.assertEqual(rows[-1], mirror_fen(d) + " Nothing")
            self.assertEqual(len(rows), 6)
            self.assertEqual([p.read_bytes() for p in paths], before)
            # Changing held-out tails cannot affect selected examples or counts.
            paths[1].write_text(mirror_fen(b) + " Checkmate Black\n")
            paths[2].write_text(mirror_fen(c) + " Nothing\n")
            second = augment(*paths, root / "derived2")
            self.assertEqual(second["counts"], counts)
            self.assertEqual((root / "derived2/train.fen").read_bytes(),
                             (root / "derived/train.fen").read_bytes())
            with self.assertRaises(ValueError):
                augment(*paths, root / "derived")
            with self.assertRaises(ValueError):
                augment(*paths, root / "splits/new")

    def test_original_conflicts_or_overlap_fail_and_mirror_conflicts_are_excluded(self):
        for rows, heldout in (([FEN + " Check", FEN + " Nothing"], []),
                              ([FEN + " Check"], [FEN + " opaque"])):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                paths = self.fixture(root, rows, heldout, [])
                with self.assertRaises(ValueError):
                    augment(*paths, root / "derived")
                self.assertFalse((root / "derived").exists())
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.fixture(root, [FEN + " Check", mirror_fen(FEN) + " Checkmate"], [], [])
            report = augment(*paths, root / "derived")
            self.assertEqual(report["counts"]["of_which_class_conflict"], 2)
            self.assertEqual(report["counts"]["added"], 0)

    def test_deterministic_files_and_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.fixture(root, [FEN + " Check White"], [], [])
            first = augment(*paths, root / "a")
            second = augment(*paths, root / "b")
            self.assertEqual(first["counts"], second["counts"])
            for name in ("train.fen", "additions.fen", "additions-source-lines.csv"):
                self.assertEqual((root / "a" / name).read_bytes(), (root / "b" / name).read_bytes())
            self.assertEqual(json.loads((root / "a/manifest.json").read_text()), first)

    def test_differential_against_production_cpp_encoder(self):
        rng = random.Random(20261108)
        fens = [FEN, mirror_fen(FEN)]
        for _ in range(512):
            ranks = []
            for _ in range(8):
                squares = [rng.choice("PNBRQKpnbrqk...") for _ in range(8)]
                rank = ""; empty = 0
                for piece in squares + ["!"]:
                    if piece == ".":
                        empty += 1
                    else:
                        if empty:
                            rank += str(empty); empty = 0
                        if piece != "!":
                            rank += piece
                ranks.append(rank)
            turn = rng.choice("wb")
            ep = rng.choice("-abcdefgh")
            ep = ep if ep == "-" else ep + ("6" if turn == "w" else "3")
            fen = "/".join(ranks) + f" {turn} Kq {ep} 7 19"
            fens.extend([fen, mirror_fen(fen)])
        with tempfile.TemporaryDirectory() as td:
            executable = Path(td) / "encoder"
            subprocess.run(["g++", "-std=c++17", "-O2", "-Isrc",
                            "tests/fixtures/augmentation_encoder.cpp", "src/chess/Encoding.cpp",
                            "-o", str(executable)], cwd=ROOT, check=True, capture_output=True)
            result = subprocess.run([str(executable)], input=("\n".join(fens) + "\n").encode(),
                                    capture_output=True, check=True)
        self.assertEqual(result.stdout, b"".join(input_key(fen) for fen in fens))


if __name__ == "__main__":
    unittest.main()
