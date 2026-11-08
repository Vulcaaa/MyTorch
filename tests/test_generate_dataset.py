"""Legal game replay, correct labels, reproducibility and held-out isolation."""

import csv
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

try:
    import chess
except ImportError:
    chess = None

from scripts.augment_dataset import input_key
if chess is not None:
    from scripts.generate_dataset import game_records, generate, position_label


@unittest.skipIf(chess is None, "optional generation dependency: scripts/requirements.txt")
class GeneratedDatasetTests(unittest.TestCase):
    def fixture(self, root):
        splits = root / "splits"
        splits.mkdir()
        paths = [splits / f"{name}.txt" for name in ("train", "validation", "test")]
        paths[0].write_text(chess.STARTING_FEN + " Nothing\n")
        for path in paths[1:]:
            path.write_text("")
        return paths

    def test_labels_include_mate_check_and_stalemate(self):
        board = chess.Board()
        self.assertEqual(position_label(board), "Nothing")
        for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
            board.push_uci(move)
        self.assertEqual(position_label(board), "Checkmate")
        self.assertEqual(position_label(chess.Board(
            "7k/8/5KQ1/8/8/8/8/8 b - - 0 1")), "Nothing")
        self.assertEqual(position_label(chess.Board(
            "7k/8/5K1Q/8/8/8/8/8 b - - 0 1")), "Check")

    def test_deterministic_games_replay_and_sample_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.fixture(root)
            settings = dict(games=12, max_plies=120, min_ply=4, stride=3, per_class=50)
            first = generate(*paths, root / "a", **settings)
            second = generate(*paths, root / "b", **settings)
            self.assertEqual(first, second)
            self.assertGreater(first["counts"]["added"], 50)
            for name in first["artifacts"]:
                self.assertEqual((root / "a" / name).read_bytes(), (root / "b" / name).read_bytes())
            replayed = {}
            for line in (root / "a/games.jsonl").read_text().splitlines():
                game = json.loads(line)
                board = chess.Board()
                for ply, uci in enumerate(game["moves"], 1):
                    self.assertFalse(board.is_game_over(claim_draw=True))
                    move = chess.Move.from_uci(uci)
                    self.assertIn(move, board.legal_moves)
                    board.push(move)
                    replayed[game["game"], ply] = board.fen(en_passant="fen")
            rows = (root / "a/train.fen").read_text().splitlines()
            self.assertEqual(rows[0], paths[0].read_text().strip())
            with (root / "a/additions-source.csv").open() as source:
                refs = list(csv.DictReader(source))
            self.assertEqual(len(refs), first["counts"]["added"])
            keys = set()
            turns = {label: set() for label in ("Nothing", "Check", "Checkmate")}
            for ref in refs:
                fen = replayed[int(ref["game"]), int(ref["ply"])]
                row = rows[int(ref["augmented_line"]) - 1]
                self.assertEqual(row, fen + " " + position_label(chess.Board(fen)))
                turns[row.split()[6]].add(fen.split()[1])
                self.assertNotIn(input_key(fen), keys)
                keys.add(input_key(fen))
            self.assertEqual(turns["Nothing"], {"w", "b"})
            self.assertEqual(turns["Check"], {"w", "b"})
            self.assertTrue(all(n <= 50 for n in first["counts"]["added_by_class"].values()))

    def test_heldouts_are_label_blind_and_originals_stay_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.fixture(root)
            settings = dict(games=2, max_plies=40, min_ply=1, stride=1)
            generate(*paths, root / "initial", **settings)
            additions = (root / "initial/additions.fen").read_text().splitlines()
            heldouts = [" ".join(row.split()[:6]) for row in additions[:2]]
            paths[1].write_text(heldouts[0] + " opaque label\n")
            paths[2].write_text(heldouts[1] + "\n")
            before = [p.read_bytes() for p in paths]
            report = generate(*paths, root / "filtered", **settings)
            self.assertEqual([p.read_bytes() for p in paths], before)
            self.assertEqual(report["counts"]["excluded"]["heldout_overlap"], 2)
            keys = {input_key(" ".join(row.split()[:6])) for row in
                    (root / "filtered/train.fen").read_text().splitlines()}
            self.assertFalse(keys & {input_key(fen) for fen in heldouts})
            paths[1].write_text(heldouts[0] + " Checkmate\n")
            generate(*paths, root / "relabeled", **settings)
            self.assertEqual((root / "filtered/train.fen").read_bytes(),
                             (root / "relabeled/train.fen").read_bytes())
            with self.assertRaises(ValueError):
                generate(*paths, root / "filtered", **settings)
            with self.assertRaises(ValueError):
                generate(*paths, root / "splits/derived", **settings)
            paths[1].write_text(paths[0].read_text())
            with self.assertRaises(ValueError):
                generate(*paths, root / "overlap", **settings)
            self.assertFalse((root / "overlap").exists())

    def test_invalid_budgets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.fixture(root)
            for settings in (dict(games=0), dict(stride=0), dict(per_class=-1),
                             dict(min_ply=-1), dict(min_ply=20, max_plies=10),
                             dict(targets=(1, 2)), dict(targets=(1, 0, 1)),
                             dict(policy="unknown"), dict(per_game=0), dict(workers=0)):
                with self.assertRaises(ValueError):
                    generate(*paths, root / "invalid", **settings)
            self.assertFalse((root / "invalid").exists())

    def test_worker_count_does_not_change_games(self):
        serial = list(game_records(9, 40, 43, "mixed", None))
        parallel = list(game_records(9, 40, 43, "mixed", None, workers=2))
        self.assertEqual(serial, parallel)

    def test_mixed_targets_are_exact_replayable_and_capped_per_game(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.fixture(root)
            settings = dict(games=100, max_plies=160, min_ply=4, stride=2,
                            policy="mixed", targets=(30, 20, 2), per_game=3)
            first = generate(*paths, root / "a", **settings)
            second = generate(*paths, root / "b", **settings)
            self.assertEqual(first, second)
            replay = generate(*paths, root / "replayed", **settings,
                              replay_games=root / "a/games.jsonl")
            self.assertEqual(replay["counts"], first["counts"])
            self.assertEqual(first["counts"]["added_by_class"],
                             dict(Nothing=30, Check=20, Checkmate=2))
            for name in first["artifacts"]:
                self.assertEqual((root / "a" / name).read_bytes(), (root / "b" / name).read_bytes())
                self.assertEqual((root / "a" / name).read_bytes(), (root / "replayed" / name).read_bytes())
            replayed = {}
            policies = set()
            for game in map(json.loads, (root / "a/games.jsonl").read_text().splitlines()):
                policies.add(game["policy"])
                board = chess.Board()
                for ply, uci in enumerate(game["moves"], 1):
                    self.assertFalse(board.is_game_over(claim_draw=True))
                    move = chess.Move.from_uci(uci)
                    self.assertIn(move, board.legal_moves)
                    board.push(move)
                    replayed[game["game"], ply] = board.fen(en_passant="fen")
            self.assertEqual(policies, {"random", "captures", "checks"})
            rows = (root / "a/train.fen").read_text().splitlines()
            per_game = Counter()
            with (root / "a/additions-source.csv").open() as source:
                for ref in csv.DictReader(source):
                    game, ply = int(ref["game"]), int(ref["ply"])
                    fen = replayed[game, ply]
                    label = position_label(chess.Board(fen))
                    self.assertEqual(rows[int(ref["augmented_line"]) - 1], fen + " " + label)
                    per_game[game, label] += 1
            self.assertLessEqual(max(per_game.values()), 3)

    def test_illegal_replay_does_not_install_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.fixture(root)
            replay = root / "games.jsonl"
            replay.write_text(json.dumps(dict(game=1, policy="random", moves=["a1a8"],
                                              ending="max_plies")) + "\n")
            with self.assertRaisesRegex(ValueError, "illegal replay move"):
                generate(*paths, root / "invalid", replay_games=replay)
            self.assertFalse((root / "invalid").exists())

    def test_unmet_targets_leave_no_partial_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "game budget exhausted"):
                generate(*paths, root / "shortfall", games=1, max_plies=2,
                         min_ply=1, targets=(1, 1, 1), policy="mixed")
            self.assertFalse((root / "shortfall").exists())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["splits"])


if __name__ == "__main__":
    unittest.main()
