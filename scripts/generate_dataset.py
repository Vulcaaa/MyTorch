"""Extend training data with reproducible random legal games, keeping held-outs fixed."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import logging
from pathlib import Path
import platform
import random
import tempfile

import chess

from scripts.augment_dataset import (
    class_counts, fingerprint, heldout_keys, input_key, read_rows,
)
from scripts.clean_dataset import parse_labeled_fen


def position_label(board: chess.Board) -> str:
    if board.is_checkmate():
        return "Checkmate"
    return "Check" if board.is_check() else "Nothing"


def choose_move(board: chess.Board, rng: random.Random, policy: str) -> chess.Move:
    legal = sorted(board.legal_moves, key=lambda move: move.uci())
    if policy == "random":
        return rng.choice(legal)
    checks = [move for move in legal if board.gives_check(move)]
    # Prefer an immediate mate when available, without a search engine.
    for move in checks:
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return move
    if policy == "checks" and checks and rng.random() < 0.7:
        return rng.choice(checks)
    captures = [move for move in legal if board.is_capture(move)]
    if captures and rng.random() < (0.8 if policy == "captures" else 0.4):
        return rng.choice(captures)
    return rng.choice(legal)


def play_game(settings: tuple[int, int, int, str]) -> dict:
    game, max_plies, seed, policy = settings
    # Independent streams let worker counts change without changing the data.
    rng = random.Random(f"{seed}:{game}")
    game_policy = ("random", "captures", "checks")[(game - 1) % 3] if policy == "mixed" else "random"
    board = chess.Board()
    moves = []
    ending = "max_plies"
    for _ in range(max_plies):
        move = choose_move(board, rng, game_policy)
        board.push(move)
        moves.append(move.uci())
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            ending = outcome.termination.name
            break
    return dict(game=game, policy=game_policy, moves=moves, ending=ending)


def game_records(games: int, max_plies: int, seed: int, policy: str,
                 replay_games: Path | None, workers: int = 1):
    """Produce game records independently of position sampling."""
    if replay_games is not None:
        for game, (_, row) in enumerate(read_rows(replay_games), 1):
            if game > games:
                break
            record = json.loads(row)
            if (record["game"] != game or not record["moves"]
                    or len(record["moves"]) > max_plies
                    or record["policy"] not in ("random", "captures", "checks")):
                raise ValueError(f"invalid replay game {game}")
            yield record
        return
    if workers == 1:
        for game in range(1, games + 1):
            yield play_game((game, max_plies, seed, policy))
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Bound queued work and return games in the same order as serial play.
        batch = workers * 8
        for start in range(1, games + 1, batch):
            settings = ((game, max_plies, seed, policy)
                        for game in range(start, min(start + batch, games + 1)))
            yield from pool.map(play_game, settings)


def generate(train: Path, validation: Path, test: Path, output: Path, *,
             games: int = 1000, max_plies: int = 240, min_ply: int = 16,
             stride: int = 4, per_class: int = 10000, seed: int = 42,
             policy: str = "random", targets: tuple[int, int, int] | None = None,
             per_game: int = 12, replay_games: Path | None = None,
             workers: int = 1) -> dict:
    """Generate legal games with bounded budgets and optional exact class targets.

    Mixed play samples across each game's trajectory. Targeted runs use class
    reservoirs so common classes also cover the later games needed for mates.
    A target shortfall fails without installing an incomplete output directory.
    """
    if min(games, max_plies, stride, per_class) < 1 or not 0 <= min_ply <= max_plies:
        raise ValueError("positive budgets required; min_ply must be in [0, max_plies]")
    if policy not in ("random", "mixed") or min(per_game, workers) < 1:
        raise ValueError("policy must be random or mixed; per_game and workers must be positive")
    if targets is not None and (len(targets) != 3 or min(targets) < 1):
        raise ValueError("targets must contain three positive class counts")
    paths = {"train": train, "validation": validation, "test": test}
    if output.exists():
        raise ValueError(f"{output}: output already exists")
    if any(output.resolve().is_relative_to(p.resolve().parent) for p in paths.values()):
        raise ValueError("output must be separate from the source split directory")
    sources = {name: fingerprint(path) for name, path in paths.items()}
    if replay_games is not None:
        paths["games"] = replay_games
        sources["games"] = fingerprint(replay_games)
    protected = set()
    for path in (validation, test):
        keys, _ = heldout_keys(path)
        protected.update(keys)
    originals = {}
    original_counts = Counter()
    original_rows = 0
    for line, row in read_rows(train):
        fen, label, _ = parse_labeled_fen(row)
        key = input_key(fen)
        if key in protected:
            raise ValueError(f"{train}:{line}: original overlaps held-out inputs")
        if key in originals and originals[key] != label:
            raise ValueError(f"{train}:{line}: conflicting original classes")
        originals[key] = label
        original_counts[label] += 1
        original_rows += 1
    if not original_rows:
        raise ValueError("empty training split")

    seen = set(originals)
    counts = Counter()
    eligible = Counter()
    selected = {label: [] for label in ("Nothing", "Check", "Checkmate")}
    sample_rng = random.Random(seed ^ 0x5A17)
    candidates = Counter()
    excluded = Counter()
    endings = Counter()
    settings = dict(games=games, max_plies=max_plies, min_ply=min_ply,
                    stride=stride, per_class=per_class, seed=seed,
                    policy=policy, targets=targets, per_game=per_game, workers=workers)
    limits = dict(zip(("Nothing", "Check", "Checkmate"), targets or (per_class,) * 3))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".generated-", dir=output.parent) as tmp:
        staging = Path(tmp) / "dataset"
        staging.mkdir()
        with (staging / "train.fen").open("w") as combined, \
                (staging / "additions.fen").open("w") as added, \
                (staging / "additions-source.csv").open("w") as refs, \
                (staging / "games.jsonl").open("w") as game_log:
            for _, row in read_rows(train):
                combined.write(row + "\n")
            refs.write("augmented_line,game,ply\n")
            for record in game_records(games, max_plies, seed, policy, replay_games, workers):
                game = record["game"]
                board = chess.Board()
                game_candidates = {label: [] for label in limits}
                first_ply = min_ply + (game - 1) % stride
                for ply, uci in enumerate(record["moves"], 1):
                    move = chess.Move.from_uci(uci)
                    if replay_games is not None and not board.is_legal(move):
                        raise ValueError(f"illegal replay move: game {game}, ply {ply}")
                    board.push(move)
                    label = position_label(board)
                    if label == "Checkmate" or (ply >= first_ply and (ply - first_ply) % stride == 0):
                        fen = board.fen(en_passant="fen")
                        key = input_key(fen)
                        candidates[label] += 1
                        game_candidates[label].append((fen, key, game, ply))
                if replay_games is not None:
                    outcome = board.outcome(claim_draw=True)
                    ending = outcome.termination.name if outcome is not None else "max_plies"
                    if (ending != record["ending"]
                            or (ending == "max_plies" and len(record["moves"]) != max_plies)):
                        raise ValueError(f"incorrect replay ending: game {game}")
                for label, pool in game_candidates.items():
                    if policy == "mixed" and len(pool) > per_game:
                        excluded["game_cap"] += len(pool) - per_game
                        pool = sorted(sample_rng.sample(pool, per_game), key=lambda row: row[3])
                    for sample in pool:
                        fen, key, _, _ = sample
                        if key in protected:
                            excluded["heldout_overlap"] += 1
                            continue
                        if key in seen:
                            excluded["duplicate_input"] += 1
                            continue
                        seen.add(key)
                        eligible[label] += 1
                        if counts[label] < limits[label]:
                            selected[label].append(sample)
                            counts[label] += 1
                        elif targets is not None:
                            slot = sample_rng.randrange(eligible[label])
                            if slot < limits[label]:
                                selected[label][slot] = sample
                        else:
                            excluded["class_cap"] += 1
                endings[record["ending"]] += 1
                game_log.write(json.dumps(record) + "\n")
                if game % 250 == 0:
                    logging.info("Games %d/%d; selected %s", game, games, dict(counts))
                if targets is not None and all(counts[label] == limit for label, limit in limits.items()):
                    break
            ordered = sorted(((*sample, label) for label, pool in selected.items() for sample in pool),
                             key=lambda sample: (sample[2], sample[3]))
            for line, (fen, _, game, ply, label) in enumerate(ordered, original_rows + 1):
                row = fen + " " + label + "\n"
                combined.write(row)
                added.write(row)
                refs.write(f"{line},{game},{ply}\n")

        if targets is not None and any(counts[label] < limit for label, limit in limits.items()):
            raise ValueError(f"game budget exhausted before class targets: {dict(counts)}; increase --games")

        # Check serialized labels and model inputs independently of selection.
        verified = set(originals)
        for line, row in read_rows(staging / "additions.fen"):
            fen, label, _ = parse_labeled_fen(row)
            board = chess.Board(fen)
            key = input_key(fen)
            if (not board.is_valid() or position_label(board) != label
                    or key in verified or key in protected):
                raise ValueError(f"generated output verification failed at line {line}")
            verified.add(key)
        for name, path in paths.items():
            if fingerprint(path) != sources[name]:
                raise ValueError(f"source changed during generation: {path}")
        artifacts = {}
        for name in ("train.fen", "additions.fen", "additions-source.csv", "games.jsonl"):
            info = fingerprint(staging / name)
            info["path"] = name
            artifacts[name] = info
        report = {
            "version": 1, "sources": sources, "settings": settings,
            "environment": {"python": platform.python_version(), "chess": chess.__version__},
            "generator_sha256": fingerprint(Path(__file__))["sha256"],
            "policy": "standard start; random or alternating random/capture/check-biased legal games; claimable draws stop games",
            "sampling": "stride offset rotates per game; mixed per-game class cap; class reservoirs when targets supplied",
            "heldout_policy": "exclude model-visible validation/test inputs; held-out labels unused",
            "counts": {"original_rows": original_rows,
                       "original_by_class": class_counts(original_counts),
                       "candidates_by_class": class_counts(candidates),
                       "eligible_by_class": class_counts(eligible),
                       "added": sum(counts.values()), "added_by_class": class_counts(counts),
                       "excluded": dict(excluded), "game_endings": dict(endings),
                       "games_played": sum(endings.values()),
                       "output_rows": original_rows + sum(counts.values()),
                       "output_by_class": class_counts(original_counts + counts)},
            "checks": {"added_unique_encoded_inputs": len(verified) - len(originals),
                       "added_validation_test_overlap": 0, "source_hashes_unchanged": True,
                       "added_legality_and_labels_verified": True},
            "artifacts": artifacts,
        }
        (staging / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-games", type=Path,
                        help="Reuse a generated games.jsonl instead of playing new games")
    parser.add_argument("--policy", choices=("random", "mixed"), default="random")
    parser.add_argument("--targets", type=int, nargs=3, metavar=("NOTHING", "CHECK", "CHECKMATE"))
    for name, default in (("games", 1000), ("max-plies", 240), ("min-ply", 16),
                          ("stride", 4), ("per-class", 10000), ("seed", 42), ("per-game", 12),
                          ("workers", 1)):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args()
    settings = {name: getattr(args, name) for name in
                ("games", "max_plies", "min_ply", "stride", "per_class", "seed", "policy", "per_game", "replay_games", "workers")}
    settings["targets"] = tuple(args.targets) if args.targets else None
    try:
        report = generate(*(args.splits / f"{name}.txt" for name in
                            ("train", "validation", "test")), args.output, **settings)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
