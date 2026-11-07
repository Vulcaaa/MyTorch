"""Read-only dataset diagnostics; no training, balancing or chess-rule oracle."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from scripts.clean_dataset import parse_labeled_fen

LABELS = ("Nothing", "Check", "Checkmate")
PIECES = "PNBRQKpnbrqk"


def position_features(fen: str) -> dict:
    """Extract descriptive features from an already validated six-field FEN."""
    board, turn, _, en_passant, *_ = fen.split()
    squares = []
    for rank, text in enumerate(board.split("/")):
        column = 0
        for symbol in text:
            if symbol.isdigit():
                column += int(symbol)
            else:
                squares.append((rank * 8 + column, symbol))
                column += 1
    counts = Counter(piece for _, piece in squares)
    return {"turn": turn, "piece_count": len(squares), "pieces": counts,
            "squares": squares, "en_passant": en_passant,
            "shortcut": f"{turn}:{len(squares)}"}


def _class_stats() -> dict:
    return {"rows": 0, "turn": Counter(), "piece_count": Counter(),
            "pieces": Counter(), "en_passant": Counter(),
            "king_count_anomalies": 0,
            "square_piece_counts": {piece: [0] * 64 for piece in PIECES}}


def analyze_split(path: str | Path) -> tuple[dict, dict]:
    """Count every nonblank row; reject malformed input with its line number.

    Returned input keys use the same equivalence as the C++ encoder. Repeated
    rows remain in distributions (important for balanced training datasets).
    """
    path = Path(path)
    classes = {label: _class_stats() for label in LABELS}
    inputs = {}
    conflicts = set()
    shortcuts = {}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            try:
                line = raw.decode("utf-8")
                if not line.strip():
                    continue
                fen, label, key = parse_labeled_fen(line)
            except (UnicodeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            features = position_features(fen)
            stats = classes[label]
            stats["rows"] += 1
            stats["turn"][features["turn"]] += 1
            stats["piece_count"][str(features["piece_count"])] += 1
            stats["pieces"].update(features["pieces"])
            stats["en_passant"][features["en_passant"]] += 1
            stats["king_count_anomalies"] += (
                features["pieces"]["K"] != 1 or features["pieces"]["k"] != 1)
            for square, piece in features["squares"]:
                stats["square_piece_counts"][piece][square] += 1
            if key in inputs and inputs[key] != label:
                conflicts.add(key)
            inputs.setdefault(key, label)
            shortcuts.setdefault(features["shortcut"], Counter())[label] += 1
    rows = sum(stats["rows"] for stats in classes.values())
    if not rows:
        raise ValueError(f"{path}: no labeled positions")
    summary = {"file": path.name, "sha256": digest.hexdigest(), "rows": rows,
               "unique_inputs": len(inputs), "repeated_input_rows": rows - len(inputs),
               "conflicting_inputs": len(conflicts), "classes": classes,
               "shortcut_groups": shortcuts}
    return summary, inputs


def input_overlap(left: dict, right: dict) -> dict:
    """Count shared model inputs, including those with different labels."""
    common = left.keys() & right.keys()
    return {"shared_inputs": len(common),
            "different_labels": sum(left[key] != right[key] for key in common)}


def _majority(counts: dict) -> str:
    # Stable ties follow the public output-class order.
    return max(LABELS, key=lambda label: counts.get(label, 0))


def _scores(confusion: list[list[int]]) -> dict:
    total = sum(map(sum, confusion))
    f1 = []
    for i in range(len(LABELS)):
        denominator = sum(confusion[i]) + sum(row[i] for row in confusion)
        f1.append(2 * confusion[i][i] / denominator if denominator else 0.0)
    return {"accuracy": sum(confusion[i][i] for i in range(3)) / total,
            "macro_f1": sum(f1) / len(LABELS), "confusion_matrix": confusion}


def shortcut_baselines(train: dict, validation: dict) -> dict:
    """Fit majority rules on train only, evaluate on aggregated validation rows.

    The second rule uses only turn and total piece count, exposing possible
    dataset shortcuts. It is a diagnostic, not a chess classifier to deliver.
    """
    majority = _majority({label: train["classes"][label]["rows"] for label in LABELS})
    rules = {group: _majority(counts) for group, counts in train["shortcut_groups"].items()}
    matrices = {name: [[0] * 3 for _ in LABELS]
                for name in ("train_majority", "turn_and_piece_count")}
    unseen = 0
    for group, counts in validation["shortcut_groups"].items():
        if group not in rules:
            unseen += sum(counts.values())
        for label, count in counts.items():
            for name, prediction in (("train_majority", majority),
                                     ("turn_and_piece_count", rules.get(group, majority))):
                matrices[name][LABELS.index(label)][LABELS.index(prediction)] += count
    return {"training_majority_class": majority, "unseen_validation_group_rows": unseen,
            **{name: _scores(matrix) for name, matrix in matrices.items()}}
