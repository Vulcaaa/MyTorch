"""Clean labeled chess datasets using the C++ loader's input contract."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


LABELS = {"Nothing", "Check", "Checkmate"}
PIECES = set("PNBRQKpnbrqk")
EXCLUDED_DIRS = {"processed", "splits"}


@dataclass
class Position:
    fen: str
    label: str
    sources: list[dict[str, object]] = field(default_factory=list)
    conflicting: bool = False


def parse_labeled_fen(line: str) -> tuple[str, str, tuple[str, str, str]]:
    """Validate a labeled row and return its FEN, class, and model input key."""
    fields = line.split()
    if len(fields) < 7:
        raise ValueError("expected six FEN fields and a label")
    board, turn, castling, en_passant, halfmove, fullmove = fields[:6]
    label = fields[6]
    if label not in LABELS:
        raise ValueError("unknown label")
    if len(fields) > 7:
        if label == "Nothing" or fields[7] not in ("White", "Black"):
            raise ValueError("invalid label suffix")
        if len(fields) > 8:
            raise ValueError("unexpected data after label")

    ranks = board.split("/")
    if len(ranks) != 8:
        raise ValueError("FEN: expected 8 ranks")
    for rank in ranks:
        width = 0
        previous_digit = False
        for symbol in rank:
            if symbol in "12345678":
                if previous_digit:
                    raise ValueError("FEN: consecutive empty-square counts")
                width += int(symbol)
                previous_digit = True
            elif symbol in PIECES:
                width += 1
                previous_digit = False
            else:
                raise ValueError("FEN: invalid piece symbol")
            if width > 8:
                raise ValueError("FEN: rank exceeds 8 squares")
        if width != 8:
            raise ValueError("FEN: each rank must contain 8 squares")
    if turn not in ("w", "b"):
        raise ValueError("FEN: side to move must be w or b")
    if castling != "-" and (not castling or any(
        right not in "KQkq" or castling.count(right) != 1 for right in castling
    )):
        raise ValueError("FEN: invalid or duplicate castling right")
    if en_passant != "-" and (
        len(en_passant) != 2
        or en_passant[0] not in "abcdefgh"
        or en_passant[1] != ("6" if turn == "w" else "3")
    ):
        raise ValueError("FEN: invalid en passant target for side to move")
    if not halfmove.isascii() or not halfmove.isdecimal():
        raise ValueError("FEN: halfmove clock must be a nonnegative integer")
    if not fullmove.isascii() or not fullmove.isdecimal() or not fullmove.strip("0"):
        raise ValueError("FEN: fullmove number must be a positive integer")

    # The C++ encoder sees the board, turn and en-passant file. It validates
    # castling and counters but does not include them in the 777 inputs.
    key = (board, turn, en_passant)
    return " ".join(fields[:6]), label, key


def _source_files(source_dir: Path, output_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise ValueError(f"source directory does not exist: {source_dir}")
    raw_dir = source_dir / "raw"
    root = raw_dir if raw_dir.is_dir() else source_dir
    files = sorted(
        path for path in root.rglob("*.txt")
        if path.is_file()
        and not any(parent.name in EXCLUDED_DIRS for parent in path.parents if parent != root)
        and not path.is_relative_to(output_dir)
    )
    if not files:
        raise ValueError(f"no .txt source files in {root}")
    return files


def clean_dataset(source_dir: str | Path = "dataset", output_dir: str | Path | None = None) -> dict:
    """Clean all source .txt files and write cleaned.txt and report.json.

    Traversal is sorted by path, then physical line. The first valid record
    supplies the output FEN. Later identical inputs with the same class are
    duplicates; any differing class excludes the entire input group.
    """
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else source_dir / "processed"
    files = _source_files(source_dir, output_dir)
    positions: dict[tuple[str, str, str], Position] = {}
    rejected: list[dict[str, object]] = []
    source_counts: dict[str, dict[str, int]] = {}
    records = 0

    for path in files:
        source = path.relative_to(source_dir).as_posix()
        source_counts[source] = {"records": 0, "retained": 0, "rejected": 0}
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                records += 1
                source_counts[source]["records"] += 1
                reference = {"file": source, "line": line_number}
                try:
                    fen, label, key = parse_labeled_fen(line)
                except ValueError as error:
                    rejected.append({**reference, "reason": str(error)})
                    source_counts[source]["rejected"] += 1
                    continue
                reference["label"] = label
                existing = positions.get(key)
                if existing is None:
                    positions[key] = Position(fen, label, [reference])
                else:
                    existing.sources.append(reference)
                    if existing.label != label:
                        existing.conflicting = True

    accepted = [position for position in positions.values() if not position.conflicting]
    conflicts = [
        {"sources": position.sources}
        for position in positions.values() if position.conflicting
    ]
    classes = Counter(position.label for position in accepted)
    duplicates = sum(len(position.sources) - 1 for position in accepted)
    for position in accepted:
        for source in {str(ref["file"]) for ref in position.sources}:
            source_counts[source]["retained"] += 1
    report = {
        "files": [path.relative_to(source_dir).as_posix() for path in files],
        "records": records,
        "accepted": len(accepted),
        "by_class": {label: classes[label] for label in ("Nothing", "Check", "Checkmate")},
        "by_source": source_counts,
        "duplicates": duplicates,
        "conflicting_records": sum(len(group["sources"]) for group in conflicts),
        "rejected": rejected,
        "conflicts": conflicts,
    }
    if not accepted:
        raise ValueError("no clean positions remain")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cleaned.txt").open("w", encoding="utf-8") as stream:
        for position in accepted:
            stream.write(f"{position.fen} {position.label}\n")
    with (output_dir / "provenance.jsonl").open("w", encoding="utf-8") as stream:
        for position in accepted:
            sources = [{"file": ref["file"], "line": ref["line"]} for ref in position.sources]
            stream.write(json.dumps(sources, separators=(",", ":")) + "\n")
    with (output_dir / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return report
