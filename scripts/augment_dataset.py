"""Build a separate, deterministic color-mirrored training split.

Held-out files supply FEN inputs only. Their labels are never parsed. The key
contains exactly the information in encodeFEN's 777 values: 64 piece identities,
side to move and en-passant file (no castling rights or move counters).
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.clean_dataset import parse_labeled_fen


CLASSES = ("Nothing", "Check", "Checkmate")


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size}


def fen_fields(fen: str) -> list[str]:
    # Reuse the loader-compatible validator with a synthetic label; this never
    # consults a held-out label, even when those labels are absent or malformed.
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError("expected exactly six FEN fields")
    parse_labeled_fen(" ".join(fields) + " Nothing")
    return fields


def input_key(fen: str) -> bytes:
    """An injective compact representation of the actual 777 binary inputs."""
    board, turn, _, ep, _, _ = fen_fields(fen)
    squares = "".join("." * int(c) if c.isdigit() else c
                      for c in board if c != "/")
    return squares.encode("ascii") + turn.encode("ascii") + bytes(
        [0 if ep == "-" else ord(ep[0]) - ord("a") + 1])


def mirror_fen(fen: str) -> str:
    """Flip ranks (a1 <-> a8), swap colors and turn; preserve both counters."""
    board, turn, castling, ep, halfmove, fullmove = fen_fields(fen)
    board = "/".join(rank.swapcase() for rank in reversed(board.split("/")))
    # Emit standard FEN order, also accepted by external chess libraries.
    swapped = castling.swapcase()
    castling = "".join(right for right in "KQkq" if right in swapped) or "-"
    ep = "-" if ep == "-" else ep[0] + str(9 - int(ep[1]))
    return " ".join((board, "b" if turn == "w" else "w", castling,
                     ep, halfmove, fullmove))


def canonical_row(row: str) -> str:
    """Normalize whitespace and castling order for a semantic round trip."""
    fields = row.split()
    fields[2] = "".join(right for right in "KQkq" if right in fields[2]) or "-"
    return " ".join(fields)


def mirror_row(row: str) -> str:
    fen, label, _ = parse_labeled_fen(row)
    fields = row.split()
    suffix = "" if len(fields) == 7 else " " + {
        "White": "Black", "Black": "White"}[fields[7]]
    return mirror_fen(fen) + " " + label + suffix


def read_rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, line.strip()


def heldout_keys(path: Path) -> tuple[set[bytes], int]:
    keys = set()
    rows = 0
    for line_number, row in read_rows(path):
        # maxsplit keeps the arbitrary label tail opaque and unused.
        fields = row.split(maxsplit=6)
        if len(fields) < 6:
            raise ValueError(f"{path}:{line_number}: missing FEN fields")
        keys.add(input_key(" ".join(fields[:6])))
        rows += 1
    return keys, rows


def class_counts(counter: Counter) -> dict:
    return {label: counter[label] for label in CLASSES}


def augment(train: Path, validation: Path, test: Path, output: Path) -> dict:
    """Keep first original rows, then first admissible mirrors, in source order.

    Same-input original class conflicts are fatal. Mirrors conflicting with an
    original are excluded; mirror-only class conflicts exclude the entire added
    group. Neither rule reads or uses held-out labels. Existing original overlap
    with held-out inputs is fatal rather than silently changing the baseline.
    """
    sources = {name: fingerprint(path) for name, path in
               (("train", train), ("validation", validation), ("test", test))}
    if output.exists():
        raise ValueError(f"{output}: output already exists")
    # Reject destinations inside the source split directory, including symlinks.
    if any(output.resolve().is_relative_to(p.resolve().parent)
           for p in (train, validation, test)):
        raise ValueError("output must be separate from the source split directory")
    protected = {}
    heldout = {}
    for name, path in (("validation", validation), ("test", test)):
        keys, rows = heldout_keys(path)
        protected[name] = keys
        heldout[name] = {"rows": rows, "unique_encoded_inputs": len(keys)}
    originals = {}
    original_counts = Counter()
    original_rows = original_duplicates = 0
    for line_number, row in read_rows(train):
        fen, label, _ = parse_labeled_fen(row)
        key = input_key(fen)
        original_rows += 1
        if key in originals:
            if originals[key][1] != label:
                raise ValueError(f"{train}:{line_number}: conflicting original classes")
            original_duplicates += 1
            continue
        if any(key in keys for keys in protected.values()):
            raise ValueError(f"{train}:{line_number}: original overlaps held-out inputs")
        originals[key] = (row, label, line_number)
        original_counts[label] += 1
    if not originals:
        raise ValueError("empty training split")

    additions = {}
    rejected = Counter()
    candidate_counts = Counter()
    candidate_class_conflicts = set()
    overlap_both = 0
    involutions = 0
    for original_key, (row, label, line_number) in originals.items():
        mirrored = mirror_row(row)
        if mirror_row(mirrored) != canonical_row(row):
            raise ValueError(f"{train}:{line_number}: mirror is not an involution")
        involutions += 1
        key = input_key(" ".join(mirrored.split()[:6]))
        if key == original_key:
            raise ValueError("mirror must invert the encoded side to move")
        candidate_counts[label] += 1
        in_validation = key in protected["validation"]
        in_test = key in protected["test"]
        if in_validation or in_test:
            rejected["validation_overlap" if in_validation else "test_overlap"] += 1
            overlap_both += in_validation and in_test
        elif key in originals:
            rejected["original_duplicate"] += 1
            if originals[key][1] != label:
                rejected["original_duplicate_class_conflict"] += 1
        elif key in additions:
            rejected["mirror_duplicate"] += 1
            if additions[key][1] != label:
                candidate_class_conflicts.add(key)
        else:
            additions[key] = (mirrored, label, line_number)
    for key in candidate_class_conflicts:
        del additions[key]
    # A bijection on model-visible inputs makes mirror-only duplicates impossible
    # after original deduplication; retain explicit guards and accounting anyway.
    added_counts = Counter(label for _, label, _ in additions.values())
    output.mkdir(parents=True)
    train_out = output / "train.fen"
    added_out = output / "additions.fen"
    provenance = output / "additions-source-lines.csv"
    with train_out.open("x", encoding="utf-8") as combined:
        for row, _, _ in originals.values():
            combined.write(row + "\n")
        with added_out.open("x", encoding="utf-8") as added, provenance.open("x") as refs:
            refs.write("augmented_line,source_train_line\n")
            for n, (row, _, source_line) in enumerate(additions.values(), len(originals) + 1):
                combined.write(row + "\n")
                added.write(row + "\n")
                refs.write(f"{n},{source_line}\n")
    # Re-read and independently enforce the output contract before returning it.
    output_keys = set()
    for line_number, row in read_rows(train_out):
        fen, _, _ = parse_labeled_fen(row)
        key = input_key(fen)
        if key in output_keys or any(key in keys for keys in protected.values()):
            raise ValueError(f"output verification failed at line {line_number}")
        output_keys.add(key)
    for name, path in (("train", train), ("validation", validation), ("test", test)):
        if fingerprint(path) != sources[name]:
            raise ValueError(f"source changed during augmentation: {path}")
    report = {
        "version": 1, "sources": sources,
        "transform": "rank reflection + piece/color/turn/castling/ep swap; unchanged class and counters; swapped optional White/Black",
        "key": "64 piece identities in FEN order + w/b + ep file (0 absent, 1..8); equivalent to all 777 encodeFEN values",
        "heldout_policy": "FEN inputs only, no held-out labels parsed or used; validation wins overlap accounting when both",
        "heldout": heldout,
        "counts": {"source_train_rows": original_rows, "original_duplicates_removed": original_duplicates,
                   "original_unique": len(originals), "original_by_class": class_counts(original_counts),
                   "mirror_candidates": sum(candidate_counts.values()),
                   "candidate_by_class": class_counts(candidate_counts),
                   "excluded_validation_overlap": rejected["validation_overlap"],
                   "excluded_test_overlap": rejected["test_overlap"],
                   "overlap_both_heldouts": overlap_both,
                   "excluded_original_duplicate": rejected["original_duplicate"],
                   "of_which_class_conflict": rejected["original_duplicate_class_conflict"],
                   "excluded_mirror_duplicate": rejected["mirror_duplicate"],
                   "excluded_mirror_conflict_groups": len(candidate_class_conflicts),
                   "added": len(additions), "added_by_class": class_counts(added_counts),
                   "output_rows": len(output_keys),
                   "output_by_class": class_counts(original_counts + added_counts)},
        "checks": {"involutions": involutions, "output_unique_encoded_inputs": len(output_keys),
                   "output_validation_overlap": 0, "output_test_overlap": 0,
                   "source_hashes_unchanged": True},
        "artifacts": {name: fingerprint(path) for name, path in
                      (("train", train_out), ("additions", added_out), ("provenance", provenance))},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = augment(*(args.splits / f"{name}.txt" for name in ("train", "validation", "test")),
                     args.output)
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
