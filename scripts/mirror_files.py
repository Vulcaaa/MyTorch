"""Add left-right (file) mirrors to a training split.

Reflecting every rank (a-file <-> h-file) keeps Check and Checkmate unchanged:
the movement rules are symmetric between files, and castling, the only
asymmetric rule, is never a legal reply to check. Castling rights are dropped
from mirrored rows because the 777 inputs do not encode them. Mirrors whose
encoded inputs already occur in the split, or in a held-out split, are skipped.

Usage:
    python3 scripts/mirror_files.py TRAIN OUTPUT --heldout VALIDATION TEST
"""

from __future__ import annotations

import argparse
from pathlib import Path

FILES = "abcdefgh"


def input_key(fields: list[str]) -> str:
    """Board, side to move and en-passant file: exactly what the model sees."""
    board, turn, _, ep = fields[:4]
    squares = "".join("." * int(c) if c.isdigit() else c for c in board if c != "/")
    return squares + turn + ("-" if ep == "-" else ep[0])


def mirror_rank(rank: str) -> str:
    cells = "".join("." * int(c) if c.isdigit() else c for c in rank)[::-1]
    out, run = "", 0
    for c in cells:
        if c == ".":
            run += 1
            continue
        out += (str(run) if run else "") + c
        run = 0
    return out + (str(run) if run else "")


def mirror_fields(fields: list[str]) -> list[str]:
    board, turn, _, ep, halfmove, fullmove = fields[:6]
    board = "/".join(mirror_rank(rank) for rank in board.split("/"))
    ep = "-" if ep == "-" else FILES[7 - FILES.index(ep[0])] + ep[1]
    return [board, turn, "-", ep, halfmove, fullmove]


def mirror_split(train: Path, output: Path, heldout: list[Path]) -> dict:
    if output.exists():
        raise SystemExit(f"{output}: output already exists")
    protected = {input_key(line.split()) for path in heldout
                 for line in path.open() if line.strip()}
    rows = [line.split() for line in train.open() if line.strip()]
    seen = {input_key(fields) for fields in rows}
    added = []
    skipped_heldout = skipped_duplicate = 0
    for fields in rows:
        mirrored = mirror_fields(fields)
        key = input_key(mirrored)
        if key in protected:
            skipped_heldout += 1
        elif key in seen:
            skipped_duplicate += 1
        else:
            seen.add(key)
            added.append(mirrored + fields[6:])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as out:
        for fields in rows + added:
            out.write(" ".join(fields) + "\n")
    return {"original": len(rows), "added": len(added), "skipped_duplicate": skipped_duplicate,
            "skipped_heldout": skipped_heldout, "total": len(rows) + len(added)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("train", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--heldout", type=Path, nargs="+", required=True,
                        help="validation and test files whose inputs must not enter training")
    args = parser.parse_args()
    print(mirror_split(args.train, args.output, args.heldout))


if __name__ == "__main__":
    main()
