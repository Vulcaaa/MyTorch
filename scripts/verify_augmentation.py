"""Audit a generated augmentation against C++ encodeFEN and python-chess.

python-chess is an independent QA oracle, not a training dependency. This audit
compares equivalent positions, never the dataset's labels against chess rules.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from scripts.augment_dataset import canonical_row, fingerprint, input_key, mirror_fen, mirror_row, read_rows


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(data: Path, oracle_path: Path | None = None) -> dict:
    if oracle_path:
        sys.path.insert(0, str(oracle_path.resolve()))
    chess = importlib.import_module("chess")
    started = time.perf_counter()
    manifest = json.loads((data / "manifest.json").read_text())
    for artifact in [*manifest["sources"].values(), *manifest["artifacts"].values()]:
        require(fingerprint(Path(artifact["path"])) == artifact, "fingerprint mismatch")
    train = Path(manifest["sources"]["train"]["path"])
    oracle_rows = 0
    rule_rows = 0
    # Include explicit castling, en-passant, check and mate positions in rule QA.
    special = [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 20",
        "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 20",
        "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
        "7k/8/5KQ1/8/8/8/8/8 b - - 0 1",
        "4k3/8/8/8/8/8/8/4R1K1 b - - 0 1",
    ]

    def compare(fen, rules=False):
        original = chess.Board(fen)
        actual = chess.Board(mirror_fen(fen))
        expected = original.mirror()
        # Raw castling rights and EP square are compared directly: python-chess
        # FEN serialization may normalize unusable rights or EP opportunities.
        require(actual.board_fen() == expected.board_fen(), "oracle piece mismatch")
        require((actual.turn, actual.castling_rights, actual.ep_square,
                 actual.halfmove_clock, actual.fullmove_number) ==
                (expected.turn, expected.castling_rights, expected.ep_square,
                 expected.halfmove_clock, expected.fullmove_number), "oracle metadata mismatch")
        if rules:
            mapped_moves = {chess.Move(chess.square_mirror(m.from_square),
                                       chess.square_mirror(m.to_square),
                                       promotion=m.promotion).uci()
                            for m in original.legal_moves}
            require(mapped_moves == {m.uci() for m in actual.legal_moves}, "legal-move symmetry mismatch")
            require((original.is_check(), original.is_checkmate(), original.is_stalemate()) ==
                    (actual.is_check(), actual.is_checkmate(), actual.is_stalemate()),
                    "chess-status symmetry mismatch")

    for fen in special:
        compare(fen, rules=True)
        rule_rows += 1
    for n, row in read_rows(train):
        fen = " ".join(row.split()[:6])
        compare(fen, rules=n % 503 == 0)
        rule_rows += n % 503 == 0
        require(mirror_row(mirror_row(row)) == canonical_row(row), "row involution failed")
        oracle_rows += 1
    print(f"python-chess checked {oracle_rows} source mirrors and {rule_rows} rule symmetries", flush=True)

    tools = data.parent / "tools"
    tools.mkdir(exist_ok=True)
    encoder = tools / "augmentation_encoder"
    command = ["g++", "-std=c++17", "-O2", "-Isrc", "tests/fixtures/augmentation_encoder.cpp",
               "src/chess/Encoding.cpp", "-o", str(encoder.resolve())]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
    cpp_rows = {}
    with tempfile.TemporaryDirectory(prefix="mytorch-augmentation-qa-") as td:
        fens = Path(td) / "fens"
        keys = Path(td) / "keys"
        for name, path in [("augmented_train", Path(manifest["artifacts"]["train"]["path"]))] + [
                (n, Path(manifest["sources"][n]["path"])) for n in ("validation", "test")]:
            with fens.open("w") as stream:
                for _, row in read_rows(path):
                    stream.write(" ".join(row.split(maxsplit=6)[:6]) + "\n")
            with fens.open("rb") as source, keys.open("wb") as dest:
                subprocess.run([str(encoder.resolve())], stdin=source, stdout=dest, check=True)
            count = 0
            with keys.open("rb") as encoded:
                for n, row in read_rows(path):
                    key = input_key(" ".join(row.split(maxsplit=6)[:6]))
                    require(key == encoded.read(66), f"C++ key mismatch: {name}:{n}")
                    count += 1
                require(not encoded.read(1), "extra C++ output")
            cpp_rows[name] = count
            print(f"C++ encodeFEN agreement: {name} {count} rows", flush=True)
    for artifact in [*manifest["sources"].values(), *manifest["artifacts"].values()]:
        require(fingerprint(Path(artifact["path"])) == artifact, "fingerprint changed during QA")
    result = {"status": "passed", "python_chess_version": chess.__version__,
              "python_chess_source": fingerprint(Path(chess.__file__)),
              "oracle_mirror_rows": oracle_rows, "legal_move_and_status_pairs": rule_rows,
              "cpp_encoded_rows": cpp_rows, "cpp_compile_command": command,
              "cpp_encoder": fingerprint(encoder),
              "cpp_encoder_source": fingerprint(ROOT / "src/chess/Encoding.cpp"),
              "cpp_probe_source": fingerprint(ROOT / "tests/fixtures/augmentation_encoder.cpp"),
              "elapsed_seconds": time.perf_counter() - started,
              "label_policy": "Only transform/involution checks for training labels; no label adjudication. Held-out labels never parsed."}
    (data / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--oracle-path", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.oracle_path), indent=2))


if __name__ == "__main__":
    main()
