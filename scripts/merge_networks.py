"""Merge trained networks, optionally with their mirror views, into one network.

Each member keeps its own hidden neurons: hidden layers are placed side by side
with block-diagonal weights, and the output layer averages the members' logits.
The result is an ordinary MyTorch model that the analyzer loads unchanged.

With --mirror-views, every member is added four times: as trained, and reading
the left-right mirror, the color mirror and both mirrors of the position. A view
only permutes first-layer input columns (and flips the side-to-move input), so
the merged network averages the predictions on all four symmetric positions.
--views selects a subset, for example `--views original,color`.

Usage:
    python3 scripts/merge_networks.py OUTPUT MODEL [MODEL ...] [--mirror-views | --views LIST]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SQUARES, PIECES, TURN, EN_PASSANT = 64, 12, 768, 769
VIEWS = {"original": (False, False), "file": (True, False),
         "color": (False, True), "both": (True, True)}


def view_index(index: int, file_flip: bool, color_flip: bool) -> int:
    """Input read by the mirrored view for input `index` of the original position."""
    if index < TURN:
        square, piece = divmod(index, PIECES)
        rank, file = divmod(square, 8)
        if file_flip:
            file = 7 - file
        if color_flip:
            rank, piece = 7 - rank, (piece + 6) % PIECES
        return (rank * 8 + file) * PIECES + piece
    if index == TURN:
        return TURN
    file = index - EN_PASSANT
    return EN_PASSANT + (7 - file if file_flip else file)


def mirror_view(model: dict, file_flip: bool, color_flip: bool) -> dict:
    view = json.loads(json.dumps(model))
    first = view["layers"][0]
    for row_index, row in enumerate(first["weights"]):
        new = [row[view_index(j, file_flip, color_flip)] for j in range(len(row))]
        if color_flip:  # the color mirror also swaps the side to move: x -> 1 - x
            first["biases"][row_index] += new[TURN]
            new[TURN] = -new[TURN]
        first["weights"][row_index] = new
    return view


def merge(models: list[dict], digits: int) -> dict:
    reference = models[0]["config"]
    depth = len(models[0]["layers"])
    for model in models:
        config = model["config"]
        if (config["hidden_activation"] != reference["hidden_activation"]
                or config["output_activation"] != "softmax" or len(model["layers"]) != depth
                or config["layers"][0] != reference["layers"][0]
                or config["layers"][-1] != reference["layers"][-1]):
            raise SystemExit("members need the same depth, input, output and activations")
    count = len(models)
    layers = []
    for i in range(depth):
        weights, biases = [], []
        if i == depth - 1:
            for c in range(len(models[0]["layers"][i]["biases"])):
                weights.append([w / count for m in models for w in m["layers"][i]["weights"][c]])
                biases.append(sum(m["layers"][i]["biases"][c] for m in models) / count)
        else:
            widths = [len(m["layers"][i - 1]["biases"]) for m in models] if i else []
            total, offset = sum(widths), 0
            for k, m in enumerate(models):
                for row, bias in zip(m["layers"][i]["weights"], m["layers"][i]["biases"]):
                    weights.append(row if i == 0 else
                                   [0.0] * offset + row + [0.0] * (total - offset - widths[k]))
                    biases.append(bias)
                offset += widths[k] if i else 0
        layers.append({"weights": weights, "biases": biases})

    def rounded(value: float) -> float:
        return float(f"{value:.{digits}g}")

    layers = [{"weights": [[rounded(w) for w in row] for row in layer["weights"]],
               "biases": [rounded(b) for b in layer["biases"]]} for layer in layers]
    sizes = [len(layers[0]["weights"][0])] + [len(layer["biases"]) for layer in layers]
    config = dict(reference, layers=sizes, optimizer="sgd")
    return {"format": "mytorch", "version": 1, "config": config, "layers": layers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--mirror-views", action="store_true", help="all four views")
    parser.add_argument("--views", default="original",
                        help="comma-separated subset of " + ",".join(VIEWS))
    parser.add_argument("--digits", type=int, default=7, help="significant digits kept")
    args = parser.parse_args()
    views = list(VIEWS) if args.mirror_views else args.views.split(",")
    if not views or any(view not in VIEWS for view in views):
        raise SystemExit("--views takes a comma-separated subset of " + ",".join(VIEWS))
    members = []
    for path in args.models:
        model = json.loads(path.read_text())
        model.pop("optimizer_state", None)
        members += [model if view == "original" else mirror_view(model, *VIEWS[view])
                    for view in views]
    merged = merge(members, args.digits)
    args.output.write_text(json.dumps(merged, separators=(",", ":")))
    print(args.output, merged["config"]["layers"])


if __name__ == "__main__":
    main()
