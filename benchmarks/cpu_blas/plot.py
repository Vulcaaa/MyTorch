#!/usr/bin/env python3
"""Render the frozen timing summary for documentation and the defense."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.evidence / "summary.json").read_text())
    manifest = json.loads((args.evidence / "manifest.json").read_text())
    if not summary["agreement_passed"]:
        raise ValueError("numerical agreement must pass before plotting speed comparisons")
    rows = summary["results"]
    epochs = manifest["settings"]["epochs"]
    labels = [("BLAS / " + str(r["threads"]) + (" thread" if r["threads"] == 1 else " threads")) if r["backend"] == "candidate-blas"
              else ("Baseline / scalar" if r["backend"] == "baseline" else "Current / scalar")
              for r in rows]
    medians = [r["seconds_per_epoch"] for r in rows]
    lower = [m - r["min_seconds"] / epochs for m, r in zip(medians, rows)]
    upper = [r["max_seconds"] / epochs - m for m, r in zip(medians, rows)]
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    bars = ax.barh(labels, medians, xerr=[lower, upper], capsize=4,
                   color=["#2563eb" if r["backend"] == "candidate-blas" else "#64748b" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Seconds per epoch, including train + validation loss (lower is better)")
    ax.set_title("MyTorch CPU training: matched FP64 / batch 32", loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(r["max_seconds"] / epochs for r in rows) * 1.35)
    for bar, row in zip(bars, rows):
        ax.text(bar.get_width() + max(medians) * 0.025, bar.get_y() + bar.get_height() / 2,
                f"{row['seconds_per_epoch']:.3f}s  ({row['speedup']:.2f}x)", va="center", fontsize=9)
    n = manifest["fixtures"]["train"]["rows"]
    v = manifest["fixtures"]["validation"]["rows"]
    fig.supxlabel(f"{n:,} train / {v:,} validation positions; {epochs} "
                  f"{'epoch' if epochs == 1 else 'epochs'} per run. "
                  "Bars: medians; whiskers: observed min/max.\n"
                  "Fixed-epoch throughput; time to target validation quality is unmeasured.", fontsize=9)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "timings.png", dpi=200)
    svg = args.output / "timings.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
