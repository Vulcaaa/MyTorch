"""Rebuild SVG figures from the portable generated-data experiment results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/report/2026-11-08-generated-60k/data.json"
OUTPUT = ROOT / "docs/report/2026-11-08-generated-60k"


def draw(data):
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    models = data["models"]
    colors = ["#64748b", "#e28c32", "#138f83", "#7356b5"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, layout="constrained")
    for ax, metric, title in zip(axes, ("macro_f1", "accuracy"), ("Macro F1", "Accuracy")):
        values = [100 * m["metrics"][metric] for m in models]
        ax.barh(range(len(models)), values, color=colors, height=0.58)
        ax.set_xlim(0, 103)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Validation score (%)")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.text(value + 1, i, f"{value:.2f}", va="center", fontsize=10)
    axes[0].set_yticks(range(len(models)),
                      [m["name"] + f"\nseed {m['seed']}" for m in models])
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False)
    fig.suptitle("Generated positions: validation comparison\n"
                 "71,555 shared positions · historical seeds and architectures differ", fontsize=12)

    loss, panels = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, layout="constrained")
    for ax, model in zip(panels, [m for m in models if "history" in m]):
        history = model["history"]
        epochs = [r["epoch"] for r in history]
        ax.plot(epochs, [r["train_loss"] for r in history], color="#64748b", label="Train", lw=1.3)
        ax.plot(epochs, [r["validation_loss"] for r in history], color="#138f83", label="Validation", lw=1.4)
        selected = model["selected_epoch"]
        best = next(r for r in history if r["epoch"] == selected)["validation_loss"]
        ax.axvline(selected, color="#7356b5", ls="--", lw=1, alpha=0.8)
        ax.scatter([selected], [best], color="#7356b5", zorder=5, s=22)
        ax.set_title(f"{model['name']}\n{model['training_seconds']/60:.1f} min · selected epoch {selected}", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.15, 1.15)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False)
    panels[0].set_ylabel("Cross-entropy loss")
    loss.suptitle("New seed-43 training runs\nEpochs contain different amounts of training work", fontsize=12)
    return {"validation-comparison.svg": fig, "training-curves.svg": loss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, fig in draw(json.loads(args.data.read_text())).items():
        fig.savefig(args.output / name)
        plt.close(fig)


if __name__ == "__main__":
    main()
