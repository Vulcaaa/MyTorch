"""Export charts from recorded results only; never train or run predictions."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    return plt


def save(fig, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        fig.savefig(temporary, format="svg", bbox_inches="tight")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        pyplot().close(fig)
    return path


def loss_chart(history_path, stopping=None, status="recorded", initial_loss=None, offset=0):
    from scripts.comparison import load_history
    rows = load_history(history_path)
    if not rows:
        return None
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    epochs = [row["epoch"] + offset for row in rows]
    for key, label in (("train_loss", "Train"), ("validation_loss", "Validation")):
        ax.plot(epochs, [row[key] for row in rows], label=label, linewidth=1.8)
    if initial_loss is not None:
        ax.scatter([offset], [initial_loss], color="tab:orange", label="Initial validation", zorder=3)
    if stopping and "best_epoch" in stopping:
        best = stopping["best_epoch"] + offset
        ax.axvline(best, color="tab:green", linestyle="--", label=f"Selected checkpoint: epoch {best}")
    else:
        best = min(rows, key=lambda row: row["validation_loss"])["epoch"] + offset
        ax.axvline(best, color="tab:green", linestyle="--", label=f"Lowest recorded validation: epoch {best}")
    # The last recorded epoch does not imply successful training or a saved model.
    end_label = "Stopped" if status == "completed" else "Last recorded"
    ax.axvline(epochs[-1], color="tab:gray", linestyle=":", label=f"{end_label}: epoch {epochs[-1]}")
    ax.set(xlabel="Epoch", ylabel="Mean loss", title=f"Training and validation loss — {status}")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    return save(fig, Path(history_path).with_suffix(".svg"))


def confusion_chart(metrics, path, title="Confusion matrix"):
    plt = pyplot()
    matrix = metrics["confusion_matrix"]
    labels, counts = matrix["labels"], matrix["counts"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    picture = ax.imshow(counts, cmap="Blues", vmin=0)
    maximum = max(max(row) for row in counts)
    for i, row in enumerate(counts):
        total = sum(row)
        for j, count in enumerate(row):
            share = f"{count / total:.1%}" if total else "—"
            ax.text(j, i, f"{count:,}\n{share}", ha="center", va="center",
                    color="white" if maximum and count > maximum / 2 else "black")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
           yticklabels=labels, xlabel="Predicted class", ylabel="Actual class",
           title=f"{title}\nCounts and percentage within each actual class")
    fig.colorbar(picture, ax=ax, label="Positions", shrink=0.8)
    return save(fig, path)


def comparison_chart(output, manifest, entries=None, note=None):
    """Show individual completed trials; leave failed/missing trials out of rankings."""
    entries = list(entries or [])
    for run in manifest.get("runs", []):
        metrics_path = output / run["directory"] / "metrics.json"
        if run["status"] != "completed" or not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())["validation"]
        if run.get("training_seconds") is None:
            continue
        entries.append((f"{run['configuration']} / seed {run['seed']}",
                        100 * metrics["macro_f1"], run["training_seconds"]))
    if len(entries) < 2:
        return None
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(entries) * 0.4)), layout="constrained")
    for ax, column, title in ((axes[0], 1, "Validation macro F1 (%)"),
                              (axes[1], 2, "Training time (seconds)")):
        bars = ax.barh([row[0] for row in entries], [row[column] for row in entries], color="tab:blue")
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.margins(x=0.18)
    axes[0].set_xlim(0, 110)
    fig.suptitle(f"Recorded trials — campaign status: {manifest['status']}")
    if note:
        fig.supxlabel(note, fontsize=9)
    return save(fig, output / "comparison.svg")


def render_run(directory, run=None, status="recorded", metadata=None):
    directory = Path(directory)
    run = run or {}
    metadata = metadata or {}
    generated = []
    history = directory / "history.csv"
    if history.exists() and len(history.read_text().splitlines()) > 1:
        stopping = run.get("stopping")
        if stopping is None and "selected_checkpoint_epoch" in metadata:
            stopping = {"best_epoch": metadata["selected_checkpoint_epoch"]}
        chart = loss_chart(history, stopping, status,
                           (stopping or {}).get("initial_validation_loss"),
                           run.get("previous_checkpoint_epoch", 0))
        if chart:
            generated.append(chart)
    metrics_path = directory / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        phases = {"": metrics} if "confusion_matrix" in metrics else metrics
        for phase, scores in phases.items():
            if not isinstance(scores, dict) or "confusion_matrix" not in scores:
                continue
            suffix = f"-{phase}" if phase else ""
            generated.append(confusion_chart(scores, directory / f"confusion-matrix{suffix}.svg",
                                             f"Confusion matrix{': ' + phase if phase else ''}"))
        if generated and "confusion_matrix" in metrics:
            (directory / "confusion-matrix.csv").unlink(missing_ok=True)
    return generated


def render_directory(directory):
    directory = Path(directory).resolve()
    config = directory / "config.json"
    metadata = json.loads(config.read_text()) if config.exists() else {}
    # Do not read or alter the outputs of an active campaign during backfill.
    if metadata.get("status") in {"preparing", "training", "generating", "evaluating"}:
        print(f"Skipped active or unfinished manifest: {directory}", file=sys.stderr)
        return []
    generated = []
    if "runs" in metadata:
        for run in metadata["runs"]:
            generated += render_run(directory / run["directory"], run, run["status"])
        chart = comparison_chart(directory, metadata)
        if chart:
            generated.append(chart)
    else:
        generated += render_run(directory, status=metadata.get("status", "recorded"), metadata=metadata)
    # Preserve hand-written readmes and conclusions; maintain one replaceable chart section.
    readme = directory / "README.md"
    if generated and readme.exists():
        text = readme.read_text()
        start, end = "<!-- charts:start -->", "<!-- charts:end -->"
        if start in text and end in text:
            before, rest = text.split(start, 1)
            text = before + rest.split(end, 1)[1]
        text = text.replace("- Notebooks display curves and confusion matrices; no HTML or image exports.",
                            "- SVG charts are exported from the saved histories and metrics.")
        text = text.replace("confusion-matrix.csv", "confusion-matrix.svg")
        links = "\n\n" + start + "\n## Charts\n\n" + "\n\n".join(
            f"![{p.stem}]({p.relative_to(directory).as_posix()})" for p in generated) + "\n" + end + "\n"
        readme.write_text(text.rstrip() + links)
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--recursive", action="store_true", help="find saved runs below this directory")
    args = parser.parse_args()
    directories = {args.directory.resolve()}
    if args.recursive:
        directories |= {p.parent.resolve() for pattern in ("config.json", "metrics.json", "history.csv")
                        for p in args.directory.rglob(pattern)}
        # A campaign handles its own nested runs.
        campaigns = {p for p in directories if (p / "config.json").exists()
                     and "runs" in json.loads((p / "config.json").read_text())}
        directories = {p for p in directories if not any(c != p and c in p.parents for c in campaigns)}
    count = 0
    for directory in sorted(directories):
        try:
            for path in render_directory(directory):
                print(path)
                count += 1
        except (OSError, ValueError, KeyError) as error:
            print(f"Charts: {directory}: {error}", file=sys.stderr)
            return 1
    print(f"Generated {count} charts; no training or prediction executed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
