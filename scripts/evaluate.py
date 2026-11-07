"""Evaluate a saved chess model through the C++ analyzer (with chart exports)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSES = ("Nothing", "Check", "Checkmate")


def read_labels(path):
    samples = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if not fields:
                continue
            if (len(fields) not in (7, 8) or fields[6] not in CLASSES
                    or (len(fields) == 8 and
                        (fields[6] == "Nothing" or fields[7] not in ("White", "Black")))):
                raise ValueError(f"{path}:{line_number}: expected six FEN fields and a valid label")
            samples.append((line_number, fields[6]))
    if not samples:
        raise ValueError(f"{path}: dataset contains no positions")
    return samples


def compute_metrics(expected, predicted):
    if not expected or len(expected) != len(predicted):
        raise ValueError("prediction count must match the non-empty labeled dataset")
    matrix = [[0] * len(CLASSES) for _ in CLASSES]
    for actual, prediction in zip(expected, predicted):
        if actual not in CLASSES or prediction not in CLASSES:
            raise ValueError("unknown class in labels or analyzer output")
        matrix[CLASSES.index(actual)][CLASSES.index(prediction)] += 1
    per_class = {}
    for index, label in enumerate(CLASSES):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return {
        "samples": len(expected),
        "accuracy": sum(matrix[i][i] for i in range(len(CLASSES))) / len(expected),
        "macro_f1": sum(values["f1"] for values in per_class.values()) / len(CLASSES),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(CLASSES), "rows": "actual", "columns": "predicted",
                             "counts": matrix},
    }


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def predict_metrics(model, dataset, analyzer):
    """Run inference and return scores plus individual predictions, without exporting files."""
    samples = read_labels(dataset)
    artifacts = {name: fingerprint(path) for name, path in
                 (("model", model), ("dataset", dataset), ("analyzer", analyzer))}
    command = [str(analyzer), "--predict", str(model), str(dataset)]
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise ValueError(f"analyzer exited with {result.returncode}: {result.stderr.strip()}")
    predicted = result.stdout.splitlines()
    metrics = compute_metrics([label for _, label in samples], predicted)
    metrics.update({"artifacts": artifacts, "command": command,
                    "prediction_seconds": elapsed, "analyzer_stderr": result.stderr})
    return metrics, samples, predicted


def evaluation_readme(metrics):
    artifacts = metrics["artifacts"]
    lines = ["# Model evaluation", "",
             f"Model: `{artifacts['model']['path']}`", "",
             f"Dataset: `{artifacts['dataset']['path']}`", "",
             f"Positions: {metrics['samples']:,}", "",
             f"Accuracy: **{metrics['accuracy']:.2%}**; macro F1: **{metrics['macro_f1']:.2%}**.", "",
             "| Class | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: |"]
    for label, values in metrics["per_class"].items():
        lines.append(f"| {label} | {values['precision']:.2%} | {values['recall']:.2%} | {values['f1']:.2%} |")
    lines += ["", "Open `notebooks/evaluation.ipynb` to inspect these scores and the confusion matrix.",
              "`metrics.json` records detailed scores and artifact fingerprints.", ""]
    return "\n".join(lines)


def evaluate(model, dataset, analyzer, output, predictions=False):
    output = Path(output)
    if output.exists():
        raise ValueError(f"{output}: output already exists; choose a new run directory")
    metrics, samples, predicted = predict_metrics(model, dataset, analyzer)
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(evaluation_readme(metrics), encoding="utf-8")
    if predictions:
        with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(["dataset_line", "actual", "predicted"])
            writer.writerows((line, label, prediction)
                             for (line, label), prediction in zip(samples, predicted))
    from scripts.charts import render_directory
    render_directory(output)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="saved .nn model")
    parser.add_argument("dataset", type=Path, help="labeled FEN dataset (use validation while tuning)")
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--analyzer", type=Path, default=ROOT / "my_torch_analyzer")
    parser.add_argument("--predictions", action="store_true", help="also export individual predictions")
    args = parser.parse_args()
    try:
        metrics = evaluate(args.model.resolve(), args.dataset.resolve(),
                           args.analyzer.resolve(), args.output.resolve(), predictions=args.predictions)
    except (OSError, ValueError) as error:
        print(f"Evaluation: {error}", file=sys.stderr)
        return 1
    print(f"Positions: {metrics['samples']} | Accuracy: {metrics['accuracy']:.4f}"
          f" | Macro F1: {metrics['macro_f1']:.4f}")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for label, values in metrics["per_class"].items():
        print(f"{label:<12} {values['precision']:>10.4f} {values['recall']:>10.4f}"
              f" {values['f1']:>10.4f} {values['support']:>10}")
    print("Confusion matrix (rows: actual; columns: Nothing, Check, Checkmate):")
    for label, row in zip(CLASSES, metrics["confusion_matrix"]["counts"]):
        print(f"{label:<12} " + " ".join(f"{count:>10}" for count in row))
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
