"""Load comparable validation reports without running inference."""

import json
import math
from pathlib import Path

from scripts.evaluate import CLASSES, fingerprint


def load_comparison(run_directories, validation_dataset):
    """Require the declared validation data and a common analyzer build.

    Rank individual runs by macro F1, then worst-class recall. This does not
    aggregate training seeds or select a final model.
    """
    directories = [Path(path).resolve() for path in run_directories]
    if not directories:
        raise ValueError("Select at least one validation run")
    if len(set(directories)) != len(directories):
        raise ValueError("Each run directory must be selected only once")
    dataset_hash = fingerprint(Path(validation_dataset))["sha256"]
    reports = []
    reference = None
    for directory in directories:
        try:
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            matrix = metrics["confusion_matrix"]
            if (matrix["labels"] != list(CLASSES) or matrix["rows"] != "actual"
                    or matrix["columns"] != "predicted"):
                raise ValueError("incompatible confusion matrix labels or axes")
            counts = matrix["counts"]
            if (len(counts) != 3 or any(len(row) != 3 for row in counts)
                    or any(type(value) is not int or value < 0 for row in counts for value in row)):
                raise ValueError("invalid confusion counts")
            supports = tuple(sum(row) for row in counts)
            if not all(supports):
                raise ValueError("validation must contain all three classes")
            if (sum(supports) != metrics["samples"] or
                    supports != tuple(metrics["per_class"][label]["support"] for label in CLASSES)):
                raise ValueError("inconsistent sample counts")
            scores = [metrics["accuracy"], metrics["macro_f1"]]
            scores += [metrics["per_class"][label][key] for label in CLASSES
                       for key in ("precision", "recall", "f1")]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
                raise ValueError("invalid classification scores")
            expected_f1 = []
            for index, label in enumerate(CLASSES):
                correct = counts[index][index]
                predicted = sum(row[index] for row in counts)
                precision = correct / predicted if predicted else 0.0
                recall = correct / supports[index]
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                expected_f1.append(f1)
                if any(not math.isclose(metrics["per_class"][label][key], value, abs_tol=1e-12)
                       for key, value in (("precision", precision), ("recall", recall), ("f1", f1))):
                    raise ValueError("scores disagree with confusion counts")
            accuracy = sum(counts[i][i] for i in range(3)) / sum(supports)
            if (not math.isclose(metrics["accuracy"], accuracy, abs_tol=1e-12) or
                    not math.isclose(metrics["macro_f1"], sum(expected_f1) / 3, abs_tol=1e-12)):
                raise ValueError("scores disagree with confusion counts")
            elapsed = metrics["prediction_seconds"]
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid prediction time")
            artifacts = metrics["artifacts"]
            if artifacts["dataset"]["sha256"] != dataset_hash:
                raise ValueError("dataset differs from the declared validation file; do not compare test runs")
            build = artifacts["analyzer"]["sha256"]
            if (not isinstance(build, str) or len(build) != 64 or
                    any(character not in "0123456789abcdef" for character in build)):
                raise ValueError("invalid analyzer fingerprint")
            identity = (build, supports)
            if reference is not None and reference != identity:
                raise ValueError("runs must use the same analyzer build and class supports")
            reference = identity
            reports.append({"run": directory.name, "directory": str(directory), "metrics": metrics,
                            "worst_recall": min(metrics["per_class"][label]["recall"] for label in CLASSES)})
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{directory.name}: {error}") from error
    return sorted(reports, key=lambda report: (-report["metrics"]["macro_f1"],
                                              -report["worst_recall"], report["directory"]))
