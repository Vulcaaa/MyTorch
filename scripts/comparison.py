"""Load comparable validation reports without running inference."""

import csv
import json
import math
from pathlib import Path
import statistics

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


def describe(values):
    """Keep singleton dispersion unknown, rather than claiming zero variance."""
    return {"mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None}


def load_history(path):
    """Optional backend log contract: epoch,train_loss,validation_loss."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["epoch", "train_loss", "validation_loss"]:
            raise ValueError(f"{path}: expected epoch,train_loss,validation_loss")
        rows = []
        for row in reader:
            try:
                epoch = int(row["epoch"])
                train, validation = float(row["train_loss"]), float(row["validation_loss"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: invalid epoch or loss") from error
            if (epoch <= (rows[-1]["epoch"] if rows else 0) or
                    any(not math.isfinite(value) or value < 0 for value in (train, validation))):
                raise ValueError(f"{path}: invalid epoch or loss")
            rows.append({"epoch": epoch, "train_loss": train, "validation_loss": validation})
    if not rows:
        raise ValueError(f"{path}: history contains no epochs")
    return rows


def load_experiment(directory):
    """Build a fresh summary from a frozen campaign and its completed reports.

    Seeds come only from the experiment plan. Partial groups remain visible but
    cannot establish a leading configuration until every planned run completes.
    """
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("unsupported experiment version")
    plan = manifest["plan"]
    seeds = plan["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("experiment seeds must be non-empty and unique")
    configurations = [config["name"] for config in plan["configurations"]]
    if not configurations or len(set(configurations)) != len(configurations):
        raise ValueError("experiment configurations must be non-empty and unique")
    completed = []
    seen = set()
    for run in manifest["runs"]:
        pair = (run["configuration"], run["seed"])
        if pair in seen or pair[0] not in configurations or pair[1] not in seeds:
            raise ValueError("duplicate or unexpected configuration/seed in experiment")
        seen.add(pair)
        if run["status"] == "completed":
            completed.append(run)
    paths = [directory / run["directory"] / "evaluation" for run in completed]
    reports = load_comparison(paths, plan["datasets"]["validation"]["path"]) if paths else []
    by_path = {report["directory"]: report for report in reports}
    grouped = {name: [] for name in configurations}
    model_hashes = set()
    for run, path in zip(completed, paths):
        report = by_path[str(path.resolve())]
        metrics = report["metrics"]
        metadata = metrics.get("experiment", {})
        if (metadata.get("configuration") != run["configuration"] or metadata.get("seed") != run["seed"] or
                metadata.get("training_seconds") != run["training_seconds"] or metadata.get("model_bytes") != run["model_bytes"]):
            raise ValueError("evaluation metadata differs from the experiment run")
        if (metrics["artifacts"]["dataset"]["sha256"] != plan["datasets"]["validation"]["sha256"] or
                metrics["artifacts"]["analyzer"]["sha256"] != manifest["tools"]["analyzer"]["sha256"]):
            raise ValueError("evaluation artifacts differ from the frozen experiment")
        trained_hash = run["trained_model"]["sha256"]
        if (metrics["artifacts"]["model"]["sha256"] != trained_hash or
                fingerprint(Path(run["trained_model"]["path"]))["sha256"] != trained_hash):
            raise ValueError("trained model differs from the evaluated model")
        if trained_hash in model_hashes:
            raise ValueError("the same model was reused across experiment runs")
        model_hashes.add(trained_hash)
        if (not math.isfinite(run["training_seconds"]) or run["training_seconds"] < 0 or
                type(run["model_bytes"]) is not int or run["model_bytes"] <= 0):
            raise ValueError("invalid training cost")
        report.update(run=f"{run['configuration']}/seed-{run['seed']}",
                      configuration=run["configuration"], seed=run["seed"])
        grouped[run["configuration"]].append(report)
    summaries = []
    for name, entries in grouped.items():
        present = sorted(report["seed"] for report in entries)
        summary = {"configuration": name, "seeds": present,
                   "missing_seeds": [seed for seed in seeds if seed not in present],
                   "complete": set(present) == set(seeds)}
        for key in ("accuracy", "macro_f1", "prediction_seconds"):
            summary[key] = describe([report["metrics"][key] for report in entries])
        summary["worst_recall"] = describe([report["worst_recall"] for report in entries])
        for key in ("training_seconds", "model_bytes"):
            summary[key] = describe([report["metrics"]["experiment"][key] for report in entries])
        summary["per_class"] = {label: {key: describe([report["metrics"]["per_class"][label][key]
                                                      for report in entries])
                                         for key in ("precision", "recall", "f1")} for label in CLASSES}
        summaries.append(summary)
    summaries.sort(key=lambda row: (not row["complete"],
                                   -(row["macro_f1"]["mean"] if row["macro_f1"]["mean"] is not None else -1),
                                   -(row["worst_recall"]["mean"] if row["worst_recall"]["mean"] is not None else -1),
                                   row["configuration"]))
    ready = (manifest["status"] == "completed" and len(seeds) >= 2 and
             all(row["complete"] for row in summaries))
    leaders = []
    if ready:
        first = summaries[0]
        leaders = [row["configuration"] for row in summaries
                   if row["macro_f1"]["mean"] == first["macro_f1"]["mean"] and
                   row["worst_recall"]["mean"] == first["worst_recall"]["mean"]]
    return {"status": manifest["status"], "expected_seeds": seeds, "selection_ready": ready,
            "leading_configurations": leaders, "configurations": summaries, "runs": reports}
