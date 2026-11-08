"""Compare three SGD continuations of the strongest color-mirror checkpoint.

Uses the original campaign's frozen analyzer; does not compile or modify it.
Stops after the three comparisons. Never reads the test split.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from scripts.evaluate import fingerprint, predict_metrics


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="frozen color-mirror campaign")
    parser.add_argument("--output", type=Path, required=True, help="new results directory")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "snapshot"
    frozen.mkdir()
    # Select only among the three original augmented checkpoints, before training.
    candidates = []
    for seed in (44, 45, 46):
        run = source / "runs" / f"seed-{seed}"
        metrics = json.loads((run / "candidate-metrics.json").read_text())
        candidates.append((metrics["macro_f1"], seed, run, metrics))
    _, seed, original, recorded = max(candidates, key=lambda item: (item[0], -item[1]))
    inputs = {}
    for name, path in {
        "model.nn": original / "model.nn",
        "analyzer": source / "tools" / "my_torch_analyzer",
        "train.fen": source / "data" / "train.fen",
        "validation.fen": source / "data" / "validation.fen",
        "recorded-metrics.json": original / "candidate-metrics.json",
        "previous-history.csv": original / "history.csv",
        "previous-config.json": original / "config.json",
    }.items():
        before = fingerprint(path)
        destination = frozen / name
        shutil.copy2(path, destination)
        assert fingerprint(destination)["sha256"] == before["sha256"], name
        destination.chmod(0o555 if name == "analyzer" else 0o444)
        inputs[name] = {"source": before, "frozen": fingerprint(destination)}
    original_config = json.loads((frozen / "previous-config.json").read_text())
    for name, role in (("model.nn", "trained_model"), ("train.fen", "train"),
                       ("validation.fen", "validation"), ("analyzer", "analyzer")):
        # Campaign records model hash in artifacts or outputs; inference below also
        # checks the exact historical confusion matrix, independently of metadata.
        expected = original_config.get("inputs", {}).get(role)
        if expected:
            assert inputs[name]["frozen"]["sha256"] == expected["sha256"], name
    model = json.loads((frozen / "model.nn").read_text())
    assert model["config"]["seed"] == seed
    assert model["config"]["layers"] == [777, 128, 64, 3]
    assert model["config"]["hidden_activation"] == "sigmoid"
    assert model["config"]["output_activation"] == "softmax"
    assert model["config"]["learning_rate"] == 0.08
    assert model["config"].get("optimizer", "sgd") == "sgd"
    for variable in ("OPENBLAS_NUM_THREADS", "OPENBLAS_DEFAULT_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[variable] = "1"
    protocol = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "source_candidates": [
            {"seed": item[1], "macro_f1": item[0]} for item in candidates],
        "inputs": inputs, "rates": [0.08, 0.02, 0.008],
        "settings": {"epochs": 40, "batch_size": 128, "patience": 20, "min_delta": 0.0001},
        "selection": "minimum validation-loss checkpoint per arm, including epoch zero; compare macro F1",
        "concurrency": "Other training may run concurrently; durations are descriptive only.",
        "continuation": "Weights restored; shuffle RNG and patience restart from the same seed in every arm.",
        "scope": "Only three resumed runs; no subsequent training, test evaluation, or model promotion.",
        "runs": [],
    }
    manifest = output / "manifest.json"
    write_json(manifest, protocol)
    try:
        baseline, _, _ = predict_metrics(frozen / "model.nn", frozen / "validation.fen", frozen / "analyzer")
        for key in ("accuracy", "macro_f1", "confusion_matrix", "per_class"):
            assert baseline[key] == recorded[key], f"baseline changed: {key}"
        write_json(output / "baseline-metrics.json", baseline)
        print(f"Verified seed {seed}: baseline macro F1 {baseline['macro_f1']:.8f}", flush=True)
        for rate in protocol["rates"]:
            run = output / f"lr-{str(rate).replace('.', '')}"
            run.mkdir()
            initial = json.loads((frozen / "model.nn").read_text())
            initial["config"]["learning_rate"] = rate
            write_json(run / "initial.nn", initial)
            loaded = json.loads((run / "initial.nn").read_text())
            expected = json.loads((frozen / "model.nn").read_text())
            expected["config"]["learning_rate"] = rate
            assert loaded == expected
            command = [str(frozen / "analyzer"), "--train", "--validation", str(frozen / "validation.fen"),
                       "--save", str(run / "model.nn"), "--history", str(run / "history.csv")]
            for key, value in protocol["settings"].items():
                command += ["--" + key.replace("_", "-"), str(value)]
            command += [str(run / "initial.nn"), str(frozen / "train.fen")]
            entry = {"learning_rate": rate, "directory": str(run), "command": command,
                     "status": "running", "initial": fingerprint(run / "initial.nn")}
            protocol["runs"].append(entry)
            write_json(manifest, protocol)
            print(f"Starting LR {rate}: at most 40 epochs, patience 20", flush=True)
            started = time.perf_counter()
            with (run / "stdout.log").open("w") as stdout, (run / "stderr.log").open("w") as stderr:
                process = subprocess.run(command, cwd=run, stdout=stdout, stderr=stderr, check=False)
            entry["wall_seconds"] = time.perf_counter() - started
            entry["returncode"] = process.returncode
            if process.returncode:
                raise RuntimeError(f"LR {rate} failed; inspect {run / 'stderr.log'}")
            with (run / "history.csv").open() as history:
                rows = list(csv.DictReader(history))
            assert rows and len(rows) <= 40
            stdout = (run / "stdout.log").read_text()
            selected = re.search(r"Training complete: best_epoch=(\d+)", stdout)
            assert selected, "missing completion summary"
            entry["selected_epoch"] = int(selected[1])
            entry["epochs_completed"] = len(rows)
            entry["final_train_loss"] = float(rows[-1]["train_loss"])
            entry["final_validation_loss"] = float(rows[-1]["validation_loss"])
            entry["model"] = fingerprint(run / "model.nn")
            metrics, _, _ = predict_metrics(run / "model.nn", frozen / "validation.fen", frozen / "analyzer")
            write_json(run / "metrics.json", metrics)
            entry["metrics"] = metrics
            entry["status"] = "completed"
            write_json(manifest, protocol)
            print(f"Completed LR {rate}: {len(rows)} epochs; selected {entry['selected_epoch']}; "
                  f"F1 {metrics['macro_f1']:.8f}; wall {entry['wall_seconds']:.1f}s", flush=True)
        for item in inputs.values():
            for location in ("source", "frozen"):
                recorded_file = item[location]
                assert fingerprint(Path(recorded_file["path"]))["sha256"] == recorded_file["sha256"]
        protocol["status"] = "completed"
        protocol["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest, protocol)
        summary = {"protocol": protocol, "baseline": baseline}
        write_json(output / "summary.json", summary)
        print("All three comparisons completed. No further training launched.", flush=True)
    except BaseException as error:
        protocol["status"] = "failed"
        protocol["error"] = str(error)
        write_json(manifest, protocol)
        raise


if __name__ == "__main__":
    main()
