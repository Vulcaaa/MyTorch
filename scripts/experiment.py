"""Prepare or run a reproducible configuration-by-seed experiment."""

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import re
import shutil
import string
import subprocess
import sys
import time
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evaluate import CLASSES, predict_metrics, fingerprint, read_labels
from scripts.comparison import load_experiment, load_history


def training_settings(settings):
    """The built-in trainer applies every declared setting; reject silent typos."""
    expected = {"epochs", "batch_size", "patience", "min_delta"}
    if not isinstance(settings, dict) or set(settings) != expected:
        raise ValueError("builtin training.settings requires epochs, batch_size, patience, min_delta")
    for name in ("epochs", "batch_size", "patience"):
        if type(settings[name]) is not int or not 0 < settings[name] <= 2**32 - 1:
            raise ValueError(f"{name} must be a positive 32-bit integer")
    delta = settings["min_delta"]
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
        raise ValueError("min_delta must be finite and nonnegative")
    return settings


def builtin_command(values, settings):
    command = [values["analyzer"], "--train", "--validation", values["validation"],
               "--save", values["trained_model"], "--history", values["history"]]
    for key, value in training_settings(settings).items():
        command += ["--" + key.replace("_", "-"), str(value)]
    return command + [values["initial_model"], values["train"]]


NETWORK_KEYS = {"layers", "hidden_activation", "output_activation", "learning_rate", "initialization", "seed"}
PLACEHOLDERS = {"root", "analyzer", "initial_model", "trained_model", "train", "validation",
                "seed", "training_config", "history"}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def network_options(path, overrides):
    options = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        key, separator, value = content.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or key not in NETWORK_KEYS or key in options or not value:
            raise ValueError(f"{path}: invalid or duplicate network option {key!r}")
        options[key] = value
    if not isinstance(overrides, dict) or set(overrides) - (NETWORK_KEYS - {"seed"}):
        raise ValueError("network overrides must use known options; seeds belong in the central seed list")
    for key, value in overrides.items():
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError(f"invalid override for {key}")
        value = str(value)
        if not value.strip() or any(character in value for character in "\n\r#"):
            raise ValueError(f"invalid override for {key}")
        options[key] = value
    options.pop("seed", None)  # Experiment seeds override the network file's default.
    if not {"layers", "hidden_activation", "output_activation", "initialization"} <= options.keys():
        raise ValueError(f"{path}: missing required network options")
    sizes = [int(size.strip()) for size in options["layers"].split(",")]
    if sizes[0] != 777 or sizes[-1] != 3:
        raise ValueError("experiment networks require 777 inputs and 3 outputs")
    return options


def load_spec(path):
    path = Path(path).resolve()
    try:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: invalid YAML: {error}") from error
    json.dumps(spec, allow_nan=False)
    required = {"version", "seeds", "datasets", "configurations", "training"}
    if (not isinstance(spec, dict) or not required <= spec.keys() or set(spec) - (required | {"hypothesis"}) or
            type(spec["version"]) is not int or spec["version"] != 1):
        raise ValueError("experiment config must contain version=1, seeds, datasets, configurations, training")
    if "hypothesis" in spec and (not isinstance(spec["hypothesis"], str) or not spec["hypothesis"].strip()):
        raise ValueError("hypothesis must be a non-empty statement")
    seeds = spec["seeds"]
    if (not isinstance(seeds, list) or not seeds or
            any(type(seed) is not int or not 0 <= seed <= 4294967295 for seed in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct unsigned 32-bit integers")
    data = spec["datasets"]
    if not isinstance(data, dict) or not {"train", "validation"} <= data.keys() or set(data) - {"train", "validation", "split_metadata"}:
        raise ValueError("datasets must declare train and validation; test is reserved for final evaluation")
    datasets = {}
    for role, relative in data.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"datasets.{role} must be a path")
        source = (path.parent / relative).resolve()
        datasets[role] = fingerprint(source)
        if role in ("train", "validation"):
            counts = Counter(label for _, label in read_labels(source))
            if any(not counts[label] for label in CLASSES):
                raise ValueError(f"{role} must contain all three classes")
            datasets[role]["class_counts"] = dict(counts)
        else:
            datasets[role]["contents"] = json.loads(source.read_text(encoding="utf-8"))
    if datasets["train"]["sha256"] == datasets["validation"]["sha256"]:
        raise ValueError("train and validation must be distinct prepared splits")
    configurations = []
    if not isinstance(spec["configurations"], list) or not spec["configurations"]:
        raise ValueError("select at least one configuration")
    for configuration in spec["configurations"]:
        if not isinstance(configuration, dict):
            raise ValueError("each configuration must be an object")
        resumed = "initial_model" in configuration
        if ((not resumed and set(configuration) != {"name", "network", "overrides"}) or
                (resumed and (not {"name", "initial_model"} <= configuration.keys() or
                              set(configuration) - {"name", "initial_model", "previous_history", "previous_checkpoint_epoch"}))):
            raise ValueError("each configuration needs name/network/overrides or name/initial_model with optional previous_history")
        name = configuration["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise ValueError("configuration names must use lowercase kebab-case")
        if resumed:
            if not isinstance(configuration["initial_model"], str) or not configuration["initial_model"]:
                raise ValueError("initial_model must be a path")
            initial = (path.parent / configuration["initial_model"]).resolve()
            saved = json.loads(initial.read_text(encoding="utf-8"))
            model_config = saved["config"]
            if seeds != [model_config["seed"]]:
                raise ValueError("continuation requires exactly the saved model's seed; it is not a new random initialization")
            if model_config["layers"][0] != 777 or model_config["layers"][-1] != 3:
                raise ValueError("experiment networks require 777 inputs and 3 outputs")
            item = {"name": name, "initial_model": fingerprint(initial)}
            if "previous_history" in configuration:
                previous = configuration["previous_history"]
                if not isinstance(previous, str) or not previous:
                    raise ValueError("previous_history must be a path")
                previous = (path.parent / previous).resolve()
                previous_rows = load_history(previous)
                if not previous_rows:
                    raise ValueError("previous_history must exist and contain epochs")
                checkpoint = configuration.get("previous_checkpoint_epoch")
                if type(checkpoint) is not int or checkpoint not in {row["epoch"] for row in previous_rows}:
                    raise ValueError("previous_history requires a declared previous_checkpoint_epoch in that history")
                item["previous_history"] = fingerprint(previous)
                item["previous_checkpoint_epoch"] = checkpoint
            elif "previous_checkpoint_epoch" in configuration:
                raise ValueError("previous_checkpoint_epoch requires previous_history")
            configurations.append(item)
            continue
        if not isinstance(configuration["network"], str) or not configuration["network"]:
            raise ValueError("configuration.network must be a path")
        source = (path.parent / configuration["network"]).resolve()
        configurations.append({"name": name, "source": fingerprint(source),
                               "options": network_options(source, configuration["overrides"])})
    if len({config["name"] for config in configurations}) != len(configurations):
        raise ValueError("configuration names must be unique")
    training = spec["training"]
    if not isinstance(training, dict) or set(training) != {"command", "settings"}:
        raise ValueError("training must declare command and settings (null until the backend is ready)")
    command = training["command"]
    if command == "builtin":
        training_settings(training["settings"])
    elif command is not None:
        if not isinstance(command, list) or not command or any(not isinstance(arg, str) or not arg for arg in command):
            raise ValueError("training.command must be an argument list, never a shell string")
        used_fields = set()
        for arg in command:
            for _, field, format_spec, conversion in string.Formatter().parse(arg):
                if field is not None and (field not in PLACEHOLDERS or format_spec or conversion):
                    raise ValueError(f"unsupported training placeholder: {field}")
                if field is not None:
                    used_fields.add(field)
        if not {"initial_model", "trained_model", "training_config"} <= used_fields:
            raise ValueError("training.command must consume {initial_model}, {trained_model}, and {training_config}")
    if training["settings"] is not None and (not isinstance(training["settings"], dict) or not training["settings"]):
        raise ValueError("training.settings must be a non-empty object or null")
    if any("initial_model" in config for config in configurations) and command != "builtin":
        raise ValueError("continuation requires the builtin trainer")
    return {"source": fingerprint(path), "spec": spec, "datasets": datasets,
            "configurations": configurations, "seeds": seeds, "training": training}


def run_command(command, cwd, log):
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
    record = {"arguments": command, "seconds": time.perf_counter() - started,
              "returncode": result.returncode}
    if result.returncode:
        raise ValueError(f"command exited with {result.returncode}; see {log}")
    return record


def write_readme(output, manifest):
    """Refresh the factual overview; leave the human conclusion untouched."""
    readme = output / "README.md"
    marker = "<!-- conclusion -->"
    conclusion = "Conclusion: pending review.\n"
    if readme.exists() and marker in readme.read_text(encoding="utf-8"):
        conclusion = readme.read_text(encoding="utf-8").split(marker, 1)[1].lstrip("\n")
    lines = [f"# {output.name}", "", f"Status: **{manifest['status']}**", "",
             manifest["plan"]["spec"].get("hypothesis", "No hypothesis declared."), "",
             "Parameters and provenance: `config.json`. Data stays in `dataset/`; code stays in Git.", ""]
    if manifest.get("error"):
        lines += [f"Failure: {manifest['error']}", ""]
    rows = []
    for run in manifest["runs"]:
        directory = output / run["directory"]
        name = f"{run['configuration']}-seed-{run['seed']}"
        lines += [f"## {name}", "", f"Status: {run['status']}", ""]
        if directory != output:
            lines += [f"Files: [{run['directory']}/]({run['directory']}/)", ""]
        stop = run.get("stopping")
        if stop:
            lines += [f"Epochs completed: {stop['epochs_completed']}; selected epoch: {stop['best_epoch']}; "
                      f"validation loss: {stop['best_validation_loss']:.6f}; stop: {stop['reason']}.", ""]
        metrics_path = directory / "metrics.json"
        if metrics_path.exists():
            phases = json.loads(metrics_path.read_text())
            lines += ["| Phase | Accuracy | Macro F1 | Recall Nothing | Recall Check | Recall Checkmate |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"]
            for phase, metrics in phases.items():
                recalls = [metrics['per_class'][label]['recall'] for label in CLASSES]
                values = [metrics['accuracy'], metrics['macro_f1'], *recalls]
                lines.append(f"| {phase} | " + " | ".join(f"{v:.2%}" for v in values) + " |")
                rows.append([run['configuration'], run['seed'], phase, *values, run.get('training_seconds')])
            lines += ["", "Scores use validation data, not the reserved test set.", ""]
    lines += ["## Files", "", "- `model.nn`: selected model (after training).",
              "- `history.csv`: training and validation loss per epoch.",
              "- `training.log`: execution diagnostics.",
              "- `metrics.json`: before/after validation scores, when evaluated.",
              "- Notebooks display curves and confusion matrices; no HTML or image exports.", "",
              "## Conclusion", "", marker, conclusion]
    readme.write_text("\n".join(lines), encoding="utf-8")
    if len(manifest["runs"]) > 1 and rows:
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["configuration", "seed", "phase", "accuracy", "macro_f1",
                             "recall_nothing", "recall_check", "recall_checkmate", "training_seconds"])
            writer.writerows(rows)


def check_artifacts(artifacts):
    for name, artifact in artifacts.items():
        if fingerprint(Path(artifact["path"]))["sha256"] != artifact["sha256"]:
            raise ValueError(f"{name} changed during the experiment; start a new experiment")


def run_experiment(config_path, output, prepare=False, generator=None, analyzer=None, train_only=False):
    plan = load_spec(config_path)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"{output}: output already exists; choose a new result directory")
    count = len(plan["configurations"]) * len(plan["seeds"])
    if train_only and count != 1:
        raise ValueError("--train-only requires one configuration and one seed")
    training = plan["training"]
    if not prepare and (training["command"] is None or training["settings"] is None):
        raise ValueError("training is not configured yet; use --prepare or configure builtin command/settings")
    tools = {"generator": fingerprint(Path(generator or ROOT / "my_torch_generator").resolve()),
             "analyzer": fingerprint(Path(analyzer or ROOT / "my_torch_analyzer").resolve())}
    manifest = {"version": 1, "status": "preparing", "created_utc": datetime.now(timezone.utc).isoformat(),
                "plan": plan, "tools": tools, "train_only": train_only,
                "machine": {"platform": platform.platform(), "python": platform.python_version()}, "runs": []}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    manifest["git"] = {"revision": revision.stdout.strip() or None,
                       "dirty": bool(dirty.stdout.strip()), "status": dirty.stdout.splitlines()}
    output.mkdir(parents=True)
    manifest_path = output / "config.json"
    write_json(manifest_path, manifest)
    active_run = None
    try:
        for config in plan["configurations"]:
            for seed in plan["seeds"]:
                directory = output if count == 1 else output / "runs" / f"{config['name']}-seed-{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                run = {"configuration": config["name"], "seed": seed,
                       "directory": str(directory.relative_to(output)), "status": "generating"}
                manifest["runs"].append(run)
                active_run = run
                write_json(manifest_path, manifest)
                with tempfile.TemporaryDirectory(prefix="mytorch-run-") as temporary:
                    work = Path(temporary)
                    check_artifacts({**plan["datasets"], **tools})
                    if "initial_model" in config:
                        check_artifacts({"source_model": config["initial_model"]})
                        initial = Path(config["initial_model"]["path"])
                        run["starting_mode"] = "continue_weights"
                        run["source_model"] = config["initial_model"]
                        if "previous_history" in config:
                            check_artifacts({"previous_history": config["previous_history"]})
                            run["previous_history"] = config["previous_history"]
                            run["previous_checkpoint_epoch"] = config["previous_checkpoint_epoch"]
                    else:
                        network_config = work / "network.conf"
                        network_config.write_text("".join(f"{key}={value}\n" for key, value in config["options"].items())
                                                  + f"seed={seed}\n", encoding="utf-8")
                        print(f"Generate {config['name']} / seed {seed}", flush=True)
                        run["generation"] = run_command([tools["generator"]["path"], str(network_config), "1"],
                                                        work, directory / "generation.log")
                        (directory / "generation.log").unlink()
                        initial = work / "models/network_1.nn"
                        run["starting_mode"] = "fresh"
                    model_config = json.loads(initial.read_text(encoding="utf-8"))["config"]
                    if model_config["seed"] != seed:
                        raise ValueError("generator did not preserve the requested seed")
                    run.update(status="prepared", network_config=model_config, initial_model=fingerprint(initial))
                    if prepare:
                        if run["starting_mode"] == "fresh":
                            shutil.copyfile(initial, directory / "initial-model.nn")
                            run["initial_model"] = fingerprint(directory / "initial-model.nn")
                        write_json(manifest_path, manifest)
                        continue
                    request = work / "training.json"
                    write_json(request, {"seed": seed, "settings": training["settings"],
                                         "train": plan["datasets"]["train"], "validation": plan["datasets"]["validation"]})
                    values = {"root": str(ROOT), "analyzer": tools["analyzer"]["path"],
                              "initial_model": str(initial), "trained_model": str(directory / "model.nn"),
                              "train": plan["datasets"]["train"]["path"],
                              "validation": plan["datasets"]["validation"]["path"],
                              "seed": str(seed), "training_config": str(request), "history": str(directory / "history.csv")}
                    builtin = training["command"] == "builtin"
                    command = (builtin_command(values, training["settings"]) if builtin else
                               [arg.format_map(values) for arg in training["command"]])
                    run["status"] = "training"
                    run["command"] = {"arguments": command, "seconds": None, "returncode": None}
                    write_json(manifest_path, manifest)
                    print(f"Train {config['name']} / seed {seed} (results: {directory})", flush=True)
                    try:
                        run["command"] = run_command(command, work if builtin else ROOT, directory / "training.log")
                    finally:
                        # The analyzer writes its detailed log in its working directory.
                        with (directory / "training.log").open("a", encoding="utf-8") as stream:
                            for log in sorted((work / "logs").glob("train_*.log")):
                                stream.write("\n" + log.read_text(encoding="utf-8"))
                    check_artifacts({"initial_model": run["initial_model"]})
                    trained = Path(values["trained_model"])
                    if not trained.is_file():
                        raise ValueError("training returned success without writing model.nn; no evaluation was run")
                    saved_model = json.loads(trained.read_text(encoding="utf-8"))
                    initial_model = json.loads(initial.read_text(encoding="utf-8"))
                    if saved_model["config"] != initial_model["config"]:
                        raise ValueError("trained model must preserve its network configuration and seed")
                    if saved_model["layers"] == initial_model["layers"] and not builtin:
                        raise ValueError("training did not change any weights or biases; refusing to evaluate an unchanged initial model")
                    history = load_history(directory / "history.csv")
                    if builtin:
                        if not history:
                            raise ValueError("builtin training must produce a history")
                        diagnostics = (directory / "training.log").read_text(encoding="utf-8")
                        complete = re.search(r"Training complete: best_epoch=(\d+) validation_loss=([\deE.+-]+)", diagnostics)
                        if complete is None:
                            raise ValueError("builtin training did not confirm its selected checkpoint")
                        best_epoch = int(complete[1])
                        if best_epoch not in {0, *(row["epoch"] for row in history)}:
                            raise ValueError("selected checkpoint is outside the training history")
                        initial_loss = re.search(r"Initial validation_loss=([\deE.+-]+)", diagnostics)
                        run["stopping"] = {"epochs_completed": len(history), "best_epoch": best_epoch,
                                           "best_validation_loss": float(complete[2]),
                                           "initial_validation_loss": float(initial_loss[1]) if initial_loss else None,
                                           "reason": "patience" if len(history) < training["settings"]["epochs"] or
                                                      "Early stopping after epoch" in diagnostics else "epoch_limit_reached",
                                           "unchanged_checkpoint": saved_model["layers"] == initial_model["layers"]}
                    run.update(training_seconds=run["command"]["seconds"], trained_model=fingerprint(trained),
                               model_bytes=trained.stat().st_size)
                    check_artifacts({**plan["datasets"], **tools})
                    if not train_only:
                        run["status"] = "evaluating"
                        write_json(manifest_path, manifest)
                        before, _, _ = predict_metrics(initial, Path(values["validation"]), Path(values["analyzer"]))
                        metrics, _, _ = predict_metrics(trained, Path(values["validation"]), Path(values["analyzer"]))
                        metrics["experiment"] = {"configuration": config["name"], "seed": seed,
                                                 "training_seconds": run["training_seconds"], "model_bytes": run["model_bytes"]}
                        if run["starting_mode"] == "fresh":
                            before["artifacts"]["model"].pop("path")
                            before["artifacts"]["model"]["generated"] = True
                        write_json(directory / "metrics.json", {"before": before, "validation": metrics})
                    if run["starting_mode"] == "fresh":
                        run["initial_model"].pop("path")
                        run["initial_model"]["generated"] = True
                    run["status"] = "completed"
                    write_json(manifest_path, manifest)
        manifest["status"] = "prepared" if prepare else "completed"
        write_json(manifest_path, manifest)
        return load_experiment(output)
    except (OSError, ValueError, KeyError, KeyboardInterrupt) as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = str(error) or "Interrupted"
        if active_run is not None and active_run["status"] != "completed":
            active_run.update(status=manifest["status"], error=manifest["error"])
        raise
    finally:
        for run in manifest["runs"]:
            initial = run.get("initial_model", {})
            if run.get("starting_mode") == "fresh" and not prepare:
                initial.pop("path", None)
                initial["generated"] = True
        write_json(manifest_path, manifest)
        write_readme(output, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new experiment directory")
    parser.add_argument("--prepare", action="store_true", help="generate initial models; never train or evaluate")
    parser.add_argument("--train-only", action="store_true", help="train one model without running classification evaluation")
    args = parser.parse_args()
    try:
        summary = run_experiment(args.config, args.output, args.prepare, train_only=args.train_only)
    except (OSError, ValueError, KeyError) as error:
        print(f"Experiment: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Experiment interrupted; completed runs and logs were retained", file=sys.stderr)
        return 130
    print(f"Status: {summary['status']} | Results: {args.output}")
    print(f"Read: {args.output / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
