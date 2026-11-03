"""Prepare or run a reproducible configuration-by-seed experiment."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import string
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evaluate import CLASSES, evaluate, fingerprint, read_labels
from scripts.comparison import load_experiment, load_history


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
    spec = json.loads(path.read_text(encoding="utf-8"))
    json.dumps(spec, allow_nan=False)
    required = {"version", "seeds", "datasets", "configurations", "training"}
    if (not isinstance(spec, dict) or set(spec) != required or
            type(spec["version"]) is not int or spec["version"] != 1):
        raise ValueError("experiment config must contain version=1, seeds, datasets, configurations, training")
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
        if not isinstance(configuration, dict) or set(configuration) != {"name", "network", "overrides"}:
            raise ValueError("each configuration needs name, network, overrides")
        name = configuration["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise ValueError("configuration names must use lowercase kebab-case")
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
    if command is not None:
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
    return {"source": fingerprint(path), "spec": spec, "datasets": datasets,
            "configurations": configurations, "seeds": seeds, "training": training}


def run_command(command, cwd, prefix):
    started = time.perf_counter()
    record = {"arguments": command, "cwd": str(cwd), "seconds": None, "returncode": None}
    write_json(prefix.with_suffix(".command.json"), record)
    result = None
    try:
        with prefix.with_suffix(".stdout.log").open("w") as stdout, prefix.with_suffix(".stderr.log").open("w") as stderr:
            result = subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, check=False)
    finally:
        record.update(seconds=time.perf_counter() - started,
                      returncode=result.returncode if result is not None else None)
        write_json(prefix.with_suffix(".command.json"), record)
    if result.returncode:
        raise ValueError(f"command exited with {result.returncode}; see {prefix}.stderr.log")
    return record["seconds"]


def check_artifacts(artifacts):
    for name, artifact in artifacts.items():
        if fingerprint(Path(artifact["path"]))["sha256"] != artifact["sha256"]:
            raise ValueError(f"{name} changed during the experiment; start a new experiment")


def run_experiment(config_path, output, prepare=False, generator=None, analyzer=None):
    plan = load_spec(config_path)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"{output}: output already exists; choose a new experiment directory")
    training = plan["training"]
    if not prepare and (training["command"] is None or training["settings"] is None):
        raise ValueError("training is not configured yet; use --prepare. Agree on command/settings with Gustave first")
    tools = {"generator": fingerprint(Path(generator or ROOT / "my_torch_generator").resolve()),
             "analyzer": fingerprint(Path(analyzer or ROOT / "my_torch_analyzer").resolve())}
    code = {path.name: fingerprint(path) for path in
            (Path(__file__), ROOT / "scripts/evaluate.py", ROOT / "scripts/comparison.py")}
    manifest = {"version": 1, "status": "preparing", "created_utc": datetime.now(timezone.utc).isoformat(),
                "plan": plan, "tools": tools, "code": code,
                "machine": {"platform": platform.platform(), "python": platform.python_version()}, "runs": []}
    output.mkdir(parents=True)
    manifest_path = output / "experiment.json"
    write_json(manifest_path, manifest)
    active_run = None
    try:
        # Prepare every configuration first, so malformed generation cannot start a partial training campaign.
        for config in plan["configurations"]:
            for seed in plan["seeds"]:
                directory = output / "runs" / config["name"] / f"seed-{seed}"
                directory.mkdir(parents=True)
                network_config = directory / "network.conf"
                network_config.write_text("".join(f"{key}={value}\n" for key, value in config["options"].items())
                                          + f"seed={seed}\n", encoding="utf-8")
                run = {"configuration": config["name"], "seed": seed,
                       "directory": str(directory.relative_to(output)), "status": "generating"}
                manifest["runs"].append(run)
                active_run = run
                write_json(manifest_path, manifest)
                print(f"Generate {config['name']} / seed {seed}", flush=True)
                run_command([tools["generator"]["path"], str(network_config), "1"], directory, directory / "generate")
                initial = directory / "models/network_1.nn"
                model_config = json.loads(initial.read_text(encoding="utf-8"))["config"]
                if model_config["seed"] != seed:
                    raise ValueError("generator did not preserve the requested seed")
                run.update(status="prepared", initial_model=fingerprint(initial))
                write_json(directory / "training.json", {"seed": seed, "settings": training["settings"],
                           "train": plan["datasets"]["train"], "validation": plan["datasets"]["validation"]})
                write_json(manifest_path, manifest)
        if prepare:
            check_artifacts({**plan["datasets"], **tools})
            manifest["status"] = "prepared"
            print("Prepared initial models only. No training or evaluation performed.", flush=True)
        else:
            manifest["status"] = "running"
            for run in manifest["runs"]:
                active_run = run
                directory = output / run["directory"]
                check_artifacts({**plan["datasets"], **tools})
                values = {"root": str(ROOT), "analyzer": tools["analyzer"]["path"],
                          "initial_model": run["initial_model"]["path"], "trained_model": str(directory / "trained.nn"),
                          "train": plan["datasets"]["train"]["path"], "validation": plan["datasets"]["validation"]["path"],
                          "seed": str(run["seed"]), "training_config": str(directory / "training.json"),
                          "history": str(directory / "history.csv")}
                command = [arg.format_map(values) for arg in training["command"]]
                run["status"] = "training"
                write_json(manifest_path, manifest)
                print(f"Train {run['configuration']} / seed {run['seed']} (logs: {directory})", flush=True)
                seconds = run_command(command, ROOT, directory / "train")
                check_artifacts({"initial_model": run["initial_model"]})
                trained = Path(values["trained_model"])
                if not trained.is_file():
                    raise ValueError("training returned success without writing trained.nn; no evaluation was run")
                saved_model = json.loads(trained.read_text(encoding="utf-8"))
                initial_model = json.loads(Path(values["initial_model"]).read_text(encoding="utf-8"))
                if saved_model["config"] != initial_model["config"]:
                    raise ValueError("trained model must preserve its network configuration and seed")
                if saved_model["layers"] == initial_model["layers"]:
                    raise ValueError("training did not change any weights or biases; refusing to evaluate an unchanged initial model")
                load_history(directory / "history.csv")
                run.update(status="evaluating", training_seconds=seconds, trained_model=fingerprint(trained),
                           model_bytes=trained.stat().st_size)
                check_artifacts({**plan["datasets"], **tools})
                write_json(manifest_path, manifest)
                print(f"Evaluate {run['configuration']} / seed {run['seed']} on validation", flush=True)
                metrics = evaluate(trained, Path(values["validation"]), Path(values["analyzer"]), directory / "evaluation")
                metrics["experiment"] = {"configuration": run["configuration"], "seed": run["seed"],
                                         "training_seconds": seconds, "model_bytes": run["model_bytes"]}
                write_json(directory / "evaluation/metrics.json", metrics)
                run["status"] = "completed"
                write_json(manifest_path, manifest)
            check_artifacts({**plan["datasets"], **tools})
            manifest["status"] = "completed"
        write_json(manifest_path, manifest)
        summary = load_experiment(output)
        write_json(output / "summary.json", summary)
    except (OSError, ValueError, KeyError, KeyboardInterrupt) as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = str(error) or "Interrupted"
        if active_run is not None and active_run["status"] != "completed":
            active_run["status"] = manifest["status"]
            active_run["error"] = manifest["error"]
        write_json(manifest_path, manifest)
        try:
            write_json(output / "summary.json", load_experiment(output))
        except (OSError, ValueError, KeyError) as summary_error:
            manifest["summary_error"] = str(summary_error)
        raise
    finally:
        write_json(manifest_path, manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new experiment directory")
    parser.add_argument("--prepare", action="store_true", help="generate initial models; never train or evaluate")
    args = parser.parse_args()
    try:
        summary = run_experiment(args.config, args.output, args.prepare)
    except (OSError, ValueError, KeyError) as error:
        print(f"Experiment: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Experiment interrupted; completed runs and logs were retained", file=sys.stderr)
        return 130
    print(f"Status: {summary['status']} | Results: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
