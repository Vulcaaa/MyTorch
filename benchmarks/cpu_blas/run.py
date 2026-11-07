#!/usr/bin/env python3
"""Freeze both source trees, then benchmark sequentially on identical fixtures."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shlex
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def source_info(root):
    files = [*root.joinpath("src").rglob("*.cpp"), *root.joinpath("src").rglob("*.hpp"),
             *root.joinpath("include").rglob("*.hpp")]
    return {str(p.relative_to(root)): sha(p) for p in sorted(files)}


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = args.baseline_root.resolve()
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "platform": platform.platform(), "machine": platform.machine(),
                "compiler": command([args.cxx, "--version"]),
                "settings": {"epochs": args.epochs, "batch_size": args.batch_size,
                             "seed": 43, "layers": [777, 128, 64, 3],
                             "hidden_activation": "sigmoid", "output_activation": "softmax",
                             "learning_rate": 0.01, "precision": "FP64"},
                "source": {}, "fixtures": {}, "builds": {}}
    try:
        manifest["cpu"] = command(["lscpu"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        manifest["cpu"] = command(["sysctl", "-n", "machdep.cpu.brand_string"])
    for name, root in (("baseline", baseline), ("candidate", ROOT)):
        snapshot = output / "sources" / name
        for directory in ("src", "include"):
            shutil.copytree(root / directory, snapshot / directory)
        manifest["source"][name] = {"path": str(root), "sha256": source_info(snapshot)}
        try:
            manifest["source"][name]["git_commit"] = command(
                ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            manifest["source"][name]["git_diff"] = command(
                ["git", "-C", str(root), "diff", "HEAD", "--", "src", "include"])
        except subprocess.CalledProcessError:
            # An exact pre-edit source snapshot can be used without its .git directory.
            if (root / "snapshot.json").exists():
                manifest["source"][name]["snapshot"] = json.loads((root / "snapshot.json").read_text())
    harness = output / "Benchmark.cpp"
    shutil.copy2(Path(__file__).with_name("Benchmark.cpp"), harness)
    shutil.copy2(__file__, output / "run.py")
    manifest["harness_sha256"] = sha(harness)
    for name, path, count in (("train", args.train, args.train_rows),
                              ("validation", args.validation, args.validation_rows)):
        fixture = output / f"{name}.fen"
        selected = []
        with path.open("rb") as stream:
            for line in stream:
                if line.strip():
                    selected.append(line)
                if len(selected) == count:
                    break
        if len(selected) != count:
            raise ValueError(f"{path}: expected at least {count} nonempty rows")
        fixture.write_bytes(b"".join(selected))
        manifest["fixtures"][name] = {"source": str(path.resolve()), "rows": count,
                                       "sha256": sha(fixture), "selection": "first nonempty rows"}
    common = [args.cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"]
    if args.blas == "accelerate":
        blas_compile = ["-DMYTORCH_USE_BLAS", "-DMYTORCH_USE_ACCELERATE", "-Wno-deprecated-declarations"]
        blas_link = ["-framework", "Accelerate"]
    else:
        blas_compile = ["-DMYTORCH_USE_BLAS"] + shlex.split(args.blas_cflags)
        blas_link = shlex.split(args.blas_libs)
    for name, source, accelerated in (("baseline", "baseline", False),
                                       ("candidate-scalar", "candidate", False),
                                       ("candidate-blas", "candidate", True)):
        root = output / "sources" / source
        executable = output / name
        sources = sorted((root / "src/neural").glob("*.cpp"))
        sources += [root / "src/chess/Dataset.cpp", root / "src/chess/Encoding.cpp", harness]
        build = common + (blas_compile if accelerated else [])
        build += ["-I" + str(root / "src"), "-I" + str(root / "include")]
        build += [str(p) for p in sources] + (blas_link if accelerated else []) + ["-o", str(executable)]
        print(f"Build {name}", flush=True)
        with (output / f"{name}-build.log").open("w") as log:
            subprocess.run(build, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest["builds"][name] = {"command": build, "sha256": sha(executable)}
        try:
            manifest["builds"][name]["libraries"] = command(
                (["otool", "-L"] if platform.system() == "Darwin" else ["ldd"]) + [str(executable)])
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    write_json(output / "manifest.json", manifest)
    print(f"Prepared source snapshots and binaries in {output}", flush=True)
    return output


def active_training():
    lines = command(["ps", "-eo", "pid=,comm=,args="]).splitlines()
    # Check actual analyzer processes, not shell command lines mentioning their name.
    return [line.strip() for line in lines
            if len(line.split(None, 2)) == 3
            and Path(line.split(None, 2)[1]).name.startswith("my_torch_analy")
            and "--train" in line.split(None, 2)[2].split()]


def parameters(model):
    data = json.loads(model.read_text())
    return [v for layer in data["layers"] for row in layer["weights"] for v in row] + [
        v for layer in data["layers"] for v in layer["biases"]]


def measure(output, args):
    manifest = json.loads((output / "manifest.json").read_text())
    settings = manifest["settings"]
    for name, build in manifest["builds"].items():
        if sha(output / name) != build["sha256"]:
            raise ValueError(f"prepared binary changed: {name}")
    for name, fixture in manifest["fixtures"].items():
        if sha(output / f"{name}.fen") != fixture["sha256"]:
            raise ValueError(f"prepared fixture changed: {name}")
    if args.wait_pid:
        print(f"Waiting for experiment process {args.wait_pid}; no timing while it runs.", flush=True)
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(15)
    running = active_training()
    if running:
        raise RuntimeError("Training still active; postpone benchmarks:\n" + "\n".join(running))
    if (output / "measurements").exists():
        raise ValueError("measurements already exist; use a fresh prepared directory")
    raw = output / "measurements"
    raw.mkdir()
    jobs = [(name, threads) for name in ("baseline", "candidate-scalar") for threads in [1]]
    jobs += [("candidate-blas", threads) for threads in args.threads]
    schedule = [(name, threads, repeat) for repeat in range(args.repeats) for name, threads in jobs]
    random.Random(43).shuffle(schedule)
    write_json(raw / "schedule.json", schedule)
    write_json(raw / "environment.json", {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threads": args.threads, "repeats": args.repeats,
        "wait_pid": args.wait_pid, "platform": platform.platform(),
        "thread_variables": ["OPENBLAS_NUM_THREADS", "OPENBLAS_DEFAULT_NUM_THREADS",
                             "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"],
    })
    results = []
    for name, threads, repeat in schedule:
        if active_training():
            raise RuntimeError("Another training started; incomplete measurements preserved")
        label = f"{name}-t{threads}-r{repeat + 1}"
        env = dict(os.environ, OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                   OPENBLAS_DEFAULT_NUM_THREADS=str(threads), VECLIB_MAXIMUM_THREADS=str(threads))
        invocation = [str(output / name), str(output / "train.fen"), str(output / "validation.fen"),
                      str(settings["epochs"]), str(settings["batch_size"]), str(settings["seed"]),
                      str(raw / f"{label}.nn")]
        print(f"Measure {label}", flush=True)
        with (raw / f"{label}.stderr.log").open("w") as log:
            text = command(invocation, env=env, stderr=log)
        result = json.loads(text)
        result.update(backend=name, threads=threads, repeat=repeat + 1, command=invocation)
        write_json(raw / f"{label}.json", result)
        results.append(result)
    reference = next(r for r in results if r["backend"] == "baseline")
    reference_model = parameters(raw / f"baseline-t1-r{reference['repeat']}.nn")
    summaries = []
    for name, threads in jobs:
        runs = [r for r in results if r["backend"] == name and r["threads"] == threads]
        differences = []
        for r in runs:
            model = parameters(raw / f"{name}-t{threads}-r{r['repeat']}.nn")
            if len(model) != len(reference_model):
                raise ValueError("model parameter shapes differ")
            differences.append(max(abs(a - b) for a, b in zip(model, reference_model)))
        summaries.append({"backend": name, "threads": threads,
                          "median_seconds": statistics.median(r["total_seconds"] for r in runs),
                          "min_seconds": min(r["total_seconds"] for r in runs),
                          "max_seconds": max(r["total_seconds"] for r in runs),
                          "max_parameter_difference": max(differences),
                          "max_loss_difference": max(abs(r["validation_loss"] - reference["validation_loss"])
                                                     for r in runs),
                          "validation_accuracy": [r["validation_accuracy"] for r in runs]})
    baseline = summaries[0]["median_seconds"]
    for row in summaries:
        row["speedup"] = baseline / row["median_seconds"]
        row["seconds_per_epoch"] = row["median_seconds"] / settings["epochs"]
        row["training_positions_per_second"] = (manifest["fixtures"]["train"]["rows"]
                                                * settings["epochs"] / row["median_seconds"])
    passed = all(r["max_parameter_difference"] < 1e-10 and r["max_loss_difference"] < 1e-10
                 and all(a == reference["validation_accuracy"] for a in r["validation_accuracy"])
                 for r in summaries)
    write_json(output / "summary.json", {"agreement_passed": passed, "results": summaries,
                                         "baseline_train_loss": reference["train_loss"],
                                         "baseline_validation_loss": reference["validation_loss"]})
    lines = ["# CPU BLAS benchmark", "", f"Numerical agreement: {'PASS' if passed else 'FAIL'}.", "",
             "| Backend | Threads | Median (s) | Speedup | Max parameter difference |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['backend']} | {row['threads']} | {row['median_seconds']:.6f} | "
                     f"{row['speedup']:.2f}x | {row['max_parameter_difference']:.3g} |")
    lines += ["", "See manifest.json for source/data hashes and build commands, and measurements/ for raw runs.",
              "Fixed-epoch throughput benchmark; this does not establish time to a quality target."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    if not passed:
        raise RuntimeError("numerical agreement failed; inspect summary.json before using performance claims")
    if args.training_doc:
        document = args.training_doc.resolve()
        content = document.read_text()
        begin, end = "<!-- cpu-blas-measurements:start -->", "<!-- cpu-blas-measurements:end -->"
        if content.count(begin) != 1 or content.count(end) != 1:
            raise ValueError("training documentation markers missing/ambiguous; raw results preserved")
        left, rest = content.split(begin)
        _, right = rest.split(end)
        note = ["Measurements completed after the existing experiment exited.", "",
                f"Local evidence directory: `{output}`.", "",
                f"Fixture: {manifest['fixtures']['train']['rows']:,} train / "
                f"{manifest['fixtures']['validation']['rows']:,} validation positions; "
                f"{settings['epochs']} epochs; batch {settings['batch_size']}; seed {settings['seed']}; "
                f"{args.repeats} runs per configuration. Both builds use `-O2` and FP64.", "",
                *lines[4:], "",
                f"Baseline final train/validation loss: {reference['train_loss']:.16g} / "
                f"{reference['validation_loss']:.16g}.", "",
                "Source snapshots, dataset hashes, build commands, raw timings, numerical comparisons "
                "and final models are retained in the evidence directory. "
                "These results measure throughput at fixed epochs, not time to a target validation quality."]
        document.write_text(left + begin + "\n\n" + "\n".join(note) + "\n\n" + end + right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prepared", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--blas", choices=["openblas", "accelerate"],
                        default="accelerate" if platform.system() == "Darwin" else "openblas")
    parser.add_argument("--blas-cflags", default=None)
    parser.add_argument("--blas-libs", default=None)
    parser.add_argument("--train-rows", type=int, default=8192)
    parser.add_argument("--validation-rows", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--training-doc", type=Path,
                        help="update only the reserved CPU BLAS measurements section after success")
    args = parser.parse_args()
    for value in [args.train_rows, args.validation_rows, args.epochs, args.batch_size,
                  args.repeats, *args.threads]:
        if value <= 0:
            parser.error("row counts, epochs, batch size, threads and repeats must be positive")
    if len(set(args.threads)) != len(args.threads):
        parser.error("thread counts must be unique")
    if args.run_prepared:
        output = args.run_prepared.resolve()
    else:
        if not all((args.baseline_root, args.train, args.validation, args.output)):
            parser.error("preparation requires --baseline-root, --train, --validation, --output")
        if args.blas == "openblas":
            if args.blas_cflags is None:
                args.blas_cflags = command(["pkg-config", "--cflags", "openblas"])
            if args.blas_libs is None:
                args.blas_libs = command(["pkg-config", "--libs", "openblas"])
        output = prepare(args)
    if not args.prepare_only:
        measure(output, args)


if __name__ == "__main__":
    main()
