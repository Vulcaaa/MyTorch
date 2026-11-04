"""Freeze and verify versioned chess datasets using the existing preparation tools."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

from scripts.clean_dataset import clean_dataset, parse_labeled_fen
from scripts.split_dataset import CLASSES, SPLITS, split_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "dataset/manifests/chess-v1.json"
SETTINGS = {"seed": 42, "ratios": [0.8, 0.1, 0.1], "balance_training": False}
TOOLS = ("scripts/clean_dataset.py", "scripts/split_dataset.py", "scripts/reference_dataset.py")
ARTIFACTS = {
    "cleaned": "processed/cleaned.txt",
    "provenance": "processed/provenance.jsonl",
    "cleaning_report": "processed/report.json",
    **{name: f"splits/{name}.txt" for name in SPLITS},
    "split_metadata": "splits/split_info.json",
}
PROVENANCE = {
    "origin": "Existing project files under dataset/raw/; external origin unknown.",
    "generator": "Unknown; filenames alone do not establish generation methods.",
    "license": "Unknown; no source license or external usage grant documented.",
    "usage": "Retained for local MyTorch experiments and the defense; redistribution rights unverified.",
    "audit": "FEN syntax and label format checked; chess legality and label correctness not audited.",
}


def confined(root, relative):
    path = Path(relative)
    if path.is_absolute() or not (root / path).resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path must stay within {root}: {relative}")
    return root / path


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def rows(path, *, allow_invalid=False):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                _, label, key = parse_labeled_fen(line)
            except ValueError as error:
                if not allow_invalid:
                    raise ValueError(f"{path}:{number}: {error}") from error
                label, key = None, None
            yield label, key


def statistics(path, *, allow_invalid=False):
    counts = Counter(label for label, _ in rows(path, allow_invalid=allow_invalid))
    return {"records": sum(counts.values()), "invalid_records": counts[None],
            "class_counts": {label: counts[label] for label in CLASSES}}


def check_file(path, expected):
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    if fingerprint(path) != {key: expected[key] for key in ("sha256", "bytes")}:
        raise ValueError(f"modified file (SHA-256/size): {path}")


def check_sources(manifest, root):
    paths = [source["path"] for source in manifest["sources"]]
    if not paths or paths != sorted(set(paths)):
        raise ValueError("sources must be a nonempty sorted list of distinct paths")
    for source in manifest["sources"]:
        relative = Path(source["path"])
        if not relative.is_relative_to("dataset/raw") or relative.suffix != ".txt":
            raise ValueError(f"source must be a .txt file under dataset/raw/: {relative}")
        path = confined(root, source["path"])
        check_file(path, source)
        if statistics(path, allow_invalid=True) != source["statistics"]:
            raise ValueError(f"source counts disagree with manifest: {path}")


def verify(manifest, *, root=ROOT, storage=None):
    """Read only: check hashes, counts, labels, coverage and disjoint model inputs."""
    if manifest["version"] != 1 or manifest["settings"] != SETTINGS:
        raise ValueError("unsupported reference version or preparation settings")
    if set(manifest["artifacts"]) != set(ARTIFACTS):
        raise ValueError("manifest must declare all reference artifacts")
    check_sources(manifest, root)
    storage = Path(storage) if storage is not None else confined(root, manifest["storage"])
    for role, relative in ARTIFACTS.items():
        entry = manifest["artifacts"][role]
        if entry["path"] != relative:
            raise ValueError(f"unexpected artifact path for {role}")
        check_file(storage / relative, entry)

    # One map checks cleaned uniqueness, split labels, duplicates, overlaps and coverage.
    inputs = {}
    counts = Counter()
    for label, key in rows(storage / ARTIFACTS["cleaned"]):
        if key in inputs:
            raise ValueError("duplicate model input in cleaned data")
        inputs[key] = (label, None)
        counts[label] += 1
    actual = {"cleaned": {label: counts[label] for label in CLASSES}}
    assigned = 0
    for role in SPLITS:
        counts = Counter()
        for label, key in rows(storage / ARTIFACTS[role]):
            if key not in inputs or inputs[key][0] != label:
                raise ValueError(f"{role}: input/label absent from cleaned data")
            owner = inputs[key][1]
            if owner is not None:
                raise ValueError(f"duplicate/overlapping model input in {owner} and {role}")
            inputs[key] = (label, role)
            assigned += 1
            counts[label] += 1
        actual[role] = {label: counts[label] for label in CLASSES}
        if any(counts[label] == 0 for label in CLASSES):
            raise ValueError(f"{role}: every class must be represented")
    if assigned != len(inputs):
        raise ValueError("splits do not cover every cleaned input")
    for role, counts in actual.items():
        expected = {"records": sum(counts.values()), "invalid_records": 0, "class_counts": counts}
        if manifest["artifacts"][role]["statistics"] != expected:
            raise ValueError(f"{role}: counts disagree with manifest")

    info = json.loads((storage / ARTIFACTS["split_metadata"]).read_text(encoding="utf-8"))
    expected_info = {
        "input_file": "cleaned.txt", "input_sha256": manifest["artifacts"]["cleaned"]["sha256"],
        "seed": SETTINGS["seed"], "ratios": dict(zip(SPLITS, SETTINGS["ratios"])),
        "balance_training": "none", "input": actual["cleaned"],
        "before_balancing": {role: actual[role] for role in SPLITS},
        "output": {role: actual[role] for role in SPLITS}, "overlapping_inputs": 0,
    }
    if info != expected_info:
        raise ValueError("split metadata disagrees with files or reference settings")
    report = json.loads((storage / ARTIFACTS["cleaning_report"]).read_text(encoding="utf-8"))
    summary = cleaning_summary(report)
    source_records = sum(source["statistics"]["records"] for source in manifest["sources"])
    if (summary != manifest["cleaning_summary"] or report["by_class"] != actual["cleaned"]
            or report["accepted"] != len(inputs) or report["records"] != source_records
            or report["records"] != report["accepted"] + report["duplicates"]
            + report["conflicting_records"] + len(report["rejected"])
            or report["files"] != [str(Path(source["path"]).relative_to("dataset"))
                                   for source in manifest["sources"]]):
        raise ValueError("cleaning report counts/sources disagree with manifest or files")
    with (storage / ARTIFACTS["provenance"]).open(encoding="utf-8") as stream:
        provenance_count = 0
        for line in stream:
            refs = json.loads(line)
            if not isinstance(refs, list) or not refs:
                raise ValueError("invalid provenance row")
            provenance_count += 1
    if provenance_count != len(inputs):
        raise ValueError("provenance count disagrees with cleaned data")
    return actual


def cleaning_summary(report):
    return {key: report[key] for key in ("records", "accepted", "duplicates", "conflicting_records")} | {
        "rejected_records": len(report["rejected"]), "class_counts": report["by_class"]}


def generate(sources, root, temporary):
    """Stage only the recorded sources, preserving cleaner traversal and report paths."""
    source_dir = temporary / "input"
    for source in sources:
        destination = source_dir / Path(source["path"]).relative_to("dataset")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(confined(root, source["path"]), destination)
        check_file(destination, source)
    output = temporary / "output"
    report = clean_dataset(source_dir, output / "processed")
    split_dataset(output / ARTIFACTS["cleaned"], output / "splits", **{
        **SETTINGS, "ratios": tuple(SETTINGS["ratios"])})
    return output, report


def prepare(name, *, root=ROOT):
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("reference name must use lowercase kebab-case")
    manifest_path = root / "dataset/manifests" / f"{name}.json"
    storage = f"dataset/references/{name}"
    destination = root / storage
    if manifest_path.exists() or destination.exists():
        raise ValueError("reference already exists; intentional changes require a new version")
    sources = []
    for path in sorted((root / "dataset/raw").rglob("*.txt")):
        sources.append({"path": path.relative_to(root).as_posix(), **fingerprint(path),
                        "statistics": statistics(path, allow_invalid=True)})
    if not sources:
        raise ValueError("no .txt sources under dataset/raw/")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    manifest = {
        "version": 1, "name": name, "storage": storage, "settings": SETTINGS,
        "environment": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                        "platform": platform.platform()},
        "tools": {"base_git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                  "files": [{"path": path, **fingerprint(root / path)} for path in TOOLS]},
        "provenance": PROVENANCE, "sources": sources,
        "commands": {"prepare": f"python3 -m scripts.reference_dataset prepare {name}",
                     "verify": f"python3 -m scripts.reference_dataset verify {manifest_path.relative_to(root)}",
                     "recreate": f"python3 -m scripts.reference_dataset recreate {manifest_path.relative_to(root)}"},
    }
    with tempfile.TemporaryDirectory(prefix="mytorch-reference-") as temporary:
        output, report = generate(sources, root, Path(temporary))
        manifest["cleaning_summary"] = cleaning_summary(report)
        manifest["artifacts"] = {
            role: {"path": relative, **fingerprint(output / relative)}
            | ({"statistics": statistics(output / relative)} if role in (*SPLITS, "cleaned") else {})
            for role, relative in ARTIFACTS.items()}
        verify(manifest, root=root, storage=output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output, destination)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest_path


def recreate(manifest, *, root=ROOT, output=None):
    destination = Path(output) if output is not None else confined(root, manifest["storage"])
    if destination.exists():
        raise ValueError(f"{destination}: output already exists; reference will not be overwritten")
    environment = manifest["environment"]
    if (platform.python_version(), platform.python_implementation()) != (
            environment["python"], environment["implementation"]):
        raise ValueError(f"recreation requires {environment['implementation']} {environment['python']}")
    for tool in manifest["tools"]["files"]:
        check_file(confined(root, tool["path"]), tool)
    check_sources(manifest, root)
    with tempfile.TemporaryDirectory(prefix="mytorch-recreate-") as temporary:
        generated, _ = generate(manifest["sources"], root, Path(temporary))
        verify(manifest, root=root, storage=generated)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(generated, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="freeze current raw sources as a new reference").add_argument("name")
    for name in ("verify", "recreate"):
        command = commands.add_parser(name)
        command.add_argument("manifest", nargs="?", default=DEFAULT_MANIFEST)
        command.add_argument("--output" if name == "recreate" else "--storage", type=Path,
                             help="alternative artifact directory (manifest remains unchanged)")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            print(f"Frozen reference: {prepare(args.name)}")
        else:
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            if args.command == "verify":
                counts = verify(manifest, storage=args.storage)
                print(f"Verified {manifest['name']}: " + ", ".join(
                    f"{role}={sum(counts[role].values())}" for role in SPLITS) + "; overlap=0")
            else:
                print(f"Recreated identical artifacts: {recreate(manifest, output=args.output)}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Reference error: {error}\n")


if __name__ == "__main__":
    main()
