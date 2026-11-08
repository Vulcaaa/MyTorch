"""Freeze prepared training variants while retaining the reference held-outs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import tempfile

from scripts.reference_dataset import ROOT, check_file, confined, fingerprint, rows


def verify(manifest: dict, *, root: Path = ROOT, storage: Path | None = None) -> dict:
    if manifest.get("kind") != "derived-training" or manifest.get("version") != 1:
        raise ValueError("unsupported derived reference manifest")
    check_file(confined(root, manifest["base"]["path"]), manifest["base"])
    base = json.loads(confined(root, manifest["base"]["path"]).read_text())
    storage = storage or confined(root, manifest["storage"])
    for entry in manifest["artifacts"].values():
        check_file(confined(storage, entry["path"]), entry)
    for role in ("validation", "test"):
        if any(manifest["artifacts"][role][key] != base["artifacts"][role][key]
               for key in ("sha256", "bytes")):
            raise ValueError(f"{role} differs from base reference")
    seen = set()
    counts = {}
    for role in ("train", "validation", "test"):
        classes = Counter()
        for label, key in rows(storage / manifest["artifacts"][role]["path"]):
            if key in seen:
                raise ValueError(f"duplicate or overlapping input in {role}")
            seen.add(key)
            classes[label] += 1
        counts[role] = dict(classes)
        if (not classes or counts[role] != manifest["artifacts"][role]["class_counts"]
                or sum(classes.values()) != manifest["artifacts"][role]["rows"]):
            raise ValueError(f"{role}: class counts disagree with manifest")
    return counts


def freeze(name: str, train: Path, base_manifest: Path, extras: dict[str, Path],
           *, root: Path = ROOT) -> Path:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("reference name must use lowercase kebab-case")
    destination = root / "dataset/references" / name
    manifest_path = root / "dataset/manifests" / f"{name}.json"
    if destination.exists() or manifest_path.exists():
        raise ValueError("reference already exists; use a new name")
    base = json.loads(base_manifest.read_text())
    sources = {"train": train, **{role: root / base["storage"] / base["artifacts"][role]["path"]
                                  for role in ("validation", "test")}, **extras}
    before = {role: fingerprint(path) for role, path in sources.items()}
    manifest = {
        "version": 1, "kind": "derived-training", "name": name,
        "storage": destination.relative_to(root).as_posix(),
        "base": {"path": base_manifest.relative_to(root).as_posix(), **fingerprint(base_manifest)},
        "sources": {role: {"path": path.relative_to(root).as_posix(), **before[role]}
                    for role, path in sources.items()},
        "policy": "Prepared train; unchanged base validation/test; unique model inputs across all splits.",
        "artifacts": {},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=destination.parent) as tmp:
        staged = Path(tmp) / name
        staged.mkdir()
        for role, path in sources.items():
            relative = f"splits/{role}.txt" if role in ("train", "validation", "test") else f"provenance/{role}"
            target = confined(staged, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            entry = {"path": relative, **fingerprint(target)}
            if fingerprint(path) != before[role] or fingerprint(target) != before[role]:
                raise ValueError(f"source changed during freezing: {path}")
            if role in ("train", "validation", "test"):
                entry["class_counts"] = dict(Counter(label for label, _ in rows(target)))
                entry["rows"] = sum(entry["class_counts"].values())
            manifest["artifacts"][role] = entry
        verify(manifest, root=root, storage=staged)
        staged.rename(destination)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("name")
    freeze_parser.add_argument("--train", type=Path, required=True)
    freeze_parser.add_argument("--base", type=Path, default=ROOT / "dataset/manifests/chess-v1.json")
    freeze_parser.add_argument("--extra", action="append", default=[], metavar="NAME=PATH")
    commands.add_parser("verify").add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            extras = {name: Path(path).resolve() for name, path in
                      (entry.split("=", 1) for entry in args.extra)}
            print(freeze(args.name, args.train.resolve(), args.base.resolve(), extras))
        else:
            manifest = json.loads(args.manifest.read_text())
            print(json.dumps(verify(manifest), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Reference error: {error}\n")


if __name__ == "__main__":
    main()
