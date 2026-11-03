"""Create reproducible, class-stratified chess dataset splits."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from scripts.clean_dataset import parse_labeled_fen


SPLITS = ("train", "validation", "test")
CLASSES = ("Nothing", "Check", "Checkmate")


@dataclass
class Group:
    label: str
    rows: list[str] = field(default_factory=list)


def _allocate(count: int, ratios: tuple[float, float, float]) -> list[int]:
    quotas = [count * ratio for ratio in ratios]
    allocated = [math.floor(quota) for quota in quotas]
    for index in sorted(range(3), key=lambda i: (-(quotas[i] - allocated[i]), i))[:count - sum(allocated)]:
        allocated[index] += 1
    for missing in (i for i in range(3) if allocated[i] == 0):
        donor = max((i for i in range(3) if allocated[i] > 1),
                    key=lambda i: (allocated[i] - quotas[i], allocated[i]))
        allocated[donor] -= 1
        allocated[missing] += 1
    return allocated


def split_dataset(
    input_path: str | Path = "dataset/processed/cleaned.txt",
    output_dir: str | Path = "dataset/splits",
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    balance_training: bool = False,
) -> dict:
    """Write train, validation, test, and split_info files from cleaned data.

    The optional balancing duplicates minority-class rows in training only.
    Every class must have at least three distinct model inputs so it can appear
    in all three splits.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios) or not math.isclose(
        sum(ratios), 1.0, rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("ratios must be three positive values summing to 1")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    groups: dict[tuple[str, str, str], Group] = {}
    input_classes: Counter[str] = Counter()
    digest = hashlib.sha256()
    with input_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                fen, label, key = parse_labeled_fen(line)
            except ValueError as error:
                raise ValueError(f"{input_path}:{line_number}: {error}") from error
            group = groups.get(key)
            if group is None:
                group = groups[key] = Group(label)
            elif group.label != label:
                raise ValueError(f"{input_path}:{line_number}: conflicting labels for one model input")
            group.rows.append(f"{fen} {label}\n")
            input_classes[label] += 1
    if not groups:
        raise ValueError(f"{input_path}: no positions to split")

    by_class: dict[str, list[tuple[tuple[str, str, str], Group]]] = {label: [] for label in CLASSES}
    for key, group in groups.items():
        by_class[group.label].append((key, group))
    for label, entries in by_class.items():
        if len(entries) < 3:
            raise ValueError(f"{label}: need at least 3 distinct positions for train, validation, and test; found {len(entries)}")

    rng = random.Random(seed)
    chosen: dict[str, list[tuple[tuple[str, str, str], Group]]] = {name: [] for name in SPLITS}
    for label in CLASSES:
        entries = by_class[label]
        rng.shuffle(entries)
        sizes = _allocate(len(entries), ratios)
        start = 0
        for name, size in zip(SPLITS, sizes):
            chosen[name].extend(entries[start:start + size])
            start += size

    keys = [{key for key, _ in chosen[name]} for name in SPLITS]
    if any(keys[i] & keys[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("model inputs overlap between splits")
    if sum(len(part) for part in keys) != len(groups):
        raise RuntimeError("some model inputs were not assigned")

    rows: dict[str, list[tuple[str, str]]] = {
        name: [(row, group.label) for _, group in chosen[name] for row in group.rows]
        for name in SPLITS
    }
    for name in SPLITS:
        rng.shuffle(rows[name])
    before = {name: Counter(label for _, label in rows[name]) for name in SPLITS}
    if balance_training:
        balance_rng = random.Random(f"{seed}:balance")
        target = max(before["train"].values())
        for label in CLASSES:
            candidates = [row for row, row_label in rows["train"] if row_label == label]
            rows["train"].extend((balance_rng.choice(candidates), label)
                                 for _ in range(target - before["train"][label]))
        balance_rng.shuffle(rows["train"])

    after = {name: Counter(label for _, label in rows[name]) for name in SPLITS}
    info = {
        "input_file": input_path.name,
        "input_sha256": digest.hexdigest(),
        "seed": seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "balance_training": "oversample" if balance_training else "none",
        "input": {label: input_classes[label] for label in CLASSES},
        "before_balancing": {
            name: {label: before[name][label] for label in CLASSES} for name in SPLITS
        },
        "output": {
            name: {label: after[name][label] for label in CLASSES} for name in SPLITS
        },
        "overlapping_inputs": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        with (output_dir / f"{name}.txt").open("w", encoding="utf-8") as stream:
            stream.writelines(row for row, _ in rows[name])
    with (output_dir / "split_info.json").open("w", encoding="utf-8") as stream:
        json.dump(info, stream, indent=2)
        stream.write("\n")
    return info
