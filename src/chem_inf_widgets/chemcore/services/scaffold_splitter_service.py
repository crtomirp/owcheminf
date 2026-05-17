from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Literal, Optional

from chem_inf_widgets.chemcore.services.scaffold_service import (
    NO_SCAFFOLD_LABEL,
    analyze_scaffolds,
)


SplitLabel = Literal["train", "validation", "test", "invalid"]


@dataclass(frozen=True)
class ScaffoldSplitAssignment:
    index: int
    split: SplitLabel
    scaffold: Optional[str]
    status: str


@dataclass(frozen=True)
class ScaffoldSplitSummary:
    split: SplitLabel
    count: int
    fraction: float


@dataclass(frozen=True)
class ScaffoldSplitResult:
    assignments: list[ScaffoldSplitAssignment]
    summaries: list[ScaffoldSplitSummary]
    scaffold_kind: str


def split_by_scaffold(
    smiles_list: list[str],
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    scaffold_kind: str = "murcko",
    random_seed: int = 42,
) -> ScaffoldSplitResult:
    if not smiles_list:
        return ScaffoldSplitResult(assignments=[], summaries=[], scaffold_kind=scaffold_kind)

    total_fraction = train_fraction + validation_fraction + test_fraction
    if total_fraction <= 0:
        raise ValueError("At least one split fraction must be > 0.")

    train_fraction /= total_fraction
    validation_fraction /= total_fraction
    test_fraction /= total_fraction

    analysis = analyze_scaffolds(smiles_list)
    scaffold_by_index: dict[int, tuple[str, str]] = {}
    groups: dict[str, list[int]] = defaultdict(list)
    valid_indices: list[int] = []

    for annotation in analysis.annotations:
        if annotation.status == "invalid":
            scaffold_by_index[annotation.index] = (annotation.status, "")
            continue

        scaffold = annotation.murcko if scaffold_kind == "murcko" else annotation.generic
        scaffold = scaffold or NO_SCAFFOLD_LABEL
        scaffold_by_index[annotation.index] = (annotation.status, scaffold)
        groups[scaffold].append(annotation.index)
        valid_indices.append(annotation.index)

    rng = random.Random(random_seed)
    grouped = list(groups.items())
    rng.shuffle(grouped)
    grouped.sort(key=lambda item: len(item[1]), reverse=True)

    n_valid = len(valid_indices)
    targets = {
        "train": train_fraction * n_valid,
        "validation": validation_fraction * n_valid,
        "test": test_fraction * n_valid,
    }
    split_members = {"train": [], "validation": [], "test": []}

    positive_splits = [split for split, frac in (("train", train_fraction), ("validation", validation_fraction), ("test", test_fraction)) if frac > 0]
    seed_groups = min(len(grouped), len(positive_splits))
    for seed_index in range(seed_groups):
        split = positive_splits[seed_index]
        _scaffold, members = grouped[seed_index]
        split_members[split].extend(members)

    for scaffold, members in grouped[seed_groups:]:
        deficits = {
            split: targets[split] - len(split_members[split])
            for split in ("train", "validation", "test")
        }
        chosen = max(deficits, key=lambda split: (deficits[split], split == "train"))
        split_members[chosen].extend(members)

    assignments: list[ScaffoldSplitAssignment] = []
    split_lookup = {}
    for split, members in split_members.items():
        for index in members:
            split_lookup[index] = split

    for index, smiles_value in enumerate(smiles_list):
        status, scaffold = scaffold_by_index.get(index, ("invalid", ""))
        if status == "invalid":
            assignments.append(
                ScaffoldSplitAssignment(index=index, split="invalid", scaffold=None, status="invalid")
            )
            continue
        split = split_lookup.get(index, "test")
        assignments.append(
            ScaffoldSplitAssignment(index=index, split=split, scaffold=scaffold, status=status)
        )

    summaries = []
    total = len(smiles_list)
    for split in ("train", "validation", "test", "invalid"):
        count = sum(1 for assignment in assignments if assignment.split == split)
        summaries.append(
            ScaffoldSplitSummary(
                split=split,
                count=count,
                fraction=0.0 if total == 0 else round(count / total, 4),
            )
        )

    return ScaffoldSplitResult(assignments=assignments, summaries=summaries, scaffold_kind=scaffold_kind)
