from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from chem_inf_widgets.chemcore.services.rdkit_safe import safe_canonical_smiles, safe_mol_from_smiles


_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
NO_SCAFFOLD_LABEL = "__no_scaffold__"


@dataclass(frozen=True)
class ActivityCliffPair:
    index_a: int
    index_b: int
    smiles_a: str
    smiles_b: str
    name_a: str
    name_b: str
    activity_a: float
    activity_b: float
    similarity: float
    activity_ratio: float
    cliff_score: float
    higher_active: str


@dataclass(frozen=True)
class ScaffoldActivitySummaryRow:
    scaffold: str
    count: int
    mean_activity: float
    best_activity: float
    worst_activity: float
    std_activity: float


@dataclass(frozen=True)
class ActivityCliffResult:
    pairs: list[ActivityCliffPair]
    valid_indices: list[int]
    failed_indices: list[int]
    unique_cliff_indices: list[int]


def _parse_mol(smiles: str) -> Optional[Chem.Mol]:
    return safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol


def _normalize_names(names: Optional[list[str]], n_rows: int) -> list[str]:
    if names is None:
        return [""] * n_rows
    out = list(names[:n_rows])
    if len(out) < n_rows:
        out.extend([""] * (n_rows - len(out)))
    return ["" if value is None else str(value).strip() for value in out]


def find_activity_cliffs(
    smiles_list: list[str],
    activities: list[float],
    *,
    names: Optional[list[str]] = None,
    similarity_threshold: float = 0.6,
    activity_fold_threshold: float = 10.0,
    activity_log_scale: bool = False,
    max_pairs: int = 500,
) -> ActivityCliffResult:
    if len(smiles_list) != len(activities):
        raise ValueError("SMILES and activity lists must have the same length.")

    norm_names = _normalize_names(names, len(smiles_list))
    valid: list[tuple[int, str, float, str, object]] = []
    failed_indices: list[int] = []

    for index, (smiles, activity, name) in enumerate(zip(smiles_list, activities, norm_names)):
        clean_smiles = (smiles or "").strip()
        try:
            activity_value = float(activity)
        except (TypeError, ValueError):
            failed_indices.append(index)
            continue
        if not math.isfinite(activity_value) or not clean_smiles:
            failed_indices.append(index)
            continue

        mol = _parse_mol(clean_smiles)
        if mol is None:
            failed_indices.append(index)
            continue

        valid.append((index, clean_smiles, activity_value, name, _MORGAN_GEN.GetFingerprint(mol)))

    if len(valid) < 2:
        return ActivityCliffResult(pairs=[], valid_indices=[row[0] for row in valid], failed_indices=failed_indices, unique_cliff_indices=[])

    pairs: list[ActivityCliffPair] = []
    cliff_index_counter: Counter[int] = Counter()

    for i in range(len(valid)):
        index_a, smiles_a, activity_a, name_a, fp_a = valid[i]
        trailing_fps = [valid[j][4] for j in range(i + 1, len(valid))]
        if not trailing_fps:
            break
        sims = DataStructs.BulkTanimotoSimilarity(fp_a, trailing_fps)

        for offset, similarity in enumerate(sims, start=i + 1):
            if similarity < float(similarity_threshold):
                continue

            index_b, smiles_b, activity_b, name_b, _fp_b = valid[offset]
            if activity_log_scale:
                delta = abs(activity_a - activity_b)
                threshold = math.log10(max(float(activity_fold_threshold), 1.0000001))
                if delta < threshold:
                    continue
                activity_ratio = round(delta, 4)
                cliff_score = round(float(similarity) * delta, 4)
                higher_active = "a" if activity_a > activity_b else "b"
            else:
                if activity_a <= 0 or activity_b <= 0:
                    continue
                ratio = max(activity_a, activity_b) / min(activity_a, activity_b)
                if ratio < float(activity_fold_threshold):
                    continue
                activity_ratio = round(ratio, 4)
                cliff_score = round(float(similarity) * math.log10(ratio), 4)
                higher_active = "a" if activity_a < activity_b else "b"

            pairs.append(
                ActivityCliffPair(
                    index_a=index_a,
                    index_b=index_b,
                    smiles_a=smiles_a,
                    smiles_b=smiles_b,
                    name_a=name_a,
                    name_b=name_b,
                    activity_a=activity_a,
                    activity_b=activity_b,
                    similarity=round(float(similarity), 4),
                    activity_ratio=activity_ratio,
                    cliff_score=cliff_score,
                    higher_active=higher_active,
                )
            )
            cliff_index_counter[index_a] += 1
            cliff_index_counter[index_b] += 1

    pairs.sort(key=lambda pair: (-pair.cliff_score, -pair.similarity, pair.index_a, pair.index_b))
    top_pairs = pairs[: max(int(max_pairs), 0)]
    unique_cliff_indices = sorted({pair.index_a for pair in top_pairs} | {pair.index_b for pair in top_pairs})

    return ActivityCliffResult(
        pairs=top_pairs,
        valid_indices=[row[0] for row in valid],
        failed_indices=sorted(set(failed_indices)),
        unique_cliff_indices=unique_cliff_indices,
    )


def scaffold_activity_summary(
    smiles_list: list[str],
    activities: list[float],
    *,
    activity_log_scale: bool = False,
) -> list[ScaffoldActivitySummaryRow]:
    if len(smiles_list) != len(activities):
        raise ValueError("SMILES and activity lists must have the same length.")

    groups: dict[str, list[float]] = {}
    for smiles, activity in zip(smiles_list, activities):
        clean_smiles = (smiles or "").strip()
        try:
            activity_value = float(activity)
        except (TypeError, ValueError):
            continue
        if not clean_smiles or not math.isfinite(activity_value):
            continue

        mol = _parse_mol(clean_smiles)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        except ValueError:
            scaffold = None
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            key = NO_SCAFFOLD_LABEL
        else:
            key = safe_canonical_smiles(scaffold, remove_hs=False) or NO_SCAFFOLD_LABEL
        groups.setdefault(key, []).append(activity_value)

    rows: list[ScaffoldActivitySummaryRow] = []
    for scaffold, values in groups.items():
        best_activity = max(values) if activity_log_scale else min(values)
        worst_activity = min(values) if activity_log_scale else max(values)
        rows.append(
            ScaffoldActivitySummaryRow(
                scaffold=scaffold,
                count=len(values),
                mean_activity=round(statistics.mean(values), 4),
                best_activity=round(best_activity, 4),
                worst_activity=round(worst_activity, 4),
                std_activity=round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            )
        )

    rows.sort(key=lambda row: (-row.best_activity, -row.count, row.scaffold) if activity_log_scale else (row.best_activity, -row.count, row.scaffold))
    return rows
