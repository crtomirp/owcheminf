from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional

from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles


DiversityMethod = Literal["maxmin", "sphere_exclusion", "butina"]

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass(frozen=True)
class DiversityMetrics:
    n_compounds: int
    mean_nn_distance: float
    mean_pairwise_dist: float
    n_singletons: int
    diversity_score: float


@dataclass(frozen=True)
class DiversitySelectionResult:
    method: DiversityMethod
    selected_indices: list[int]
    valid_indices: list[int]
    failed_indices: list[int]
    metrics_input: DiversityMetrics
    metrics_selected: DiversityMetrics


def _compute_fps(smiles_list: list[str]) -> tuple[list, list[int]]:
    fps = []
    valid_indices = []
    for index, smiles in enumerate(smiles_list):
        mol = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol
        if mol is None:
            continue
        fps.append(_MORGAN_GEN.GetFingerprint(mol))
        valid_indices.append(index)
    return fps, valid_indices


def maxmin_selection(
    smiles_list: list[str],
    n_select: int,
    seed_idx: int = 0,
    random_seed: int = 42,
) -> list[int]:
    fps, valid_indices = _compute_fps(smiles_list)
    n_fps = len(fps)
    if n_fps == 0:
        return []

    n_select = min(max(1, int(n_select)), n_fps)
    rng = random.Random(random_seed)
    if seed_idx < 0 or seed_idx >= n_fps:
        seed_idx = rng.randint(0, n_fps - 1)

    seed_sims = DataStructs.BulkTanimotoSimilarity(fps[seed_idx], fps)
    min_dist = [1.0 - similarity for similarity in seed_sims]
    min_dist[seed_idx] = -1.0
    selected = [seed_idx]

    while len(selected) < n_select:
        next_idx = max(range(n_fps), key=lambda idx: min_dist[idx])
        if min_dist[next_idx] < 0:
            break

        selected.append(next_idx)
        min_dist[next_idx] = -1.0
        new_sims = DataStructs.BulkTanimotoSimilarity(fps[next_idx], fps)
        for idx in range(n_fps):
            if min_dist[idx] < 0:
                continue
            new_dist = 1.0 - new_sims[idx]
            if new_dist < min_dist[idx]:
                min_dist[idx] = new_dist

    return [valid_indices[idx] for idx in selected]


def sphere_exclusion(
    smiles_list: list[str],
    radius: float = 0.35,
    random_seed: int = 42,
) -> list[int]:
    fps, valid_indices = _compute_fps(smiles_list)
    n_fps = len(fps)
    if n_fps == 0:
        return []

    rng = random.Random(random_seed)
    order = list(range(n_fps))
    rng.shuffle(order)

    similarity_threshold = 1.0 - float(radius)
    excluded = set()
    selected = []

    for idx in order:
        if idx in excluded:
            continue
        selected.append(idx)
        sims = DataStructs.BulkTanimotoSimilarity(fps[idx], fps)
        for other_idx, similarity in enumerate(sims):
            if other_idx != idx and similarity >= similarity_threshold:
                excluded.add(other_idx)

    return [valid_indices[idx] for idx in selected]


def butina_cluster_selection(
    smiles_list: list[str],
    n_clusters: int = 10,
    threshold: float = 0.4,
) -> list[int]:
    fps, valid_indices = _compute_fps(smiles_list)
    n_fps = len(fps)
    if n_fps == 0:
        return []

    n_clusters = min(max(1, int(n_clusters)), n_fps)
    dists = []
    for idx in range(1, n_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[idx], fps[:idx])
        dists.extend(1.0 - similarity for similarity in sims)

    clusters = Butina.ClusterData(dists, n_fps, float(threshold), isDistData=True)
    sorted_clusters = sorted(clusters, key=len, reverse=True)

    selected = []
    for cluster in sorted_clusters[:n_clusters]:
        if cluster:
            selected.append(cluster[0])
    return [valid_indices[idx] for idx in selected]


def diversity_metrics(
    smiles_list: list[str],
    sample_size: int = 500,
    random_seed: int = 42,
) -> DiversityMetrics:
    fps, _valid_indices = _compute_fps(smiles_list)
    n_fps = len(fps)
    if n_fps < 2:
        return DiversityMetrics(
            n_compounds=n_fps,
            mean_nn_distance=1.0,
            mean_pairwise_dist=1.0,
            n_singletons=n_fps,
            diversity_score=1.0,
        )

    rng = random.Random(random_seed)
    fps_sample = fps
    if n_fps > int(sample_size):
        sample_indices = rng.sample(range(n_fps), int(sample_size))
        fps_sample = [fps[idx] for idx in sample_indices]

    nn_distances = []
    pairwise_sample = []
    for idx, fp in enumerate(fps_sample):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_sample)
        sims_without_self = [similarity for j, similarity in enumerate(sims) if j != idx]
        nn_sim = max(sims_without_self) if sims_without_self else 0.0
        nn_distances.append(1.0 - nn_sim)
        pairwise_sample.extend(1.0 - similarity for similarity in sims_without_self[:20])

    mean_nn = sum(nn_distances) / len(nn_distances)
    mean_pairwise = sum(pairwise_sample) / len(pairwise_sample) if pairwise_sample else 0.0
    n_singletons = sum(1 for distance in nn_distances if distance > 0.5)

    return DiversityMetrics(
        n_compounds=n_fps,
        mean_nn_distance=round(mean_nn, 4),
        mean_pairwise_dist=round(mean_pairwise, 4),
        n_singletons=n_singletons,
        diversity_score=round(mean_pairwise, 4),
    )


def select_diverse_subset(
    smiles_list: list[str],
    *,
    method: DiversityMethod = "maxmin",
    n_select: int = 25,
    seed_idx: int = 0,
    radius: float = 0.35,
    n_clusters: Optional[int] = None,
    threshold: float = 0.4,
    random_seed: int = 42,
) -> DiversitySelectionResult:
    method_name = method.strip().lower()
    metrics_input = diversity_metrics(smiles_list, random_seed=random_seed)
    fps, valid_indices = _compute_fps(smiles_list)
    failed_indices = [idx for idx in range(len(smiles_list)) if idx not in set(valid_indices)]

    if method_name == "maxmin":
        selected_indices = maxmin_selection(
            smiles_list,
            n_select=n_select,
            seed_idx=seed_idx,
            random_seed=random_seed,
        )
    elif method_name == "sphere_exclusion":
        selected_indices = sphere_exclusion(
            smiles_list,
            radius=radius,
            random_seed=random_seed,
        )
    elif method_name == "butina":
        selected_indices = butina_cluster_selection(
            smiles_list,
            n_clusters=n_clusters if n_clusters is not None else n_select,
            threshold=threshold,
        )
    else:
        raise ValueError(f"Unsupported diversity method: {method!r}")

    selected_smiles = [smiles_list[idx] for idx in selected_indices]
    metrics_selected = diversity_metrics(selected_smiles, random_seed=random_seed)
    return DiversitySelectionResult(
        method=method_name,  # type: ignore[arg-type]
        selected_indices=selected_indices,
        valid_indices=valid_indices,
        failed_indices=failed_indices,
        metrics_input=metrics_input,
        metrics_selected=metrics_selected,
    )
