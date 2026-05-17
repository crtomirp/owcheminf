from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition

from chem_inf_widgets.chemcore.services.rdkit_safe import (
    safe_canonical_smiles,
    safe_mol_from_smiles,
)
from chem_inf_widgets.chemcore.services.scaffold_service import get_murcko_scaffold


@dataclass(frozen=True)
class RGroupRow:
    index: int
    core: str
    groups: dict[str, str]


@dataclass(frozen=True)
class RGroupDecompositionResult:
    core: str
    rows: list[RGroupRow]
    matched_indices: list[int]
    unmatched_indices: list[int]
    group_labels: list[str]


def _parse_mol(smiles: str) -> Optional[Chem.Mol]:
    return safe_mol_from_smiles((smiles or "").strip(), sanitize=True, remove_hs=True).mol


def _choose_auto_core(smiles_list: list[str]) -> Optional[str]:
    scaffolds = [get_murcko_scaffold(smiles) for smiles in smiles_list]
    counts = Counter(scaffold for scaffold in scaffolds if scaffold)
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _mol_to_smiles_or_empty(mol) -> str:
    return safe_canonical_smiles(mol, remove_hs=False)


def decompose_rgroups(
    smiles_list: list[str],
    *,
    core_smarts: Optional[str] = None,
) -> RGroupDecompositionResult:
    core_text = (core_smarts or "").strip() or _choose_auto_core(smiles_list)
    if not core_text:
        raise ValueError("Could not infer a common core. Provide a core SMARTS/SMILES.")

    core_mol = Chem.MolFromSmarts(core_text)
    if core_mol is None:
        core_mol = safe_mol_from_smiles(core_text, sanitize=True, remove_hs=True).mol
    if core_mol is None:
        raise ValueError("Invalid core SMARTS/SMILES.")

    mols = []
    valid_map = []
    for index, smiles in enumerate(smiles_list):
        mol = _parse_mol(smiles)
        if mol is not None:
            mols.append(mol)
            valid_map.append(index)

    if not mols:
        raise ValueError("No valid molecules available for decomposition.")

    params = rdRGroupDecomposition.RGroupDecompositionParameters()
    params.removeAllHydrogenRGroups = True
    params.removeHydrogensPostMatch = True
    rows_raw, unmatched_raw = rdRGroupDecomposition.RGroupDecompose([core_mol], mols, True, True, params)
    rows_raw = rows_raw or []
    unmatched_raw = list(unmatched_raw or [])

    rows: list[RGroupRow] = []
    group_labels = set()
    matched_indices = []
    for local_index, row in enumerate(rows_raw):
        global_index = valid_map[local_index]
        groups = {}
        for key, value in row.items():
            if key == "Core":
                continue
            group_labels.add(key)
            groups[key] = _mol_to_smiles_or_empty(value)
        rows.append(
            RGroupRow(
                index=global_index,
                core=_mol_to_smiles_or_empty(row.get("Core")) or core_text,
                groups=groups,
            )
        )
        matched_indices.append(global_index)

    unmatched_indices = sorted({valid_map[idx] for idx in unmatched_raw if idx < len(valid_map)})
    return RGroupDecompositionResult(
        core=core_text,
        rows=rows,
        matched_indices=sorted(matched_indices),
        unmatched_indices=unmatched_indices,
        group_labels=sorted(group_labels),
    )
