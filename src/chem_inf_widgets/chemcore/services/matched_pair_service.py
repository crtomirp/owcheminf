from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdFMCS

from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles


@dataclass(frozen=True)
class MatchedPairRow:
    index_a: int
    index_b: int
    smiles_a: str
    smiles_b: str
    transformation: str
    shared_heavy_atoms: int
    delta_property: Optional[float]


def _parse_mol(smiles: str):
    return safe_mol_from_smiles((smiles or "").strip(), sanitize=True, remove_hs=True).mol


def _fragment_from_match(mol: Chem.Mol, match_atoms: tuple[int, ...]) -> str:
    diff_atoms = sorted(set(range(mol.GetNumAtoms())) - set(match_atoms))
    if not diff_atoms:
        return "H"
    try:
        fragment = Chem.MolFragmentToSmiles(mol, atomsToUse=diff_atoms, canonical=True)
    except Exception:
        return "?"
    return fragment or "?"


def find_matched_pairs(
    smiles_list: list[str],
    property_values: Optional[list[Optional[float]]] = None,
    *,
    min_shared_atoms: int = 4,
    max_pairs: int = 500,
) -> list[MatchedPairRow]:
    if len(smiles_list) < 2:
        return []

    mols = [(_parse_mol(smiles), (smiles or "").strip()) for smiles in smiles_list]
    rows: list[MatchedPairRow] = []
    for index_a in range(len(mols)):
        mol_a, smiles_a = mols[index_a]
        if mol_a is None:
            continue
        for index_b in range(index_a + 1, len(mols)):
            mol_b, smiles_b = mols[index_b]
            if mol_b is None:
                continue
            try:
                mcs = rdFMCS.FindMCS(
                    [mol_a, mol_b],
                    completeRingsOnly=True,
                    ringMatchesRingOnly=True,
                    timeout=2,
                )
            except Exception:
                continue
            if mcs.numAtoms < min_shared_atoms or not mcs.smartsString:
                continue
            core = Chem.MolFromSmarts(mcs.smartsString)
            if core is None:
                continue
            match_a = mol_a.GetSubstructMatch(core)
            match_b = mol_b.GetSubstructMatch(core)
            if not match_a or not match_b:
                continue

            frag_a = _fragment_from_match(mol_a, match_a)
            frag_b = _fragment_from_match(mol_b, match_b)
            if frag_a == frag_b:
                continue

            delta_property = None
            if property_values is not None:
                value_a = property_values[index_a]
                value_b = property_values[index_b]
                if value_a is not None and value_b is not None:
                    delta_property = round(float(value_b) - float(value_a), 4)

            rows.append(
                MatchedPairRow(
                    index_a=index_a,
                    index_b=index_b,
                    smiles_a=smiles_a,
                    smiles_b=smiles_b,
                    transformation=f"{frag_a} -> {frag_b}",
                    shared_heavy_atoms=mcs.numAtoms,
                    delta_property=delta_property,
                )
            )
            if len(rows) >= max_pairs:
                return rows
    rows.sort(key=lambda row: (-row.shared_heavy_atoms, abs(row.delta_property or 0.0)))
    return rows
