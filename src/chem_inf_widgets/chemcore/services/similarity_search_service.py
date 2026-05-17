from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint

from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles


@dataclass(frozen=True)
class SimilarityHit:
    query_index: int
    reference_index: int
    query_smiles: str
    reference_smiles: str
    similarity: float


def _parse_mol(smiles: str) -> Optional[Chem.Mol]:
    return safe_mol_from_smiles((smiles or "").strip(), sanitize=True, remove_hs=True).mol


def _fingerprint(mol: Chem.Mol, fp_type: str, radius: int, n_bits: int):
    fp_type = (fp_type or "morgan").lower()
    if fp_type == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    if fp_type == "rdkit":
        return RDKFingerprint(mol, fpSize=n_bits)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def find_similarity_hits(
    query_smiles: list[str],
    reference_smiles: list[str],
    *,
    top_k: int = 5,
    min_similarity: float = 0.0,
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
    include_self: bool = False,
) -> list[SimilarityHit]:
    if not query_smiles or not reference_smiles:
        return []

    query_mols = [(_parse_mol(smiles), (smiles or "").strip()) for smiles in query_smiles]
    ref_mols = [(_parse_mol(smiles), (smiles or "").strip()) for smiles in reference_smiles]

    ref_fps = []
    for mol, smiles_value in ref_mols:
        if mol is None:
            ref_fps.append(None)
            continue
        ref_fps.append(_fingerprint(mol, fp_type, radius, n_bits))

    hits: list[SimilarityHit] = []
    for q_idx, (q_mol, q_smiles) in enumerate(query_mols):
        if q_mol is None:
            continue
        q_fp = _fingerprint(q_mol, fp_type, radius, n_bits)
        scores = []
        for r_idx, (r_mol, r_smiles) in enumerate(ref_mols):
            r_fp = ref_fps[r_idx]
            if r_mol is None or r_fp is None:
                continue
            if not include_self and q_smiles and q_smiles == r_smiles:
                continue
            score = float(DataStructs.TanimotoSimilarity(q_fp, r_fp))
            if score >= min_similarity:
                scores.append((score, r_idx, r_smiles))
        scores.sort(key=lambda item: (-item[0], item[1]))
        for score, r_idx, r_smiles in scores[: max(1, int(top_k))]:
            hits.append(
                SimilarityHit(
                    query_index=q_idx,
                    reference_index=r_idx,
                    query_smiles=q_smiles,
                    reference_smiles=r_smiles,
                    similarity=round(score, 4),
                )
            )
    return hits
