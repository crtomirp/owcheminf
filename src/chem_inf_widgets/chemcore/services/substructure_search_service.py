from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from chem_inf_widgets.chemcore.services.rdkit_safe import safe_canonical_smiles, safe_mol_from_smiles


SEARCH_TYPES = ("substructure", "superstructure", "similarity", "exact")
FP_TYPES = ("morgan", "rdkit")


@dataclass(frozen=True)
class SearchConfig:
    search_type: str = "substructure"  # substructure | superstructure | similarity | exact
    similarity_threshold: float = 0.3

    fp_type: str = "morgan"
    morgan_radius: int = 2
    morgan_nbits: int = 2048

    # if True, for substructure will return atom indices to highlight
    return_highlight_atoms: bool = True


@dataclass(frozen=True)
class SearchHit:
    idx: int
    highlighted_atoms_csv: str
    similarity: float  # NaN if not applicable


def canonical_smiles_no_h(smiles: str) -> str:
    s = (smiles or "").strip()
    if not s:
        return ""
    parsed = safe_mol_from_smiles(s)
    if parsed.mol is None:
        return s
    return parsed.canonical_smiles or s


def looks_like_smarts(q: str) -> bool:
    """
    Conservative SMARTS detector.
    Only triggers on tokens that are very uncommon in plain SMILES.
    """
    s = (q or "").strip()
    if not s:
        return False

    tokens = ["[#", ";", "!", "&", ",", "~", "@", "*", "$("]
    if any(t in s for t in tokens):
        return True

    # ':' is used for atom mapping in SMILES too, so do NOT trigger on ':' alone.
    return False


def normalize_query_string(query: str, search_type: str) -> str:
    """
    Normalize query for stable behavior:
    - sub/super: keep as-is (SMARTS must not be canonicalized)
    - similarity/exact: canonicalize SMILES without explicit H
    """
    q = (query or "").strip()
    if not q:
        return ""
    if search_type in ("similarity", "exact"):
        return canonical_smiles_no_h(q)
    return q


def parse_query_mol_auto(query: str, search_type: str) -> Chem.Mol:
    """
    For sub/superstructure:
      - auto-detect SMARTS; else try SMILES; if SMILES fails, fallback SMARTS
    For similarity/exact:
      - SMILES only
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("Empty query.")

    if search_type in ("similarity", "exact"):
        parsed = safe_mol_from_smiles(q)
        mol = parsed.mol
        if mol is None:
            raise ValueError("Invalid SMILES query.")
        mol.UpdatePropertyCache(False)
        _ = mol.GetRingInfo()
        return mol

    # substructure / superstructure
    if looks_like_smarts(q):
        mol = Chem.MolFromSmarts(q)
        if mol is None:
            raise ValueError("Invalid SMARTS query.")
        mol.UpdatePropertyCache(False)
        _ = mol.GetRingInfo()
        return mol

    # try SMILES first
    parsed = safe_mol_from_smiles(q)
    mol = parsed.mol
    if mol is not None:
        mol.UpdatePropertyCache(False)
        _ = mol.GetRingInfo()
        return mol

    # fallback: try SMARTS
    mol = Chem.MolFromSmarts(q)
    if mol is None:
        raise ValueError("Invalid query (neither SMILES nor SMARTS).")
    mol.UpdatePropertyCache(False)
    _ = mol.GetRingInfo()
    return mol


def _fingerprint(mol: Chem.Mol, cfg: SearchConfig):
    if cfg.fp_type == "rdkit":
        return Chem.RDKFingerprint(mol)
    return AllChem.GetMorganFingerprintAsBitVect(mol, cfg.morgan_radius, nBits=cfg.morgan_nbits)


def _match_one(mol: Chem.Mol, query_mol: Chem.Mol, cfg: SearchConfig) -> Tuple[bool, str, float]:
    st = cfg.search_type

    if st == "substructure":
        if not mol.HasSubstructMatch(query_mol):
            return False, "", float("nan")
        if cfg.return_highlight_atoms:
            match = mol.GetSubstructMatch(query_mol)
            return True, ",".join(map(str, match)), float("nan")
        return True, "", float("nan")

    if st == "superstructure":
        ok = query_mol.HasSubstructMatch(mol)
        return ok, "", float("nan")

    if st == "exact":
        sm1 = safe_canonical_smiles(mol)
        sm2 = safe_canonical_smiles(query_mol)
        return (sm1 == sm2), "", float("nan")

    if st == "similarity":
        fp1 = _fingerprint(mol, cfg)
        fp2 = _fingerprint(query_mol, cfg)
        sim = float(DataStructs.TanimotoSimilarity(fp1, fp2))
        return sim >= float(cfg.similarity_threshold), "", sim

    raise ValueError(f"Unsupported search_type: {st}")


def search_smiles(
    smiles_list: Sequence[str],
    query: str,
    cfg: SearchConfig,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[SearchHit]:
    if cfg.search_type not in SEARCH_TYPES:
        raise ValueError(f"search_type must be one of: {SEARCH_TYPES}")

    query_norm = normalize_query_string(query, cfg.search_type)
    query_mol = parse_query_mol_auto(query_norm, cfg.search_type)

    hits: List[SearchHit] = []
    n = len(smiles_list)

    for i, smi in enumerate(smiles_list):
        if progress_cb:
            progress_cb(i + 1, n)

        smi0 = (str(smi) if smi is not None else "").strip()
        if not smi0:
            continue

        parsed = safe_mol_from_smiles(smi0)
        mol = parsed.mol
        if mol is None:
            continue

        ok, atoms_csv, sim = _match_one(mol, query_mol, cfg)
        if ok:
            hits.append(SearchHit(idx=i, highlighted_atoms_csv=atoms_csv, similarity=sim))

    return hits
 
