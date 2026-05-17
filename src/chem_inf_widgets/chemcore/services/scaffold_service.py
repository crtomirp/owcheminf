from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from chem_inf_widgets.chemcore.services.rdkit_safe import safe_canonical_smiles, safe_mol_from_smiles


ScaffoldKind = Literal["murcko", "generic"]
NO_SCAFFOLD_LABEL = "__no_scaffold__"


@dataclass(frozen=True)
class ScaffoldAnnotation:
    index: int
    smiles: str
    murcko: Optional[str]
    generic: Optional[str]
    status: str


@dataclass(frozen=True)
class ScaffoldSummaryRow:
    scaffold: str
    kind: ScaffoldKind
    count: int
    fraction: float


@dataclass(frozen=True)
class ScaffoldAnalysisResult:
    annotations: list[ScaffoldAnnotation]
    failed_indices: list[int]
    valid_count: int
    murcko_counts: list[tuple[str, int]]
    generic_counts: list[tuple[str, int]]


def _parse_mol(smiles: str) -> Optional[Chem.Mol]:
    return safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol


def _murcko_from_mol(mol: Chem.Mol) -> Optional[str]:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except ValueError:
        return None
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    smiles = safe_canonical_smiles(scaffold, remove_hs=False)
    return smiles or None


def _generic_from_mol(mol: Chem.Mol) -> Optional[str]:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except ValueError:
        return None
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    try:
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    except ValueError:
        return None
    if generic is None or generic.GetNumAtoms() == 0:
        return None
    smiles = safe_canonical_smiles(generic, remove_hs=False)
    return smiles or None


def get_murcko_scaffold(smiles: str) -> Optional[str]:
    mol = _parse_mol((smiles or "").strip())
    if mol is None:
        return None
    return _murcko_from_mol(mol)


def get_generic_scaffold(smiles: str) -> Optional[str]:
    mol = _parse_mol((smiles or "").strip())
    if mol is None:
        return None
    return _generic_from_mol(mol)


def _sort_counts(counts: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def analyze_scaffolds(
    smiles_list: list[str],
    *,
    no_scaffold_label: str = NO_SCAFFOLD_LABEL,
) -> ScaffoldAnalysisResult:
    annotations: list[ScaffoldAnnotation] = []
    failed_indices: list[int] = []
    murcko_counts: Counter[str] = Counter()
    generic_counts: Counter[str] = Counter()
    valid_count = 0

    for index, value in enumerate(smiles_list):
        smiles = (value or "").strip()
        mol = _parse_mol(smiles) if smiles else None
        if mol is None:
            annotations.append(
                ScaffoldAnnotation(
                    index=index,
                    smiles=smiles,
                    murcko=None,
                    generic=None,
                    status="invalid",
                )
            )
            failed_indices.append(index)
            continue

        valid_count += 1
        murcko = _murcko_from_mol(mol)
        generic = _generic_from_mol(mol)

        if murcko is None:
            annotations.append(
                ScaffoldAnnotation(
                    index=index,
                    smiles=smiles,
                    murcko=no_scaffold_label,
                    generic=no_scaffold_label,
                    status="acyclic",
                )
            )
            murcko_counts[no_scaffold_label] += 1
            generic_counts[no_scaffold_label] += 1
            continue

        generic_key = generic or no_scaffold_label
        annotations.append(
            ScaffoldAnnotation(
                index=index,
                smiles=smiles,
                murcko=murcko,
                generic=generic_key,
                status="ok",
            )
        )
        murcko_counts[murcko] += 1
        generic_counts[generic_key] += 1

    return ScaffoldAnalysisResult(
        annotations=annotations,
        failed_indices=failed_indices,
        valid_count=valid_count,
        murcko_counts=_sort_counts(murcko_counts),
        generic_counts=_sort_counts(generic_counts),
    )


def build_scaffold_summary(
    result: ScaffoldAnalysisResult,
    *,
    kind: ScaffoldKind = "murcko",
    include_acyclic: bool = True,
    top_n: Optional[int] = None,
    no_scaffold_label: str = NO_SCAFFOLD_LABEL,
) -> list[ScaffoldSummaryRow]:
    counts = result.murcko_counts if kind == "murcko" else result.generic_counts
    summary_rows: list[ScaffoldSummaryRow] = []
    for scaffold, count in counts:
        if not include_acyclic and scaffold == no_scaffold_label:
            continue
        summary_rows.append(
            ScaffoldSummaryRow(
                scaffold=scaffold,
                kind=kind,
                count=count,
                fraction=0.0 if result.valid_count == 0 else round(count / result.valid_count, 4),
            )
        )

    if top_n is not None and top_n > 0:
        return summary_rows[:top_n]
    return summary_rows
