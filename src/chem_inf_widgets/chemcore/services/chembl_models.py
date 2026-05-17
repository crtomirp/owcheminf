from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ChemBLTargetRecord:
    chembl_id: str
    pref_name: str
    organism: str
    target_type: str


@dataclass(frozen=True)
class ChemBLAssayRecord:
    assay_chembl_id: str
    description: str
    assay_type: str
    confidence_score: Optional[int]
    organism: str


@dataclass(frozen=True)
class ChemBLBioactivityRecord:
    molecule_chembl_id: str
    target_chembl_id: str
    smiles: str

    # activity fields (may be missing depending on record)
    standard_type: str
    standard_value: Optional[float]
    standard_units: str
    pchembl_value: Optional[float]

    # optional “normalized” convenience (only if type matches)
    ic50_nM: Optional[float]


@dataclass(frozen=True)
class ChemBLMoleculeRecord:
    chembl_id: str
    pref_name: str
    canonical_smiles: str
