from dataclasses import dataclass
from typing import Optional


@dataclass
class ChemBLRecord:
    chembl_id: str
    smiles: str
    pchembl: Optional[float]
    ic50_nM: Optional[float]
    target_chembl_id: Optional[str]

