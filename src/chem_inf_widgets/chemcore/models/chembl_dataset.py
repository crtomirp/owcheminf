from dataclasses import dataclass
from typing import List, Dict, Any
from ..mol import ChemMol


@dataclass
class ChemBLDataset:
    mols: List[ChemMol]
    props: List[Dict[str, Any]]

    def __len__(self):
        return len(self.mols)

