from typing import List
from ..models.chembl_dataset import ChemBLDataset
from ..mol import ChemMol
from ..models.chembl_record import ChemBLRecord



def records_to_dataset(records: list[ChemBLRecord]) -> ChemBLDataset:
    mols = []
    props = []

    for r in records:
        if not r.smiles:
            continue

        mol = ChemMol.from_smiles(r.smiles)
        mols.append(mol)

        props.append({
            "ChEMBL ID": r.chembl_id,
            "pChEMBL": r.pchembl,
            "IC50_nM": r.ic50_nM,
            "Target ChEMBL ID": r.target_chembl_id,
        })

    return ChemBLDataset(mols=mols, props=props)


