import requests
from typing import List
from .rdkit_utils import canonical_smiles
from ..models.chembl_record import ChemBLRecord


class ChEMBLClient:
    BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

    def fetch_bioactivities(self, target_chembl_id: str) -> List[ChemBLRecord]:
        records = []
        url = self.BASE
        params = {
            "target_chembl_id": target_chembl_id,
            "limit": 1000,
        }

        while url:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            for row in data.get("activities", []):
                smi = canonical_smiles(row.get("canonical_smiles", ""))

                records.append(
                    ChemBLRecord(
                        chembl_id=row.get("molecule_chembl_id"),
                        smiles=smi,
                        pchembl=row.get("pchembl_value"),
                        ic50_nM=row.get("standard_value"),
                        target_chembl_id=row.get("target_chembl_id"),
                    )
                )

            url = data.get("page_meta", {}).get("next")

        return records
