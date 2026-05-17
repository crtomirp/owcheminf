import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.chembl_client import ChEMBLClient  # noqa: E402


class ChemblClientTests(unittest.TestCase):
    @patch("chem_inf_widgets.chemcore.services.chembl_client.requests.get")
    def test_fetch_bioactivities_canonicalizes_smiles(self, mock_get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "activities": [
                {
                    "molecule_chembl_id": "CHEMBL1",
                    "canonical_smiles": "C(C)O",
                    "pchembl_value": 6.5,
                    "standard_value": 120.0,
                    "target_chembl_id": "CHEMBLT1",
                },
                {
                    "molecule_chembl_id": "CHEMBL2",
                    "canonical_smiles": "not_a_smiles",
                    "pchembl_value": 5.0,
                    "standard_value": 500.0,
                    "target_chembl_id": "CHEMBLT1",
                },
            ],
            "page_meta": {"next": None},
        }
        mock_get.return_value = response

        records = ChEMBLClient().fetch_bioactivities("CHEMBLT1")

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].smiles, "CCO")
        self.assertEqual(records[1].smiles, "")


if __name__ == "__main__":
    unittest.main()
