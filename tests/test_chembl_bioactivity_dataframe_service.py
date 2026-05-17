import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.chembl_bioactivity_dataframe_service import (  # noqa: E402
    calculate_drug_properties,
    convert_activity_to_nm,
    fetch_bioactivity_dataframe,
    filter_output_columns,
    normalize_smiles_column,
    process_ic50_values,
)


class ChemblBioactivityDataframeServiceTests(unittest.TestCase):
    @patch("chem_inf_widgets.chemcore.services.chembl_bioactivity_dataframe_service.requests.get")
    def test_fetch_bioactivity_dataframe_follows_pagination(self, mock_get):
        first = Mock()
        first.raise_for_status.return_value = None
        first.json.return_value = {
            "activities": [
                {
                    "canonical_smiles": "CCO",
                    "pchembl_value": "6.5",
                    "molecule_chembl_id": "CHEMBL1",
                },
                {
                    "canonical_smiles": "CCC",
                    "pchembl_value": None,
                    "molecule_chembl_id": "CHEMBL2",
                },
            ],
            "page_meta": {"next": "https://next.example/page-2"},
        }

        second = Mock()
        second.raise_for_status.return_value = None
        second.json.return_value = {
            "activities": [
                {
                    "canonical_smiles": "CCN",
                    "pchembl_value": "7.1",
                    "molecule_chembl_id": "CHEMBL3",
                }
            ],
            "page_meta": {"next": None},
        }

        mock_get.side_effect = [first, second]

        df = fetch_bioactivity_dataframe("CHEMBLT1", standard_type="IC50", limit=1000, timeout=30)

        self.assertEqual(list(df["molecule_chembl_id"]), ["CHEMBL1", "CHEMBL3"])
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(
            mock_get.call_args_list[0].kwargs["params"],
            {
                "target_chembl_id": "CHEMBLT1",
                "standard_type": "IC50",
                "limit": 1000,
            },
        )
        self.assertIsNone(mock_get.call_args_list[1].kwargs["params"])

    def test_convert_activity_to_nm(self):
        self.assertEqual(convert_activity_to_nm(1.5, "uM"), 1500.0)
        self.assertEqual(convert_activity_to_nm(50, "nM"), 50.0)
        self.assertTrue(pd.isna(convert_activity_to_nm("bad", "nM")))

    def test_process_and_normalize_dataframe(self):
        df = pd.DataFrame(
            [
                {
                    "canonical_smiles": "CCO",
                    "standard_value": 2.0,
                    "standard_units": "uM",
                    "molecule_chembl_id": "CHEMBL1",
                }
            ]
        )

        processed = process_ic50_values(df)
        normalized = normalize_smiles_column(processed)

        self.assertIn("IC50_nM", normalized.columns)
        self.assertEqual(normalized.at[0, "IC50_nM"], 2000.0)
        self.assertIn("SMILES", normalized.columns)
        self.assertNotIn("canonical_smiles", normalized.columns)

    def test_calculate_drug_properties_and_filter_columns(self):
        df = pd.DataFrame(
            [
                {
                    "SMILES": "CCO",
                    "molecule_chembl_id": "CHEMBL1",
                    "target_chembl_id": "CHEMBLT1",
                    "pchembl_value": 6.3,
                    "IC50_nM": 500.0,
                    "extra_col": "ignored",
                }
            ]
        )

        enriched = calculate_drug_properties(df)
        filtered = filter_output_columns(enriched)

        self.assertIn("mw", filtered.columns)
        self.assertIn("logp", filtered.columns)
        self.assertIn("SMILES", filtered.columns)
        self.assertNotIn("extra_col", filtered.columns)


if __name__ == "__main__":
    unittest.main()
