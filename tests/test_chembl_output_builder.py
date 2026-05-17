import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.chembl_models import ChemBLBioactivityRecord  # noqa: E402
from chem_inf_widgets.chemcore.services.chembl_molecule_service import ChemBLMoleculePropsRecord  # noqa: E402
from chem_inf_widgets.chemcore.services.chembl_output_builder import (  # noqa: E402
    aggregate_bio_by_molecule,
    build_bioactivity_outputs,
    build_molecule_outputs,
    derive_prop_keys_from_records,
)


BIO_FIELD_SPECS = [
    ("pChEMBL", "num"),
    ("standard_value", "num"),
    ("IC50_nM", "num"),
    ("standard_type", "meta"),
    ("standard_units", "meta"),
    ("assay_chembl_id", "meta"),
    ("target_chembl_id", "meta"),
    ("molecule_chembl_id", "meta"),
    ("pref_name", "meta"),
    ("SMILES", "smiles"),
]


class ChemblOutputBuilderTests(unittest.TestCase):
    def test_aggregate_bio_by_molecule(self):
        recs = [
            ChemBLBioactivityRecord(
                molecule_chembl_id="CHEMBL1",
                target_chembl_id="CHEMBLT1",
                smiles="CCO",
                standard_type="IC50",
                standard_value=120.0,
                standard_units="nM",
                pchembl_value=6.5,
                ic50_nM=120.0,
            ),
            ChemBLBioactivityRecord(
                molecule_chembl_id="CHEMBL1",
                target_chembl_id="CHEMBLT1",
                smiles="CCO",
                standard_type="IC50",
                standard_value=80.0,
                standard_units="nM",
                pchembl_value=7.0,
                ic50_nM=80.0,
            ),
        ]

        aggregated = aggregate_bio_by_molecule(
            recs,
            ["pChEMBL", "standard_value", "standard_type", "standard_units"],
            BIO_FIELD_SPECS,
        )

        self.assertEqual(aggregated["CHEMBL1"]["bio_n"], 2)
        self.assertEqual(aggregated["CHEMBL1"]["pChEMBL"], 7.0)
        self.assertEqual(aggregated["CHEMBL1"]["standard_value"], 80.0)
        self.assertEqual(aggregated["CHEMBL1"]["standard_type"], "IC50")
        self.assertEqual(aggregated["CHEMBL1"]["standard_units"], "nM")

    def test_derive_prop_keys_from_records_prefers_known_qsar_fields(self):
        props_by_id = {
            "CHEMBL1": ChemBLMoleculePropsRecord(
                chembl_id="CHEMBL1",
                pref_name="demo",
                canonical_smiles="CCO",
                props={"rtb": 1, "custom_b": 2, "alogp": 3.1, "custom_a": 5},
            )
        }

        keys = derive_prop_keys_from_records(props_by_id, max_keys=4)

        self.assertEqual(keys[:2], ["alogp", "rtb"])
        self.assertEqual(len(keys), 4)

    def test_build_outputs_skip_invalid_smiles_in_molecule_list(self):
        recs = [
            ChemBLBioactivityRecord(
                molecule_chembl_id="CHEMBL1",
                target_chembl_id="CHEMBLT1",
                smiles="CCO",
                standard_type="IC50",
                standard_value=120.0,
                standard_units="nM",
                pchembl_value=6.5,
                ic50_nM=120.0,
            ),
            ChemBLBioactivityRecord(
                molecule_chembl_id="CHEMBL2",
                target_chembl_id="CHEMBLT1",
                smiles="not_a_smiles",
                standard_type="IC50",
                standard_value=500.0,
                standard_units="nM",
                pchembl_value=5.0,
                ic50_nM=500.0,
            ),
        ]

        table, mols = build_bioactivity_outputs(
            recs,
            prop_keys=[],
            props_by_id={},
            selected_bio_fields=["pChEMBL"],
            bio_field_specs=BIO_FIELD_SPECS,
        )
        self.assertIsNotNone(table)
        self.assertEqual(len(table), 2)
        self.assertEqual(len(mols), 1)
        self.assertEqual(mols[0].name, "CHEMBL1")

        mol_table, mol_outputs = build_molecule_outputs(
            mols=[
                type("M", (), {"chembl_id": "CHEMBL1", "pref_name": "demo", "canonical_smiles": "CCO"})(),
                type("M", (), {"chembl_id": "CHEMBL2", "pref_name": "bad", "canonical_smiles": "not_a_smiles"})(),
            ],
            props_by_id={},
            prop_keys=[],
            recs=[],
            selected_bio_fields=[],
            bio_field_specs=BIO_FIELD_SPECS,
        )
        self.assertIsNotNone(mol_table)
        self.assertEqual(len(mol_table), 2)
        self.assertEqual(len(mol_outputs), 1)
        self.assertEqual(mol_outputs[0].name, "CHEMBL1")


if __name__ == "__main__":
    unittest.main()
