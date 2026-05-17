import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


try:
    from Orange.data import Domain, StringVariable, Table
except Exception:  # pragma: no cover
    Domain = None
    StringVariable = None
    Table = None


@unittest.skipIf(Table is None, "Orange is not available")
class FromOrangeSmokeTests(unittest.TestCase):
    def test_table_to_chemmols_and_back(self):
        from chem_inf_widgets.chemcore.services.from_orange import chemmols_to_table, table_to_chemmols

        domain = Domain([], metas=[StringVariable("SMILES"), StringVariable("Name")])
        table = Table.from_list(domain, [["CCO", "ethanol"], ["O", "water"]])

        mols = table_to_chemmols(table)
        self.assertEqual(len(mols), 2)
        self.assertEqual(mols[0].name, "ethanol")
        self.assertEqual(mols[0].props["SMILES"], "CCO")

        roundtrip = chemmols_to_table(mols)
        self.assertEqual(len(roundtrip), 2)
        self.assertEqual(roundtrip[0, "SMILES"], "CCO")
        self.assertEqual(roundtrip[0, "Name"], "ethanol")

    def test_table_to_chemmols_with_report(self):
        from chem_inf_widgets.chemcore.services.from_orange import table_to_chemmols_with_report

        domain = Domain([], metas=[StringVariable("SMILES"), StringVariable("Name")])
        table = Table.from_list(
            domain,
            [
                ["CCO", "ethanol"],
                ["not-a-smiles", "broken"],
                ["", "empty"],
                ["O", "water"],
            ],
        )

        mols, report = table_to_chemmols_with_report(table)

        self.assertEqual(len(mols), 2)
        self.assertEqual(report.n_rows, 4)
        self.assertEqual(report.n_valid, 2)
        self.assertEqual(report.n_invalid, 2)
        self.assertEqual(report.skipped_rows, [2, 3])
        self.assertEqual(report.smiles_column, "SMILES")
        self.assertEqual(report.name_column, "Name")
        self.assertEqual(mols[0].props["SMILES"], "CCO")
        self.assertTrue(any("Row 2" in msg for msg in report.errors))
        self.assertTrue(any("Row 3" in msg for msg in report.errors))

    def test_dataset_to_table(self):
        from chem_inf_widgets.chemcore.models.chembl_dataset import ChemBLDataset
        from chem_inf_widgets.chemcore.mol import ChemMol
        from chem_inf_widgets.chemcore.services.from_orange import dataset_to_table

        dataset = ChemBLDataset(
            mols=[ChemMol.from_smiles("CCO", name="ethanol")],
            props=[{"chembl_id": "CHEMBL1", "target": "demo", "activity": 6.5}],
        )
        table = dataset_to_table(dataset)

        self.assertEqual(len(table), 1)
        self.assertEqual(table[0, "chembl_id"], "CHEMBL1")
        self.assertEqual(table[0, "target"], "demo")
        self.assertIn("activity", [v.name for v in table.domain.attributes])
        self.assertNotIn("activity", [v.name for v in table.domain.metas])
        self.assertAlmostEqual(float(table[0, "activity"]), 6.5)


if __name__ == "__main__":
    unittest.main()
