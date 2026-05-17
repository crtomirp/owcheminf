import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.admet.drug_filter_service import (  # noqa: E402
    FilterConfig,
    _compiled_pains_rules_default,
    canonical_smiles,
    filter_smiles,
)


class DrugFilterServiceTests(unittest.TestCase):
    def test_canonical_smiles_preserves_invalid_input(self):
        self.assertEqual(canonical_smiles("not_a_smiles"), "not_a_smiles")

    def test_packaged_pains_rules_are_available(self):
        rules = _compiled_pains_rules_default()
        self.assertTrue(rules, "Expected packaged PAINS rules to be compiled")

    def test_filter_smiles_selection_modes(self):
        smiles = ["CCO", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"]

        within = filter_smiles(
            smiles,
            FilterConfig(filter_rule="Lipinski", selection_mode="Within Criteria"),
        )
        self.assertEqual([row.smiles for row in within], ["CCO"])
        self.assertEqual(within[0].criteria, "Pass")

        outside = filter_smiles(
            smiles,
            FilterConfig(filter_rule="Lipinski", selection_mode="Out of Criteria"),
        )
        self.assertEqual([row.smiles for row in outside], ["CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"])
        self.assertEqual(outside[0].criteria, "Fail")

    def test_filter_smiles_can_highlight_pains_atoms(self):
        rows = filter_smiles(
            ["c1ccccc1N=Nc2ccccc2"],
            FilterConfig(
                filter_rule="None",
                selection_mode="Forward All Molecules",
                highlight_pains_atoms=True,
            ),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].pains_match, 1.0)
        self.assertNotEqual(rows[0].highlighted_atoms, "")

    def test_invalid_smiles_are_skipped(self):
        rows = filter_smiles(
            ["CCO", "not_a_smiles", ""],
            FilterConfig(filter_rule="None", selection_mode="Forward All Molecules"),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].smiles, "CCO")


if __name__ == "__main__":
    unittest.main()
