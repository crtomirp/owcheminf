import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class RdkitSafeTests(unittest.TestCase):
    def test_safe_mol_from_smiles_valid(self):
        from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles

        result = safe_mol_from_smiles("CCO")

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.mol)
        self.assertEqual(result.canonical_smiles, "CCO")
        self.assertIsNone(result.error)

    def test_safe_mol_from_smiles_invalid(self):
        from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles

        result = safe_mol_from_smiles("not-a-smiles")

        self.assertFalse(result.ok)
        self.assertIsNone(result.mol)
        self.assertIsNotNone(result.error)

    def test_safe_mol_to_inchikey(self):
        from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles, safe_mol_to_inchikey

        parsed = safe_mol_from_smiles("CCO")
        inchikey = safe_mol_to_inchikey(parsed.mol)

        self.assertTrue(parsed.ok)
        self.assertTrue(bool(inchikey))

    def test_safe_standardize_mol(self):
        from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles, safe_standardize_mol

        parsed = safe_mol_from_smiles("C(C)O")
        result = safe_standardize_mol(parsed.mol)

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.mol)
        self.assertTrue(bool(result.standardized_smiles))


if __name__ == "__main__":
    unittest.main()
