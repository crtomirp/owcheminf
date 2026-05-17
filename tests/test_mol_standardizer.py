import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.mol import ChemMol  # noqa: E402
from chem_inf_widgets.chemcore.services.mol_standardizer import MolStandardizer  # noqa: E402


class MolStandardizerTests(unittest.TestCase):
    def test_standardize_smiles_invalid(self):
        service = MolStandardizer()

        result = service.standardize_smiles("not_a_smiles")

        self.assertFalse(result.ok)
        self.assertEqual(result.output_smiles, "")

    def test_standardize_chemmols_uses_rdkit_fallback_smiles(self):
        service = MolStandardizer()
        chem_mol = ChemMol.from_smiles("C(C)O", name="ethanol")
        chem_mol.set_prop("SMILES", "")

        out_mols, results = service.standardize_chemmols([chem_mol])

        self.assertEqual(len(out_mols), 1)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].ok)
        self.assertTrue(bool(out_mols[0].get_prop("SMILES_STD")))


if __name__ == "__main__":
    unittest.main()
