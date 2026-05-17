import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.mol import ChemMol  # noqa: E402


class ChemMolTests(unittest.TestCase):
    def test_to_rdkit_returns_a_copy(self):
        mol = ChemMol.from_smiles("CCO", name="ethanol")

        rdkit_mol = mol.to_rdkit()

        self.assertIsNotNone(rdkit_mol)
        self.assertIsNot(rdkit_mol, mol.mol)
        self.assertEqual(mol.smiles(), "CCO")


if __name__ == "__main__":
    unittest.main()
