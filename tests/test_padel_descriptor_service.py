import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.mol import ChemMol  # noqa: E402
from chem_inf_widgets.chemcore.services.padel_descriptor_service import PadelDescriptorService  # noqa: E402


class PadelDescriptorServiceTests(unittest.TestCase):
    def test_chemmols_to_smiles_uses_smiles_fallback(self):
        chem_mol = ChemMol.from_smiles("CCO", name="ethanol")
        chem_mol.mol = None
        chem_mol.set_prop("SMILES", "CCO")

        smiles = PadelDescriptorService.chemmols_to_smiles([chem_mol])

        self.assertEqual(smiles, ["CCO"])

    def test_numeric_or_none_handles_nan_and_strings(self):
        self.assertIsNone(PadelDescriptorService.numeric_or_none(np.nan))
        self.assertEqual(PadelDescriptorService.numeric_or_none("4.25"), 4.25)
        self.assertIsNone(PadelDescriptorService.numeric_or_none("abc"))


if __name__ == "__main__":
    unittest.main()
