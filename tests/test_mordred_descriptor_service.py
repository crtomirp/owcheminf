import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.mol import ChemMol  # noqa: E402

try:
    from chem_inf_widgets.chemcore.services.mordred_descriptor_service import (  # noqa: E402
        MORDRED_AVAILABLE,
        MordredDescriptorService,
    )
except Exception:  # pragma: no cover
    MORDRED_AVAILABLE = False
    MordredDescriptorService = None


@unittest.skipIf(not MORDRED_AVAILABLE or MordredDescriptorService is None, "mordred is not available")
class MordredDescriptorServiceTests(unittest.TestCase):
    def test_chemmols_to_mols_uses_smiles_fallback(self):
        chem_mol = ChemMol.from_smiles("CCO", name="ethanol")
        chem_mol.mol = None
        chem_mol.set_prop("SMILES", "CCO")

        mols, valid_idx = MordredDescriptorService.chemmols_to_mols([chem_mol])

        self.assertEqual(valid_idx, [0])
        self.assertIsNotNone(mols[0])

    def test_numeric_or_none_handles_nan_and_strings(self):
        self.assertIsNone(MordredDescriptorService.numeric_or_none(np.nan))
        self.assertEqual(MordredDescriptorService.numeric_or_none("3.5"), 3.5)
        self.assertIsNone(MordredDescriptorService.numeric_or_none("abc"))


if __name__ == "__main__":
    unittest.main()
