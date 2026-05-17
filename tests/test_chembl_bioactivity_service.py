import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.chembl_bioactivity_service import (  # noqa: E402
    canonical_smiles_no_h,
)


class ChemblBioactivityServiceTests(unittest.TestCase):
    def test_canonical_smiles_no_h_returns_canonical_smiles(self):
        self.assertEqual(canonical_smiles_no_h("C(C)O"), "CCO")

    def test_canonical_smiles_no_h_keeps_invalid_input_as_text(self):
        self.assertEqual(canonical_smiles_no_h("not-a-smiles"), "not-a-smiles")


if __name__ == "__main__":
    unittest.main()
