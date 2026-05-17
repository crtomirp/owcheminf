import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.rgroup_decomposition_service import decompose_rgroups  # noqa: E402


class RGroupDecompositionServiceTests(unittest.TestCase):
    def test_decompose_rgroups_uses_auto_core(self):
        smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1Cl"]
        result = decompose_rgroups(smiles)
        self.assertTrue(result.core)
        self.assertGreaterEqual(len(result.rows), 1)


if __name__ == "__main__":
    unittest.main()
