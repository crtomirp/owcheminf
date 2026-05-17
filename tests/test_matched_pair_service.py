import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.matched_pair_service import find_matched_pairs  # noqa: E402


class MatchedPairServiceTests(unittest.TestCase):
    def test_find_matched_pairs_returns_transformations(self):
        smiles = ["c1ccccc1O", "c1ccccc1N", "CCO"]
        rows = find_matched_pairs(smiles, [1.0, 2.5, 0.1], max_pairs=10)
        self.assertTrue(any("->" in row.transformation for row in rows))


if __name__ == "__main__":
    unittest.main()
