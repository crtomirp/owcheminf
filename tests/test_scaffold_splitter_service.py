import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.scaffold_splitter_service import split_by_scaffold  # noqa: E402


class ScaffoldSplitterServiceTests(unittest.TestCase):
    def test_split_by_scaffold_returns_expected_partitions(self):
        smiles = [
            "c1ccccc1O",
            "c1ccccc1N",
            "CCO",
            "CCN",
            "c1ccncc1",
            "c1ccncc1O",
        ]
        result = split_by_scaffold(smiles, random_seed=0)
        splits = {assignment.split for assignment in result.assignments}
        self.assertIn("train", splits)
        self.assertIn("test", splits)
        self.assertEqual(len(result.assignments), len(smiles))


if __name__ == "__main__":
    unittest.main()
