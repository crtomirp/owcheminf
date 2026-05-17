import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.activity_cliff_service import (  # noqa: E402
    NO_SCAFFOLD_LABEL,
    find_activity_cliffs,
    scaffold_activity_summary,
)


class ActivityCliffServiceTests(unittest.TestCase):
    def setUp(self):
        self.smiles = [
            "CC(=O)Nc1ccc(cc1)O",
            "CC(=O)Nc1ccc(cc1)Cl",
            "CC(=O)Nc1ccc(cc1)F",
            "CCO",
            "not_a_smiles",
        ]
        self.activities = [100.0, 1.0, 80.0, 50.0, 10.0]
        self.names = ["phenol", "chloro", "fluoro", "ethanol", "bad"]

    def test_find_activity_cliffs_returns_ranked_pairs(self):
        result = find_activity_cliffs(
            self.smiles,
            self.activities,
            names=self.names,
            similarity_threshold=0.6,
            activity_fold_threshold=10.0,
            max_pairs=10,
        )

        self.assertEqual(result.failed_indices, [4])
        self.assertTrue(result.pairs)
        top = result.pairs[0]
        self.assertEqual((top.name_a, top.name_b), ("phenol", "chloro"))
        self.assertGreaterEqual(top.similarity, 0.6)
        self.assertGreaterEqual(top.activity_ratio, 10.0)
        self.assertEqual(top.higher_active, "b")

    def test_find_activity_cliffs_supports_log_scale(self):
        result = find_activity_cliffs(
            ["CCO", "CCCO", "CCN"],
            [5.0, 6.2, 5.1],
            similarity_threshold=0.5,
            activity_fold_threshold=10.0,
            activity_log_scale=True,
        )

        self.assertEqual(len(result.pairs), 1)
        self.assertAlmostEqual(result.pairs[0].activity_ratio, 1.2, places=3)

    def test_scaffold_activity_summary_groups_by_scaffold(self):
        rows = scaffold_activity_summary(self.smiles, self.activities)

        self.assertEqual(rows[0].scaffold, "c1ccccc1")
        self.assertEqual(rows[0].count, 3)
        self.assertEqual(rows[0].best_activity, 1.0)
        self.assertTrue(any(row.scaffold == NO_SCAFFOLD_LABEL for row in rows))

    def test_mismatched_input_lengths_raise(self):
        with self.assertRaises(ValueError):
            find_activity_cliffs(["CCO"], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
