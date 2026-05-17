import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.diversity_service import (  # noqa: E402
    butina_cluster_selection,
    diversity_metrics,
    maxmin_selection,
    select_diverse_subset,
    sphere_exclusion,
)


class DiversityServiceTests(unittest.TestCase):
    def setUp(self):
        self.smiles = [
            "CCO",
            "CCCO",
            "c1ccccc1",
            "c1ccncc1",
            "CC(=O)O",
            "not_a_smiles",
        ]

    def test_maxmin_selection_returns_requested_count_from_valid_rows(self):
        selected = maxmin_selection(self.smiles, n_select=3, seed_idx=0, random_seed=7)

        self.assertEqual(len(selected), 3)
        self.assertTrue(all(idx < 5 for idx in selected))

    def test_sphere_exclusion_drops_invalid_smiles(self):
        selected = sphere_exclusion(self.smiles, radius=0.35, random_seed=7)

        self.assertTrue(selected)
        self.assertTrue(all(idx < 5 for idx in selected))

    def test_butina_cluster_selection_limits_to_requested_cluster_count(self):
        selected = butina_cluster_selection(self.smiles, n_clusters=2, threshold=0.4)

        self.assertLessEqual(len(selected), 2)
        self.assertTrue(all(idx < 5 for idx in selected))

    def test_diversity_metrics_reports_valid_compound_count(self):
        metrics = diversity_metrics(self.smiles, random_seed=7)

        self.assertEqual(metrics.n_compounds, 5)
        self.assertGreaterEqual(metrics.diversity_score, 0.0)

    def test_select_diverse_subset_reports_failed_indices_and_selected_metrics(self):
        result = select_diverse_subset(
            self.smiles,
            method="maxmin",
            n_select=2,
            random_seed=7,
        )

        self.assertEqual(result.failed_indices, [5])
        self.assertEqual(len(result.selected_indices), 2)
        self.assertEqual(result.metrics_input.n_compounds, 5)
        self.assertEqual(result.metrics_selected.n_compounds, 2)


if __name__ == "__main__":
    unittest.main()
