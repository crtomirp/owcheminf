import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.scaffold_service import (  # noqa: E402
    NO_SCAFFOLD_LABEL,
    analyze_scaffolds,
    build_scaffold_summary,
    get_generic_scaffold,
    get_murcko_scaffold,
)


class ScaffoldServiceTests(unittest.TestCase):
    def test_get_murcko_scaffold_extracts_ring_core(self):
        scaffold = get_murcko_scaffold("Cc1ccccc1")

        self.assertEqual(scaffold, "c1ccccc1")

    def test_get_generic_scaffold_normalizes_heteroatoms(self):
        scaffold = get_generic_scaffold("c1ccncc1")

        self.assertEqual(scaffold, "C1CCCCC1")

    def test_analyze_scaffolds_reports_counts_and_invalid_rows(self):
        result = analyze_scaffolds(
            [
                "Cc1ccccc1",
                "Oc1ccccc1",
                "CCO",
                "not_a_smiles",
            ]
        )

        self.assertEqual(result.valid_count, 3)
        self.assertEqual(result.failed_indices, [3])
        self.assertEqual(result.murcko_counts[0], ("c1ccccc1", 2))
        self.assertIn((NO_SCAFFOLD_LABEL, 1), result.murcko_counts)
        self.assertEqual(result.annotations[2].status, "acyclic")
        self.assertEqual(result.annotations[3].status, "invalid")

    def test_build_scaffold_summary_can_exclude_acyclic_rows(self):
        result = analyze_scaffolds(["Cc1ccccc1", "CCO", "CC1CCCCC1"])

        summary_rows = build_scaffold_summary(result, kind="murcko", include_acyclic=False)

        self.assertEqual([row.scaffold for row in summary_rows], ["C1CCCCC1", "c1ccccc1"])
        self.assertTrue(all(row.scaffold != NO_SCAFFOLD_LABEL for row in summary_rows))

    def test_build_generic_summary_respects_top_n(self):
        result = analyze_scaffolds(["Cc1ccccc1", "c1ccncc1", "CC1CCCCC1", "CCO"])

        summary_rows = build_scaffold_summary(result, kind="generic", top_n=1)

        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(summary_rows[0].scaffold, "C1CCCCC1")
        self.assertAlmostEqual(summary_rows[0].fraction, 0.75, places=4)


if __name__ == "__main__":
    unittest.main()
