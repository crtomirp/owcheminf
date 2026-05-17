import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.descriptors.descriptors import compute_fingerprints_from_smiles  # noqa: E402


class LegacyDescriptorFingerprintsTests(unittest.TestCase):
    def test_compute_fingerprints_skips_invalid_smiles(self):
        result = compute_fingerprints_from_smiles(["CCO", "not_a_smiles"], fp_type="morgan", bit_size=64)

        self.assertEqual(result.valid_indices, [0])
        self.assertEqual(result.failed_indices, [1])
        self.assertEqual(result.X.shape[0], 1)
        self.assertEqual(result.X.shape[1], 64)

    def test_compute_fingerprints_reports_empty_result_for_all_invalid(self):
        result = compute_fingerprints_from_smiles(["", "not_a_smiles"], fp_type="maccs")

        self.assertEqual(result.valid_indices, [])
        self.assertEqual(result.failed_indices, [0, 1])
        self.assertEqual(result.X.shape, (0, 0))


if __name__ == "__main__":
    unittest.main()
