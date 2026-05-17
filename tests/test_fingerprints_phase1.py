import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chem_inf_widgets.chemcore.descriptors.fingerprints import compute_fingerprints_from_smiles  # noqa: E402


class FingerprintsPhase1Tests(unittest.TestCase):
    def test_result_contains_provenance_and_errors(self):
        result = compute_fingerprints_from_smiles(["c1ccncc1", "not_a_smiles"], fp_type="morgan", bit_size=128)

        self.assertEqual(result.valid_indices, [0])
        self.assertEqual(result.failed_indices, [1])
        self.assertEqual(result.bit_size, 128)
        self.assertEqual(result.radius, 2)
        self.assertIsInstance(result.params, dict)
        self.assertTrue(result.errors)
        self.assertTrue(result.bit_names[0].startswith("morgan_"))

    def test_maccs_uses_effective_size_and_named_bits(self):
        result = compute_fingerprints_from_smiles(["CCO"], fp_type="maccs", bit_size=1024)

        self.assertEqual(result.X.shape[1], 167)
        self.assertEqual(result.bit_size, 167)
        self.assertEqual(result.bit_names[0], "MACCS_000")


if __name__ == "__main__":
    unittest.main()
