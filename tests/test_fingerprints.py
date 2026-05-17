import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.descriptors.fingerprints import compute_fingerprints_from_smiles  # noqa: E402


class FingerprintsTests(unittest.TestCase):
    def test_compute_fingerprints_skips_invalid_smiles(self):
        result = compute_fingerprints_from_smiles(["CCO", "not_a_smiles", ""], fp_type="morgan", bit_size=64)

        self.assertEqual(result.valid_indices, [0])
        self.assertEqual(result.failed_indices, [1, 2])
        self.assertEqual(result.X.shape[0], 1)
        self.assertEqual(result.X.shape[1], 64)

    def test_compute_fingerprints_honors_cancel(self):
        seen = []

        def progress_cb(pct: int):
            seen.append(pct)

        calls = {"n": 0}

        def cancel_cb():
            calls["n"] += 1
            return calls["n"] > 1

        result = compute_fingerprints_from_smiles(
            ["CCO", "CCN", "CCC"],
            fp_type="rdkit",
            bit_size=32,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

        self.assertEqual(result.valid_indices, [0])
        self.assertEqual(result.failed_indices, [])
        self.assertTrue(seen)


if __name__ == "__main__":
    unittest.main()
