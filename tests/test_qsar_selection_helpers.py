import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.qsar_regression_service import (  # noqa: E402
    lasso_selection_indices,
    rectangle_selection_indices,
    selection_overlay_offsets,
    selection_status_text,
)


class QSARSelectionHelpersTests(unittest.TestCase):
    def test_rectangle_selection_indices(self):
        preds = np.array([0.1, 0.5, 0.9])
        ys = np.array([1.0, 2.0, 3.0])

        selected = rectangle_selection_indices(preds, ys, 0.0, 0.5, 0.6, 2.5)

        self.assertEqual(selected.tolist(), [0, 1])

    def test_lasso_selection_indices(self):
        preds = np.array([0.1, 0.5, 0.9])
        ys = np.array([1.0, 2.0, 3.0])
        vertices = [(0.0, 0.5), (0.6, 0.5), (0.6, 2.5), (0.0, 2.5)]

        selected = lasso_selection_indices(preds, ys, vertices)

        self.assertEqual(selected.tolist(), [0, 1])

    def test_selection_overlay_offsets(self):
        preds = np.array([0.1, 0.5, 0.9])
        ys = np.array([1.0, 2.0, 3.0])
        residuals = np.array([0.2, -0.1, 0.3])

        left, right = selection_overlay_offsets(preds, ys, residuals, np.array([1, 2]))

        self.assertEqual(left.shape, (2, 2))
        self.assertEqual(right.shape, (2, 2))
        self.assertTrue(np.allclose(left[0], [0.5, 2.0]))
        self.assertTrue(np.allclose(right[1], [0.9, 0.3]))

    def test_selection_status_text(self):
        text = selection_status_text("Random Forest", "test", 3)

        self.assertIn("Random Forest", text)
        self.assertIn("Selected 3 compounds", text)


if __name__ == "__main__":
    unittest.main()
