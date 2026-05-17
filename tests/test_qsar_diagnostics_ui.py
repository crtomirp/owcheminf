import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.widgets.qsar_diagnostics_ui import (  # noqa: E402
    build_diagnostic_selection_context,
    clear_selection_overlays,
    selection_plot_values,
    set_selector_mode,
    update_selection_overlays,
)


class _Selector:
    def __init__(self):
        self.active = None

    def set_active(self, state):
        self.active = bool(state)


class _Overlay:
    def __init__(self):
        self.offsets = None

    def set_offsets(self, offsets):
        self.offsets = np.asarray(offsets)


class _Canvas:
    def __init__(self):
        self.draw_calls = 0

    def draw_idle(self):
        self.draw_calls += 1


class _Bundle:
    def __init__(self):
        self.rect_left = _Selector()
        self.rect_right = _Selector()
        self.lasso_left = _Selector()
        self.lasso_right = _Selector()


class QSARDiagnosticsUiTests(unittest.TestCase):
    def test_set_selector_mode(self):
        bundle = _Bundle()
        set_selector_mode(bundle, use_lasso=True)
        self.assertFalse(bundle.rect_left.active)
        self.assertFalse(bundle.rect_right.active)
        self.assertTrue(bundle.lasso_left.active)
        self.assertTrue(bundle.lasso_right.active)

    def test_selection_plot_values(self):
        context = build_diagnostic_selection_context(
            canvas=_Canvas(),
            figure=None,
            preds=[1.0, 2.0],
            y=[1.1, 1.9],
            residuals=[0.1, -0.1],
            table=None,
            overlay_left=_Overlay(),
            overlay_right=_Overlay(),
        )
        preds, y = selection_plot_values(context, left_plot=True)
        self.assertEqual(preds.tolist(), [1.0, 2.0])
        self.assertEqual(y.tolist(), [1.1, 1.9])
        preds, residuals = selection_plot_values(context, left_plot=False)
        self.assertEqual(residuals.tolist(), [0.1, -0.1])

    def test_update_and_clear_selection_overlays(self):
        canvas = _Canvas()
        left = _Overlay()
        right = _Overlay()
        context = build_diagnostic_selection_context(
            canvas=canvas,
            figure=None,
            preds=[1.0, 2.0, 3.0],
            y=[1.1, 1.9, 2.8],
            residuals=[0.1, -0.1, -0.2],
            table=None,
            overlay_left=left,
            overlay_right=right,
        )

        update_selection_overlays(context, np.array([0, 2], dtype=int))
        self.assertEqual(left.offsets.shape, (2, 2))
        self.assertEqual(right.offsets.shape, (2, 2))
        self.assertEqual(canvas.draw_calls, 1)

        clear_selection_overlays(context)
        self.assertEqual(left.offsets.shape, (0, 2))
        self.assertEqual(right.offsets.shape, (0, 2))
        self.assertEqual(canvas.draw_calls, 2)


if __name__ == "__main__":
    unittest.main()
