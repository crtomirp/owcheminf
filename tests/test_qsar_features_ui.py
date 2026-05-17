import unittest

import numpy as np
from AnyQt.QtWidgets import QApplication

from chem_inf_widgets.widgets import qsar_features_ui


_APP = QApplication.instance() or QApplication([])


class TestQSARFeaturesUI(unittest.TestCase):
    def test_build_feature_message_label(self):
        label = qsar_features_ui.build_feature_message_label("<b>Hello</b>")
        self.assertTrue(label.wordWrap())
        self.assertIn("padding:12px", label.styleSheet())
        self.assertEqual(label.text(), "<b>Hello</b>")

    def test_build_feature_chart_figure(self):
        fig = qsar_features_ui.build_feature_chart_figure(
            ("A", "B"),
            np.asarray([0.3, -0.1], dtype=float),
            ("#16a34a", "#dc2626"),
            value_label="Coefficient",
            chart_title="QSAR Test",
        )
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(fig.axes[0].get_title(), "QSAR Test")
        self.assertEqual(fig.axes[0].get_xlabel(), "Coefficient")

    def test_build_features_table(self):
        table = qsar_features_ui.build_features_table(
            ["desc1", "desc2"],
            [0.25, -0.75],
            "Coefficient",
            ses=[0.1, 0.2],
            ts=[2.5, -3.75],
            ps=[0.04, 0.2],
            vifs=[1.5, 12.0],
        )
        self.assertEqual(table.rowCount(), 2)
        self.assertEqual(table.columnCount(), 7)
        rows = [
            [None if table.item(r, c) is None else table.item(r, c).data(0) for c in range(table.columnCount())]
            for r in range(table.rowCount())
        ]
        self.assertIn("desc1", [row[1] for row in rows])
        self.assertIn("desc2", [row[1] for row in rows])
        self.assertIn(-0.75, [row[2] for row in rows])


if __name__ == "__main__":
    unittest.main()
