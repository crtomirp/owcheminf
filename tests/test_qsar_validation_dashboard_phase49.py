from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication

from chem_inf_widgets.widgets.ow_qsar_validation_dashboard import (
    OWQSARValidationDashboard,
    _dataframe_to_orange,
    _residual_reference_levels,
)


_APP = QApplication.instance() or QApplication([])


def _diagnostics_df() -> pd.DataFrame:
    observed = np.array([5.1, 5.8, 6.2, 6.7, 7.0], dtype=float)
    predicted = np.array([5.0, 6.0, 6.0, 6.9, 6.8], dtype=float)
    return pd.DataFrame(
        {
            "compound_id": [f"C{i:03d}" for i in range(len(observed))],
            "name": [f"cmpd_{i:03d}" for i in range(len(observed))],
            "split": ["train", "train", "test", "test", "test"],
            "observed": observed,
            "predicted": predicted,
            "residual": observed - predicted,
        }
    )


def test_qsar_validation_reference_levels_include_std_bands():
    levels = _residual_reference_levels([0.0, 1.0, 2.0, 3.0])

    assert levels["mean"] == pytest.approx(1.5)
    assert levels["std"] == pytest.approx(1.1180339887)
    assert levels["plus_1std"] == pytest.approx(levels["mean"] + levels["std"])
    assert levels["minus_2std"] == pytest.approx(levels["mean"] - 2.0 * levels["std"])


def test_qsar_validation_widget_defaults_to_rectangle_selection():
    widget = OWQSARValidationDashboard()
    try:
        assert widget._cmb_selection_tool.currentText() == "Rectangle"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_qsar_validation_widget_builds_diagnostics_and_selected_rows():
    widget = OWQSARValidationDashboard()
    try:
        df = _diagnostics_df()
        widget._diagnostics_table = _dataframe_to_orange(df)
        widget._update_diagnostics(df)

        assert widget._diagnostic_canvas is not None
        assert widget._diagnostic_context is not None
        assert "combined" in widget._diagnostic_selectors

        widget._publish_selection(np.array([1, 3], dtype=int))

        assert widget._selected_table.rowCount() == 2
        assert widget._tabs.tabText(widget._tabs.indexOf(widget._tab_selected)) == "Selected (2)"
        assert "Selected 2 compounds" in widget._diagnostics_hint.text()
    finally:
        widget.onDeleteWidget()
        widget.close()
