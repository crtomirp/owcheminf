from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication

from chem_inf_widgets.widgets.ow_qsar_validation_dashboard import _dataframe_to_orange
from chem_inf_widgets.widgets.ow_qsar_regression import (
    OWQSARRegression,
    _residual_reference_levels as regression_reference_levels,
)
from chem_inf_widgets.widgets.ow_mlr_model_selection import (
    OWMLRModelSelection,
    _residual_reference_levels as mlr_reference_levels,
)


_APP = QApplication.instance() or QApplication([])


def _selected_table():
    return _dataframe_to_orange(
        pd.DataFrame(
            {
                "compound_id": ["A", "B"],
                "split": ["test", "test"],
                "observed": [5.0, 6.0],
                "predicted": [5.2, 5.8],
                "residual": [-0.2, 0.2],
            }
        )
    )


def test_qsar_regression_reference_levels_include_std_bands():
    levels = regression_reference_levels([0.0, 1.0, 2.0, 3.0])

    assert levels["mean"] == pytest.approx(1.5)
    assert levels["std"] == pytest.approx(1.1180339887)
    assert levels["plus_1std"] == pytest.approx(levels["mean"] + levels["std"])
    assert levels["minus_2std"] == pytest.approx(levels["mean"] - 2.0 * levels["std"])


def test_qsar_regression_widget_updates_selected_tab():
    widget = OWQSARRegression()
    try:
        assert widget.selection_tool_combo.currentText() == "Rectangle"

        widget._update_selected_table(_selected_table())

        assert widget.selected_table.rowCount() == 2
        assert widget.tabs.tabText(widget.tabs.indexOf(widget.selected_tab)) == "Selected (2)"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_qsar_regression_auto_qsar_disables_manual_algorithm_choice():
    widget = OWQSARRegression()
    try:
        widget.enable_auto_qsar = True
        widget._update_auto_qsar_state()

        assert widget.algorithm_combo.isEnabled() is False
        assert "ignored" in widget.auto_qsar_hint.text().lower()
        assert widget.mode_chip_label.text() == "Auto QSAR mode"

        widget.enable_auto_qsar = False
        widget._update_auto_qsar_state()
        assert widget.mode_chip_label.text() == "Manual mode"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_mlr_reference_levels_include_std_bands():
    levels = mlr_reference_levels([0.0, 1.0, 2.0, 3.0])

    assert levels["mean"] == pytest.approx(1.5)
    assert levels["std"] == pytest.approx(1.1180339887)
    assert levels["plus_1std"] == pytest.approx(levels["mean"] + levels["std"])
    assert levels["minus_2std"] == pytest.approx(levels["mean"] - 2.0 * levels["std"])


def test_mlr_widget_updates_selected_tab():
    widget = OWMLRModelSelection()
    try:
        assert widget._cmb_selection_tool.currentText() == "Rectangle"

        widget._update_selected_table(_selected_table())

        assert widget._selected_table.rowCount() == 2
        assert widget.diagnostic_tabs.tabText(widget.diagnostic_tabs.indexOf(widget._selected_tab)) == "Selected (2)"
    finally:
        widget.close()
