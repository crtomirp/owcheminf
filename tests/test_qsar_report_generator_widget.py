from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication

from chem_inf_widgets.widgets.ow_qsar_report_generator import (
    OWQSARReportGenerator,
    _df_to_table,
)


_APP = QApplication.instance() or QApplication([])


def _wait_until_finished(widget: OWQSARReportGenerator, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        _APP.processEvents()
        if widget._btn_run.isEnabled() and not widget._progress.isVisible():
            _APP.processEvents()
            return
        time.sleep(0.01)
    raise AssertionError("QSAR report generator did not finish in time.")


def test_qsar_report_generator_accepts_noncanonical_prediction_and_metrics_inputs():
    widget = OWQSARReportGenerator()
    try:
        dataset = pd.DataFrame(
            {
                "compound_id": ["C001", "C002", "C003", "C004"],
                "pActivity": [5.1, 5.4, 5.8, 6.0],
                "MolWt": [110.0, 124.0, 138.0, 149.0],
                "LogP": [1.1, 1.4, 1.9, 2.2],
            }
        )
        predictions = pd.DataFrame(
            {
                "compound_id": ["C001", "C002", "C003", "C004"],
                "actual_value": [5.1, 5.4, 5.8, 6.0],
                "predicted_pActivity": [5.0, 5.5, 5.75, 6.1],
                "dataset": ["train", "train", "external", "cv"],
            }
        )
        metrics = pd.DataFrame(
            [
                {
                    "train_r2": 0.91,
                    "test_r2": 0.82,
                    "cv_r2": 0.79,
                    "rmse_train": 0.12,
                    "rmse_test": 0.24,
                    "rmse_cv": 0.27,
                    "train_mae": 0.09,
                    "test_mae": 0.18,
                    "cv_mae": 0.22,
                }
            ]
        )
        feature_importance = pd.DataFrame(
            {
                "feature": ["MolWt", "LogP", "TPSA"],
                "normalized_importance": [1.0, 0.66, 0.31],
            }
        )

        widget.set_dataset(_df_to_table(dataset))
        widget.set_predictions(_df_to_table(predictions))
        widget.set_metrics(_df_to_table(metrics))
        widget._data["feature_importance"] = _df_to_table(feature_importance)
        widget.commit()
        _wait_until_finished(widget)

        status = widget._lbl_status.text().lower()
        report_text = widget._report_browser.toPlainText()

        assert "report ready" in status
        assert "sections" in status
        assert "actual_value" in report_text
        assert "predicted_pactivity" in report_text.lower()
        assert not widget._imp_placeholder.isVisible()
        assert len(widget._op_plot.getPlotItem().items) >= 2
        assert len(widget._res_plot.getPlotItem().items) >= 2
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_qsar_report_generator_can_export_pdf(tmp_path: Path):
    widget = OWQSARReportGenerator()
    try:
        dataset = pd.DataFrame(
            {
                "compound_id": ["C001", "C002", "C003"],
                "pActivity": [5.1, 5.4, 5.8],
                "MolWt": [110.0, 124.0, 138.0],
            }
        )
        metrics = pd.DataFrame([{"train_r2": 0.91, "test_r2": 0.82, "cv_r2": 0.79}])

        widget.set_dataset(_df_to_table(dataset))
        widget.set_metrics(_df_to_table(metrics))
        widget.commit()
        _wait_until_finished(widget)

        out_path = tmp_path / "qsar_widget_report.pdf"
        saved = widget._save_report_pdf(str(out_path))

        assert saved.endswith(".pdf")
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert widget._btn_pdf.isEnabled()
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_qsar_report_generator_sets_plot_ranges_when_tabs_are_hidden():
    widget = OWQSARReportGenerator()
    try:
        widget.show()
        _APP.processEvents()
        dataset = pd.DataFrame(
            {
                "compound_id": ["C001", "C002", "C003", "C004"],
                "pActivity": [5.1, 5.4, 5.8, 6.0],
                "MolWt": [110.0, 124.0, 138.0, 149.0],
            }
        )
        predictions = pd.DataFrame(
            {
                "compound_id": ["C001", "C002", "C003", "C004"],
                "observed": [5.1, 5.4, 5.8, 6.0],
                "predicted": [5.0, 5.5, 5.75, 6.1],
                "split": ["train", "train", "test", "cross_validation"],
            }
        )
        metrics = pd.DataFrame(
            [
                {"group": "train", "metric": "train_r2", "value": 0.91},
                {"group": "test", "metric": "test_r2", "value": 0.82},
                {"group": "cross_validation", "metric": "cv_r2", "value": 0.79},
                {"group": "train", "metric": "train_rmse", "value": 0.12},
                {"group": "test", "metric": "test_rmse", "value": 0.24},
                {"group": "cross_validation", "metric": "cv_rmse", "value": 0.27},
            ]
        )

        widget.set_dataset(_df_to_table(dataset))
        widget.set_predictions(_df_to_table(predictions))
        widget.set_metrics(_df_to_table(metrics))
        widget.commit()
        _wait_until_finished(widget)

        widget._tabs.setCurrentIndex(1)
        _APP.processEvents()
        widget._tabs.setCurrentIndex(2)
        _APP.processEvents()
        widget._tabs.setCurrentIndex(3)
        _APP.processEvents()

        obs_range = widget._op_plot.getPlotItem().viewRange()
        res_range = widget._res_plot.getPlotItem().viewRange()
        met_range = widget._met_r2.getPlotItem().viewRange()

        assert obs_range[0][1] > 5.0
        assert res_range[0][1] > 5.0
        assert res_range[1][0] < 0.0 < res_range[1][1]
        assert met_range[0][1] > 2.0
        assert met_range[1][1] > 0.7
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_qsar_report_generator_hover_handlers_accept_pyqtgraph_signal_shape():
    class _DummyPos:
        def __init__(self, x: float, y: float):
            self._x = x
            self._y = y

        def x(self):
            return self._x

        def y(self):
            return self._y

    class _DummyPoint:
        def __init__(self, label: str, x: float, y: float):
            self._label = label
            self._pos = _DummyPos(x, y)

        def data(self):
            return self._label

        def pos(self):
            return self._pos

    widget = OWQSARReportGenerator()
    try:
        point = _DummyPoint("C001", 5.2, 5.1)
        widget._on_hover_op([point], None, widget._op_hover)
        assert "C001" in widget._op_hover.text()
        assert "obs=5.200" in widget._op_hover.text()

        widget._on_hover_res([point], None, widget._res_hover)
        assert "C001" in widget._res_hover.text()
        assert "pred=5.200" in widget._res_hover.text()
    finally:
        widget.onDeleteWidget()
        widget.close()
