from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication

from chem_inf_widgets.widgets.ow_model_explanation import (
    OWModelExplanation,
    _df_to_table,
)


_APP = QApplication.instance() or QApplication([])


def _demo_df(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    x1 = np.linspace(0.0, 1.0, n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 5.0 + 1.1 * x1 - 0.5 * x2 + rng.normal(0.0, 0.05, size=n)
    return pd.DataFrame(
        {
            "compound_id": [f"C{i:03d}" for i in range(n)],
            "pActivity": y,
            "desc_1": x1,
            "desc_2": x2,
            "desc_3": x3,
        }
    )


def _wait_until_finished(widget: OWModelExplanation, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        _APP.processEvents()
        worker = widget._worker
        if worker is None or not worker.isRunning():
            _APP.processEvents()
            return
        time.sleep(0.01)
    raise AssertionError("Model explanation worker did not finish in time.")


def test_model_explanation_widget_exposes_full_method_list():
    widget = OWModelExplanation()
    try:
        methods = [widget._method_combo.itemText(i) for i in range(widget._method_combo.count())]
        assert methods == [
            "Auto",
            "Model importance",
            "Coefficient",
            "Permutation",
            "Univariate",
        ]
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_model_explanation_widget_runs_with_fallback_model():
    widget = OWModelExplanation()
    try:
        widget.set_data(_df_to_table(_demo_df()))
        widget.commit()
        _wait_until_finished(widget)

        status = widget._status_chip.text().lower()
        assert "model_importance" in status or "coefficient" in status or "permutation" in status or "univariate" in status
        assert "Model Explanation Summary" in widget._summary_browser.toPlainText()
        assert "desc_" in widget._summary_browser.toPlainText()
        assert "Approximate local contributions" in widget._local_browser.toPlainText()
        assert "C000" in widget._local_browser.toPlainText()
    finally:
        widget.onDeleteWidget()
        widget.close()
