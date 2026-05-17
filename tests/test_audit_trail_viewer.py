from __future__ import annotations

import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication

from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table
from chem_inf_widgets.widgets.ow_audit_trail_viewer import (
    OWAuditTrailViewer,
    _summary_rows,
    _table_records,
)


_APP = QApplication.instance() or QApplication([])


def _audit_table():
    records = [
        {
            "compound_id": "A",
            "row_id": "row-a",
            "transform_log": "import_csv|molecule_qc",
            "qc_flags": "duplicate_structure",
            "dropped_reason": "",
        },
        {
            "compound_id": "B",
            "row_id": "row-b",
            "transform_log": "import_csv|molecule_qc|standardize_qsar_ready",
            "qc_flags": "",
            "dropped_reason": "invalid_structure",
        },
        {
            "compound_id": "C",
            "source_row_ids": "row-c1;row-c2",
            "source_transform_logs": "import_csv|molecule_qc|standardize_qsar_ready",
            "source_qc_flags_all": "multi_fragment|duplicate_structure",
            "source_dropped_reasons": "",
        },
    ]
    return records_to_orange_table(records, meta_columns=list(records[0].keys()) + list(records[2].keys()), name="Audit Input")


def test_audit_trail_viewer_helpers_read_direct_and_aggregated_columns():
    table = _audit_table()
    rows = _table_records(table)

    assert len(rows) == 3
    assert rows[0]["row_id"] == "row-a"
    assert rows[1]["dropped"] == "1"
    assert rows[2]["row_id"] == "row-c1;row-c2"
    assert rows[2]["qc_flags"] == "multi_fragment|duplicate_structure"

    summary = _summary_rows(rows, filtered_rows=2)
    values = {row["metric"]: row["value"] for row in summary}
    assert values["total_rows"] == 3
    assert values["flagged_rows"] == 3
    assert values["filtered_rows"] == 2


def test_audit_trail_viewer_filters_rows_and_updates_table():
    widget = OWAuditTrailViewer()
    try:
        widget.set_data(_audit_table())
        assert widget.results_table.rowCount() == 3

        widget.mode_combo.setCurrentIndex(1)  # flagged
        widget._on_filter_changed()
        assert widget.results_table.rowCount() == 3

        widget.mode_combo.setCurrentIndex(2)  # dropped
        widget._on_filter_changed()
        assert widget.results_table.rowCount() == 1
        assert widget.results_table.item(0, 3).text() == "invalid_structure"

        widget.mode_combo.setCurrentIndex(0)
        widget.search_edit.setText("multi_fragment")
        widget._on_filter_changed()
        assert widget.results_table.rowCount() == 1
        assert widget.results_table.item(0, 1).text() == "row-c1;row-c2"
        assert "Audit view" in widget.status_label.text()
    finally:
        widget.onDeleteWidget()
        widget.close()
