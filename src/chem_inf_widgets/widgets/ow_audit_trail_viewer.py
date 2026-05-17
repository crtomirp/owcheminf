from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from Orange.data import Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.report_table_utils import summary_rows_to_table


DIRECT_AUDIT_COLUMNS = ("row_id", "transform_log", "qc_flags", "dropped_reason")
FALLBACK_AUDIT_COLUMNS = (
    ("source_row_ids", "source_transform_logs", "source_qc_flags_all", "source_dropped_reasons"),
    ("source_row_id", "source_transform_log", "source_qc_flags", "source_dropped_reason"),
)


def _table_column_names(data: Optional[Table]) -> list[str]:
    if data is None:
        return []
    return [
        *(var.name for var in data.domain.attributes),
        *(var.name for var in data.domain.class_vars),
        *(var.name for var in data.domain.metas),
    ]


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"", "?", "None", "nan"}:
        return ""
    return text


def _table_records(data: Optional[Table]) -> list[dict[str, str]]:
    if data is None:
        return []
    columns = _table_column_names(data)
    rows: list[dict[str, str]] = []
    for row_idx in range(len(data)):
        row_map = {}
        for name in columns:
            try:
                row_map[name] = _clean_text(data[row_idx][name])
            except Exception:
                row_map[name] = ""

        audit = {key: row_map.get(key, "") for key in DIRECT_AUDIT_COLUMNS}
        if not any(audit.values()):
            for fallback in FALLBACK_AUDIT_COLUMNS:
                candidate = {key: row_map.get(source, "") for key, source in zip(DIRECT_AUDIT_COLUMNS, fallback)}
                if any(candidate.values()):
                    audit = candidate
                    break

        transform_steps = [part for part in str(audit["transform_log"]).split("|") if part]
        dropped = bool(str(audit["dropped_reason"]).strip())
        flagged = bool(str(audit["qc_flags"]).strip()) or dropped
        rows.append(
            {
                "row_index": str(row_idx + 1),
                "row_id": str(audit["row_id"]).strip(),
                "transform_log": str(audit["transform_log"]).strip(),
                "qc_flags": str(audit["qc_flags"]).strip(),
                "dropped_reason": str(audit["dropped_reason"]).strip(),
                "flagged": "1" if flagged else "0",
                "dropped": "1" if dropped else "0",
                "provenance_missing": "1" if not (audit["row_id"] or audit["transform_log"]) else "0",
                "n_steps": str(len(transform_steps)),
            }
        )
    return rows


def _summary_rows(records: Iterable[dict[str, str]], filtered_rows: int) -> list[dict[str, object]]:
    rows = list(records)
    steps = Counter(
        step
        for row in rows
        for step in str(row.get("transform_log", "")).split("|")
        if step
    )
    return [
        {"metric": "total_rows", "value": len(rows), "description": "Rows available on input."},
        {"metric": "flagged_rows", "value": sum(row.get("flagged") == "1" for row in rows), "description": "Rows with qc_flags or dropped_reason."},
        {"metric": "dropped_rows", "value": sum(row.get("dropped") == "1" for row in rows), "description": "Rows with a dropped_reason."},
        {"metric": "missing_provenance_rows", "value": sum(row.get("provenance_missing") == "1" for row in rows), "description": "Rows missing both row_id and transform_log."},
        {"metric": "filtered_rows", "value": int(filtered_rows), "description": "Rows visible after the current filter."},
        {"metric": "unique_transform_steps", "value": len(steps), "description": "Distinct transform steps found in transform_log."},
        {"metric": "top_transform_step", "value": steps.most_common(1)[0][0] if steps else "", "description": "Most frequent transform_log step."},
    ]


class OWAuditTrailViewer(OWWidget):
    name = "Audit Trail Viewer"
    description = "Inspect row_id, transform_log, qc_flags, and dropped_reason across workflow tables."
    icon = "icons/standardization_filtering/owaudittrailviewerwidget.svg"
    priority = 106

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        filtered_data = Output("Filtered Data", Table)
        flagged_data = Output("Flagged Data", Table)
        dropped_data = Output("Dropped Data", Table)
        audit_summary = Output("Audit Summary", Table)

    want_main_area = True

    filter_mode = Setting("all")
    search_text = Setting("")

    FILTER_OPTIONS = (
        ("all", "All Rows"),
        ("flagged", "Flagged Rows"),
        ("dropped", "Dropped Rows"),
        ("missing", "Missing Provenance"),
    )

    def __init__(self) -> None:
        super().__init__()
        self._data: Optional[Table] = None
        self._records: list[dict[str, str]] = []
        self._build_ui()
        self._set_status("Connect a table to inspect workflow provenance.")

    def _build_ui(self) -> None:
        root = self.controlArea
        root.setMinimumWidth(360)

        box = QGroupBox("Audit Filter")
        layout = QVBoxLayout(box)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        for key, label in self.FILTER_OPTIONS:
            self.mode_combo.addItem(label, key)
        self.mode_combo.setCurrentIndex(next((i for i, (key, _) in enumerate(self.FILTER_OPTIONS) if key == self.filter_mode), 0))
        self.mode_combo.currentIndexChanged.connect(self._on_filter_changed)
        mode_row.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_row)

        self.search_edit = QLineEdit(self.search_text)
        self.search_edit.setPlaceholderText("Search qc_flags, dropped_reason, or transform_log")
        self.search_edit.textChanged.connect(self._on_filter_changed)
        layout.addWidget(self.search_edit)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color:#475467;")
        layout.addWidget(self.status_label)

        root.layout().addWidget(box)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["Row", "Row ID", "QC Flags", "Dropped", "Transform Log"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.mainArea.layout().addWidget(self.results_table)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._data = data
        self._records = _table_records(data)
        self._rebuild()

    def _on_filter_changed(self) -> None:
        self.filter_mode = str(self.mode_combo.currentData() or "all")
        self.search_text = str(self.search_edit.text() or "")
        self._rebuild()

    def _filtered_indices(self) -> list[int]:
        mode = str(self.filter_mode or "all")
        token = str(self.search_text or "").strip().lower()
        indices: list[int] = []
        for idx, row in enumerate(self._records):
            if mode == "flagged" and row.get("flagged") != "1":
                continue
            if mode == "dropped" and row.get("dropped") != "1":
                continue
            if mode == "missing" and row.get("provenance_missing") != "1":
                continue
            haystack = " ".join(
                [
                    row.get("row_id", ""),
                    row.get("qc_flags", ""),
                    row.get("dropped_reason", ""),
                    row.get("transform_log", ""),
                ]
            ).lower()
            if token and token not in haystack:
                continue
            indices.append(idx)
        return indices

    def _rebuild(self) -> None:
        if self._data is None:
            self.results_table.setRowCount(0)
            self.Outputs.filtered_data.send(None)
            self.Outputs.flagged_data.send(None)
            self.Outputs.dropped_data.send(None)
            self.Outputs.audit_summary.send(None)
            self._set_status("No data.")
            return

        filtered_indices = self._filtered_indices()
        self._populate_table(filtered_indices)
        self.Outputs.filtered_data.send(self._subset(self._data, filtered_indices))
        self.Outputs.flagged_data.send(self._subset(self._data, [i for i, row in enumerate(self._records) if row.get("flagged") == "1"]))
        self.Outputs.dropped_data.send(self._subset(self._data, [i for i, row in enumerate(self._records) if row.get("dropped") == "1"]))
        self.Outputs.audit_summary.send(summary_rows_to_table(_summary_rows(self._records, len(filtered_indices)), name="Audit Summary"))
        self._set_status(
            f"Audit view: {len(filtered_indices)}/{len(self._records)} rows visible, "
            f"{sum(row.get('flagged') == '1' for row in self._records)} flagged, "
            f"{sum(row.get('dropped') == '1' for row in self._records)} dropped."
        )

    def _populate_table(self, indices: list[int]) -> None:
        self.results_table.setRowCount(len(indices))
        for row_idx, src_idx in enumerate(indices):
            row = self._records[src_idx]
            values = [
                row.get("row_index", ""),
                row.get("row_id", ""),
                row.get("qc_flags", ""),
                row.get("dropped_reason", ""),
                row.get("transform_log", ""),
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col_idx == 0:
                    item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row_idx, col_idx, item)
        self.results_table.resizeColumnsToContents()

    @staticmethod
    def _subset(data: Optional[Table], indices: list[int]) -> Optional[Table]:
        if data is None:
            return None
        if not indices:
            return Table.from_numpy(
                data.domain,
                X=np.empty((0, len(data.domain.attributes)), dtype=float),
                Y=np.empty((0, len(data.domain.class_vars)), dtype=float) if data.domain.class_vars else None,
                metas=np.empty((0, len(data.domain.metas)), dtype=object),
            )
        return data[indices]
