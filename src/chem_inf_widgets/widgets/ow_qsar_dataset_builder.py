from __future__ import annotations

import math
from typing import Any

import numpy as np
from AnyQt.QtCore import QThread, Qt, pyqtSignal
from AnyQt.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QSizePolicy, QVBoxLayout, QWidget,
)
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table
from chem_inf_widgets.chemcore.services.qsar_target_contract import build_qsar_ready_table
from chem_inf_widgets.chemcore.services.qsar_dataset_builder_service import (
    QSARDatasetBuilderConfig,
    QSARDatasetBuilderResult,
    build_qsar_dataset,
    smart_detect_columns,
)

_AUTO = "— auto-detect —"

# ── helpers ────────────────────────────────────────────────────────────────

def _is_nan(value: Any) -> bool:
    try:
        return isinstance(value, float) and math.isnan(value)
    except Exception:
        return False


def _orange_value_to_python(var, value: Any) -> Any:
    if _is_nan(value):
        return ""
    if hasattr(var, "values") and getattr(var, "values", None) and isinstance(value, (int, float, np.floating)):
        try:
            idx = int(value)
            if 0 <= idx < len(var.values):
                return var.values[idx]
        except Exception:
            pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _table_to_records(data: Table) -> list[dict[str, Any]]:
    variables = list(data.domain.attributes) + list(data.domain.class_vars) + list(data.domain.metas)
    columns = {var.name: data.get_column(var) for var in variables}
    rows: list[dict[str, Any]] = []
    n = len(data)
    for i in range(n):
        row: dict[str, Any] = {}
        for var in variables:
            col = columns[var.name]
            row[var.name] = _orange_value_to_python(var, col[i])
        rows.append(row)
    return rows


def _table_from_records(
    records: list[dict[str, Any]], *, class_column: str | None = None, name: str = "Data"
) -> Table | None:
    return records_to_orange_table(
        records, class_column=class_column, name=name, numeric_as_attributes=True
    )


class _QSARDatasetWorker(QThread):
    finished = pyqtSignal(object)
    failed   = pyqtSignal(str)

    def __init__(self, records, config, parent=None):
        super().__init__(parent)
        self.records = records
        self.config  = config

    def run(self) -> None:
        try:
            self.finished.emit(build_qsar_dataset(self.records, self.config))
        except Exception as exc:
            self.failed.emit(str(exc))


# ── small column-selector row ──────────────────────────────────────────────

class _ColRow(QWidget):
    """Label + QComboBox + confidence badge in one horizontal row."""

    def __init__(self, label: str, callback, parent=None):
        super().__init__(parent)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 2, 0, 2)
        hl.setSpacing(6)

        lbl = QLabel(label)
        lbl.setFixedWidth(72)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hl.addWidget(lbl)

        self.combo = QComboBox()
        self.combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo.setMinimumContentsLength(18)
        self.combo.currentIndexChanged.connect(callback)
        hl.addWidget(self.combo, 1)

        self.badge = QLabel()
        self.badge.setFixedWidth(60)
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setStyleSheet(
            "border-radius:8px; padding:1px 4px; font-size:10px; color:#6b7280;"
        )
        hl.addWidget(self.badge)

    def populate(self, cols: list[str], selected: str | None = None) -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(_AUTO)
        for c in cols:
            self.combo.addItem(c)
        if selected and selected in cols:
            self.combo.setCurrentText(selected)
        else:
            self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)

    def value(self) -> str | None:
        """Return selected column name, or None when auto-detect is chosen."""
        t = self.combo.currentText()
        return None if t == _AUTO else t

    def set_badge(self, conf: float, method: str, is_pact: bool = False) -> None:
        if is_pact:
            self.badge.setText("⚡ pAct")
            self.badge.setStyleSheet(
                "border-radius:8px; padding:1px 4px; font-size:10px;"
                "background:#fef3c7; color:#92400e;"
            )
            return
        if conf <= 0:
            self.badge.setText("")
            self.badge.setStyleSheet("border-radius:8px; padding:1px 4px;")
            return
        pct = int(conf * 100)
        tag = {"name": "N", "content": "C"}.get(method, "?")
        if conf >= 0.80:
            bg, fg = "#d1fae5", "#065f46"
        elif conf >= 0.50:
            bg, fg = "#fef3c7", "#92400e"
        else:
            bg, fg = "#fee2e2", "#991b1b"
        self.badge.setText(f"{tag} {pct}%")
        self.badge.setStyleSheet(
            f"border-radius:8px; padding:1px 4px; font-size:10px;"
            f"background:{bg}; color:{fg};"
        )

    def clear_badge(self) -> None:
        self.badge.setText("")
        self.badge.setStyleSheet("border-radius:8px; padding:1px 4px;")


# ── widget ─────────────────────────────────────────────────────────────────

class OWQSARDatasetBuilder(OWWidget):
    name        = "QSAR Dataset Builder"
    description = "Curate activity records into a QSAR-ready table with pActivity, duplicate aggregation, and a curation report."
    icon        = "icons/modeling/ow_qsar_dataset_builder.png"
    priority    = 130
    keywords    = ["QSAR", "dataset", "curation", "activity", "pActivity", "ChEMBL"]

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        qsar_ready_data  = Output("QSAR Ready Data",   Table)
        rejected_records = Output("Rejected Records",  Table)
        curation_report  = Output("Curation Report",   Table)
        dataset_summary  = Output("Dataset Summary",   Table)

    want_main_area = False

    # Column settings (persist selected names across sessions)
    smiles_column    = Setting("")
    name_column      = Setting("")
    activity_column  = Setting("")
    unit_column      = Setting("")
    relation_column  = Setting("")
    endpoint_column  = Setting("")

    target_endpoint      = Setting("")
    target_unit          = Setting("nM")
    relation_policy      = Setting(0)
    aggregation          = Setting(0)
    duplicate_key        = Setting(0)
    min_pactivity        = Setting(0.0)
    max_pactivity        = Setting(14.0)
    use_pactivity_range  = Setting(False)
    auto_run             = Setting(True)

    relation_options  = ["Exact values only", "Allow inequalities"]
    aggregation_options = ["median", "mean", "min", "max", "first"]
    duplicate_options = ["standard_inchikey", "canonical_smiles", "raw_smiles"]

    def __init__(self):
        super().__init__()
        self.data:   Table | None              = None
        self.worker: _QSARDatasetWorker | None = None
        self._cols:  list[str]                 = []

        ca = self.controlArea

        # ── Input info ────────────────────────────────────────────────────
        info_box = gui.widgetBox(ca, "Input")
        self._input_label = gui.label(info_box, self, "No data on input.")

        # ── Column selectors ──────────────────────────────────────────────
        col_box = QGroupBox("Columns")
        col_vl  = QVBoxLayout(col_box)
        col_vl.setSpacing(0)
        col_vl.setContentsMargins(8, 6, 8, 6)

        self._row_smiles   = _ColRow("SMILES",    self._on_col_changed)
        self._row_name     = _ColRow("Name / ID", self._on_col_changed)
        self._row_activity = _ColRow("Activity",  self._on_col_changed)
        self._row_unit     = _ColRow("Unit",      self._on_col_changed)
        self._row_relation = _ColRow("Relation",  self._on_col_changed)
        self._row_endpoint = _ColRow("Endpoint",  self._on_col_changed)

        for row in (self._row_smiles, self._row_name, self._row_activity,
                    self._row_unit,   self._row_relation, self._row_endpoint):
            col_vl.addWidget(row)

        # Re-detect / Clear buttons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        from AnyQt.QtWidgets import QPushButton
        self._btn_redetect = QPushButton("Re-detect")
        self._btn_redetect.setFixedHeight(24)
        self._btn_redetect.clicked.connect(self._redetect)
        self._btn_clear = QPushButton("Clear all")
        self._btn_clear.setFixedHeight(24)
        self._btn_clear.clicked.connect(self._clear_columns)
        btn_row.addWidget(self._btn_redetect)
        btn_row.addWidget(self._btn_clear)
        col_vl.addLayout(btn_row)

        # Detection status line
        self._det_status = QLabel("")
        self._det_status.setWordWrap(True)
        self._det_status.setStyleSheet("font-size:10px; color:#6b7280; padding:2px 0;")
        col_vl.addWidget(self._det_status)

        ca.layout().addWidget(col_box)

        # ── Curation ──────────────────────────────────────────────────────
        curation = gui.widgetBox(ca, "Curation")
        gui.lineEdit(curation, self, "target_endpoint", label="Endpoint filter:",
                     orientation="horizontal", labelWidth=100, callback=self._settings_changed)
        gui.lineEdit(curation, self, "target_unit", label="Target unit:",
                     orientation="horizontal", labelWidth=100, callback=self._settings_changed)
        gui.comboBox(curation, self, "relation_policy", label="Relations:",
                     orientation="horizontal", labelWidth=100,
                     items=self.relation_options, callback=self._settings_changed)
        gui.comboBox(curation, self, "aggregation", label="Aggregation:",
                     orientation="horizontal", labelWidth=100,
                     items=self.aggregation_options, callback=self._settings_changed)
        gui.comboBox(curation, self, "duplicate_key", label="Dup. key:",
                     orientation="horizontal", labelWidth=100,
                     items=self.duplicate_options, callback=self._settings_changed)

        pact_row = gui.hBox(curation)
        gui.checkBox(pact_row, self, "use_pactivity_range", "pActivity range",
                     callback=self._pactivity_toggled)
        self._min_spin = gui.doubleSpin(pact_row, self, "min_pactivity",
                                        minv=0.0, maxv=20.0, step=0.5, label="min",
                                        callback=self._settings_changed)
        pact_row.layout().addWidget(QLabel("–"))
        self._max_spin = gui.doubleSpin(pact_row, self, "max_pactivity",
                                        minv=0.0, maxv=20.0, step=0.5, label="max",
                                        callback=self._settings_changed)
        self._min_spin.setEnabled(self.use_pactivity_range)
        self._max_spin.setEnabled(self.use_pactivity_range)

        # ── Run ───────────────────────────────────────────────────────────
        run_box = gui.hBox(ca)
        gui.checkBox(run_box, self, "auto_run", "Auto-run",
                     callback=self._settings_changed)
        gui.button(run_box, self, "Build QSAR dataset", callback=self.commit)

        self._status_label = gui.label(ca, self, "Ready.")

    # ── Column-combo helpers ──────────────────────────────────────────────

    def _col_rows(self):
        return [
            (self._row_smiles,   "smiles_column"),
            (self._row_name,     "name_column"),
            (self._row_activity, "activity_column"),
            (self._row_unit,     "unit_column"),
            (self._row_relation, "relation_column"),
            (self._row_endpoint, "endpoint_column"),
        ]

    def _populate_combos(self, cols: list[str], selected: dict[str, str | None]) -> None:
        """Fill all combos with cols, choosing the given selected name per row."""
        for row, attr in self._col_rows():
            row.combo.blockSignals(True)
            row.populate(cols, selected.get(attr))
            row.combo.blockSignals(False)

    def _read_combos(self) -> None:
        """Sync combo selections back to Setting attributes."""
        for row, attr in self._col_rows():
            setattr(self, attr, row.value() or "")

    def _on_col_changed(self) -> None:
        self._read_combos()
        self._settings_changed()

    # ── Input handler ────────────────────────────────────────────────────

    @Inputs.data
    def set_data(self, data: Table | None) -> None:
        self.data = data
        if data is None:
            self._input_label.setText("No data on input.")
            self._det_status.setText("")
            for row, _ in self._col_rows():
                row.combo.blockSignals(True)
                row.combo.clear()
                row.combo.addItem(_AUTO)
                row.combo.blockSignals(False)
                row.clear_badge()
            self._cols = []
            self._clear_outputs()
            return

        self._cols = [
            v.name for v in
            list(data.domain.attributes) + list(data.domain.class_vars) + list(data.domain.metas)
        ]
        preview = ", ".join(self._cols[:6]) + ("…" if len(self._cols) > 6 else "")
        self._input_label.setText(f"{len(data)} rows × {len(self._cols)} cols — {preview}")

        records = _table_to_records(data)
        self._run_detection(records, force=False)

        if self.auto_run:
            self.commit()

    # ── Auto-detection ────────────────────────────────────────────────────

    def _run_detection(self, records: list, *, force: bool = True) -> None:
        """Run smart_detect_columns, populate combos, update badges."""
        det = smart_detect_columns(self._cols, records)

        # Fields to detect; only override empty settings (unless force=True)
        _field_attr = [
            ("smiles_column",   "smiles_column"),
            ("name_column",     "name_column"),
            ("activity_column", "activity_column"),
            ("unit_column",     "unit_column"),
            ("relation_column", "relation_column"),
            ("endpoint_column", "endpoint_column"),
        ]
        selected: dict[str, str | None] = {}
        for field, attr in _field_attr:
            if force or not getattr(self, attr):
                val = det.get(field)
                if val:
                    setattr(self, attr, val)
            selected[attr] = getattr(self, attr) or None

        # Populate all combos
        self._populate_combos(self._cols, selected)

        # Set badges
        is_pact = det.get("is_pactivity", False)
        for field, attr in _field_attr:
            row = getattr(self, f"_row_{attr.replace('_column', '')}")
            conf   = det.get(f"{field}_confidence", 0.0)
            method = det.get(f"{field}_method", "none")
            use_pact_badge = (field == "activity_column" and is_pact)
            row.set_badge(conf, method, is_pact=use_pact_badge)

        # Auto-populate target endpoint & unit
        if force or not self.target_endpoint:
            ep = det.get("suggested_endpoint") or ""
            if ep:
                self.target_endpoint = ep
                self.controls.target_endpoint.setText(ep)

        if force or self.target_unit == "nM":
            su = det.get("suggested_unit") or ""
            if su and su.strip().lower() not in ("", "n/a", "none"):
                self.target_unit = su
                self.controls.target_unit.setText(su)

        # Status line
        found = sum(1 for _, attr in _field_attr if getattr(self, attr))
        pact_note = " · ⚡ pre-computed pActivity" if is_pact else ""
        self._det_status.setText(
            f"Auto-detected {found}/6 columns{pact_note}"
        )

    def _redetect(self) -> None:
        if self.data is None:
            return
        for _, attr in self._col_rows():
            setattr(self, attr, "")
        self.target_endpoint = ""
        self.target_unit = "nM"
        self._run_detection(_table_to_records(self.data), force=True)

    def _clear_columns(self) -> None:
        for row, attr in self._col_rows():
            setattr(self, attr, "")
            row.combo.blockSignals(True)
            row.combo.setCurrentIndex(0)
            row.combo.blockSignals(False)
            row.clear_badge()
        self._det_status.setText("All column fields cleared.")

    # ── Settings / commit ─────────────────────────────────────────────────

    def _pactivity_toggled(self) -> None:
        self._min_spin.setEnabled(self.use_pactivity_range)
        self._max_spin.setEnabled(self.use_pactivity_range)
        self._settings_changed()

    def _settings_changed(self) -> None:
        if self.auto_run and self.data is not None:
            self.commit()

    def _config(self) -> QSARDatasetBuilderConfig:
        self._read_combos()
        return QSARDatasetBuilderConfig(
            smiles_column    = self.smiles_column.strip()   or None,
            name_column      = self.name_column.strip()     or None,
            activity_column  = self.activity_column.strip() or None,
            unit_column      = self.unit_column.strip()     or None,
            relation_column  = self.relation_column.strip() or None,
            endpoint_column  = self.endpoint_column.strip() or None,
            target_endpoint  = self.target_endpoint.strip(),
            target_unit      = self.target_unit.strip() or "nM",
            relation_policy  = "exact_only" if self.relation_policy == 0 else "allow_inequalities",
            aggregation      = self.aggregation_options[self.aggregation],
            duplicate_key    = self.duplicate_options[self.duplicate_key],
            min_pactivity    = self.min_pactivity if self.use_pactivity_range else None,
            max_pactivity    = self.max_pactivity if self.use_pactivity_range else None,
        )

    def commit(self) -> None:
        if self.data is None:
            self._clear_outputs()
            return
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(200)
        records = _table_to_records(self.data)
        self._status_label.setText("Building QSAR-ready dataset…")
        self.worker = _QSARDatasetWorker(records, self._config())
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, result: QSARDatasetBuilderResult) -> None:
        ready    = build_qsar_ready_table(result.prepared_records, name="QSAR Ready Data")
        rejected = _table_from_records(result.rejected_records, name="Rejected Records")
        report   = _table_from_records(result.curation_report,  name="Curation Report")
        summary  = _table_from_records([result.summary],        name="Dataset Summary")
        self.Outputs.qsar_ready_data.send(ready)
        self.Outputs.rejected_records.send(rejected)
        self.Outputs.curation_report.send(report)
        self.Outputs.dataset_summary.send(summary)
        self._status_label.setText(
            f"Prepared {result.summary.get('prepared_compounds', 0)} compounds; "
            f"rejected {result.summary.get('rejected_records', 0)} records; "
            f"duplicate groups {result.summary.get('duplicate_groups', 0)}."
        )

    def _on_failed(self, message: str) -> None:
        self._status_label.setText(f"Failed: {message}")
        self._clear_outputs()

    def _clear_outputs(self) -> None:
        self.Outputs.qsar_ready_data.send(None)
        self.Outputs.rejected_records.send(None)
        self.Outputs.curation_report.send(None)
        self.Outputs.dataset_summary.send(None)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWQSARDatasetBuilder).run()
