from __future__ import annotations

import os
from typing import Optional, Sequence

from AnyQt.QtCore import pyqtSlot
from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)
from Orange.data import Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.molecule_export_service import (
    MoleculeExportConfig,
    MoleculeExportResult,
    export_molecule_data,
    export_records_as_dicts,
    export_summary_as_rows,
)
from chem_inf_widgets.chemcore.services.orange_table_utils import format_domain_role_summary
from chem_inf_widgets.chemcore.services.report_table_utils import report_rows_to_table, summary_rows_to_table
from chem_inf_widgets.widgets.ui_helpers import format_done_status, format_failed_status, format_no_input_status


class OWMoleculeExportHub(OWWidget):
    name = "Molecule Export Hub"
    description = "Export molecules from Data or Molecules input to CSV/TSV/TXT/SMI/SDF with an audit report."
    icon = "icons/input_output/owmolexport.png"
    priority = 101

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        export_report = Output("Export Report", Table)
        failed_records = Output("Failed Records", Table)
        export_summary = Output("Export Summary", Table)

    output_path = Setting("")
    output_format = Setting("auto")
    smiles_column = Setting("")
    name_column = Setting("")
    delimiter = Setting("")
    sanitize = Setting(True)
    include_props = Setting(True)
    write_name = Setting(True)
    include_header = Setting(True)
    use_canonical_smiles = Setting(True)
    auto_run = Setting(False)

    def __init__(self) -> None:
        super().__init__()
        self.executor = ThreadExecutor(self)
        self._data: Optional[Table] = None
        self._molecules: Optional[list[ChemMol]] = None
        self._build_ui()
        self._set_status("Connect Data or Molecules and choose an output file.")
        self._refresh_preview()

    def _build_ui(self) -> None:
        self.mainArea.hide()
        root = self.controlArea
        root.setMinimumWidth(410)

        file_box = QGroupBox("Output file")
        file_layout = QVBoxLayout(file_box)
        self.path_edit = QLineEdit(self.output_path)
        self.path_edit.setPlaceholderText("Path to .csv, .tsv, .txt, .smi, .smiles, .sdf, or .sd")
        self.path_edit.textChanged.connect(self._path_changed)
        file_layout.addWidget(self.path_edit)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        file_layout.addWidget(self.btn_browse)
        root.layout().addWidget(file_box)

        export_box = QGroupBox("Export mapping")
        export_form = QFormLayout(export_box)
        self.format_combo = QComboBox()
        self.format_combo.addItem("Auto-detect from file extension", "auto")
        self.format_combo.addItem("CSV", "csv")
        self.format_combo.addItem("TSV", "tsv")
        self.format_combo.addItem("TXT", "txt")
        self.format_combo.addItem("SMILES / SMI", "smi")
        self.format_combo.addItem("SDF", "sdf")
        index = self.format_combo.findData(self.output_format or "auto")
        self.format_combo.setCurrentIndex(max(index, 0))
        self.format_combo.currentIndexChanged.connect(self._settings_changed)

        self.smiles_edit = QLineEdit(self.smiles_column)
        self.smiles_edit.setPlaceholderText("Auto-detect for Data input")
        self.smiles_edit.textChanged.connect(self._settings_changed)

        self.name_edit = QLineEdit(self.name_column)
        self.name_edit.setPlaceholderText("Auto-detect for Data input")
        self.name_edit.textChanged.connect(self._settings_changed)

        self.delim_edit = QLineEdit(self.delimiter)
        self.delim_edit.setPlaceholderText("Optional override; use \\t for tab")
        self.delim_edit.textChanged.connect(self._settings_changed)

        export_form.addRow("Format", self.format_combo)
        export_form.addRow("SMILES column", self.smiles_edit)
        export_form.addRow("Name column", self.name_edit)
        export_form.addRow("Delimiter", self.delim_edit)
        root.layout().addWidget(export_box)

        options_box = QGroupBox("Export options")
        options_layout = QVBoxLayout(options_box)
        self.cb_sanitize = QCheckBox("Sanitize molecules when converting Data input")
        self.cb_sanitize.setChecked(bool(self.sanitize))
        self.cb_include_props = QCheckBox("Include properties / metadata")
        self.cb_include_props.setChecked(bool(self.include_props))
        self.cb_write_name = QCheckBox("Write name / identifier column")
        self.cb_write_name.setChecked(bool(self.write_name))
        self.cb_include_header = QCheckBox("Write header row for text exports")
        self.cb_include_header.setChecked(bool(self.include_header))
        self.cb_canonical = QCheckBox("Prefer canonical SMILES in text exports")
        self.cb_canonical.setChecked(bool(self.use_canonical_smiles))
        self.cb_auto = QCheckBox("Auto-run")
        self.cb_auto.setChecked(bool(self.auto_run))
        for checkbox in (
            self.cb_sanitize,
            self.cb_include_props,
            self.cb_write_name,
            self.cb_include_header,
            self.cb_canonical,
            self.cb_auto,
        ):
            checkbox.stateChanged.connect(self._settings_changed)
            options_layout.addWidget(checkbox)
        root.layout().addWidget(options_box)

        self.btn_export = QPushButton("Export molecules")
        self.btn_export.clicked.connect(self._run)
        root.layout().addWidget(self.btn_export)

        self.lbl = QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#475467;")
        root.layout().addWidget(self.lbl)

        preview_box = QGroupBox("Export payload")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_label = QLabel("Connect Data or Molecules to inspect the active export payload.")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("color:#475467;")
        preview_layout.addWidget(self.preview_label)
        root.layout().addWidget(preview_box)

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._data = data
        self._refresh_preview()
        if self.auto_run and self.output_path and self._has_input():
            self._run()

    @Inputs.molecules
    def set_molecules(self, molecules: Optional[list]) -> None:
        self._molecules = molecules
        self._refresh_preview()
        if self.auto_run and self.output_path and self._has_input():
            self._run()

    def _path_changed(self, value: str) -> None:
        self.output_path = value
        if self.auto_run and value and self._has_input():
            self._run()

    def _settings_changed(self) -> None:
        self.output_format = str(self.format_combo.currentData() or "auto")
        self.smiles_column = self.smiles_edit.text().strip()
        self.name_column = self.name_edit.text().strip()
        self.delimiter = self.delim_edit.text()
        self.sanitize = bool(self.cb_sanitize.isChecked())
        self.include_props = bool(self.cb_include_props.isChecked())
        self.write_name = bool(self.cb_write_name.isChecked())
        self.include_header = bool(self.cb_include_header.isChecked())
        self.use_canonical_smiles = bool(self.cb_canonical.isChecked())
        self.auto_run = bool(self.cb_auto.isChecked())
        self._refresh_preview()

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select export file",
            self.output_path or "molecules_export.sdf",
            "Molecule exports (*.csv *.tsv *.txt *.smi *.smiles *.sdf *.sd);;All files (*)",
        )
        if path:
            self.output_path = path
            self.path_edit.setText(path)

    def _set_status(self, text: str) -> None:
        self.lbl.setText(text)

    def _set_busy(self, busy: bool, text: str) -> None:
        self.btn_export.setEnabled(not busy)
        self.btn_browse.setEnabled(not busy)
        self._set_status(text)
        if busy:
            self.progressBarInit()
        else:
            self.progressBarFinished()

    def _config(self) -> MoleculeExportConfig:
        return MoleculeExportConfig(
            output_format=self.output_format or None,
            smiles_column=self.smiles_column or None,
            name_column=self.name_column or None,
            delimiter=self.delimiter or None,
            sanitize=bool(self.sanitize),
            include_props=bool(self.include_props),
            write_name=bool(self.write_name),
            include_header=bool(self.include_header),
            use_canonical_smiles=bool(self.use_canonical_smiles),
        )

    def _active_source(self) -> tuple[Optional[Table], Optional[Sequence[ChemMol]], str]:
        if self._molecules is not None:
            return None, self._molecules, "Molecules"
        if self._data is not None:
            return self._data, None, "Data"
        return None, None, ""

    def _has_input(self) -> bool:
        return self._molecules is not None or self._data is not None

    def _run(self) -> None:
        self._settings_changed()
        path = (self.output_path or "").strip()
        if not path:
            self._set_status(format_no_input_status("output file path"))
            self._send_empty()
            return
        data, molecules, _source_label = self._active_source()
        if data is None and molecules is None:
            self._set_status(format_no_input_status("input data or molecules"))
            self._send_empty()
            return
        self._set_busy(True, "Exporting molecules…")
        future = self.executor.submit(self._run_background, path, self._config(), data, molecules)
        future.add_done_callback(self._on_done)

    def _run_background(
        self,
        path: str,
        config: MoleculeExportConfig,
        data: Optional[Table],
        molecules: Optional[Sequence[ChemMol]],
    ) -> tuple[Table, Table, Table, MoleculeExportResult]:
        result = export_molecule_data(path, data=data, molecules=molecules, config=config)
        report = self._records_to_table(result.records)
        failed = self._records_to_table(result.failed_records)
        summary = self._summary_to_table(result.summary)
        return report, failed, summary, result

    def _on_done(self, future) -> None:
        try:
            payload = future.result()
            methodinvoke(self, "_apply_outputs", (object,))(payload)
        except Exception as exc:
            methodinvoke(self, "_apply_error", (str,))(str(exc))

    @pyqtSlot(str)
    def _apply_error(self, message: str) -> None:
        self._set_busy(False, format_failed_status(message))
        self._send_empty()

    @pyqtSlot(object)
    def _apply_outputs(self, payload: object) -> None:
        report, failed, summary, result = payload
        self._set_busy(False, format_done_status(
            f"written={result.summary.written_records}/{result.summary.total_records}",
            f"failed={result.summary.failed_records}",
            f"format={result.summary.output_format}",
            f"file={os.path.basename(result.summary.output_path)}",
            prefix="Export complete",
        ))
        self.preview_label.setText(self._format_export_columns(result.summary))
        self.Outputs.export_report.send(report)
        self.Outputs.failed_records.send(failed)
        self.Outputs.export_summary.send(summary)

    def _send_empty(self) -> None:
        self.Outputs.export_report.send(None)
        self.Outputs.failed_records.send(None)
        self.Outputs.export_summary.send(None)

    def _refresh_preview(self) -> None:
        if self._molecules is not None:
            keys = sorted({str(key) for molecule in self._molecules for key in (molecule.props or {}).keys()}) if self._molecules else []
            shown = ", ".join(keys[:10]) if keys else "none"
            if len(keys) > 10:
                shown += f", … (+{len(keys) - 10})"
            extra = "\nActive source: Molecules input takes precedence when both inputs are connected." if self._data is not None else ""
            self.preview_label.setText(
                f"Input molecules: {len(self._molecules)}\n"
                f"Available props: {shown}{extra}"
            )
            return

        if self._data is not None:
            self.preview_label.setText(format_domain_role_summary(self._data.domain))
            return

        self.preview_label.setText("Connect Data or Molecules to inspect the active export payload.")

    @staticmethod
    def _records_to_table(records) -> Table:
        rows = export_records_as_dicts(records)
        return report_rows_to_table(
            rows,
            numeric_columns=["row_index", "ok", "written"],
            meta_columns=[
                "source_kind",
                "source_name",
                "name",
                "mol_id",
                "input_smiles",
                "canonical_smiles",
                "output_smiles",
                "inchikey",
                "status",
                "error",
                "props_json",
            ],
            name="Export Report",
        )

    @staticmethod
    def _summary_to_table(summary) -> Table:
        return summary_rows_to_table(
            export_summary_as_rows(summary),
            name="Export Summary",
        )

    @staticmethod
    def _format_export_columns(summary) -> str:
        columns = ", ".join(summary.columns) if summary.columns else "none"
        return (
            "Export payload:\n"
            f"Format: {summary.output_format}\n"
            f"Columns/properties: {columns}\n"
            f"Source: {summary.source_kind}"
        )


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWMoleculeExportHub).run()
