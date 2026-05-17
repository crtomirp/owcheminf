from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
from AnyQt.QtCore import pyqtSlot
from AnyQt.QtWidgets import QFileDialog, QCheckBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QVBoxLayout
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.from_orange import chemmols_to_table
from chem_inf_widgets.chemcore.services.curation_summary import curation_summary_to_table, summary_from_import
from chem_inf_widgets.chemcore.services.orange_table_utils import format_domain_role_summary
from chem_inf_widgets.chemcore.services.molecule_import_service import (
    MoleculeImportConfig,
    MoleculeImportResult,
    import_molecule_file,
    import_records_as_dicts,
    import_summary_as_rows,
)
from chem_inf_widgets.chemcore.services.report_table_utils import report_rows_to_table, summary_rows_to_table
from chem_inf_widgets.widgets.ui_helpers import format_done_status, format_failed_status, format_no_input_status


class OWMoleculeImportHub(OWWidget):
    name = "Molecule Import Hub"
    description = "Import molecules from CSV/TSV/SMI/TXT/SDF with automatic structure-column detection and an import report."
    icon = "icons/input_output/owmolimport.png"
    priority = 100

    class Outputs:
        data = Output("Data", Table)
        molecules = Output("Molecules", list, auto_summary=False)
        accepted_data = Output("Accepted Data", Table)
        accepted_molecules = Output("Accepted Molecules", list, auto_summary=False)
        rejected_records = Output("Rejected Records", Table)
        import_report = Output("Import Report", Table)
        failed_records = Output("Failed Records", Table)
        import_summary = Output("Import Summary", Table)
        curation_summary = Output("Curation Summary", Table)

    file_path = Setting("")
    smiles_column = Setting("")
    name_column = Setting("")
    delimiter = Setting("")
    sanitize = Setting(True)
    remove_hs = Setting(True)
    flag_duplicates = Setting(True)
    reject_duplicate_structures = Setting(False)
    auto_run = Setting(False)

    def __init__(self) -> None:
        super().__init__()
        self.executor = ThreadExecutor(self)
        self._build_ui()
        self._set_status("Select a molecule file to import.")

    def _build_ui(self) -> None:
        self.mainArea.hide()
        root = self.controlArea
        root.setMinimumWidth(390)

        file_box = QGroupBox("Input file")
        file_layout = QVBoxLayout(file_box)
        self.path_edit = QLineEdit(self.file_path)
        self.path_edit.setPlaceholderText("Path to .csv, .tsv, .smi, .txt, .sdf, or .sd")
        self.path_edit.textChanged.connect(self._path_changed)
        file_layout.addWidget(self.path_edit)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        file_layout.addWidget(self.btn_browse)
        root.layout().addWidget(file_box)

        columns_box = QGroupBox("Column detection")
        form = QFormLayout(columns_box)
        self.smiles_edit = QLineEdit(self.smiles_column)
        self.smiles_edit.setPlaceholderText("Auto-detect, or type column name")
        self.name_edit = QLineEdit(self.name_column)
        self.name_edit.setPlaceholderText("Auto-detect, or type column name")
        self.delim_edit = QLineEdit(self.delimiter)
        self.delim_edit.setPlaceholderText("Auto-detect; use \\t for tab")
        for line in (self.smiles_edit, self.name_edit, self.delim_edit):
            line.textChanged.connect(self._settings_changed)
        form.addRow("SMILES column", self.smiles_edit)
        form.addRow("Name column", self.name_edit)
        form.addRow("Delimiter", self.delim_edit)
        root.layout().addWidget(columns_box)

        opts = QGroupBox("RDKit parsing")
        opts_layout = QVBoxLayout(opts)
        self.cb_sanitize = QCheckBox("Sanitize molecules")
        self.cb_sanitize.setChecked(bool(self.sanitize))
        self.cb_remove_hs = QCheckBox("Remove explicit hydrogens")
        self.cb_remove_hs.setChecked(bool(self.remove_hs))
        self.cb_flag_duplicates = QCheckBox("Detect duplicate structures by InChIKey")
        self.cb_flag_duplicates.setChecked(bool(self.flag_duplicates))
        self.cb_reject_duplicates = QCheckBox("Reject duplicate structures after first occurrence")
        self.cb_reject_duplicates.setChecked(bool(self.reject_duplicate_structures))
        self.cb_auto = QCheckBox("Auto-run")
        self.cb_auto.setChecked(bool(self.auto_run))
        for cb in (self.cb_sanitize, self.cb_remove_hs, self.cb_flag_duplicates, self.cb_reject_duplicates, self.cb_auto):
            cb.stateChanged.connect(self._settings_changed)
            opts_layout.addWidget(cb)
        root.layout().addWidget(opts)

        self.btn_load = QPushButton("Import molecules")
        self.btn_load.clicked.connect(self._run)
        root.layout().addWidget(self.btn_load)

        self.lbl = QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#475467;")
        root.layout().addWidget(self.lbl)

        roles_box = QGroupBox("Output column roles")
        roles_layout = QVBoxLayout(roles_box)
        self.roles_label = QLabel("Import a file to inspect attributes, class variables, and metas.")
        self.roles_label.setWordWrap(True)
        self.roles_label.setStyleSheet("color:#475467;")
        roles_layout.addWidget(self.roles_label)
        root.layout().addWidget(roles_box)

    def _path_changed(self, value: str) -> None:
        self.file_path = value
        if self.auto_run and value:
            self._run()

    def _settings_changed(self) -> None:
        self.smiles_column = self.smiles_edit.text().strip()
        self.name_column = self.name_edit.text().strip()
        self.delimiter = self.delim_edit.text()
        self.sanitize = bool(self.cb_sanitize.isChecked())
        self.remove_hs = bool(self.cb_remove_hs.isChecked())
        self.flag_duplicates = bool(self.cb_flag_duplicates.isChecked())
        self.reject_duplicate_structures = bool(self.cb_reject_duplicates.isChecked())
        self.auto_run = bool(self.cb_auto.isChecked())

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select molecule file",
            "",
            "Molecule files (*.csv *.tsv *.txt *.smi *.smiles *.sdf *.sd);;All files (*)",
        )
        if path:
            self.file_path = path
            self.path_edit.setText(path)

    def _set_status(self, text: str) -> None:
        self.lbl.setText(text)

    def _set_busy(self, busy: bool, text: str) -> None:
        self.btn_load.setEnabled(not busy)
        self.btn_browse.setEnabled(not busy)
        self._set_status(text)
        if busy:
            self.progressBarInit()
        else:
            self.progressBarFinished()

    def _config(self) -> MoleculeImportConfig:
        return MoleculeImportConfig(
            smiles_column=self.smiles_column or None,
            name_column=self.name_column or None,
            delimiter=self.delimiter or None,
            sanitize=bool(self.sanitize),
            remove_hs=bool(self.remove_hs),
            flag_duplicates=bool(self.flag_duplicates),
            reject_duplicate_structures=bool(self.reject_duplicate_structures),
            duplicate_key="inchikey",
        )

    def _run(self) -> None:
        self._settings_changed()
        path = (self.file_path or "").strip()
        if not path:
            self._set_status(format_no_input_status("molecule file"))
            self._send_empty()
            return
        if not os.path.exists(path):
            self._set_status(format_failed_status(f"File not found: {path}"))
            self._send_empty()
            return
        self._set_busy(True, "Importing molecules…")
        fut = self.executor.submit(self._run_background, path, self._config())
        fut.add_done_callback(self._on_done)

    def _run_background(self, path: str, cfg: MoleculeImportConfig) -> Tuple[Table, List[ChemMol], Table, List[ChemMol], Table, Table, Table, Table, Table, MoleculeImportResult]:
        result = import_molecule_file(path, cfg)
        data = chemmols_to_table(result.mols)
        accepted_mols = self._accepted_molecules(result)
        accepted_data = chemmols_to_table(accepted_mols)
        rejected = self._records_to_table(result.rejected_records)
        report = self._records_to_table(result.records)
        failed = self._records_to_table(result.failed_records)
        summary = self._summary_to_table(result)
        curation = curation_summary_to_table(summary_from_import(result.summary))
        return data, result.mols, accepted_data, accepted_mols, rejected, report, failed, summary, curation, result

    def _on_done(self, fut) -> None:
        try:
            payload = fut.result()
            methodinvoke(self, "_apply_outputs", (object,))(payload)
        except Exception as exc:
            methodinvoke(self, "_apply_error", (str,))(str(exc))

    @pyqtSlot(str)
    def _apply_error(self, msg: str) -> None:
        self._set_busy(False, format_failed_status(msg))
        self._send_empty()

    @pyqtSlot(object)
    def _apply_outputs(self, payload: object) -> None:
        data, mols, accepted_data, accepted_mols, rejected, report, failed, summary, curation, result = payload
        self._set_busy(False, format_done_status(
            f"valid={result.summary.valid_records}/{result.summary.total_records}",
            f"accepted={result.summary.accepted_records}",
            f"rejected={result.summary.rejected_records}",
            f"duplicates={result.summary.duplicate_records}",
            f"format={result.summary.source_format}",
            prefix="Import complete",
        ))
        self.roles_label.setText(format_domain_role_summary(data.domain))
        self.Outputs.data.send(data)
        self.Outputs.molecules.send(mols)
        self.Outputs.accepted_data.send(accepted_data)
        self.Outputs.accepted_molecules.send(accepted_mols)
        self.Outputs.rejected_records.send(rejected)
        self.Outputs.import_report.send(report)
        self.Outputs.failed_records.send(failed)
        self.Outputs.import_summary.send(summary)
        self.Outputs.curation_summary.send(curation)

    def _send_empty(self) -> None:
        self.Outputs.data.send(None)
        self.Outputs.molecules.send([])
        self.Outputs.accepted_data.send(None)
        self.Outputs.accepted_molecules.send([])
        self.Outputs.rejected_records.send(None)
        self.Outputs.import_report.send(None)
        self.Outputs.failed_records.send(None)
        self.Outputs.import_summary.send(None)
        self.Outputs.curation_summary.send(None)
        if hasattr(self, "roles_label"):
            self.roles_label.setText("Import a file to inspect attributes, class variables, and metas.")

    @staticmethod
    def _accepted_molecules(result: MoleculeImportResult) -> List[ChemMol]:
        accepted: List[ChemMol] = []
        valid_records = [r for r in result.records if r.ok]
        for cm, rec in zip(result.mols, valid_records):
            if rec.accepted:
                accepted.append(cm)
        return accepted

    @staticmethod
    def _records_to_table(records) -> Table:
        rows = import_records_as_dicts(records)
        numeric_cols = ["row_index", "ok", "accepted", "duplicate_count", "duplicate_group_index"]
        string_cols = [
            "source_format",
            "source_name",
            "name",
            "mol_id",
            "input_smiles",
            "canonical_smiles",
            "inchikey",
            "status",
            "error",
            "warnings",
            "duplicate_key",
            "rejection_reason",
            "qc_flags",
            "dropped_reason",
            "props_json",
        ]
        return report_rows_to_table(
            rows,
            numeric_columns=numeric_cols,
            meta_columns=string_cols,
            name="Import Report",
        )

    @staticmethod
    def _summary_to_table(result: MoleculeImportResult) -> Table:
        return summary_rows_to_table(
            import_summary_as_rows(result.summary),
            name="Import Summary",
        )
