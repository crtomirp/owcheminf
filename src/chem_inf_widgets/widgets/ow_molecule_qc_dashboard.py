from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from AnyQt.QtCore import pyqtSlot
from AnyQt.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QPushButton, QSpinBox, QVBoxLayout
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.from_orange import table_to_chemmols_with_report
from chem_inf_widgets.chemcore.molecule_contract import DROPPED_REASON, QC_FLAGS, ROW_ID, SOURCE_ROW_INDEX, TRANSFORM_LOG
from chem_inf_widgets.chemcore.services.curation_summary import curation_summary_to_table, summary_from_qc
from chem_inf_widgets.chemcore.services.molecule_qc_service import (
    MoleculeQCConfig,
    MoleculeQCResult,
    annotate_chemmols_with_qc,
    qc_partition_indices,
    qc_records_as_dicts,
    qc_summary_as_rows,
    run_molecule_qc,
)
from chem_inf_widgets.chemcore.services.report_table_utils import report_rows_to_table, summary_rows_to_table
from chem_inf_widgets.widgets.ui_helpers import format_done_status, format_failed_status, format_no_input_status, format_table_report, set_widget_warning


class OWMoleculeQCDashboard(OWWidget):
    name = "Molecule QC Dashboard"
    description = "Detect invalid structures, salts/fragments, duplicates, charges, metals, stereochemistry gaps, and other molecule quality issues."
    icon = "icons/standardization_filtering/owmolocwidget.png"
    priority = 105

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        modeling_data = Output("Modeling Data", Table)
        annotated_data = Output("QC Annotated Data", Table)
        clean_data = Output("Clean Data", Table)
        problem_data = Output("Problem Data", Table)
        rejected_data = Output("Rejected Data", Table)
        qc_report = Output("QC Report", Table)
        qc_summary = Output("QC Summary", Table)
        curation_summary = Output("Curation Summary", Table)
        annotated_molecules = Output("QC Annotated Molecules", list, auto_summary=False)
        clean_molecules = Output("Clean Molecules", list, auto_summary=False)
        problem_molecules = Output("Problem Molecules", list, auto_summary=False)
        rejected_molecules = Output("Rejected Molecules", list, auto_summary=False)

    duplicate_key = Setting("canonical_smiles")
    max_mw = Setting(900.0)
    min_heavy_atoms = Setting(3)
    max_heavy_atoms = Setting(90)
    max_fragments = Setting(1)
    flag_metals = Setting(True)
    flag_isotopes = Setting(True)
    flag_radicals = Setting(True)
    flag_formal_charge = Setting(True)
    flag_missing_chiral_stereo = Setting(True)
    flag_missing_double_bond_stereo = Setting(True)
    auto_run = Setting(False)

    def __init__(self) -> None:
        super().__init__()
        self.executor = ThreadExecutor(self)
        self._in_table: Optional[Table] = None
        self._in_molecules: List[ChemMol] = []
        self._table_report = None
        self._build_ui()
        self._update_status("Ready")

    def _build_ui(self) -> None:
        self.mainArea.hide()
        root = self.controlArea
        root.setMinimumWidth(360)

        input_box = QGroupBox("Input and execution")
        input_layout = QVBoxLayout(input_box)
        self.cb_auto = QCheckBox("Auto-run")
        self.cb_auto.setChecked(bool(self.auto_run))
        self.cb_auto.stateChanged.connect(self._on_settings_changed)
        input_layout.addWidget(self.cb_auto)
        root.layout().addWidget(input_box)

        thresholds = QGroupBox("Thresholds")
        form = QFormLayout(thresholds)
        self.spin_min_ha = QSpinBox()
        self.spin_min_ha.setRange(0, 500)
        self.spin_min_ha.setValue(int(self.min_heavy_atoms))
        self.spin_max_ha = QSpinBox()
        self.spin_max_ha.setRange(1, 1000)
        self.spin_max_ha.setValue(int(self.max_heavy_atoms))
        self.spin_max_frags = QSpinBox()
        self.spin_max_frags.setRange(1, 20)
        self.spin_max_frags.setValue(int(self.max_fragments))
        self.spin_max_mw = QDoubleSpinBox()
        self.spin_max_mw.setRange(0.0, 10000.0)
        self.spin_max_mw.setDecimals(1)
        self.spin_max_mw.setSingleStep(50.0)
        self.spin_max_mw.setValue(float(self.max_mw))
        for w in (self.spin_min_ha, self.spin_max_ha, self.spin_max_frags, self.spin_max_mw):
            w.valueChanged.connect(self._on_settings_changed)
        form.addRow("Min heavy atoms", self.spin_min_ha)
        form.addRow("Max heavy atoms", self.spin_max_ha)
        form.addRow("Max fragments", self.spin_max_frags)
        form.addRow("Max MW", self.spin_max_mw)
        root.layout().addWidget(thresholds)

        dup_box = QGroupBox("Duplicate detection")
        dup_layout = QFormLayout(dup_box)
        self.combo_dup = QComboBox()
        self.combo_dup.addItems(["canonical_smiles", "inchikey"])
        self.combo_dup.setCurrentText(str(self.duplicate_key))
        self.combo_dup.currentTextChanged.connect(self._on_settings_changed)
        dup_layout.addRow("Duplicate key", self.combo_dup)
        root.layout().addWidget(dup_box)

        flags = QGroupBox("Issue flags")
        flags_layout = QVBoxLayout(flags)
        self.cb_metals = QCheckBox("Flag metals/metalloids")
        self.cb_isotopes = QCheckBox("Flag isotopes")
        self.cb_radicals = QCheckBox("Flag radicals")
        self.cb_charge = QCheckBox("Flag non-zero net formal charge")
        self.cb_chiral = QCheckBox("Flag unassigned chiral centers")
        self.cb_db = QCheckBox("Flag unassigned double-bond stereo")
        checkboxes = [
            (self.cb_metals, self.flag_metals),
            (self.cb_isotopes, self.flag_isotopes),
            (self.cb_radicals, self.flag_radicals),
            (self.cb_charge, self.flag_formal_charge),
            (self.cb_chiral, self.flag_missing_chiral_stereo),
            (self.cb_db, self.flag_missing_double_bond_stereo),
        ]
        for cb, value in checkboxes:
            cb.setChecked(bool(value))
            cb.stateChanged.connect(self._on_settings_changed)
            flags_layout.addWidget(cb)
        root.layout().addWidget(flags)

        self.btn_run = QPushButton("Run QC")
        self.btn_run.clicked.connect(self._on_run)
        root.layout().addWidget(self.btn_run)

        self.lbl = QLabel("Ready")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#475467;")
        root.layout().addWidget(self.lbl)

    def _update_status(self, text: str) -> None:
        self.lbl.setText(text)

    def _set_busy(self, busy: bool, text: str) -> None:
        self.btn_run.setEnabled(not busy)
        self._update_status(text)
        if busy:
            self.progressBarInit()
        else:
            self.progressBarFinished()

    def _on_settings_changed(self) -> None:
        self.auto_run = bool(self.cb_auto.isChecked())
        self.min_heavy_atoms = int(self.spin_min_ha.value())
        self.max_heavy_atoms = int(self.spin_max_ha.value())
        self.max_fragments = int(self.spin_max_frags.value())
        self.max_mw = float(self.spin_max_mw.value())
        self.duplicate_key = str(self.combo_dup.currentText())
        self.flag_metals = bool(self.cb_metals.isChecked())
        self.flag_isotopes = bool(self.cb_isotopes.isChecked())
        self.flag_radicals = bool(self.cb_radicals.isChecked())
        self.flag_formal_charge = bool(self.cb_charge.isChecked())
        self.flag_missing_chiral_stereo = bool(self.cb_chiral.isChecked())
        self.flag_missing_double_bond_stereo = bool(self.cb_db.isChecked())

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._in_table = data
        self._table_report = None
        if data is not None and len(data) > 0:
            try:
                _mols, self._table_report = table_to_chemmols_with_report(data)
            except Exception as exc:
                self._table_report = None
                set_widget_warning(self, f"Could not pre-parse input table: {exc}")
        self._update_status(self._input_summary())
        if self.auto_run:
            self._on_run()

    @Inputs.molecules
    def set_molecules(self, mols: Optional[list]) -> None:
        self._in_molecules = [m for m in (mols or []) if isinstance(m, ChemMol)]
        self._update_status(self._input_summary())
        if self.auto_run:
            self._on_run()

    def _input_summary(self) -> str:
        n_tab = 0 if self._in_table is None else len(self._in_table)
        n_mol = len(self._in_molecules)
        if self._table_report is not None:
            return format_table_report(self._table_report, prefix="Input") + f", Molecules={n_mol}"
        return f"Input: Table rows={n_tab}, Molecules={n_mol}"

    def _make_config(self) -> MoleculeQCConfig:
        return MoleculeQCConfig(
            duplicate_key=str(self.duplicate_key),
            max_mw=float(self.max_mw),
            min_heavy_atoms=int(self.min_heavy_atoms),
            max_heavy_atoms=int(self.max_heavy_atoms),
            max_fragments=int(self.max_fragments),
            flag_metals=bool(self.flag_metals),
            flag_isotopes=bool(self.flag_isotopes),
            flag_radicals=bool(self.flag_radicals),
            flag_formal_charge=bool(self.flag_formal_charge),
            flag_missing_chiral_stereo=bool(self.flag_missing_chiral_stereo),
            flag_missing_double_bond_stereo=bool(self.flag_missing_double_bond_stereo),
        )

    def _on_run(self) -> None:
        if (self._in_table is None or len(self._in_table) == 0) and not self._in_molecules:
            self._update_status(format_no_input_status())
            self._send_empty()
            return
        self._set_busy(True, "Running molecule QC…")
        fut = self.executor.submit(self._run_background, self._in_table, self._in_molecules, self._make_config())
        fut.add_done_callback(self._on_done)

    def _run_background(self, data: Optional[Table], mols: Sequence[ChemMol], cfg: MoleculeQCConfig) -> Tuple[Optional[Table], Optional[Table], Optional[Table], Optional[Table], Optional[Table], Table, Table, Table, List[ChemMol], List[ChemMol], List[ChemMol], List[ChemMol], MoleculeQCResult]:
        items: List[Any]
        source_mols: List[ChemMol] = []
        skipped_source_rows: List[int] = []
        conversion_errors: List[str] = []
        if mols:
            items = list(mols)
            source_mols = list(mols)
        elif data is not None and len(data) > 0:
            source_mols, conversion_report = table_to_chemmols_with_report(data)
            skipped_source_rows = [max(0, int(i) - 1) for i in conversion_report.skipped_rows]
            conversion_errors = list(conversion_report.errors or [])
            items = list(source_mols)
        else:
            items = []

        result = run_molecule_qc(items, cfg)
        partitions = qc_partition_indices(result)
        annotated_mols = annotate_chemmols_with_qc(source_mols, result.records)
        report_table = self._records_to_table(result.records)
        summary_table = self._summary_to_table(result)
        curation_table = curation_summary_to_table(summary_from_qc(result.summary))

        clean_mols = [annotated_mols[i] for i in partitions["clean"] if i < len(annotated_mols)]
        problem_mols = [annotated_mols[i] for i in partitions["problem"] if i < len(annotated_mols)]
        rejected_mols = [annotated_mols[i] for i in partitions["rejected"] if i < len(annotated_mols)]

        modeling_data = None
        annotated_data = None
        clean_data = None
        problem_data = None
        rejected_data = None
        if data is not None and not mols:
            annotated_data = self._annotated_table(data, result.records, annotated_mols, skipped_source_rows, conversion_errors)
            row_indices = self._source_row_indices(annotated_mols)
            clean_rows = [row_indices[i] for i in partitions["clean"] if i < len(row_indices)]
            problem_rows = [row_indices[i] for i in partitions["problem"] if i < len(row_indices)]
            rejected_rows = [row_indices[i] for i in partitions["rejected"] if i < len(row_indices)] + list(skipped_source_rows)
            clean_data = self._subset_table(annotated_data, clean_rows)
            problem_data = self._subset_table(annotated_data, problem_rows)
            rejected_data = self._subset_table(annotated_data, sorted(set(rejected_rows)))
            modeling_data = self._modeling_table(data, result.records, source_mols, partitions["clean"])
        return modeling_data, annotated_data, clean_data, problem_data, rejected_data, report_table, summary_table, curation_table, annotated_mols, clean_mols, problem_mols, rejected_mols, result

    def _on_done(self, fut) -> None:
        try:
            payload = fut.result()
            methodinvoke(self, "_apply_outputs", (object,))(payload)
        except Exception as e:
            methodinvoke(self, "_apply_error", (str,))(str(e))

    @pyqtSlot(str)
    def _apply_error(self, msg: str) -> None:
        self._set_busy(False, format_failed_status(msg))
        self._send_empty()

    @pyqtSlot(object)
    def _apply_outputs(self, payload: object) -> None:
        modeling_data, annotated_data, clean_data, problem_data, rejected_data, report_table, summary_table, curation_table, annotated_mols, clean_mols, problem_mols, rejected_mols, result = payload
        self._set_busy(
            False,
            format_done_status(
                f"total={result.summary.total}",
                f"clean={result.summary.clean}",
                f"problem={len(problem_mols)}",
                f"rejected={len(rejected_mols)}",
                f"invalid={result.summary.invalid}",
                f"duplicate groups={result.summary.duplicate_groups}",
                prefix="QC complete",
            ),
        )
        self.Outputs.modeling_data.send(modeling_data)
        self.Outputs.annotated_data.send(annotated_data)
        self.Outputs.clean_data.send(clean_data)
        self.Outputs.problem_data.send(problem_data)
        self.Outputs.rejected_data.send(rejected_data)
        self.Outputs.qc_report.send(report_table)
        self.Outputs.qc_summary.send(summary_table)
        self.Outputs.curation_summary.send(curation_table)
        self.Outputs.annotated_molecules.send(annotated_mols)
        self.Outputs.clean_molecules.send(clean_mols)
        self.Outputs.problem_molecules.send(problem_mols)
        self.Outputs.rejected_molecules.send(rejected_mols)

    def _send_empty(self) -> None:
        self.Outputs.modeling_data.send(None)
        self.Outputs.annotated_data.send(None)
        self.Outputs.clean_data.send(None)
        self.Outputs.problem_data.send(None)
        self.Outputs.rejected_data.send(None)
        self.Outputs.qc_report.send(None)
        self.Outputs.qc_summary.send(None)
        self.Outputs.curation_summary.send(None)
        self.Outputs.annotated_molecules.send([])
        self.Outputs.clean_molecules.send([])
        self.Outputs.problem_molecules.send([])
        self.Outputs.rejected_molecules.send([])


    @staticmethod
    def _source_row_indices(mols: Sequence[ChemMol]) -> List[int]:
        indices: List[int] = []
        for fallback, cm in enumerate(mols):
            value = None
            try:
                value = (cm.props or {}).get(SOURCE_ROW_INDEX)
            except Exception:
                value = None
            try:
                idx = int(value) - 1 if value not in (None, "") else fallback
            except Exception:
                idx = fallback
            indices.append(max(0, idx))
        return indices

    @staticmethod
    def _conversion_error_by_row(errors: Sequence[str]) -> Dict[int, str]:
        out: Dict[int, str] = {}
        for text in errors:
            msg = str(text or "")
            # Current converter messages start with "Row N:". Keep this parser
            # deliberately defensive so older reports still work.
            row_idx = None
            if msg.lower().startswith("row "):
                head = msg.split(":", 1)[0]
                try:
                    row_idx = int(head.split()[1]) - 1
                except Exception:
                    row_idx = None
            if row_idx is not None:
                out[row_idx] = msg
        return out

    @staticmethod
    def _annotated_table(
        data: Table,
        records,
        source_mols: Sequence[ChemMol],
        skipped_source_rows: Sequence[int],
        conversion_errors: Sequence[str],
    ) -> Table:
        n_rows = len(data)
        status = [""] * n_rows
        severity = [""] * n_rows
        issue_codes = [""] * n_rows
        issues = [""] * n_rows
        n_issues = [0.0] * n_rows
        duplicate_key = [""] * n_rows
        duplicate_count = [0.0] * n_rows
        qc_valid_structure = [0.0] * n_rows
        row_id = [""] * n_rows
        transform_log = [""] * n_rows
        qc_flags = [""] * n_rows
        dropped_reason = [""] * n_rows

        row_indices = OWMoleculeQCDashboard._source_row_indices(source_mols)
        for rec_idx, rec in enumerate(records):
            if rec_idx >= len(row_indices):
                continue
            row = row_indices[rec_idx]
            if row < 0 or row >= n_rows:
                continue
            status[row] = str(rec.status)
            severity[row] = str(rec.severity)
            issue_codes[row] = ";".join(rec.issue_codes)
            issues[row] = " | ".join(rec.issues)
            n_issues[row] = float(rec.n_issues)
            duplicate_key[row] = str(rec.duplicate_key or "")
            duplicate_count[row] = float(rec.duplicate_count or 0)
            qc_valid_structure[row] = 1.0 if rec.ok_parse else 0.0
            if rec_idx < len(source_mols):
                props = source_mols[rec_idx].props if isinstance(source_mols[rec_idx].props, dict) else {}
                row_id[row] = str(props.get(ROW_ID, "") or "")
                transform_log[row] = str(props.get(TRANSFORM_LOG, "") or "")
                qc_flags[row] = str(props.get(QC_FLAGS, "") or "")
                dropped_reason[row] = str(props.get(DROPPED_REASON, "") or "")

        errors_by_row = OWMoleculeQCDashboard._conversion_error_by_row(conversion_errors)
        for row in skipped_source_rows:
            if row < 0 or row >= n_rows:
                continue
            status[row] = "Invalid"
            severity[row] = "ERROR"
            issue_codes[row] = "INVALID_STRUCTURE"
            issues[row] = errors_by_row.get(row, "Could not parse molecule from the input table.")
            n_issues[row] = 1.0
            qc_valid_structure[row] = 0.0
            qc_flags[row] = "invalid_structure"
            dropped_reason[row] = "invalid_structure"

        new_attr_names = {"qc_n_issues", "qc_duplicate_count", "qc_valid_structure"}
        existing_attr_names = [v.name for v in data.domain.attributes]
        existing_meta_names = [v.name for v in data.domain.metas]
        existing_names = {v.name for v in list(data.domain.attributes) + list(data.domain.class_vars) + list(data.domain.metas)}
        attrs = list(data.domain.attributes)
        metas = list(data.domain.metas)

        # Keep numeric QC fields as attributes unless a name collision exists.
        numeric_specs = [
            ("qc_n_issues", n_issues),
            ("qc_duplicate_count", duplicate_count),
            ("qc_valid_structure", qc_valid_structure),
        ]
        numeric_attr_specs = [(name, values) for name, values in numeric_specs if name not in existing_names]
        string_specs = [
            ("qc_status", status),
            ("qc_severity", severity),
            ("qc_issue_codes", issue_codes),
            ("qc_issues", issues),
            ("qc_duplicate_key", duplicate_key),
            ("row_id", row_id),
            ("transform_log", transform_log),
            ("qc_flags", qc_flags),
            ("dropped_reason", dropped_reason),
        ]
        string_meta_specs = [(name, values) for name, values in string_specs if name not in existing_names]

        attrs = attrs + [ContinuousVariable(name) for name, _ in numeric_attr_specs]
        metas = metas + [StringVariable(name) for name, _ in string_meta_specs]
        domain = Domain(attrs, data.domain.class_vars, metas)

        # Orange can hand back read-only array views here depending on the
        # upstream slicing history, so QC annotations always write into an
        # owned copy.
        base_x = np.array(data.X, dtype=float, copy=True) if len(data.domain.attributes) else np.empty((n_rows, 0), dtype=float)
        attr_index = {name: idx for idx, name in enumerate(existing_attr_names)}
        for name, values in numeric_specs:
            idx = attr_index.get(name)
            if idx is None or idx >= base_x.shape[1]:
                continue
            for row_i in range(n_rows):
                base_x[row_i, idx] = float(values[row_i])
        extra_x = np.asarray([[float(values[i]) for _, values in numeric_attr_specs] for i in range(n_rows)], dtype=float) if numeric_attr_specs else np.empty((n_rows, 0), dtype=float)
        X = np.hstack([base_x, extra_x]) if extra_x.size else base_x

        base_m = np.array(data.metas, dtype=object, copy=True) if len(data.domain.metas) else np.empty((n_rows, 0), dtype=object)
        meta_index = {name: idx for idx, name in enumerate(existing_meta_names)}
        for name, values in string_specs:
            idx = meta_index.get(name)
            if idx is None or idx >= base_m.shape[1]:
                continue
            for row_i in range(n_rows):
                base_m[row_i, idx] = str(values[row_i])
        extra_m = np.asarray([[str(values[i]) for _, values in string_meta_specs] for i in range(n_rows)], dtype=object) if string_meta_specs else np.empty((n_rows, 0), dtype=object)
        M = np.hstack([base_m, extra_m]) if extra_m.size else base_m

        Y = np.array(data.Y, dtype=float, copy=True) if len(data.domain.class_vars) else None
        return Table.from_numpy(domain, X=X, Y=Y, metas=M)

    @staticmethod
    def _safe_subset_indices(data: Table, indices: Sequence[int]) -> List[int]:
        """Return stable, unique, in-bounds row indices for Orange table slicing.

        Some upstream widgets preserve a ``source_row_index`` provenance column.
        Depending on the producer this value can be one-based, zero-based, or
        copied from a previous table.  QC partitioning must never crash because
        of one stale/off-by-one provenance value; invalid indices are ignored
        and duplicates are collapsed while preserving order.
        """
        n_rows = len(data)
        out: List[int] = []
        seen = set()
        for value in indices:
            try:
                idx = int(value)
            except Exception:
                continue
            if idx < 0 or idx >= n_rows or idx in seen:
                continue
            out.append(idx)
            seen.add(idx)
        return out

    @staticmethod
    def _empty_like_table(data: Table) -> Table:
        return Table.from_numpy(
            data.domain,
            X=np.empty((0, len(data.domain.attributes))),
            Y=np.empty((0, len(data.domain.class_vars))) if len(data.domain.class_vars) else None,
            metas=np.empty((0, len(data.domain.metas)), dtype=object),
        )

    @staticmethod
    def _subset_table(data: Optional[Table], indices: Sequence[int]) -> Optional[Table]:
        if data is None:
            return None
        safe_indices = OWMoleculeQCDashboard._safe_subset_indices(data, indices)
        if not safe_indices:
            return OWMoleculeQCDashboard._empty_like_table(data)
        return data[safe_indices]


    @staticmethod
    def _is_structure_column(name: str) -> bool:
        key = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
        return key in {
            "smiles",
            "smile",
            "canonical_smiles",
            "can_smiles",
            "input_smiles",
            "original_smiles",
            "standardized_smiles",
            "smiles_orig",
            "smiles_std",
            "mol_smiles",
            "inchi",
            "inchikey",
            "inchi_key",
            "rdkit_mol",
            "mol",
        }

    @staticmethod
    def _unique_variable_name(base: str, existing: set[str]) -> str:
        name = base
        counter = 2
        while name in existing:
            name = f"{base}_{counter}"
            counter += 1
        existing.add(name)
        return name

    @staticmethod
    def _is_modeling_excluded_column(name: str) -> bool:
        key = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
        if OWMoleculeQCDashboard._is_structure_column(key):
            return True
        return key.startswith(("qc_", "standardization_", "curation_", "std_", "import_"))

    @staticmethod
    def _modeling_table(
        data: Table,
        records,
        source_mols: Sequence[ChemMol],
        clean_partition: Sequence[int],
    ) -> Table:
        """Return a slim clean-data table intended for downstream modelling.

        The regular QC outputs intentionally carry audit/provenance columns.  This
        output is deliberately minimal: clean/accepted input rows only, original
        non-structure data columns preserved, and exactly two fresh structure
        identifiers added as metas: ``SMILES`` and ``inchikey``.
        """
        row_indices = OWMoleculeQCDashboard._source_row_indices(source_mols)
        selected_pairs: List[Tuple[int, Any]] = []
        seen_rows = set()
        n_rows = len(data)
        for rec_idx in clean_partition:
            if rec_idx >= len(row_indices) or rec_idx >= len(records):
                continue
            row = row_indices[rec_idx]
            if row < 0 or row >= n_rows or row in seen_rows:
                continue
            selected_pairs.append((row, records[rec_idx]))
            seen_rows.add(row)

        keep_attrs_idx = [
            i for i, var in enumerate(data.domain.attributes)
            if not OWMoleculeQCDashboard._is_modeling_excluded_column(var.name)
        ]
        keep_class_idx = [
            i for i, var in enumerate(data.domain.class_vars)
            if not OWMoleculeQCDashboard._is_modeling_excluded_column(var.name)
        ]
        keep_meta_idx = [
            i for i, var in enumerate(data.domain.metas)
            if not OWMoleculeQCDashboard._is_modeling_excluded_column(var.name)
        ]

        attrs = [data.domain.attributes[i] for i in keep_attrs_idx]
        class_vars = [data.domain.class_vars[i] for i in keep_class_idx]
        metas = [data.domain.metas[i] for i in keep_meta_idx]
        existing_names = {var.name for var in attrs + class_vars + metas}
        smiles_name = OWMoleculeQCDashboard._unique_variable_name("SMILES", existing_names)
        inchikey_name = OWMoleculeQCDashboard._unique_variable_name("inchikey", existing_names)
        metas = metas + [StringVariable(smiles_name), StringVariable(inchikey_name)]
        domain = Domain(attrs, class_vars, metas)

        if not selected_pairs:
            return Table.from_numpy(
                domain,
                X=np.empty((0, len(attrs)), dtype=float),
                Y=np.empty((0, len(class_vars)), dtype=float) if class_vars else None,
                metas=np.empty((0, len(metas)), dtype=object),
            )

        rows = [row for row, _ in selected_pairs]
        base_x = np.asarray(data.X, dtype=float)[:, keep_attrs_idx] if keep_attrs_idx else np.empty((len(data), 0), dtype=float)
        X = base_x[rows, :] if keep_attrs_idx else np.empty((len(rows), 0), dtype=float)

        if keep_class_idx:
            base_y = np.asarray(data.Y, dtype=float)
            if base_y.ndim == 1:
                base_y = base_y.reshape((-1, 1))
            Y = base_y[:, keep_class_idx][rows, :]
        else:
            Y = None

        base_metas = np.asarray(data.metas, dtype=object)[:, keep_meta_idx] if keep_meta_idx else np.empty((len(data), 0), dtype=object)
        extra_metas = np.asarray(
            [[str(getattr(rec, "canonical_smiles", "") or ""), str(getattr(rec, "inchikey", "") or "")] for _, rec in selected_pairs],
            dtype=object,
        )
        M = np.hstack([base_metas[rows, :], extra_metas]) if keep_meta_idx else extra_metas
        return Table.from_numpy(domain, X=X, Y=Y, metas=M)

    @staticmethod
    def _records_to_table(records) -> Table:
        rows = qc_records_as_dicts(records)
        string_cols = [
            "name",
            "input_smiles",
            "canonical_smiles",
            "inchikey",
            "status",
            "severity",
            "issue_codes",
            "issues",
            "duplicate_key",
            "parse_error",
            "parse_warnings",
            "qc_flags",
            "dropped_reason",
        ]
        numeric_cols = ["row_index", "n_issues", "n_fragments", "largest_fragment_atoms", "heavy_atoms", "molecular_weight", "formal_charge", "n_rings", "n_hetero_atoms", "has_metal", "has_isotope", "has_radical", "possible_chiral_centers", "unassigned_chiral_centers", "possible_double_bond_stereo", "unassigned_double_bond_stereo", "duplicate_count"]
        return report_rows_to_table(
            rows,
            numeric_columns=numeric_cols,
            meta_columns=string_cols,
            name="QC Report",
        )

    @staticmethod
    def _summary_to_table(result: MoleculeQCResult) -> Table:
        return summary_rows_to_table(
            qc_summary_as_rows(result.summary),
            name="QC Summary",
        )
