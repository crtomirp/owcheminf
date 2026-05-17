from __future__ import annotations

from typing import List, Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.from_orange import TableMolConversionReport, table_to_chemmols_with_report
from chem_inf_widgets.chemcore.services.similarity_search_service import find_similarity_hits
from chem_inf_widgets.widgets.ui_helpers import (
    format_done_status,
    format_required_inputs_status,
    format_table_report,
    format_waiting_status,
    set_widget_warning,
)


def _find_smiles_vars(data: Table) -> List[StringVariable]:
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    preferred = [
        variable
        for variable in variables
        if isinstance(variable, StringVariable) and variable.name.strip().lower() in {"smiles", "canonical_smiles", "smile"}
    ]
    if preferred:
        return preferred + [variable for variable in variables if isinstance(variable, StringVariable) and variable not in preferred]
    return [variable for variable in variables if isinstance(variable, StringVariable)]


def _table_smiles(data: Table, var_name: str) -> List[str]:
    variable = next((var for var in _find_smiles_vars(data) if var.name == var_name), None)
    if variable is None:
        raise ValueError("No SMILES column selected.")
    return ["" if value is None else str(value).strip() for value in data.get_column(variable)]


class OWSimilaritySearch(OWWidget):
    name = "Similarity Search"
    description = "Return the nearest neighbors of each query compound in a reference library."
    icon = "icons/standardization_filtering/owsimilaritysearchwidget.svg"
    priority = 140

    class Inputs:
        query_data = Input("Query Data", Table)
        reference_data = Input("Reference Data", Table)

    class Outputs:
        neighbor_pairs = Output("Neighbor Pairs", Table)
        hit_compounds = Output("Hit Compounds", Table)

    query_smiles_var_name: str = Setting("")
    reference_smiles_var_name: str = Setting("")
    fp_type_idx: int = Setting(0)
    top_k: int = Setting(5)
    min_similarity: float = Setting(0.3)
    radius: int = Setting(2)
    n_bits: int = Setting(2048)
    auto_run: bool = Setting(True)

    _FP_TYPES = [("Morgan", "morgan"), ("RDKit", "rdkit"), ("MACCS", "maccs")]

    def __init__(self) -> None:
        super().__init__()
        self.query_data: Optional[Table] = None
        self.reference_data: Optional[Table] = None
        self._query_report: Optional[TableMolConversionReport] = None
        self._reference_report: Optional[TableMolConversionReport] = None
        self.mainArea.hide()

        root = QWidget(self.controlArea)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        self.controlArea.layout().addWidget(root)

        self.status_label = QLabel("Waiting for query and reference data…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignLeft)

        self.query_combo = QComboBox()
        self.query_combo.currentTextChanged.connect(self._on_query_smiles_changed)
        form.addRow("Query SMILES:", self.query_combo)

        self.reference_combo = QComboBox()
        self.reference_combo.currentTextChanged.connect(self._on_reference_smiles_changed)
        form.addRow("Reference SMILES:", self.reference_combo)

        self.fp_combo = QComboBox()
        self.fp_combo.addItems([label for label, _kind in self._FP_TYPES])
        self.fp_combo.setCurrentIndex(int(self.fp_type_idx))
        self.fp_combo.currentIndexChanged.connect(self._on_fp_type_changed)
        form.addRow("Fingerprint:", self.fp_combo)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 1000)
        self.top_k_spin.setValue(int(self.top_k))
        self.top_k_spin.valueChanged.connect(self._on_top_k_changed)
        form.addRow("Top-k:", self.top_k_spin)

        self.min_sim_spin = QDoubleSpinBox()
        self.min_sim_spin.setRange(0.0, 1.0)
        self.min_sim_spin.setSingleStep(0.05)
        self.min_sim_spin.setValue(float(self.min_similarity))
        self.min_sim_spin.valueChanged.connect(self._on_min_similarity_changed)
        form.addRow("Min similarity:", self.min_sim_spin)

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 6)
        self.radius_spin.setValue(int(self.radius))
        self.radius_spin.valueChanged.connect(self._on_radius_changed)
        form.addRow("Morgan radius:", self.radius_spin)

        self.nbits_spin = QSpinBox()
        self.nbits_spin.setRange(64, 8192)
        self.nbits_spin.setSingleStep(64)
        self.nbits_spin.setValue(int(self.n_bits))
        self.nbits_spin.valueChanged.connect(self._on_n_bits_changed)
        form.addRow("Fingerprint bits:", self.nbits_spin)
        layout.addWidget(form_widget)

        self.auto_run_check = QCheckBox("Auto-run")
        self.auto_run_check.setChecked(bool(self.auto_run))
        self.auto_run_check.toggled.connect(self._on_auto_run_toggled)
        layout.addWidget(self.auto_run_check)

        run_button = QPushButton("Run similarity search")
        run_button.clicked.connect(self.commit)
        layout.addWidget(run_button)
        layout.addStretch(1)

    @Inputs.query_data
    def set_query_data(self, data: Optional[Table]) -> None:
        self.query_data = data
        self._populate_combo(self.query_combo, data, "query_smiles_var_name")
        self._refresh_reports()
        self._update_status()
        self._maybe_autorun()

    @Inputs.reference_data
    def set_reference_data(self, data: Optional[Table]) -> None:
        self.reference_data = data
        self._populate_combo(self.reference_combo, data, "reference_smiles_var_name")
        self._refresh_reports()
        self._update_status()
        self._maybe_autorun()

    def _populate_combo(self, combo: QComboBox, data: Optional[Table], attr_name: str) -> None:
        combo.blockSignals(True)
        try:
            combo.clear()
            if data is None:
                return
            smiles_vars = _find_smiles_vars(data)
            combo.addItems([variable.name for variable in smiles_vars])
            if smiles_vars:
                names = [variable.name for variable in smiles_vars]
                current = getattr(self, attr_name)
                if current in names:
                    combo.setCurrentText(current)
                else:
                    setattr(self, attr_name, names[0])
                    combo.setCurrentText(names[0])
        finally:
            combo.blockSignals(False)

    def _on_query_smiles_changed(self, text: str) -> None:
        self.query_smiles_var_name = text
        self._refresh_reports()
        self._update_status()
        self._maybe_autorun()

    def _on_reference_smiles_changed(self, text: str) -> None:
        self.reference_smiles_var_name = text
        self._refresh_reports()
        self._update_status()
        self._maybe_autorun()

    def _on_fp_type_changed(self, index: int) -> None:
        self.fp_type_idx = int(index)
        self._maybe_autorun()

    def _on_top_k_changed(self, value: int) -> None:
        self.top_k = int(value)
        self._maybe_autorun()

    def _on_min_similarity_changed(self, value: float) -> None:
        self.min_similarity = float(value)
        self._maybe_autorun()

    def _on_radius_changed(self, value: int) -> None:
        self.radius = int(value)
        self._maybe_autorun()

    def _on_n_bits_changed(self, value: int) -> None:
        self.n_bits = int(value)
        self._maybe_autorun()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        self.auto_run = bool(checked)
        self._maybe_autorun()

    def _refresh_reports(self) -> None:
        self._query_report = None
        self._reference_report = None
        if self.query_data is not None and len(self.query_data) > 0 and self.query_smiles_var_name:
            try:
                _mols, self._query_report = table_to_chemmols_with_report(
                    self.query_data,
                    smiles_var=self.query_smiles_var_name,
                )
            except Exception:
                self._query_report = None
        if self.reference_data is not None and len(self.reference_data) > 0 and self.reference_smiles_var_name:
            try:
                _mols, self._reference_report = table_to_chemmols_with_report(
                    self.reference_data,
                    smiles_var=self.reference_smiles_var_name,
                )
            except Exception:
                self._reference_report = None

    def _update_status(self) -> None:
        if self.query_data is None or self.reference_data is None:
            self.status_label.setText(format_waiting_status("query and reference data"))
            return

        parts = []
        if self._query_report is not None:
            parts.append(format_table_report(self._query_report, prefix="Query"))
        else:
            parts.append(f"Query: rows={len(self.query_data)}")

        if self._reference_report is not None:
            parts.append(format_table_report(self._reference_report, prefix="Reference"))
        else:
            parts.append(f"Reference: rows={len(self.reference_data)}")

        self.status_label.setText(" | ".join(parts))

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and self.query_data is not None and self.reference_data is not None:
            self.commit()

    def commit(self) -> None:
        if self.query_data is None or self.reference_data is None:
            self.status_label.setText(format_required_inputs_status("Query data", "Reference data"))
            self.Outputs.neighbor_pairs.send(None)
            self.Outputs.hit_compounds.send(None)
            return

        query_smiles = _table_smiles(self.query_data, self.query_smiles_var_name)
        reference_smiles = _table_smiles(self.reference_data, self.reference_smiles_var_name)
        fp_type = self._FP_TYPES[self.fp_type_idx][1]
        hits = find_similarity_hits(
            query_smiles,
            reference_smiles,
            top_k=int(self.top_k),
            min_similarity=float(self.min_similarity),
            fp_type=fp_type,
            radius=int(self.radius),
            n_bits=int(self.n_bits),
        )
        self.Outputs.neighbor_pairs.send(self._pairs_table(hits))
        ref_indices = sorted({hit.reference_index for hit in hits})
        self.Outputs.hit_compounds.send(self.reference_data[ref_indices] if ref_indices else self.reference_data[:0])
        set_widget_warning(
            self,
            None if (self._query_report is None and self._reference_report is None) else "; ".join(
                part for part in [
                    f"Query invalid skipped: {self._query_report.n_invalid}" if self._query_report and self._query_report.n_invalid else "",
                    f"Reference invalid skipped: {self._reference_report.n_invalid}" if self._reference_report and self._reference_report.n_invalid else "",
                ] if part
            ),
        )
        self.status_label.setText(
            format_done_status(
                f"neighbor pairs={len(hits)}",
                f"unique hits={len(ref_indices)}",
            )
            + (
                f" | query valid={self._query_report.n_valid}, reference valid={self._reference_report.n_valid}"
                if self._query_report is not None and self._reference_report is not None
                else ""
            )
        )

    def _pairs_table(self, hits) -> Table:
        metas = [
            StringVariable("Query SMILES"),
            StringVariable("Reference SMILES"),
        ]
        attrs = [
            ContinuousVariable("Query Index"),
            ContinuousVariable("Reference Index"),
            ContinuousVariable("Similarity"),
        ]
        domain = Domain(attrs, metas=metas)
        X = np.array([[hit.query_index, hit.reference_index, hit.similarity] for hit in hits], dtype=float) if hits else np.zeros((0, 3))
        metas_arr = np.array([[hit.query_smiles, hit.reference_smiles] for hit in hits], dtype=object) if hits else np.zeros((0, 2), dtype=object)
        return Table.from_numpy(domain, X=X, metas=metas_arr)
