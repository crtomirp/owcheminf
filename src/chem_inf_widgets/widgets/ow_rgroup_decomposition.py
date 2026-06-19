from __future__ import annotations

from typing import List, Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from Orange.data import Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.rgroup_decomposition_service import decompose_rgroups


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


class OWRGroupDecomposition(OWWidget):
    name = "R-Group Decomposition"
    description = "Break a compound series into a shared core and R-group substituents."
    icon = "icons/analysis/owrgroupdecompositionwidget.svg"
    priority = 141

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        rgroup_table = Output("R-Group Table", Table)
        matched_data = Output("Matched Data", Table)
        unmatched_data = Output("Unmatched Data", Table)

    smiles_var_name: str = Setting("")
    core_text: str = Setting("")
    auto_run: bool = Setting(True)

    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[Table] = None
        self.mainArea.hide()

        root = QWidget(self.controlArea)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        self.controlArea.layout().addWidget(root)

        self.status_label = QLabel("Waiting for input…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignLeft)

        self.smiles_combo = QComboBox()
        self.smiles_combo.currentTextChanged.connect(self._on_smiles_changed)
        form.addRow("SMILES column:", self.smiles_combo)

        self.core_edit = QLineEdit()
        self.core_edit.setText(self.core_text)
        self.core_edit.textChanged.connect(self._on_core_text_changed)
        self.core_edit.editingFinished.connect(self._on_core_edit_finished)
        form.addRow("Core SMARTS/SMILES:", self.core_edit)
        layout.addWidget(form_widget)

        self.auto_run_check = QCheckBox("Auto-run")
        self.auto_run_check.setChecked(bool(self.auto_run))
        self.auto_run_check.toggled.connect(self._on_auto_run_toggled)
        layout.addWidget(self.auto_run_check)

        run_button = QPushButton("Run decomposition")
        run_button.clicked.connect(self.commit)
        layout.addWidget(run_button)
        layout.addStretch(1)

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self.data = data
        self._populate_smiles_combo()
        self.status_label.setText("Input loaded." if data is not None else "Waiting for input…")
        self._maybe_autorun()

    def _populate_smiles_combo(self) -> None:
        self.smiles_combo.blockSignals(True)
        try:
            self.smiles_combo.clear()
            if self.data is None:
                return
            smiles_vars = _find_smiles_vars(self.data)
            self.smiles_combo.addItems([variable.name for variable in smiles_vars])
            if smiles_vars:
                names = [variable.name for variable in smiles_vars]
                if self.smiles_var_name in names:
                    self.smiles_combo.setCurrentText(self.smiles_var_name)
                else:
                    self.smiles_var_name = names[0]
                    self.smiles_combo.setCurrentText(names[0])
        finally:
            self.smiles_combo.blockSignals(False)

    def _on_smiles_changed(self, text: str) -> None:
        self.smiles_var_name = text
        self._maybe_autorun()

    def _on_core_text_changed(self, text: str) -> None:
        self.core_text = text

    def _on_core_edit_finished(self) -> None:
        self._maybe_autorun()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        self.auto_run = bool(checked)
        self._maybe_autorun()

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and self.data is not None and len(self.data) > 0:
            self.commit()

    def commit(self) -> None:
        if self.data is None or len(self.data) == 0:
            self.Outputs.rgroup_table.send(None)
            self.Outputs.matched_data.send(None)
            self.Outputs.unmatched_data.send(None)
            self.status_label.setText("No input data.")
            return

        try:
            result = decompose_rgroups(_table_smiles(self.data, self.smiles_var_name), core_smarts=self.core_text or None)
        except Exception as exc:
            self.status_label.setText(f"Failed: {exc}")
            self.Outputs.rgroup_table.send(None)
            self.Outputs.matched_data.send(None)
            self.Outputs.unmatched_data.send(None)
            return

        self.Outputs.rgroup_table.send(self._result_table(result))
        self.Outputs.matched_data.send(self.data[result.matched_indices] if result.matched_indices else self.data[:0])
        self.Outputs.unmatched_data.send(self.data[result.unmatched_indices] if result.unmatched_indices else self.data[:0])
        self.status_label.setText(
            f"Core: {result.core} | matched={len(result.matched_indices)}, unmatched={len(result.unmatched_indices)}"
        )

    def _result_table(self, result) -> Table:
        metas = [StringVariable("Core")] + [StringVariable(label) for label in result.group_labels]
        domain = Domain([], metas=metas)
        rows = []
        for row in result.rows:
            rows.append([row.core] + [row.groups.get(label, "") for label in result.group_labels])
        metas_arr = np.array(rows, dtype=object) if rows else np.zeros((0, len(metas)), dtype=object)
        return Table.from_numpy(domain, X=np.zeros((len(rows), 0), dtype=float), metas=metas_arr)
