from __future__ import annotations

from typing import List, Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.matched_pair_service import find_matched_pairs
from chem_inf_widgets.widgets.ui_helpers import (
    format_loaded_status,
    format_no_input_status,
    format_result_count_status,
    format_waiting_status,
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


def _find_numeric_candidates(data: Table) -> List[ContinuousVariable]:
    variables = list(data.domain.attributes) + list(data.domain.class_vars)
    return [variable for variable in variables if getattr(variable, "is_continuous", False)]


class OWMatchedMolecularPairs(OWWidget):
    name = "Matched Molecular Pairs"
    description = "Find molecule pairs with a shared core and a local transformation."
    icon = "icons/analysis/owmatchedmolecularpairswidget.svg"
    priority = 142

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        pair_table = Output("Pair Table", Table)
        pair_compounds = Output("Pair Compounds", Table)

    smiles_var_name: str = Setting("")
    property_var_name: str = Setting("")
    min_shared_atoms: int = Setting(4)
    max_pairs: int = Setting(250)
    auto_run: bool = Setting(False)

    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[Table] = None
        self.mainArea.hide()

        root = QWidget(self.controlArea)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        self.controlArea.layout().addWidget(root)

        self.status_label = QLabel(format_waiting_status())
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignLeft)

        self.smiles_combo = QComboBox()
        self.smiles_combo.currentTextChanged.connect(self._on_smiles_changed)
        form.addRow("SMILES column:", self.smiles_combo)

        self.property_combo = QComboBox()
        self.property_combo.currentTextChanged.connect(self._on_property_changed)
        form.addRow("Property column:", self.property_combo)

        self.shared_spin = QSpinBox()
        self.shared_spin.setRange(1, 50)
        self.shared_spin.setValue(int(self.min_shared_atoms))
        self.shared_spin.valueChanged.connect(self._on_shared_changed)
        form.addRow("Min shared atoms:", self.shared_spin)

        self.max_pairs_spin = QSpinBox()
        self.max_pairs_spin.setRange(1, 5000)
        self.max_pairs_spin.setValue(int(self.max_pairs))
        self.max_pairs_spin.valueChanged.connect(self._on_max_pairs_changed)
        form.addRow("Max pairs:", self.max_pairs_spin)
        layout.addWidget(form_widget)

        self.auto_run_check = QCheckBox("Auto-run")
        self.auto_run_check.setChecked(bool(self.auto_run))
        self.auto_run_check.toggled.connect(self._on_auto_run_toggled)
        layout.addWidget(self.auto_run_check)

        run_button = QPushButton("Find matched pairs")
        run_button.clicked.connect(self.commit)
        layout.addWidget(run_button)
        layout.addStretch(1)

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self.data = data
        self._populate_combos()
        self.status_label.setText(
            format_loaded_status(len(data), item_label="rows") if data is not None else format_waiting_status()
        )
        self._maybe_autorun()

    def _on_smiles_changed(self, text: str) -> None:
        self.smiles_var_name = text
        self._maybe_autorun()

    def _on_property_changed(self, text: str) -> None:
        self.property_var_name = text
        self._maybe_autorun()

    def _on_shared_changed(self, value: int) -> None:
        self.min_shared_atoms = int(value)
        self._maybe_autorun()

    def _on_max_pairs_changed(self, value: int) -> None:
        self.max_pairs = int(value)
        self._maybe_autorun()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        self.auto_run = bool(checked)
        self._maybe_autorun()

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and self.data is not None and len(self.data) > 0:
            self.commit()

    def _populate_combos(self) -> None:
        self.smiles_combo.blockSignals(True)
        self.property_combo.blockSignals(True)
        try:
            self.smiles_combo.clear()
            self.property_combo.clear()
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

            numeric_vars = _find_numeric_candidates(self.data)
            self.property_combo.addItem("")
            self.property_combo.addItems([variable.name for variable in numeric_vars])
            if self.property_var_name and self.property_var_name in [variable.name for variable in numeric_vars]:
                self.property_combo.setCurrentText(self.property_var_name)
        finally:
            self.smiles_combo.blockSignals(False)
            self.property_combo.blockSignals(False)

    def commit(self) -> None:
        if self.data is None or len(self.data) == 0:
            self.Outputs.pair_table.send(None)
            self.Outputs.pair_compounds.send(None)
            self.status_label.setText(format_no_input_status("input data"))
            return

        smiles = _table_smiles(self.data, self.smiles_var_name)
        property_values = None
        if self.property_var_name:
            variable = next((var for var in _find_numeric_candidates(self.data) if var.name == self.property_var_name), None)
            if variable is not None:
                property_values = [
                    None if np.isnan(value) else float(value)
                    for value in np.asarray(self.data.get_column(variable), dtype=float)
                ]

        rows = find_matched_pairs(
            smiles,
            property_values,
            min_shared_atoms=int(self.min_shared_atoms),
            max_pairs=int(self.max_pairs),
        )
        self.Outputs.pair_table.send(self._pairs_table(rows))
        compound_indices = sorted({row.index_a for row in rows} | {row.index_b for row in rows})
        self.Outputs.pair_compounds.send(self.data[compound_indices] if compound_indices else self.data[:0])
        self.status_label.setText(format_result_count_status(len(rows), item_label="matched pairs"))

    def _pairs_table(self, rows) -> Table:
        attrs = [
            ContinuousVariable("Index A"),
            ContinuousVariable("Index B"),
            ContinuousVariable("Shared Heavy Atoms"),
            ContinuousVariable("Delta Property"),
        ]
        metas = [
            StringVariable("SMILES A"),
            StringVariable("SMILES B"),
            StringVariable("Transformation"),
        ]
        domain = Domain(attrs, metas=metas)
        X = np.array(
            [[row.index_a, row.index_b, row.shared_heavy_atoms, np.nan if row.delta_property is None else row.delta_property] for row in rows],
            dtype=float,
        ) if rows else np.zeros((0, 4), dtype=float)
        metas_arr = np.array([[row.smiles_a, row.smiles_b, row.transformation] for row in rows], dtype=object) if rows else np.zeros((0, 3), dtype=object)
        return Table.from_numpy(domain, X=X, metas=metas_arr)
