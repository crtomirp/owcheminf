from __future__ import annotations

from typing import List, Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.scaffold_splitter_service import split_by_scaffold
from chem_inf_widgets.widgets.ui_helpers import (
    format_done_status,
    format_loaded_status,
    format_no_input_status,
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


class OWScaffoldSplitter(OWWidget):
    name = "Scaffold Splitter"
    description = "Split a dataset into train/validation/test partitions by scaffold groups."
    icon = "icons/analysis/owscaffoldsplitterwidget.svg"
    priority = 139

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        train_data = Output("Train Data", Table)
        validation_data = Output("Validation Data", Table)
        test_data = Output("Test Data", Table)
        summary = Output("Split Summary", Table)

    smiles_var_name: str = Setting("")
    scaffold_kind_idx: int = Setting(0)
    train_fraction: float = Setting(0.7)
    validation_fraction: float = Setting(0.15)
    test_fraction: float = Setting(0.15)
    random_seed: int = Setting(42)
    auto_run: bool = Setting(True)

    _KINDS = [("Murcko", "murcko"), ("Generic Murcko", "generic")]

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

        self.kind_combo = QComboBox()
        self.kind_combo.addItems([label for label, _kind in self._KINDS])
        self.kind_combo.setCurrentIndex(int(self.scaffold_kind_idx))
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        form.addRow("Scaffold kind:", self.kind_combo)

        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0.0, 1.0)
        self.train_spin.setSingleStep(0.05)
        self.train_spin.setValue(float(self.train_fraction))
        self.train_spin.valueChanged.connect(self._on_train_fraction_changed)
        form.addRow("Train fraction:", self.train_spin)

        self.validation_spin = QDoubleSpinBox()
        self.validation_spin.setRange(0.0, 1.0)
        self.validation_spin.setSingleStep(0.05)
        self.validation_spin.setValue(float(self.validation_fraction))
        self.validation_spin.valueChanged.connect(self._on_validation_fraction_changed)
        form.addRow("Validation fraction:", self.validation_spin)

        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.0, 1.0)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(float(self.test_fraction))
        self.test_spin.valueChanged.connect(self._on_test_fraction_changed)
        form.addRow("Test fraction:", self.test_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        self.seed_spin.setValue(int(self.random_seed))
        self.seed_spin.valueChanged.connect(self._on_random_seed_changed)
        form.addRow("Random seed:", self.seed_spin)
        layout.addWidget(form_widget)

        self.auto_run_check = QCheckBox("Auto-run")
        self.auto_run_check.setChecked(bool(self.auto_run))
        self.auto_run_check.toggled.connect(self._on_auto_run_toggled)
        layout.addWidget(self.auto_run_check)

        run_button = QPushButton("Create scaffold split")
        run_button.clicked.connect(self.commit)
        layout.addWidget(run_button)
        layout.addStretch(1)

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self.data = data
        self._populate_smiles_combo()
        self.status_label.setText(
            format_loaded_status(len(data), item_label="rows") if data is not None else format_waiting_status()
        )
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
                    self.smiles_combo.setCurrentText(self.smiles_var_name)
        finally:
            self.smiles_combo.blockSignals(False)

    def _on_smiles_changed(self, text: str) -> None:
        self.smiles_var_name = text
        self._maybe_autorun()

    def _on_kind_changed(self, index: int) -> None:
        self.scaffold_kind_idx = int(index)
        self._maybe_autorun()

    def _on_train_fraction_changed(self, value: float) -> None:
        self.train_fraction = float(value)
        self._maybe_autorun()

    def _on_validation_fraction_changed(self, value: float) -> None:
        self.validation_fraction = float(value)
        self._maybe_autorun()

    def _on_test_fraction_changed(self, value: float) -> None:
        self.test_fraction = float(value)
        self._maybe_autorun()

    def _on_random_seed_changed(self, value: int) -> None:
        self.random_seed = int(value)
        self._maybe_autorun()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        self.auto_run = bool(checked)
        self._maybe_autorun()

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and self.data is not None and len(self.data) > 0:
            self.commit()

    def commit(self) -> None:
        if self.data is None or len(self.data) == 0:
            self.Outputs.train_data.send(None)
            self.Outputs.validation_data.send(None)
            self.Outputs.test_data.send(None)
            self.Outputs.summary.send(None)
            self.status_label.setText(format_no_input_status("input data"))
            return

        smiles = _table_smiles(self.data, self.smiles_var_name)
        kind = self._KINDS[self.scaffold_kind_idx][1]
        result = split_by_scaffold(
            smiles,
            train_fraction=float(self.train_fraction),
            validation_fraction=float(self.validation_fraction),
            test_fraction=float(self.test_fraction),
            scaffold_kind=kind,
            random_seed=int(self.random_seed),
        )
        split_map = {assignment.index: assignment.split for assignment in result.assignments}

        train_idx = [index for index, split in split_map.items() if split == "train"]
        val_idx = [index for index, split in split_map.items() if split == "validation"]
        test_idx = [index for index, split in split_map.items() if split == "test"]

        self.Outputs.train_data.send(self.data[train_idx] if train_idx else self.data[:0])
        self.Outputs.validation_data.send(self.data[val_idx] if val_idx else self.data[:0])
        self.Outputs.test_data.send(self.data[test_idx] if test_idx else self.data[:0])
        self.Outputs.summary.send(self._summary_table(result))
        self.status_label.setText(
            format_done_status(
                f"train={len(train_idx)}",
                f"validation={len(val_idx)}",
                f"test={len(test_idx)}",
            )
        )

    def _summary_table(self, result) -> Table:
        split_var = StringVariable("Split")
        count_var = ContinuousVariable("Count")
        fraction_var = ContinuousVariable("Fraction")
        domain = Domain([count_var, fraction_var], metas=[split_var])
        X = np.array([[row.count, row.fraction] for row in result.summaries], dtype=float)
        metas = np.array([[row.split] for row in result.summaries], dtype=object)
        return Table.from_numpy(domain, X=X, metas=metas)
