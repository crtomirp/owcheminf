from __future__ import annotations

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from Orange.data import (
    ContinuousVariable,
    DiscreteVariable,
    Domain,
    StringVariable,
    Table,
)
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.splitter_service import SplitConfig, split_dataset
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_failed_status,
    format_loaded_status,
    format_no_input_status,
    format_waiting_status,
)
from chem_inf_widgets.widgets.utils import send_output_values, show_service_issues


def _find_smiles_vars(data: Table) -> list[StringVariable]:
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    preferred = [
        variable
        for variable in variables
        if isinstance(variable, StringVariable) and variable.name.strip().lower() in {"smiles", "canonical_smiles", "smile"}
    ]
    if preferred:
        return [
            *preferred,
            *[variable for variable in variables if isinstance(variable, StringVariable) and variable not in preferred],
        ]
    return [variable for variable in variables if isinstance(variable, StringVariable)]


def _target_candidate_vars(data: Table) -> list:
    variables = list(data.domain.class_vars) + list(data.domain.attributes) + list(data.domain.metas)
    out = []
    seen = set()
    for variable in variables:
        if isinstance(variable, (ContinuousVariable, DiscreteVariable)) and variable.name not in seen:
            out.append(variable)
            seen.add(variable.name)
    return out


class OWScaffoldSplitter(OWWidget):
    name = "Scaffold Splitter"
    description = "Split a dataset into train/validation/test partitions using scaffold, random, or activity-stratified logic."
    icon = "icons/analysis/owscaffoldsplitterwidget.svg"
    priority = 139

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        train_data = Output("Train Data", Table, default=True)
        validation_data = Output("Validation Data", Table)
        test_data = Output("Test Data", Table)
        summary = Output("Split Summary", Table)

    smiles_var_name: str = Setting("")
    target_var_name: str = Setting("")
    method_idx: int = Setting(0)
    scaffold_kind_idx: int = Setting(0)
    train_fraction: float = Setting(0.7)
    validation_fraction: float = Setting(0.15)
    test_fraction: float = Setting(0.15)
    random_seed: int = Setting(42)
    auto_run: bool = Setting(True)

    _KINDS = [("Murcko", "murcko"), ("Generic Murcko", "generic")]
    _METHODS = [("Scaffold", "scaffold"), ("Random", "random"), ("Activity-stratified", "activity_stratified")]

    def __init__(self) -> None:
        super().__init__()
        self.data: Table | None = None
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

        self.method_combo = QComboBox()
        self.method_combo.addItems([label for label, _method in self._METHODS])
        self.method_combo.setCurrentIndex(int(self.method_idx))
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("Split method:", self.method_combo)

        self.target_combo = QComboBox()
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        form.addRow("Target column:", self.target_combo)

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

        run_button = QPushButton("Create split")
        run_button.clicked.connect(self.commit)
        layout.addWidget(run_button)
        layout.addStretch(1)

    @Inputs.data
    def set_data(self, data: Table | None) -> None:
        self.data = data
        clear_widget_messages(self)
        self._populate_smiles_combo()
        self._populate_target_combo()
        self._update_control_visibility()
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

    def _populate_target_combo(self) -> None:
        self.target_combo.blockSignals(True)
        try:
            self.target_combo.clear()
            if self.data is None:
                return
            target_vars = _target_candidate_vars(self.data)
            self.target_combo.addItems([variable.name for variable in target_vars])
            if target_vars:
                names = [variable.name for variable in target_vars]
                if self.target_var_name in names:
                    self.target_combo.setCurrentText(self.target_var_name)
                else:
                    self.target_var_name = names[0]
                    self.target_combo.setCurrentText(self.target_var_name)
            else:
                self.target_var_name = ""
        finally:
            self.target_combo.blockSignals(False)

    def _update_control_visibility(self) -> None:
        method = self._METHODS[int(self.method_idx)][1]
        is_scaffold = method == "scaffold"
        is_stratified = method == "activity_stratified"
        self.smiles_combo.setEnabled(is_scaffold)
        self.kind_combo.setEnabled(is_scaffold)
        self.target_combo.setEnabled(is_stratified)

    def _on_smiles_changed(self, text: str) -> None:
        self.smiles_var_name = text
        self._maybe_autorun()

    def _on_method_changed(self, index: int) -> None:
        self.method_idx = int(index)
        self._update_control_visibility()
        self._maybe_autorun()

    def _on_target_changed(self, text: str) -> None:
        self.target_var_name = text
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
            send_output_values(
                (self.Outputs.train_data, None),
                (self.Outputs.validation_data, None),
                (self.Outputs.test_data, None),
                (self.Outputs.summary, None),
            )
            self.status_label.setText(format_no_input_status("input data"))
            return

        clear_widget_messages(self)
        result = split_dataset(
            self.data,
            SplitConfig(
                method=self._METHODS[int(self.method_idx)][1],
                test_size=float(self.test_fraction),
                validation_size=float(self.validation_fraction),
                random_state=int(self.random_seed),
                target_column=self.target_var_name or None,
                scaffold_kind=self._KINDS[int(self.scaffold_kind_idx)][1],
            ),
        )

        show_service_issues(self, result.issues, subject="splitter")
        if any(issue.severity == "error" for issue in result.issues):
            send_output_values(
                (self.Outputs.train_data, None),
                (self.Outputs.validation_data, None),
                (self.Outputs.test_data, None),
                (self.Outputs.summary, None),
            )
            self.status_label.setText(format_failed_status(result.issues[0].message))
            return

        train_idx = result.train_indices
        val_idx = result.validation_indices
        test_idx = result.test_indices

        send_output_values(
            (self.Outputs.train_data, self.data[train_idx] if train_idx else self.data[:0]),
            (self.Outputs.validation_data, self.data[val_idx] if val_idx else self.data[:0]),
            (self.Outputs.test_data, self.data[test_idx] if test_idx else self.data[:0]),
            (self.Outputs.summary, self._summary_table(result)),
        )
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
        n_rows = 0 if self.data is None else len(self.data)
        total = max(n_rows, 1)
        assigned = set(result.train_indices) | set(result.validation_indices) | set(result.test_indices)
        rows = [
            ("train", len(result.train_indices)),
            ("validation", len(result.validation_indices)),
            ("test", len(result.test_indices)),
        ]
        if len(assigned) < n_rows:
            rows.append(("unassigned", n_rows - len(assigned)))
        X = np.array([[count, round(count / total, 4)] for _name, count in rows], dtype=float)
        metas = np.array([[name] for name, _count in rows], dtype=object)
        return Table.from_numpy(domain, X=X, metas=metas)
