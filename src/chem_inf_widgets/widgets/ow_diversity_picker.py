from __future__ import annotations

from typing import List, Optional

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from Orange.data import StringVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.diversity_service import (
    DiversitySelectionResult,
    select_diverse_subset,
)
from chem_inf_widgets.widgets.ui_helpers import (
    format_done_status,
    format_error_status,
    format_no_input_status,
    set_widget_error,
)


def _find_smiles_vars(data: Table) -> List[StringVariable]:
    wanted = {"smiles", "canonical_smiles", "smile"}
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)

    preferred = [var for var in variables if isinstance(var, StringVariable) and var.name.strip().lower() in wanted]
    if preferred:
        return preferred + [var for var in variables if isinstance(var, StringVariable) and var not in preferred]
    return [var for var in variables if isinstance(var, StringVariable)]


def _table_smiles(data: Table, var_name: str) -> List[str]:
    variables = _find_smiles_vars(data)
    selected_var = next((var for var in variables if var.name == var_name), None)
    if selected_var is None:
        raise ValueError("No SMILES column selected.")

    col = data.get_column(selected_var)
    return ["" if value is None else str(value).strip() for value in col]


def _molecule_smiles(molecules: List[ChemMol]) -> List[str]:
    smiles = []
    for molecule in molecules:
        value = molecule.get_prop("SMILES") or molecule.get_prop("smiles")
        if isinstance(value, str) and value.strip():
            smiles.append(value.strip())
            continue
        try:
            smiles.append(molecule.canonical_smiles())
        except Exception:
            smiles.append("")
    return smiles


class OWDiversityPicker(OWWidget):
    name = "Diversity Picker"
    description = "Select a diverse subset of compounds using MaxMin, sphere exclusion, or Butina clustering."
    icon = "icons/analysis/owdiversitypickerwidget.svg"
    priority = 134

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        selected_data = Output("Selected Data", Table)
        remainder_data = Output("Remainder Data", Table)
        selected_molecules = Output("Selected Molecules", list, auto_summary=False)
        remainder_molecules = Output("Remainder Molecules", list, auto_summary=False)

    method_idx: int = Setting(0)
    smiles_var_name: str = Setting("")
    n_select: int = Setting(25)
    seed_idx: int = Setting(0)
    sphere_radius: float = Setting(0.35)
    butina_threshold: float = Setting(0.40)
    random_seed: int = Setting(42)
    auto_run: bool = Setting(True)

    _METHODS = [
        ("MaxMin", "maxmin"),
        ("Sphere exclusion", "sphere_exclusion"),
        ("Butina clusters", "butina"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[Table] = None
        self.molecules: List[ChemMol] = []

        self.mainArea.hide()

        root = QWidget(self.controlArea)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        self.controlArea.layout().addWidget(root)

        self.status_label = QLabel("Waiting for input…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)

        self.smiles_combo = QComboBox()
        self.smiles_combo.currentTextChanged.connect(self._on_smiles_changed)
        form.addRow("SMILES column:", self.smiles_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems([label for label, _method in self._METHODS])
        self.method_combo.setCurrentIndex(int(self.method_idx))
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("Method:", self.method_combo)

        self.n_select_spin = QSpinBox()
        self.n_select_spin.setRange(1, 100000)
        self.n_select_spin.setValue(int(self.n_select))
        self.n_select_spin.valueChanged.connect(self._on_n_select_changed)
        form.addRow("Target count / clusters:", self.n_select_spin)

        self.seed_idx_spin = QSpinBox()
        self.seed_idx_spin.setRange(0, 100000)
        self.seed_idx_spin.setValue(int(self.seed_idx))
        self.seed_idx_spin.valueChanged.connect(self._on_seed_idx_changed)
        form.addRow("Seed index (MaxMin):", self.seed_idx_spin)

        self.sphere_radius_spin = QDoubleSpinBox()
        self.sphere_radius_spin.setRange(0.01, 0.99)
        self.sphere_radius_spin.setSingleStep(0.01)
        self.sphere_radius_spin.setDecimals(2)
        self.sphere_radius_spin.setValue(float(self.sphere_radius))
        self.sphere_radius_spin.valueChanged.connect(self._on_sphere_radius_changed)
        form.addRow("Sphere radius:", self.sphere_radius_spin)

        self.butina_threshold_spin = QDoubleSpinBox()
        self.butina_threshold_spin.setRange(0.01, 1.00)
        self.butina_threshold_spin.setSingleStep(0.01)
        self.butina_threshold_spin.setDecimals(2)
        self.butina_threshold_spin.setValue(float(self.butina_threshold))
        self.butina_threshold_spin.valueChanged.connect(self._on_butina_threshold_changed)
        form.addRow("Butina distance threshold:", self.butina_threshold_spin)

        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 1_000_000)
        self.random_seed_spin.setValue(int(self.random_seed))
        self.random_seed_spin.valueChanged.connect(self._on_random_seed_changed)
        form.addRow("Random seed:", self.random_seed_spin)

        layout.addWidget(form_widget)

        self.auto_run_check = QCheckBox("Auto-run")
        self.auto_run_check.setChecked(bool(self.auto_run))
        self.auto_run_check.toggled.connect(self._on_auto_run_toggled)
        layout.addWidget(self.auto_run_check)

        self.run_button = QPushButton("Select diverse subset")
        self.run_button.clicked.connect(self.commit)
        layout.addWidget(self.run_button)

        layout.addStretch(1)
        self._update_smiles_controls()
        self._update_method_controls()

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self.data = data
        self._populate_smiles_combo()
        self._set_status(self._input_summary())
        self._maybe_autorun()

    @Inputs.molecules
    def set_molecules(self, molecules: Optional[list]) -> None:
        self.molecules = [molecule for molecule in (molecules or []) if isinstance(molecule, ChemMol)]
        self._set_status(self._input_summary())
        self._maybe_autorun()

    def _populate_smiles_combo(self) -> None:
        self.smiles_combo.blockSignals(True)
        try:
            self.smiles_combo.clear()
            if self.data is None:
                self._update_smiles_controls()
                return

            smiles_vars = _find_smiles_vars(self.data)
            self.smiles_combo.addItems([var.name for var in smiles_vars])
            if smiles_vars:
                if self.smiles_var_name and self.smiles_var_name in [var.name for var in smiles_vars]:
                    self.smiles_combo.setCurrentText(self.smiles_var_name)
                else:
                    self.smiles_var_name = smiles_vars[0].name
                    self.smiles_combo.setCurrentText(self.smiles_var_name)
        finally:
            self.smiles_combo.blockSignals(False)
        self._update_smiles_controls()

    def _update_smiles_controls(self) -> None:
        has_table = self.data is not None
        self.smiles_combo.setEnabled(has_table)

    def _update_method_controls(self) -> None:
        method = self._METHODS[self.method_idx][1]
        is_maxmin = method == "maxmin"
        is_sphere = method == "sphere_exclusion"
        is_butina = method == "butina"

        self.seed_idx_spin.setEnabled(is_maxmin)
        self.sphere_radius_spin.setEnabled(is_sphere)
        self.butina_threshold_spin.setEnabled(is_butina)

    def _on_smiles_changed(self, text: str) -> None:
        self.smiles_var_name = text
        self._maybe_autorun()

    def _on_method_changed(self, index: int) -> None:
        self.method_idx = int(index)
        self._update_method_controls()
        self._maybe_autorun()

    def _on_n_select_changed(self, value: int) -> None:
        self.n_select = int(value)
        self._maybe_autorun()

    def _on_seed_idx_changed(self, value: int) -> None:
        self.seed_idx = int(value)
        self._maybe_autorun()

    def _on_sphere_radius_changed(self, value: float) -> None:
        self.sphere_radius = float(value)
        self._maybe_autorun()

    def _on_butina_threshold_changed(self, value: float) -> None:
        self.butina_threshold = float(value)
        self._maybe_autorun()

    def _on_random_seed_changed(self, value: int) -> None:
        self.random_seed = int(value)
        self._maybe_autorun()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        self.auto_run = bool(checked)
        self._maybe_autorun()

    def _input_summary(self) -> str:
        table_rows = 0 if self.data is None else len(self.data)
        return f"Input: Table rows={table_rows}, Molecules={len(self.molecules)}"

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and (self.data is not None or self.molecules):
            self.commit()

    def _input_smiles(self) -> List[str]:
        if self.data is not None:
            return _table_smiles(self.data, self.smiles_var_name)
        return _molecule_smiles(self.molecules)

    @staticmethod
    def _subset_table(data: Optional[Table], indices: List[int]) -> Optional[Table]:
        if data is None:
            return None
        return data[indices] if indices else data[:0]

    @staticmethod
    def _subset_molecules(molecules: List[ChemMol], indices: List[int]) -> List[ChemMol]:
        return [molecules[idx] for idx in indices if 0 <= idx < len(molecules)]

    def commit(self) -> None:
        if self.data is None and not self.molecules:
            self._set_status(format_no_input_status())
            self.Outputs.selected_data.send(None)
            self.Outputs.remainder_data.send(None)
            self.Outputs.selected_molecules.send([])
            self.Outputs.remainder_molecules.send([])
            return

        try:
            smiles = self._input_smiles()
        except ValueError as exc:
            set_widget_error(self, str(exc))
            self._set_status(format_error_status(str(exc)))
            return

        method = self._METHODS[self.method_idx][1]
        result = select_diverse_subset(
            smiles,
            method=method,
            n_select=int(self.n_select),
            seed_idx=int(self.seed_idx),
            radius=float(self.sphere_radius),
            n_clusters=int(self.n_select),
            threshold=float(self.butina_threshold),
            random_seed=int(self.random_seed),
        )
        self._send_outputs(result, len(smiles))

    def _send_outputs(self, result: DiversitySelectionResult, total_count: int) -> None:
        selected_indices = result.selected_indices
        remainder_indices = [idx for idx in range(total_count) if idx not in set(selected_indices)]

        selected_data = self._subset_table(self.data, selected_indices)
        remainder_data = self._subset_table(self.data, remainder_indices)

        aligned_molecules = self.molecules if len(self.molecules) == total_count else []
        selected_molecules = self._subset_molecules(aligned_molecules, selected_indices)
        remainder_molecules = self._subset_molecules(aligned_molecules, remainder_indices)

        self.Outputs.selected_data.send(selected_data)
        self.Outputs.remainder_data.send(remainder_data)
        self.Outputs.selected_molecules.send(selected_molecules)
        self.Outputs.remainder_molecules.send(remainder_molecules)

        input_metrics = result.metrics_input
        selected_metrics = result.metrics_selected
        self._set_status(
            format_done_status(
                f"selected={len(selected_indices)}/{input_metrics.n_compounds}",
                f"invalid skipped={len(result.failed_indices)}",
                f"diversity {input_metrics.diversity_score:.4f}->{selected_metrics.diversity_score:.4f}",
                prefix="Selected",
            )
        )


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWDiversityPicker).run()
