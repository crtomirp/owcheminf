from __future__ import annotations

from typing import List, Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.activity_cliff_service import (
    ActivityCliffResult,
    find_activity_cliffs,
    scaffold_activity_summary,
)
from chem_inf_widgets.widgets.ui_helpers import (
    format_done_status,
    format_error_status,
    format_no_input_status,
    set_widget_error,
)


def _string_vars(data: Table) -> List[StringVariable]:
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    wanted = {"smiles", "canonical_smiles", "smile"}
    preferred = [var for var in variables if isinstance(var, StringVariable) and var.name.strip().lower() in wanted]
    if preferred:
        return preferred + [var for var in variables if isinstance(var, StringVariable) and var not in preferred]
    return [var for var in variables if isinstance(var, StringVariable)]


def _continuous_vars(data: Table) -> List[ContinuousVariable]:
    variables = list(data.domain.class_vars) + list(data.domain.attributes) + list(data.domain.metas)
    out: List[ContinuousVariable] = []
    seen = set()
    for var in variables:
        if isinstance(var, ContinuousVariable) and var.name not in seen:
            out.append(var)
            seen.add(var.name)
    return out


def _table_str_list(data: Table, var_name: str) -> List[str]:
    variable = next((var for var in _string_vars(data) if var.name == var_name), None)
    if variable is None:
        raise ValueError("No SMILES column selected.")
    col = data.get_column(variable)
    return ["" if value is None else str(value).strip() for value in col]


def _table_float_list(data: Table, var_name: str) -> List[float]:
    variable = next((var for var in _continuous_vars(data) if var.name == var_name), None)
    if variable is None:
        raise ValueError("No activity column selected.")
    col = data.get_column(variable)
    return [float(value) if value is not None else float("nan") for value in col]


def _table_name_list(data: Table) -> List[str]:
    variables = _string_vars(data)
    for candidate in ("name", "title", "compound", "compound_name"):
        variable = next((var for var in variables if var.name.strip().lower() == candidate), None)
        if variable is not None:
            col = data.get_column(variable)
            return ["" if value is None else str(value).strip() for value in col]
    return [""] * len(data)


def _pairs_table(result: ActivityCliffResult) -> Table:
    domain = Domain(
        [
            ContinuousVariable("Similarity"),
            ContinuousVariable("Activity Ratio"),
            ContinuousVariable("Cliff Score"),
            ContinuousVariable("Activity A"),
            ContinuousVariable("Activity B"),
        ],
        metas=[
            StringVariable("Name A"),
            StringVariable("Name B"),
            StringVariable("SMILES A"),
            StringVariable("SMILES B"),
            StringVariable("Higher Active"),
        ],
    )
    if not result.pairs:
        return Table.from_numpy(
            domain,
            X=np.zeros((0, 5), dtype=float),
            metas=np.zeros((0, 5), dtype=object),
        )

    X = np.array(
        [
            [pair.similarity, pair.activity_ratio, pair.cliff_score, pair.activity_a, pair.activity_b]
            for pair in result.pairs
        ],
        dtype=float,
    )
    metas = np.array(
        [
            [pair.name_a, pair.name_b, pair.smiles_a, pair.smiles_b, pair.higher_active]
            for pair in result.pairs
        ],
        dtype=object,
    )
    return Table.from_numpy(domain, X=X, metas=metas)


def _scaffold_summary_table(rows) -> Table:
    domain = Domain(
        [
            ContinuousVariable("Count"),
            ContinuousVariable("Mean Activity"),
            ContinuousVariable("Best Activity"),
            ContinuousVariable("Worst Activity"),
            ContinuousVariable("Std Activity"),
        ],
        metas=[StringVariable("Scaffold")],
    )
    if not rows:
        return Table.from_numpy(
            domain,
            X=np.zeros((0, 5), dtype=float),
            metas=np.zeros((0, 1), dtype=object),
        )

    X = np.array(
        [
            [float(row.count), row.mean_activity, row.best_activity, row.worst_activity, row.std_activity]
            for row in rows
        ],
        dtype=float,
    )
    metas = np.array([[row.scaffold] for row in rows], dtype=object)
    return Table.from_numpy(domain, X=X, metas=metas)


class OWActivityCliffFinder(OWWidget):
    name = "Activity Cliff Finder"
    description = "Find highly similar compounds with large activity differences."
    icon = "icons/analysis/owactivityclifffinderwidget.png"
    priority = 137

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        cliff_pairs = Output("Cliff Pairs", Table)
        cliff_compounds = Output("Cliff Compounds", Table)
        scaffold_summary = Output("Scaffold Summary", Table)

    smiles_var_name: str = Setting("")
    activity_var_name: str = Setting("")
    similarity_threshold: float = Setting(0.6)
    activity_fold_threshold: float = Setting(10.0)
    activity_scale_idx: int = Setting(0)
    max_pairs: int = Setting(250)
    auto_run: bool = Setting(True)

    _ACTIVITY_SCALES = [
        ("Linear potency (IC50/Ki/EC50)", False),
        ("Log potency (pIC50/pKi)", True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[Table] = None

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

        self.activity_combo = QComboBox()
        self.activity_combo.currentTextChanged.connect(self._on_activity_changed)
        form.addRow("Activity column:", self.activity_combo)

        self.scale_combo = QComboBox()
        self.scale_combo.addItems([label for label, _value in self._ACTIVITY_SCALES])
        self.scale_combo.setCurrentIndex(int(self.activity_scale_idx))
        self.scale_combo.currentIndexChanged.connect(self._on_scale_changed)
        form.addRow("Activity scale:", self.scale_combo)

        self.similarity_spin = QDoubleSpinBox()
        self.similarity_spin.setRange(0.1, 1.0)
        self.similarity_spin.setSingleStep(0.05)
        self.similarity_spin.setDecimals(2)
        self.similarity_spin.setValue(float(self.similarity_threshold))
        self.similarity_spin.valueChanged.connect(self._on_similarity_changed)
        form.addRow("Similarity threshold:", self.similarity_spin)

        self.activity_spin = QDoubleSpinBox()
        self.activity_spin.setRange(1.1, 1_000_000.0)
        self.activity_spin.setSingleStep(1.0)
        self.activity_spin.setDecimals(2)
        self.activity_spin.setValue(float(self.activity_fold_threshold))
        self.activity_spin.valueChanged.connect(self._on_activity_threshold_changed)
        form.addRow("Activity fold threshold:", self.activity_spin)

        self.max_pairs_spin = QSpinBox()
        self.max_pairs_spin.setRange(1, 100_000)
        self.max_pairs_spin.setValue(int(self.max_pairs))
        self.max_pairs_spin.valueChanged.connect(self._on_max_pairs_changed)
        form.addRow("Max pairs:", self.max_pairs_spin)

        layout.addWidget(form_widget)

        self.auto_run_check = QCheckBox("Auto-run")
        self.auto_run_check.setChecked(bool(self.auto_run))
        self.auto_run_check.toggled.connect(self._on_auto_run_toggled)
        layout.addWidget(self.auto_run_check)

        self.run_button = QPushButton("Find activity cliffs")
        self.run_button.clicked.connect(self.commit)
        layout.addWidget(self.run_button)

        layout.addStretch(1)
        self._update_controls()

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self.data = data
        self._populate_combos()
        self._set_status(self._input_summary())
        self._maybe_autorun()

    def _populate_combos(self) -> None:
        self.smiles_combo.blockSignals(True)
        self.activity_combo.blockSignals(True)
        try:
            self.smiles_combo.clear()
            self.activity_combo.clear()
            if self.data is None:
                return

            smiles_vars = _string_vars(self.data)
            activity_vars = _continuous_vars(self.data)
            self.smiles_combo.addItems([var.name for var in smiles_vars])
            self.activity_combo.addItems([var.name for var in activity_vars])

            if smiles_vars:
                names = [var.name for var in smiles_vars]
                selected = self.smiles_var_name if self.smiles_var_name in names else names[0]
                self.smiles_var_name = selected
                self.smiles_combo.setCurrentText(selected)

            if activity_vars:
                names = [var.name for var in activity_vars]
                selected = self.activity_var_name if self.activity_var_name in names else names[0]
                self.activity_var_name = selected
                self.activity_combo.setCurrentText(selected)
        finally:
            self.smiles_combo.blockSignals(False)
            self.activity_combo.blockSignals(False)
        self._update_controls()

    def _update_controls(self) -> None:
        has_data = self.data is not None
        self.smiles_combo.setEnabled(has_data)
        self.activity_combo.setEnabled(has_data)
        self.run_button.setEnabled(has_data)

    def _on_smiles_changed(self, text: str) -> None:
        self.smiles_var_name = text
        self._maybe_autorun()

    def _on_activity_changed(self, text: str) -> None:
        self.activity_var_name = text
        self._maybe_autorun()

    def _on_scale_changed(self, index: int) -> None:
        self.activity_scale_idx = int(index)
        self._maybe_autorun()

    def _on_similarity_changed(self, value: float) -> None:
        self.similarity_threshold = float(value)
        self._maybe_autorun()

    def _on_activity_threshold_changed(self, value: float) -> None:
        self.activity_fold_threshold = float(value)
        self._maybe_autorun()

    def _on_max_pairs_changed(self, value: int) -> None:
        self.max_pairs = int(value)
        self._maybe_autorun()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        self.auto_run = bool(checked)
        self._maybe_autorun()

    def _input_summary(self) -> str:
        return f"Input: Table rows={0 if self.data is None else len(self.data)}"

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and self.data is not None:
            self.commit()

    def commit(self) -> None:
        self.clear_messages()
        if self.data is None:
            self._set_status(format_no_input_status())
            self.Outputs.cliff_pairs.send(None)
            self.Outputs.cliff_compounds.send(None)
            self.Outputs.scaffold_summary.send(None)
            return

        try:
            smiles = _table_str_list(self.data, self.smiles_var_name)
            activities = _table_float_list(self.data, self.activity_var_name)
            names = _table_name_list(self.data)
            result = find_activity_cliffs(
                smiles,
                activities,
                names=names,
                similarity_threshold=float(self.similarity_threshold),
                activity_fold_threshold=float(self.activity_fold_threshold),
                activity_log_scale=bool(self._ACTIVITY_SCALES[self.activity_scale_idx][1]),
                max_pairs=int(self.max_pairs),
            )
            summary_rows = scaffold_activity_summary(
                smiles,
                activities,
                activity_log_scale=bool(self._ACTIVITY_SCALES[self.activity_scale_idx][1]),
            )
        except ValueError as exc:
            set_widget_error(self, str(exc))
            self._set_status(format_error_status(str(exc)))
            self.Outputs.cliff_pairs.send(None)
            self.Outputs.cliff_compounds.send(None)
            self.Outputs.scaffold_summary.send(None)
            return

        self.Outputs.cliff_pairs.send(_pairs_table(result))
        self.Outputs.cliff_compounds.send(self.data[result.unique_cliff_indices] if result.unique_cliff_indices else self.data[:0])
        self.Outputs.scaffold_summary.send(_scaffold_summary_table(summary_rows))

        self._set_status(
            format_done_status(
                f"cliff pairs={len(result.pairs)}",
                f"unique compounds={len(result.unique_cliff_indices)}",
                f"valid rows={len(result.valid_indices)}",
                f"invalid skipped={len(result.failed_indices)}",
                prefix="Found",
            )
        )


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWActivityCliffFinder).run()
