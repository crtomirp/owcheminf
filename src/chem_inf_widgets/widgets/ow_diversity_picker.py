from __future__ import annotations

from html import escape

import numpy as np
import pyqtgraph as pg
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QPixmap
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services import mol_depict
from chem_inf_widgets.chemcore.services.diversity_service import (
    DiversitySelectionResult,
    select_diverse_subset,
)
from chem_inf_widgets.chemcore.services.orange_table_utils import safe_table_from_numpy
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_error_status,
    format_no_input_status,
    set_widget_error,
)

_LABEL_CANDIDATES = {
    "name",
    "compound_name",
    "compound_id",
    "molecule_id",
    "chembl_id",
    "identifier",
    "id",
    "title",
}


def _string_vars(data: Table) -> list[StringVariable]:
    return [
        var
        for var in list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
        if isinstance(var, StringVariable)
    ]


def _find_smiles_vars(data: Table) -> list[StringVariable]:
    wanted = {"smiles", "canonical_smiles", "smile"}
    variables = _string_vars(data)

    preferred = [var for var in variables if isinstance(var, StringVariable) and var.name.strip().lower() in wanted]
    if preferred:
        return preferred + [var for var in variables if isinstance(var, StringVariable) and var not in preferred]
    return [var for var in variables if isinstance(var, StringVariable)]


def _table_smiles(data: Table, var_name: str) -> list[str]:
    variables = _find_smiles_vars(data)
    selected_var = next((var for var in variables if var.name == var_name), None)
    if selected_var is None:
        raise ValueError("No SMILES column selected.")

    col = data.get_column(selected_var)
    return ["" if value is None else str(value).strip() for value in col]


def _molecule_smiles(molecules: list[ChemMol]) -> list[str]:
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


def _find_label_var_name(data: Table, *, exclude_name: str = "") -> str:
    normalized_exclude = str(exclude_name or "").strip().lower()
    variables = [var for var in _string_vars(data) if var.name.strip().lower() != normalized_exclude]
    preferred = [var for var in variables if var.name.strip().lower() in _LABEL_CANDIDATES]
    if preferred:
        return preferred[0].name
    return variables[0].name if variables else ""


def _styled_plot(title: str = "") -> pg.PlotWidget:
    plot_widget = pg.PlotWidget(title=title)
    plot_widget.setBackground("#FFFFFF")
    plot_widget.getPlotItem().getAxis("left").setPen(pg.mkPen("#CBD5E1"))
    plot_widget.getPlotItem().getAxis("bottom").setPen(pg.mkPen("#CBD5E1"))
    plot_widget.getPlotItem().getAxis("left").setTextPen(pg.mkPen("#475569"))
    plot_widget.getPlotItem().getAxis("bottom").setTextPen(pg.mkPen("#475569"))
    plot_widget.showGrid(x=True, y=True, alpha=0.18)
    plot_widget.setMenuEnabled(False)
    return plot_widget


def _set_plot_range(plot: pg.PlotWidget, x_values: np.ndarray, y_values: np.ndarray) -> None:
    if x_values.size == 0 or y_values.size == 0:
        return
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = max(x_span * 0.08, 0.15 if x_span == 0 else 0.0)
    y_pad = max(y_span * 0.08, 0.15 if y_span == 0 else 0.0)
    plot.setXRange(x_min - x_pad, x_max + x_pad, padding=0.0)
    plot.setYRange(y_min - y_pad, y_max + y_pad, padding=0.0)


def _count_card(label: str, value: str) -> str:
    return (
        "<div style='display:inline-block; min-width:160px; margin:6px; padding:10px 12px; "
        "border:1px solid #D0D5DD; border-radius:10px; background:#F8FAFC;'>"
        f"<div style='font-size:12px; color:#475467;'>{escape(label)}</div>"
        f"<div style='font-size:22px; font-weight:600; color:#101828;'>{escape(value)}</div>"
        "</div>"
    )


def _summary_html(result: DiversitySelectionResult) -> str:
    cards = "".join(
        [
            _count_card("Fingerprint", "Morgan r=2, 2048 bits"),
            _count_card("Projection", "PCA"),
            _count_card("Picker", "MaxMinPicker" if result.method == "maxmin" else result.method.replace("_", " ").title()),
            _count_card("Valid", str(result.metrics_input.n_compounds)),
            _count_card("Selected", str(len(result.selected_indices))),
            _count_card("Invalid skipped", str(len(result.failed_indices))),
        ]
    )
    explained = result.explained_variance or []
    explained_html = "".join(
        f"<li>PC{index + 1}: {value * 100:.2f}% variance</li>"
        for index, value in enumerate(explained[:2])
    ) or "<li>Explained variance unavailable.</li>"
    return (
        "<html><body style='font-family: sans-serif;'>"
        "<h2>Diversity Picker</h2>"
        f"{cards}"
        "<h3>Diversity</h3>"
        "<ul>"
        f"<li>Input diversity score: {result.metrics_input.diversity_score:.4f}</li>"
        f"<li>Selected diversity score: {result.metrics_selected.diversity_score:.4f}</li>"
        f"<li>Mean nearest-neighbour distance: {result.metrics_selected.mean_nn_distance:.4f}</li>"
        "</ul>"
        "<h3>PCA projection</h3>"
        f"<ul>{explained_html}</ul>"
        "<h3>Display legend</h3>"
        "<ul>"
        "<li>Blue circles: all valid compounds</li>"
        "<li>Orange stars: selected compounds from the diversity picker</li>"
        "<li>Dark outlined circles: compounds currently inspected from the plot</li>"
        "</ul>"
        "</body></html>"
    )


def _unique_variable_name(existing_names: set[str], wanted: str) -> str:
    if wanted not in existing_names:
        existing_names.add(wanted)
        return wanted
    index = 2
    while f"{wanted}_{index}" in existing_names:
        index += 1
    out = f"{wanted}_{index}"
    existing_names.add(out)
    return out


def _annotation_arrays(result: DiversitySelectionResult, total_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(result.coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] != int(total_count):
        coordinates = np.full((int(total_count), 2), np.nan, dtype=float)
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(coordinates.shape[0], dtype=float)])
    elif coordinates.shape[1] == 0:
        coordinates = np.full((int(total_count), 2), np.nan, dtype=float)
    elif coordinates.shape[1] > 2:
        coordinates = coordinates[:, :2]

    selected_mask = np.zeros(int(total_count), dtype=float)
    for index in result.selected_indices:
        if 0 <= int(index) < len(selected_mask):
            selected_mask[int(index)] = 1.0

    rank = np.full(int(total_count), np.nan, dtype=float)
    for index, value in enumerate(result.selection_ranks):
        if value is not None and 0 <= int(index) < len(rank):
            rank[int(index)] = float(value)

    return coordinates[:, 0], coordinates[:, 1], selected_mask, rank


def _annotated_table_from_data(data: Table, result: DiversitySelectionResult) -> Table:
    total_count = len(data)
    x_col, y_col, selected_col, rank_col = _annotation_arrays(result, total_count)

    existing_names = {var.name for var in list(data.domain.attributes) + list(data.domain.class_vars) + list(data.domain.metas)}
    x_var = ContinuousVariable(_unique_variable_name(existing_names, "chem_space_x"))
    y_var = ContinuousVariable(_unique_variable_name(existing_names, "chem_space_y"))
    selected_var = ContinuousVariable(_unique_variable_name(existing_names, "diversity_selected"))
    rank_var = ContinuousVariable(_unique_variable_name(existing_names, "diversity_rank"))

    attrs = list(data.domain.attributes) + [x_var, y_var, selected_var, rank_var]
    domain = Domain(attrs, data.domain.class_vars, data.domain.metas)
    x_base = np.asarray(data.X, dtype=float)
    x_extra = np.column_stack([x_col, y_col, selected_col, rank_col]).astype(float, copy=False)
    x_out = np.hstack([x_base, x_extra]) if x_base.size else x_extra
    y_out = data.Y if len(data.domain.class_vars) else None
    metas_out = data.metas if len(data.domain.metas) else None
    return safe_table_from_numpy(
        domain,
        X=x_out,
        Y=y_out,
        metas=metas_out,
        name=getattr(data, "name", "Diversity Annotated Data") or "Diversity Annotated Data",
    )


def _annotated_table_from_molecules(molecules: list[ChemMol], result: DiversitySelectionResult) -> Table:
    total_count = len(molecules)
    x_col, y_col, selected_col, rank_col = _annotation_arrays(result, total_count)
    attrs = [
        ContinuousVariable("chem_space_x"),
        ContinuousVariable("chem_space_y"),
        ContinuousVariable("diversity_selected"),
        ContinuousVariable("diversity_rank"),
    ]
    name_var = StringVariable("Name")
    smiles_var = StringVariable("SMILES")
    smiles_var.attributes["format"] = "SMILES"
    domain = Domain(attrs, metas=[name_var, smiles_var])

    metas = np.empty((total_count, 2), dtype=object)
    for index, molecule in enumerate(molecules):
        metas[index, 0] = str(molecule.name or "")
        metas[index, 1] = str(_molecule_smiles([molecule])[0] if molecules else "")

    x_out = np.column_stack([x_col, y_col, selected_col, rank_col]).astype(float, copy=False)
    return safe_table_from_numpy(
        domain,
        X=x_out,
        metas=metas,
        name="Diversity Annotated Data",
    )


def _mol_pixmap(smiles: str, size: int = 320) -> QPixmap | None:
    clean = str(smiles or "").strip()
    if not clean:
        return None
    mol = safe_mol_from_smiles(clean, sanitize=True, remove_hs=True).mol
    if mol is None:
        return None
    png = mol_depict.render_mol_png(
        mol,
        size=size,
        suppress_isolated_metal_radicals=True,
    )
    pixmap = QPixmap()
    pixmap.loadFromData(png)
    return pixmap


class OWDiversityPicker(OWWidget):
    name = "Diversity Picker"
    description = "Select a diverse subset of compounds using MaxMin, sphere exclusion, or Butina clustering."
    icon = "icons/analysis/owdiversitypickerwidget.svg"
    priority = 134
    want_main_area = True

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output("Annotated Data", Table)
        remainder_data = Output("Remainder Data", Table)
        inspected_data = Output("Inspected Data", Table)
        inspected_molecules = Output("Inspected Molecules", list, auto_summary=False)
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
        self.data: Table | None = None
        self.molecules: list[ChemMol] = []
        self._last_result: DiversitySelectionResult | None = None
        self._last_annotated: Table | None = None
        self._all_points_item: pg.ScatterPlotItem | None = None
        self._selected_points_item: pg.ScatterPlotItem | None = None
        self._inspection_points_item: pg.ScatterPlotItem | None = None
        self._inspected_indices: list[int] = []
        self._syncing_inspection_list = False

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

        self._summary_browser = QTextBrowser()
        self._summary_browser.setOpenExternalLinks(False)
        self._plot_widget = _styled_plot("Chemical Space Projection")

        tabs = QTabWidget()
        self._tabs = tabs
        tabs.addTab(self._summary_browser, "Summary")

        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        plot_layout.addWidget(self._plot_widget, 1)
        self._plot_legend = QLabel(
            "Circles = all valid compounds, stars = picker-selected compounds, outlined circles = inspected points. "
            "Click to inspect; Ctrl-click adds or removes points."
        )
        self._plot_legend.setWordWrap(True)
        plot_layout.addWidget(self._plot_legend)
        self._hover_label = QLabel("Hover over a point to preview the compound; click to inspect it.")
        self._hover_label.setWordWrap(True)
        self._hover_label.setStyleSheet("color:#475467;")
        plot_layout.addWidget(self._hover_label)
        tabs.addTab(plot_tab, "Projection")

        inspection_tab = QWidget()
        inspection_layout = QHBoxLayout(inspection_tab)
        inspection_list_panel = QWidget()
        inspection_list_layout = QVBoxLayout(inspection_list_panel)
        inspection_list_layout.setContentsMargins(0, 0, 0, 0)
        inspection_list_layout.setSpacing(8)
        self._inspection_summary_label = QLabel("No compounds are currently inspected.")
        self._inspection_summary_label.setWordWrap(True)
        self._inspection_summary_label.setStyleSheet("color:#475467;")
        inspection_list_layout.addWidget(self._inspection_summary_label)
        self._inspect_selected_button = QPushButton("Inspect picked subset")
        self._inspect_selected_button.clicked.connect(self._inspect_picker_subset)
        inspection_list_layout.addWidget(self._inspect_selected_button)
        self._clear_inspection_button = QPushButton("Clear inspection")
        self._clear_inspection_button.clicked.connect(self._clear_inspection_selection)
        inspection_list_layout.addWidget(self._clear_inspection_button)
        self._inspection_list = QListWidget()
        self._inspection_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._inspection_list.currentItemChanged.connect(self._on_inspection_item_changed)
        self._inspection_list.itemSelectionChanged.connect(self._on_inspection_selection_changed)
        self._inspection_list.setMinimumWidth(260)
        inspection_list_layout.addWidget(self._inspection_list, 1)
        inspection_layout.addWidget(inspection_list_panel, 0)

        inspection_detail = QWidget()
        inspection_detail_layout = QVBoxLayout(inspection_detail)
        self._structure_label = QLabel("Click points in the projection to inspect structures.")
        self._structure_label.setAlignment(Qt.AlignCenter)
        self._structure_label.setMinimumHeight(340)
        self._structure_label.setStyleSheet(
            "border:1px solid #D0D5DD; border-radius:8px; background:#FFFFFF; color:#475467; padding:10px;"
        )
        self._inspection_browser = QTextBrowser()
        self._inspection_browser.setOpenExternalLinks(False)
        inspection_detail_layout.addWidget(self._structure_label, 1)
        inspection_detail_layout.addWidget(self._inspection_browser, 1)
        inspection_layout.addWidget(inspection_detail, 1)
        self._inspection_tab_index = tabs.addTab(inspection_tab, "Inspection")

        self.mainArea.layout().addWidget(tabs)

        self._update_smiles_controls()
        self._update_method_controls()

    @Inputs.data
    def set_data(self, data: Table | None) -> None:
        self.data = data
        self._populate_smiles_combo()
        self._set_status(self._input_summary())
        self._maybe_autorun()

    @Inputs.molecules
    def set_molecules(self, molecules: list | None) -> None:
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
                names = [var.name for var in smiles_vars]
                if self.smiles_var_name and self.smiles_var_name in names:
                    self.smiles_combo.setCurrentText(self.smiles_var_name)
                else:
                    self.smiles_var_name = smiles_vars[0].name
                    self.smiles_combo.setCurrentText(self.smiles_var_name)
        finally:
            self.smiles_combo.blockSignals(False)
        self._update_smiles_controls()

    def _update_smiles_controls(self) -> None:
        self.smiles_combo.setEnabled(self.data is not None)

    def _update_method_controls(self) -> None:
        method = self._METHODS[self.method_idx][1]
        self.seed_idx_spin.setEnabled(method == "maxmin")
        self.sphere_radius_spin.setEnabled(method == "sphere_exclusion")
        self.butina_threshold_spin.setEnabled(method == "butina")

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

    def _inspect_picker_subset(self) -> None:
        if self._last_result is None:
            self._publish_inspection([])
            return
        self._publish_inspection(list(self._last_result.selected_indices))

    def _clear_inspection_selection(self) -> None:
        self._publish_inspection([])

    def _input_summary(self) -> str:
        table_rows = 0 if self.data is None else len(self.data)
        return f"Input: Table rows={table_rows}, Molecules={len(self.molecules)}"

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and (self.data is not None or self.molecules):
            self.commit()

    def _input_smiles(self) -> list[str]:
        if self.data is not None:
            return _table_smiles(self.data, self.smiles_var_name)
        return _molecule_smiles(self.molecules)

    @staticmethod
    def _subset_table(data: Table | None, indices: list[int]) -> Table | None:
        if data is None:
            return None
        return data[indices] if indices else data[:0]

    @staticmethod
    def _subset_molecules(molecules: list[ChemMol], indices: list[int]) -> list[ChemMol]:
        return [molecules[idx] for idx in indices if 0 <= idx < len(molecules)]

    def _inspected_molecules(self, row_indices: list[int]) -> list[ChemMol]:
        if not row_indices:
            return []

        aligned_molecules = (
            self.molecules
            if len(self.molecules) == (len(self.data) if self.data is not None else len(self.molecules))
            else []
        )
        if aligned_molecules:
            return self._subset_molecules(aligned_molecules, row_indices)

        inspected: list[ChemMol] = []
        for row_index in row_indices:
            smiles = self._row_smiles(int(row_index))
            parsed = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True)
            if parsed.mol is None:
                continue
            chem_mol = ChemMol(mol=parsed.mol, name=self._row_label(int(row_index)))
            chem_mol.set_prop("SMILES", smiles)
            inspected.append(chem_mol)
        return inspected

    def _clear_visuals(self) -> None:
        self._summary_browser.clear()
        self._plot_widget.clear()
        self._all_points_item = None
        self._selected_points_item = None
        self._inspection_points_item = None
        self._last_annotated = None
        self._inspected_indices = []
        self._inspection_list.clear()
        self._inspection_summary_label.setText("No compounds are currently inspected.")
        self._inspection_browser.clear()
        self._structure_label.setPixmap(QPixmap())
        self._structure_label.setText("Click points in the projection to inspect structures.")
        self._hover_label.setText("Hover over a point to preview the compound; click to inspect it.")

    def _row_smiles(self, row_index: int) -> str:
        if self.data is not None and 0 <= int(row_index) < len(self.data):
            try:
                value = self.data[row_index, self.smiles_var_name]
            except (IndexError, KeyError, TypeError, ValueError):
                value = ""
            return "" if value is None else str(value).strip()
        if 0 <= int(row_index) < len(self.molecules):
            return _molecule_smiles([self.molecules[int(row_index)]])[0]
        return ""

    def _row_label(self, row_index: int) -> str:
        if self.data is not None and 0 <= int(row_index) < len(self.data):
            label_var_name = _find_label_var_name(self.data, exclude_name=self.smiles_var_name)
            if label_var_name:
                try:
                    value = self.data[row_index, label_var_name]
                except (IndexError, KeyError, TypeError, ValueError):
                    value = ""
                if value is not None and str(value).strip():
                    return str(value).strip()
        if 0 <= int(row_index) < len(self.molecules):
            name = str(self.molecules[int(row_index)].name or "").strip()
            if name:
                return name
        return f"Compound {int(row_index) + 1}"

    def _row_detail_html(self, row_index: int) -> str:
        label = escape(self._row_label(row_index))
        smiles = escape(self._row_smiles(row_index) or "N/A")
        selected_flag = "Yes" if self._last_result and int(row_index) in set(self._last_result.selected_indices) else "No"
        rank_value = ""
        x_value = float("nan")
        y_value = float("nan")
        if self._last_result is not None and 0 <= int(row_index) < len(self._last_result.selection_ranks):
            rank = self._last_result.selection_ranks[int(row_index)]
            rank_value = "" if rank is None else str(rank)
            coordinates = np.asarray(self._last_result.coordinates, dtype=float)
            if coordinates.ndim == 2 and 0 <= int(row_index) < coordinates.shape[0]:
                x_value = float(coordinates[int(row_index), 0])
                y_value = float(coordinates[int(row_index), 1]) if coordinates.shape[1] > 1 else 0.0
        coord_html = (
            f"{x_value:.4f}, {y_value:.4f}"
            if np.isfinite(x_value) and np.isfinite(y_value)
            else "Unavailable"
        )
        rank_html = escape(rank_value or "Not picked")
        return (
            "<html><body style='font-family:sans-serif;'>"
            f"<h3>{label}</h3>"
            "<ul>"
            f"<li>Row: {int(row_index) + 1}</li>"
            f"<li>SMILES: {smiles}</li>"
            f"<li>Picker selected: {escape(selected_flag)}</li>"
            f"<li>Diversity rank: {rank_html}</li>"
            f"<li>Chemical-space coordinates: {escape(coord_html)}</li>"
            "</ul>"
            "</body></html>"
        )

    def _hover_text(self, row_index: int) -> str:
        label = self._row_label(row_index)
        selected_set = set(self._last_result.selected_indices) if self._last_result is not None else set()
        selected_text = "picked" if int(row_index) in selected_set else "not picked"
        rank_text = "no rank"
        x_value = float("nan")
        y_value = float("nan")
        if self._last_result is not None and 0 <= int(row_index) < len(self._last_result.selection_ranks):
            rank = self._last_result.selection_ranks[int(row_index)]
            if rank is not None:
                rank_text = f"rank #{int(rank)}"
            coordinates = np.asarray(self._last_result.coordinates, dtype=float)
            if coordinates.ndim == 2 and 0 <= int(row_index) < coordinates.shape[0]:
                x_value = float(coordinates[int(row_index), 0])
                y_value = float(coordinates[int(row_index), 1]) if coordinates.shape[1] > 1 else 0.0
        coord_text = (
            f"({x_value:.3f}, {y_value:.3f})"
            if np.isfinite(x_value) and np.isfinite(y_value)
            else "(coordinates unavailable)"
        )
        return f"Hover: {label}  [{selected_text}, {rank_text}]  {coord_text}"

    def _on_points_hovered(self, _item, points, _event) -> None:
        try:
            has_points = points is not None and len(points) > 0
        except TypeError:
            has_points = points is not None
        if has_points:
            row_index = points[0].data()
            if row_index is not None:
                self._hover_label.setText(self._hover_text(int(row_index)))
                return
        self._hover_label.setText("Hover over a point to preview the compound; click to inspect it.")

    def _update_structure_preview(self, row_index: int | None) -> None:
        if row_index is None:
            self._structure_label.setPixmap(QPixmap())
            self._structure_label.setText("Click points in the projection to inspect structures.")
            self._inspection_browser.clear()
            return

        smiles = self._row_smiles(int(row_index))
        pixmap = _mol_pixmap(smiles, size=320)
        if pixmap is None or pixmap.isNull():
            self._structure_label.setPixmap(QPixmap())
            self._structure_label.setText("No structure depiction available for the current selection.")
        else:
            self._structure_label.setText("")
            self._structure_label.setPixmap(
                pixmap.scaled(320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        self._inspection_browser.setHtml(self._row_detail_html(int(row_index)))

    def _update_inspection_list(self, row_indices: list[int]) -> None:
        self._syncing_inspection_list = True
        self._inspection_list.blockSignals(True)
        try:
            self._inspection_list.clear()
            selected_set = set(self._last_result.selected_indices) if self._last_result is not None else set()
            selection_ranks = self._last_result.selection_ranks if self._last_result is not None else []
            first_item: QListWidgetItem | None = None
            for row_index in row_indices:
                label = self._row_label(row_index)
                text = f"{int(row_index) + 1}. {label}"
                if row_index in selected_set:
                    rank = selection_ranks[int(row_index)] if 0 <= int(row_index) < len(selection_ranks) else None
                    rank_text = "" if rank is None else f" #{int(rank)}"
                    text += f" [picked{rank_text}]"
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, int(row_index))
                self._inspection_list.addItem(item)
                item.setSelected(True)
                if first_item is None:
                    first_item = item
            if first_item is not None:
                self._inspection_list.setCurrentItem(first_item)
        finally:
            self._inspection_list.blockSignals(False)
            self._syncing_inspection_list = False

        selected_set = set(self._last_result.selected_indices) if self._last_result is not None else set()
        n_picked = sum(1 for row_index in row_indices if row_index in selected_set)
        if row_indices:
            self._inspection_summary_label.setText(
                f"Inspecting {len(row_indices)} compound(s); {n_picked} from the diversity-picked subset."
            )
        else:
            self._inspection_summary_label.setText("No compounds are currently inspected.")

        if row_indices:
            self._update_structure_preview(int(row_indices[0]))
        else:
            self._update_structure_preview(None)

    def _on_inspection_item_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            self._update_structure_preview(None)
            return
        row_index = current.data(Qt.UserRole)
        self._update_structure_preview(None if row_index is None else int(row_index))

    def _selected_inspection_rows_from_list(self) -> list[int]:
        return sorted(
            {
                int(item.data(Qt.UserRole))
                for item in self._inspection_list.selectedItems()
                if item.data(Qt.UserRole) is not None
            }
        )

    def _send_inspection_outputs(self, row_indices: list[int]) -> None:
        if self._last_annotated is None or not row_indices:
            self.Outputs.inspected_data.send(None)
            self.Outputs.inspected_molecules.send([])
            return
        self.Outputs.inspected_data.send(self._subset_table(self._last_annotated, row_indices))
        self.Outputs.inspected_molecules.send(self._inspected_molecules(row_indices))

    def _on_inspection_selection_changed(self) -> None:
        if self._syncing_inspection_list:
            return
        row_indices = self._selected_inspection_rows_from_list()
        self._inspected_indices = row_indices
        self._update_inspection_overlay(row_indices)
        self._send_inspection_outputs(row_indices)

    def _update_plot(self, result: DiversitySelectionResult) -> None:
        self._plot_widget.clear()
        self._all_points_item = None
        self._selected_points_item = None
        self._inspection_points_item = None

        coordinates = np.asarray(result.coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[0] == 0:
            return
        if coordinates.shape[1] == 1:
            coordinates = np.column_stack([coordinates[:, 0], np.zeros(coordinates.shape[0], dtype=float)])

        valid_mask = np.isfinite(coordinates[:, 0]) & np.isfinite(coordinates[:, 1])
        if not np.any(valid_mask):
            return

        valid_indices = np.flatnonzero(valid_mask)
        x_values = coordinates[valid_indices, 0]
        y_values = coordinates[valid_indices, 1]
        spots = [
            {
                "pos": (float(coordinates[row_index, 0]), float(coordinates[row_index, 1])),
                "data": int(row_index),
                "brush": pg.mkBrush(59, 130, 246, 180),
                "pen": pg.mkPen(None),
                "size": 8,
            }
            for row_index in valid_indices
        ]
        self._all_points_item = pg.ScatterPlotItem(
            spots=spots,
            symbol="o",
            hoverable=True,
            hoverSize=11,
            hoverBrush=pg.mkBrush(37, 99, 235, 220),
        )
        self._all_points_item.sigClicked.connect(self._on_points_clicked)
        self._all_points_item.sigHovered.connect(self._on_points_hovered)
        self._plot_widget.addItem(self._all_points_item)

        selected_rows = [index for index in result.selected_indices if 0 <= index < coordinates.shape[0]]
        if selected_rows:
            selected_coords = coordinates[selected_rows, :]
            finite_selected = np.isfinite(selected_coords[:, 0]) & np.isfinite(selected_coords[:, 1])
            selected_spots = [
                {
                    "pos": (float(selected_coords[idx, 0]), float(selected_coords[idx, 1])),
                    "data": int(selected_rows[idx]),
                    "brush": pg.mkBrush(251, 146, 60, 230),
                    "pen": pg.mkPen("#C2410C", width=1.5),
                    "size": 16,
                }
                for idx in np.flatnonzero(finite_selected)
            ]
            self._selected_points_item = pg.ScatterPlotItem(
                spots=selected_spots,
                symbol="star",
                hoverable=True,
                hoverSize=19,
                hoverBrush=pg.mkBrush(249, 115, 22, 255),
            )
            self._selected_points_item.sigClicked.connect(self._on_points_clicked)
            self._selected_points_item.sigHovered.connect(self._on_points_hovered)
            self._plot_widget.addItem(self._selected_points_item)

        self._inspection_points_item = pg.ScatterPlotItem(
            symbol="o",
            size=18,
            pen=pg.mkPen("#0F172A", width=2),
            brush=pg.mkBrush(15, 23, 42, 35),
        )
        self._plot_widget.addItem(self._inspection_points_item)

        self._plot_widget.setLabel("bottom", "chem_space_x")
        self._plot_widget.setLabel("left", "chem_space_y")
        _set_plot_range(self._plot_widget, x_values, y_values)
        self._update_inspection_overlay(self._inspected_indices)

    def _update_inspection_overlay(self, row_indices: list[int]) -> None:
        if self._inspection_points_item is None or self._last_result is None:
            return
        coordinates = np.asarray(self._last_result.coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[0] == 0:
            self._inspection_points_item.setData(x=[], y=[])
            return
        valid_rows = [
            int(row_index)
            for row_index in row_indices
            if 0 <= int(row_index) < coordinates.shape[0]
            and np.isfinite(coordinates[int(row_index), 0])
            and np.isfinite(coordinates[int(row_index), 1 if coordinates.shape[1] > 1 else 0])
        ]
        if not valid_rows:
            self._inspection_points_item.setData(x=[], y=[])
            return
        self._inspection_points_item.setData(
            x=[float(coordinates[row_index, 0]) for row_index in valid_rows],
            y=[
                float(coordinates[row_index, 1]) if coordinates.shape[1] > 1 else 0.0
                for row_index in valid_rows
            ],
        )

    def _publish_inspection(self, row_indices: list[int]) -> None:
        clean_indices = sorted(
            {
                int(row_index)
                for row_index in row_indices
                if self._last_result is not None and 0 <= int(row_index) < len(self._last_result.coordinates)
            }
        )
        self._inspected_indices = clean_indices
        self._update_inspection_overlay(clean_indices)
        self._update_inspection_list(clean_indices)
        if clean_indices:
            self._tabs.setCurrentIndex(self._inspection_tab_index)
        self._send_inspection_outputs(clean_indices)

    def _merged_inspection_selection(self, clicked_indices: list[int], modifiers: Qt.KeyboardModifiers) -> list[int]:
        clean_clicked = [int(row_index) for row_index in clicked_indices]
        if modifiers & (Qt.ControlModifier | Qt.MetaModifier):
            merged = list(self._inspected_indices)
            for row_index in clean_clicked:
                if row_index in merged:
                    merged.remove(row_index)
                else:
                    merged.append(row_index)
            return merged
        return clean_clicked

    def _on_points_clicked(self, _item, points, event) -> None:
        clicked_indices = [int(point.data()) for point in points if point.data() is not None]
        modifiers = event.modifiers() if hasattr(event, "modifiers") else Qt.NoModifier
        self._publish_inspection(self._merged_inspection_selection(clicked_indices, modifiers))

    def _annotated_table(self, result: DiversitySelectionResult) -> Table | None:
        if self.data is not None:
            return _annotated_table_from_data(self.data, result)
        if self.molecules:
            return _annotated_table_from_molecules(self.molecules, result)
        return None

    def commit(self) -> None:
        clear_widget_messages(self)
        if self.data is None and not self.molecules:
            self._clear_visuals()
            self._set_status(format_no_input_status())
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
            self.Outputs.remainder_data.send(None)
            self.Outputs.inspected_data.send(None)
            self.Outputs.inspected_molecules.send([])
            self.Outputs.selected_molecules.send([])
            self.Outputs.remainder_molecules.send([])
            return

        try:
            smiles = self._input_smiles()
        except ValueError as exc:
            self._clear_visuals()
            set_widget_error(self, str(exc))
            self._set_status(format_error_status(str(exc)))
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
            self.Outputs.remainder_data.send(None)
            self.Outputs.inspected_data.send(None)
            self.Outputs.inspected_molecules.send([])
            self.Outputs.selected_molecules.send([])
            self.Outputs.remainder_molecules.send([])
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
        self._last_result = result
        self._send_outputs(result, len(smiles))

    def _send_outputs(self, result: DiversitySelectionResult, total_count: int) -> None:
        selected_indices = list(result.selected_indices)
        remainder_indices = [idx for idx in range(total_count) if idx not in set(selected_indices)]

        annotated = self._annotated_table(result)
        self._last_annotated = annotated
        selected_data = self._subset_table(annotated, selected_indices)
        remainder_data = self._subset_table(annotated, remainder_indices)

        aligned_molecules = self.molecules if len(self.molecules) == total_count else []
        selected_molecules = self._subset_molecules(aligned_molecules, selected_indices)
        remainder_molecules = self._subset_molecules(aligned_molecules, remainder_indices)

        self.Outputs.selected_data.send(selected_data)
        self.Outputs.annotated_data.send(annotated)
        self.Outputs.remainder_data.send(remainder_data)
        self.Outputs.selected_molecules.send(selected_molecules)
        self.Outputs.remainder_molecules.send(remainder_molecules)

        self._summary_browser.setHtml(_summary_html(result))
        self._update_plot(result)
        self._publish_inspection([])

        input_metrics = result.metrics_input
        selected_metrics = result.metrics_selected
        self._set_status(
            format_done_status(
                f"selected={len(selected_indices)}/{input_metrics.n_compounds}",
                f"invalid skipped={len(result.failed_indices)}",
                f"PCA diversity {input_metrics.diversity_score:.4f}->{selected_metrics.diversity_score:.4f}",
                prefix="Selected",
            )
        )


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWDiversityPicker).run()
