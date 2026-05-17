from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.from_orange import table_to_chemmols_with_report
from chem_inf_widgets.chemcore.services.orange_table_utils import safe_table_from_numpy
from chem_inf_widgets.chemcore.services.rdkit_descriptor_service import (
    RDKIT_DESCRIPTOR_CATEGORIES,
    RDKIT_DESCRIPTOR_PRESETS,
    RdkitDescriptorPreset,
    RdkitDescriptorService,
)
from chem_inf_widgets.widgets.ui_helpers import (
    format_done_status,
    format_no_input_status,
    format_skip_warning,
    format_table_report,
    set_widget_warning,
)


def _find_smiles_var(data: Table) -> Optional[StringVariable]:
    if data is None:
        return None
    all_vars = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    for var in all_vars:
        if var.name.strip().lower() == "smiles" and isinstance(var, StringVariable):
            return var
    for var in all_vars:
        if "smiles" in var.name.strip().lower() and isinstance(var, StringVariable):
            return var
    return None


def _unique_preserve_order(names: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _make_unique_name(name: str, taken: set[str], prefix: str = "rdkit_") -> str:
    candidate = name
    if candidate in taken:
        candidate = f"{prefix}{name}"
    if candidate not in taken:
        taken.add(candidate)
        return candidate
    i = 2
    while True:
        candidate_i = f"{candidate}_{i}"
        if candidate_i not in taken:
            taken.add(candidate_i)
            return candidate_i
        i += 1


def _list_widget_texts(widget: QListWidget) -> List[str]:
    return [widget.item(i).text() for i in range(widget.count())]


class OWRdkitDescriptors(OWWidget):
    name = "RDKit Descriptors"
    description = "Compute categorized RDKit molecular descriptors from SMILES tables or ChemMol objects."
    icon = "icons/descriptors/owmoldescriptorwidget.png"
    priority = 131
    keywords = ["rdkit", "descriptor", "descriptors", "qsar", "smiles", "physchem"]
    want_main_area = False
    resizing_enabled = True

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        data = Output("Data", Table)
        molecules = Output("Molecules", list, auto_summary=False)

    preset_key: str = Setting("recommended_qsar")
    selected_descriptors: List[str] = Setting([])
    checked_categories: List[str] = Setting([])
    write_to_molecules: bool = Setting(False)
    auto_run: bool = Setting(False)

    def __init__(self) -> None:
        super().__init__()
        self._data: Optional[Table] = None
        self._molecules: List[ChemMol] = []
        self._table_report = None
        self._service = RdkitDescriptorService()
        self._descriptor_category: Dict[str, str] = {d.name: d.category for d in self._service.list_descriptors()}
        self._presets = RDKIT_DESCRIPTOR_PRESETS
        self._preset_map: Dict[str, RdkitDescriptorPreset] = {preset.key: preset for preset in self._presets}
        self._build_ui()
        self._bootstrap_defaults()
        self._refresh_available_descriptors()
        self._set_status("Waiting for input…")

    def _build_ui(self) -> None:
        root = self.controlArea

        self.lbl = QLabel("Waiting for input…")
        self.lbl.setWordWrap(True)
        root.layout().addWidget(self.lbl)

        preset_box = QGroupBox("Preset")
        preset_layout = QVBoxLayout(preset_box)
        self.combo_preset = QComboBox()
        for preset in self._presets:
            self.combo_preset.addItem(preset.label, preset.key)
        self.combo_preset.setCurrentIndex(max(0, self.combo_preset.findData(self.preset_key)))
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.combo_preset)
        self.lbl_preset_desc = QLabel("")
        self.lbl_preset_desc.setWordWrap(True)
        self.lbl_preset_desc.setStyleSheet("color: #666; font-size: 11px;")
        preset_layout.addWidget(self.lbl_preset_desc)
        root.layout().addWidget(preset_box)

        self.category_box = QGroupBox("Descriptor categories")
        category_layout = QVBoxLayout(self.category_box)
        self.list_categories = QListWidget()
        self.list_categories.setSelectionMode(QAbstractItemView.NoSelection)
        self.list_categories.setFixedHeight(125)
        self.list_categories.itemChanged.connect(self._on_categories_changed)
        category_layout.addWidget(self.list_categories)
        root.layout().addWidget(self.category_box)

        selector = QGroupBox("Descriptor selection")
        selector_layout = QHBoxLayout(selector)
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.available_list.setMinimumHeight(145)
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.selected_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.selected_list.setMinimumHeight(145)
        self.selected_list.model().rowsMoved.connect(self._on_order_changed)

        button_col = QVBoxLayout()
        self.btn_add = QPushButton("Add →")
        self.btn_remove = QPushButton("← Remove")
        self.btn_add_all = QPushButton("Add all")
        self.btn_clear = QPushButton("Clear")
        self.btn_refresh = QPushButton("↺ Refresh")
        self.btn_add.clicked.connect(self._on_add)
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_add_all.clicked.connect(self._on_add_all)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_refresh.clicked.connect(self._refresh_available_descriptors)
        button_col.addStretch(1)
        for btn in (self.btn_refresh, self.btn_add, self.btn_remove, self.btn_add_all, self.btn_clear):
            button_col.addWidget(btn)
        button_col.addStretch(1)

        selector_layout.addWidget(self._wrap("Available", self.available_list), 1)
        selector_layout.addLayout(button_col)
        selector_layout.addWidget(self._wrap("Selected", self.selected_list), 1)
        root.layout().addWidget(selector)

        self.cb_write_mols = QCheckBox("Write descriptor values into Molecules")
        self.cb_write_mols.setChecked(bool(self.write_to_molecules))
        self.cb_write_mols.toggled.connect(self._on_options_changed)
        self.cb_auto = QCheckBox("Auto-run")
        self.cb_auto.setChecked(bool(self.auto_run))
        self.cb_auto.toggled.connect(self._on_auto_run_toggled)
        self.btn_compute = QPushButton("Compute")
        self.btn_compute.clicked.connect(self.commit)
        root.layout().addWidget(self.cb_write_mols)
        root.layout().addWidget(self.cb_auto)
        root.layout().addWidget(self.btn_compute)

    @staticmethod
    def _wrap(title: str, widget) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.addWidget(widget)
        return box

    def _set_status(self, text: str) -> None:
        self.lbl.setText(text)

    def _current_preset(self) -> RdkitDescriptorPreset:
        return self._preset_map.get(self.preset_key, self._preset_map["recommended_qsar"])

    def _bootstrap_defaults(self) -> None:
        self._render_categories()
        if not self.selected_descriptors:
            preset = self._current_preset()
            if preset.descriptors:
                self.selected_descriptors = [name for name in preset.descriptors if name in self._descriptor_category]
            else:
                self.selected_descriptors = self._service.descriptor_names_for_categories(preset.categories)
        self._render_selected_list()
        self._sync_preset_controls()

    def _render_categories(self) -> None:
        self.list_categories.blockSignals(True)
        self.list_categories.clear()
        active = set(self.checked_categories or self._current_preset().categories)
        for category in list(RDKIT_DESCRIPTOR_CATEGORIES.keys()) + ["Other RDKit descriptors"]:
            item = QListWidgetItem(category)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if category in active else Qt.Unchecked)
            self.list_categories.addItem(item)
        self.list_categories.blockSignals(False)
        self.checked_categories = [
            self.list_categories.item(i).text()
            for i in range(self.list_categories.count())
            if self.list_categories.item(i).checkState() == Qt.Checked
        ]

    def _render_selected_list(self) -> None:
        self.selected_list.clear()
        for name in _unique_preserve_order(self.selected_descriptors):
            if name in self._descriptor_category:
                self.selected_list.addItem(QListWidgetItem(name))
        self.selected_descriptors = _list_widget_texts(self.selected_list)

    def _sync_preset_controls(self) -> None:
        preset = self._current_preset()
        self.lbl_preset_desc.setText(preset.description)
        self.category_box.setVisible(preset.key == "custom")

    def _on_preset_changed(self) -> None:
        self.preset_key = self.combo_preset.currentData() or "recommended_qsar"
        preset = self._current_preset()
        self.checked_categories = list(preset.categories)
        self._render_categories()
        if preset.descriptors:
            self.selected_descriptors = [name for name in preset.descriptors if name in self._descriptor_category]
        elif preset.key == "all":
            self.selected_descriptors = sorted(self._descriptor_category)
        else:
            self.selected_descriptors = self._service.descriptor_names_for_categories(preset.categories)
        self._render_selected_list()
        self._refresh_available_descriptors()
        self._sync_preset_controls()
        self._maybe_autorun()

    def _on_categories_changed(self) -> None:
        self.checked_categories = [
            self.list_categories.item(i).text()
            for i in range(self.list_categories.count())
            if self.list_categories.item(i).checkState() == Qt.Checked
        ]
        self._refresh_available_descriptors()

    def _active_categories(self) -> List[str]:
        if self._current_preset().key != "custom":
            return list(self._current_preset().categories)
        return list(self.checked_categories or [])

    def _refresh_available_descriptors(self) -> None:
        selected = set(self._read_selected_names())
        active = set(self._active_categories())
        self.available_list.clear()
        for name in sorted(self._descriptor_category):
            if name in selected:
                continue
            if self._current_preset().key == "all" or self._descriptor_category[name] in active:
                self.available_list.addItem(QListWidgetItem(name))

    def _read_selected_names(self) -> List[str]:
        return _list_widget_texts(self.selected_list)

    def _on_add(self) -> None:
        for item in list(self.available_list.selectedItems()):
            row = self.available_list.row(item)
            self.available_list.takeItem(row)
            self.selected_list.addItem(QListWidgetItem(item.text()))
        self.selected_descriptors = self._read_selected_names()
        self._maybe_autorun()

    def _on_remove(self) -> None:
        for item in list(self.selected_list.selectedItems()):
            row = self.selected_list.row(item)
            self.selected_list.takeItem(row)
        self.selected_descriptors = self._read_selected_names()
        self._refresh_available_descriptors()
        self._maybe_autorun()

    def _on_add_all(self) -> None:
        while self.available_list.count() > 0:
            item = self.available_list.takeItem(0)
            self.selected_list.addItem(QListWidgetItem(item.text()))
        self.selected_descriptors = self._read_selected_names()
        self._maybe_autorun()

    def _on_clear(self) -> None:
        self.selected_list.clear()
        self.selected_descriptors = []
        self._refresh_available_descriptors()
        self._maybe_autorun()

    def _on_order_changed(self, *_args) -> None:
        self.selected_descriptors = self._read_selected_names()
        self._maybe_autorun()

    def _on_options_changed(self) -> None:
        self.write_to_molecules = bool(self.cb_write_mols.isChecked())
        self._maybe_autorun()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        self.auto_run = bool(checked)
        self._maybe_autorun()

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._data = data
        self._table_report = None
        if data is not None:
            try:
                _mols, self._table_report = table_to_chemmols_with_report(data)
            except Exception:
                self._table_report = None
        self._set_status(self._input_summary())
        self._maybe_autorun()

    @Inputs.molecules
    def set_molecules(self, molecules: Optional[list]) -> None:
        self._molecules = [m for m in (molecules or []) if isinstance(m, ChemMol)]
        self._set_status(self._input_summary())
        self._maybe_autorun()

    def _input_summary(self) -> str:
        n_tab = 0 if self._data is None else len(self._data)
        n_mol = len(self._molecules)
        parts = []
        if self._table_report is not None:
            parts.append(format_table_report(self._table_report, prefix="Table", valid_label="valid SMILES"))
        else:
            parts.append(f"Table rows={n_tab}")
        parts.append(f"Molecules={n_mol}")
        parts.append(f"Preset={self._current_preset().label}")
        return "Input: " + " | ".join(parts)

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and (self._data is not None or self._molecules):
            self.commit()

    def commit(self) -> None:
        selected = _unique_preserve_order(self._read_selected_names())
        if not selected:
            self.Outputs.data.send(self._data)
            self.Outputs.molecules.send(self._molecules)
            self._set_status("No selected descriptors.")
            return
        if (self._data is None or len(self._data) == 0) and not self._molecules:
            self.Outputs.data.send(None)
            self.Outputs.molecules.send([])
            self._set_status(format_no_input_status())
            return

        base_table = self._data
        rdkit_mols = []
        valid_idx = []
        n_total = 0
        if base_table is not None and len(base_table) > 0:
            n_total = len(base_table)
            smiles_var = _find_smiles_var(base_table)
            if smiles_var is not None:
                smiles = [str(row[smiles_var]) if row[smiles_var] is not None else "" for row in base_table]
                rdkit_mols, valid_idx = self._service.smiles_to_mols(smiles)
        if (base_table is None or n_total == 0 or not valid_idx) and self._molecules:
            base_table = None
            n_total = len(self._molecules)
            rdkit_mols, valid_idx = self._service.chemmols_to_mols(self._molecules)

        valid_mols = [mol for mol in rdkit_mols if mol is not None]
        if not valid_mols:
            self.Outputs.data.send(base_table)
            self.Outputs.molecules.send(self._molecules)
            self._set_status("No valid molecules for RDKit descriptor calculation.")
            return

        df_valid = self._service.compute(valid_mols, selected)
        df_full = self._service.df_to_full_length(df_valid, valid_idx, n_total)
        out_table = self._attach_df_to_table(base_table, df_full, selected)
        out_mols = list(self._molecules)
        if self.write_to_molecules and out_mols and len(out_mols) == len(df_full):
            for row_idx, chem_mol in enumerate(out_mols):
                if chem_mol is None:
                    continue
                for col in selected:
                    if col not in df_full.columns:
                        continue
                    value = self._service.numeric_or_none(df_full.iloc[row_idx][col])
                    if value is not None:
                        chem_mol.set_prop(col, value)

        invalid = n_total - len(valid_idx)
        set_widget_warning(self, format_skip_warning(invalid, subject="input rows", action="were skipped during RDKit descriptor computation"))
        self._set_status(format_done_status(f"rows={n_total}", f"descriptors={len(selected)}", f"invalid={invalid}"))
        self.Outputs.data.send(out_table)
        self.Outputs.molecules.send(out_mols)

    def _attach_df_to_table(self, base: Optional[Table], df_full, selected: Sequence[str]) -> Table:
        selected = [name for name in _unique_preserve_order(selected) if name in df_full.columns]
        if not selected:
            return base if base is not None else safe_table_from_numpy(Domain([]), X=np.empty((len(df_full), 0)))
        x_desc = df_full[selected].to_numpy(dtype=float, copy=False)
        if base is None:
            taken: set[str] = set()
            desc_vars = [ContinuousVariable(_make_unique_name(name, taken)) for name in selected]
            return safe_table_from_numpy(Domain(desc_vars), X=x_desc, metas=np.zeros((len(x_desc), 0), dtype=object))

        dom0 = base.domain
        taken = {v.name for v in list(dom0.attributes) + list(dom0.class_vars) + list(dom0.metas)}
        desc_vars = [ContinuousVariable(_make_unique_name(name, taken)) for name in selected]
        domain_out = Domain(list(dom0.attributes) + desc_vars, dom0.class_vars, metas=dom0.metas)
        x0 = base.X
        x_out = np.hstack([x0, x_desc]) if x0.size else x_desc
        return safe_table_from_numpy(domain_out, X=x_out, Y=base.Y, metas=base.metas, name=getattr(base, "name", "Data"))
