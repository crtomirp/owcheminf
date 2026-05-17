from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from AnyQt.QtCore import Qt, pyqtSlot
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from Orange.data import Domain, ContinuousVariable, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from rdkit import Chem

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.from_orange import (
    TableMolConversionReport,
    table_to_chemmols_with_report,
)
from chem_inf_widgets.chemcore.services.mordred_descriptor_service import (
    MORDRED_AVAILABLE,
    MordredComputeConfig,
    MordredDescriptorService,
)
from chem_inf_widgets.chemcore.services.orange_table_utils import safe_table_from_numpy
from chem_inf_widgets.widgets.ui_helpers import (
    format_done_status,
    format_failed_status,
    format_no_input_status,
    format_skip_warning,
    format_table_report,
    format_waiting_status,
    set_widget_error,
    set_widget_warning,
)

# ------------------------------ Categories / Groups ------------------------------

CATEGORIES: Dict[str, List[str]] = {
    "Constitutional": [
        "mordred.AtomCount",
        "mordred.BondCount",
        "mordred.Constitutional",
        "mordred.RingCount",
        "mordred.RotatableBond",
        "mordred.FragmentComplexity",
        "mordred.CarbonTypes",
    ],
    "Topological indices": [
        "mordred.Chi",
        "mordred.KappaShapeIndex",
        "mordred.ZagrebIndex",
        "mordred.BalabanJ",
        "mordred.TopologicalIndex",
        "mordred.PathCount",
        "mordred.WalkCount",
        "mordred.VertexAdjacencyInformation",
        "mordred.TopologicalCharge",
        "mordred.ABCIndex",
    ],
    "Matrices & autocorrelation": [
        "mordred.AdjacencyMatrix",
        "mordred.DetourMatrix",
        "mordred.DistanceMatrix",
        "mordred.MolecularDistanceEdge",
        "mordred.WienerIndex",
        "mordred.EccentricConnectivityIndex",
        "mordred.Autocorrelation",
    ],
    "3D / geometry (ignored if 2D mode)": [
        "mordred.MomentOfInertia",
        "mordred.GeometricalIndex",
        "mordred.McGowanVolume",
        "mordred.PBF",
        "mordred.VdwVolumeABC",
    ],
    "Physicochemical": [
        "mordred.Polarizability",
        "mordred.Weight",
        "mordred.GravitationalIndex",
        "mordred.InformationContent",
    ],
    "Electronic / QSAR": [
        "mordred.EState",
        "mordred.CPSA",
        "mordred.HydrogenBond",
        "mordred.LogS",
        "mordred.SLogP",
        "mordred.TopoPSA",
        "mordred.Lipinski",
        "mordred.BCUT",
    ],
    "Specialized": [
        "mordred.MoRSE",
        "mordred.MoeType",
        "mordred.ExtendedTopochemicalAtom",
        "mordred.Framework",
        "mordred.MolecularId",
        "mordred.BaryszMatrix",
        "mordred.AcidBase",
        "mordred.Aromatic",
        "mordred.BertzCT",
    ],
}

DEFAULT_GROUPS = {"mordred.Weight", "mordred.RotatableBond", "mordred.HydrogenBond", "mordred.SLogP"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MordredPreset:
    key: str
    label: str
    description: str
    categories: Tuple[str, ...]
    groups: Tuple[str, ...]


MORDRED_PRESETS: Tuple[MordredPreset, ...] = (
    MordredPreset(
        key="custom",
        label="Custom / manual module filters",
        description="Choose categories and module prefixes manually, then pick individual descriptors.",
        categories=tuple(CATEGORIES.keys()),
        groups=tuple(),
    ),
    MordredPreset(
        key="recommended_qsar",
        label="Recommended QSAR core",
        description="Balanced default subset around size, lipophilicity, H-bond, and rotatable-bond families.",
        categories=tuple(CATEGORIES.keys()),
        groups=tuple(sorted(DEFAULT_GROUPS)),
    ),
    MordredPreset(
        key="constitutional_counts",
        label="Descriptor family: constitutional and counts",
        description="Counts, atom/bond composition, constitutional fragments, and ring-centric families.",
        categories=("Constitutional",),
        groups=tuple(CATEGORIES["Constitutional"]),
    ),
    MordredPreset(
        key="topology_connectivity",
        label="Descriptor family: topology and connectivity",
        description="Topological indices, matrices, path and walk descriptors, and graph-derived connectivity families.",
        categories=("Topological indices", "Matrices & autocorrelation"),
        groups=tuple(CATEGORIES["Topological indices"] + CATEGORIES["Matrices & autocorrelation"]),
    ),
    MordredPreset(
        key="physchem_qsar",
        label="Descriptor family: physicochemical / QSAR",
        description="Lipophilicity, H-bond, EState, BCUT, TPSA, and other common QSAR-oriented families.",
        categories=("Physicochemical", "Electronic / QSAR"),
        groups=tuple(CATEGORIES["Physicochemical"] + CATEGORIES["Electronic / QSAR"]),
    ),
    MordredPreset(
        key="geometry_3d",
        label="Descriptor family: 3D / geometry",
        description="3D geometry and conformational families. Disable 'Ignore 3D' to expose these descriptors.",
        categories=("3D / geometry (ignored if 2D mode)",),
        groups=tuple(CATEGORIES["3D / geometry (ignored if 2D mode)"]),
    ),
    MordredPreset(
        key="specialized",
        label="Descriptor family: specialized",
        description="Specialized Mordred modules such as MoRSE, ETA, framework, aromaticity, and Barysz families.",
        categories=("Specialized",),
        groups=tuple(CATEGORIES["Specialized"]),
    ),
    MordredPreset(
        key="all_families",
        label="All available Mordred families",
        description="Expose the full Mordred catalog allowed by the current 2D/3D setting.",
        categories=tuple(CATEGORIES.keys()),
        groups=tuple(sorted({group for groups in CATEGORIES.values() for group in groups})),
    ),
)

MORDRED_PRESET_MAP: Dict[str, MordredPreset] = {preset.key: preset for preset in MORDRED_PRESETS}

# ------------------------------ Helpers ------------------------------


def _find_smiles_var(data: Table) -> Optional[StringVariable]:
    if data is None:
        return None
    all_vars = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    # exact match first
    for v in all_vars:
        if v.name.strip().lower() == "smiles" and isinstance(v, StringVariable):
            return v
    # substring match
    for v in all_vars:
        if "smiles" in v.name.strip().lower() and isinstance(v, StringVariable):
            return v
    return None


def _unique_preserve_order(names: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _make_unique_name(name: str, taken: set[str], prefix: str = "mordred_") -> str:
    candidate = name
    if candidate in taken:
        candidate = f"{prefix}{name}"
    if candidate not in taken:
        taken.add(candidate)
        return candidate

    i = 2
    while True:
        cand2 = f"{candidate}_{i}"
        if cand2 not in taken:
            taken.add(cand2)
            return cand2
        i += 1


def _list_widget_texts(list_widget: QListWidget) -> List[str]:
    return [list_widget.item(i).text() for i in range(list_widget.count())]


# ------------------------------ Widget ------------------------------


class OWMolDescriptor(OWWidget):
    name = "Mol Descriptors 2"
    description = "Compute Mordred descriptors from Table (SMILES) and/or Molecules (ChemMol)."
    icon = "icons/descriptors/owmoldescriptorwidget.png"
    priority = 132
    want_main_area = False

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        data = Output("Data", Table)
        molecules = Output("Molecules", list, auto_summary=False)

    # settings
    preset_key: str = Setting("recommended_qsar")
    selected_descriptors: List[str] = Setting([])
    checked_categories: List[str] = Setting([])
    checked_groups: List[str] = Setting([])
    ignore_3d: bool = Setting(True)
    nproc: int = Setting(1)  # safe default: single Mordred worker in Orange GUI
    write_to_molecules: bool = Setting(False)
    auto_run: bool = Setting(False)

    def __init__(self) -> None:
        super().__init__()
        self.executor = ThreadExecutor(self)

        self._data: Optional[Table] = None
        self._molecules: List[ChemMol] = []
        self._table_report: Optional[TableMolConversionReport] = None

        self._service: Optional[MordredDescriptorService] = None
        self._desc_map: Dict[str, str] = {}
        self._presets: Tuple[MordredPreset, ...] = MORDRED_PRESETS
        self._preset_map: Dict[str, MordredPreset] = MORDRED_PRESET_MAP

        # background task control
        self._task_id: int = 0
        self._future = None

        self._build_ui()
        if MORDRED_AVAILABLE:
            self._initialize_service()
            self._bootstrap_defaults()
            self._refresh_available_descriptors()
            self._set_status(format_waiting_status())
        else:
            self._set_mordred_unavailable_state()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        root = self.controlArea

        self.lbl = QLabel("Waiting for input…")
        self.lbl.setWordWrap(True)
        root.layout().addWidget(self.lbl)

        # preset row (no description label — wastes space)
        preset_box = QGroupBox("Preset")
        preset_form = QFormLayout(preset_box)
        preset_form.setContentsMargins(6, 4, 6, 4)
        self.combo_preset = QComboBox()
        for preset in self._presets:
            self.combo_preset.addItem(preset.label, preset.key)
        idx = max(0, self.combo_preset.findData(self.preset_key))
        self.combo_preset.setCurrentIndex(idx)
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        preset_form.addRow(self.combo_preset)
        self.lbl_preset_desc = QLabel("")
        self.lbl_preset_desc.setWordWrap(True)
        self.lbl_preset_desc.setStyleSheet("color: #666; font-size: 11px;")
        preset_form.addRow(self.lbl_preset_desc)
        root.layout().addWidget(preset_box)

        # compact options
        opt = QGroupBox("Options")
        opt_form = QFormLayout(opt)
        opt_form.setContentsMargins(6, 4, 6, 4)
        self.cb_ignore3d = QCheckBox("Ignore 3D (2D only)")
        self.cb_ignore3d.setChecked(bool(self.ignore_3d))
        self.cb_ignore3d.toggled.connect(self._on_options_changed)
        self.spin_nproc = QSpinBox()
        self.spin_nproc.setRange(1, 64)
        self.spin_nproc.setValue(max(1, int(self.nproc)))
        self.spin_nproc.setFixedWidth(52)
        self.spin_nproc.valueChanged.connect(self._on_options_changed)
        self.cb_write_mols = QCheckBox("Write to Molecules")
        self.cb_write_mols.setChecked(bool(self.write_to_molecules))
        self.cb_write_mols.toggled.connect(self._on_options_changed)
        opt_form.addRow(self.cb_ignore3d)
        opt_form.addRow("nproc (1=safe):", self.spin_nproc)
        opt_form.addRow(self.cb_write_mols)
        root.layout().addWidget(opt)

        # categories + groups — only visible in custom mode
        self._catalog_scope_box = QGroupBox("Catalog scope (custom only)")
        hv = QHBoxLayout(self._catalog_scope_box)
        hv.setContentsMargins(4, 4, 4, 4)
        hv.setSpacing(4)

        self.list_categories = QListWidget()
        self.list_categories.setSelectionMode(QAbstractItemView.NoSelection)
        self.list_categories.setFixedHeight(110)
        self.list_categories.itemChanged.connect(self._on_categories_changed)

        self.list_groups = QListWidget()
        self.list_groups.setSelectionMode(QAbstractItemView.NoSelection)
        self.list_groups.setFixedHeight(110)
        self.list_groups.itemChanged.connect(self._on_groups_changed)

        hv.addWidget(self._wrap("Categories", self.list_categories))
        hv.addWidget(self._wrap("Groups", self.list_groups))
        root.layout().addWidget(self._catalog_scope_box)

        # dual-list descriptor selection
        sel = QGroupBox("Descriptor selection")
        h = QHBoxLayout(sel)
        h.setContentsMargins(6, 6, 6, 6)
        h.setSpacing(8)

        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.available_list.setMinimumWidth(160)
        self.available_list.setMinimumHeight(140)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)
        self.btn_refresh = QPushButton("↺ Refresh")
        self.btn_add     = QPushButton("Add →")
        self.btn_rem     = QPushButton("← Remove")
        self.btn_all     = QPushButton("Add all")
        self.btn_none    = QPushButton("Clear")

        for b in (self.btn_refresh, self.btn_add, self.btn_rem,
                  self.btn_all, self.btn_none):
            b.setMinimumWidth(80)

        self.btn_refresh.clicked.connect(self._refresh_available_descriptors)
        self.btn_add.clicked.connect(self._on_add)
        self.btn_rem.clicked.connect(self._on_remove)
        self.btn_all.clicked.connect(self._on_add_all)
        self.btn_none.clicked.connect(self._on_remove_all)

        btn_col.addStretch(1)
        btn_col.addWidget(self.btn_refresh)
        btn_col.addSpacing(8)
        btn_col.addWidget(self.btn_add)
        btn_col.addWidget(self.btn_rem)
        btn_col.addSpacing(8)
        btn_col.addWidget(self.btn_all)
        btn_col.addWidget(self.btn_none)
        btn_col.addStretch(1)

        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.selected_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.selected_list.setMinimumWidth(160)
        self.selected_list.setMinimumHeight(140)
        self.selected_list.model().rowsMoved.connect(self._on_order_changed)

        h.addWidget(self._wrap("Available", self.available_list), 1)
        h.addLayout(btn_col)
        h.addWidget(self._wrap("Selected  (drag to reorder)", self.selected_list), 1)
        root.layout().addWidget(sel)

        # run
        self.btn_compute = QPushButton("Compute")
        self.btn_compute.clicked.connect(self.commit)
        self.cb_auto_run = QCheckBox("Auto-run")
        self.cb_auto_run.setChecked(bool(self.auto_run))
        self.cb_auto_run.toggled.connect(self._on_auto_run_toggled)
        root.layout().addWidget(self.cb_auto_run)
        root.layout().addWidget(self.btn_compute)

    @staticmethod
    def _wrap(title: str, w) -> QGroupBox:
        box = QGroupBox(title)
        v = QVBoxLayout(box)
        v.addWidget(w)
        return box

    def _set_status(self, text: str) -> None:
        self.lbl.setText(text)

    def _set_busy(self, busy: bool, text: str) -> None:
        widgets = [
            self.combo_preset,
            self.btn_refresh,
            self.btn_compute,
            self.btn_add,
            self.btn_rem,
            self.btn_all,
            self.btn_none,
        ]
        for widget in widgets:
            widget.setEnabled(not busy)
        self._set_status(text)
        if busy:
            self.progressBarInit()
        else:
            self.progressBarFinished()
            self._sync_preset_controls()

    def _current_preset(self) -> MordredPreset:
        return self._preset_map.get(self.preset_key, self._preset_map["custom"])

    def _sync_preset_controls(self) -> None:
        preset = self._current_preset()
        self.lbl_preset_desc.setText(preset.description)
        is_custom = preset.key == "custom"

        self._catalog_scope_box.setVisible(is_custom)
        self.list_categories.setEnabled(is_custom and MORDRED_AVAILABLE)
        self.list_groups.setEnabled(is_custom and MORDRED_AVAILABLE)

    def _active_group_filters(self) -> List[str]:
        preset = self._current_preset()
        if preset.key != "custom":
            return list(preset.groups)
        return list(self.checked_groups or [])

    def _apply_preset_scope(self) -> None:
        preset = self._current_preset()
        if preset.key == "custom":
            self._sync_preset_controls()
            return

        self.checked_categories = list(preset.categories)
        self.checked_groups = list(preset.groups)
        self._render_categories()
        self._render_groups_from_categories()
        self._sync_preset_controls()

    def _initialize_service(self) -> None:
        self._service = MordredDescriptorService(
            MordredComputeConfig(ignore_3d=self.ignore_3d, nproc=max(1, int(self.nproc)))
        )
        self._desc_map = {d.name: d.module for d in self._service.list_descriptors()}

    def _set_mordred_unavailable_state(self) -> None:
        message = "Optional package 'mordred' is not installed. Install the 'descriptors' extra to enable this widget."
        self.btn_compute.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_rem.setEnabled(False)
        self.btn_all.setEnabled(False)
        self.btn_none.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self.combo_preset.setEnabled(False)
        self.available_list.setEnabled(False)
        self.selected_list.setEnabled(False)
        self.list_categories.setEnabled(False)
        self.list_groups.setEnabled(False)
        self.cb_ignore3d.setEnabled(False)
        self.spin_nproc.setEnabled(False)
        self.cb_write_mols.setEnabled(False)
        self._set_status(message)

    # ---------------- defaults ----------------

    def _bootstrap_defaults(self) -> None:
        if self.preset_key not in self._preset_map:
            self.preset_key = "recommended_qsar"

        if not self.checked_categories:
            self.checked_categories = list(CATEGORIES.keys())

        if not self.checked_groups:
            self.checked_groups = sorted(DEFAULT_GROUPS)

        if not self.selected_descriptors:
            for name, mod in self._desc_map.items():
                if mod in DEFAULT_GROUPS:
                    self.selected_descriptors.append(name)

        self._render_categories()
        self._render_groups_from_categories()
        self._apply_preset_scope()
        self._render_selected_list()

    def _render_categories(self) -> None:
        self.list_categories.blockSignals(True)
        try:
            self.list_categories.clear()
            checked = set(self.checked_categories or [])
            for cat in CATEGORIES.keys():
                it = QListWidgetItem(cat)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if cat in checked else Qt.Unchecked)
                self.list_categories.addItem(it)
        finally:
            self.list_categories.blockSignals(False)

    def _render_groups_from_categories(self) -> None:
        preset = self._current_preset()
        active_cats = list(preset.categories) if preset.key != "custom" else self._get_checked(self.list_categories)

        groups: List[str] = []
        for cat in active_cats:
            groups.extend(CATEGORIES.get(cat, []))
        groups = sorted(dict.fromkeys(groups))

        self.list_groups.blockSignals(True)
        try:
            self.list_groups.clear()
            checked = set(self.checked_groups or [])
            for g in groups:
                it = QListWidgetItem(g)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if g in checked else Qt.Unchecked)
                self.list_groups.addItem(it)
        finally:
            self.list_groups.blockSignals(False)

    def _render_selected_list(self) -> None:
        self.selected_list.blockSignals(True)
        try:
            self.selected_list.clear()
            for n in self.selected_descriptors or []:
                self.selected_list.addItem(QListWidgetItem(n))
        finally:
            self.selected_list.blockSignals(False)

    @staticmethod
    def _get_checked(listw: QListWidget) -> List[str]:
        out: List[str] = []
        for i in range(listw.count()):
            it = listw.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.text())
        return out

    # ---------------- UI reactions ----------------

    def _on_options_changed(self) -> None:
        if not MORDRED_AVAILABLE:
            return
        self.ignore_3d = bool(self.cb_ignore3d.isChecked())
        self.nproc = int(self.spin_nproc.value())
        self.write_to_molecules = bool(self.cb_write_mols.isChecked())

        self._initialize_service()
        self._refresh_available_descriptors()
        self._maybe_autorun()

    def _on_preset_changed(self) -> None:
        self.preset_key = str(self.combo_preset.currentData() or "custom")
        self._apply_preset_scope()
        self.selected_descriptors = [
            name for name in self._read_selected_names()
            if name in self._descriptor_names_for_active_groups()
        ]
        self._render_selected_list()
        self._refresh_available_descriptors()
        self._set_status(self._input_summary())
        self._maybe_autorun()

    def _on_categories_changed(self) -> None:
        self.checked_categories = self._get_checked(self.list_categories)
        self._render_groups_from_categories()
        self._refresh_available_descriptors()
        self._maybe_autorun()

    def _on_groups_changed(self) -> None:
        self.checked_groups = self._get_checked(self.list_groups)
        self._refresh_available_descriptors()
        self._maybe_autorun()

    def _descriptor_names_for_active_groups(self) -> List[str]:
        active_groups = set(self._active_group_filters())
        if not active_groups:
            return []
        return [
            name
            for name, mod in sorted(self._desc_map.items(), key=lambda item: (item[1], item[0]))
            if any(mod.startswith(group) for group in active_groups)
        ]

    def _refresh_available_descriptors(self) -> None:
        active_groups = set(self._active_group_filters())
        selected = set(self._read_selected_names())

        self.available_list.clear()
        if not active_groups:
            return

        for name, mod in sorted(self._desc_map.items(), key=lambda x: (x[1], x[0])):
            if name in selected:
                continue
            if any(mod.startswith(g) for g in active_groups):
                self.available_list.addItem(QListWidgetItem(name))

    def _read_selected_names(self) -> List[str]:
        return _list_widget_texts(self.selected_list)

    def _on_add(self) -> None:
        items = self.available_list.selectedItems()
        if not items:
            return
        for it in items:
            name = it.text()
            row = self.available_list.row(it)
            self.available_list.takeItem(row)
            self.selected_list.addItem(QListWidgetItem(name))
        self.selected_descriptors = self._read_selected_names()
        self._maybe_autorun()

    def _on_remove(self) -> None:
        items = self.selected_list.selectedItems()
        if not items:
            return
        for it in items:
            name = it.text()
            row = self.selected_list.row(it)
            self.selected_list.takeItem(row)
            mod = self._desc_map.get(name, "")
            if any(mod.startswith(g) for g in set(self._active_group_filters())):
                self.available_list.addItem(QListWidgetItem(name))
        self.selected_descriptors = self._read_selected_names()
        self._maybe_autorun()

    def _on_add_all(self) -> None:
        while self.available_list.count() > 0:
            it = self.available_list.takeItem(0)
            self.selected_list.addItem(QListWidgetItem(it.text()))
        self.selected_descriptors = self._read_selected_names()
        self._maybe_autorun()

    def _on_remove_all(self) -> None:
        while self.selected_list.count() > 0:
            it = self.selected_list.takeItem(0)
            self.available_list.addItem(QListWidgetItem(it.text()))
        self.selected_descriptors = []
        self._refresh_available_descriptors()
        self._maybe_autorun()

    def _on_order_changed(self, *_args) -> None:
        self.selected_descriptors = self._read_selected_names()
        self._maybe_autorun()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        self.auto_run = bool(checked)
        self._maybe_autorun()

    # ---------------- Inputs ----------------

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
    def set_molecules(self, mols: Optional[list]) -> None:
        self._molecules = [m for m in (mols or []) if isinstance(m, ChemMol)]
        self._set_status(self._input_summary())
        self._maybe_autorun()

    def _input_summary(self) -> str:
        n_tab = 0 if self._data is None else len(self._data)
        n_mol = len(self._molecules)
        preset = self._current_preset().label
        parts = []
        if self._table_report is not None:
            parts.append(
                format_table_report(
                    self._table_report,
                    prefix="Table",
                    valid_label="valid SMILES",
                )
            )
        else:
            parts.append(f"Table rows={n_tab}")
        parts.append(f"Molecules={n_mol}")
        parts.append(f"Preset={preset}")
        if self.preset_key == "geometry_3d" and self.ignore_3d:
            parts.append("Note: disable 'Ignore 3D descriptors' to expose 3D modules.")
        return "Input: " + " | ".join(parts)

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and MORDRED_AVAILABLE and (self._data is not None or self._molecules):
            self.commit()

    # ---------------- Compute / Commit (background) ----------------

    def commit(self) -> None:
        if not MORDRED_AVAILABLE:
            set_widget_error(self, "Optional package 'mordred' is not installed.")
            self._set_status("Optional package 'mordred' is not installed.")
            return
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

        # invalidate previous job(s)
        self._task_id += 1
        task_id = self._task_id

        self._set_busy(True, "Computing descriptors (background)…")

        # best-effort cancel
        if self._future is not None:
            try:
                self._future.cancel()
            except RuntimeError as exc:
                logger.debug("Failed to cancel previous descriptor task: %s", exc)

        fut = self.executor.submit(
            self._compute_background,
            task_id,
            self._data,
            self._molecules,
            selected,
            bool(self.ignore_3d),
            int(self.nproc),
            bool(self.write_to_molecules),
        )
        setattr(fut, "_descriptor_task_id", task_id)
        self._future = fut
        fut.add_done_callback(self._on_done)

    def _compute_background(
        self,
        task_id: int,
        data: Optional[Table],
        mols: Sequence[ChemMol],
        selected: Sequence[str],
        ignore_3d: bool,
        nproc: int,
        write_to_molecules: bool,
    ) -> Tuple[int, Optional[Table], List[ChemMol], int, int, int]:
        cfg = MordredComputeConfig(ignore_3d=ignore_3d, nproc=max(1, int(nproc)))
        service = MordredDescriptorService(cfg)

        base_table = data

        rdkit_mols: List[Optional[Chem.Mol]] = []
        valid_idx: List[int] = []
        n_total = 0

        # Prefer Table SMILES if available (keeps row alignment)
        if base_table is not None and len(base_table) > 0:
            n_total = len(base_table)
            smi_var = _find_smiles_var(base_table)
            if smi_var is not None:
                smiles = [str(r[smi_var]) if r[smi_var] is not None else "" for r in base_table]
                rdkit_mols, valid_idx = service.smiles_to_mols(smiles)

        # Fallback to Molecules if no valid SMILES in table
        if (base_table is None or n_total == 0 or not valid_idx) and mols:
            base_table = None
            n_total = len(mols)
            rdkit_mols, valid_idx = service.chemmols_to_mols(mols)

        valid_mols = [m for m in rdkit_mols if m is not None]
        if not valid_mols:
            # nothing to compute
            return task_id, base_table, list(mols), n_total, 0, n_total

        df_valid = service.compute(valid_mols, selected, cfg=cfg)
        df_full = service.df_to_full_length(df_valid, valid_idx, n_total)

        out_table = self._attach_df_to_table(base_table, df_full, selected)

        out_mols = list(mols)
        if write_to_molecules and out_mols and len(out_mols) == len(df_full):
            for i, cm in enumerate(out_mols):
                if cm is None:
                    continue
                for col in selected:
                    v = df_full.iloc[i][col] if i < len(df_full) else np.nan
                    numeric_value = service.numeric_or_none(v)
                    if numeric_value is None:
                        continue
                    cm.set_prop(col, numeric_value)

        invalid_count = n_total - len(valid_idx)
        return task_id, out_table, out_mols, n_total, len(selected), invalid_count

    def _attach_df_to_table(self, base: Optional[Table], df_full, selected: Sequence[str]) -> Table:
        """
        Append descriptor attributes to incoming table (if present), ensuring unique names.
        If no base table is present, create a descriptor-only table.
        """
        selected = _unique_preserve_order([str(x) for x in selected if x])
        # keep only descriptors that were actually computed (ignore_3d or version mismatch may drop some)
        selected = [s for s in selected if s in df_full.columns]
        if not selected:
            return base if base is not None else safe_table_from_numpy(Domain([]), X=np.empty((len(df_full), 0)))
        X_desc = df_full[selected].to_numpy(dtype=float, copy=False)

        if base is None:
            taken: set[str] = set()
            desc_vars: List[ContinuousVariable] = []
            for name in selected:
                uname = _make_unique_name(name, taken)
                desc_vars.append(ContinuousVariable(uname))
            dom = Domain(desc_vars, metas=[])
            return safe_table_from_numpy(dom, X=X_desc, metas=np.zeros((len(X_desc), 0), dtype=object))

        dom0 = base.domain
        taken = {v.name for v in (list(dom0.attributes) + list(dom0.class_vars) + list(dom0.metas))}

        desc_vars: List[ContinuousVariable] = []
        for name in selected:
            uname = _make_unique_name(name, taken)
            desc_vars.append(ContinuousVariable(uname))

        # keep original attributes and append descriptors (more useful in Orange workflows)
        attrs_out = list(dom0.attributes) + desc_vars
        dom_out = Domain(attrs_out, dom0.class_vars, metas=dom0.metas)

        X0 = base.X
        X_out = np.hstack([X0, X_desc]) if X0.size else X_desc
        return safe_table_from_numpy(dom_out, X=X_out, Y=base.Y, metas=base.metas, name=getattr(base, "name", "Data"))

    def _on_done(self, fut) -> None:
        try:
            task_id, table, mols, n_total, n_desc, invalid = fut.result()
            methodinvoke(self, "_apply_outputs", (int, object, object, int, int, int))(
                task_id, table, mols, n_total, n_desc, invalid
            )
        except Exception as exc:
            task_id = getattr(fut, "_descriptor_task_id", self._task_id)
            methodinvoke(self, "_apply_error", (int, str))(task_id, str(exc))

    @pyqtSlot(int, str)
    def _apply_error(self, task_id: int, msg: str) -> None:
        # ignore stale results
        if task_id != self._task_id:
            return
        self._set_busy(False, format_failed_status(msg))
        self.Outputs.data.send(None)
        self.Outputs.molecules.send([])

    @pyqtSlot(int, object, object, int, int, int)
    def _apply_outputs(self, task_id: int, table: object, mols: object, n_total: int, n_desc: int, invalid: int) -> None:
        # ignore stale results
        if task_id != self._task_id:
            return
        set_widget_warning(
            self,
            format_skip_warning(
                invalid,
                subject="input rows",
                action="were skipped during descriptor computation",
            ),
        )
        if self._table_report is not None:
            status = format_done_status(
                f"descriptors={n_desc}",
                f"valid rows={n_total - invalid}/{self._table_report.n_rows}",
                f"invalid={invalid}",
            )
        else:
            status = format_done_status(
                f"rows={n_total}",
                f"descriptors={n_desc}",
                f"invalid={invalid}",
            )
        self._set_busy(False, status)
        self.Outputs.data.send(table)
        self.Outputs.molecules.send(mols)
