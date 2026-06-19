from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from AnyQt.QtCore import pyqtSlot
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
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

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.padel_descriptor_service import (
    PadelComputeConfig,
    PadelDescriptorService,
    PadelPreset,
)


logger = logging.getLogger(__name__)


def _find_smiles_var(data: Table) -> Optional[StringVariable]:
    if data is None:
        return None
    for v in data.domain.metas:
        if v.name.strip().lower() == "smiles":
            return v if isinstance(v, StringVariable) else None
    for v in list(data.domain.metas) + list(data.domain.attributes):
        if "smiles" in v.name.strip().lower():
            return v if isinstance(v, StringVariable) else None
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


def _make_unique_name(name: str, taken: set[str], prefix: str = "padel_") -> str:
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


class OWPadelDescriptors(OWWidget):
    name = "PaDEL Descriptors"
    description = "Compute PaDEL descriptors from Table (SMILES) and/or Molecules (ChemMol)."
    icon = "icons/descriptors/owmoldescriptorwidget.png"
    priority = 134
    keywords = ["padel", "descriptor", "descriptors", "fingerprints", "smiles"]
    want_main_area = False
    resizing_enabled = True

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        data = Output("Data", Table)
        molecules = Output("Molecules", list, auto_summary=False)

    preset_key: str = Setting("custom")
    selected_descriptors: List[str] = Setting([])
    calculate_2d: bool = Setting(True)
    calculate_3d: bool = Setting(False)
    fingerprints: bool = Setting(False)
    convert_3d: bool = Setting(False)
    remove_salt: bool = Setting(False)
    detect_aromaticity: bool = Setting(False)
    standardize_nitro: bool = Setting(False)
    standardize_tautomers: bool = Setting(False)
    threads: int = Setting(0)          # 0 = PaDEL default
    timeout: int = Setting(300)        # 0 = unlimited
    maxruntime: int = Setting(0)       # 0 = unlimited
    write_to_molecules: bool = Setting(False)
    auto_run: bool = Setting(False)

    def __init__(self) -> None:
        super().__init__()
        self.executor = ThreadExecutor(self)

        self._data: Optional[Table] = None
        self._molecules: List[ChemMol] = []
        self._service = PadelDescriptorService(self._build_cfg())
        self._available_names: List[str] = []
        self._presets: List[PadelPreset] = PadelDescriptorService.list_presets()
        self._preset_map: Dict[str, PadelPreset] = {preset.key: preset for preset in self._presets}
        self._dependency_ready = False
        self._dependency_message = ""

        self._task_id: int = 0
        self._future = None
        self._autorun_after_catalog = False

        self._build_ui()
        self._sync_preset_controls(initial=True)
        self._render_selected_list()
        self._refresh_dependency_state()
        if self._dependency_ready:
            self._set_status("Initializing PaDEL descriptor catalog…")
            self._refresh_catalog()
        else:
            self._apply_dependency_state()

    def _build_ui(self) -> None:
        root = self.controlArea

        self.lbl = QLabel("Waiting for input…")
        self.lbl.setWordWrap(True)
        root.layout().addWidget(self.lbl)

        preset_box = QGroupBox("PaDEL XML presets")
        preset_layout = QVBoxLayout(preset_box)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset"))

        self.combo_preset = QComboBox()
        for preset in self._presets:
            self.combo_preset.addItem(preset.label, preset.key)
        idx = max(0, self.combo_preset.findData(self.preset_key))
        self.combo_preset.setCurrentIndex(idx)
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.combo_preset, 1)
        preset_layout.addLayout(preset_row)

        self.lbl_preset_desc = QLabel("")
        self.lbl_preset_desc.setWordWrap(True)
        preset_layout.addWidget(self.lbl_preset_desc)
        root.layout().addWidget(preset_box)

        opt = QGroupBox("PaDEL options")
        gl = QGridLayout(opt)

        self.cb_2d = QCheckBox("Calculate 2D descriptors")
        self.cb_2d.setChecked(bool(self.calculate_2d))
        self.cb_2d.toggled.connect(self._on_options_changed)

        self.cb_3d = QCheckBox("Calculate 3D descriptors")
        self.cb_3d.setChecked(bool(self.calculate_3d))
        self.cb_3d.toggled.connect(self._on_options_changed)

        self.cb_fp = QCheckBox("Include PaDEL fingerprints")
        self.cb_fp.setChecked(bool(self.fingerprints))
        self.cb_fp.toggled.connect(self._on_options_changed)

        self.cb_convert3d = QCheckBox("Convert molecules to 3D before calculation")
        self.cb_convert3d.setChecked(bool(self.convert_3d))
        self.cb_convert3d.toggled.connect(self._on_options_changed)

        self.cb_remove_salt = QCheckBox("Remove salts")
        self.cb_remove_salt.setChecked(bool(self.remove_salt))
        self.cb_remove_salt.toggled.connect(self._on_options_changed)

        self.cb_detect_arom = QCheckBox("Detect aromaticity")
        self.cb_detect_arom.setChecked(bool(self.detect_aromaticity))
        self.cb_detect_arom.toggled.connect(self._on_options_changed)

        self.cb_nitro = QCheckBox("Standardize nitro groups")
        self.cb_nitro.setChecked(bool(self.standardize_nitro))
        self.cb_nitro.toggled.connect(self._on_options_changed)

        self.cb_tauts = QCheckBox("Standardize tautomers")
        self.cb_tauts.setChecked(bool(self.standardize_tautomers))
        self.cb_tauts.toggled.connect(self._on_options_changed)

        self.cb_write_mols = QCheckBox("Write descriptor values into Molecules (ChemMol props)")
        self.cb_write_mols.setChecked(bool(self.write_to_molecules))
        self.cb_write_mols.toggled.connect(self._on_options_changed)

        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(0, 128)
        self.spin_threads.setValue(int(self.threads))
        self.spin_threads.valueChanged.connect(self._on_options_changed)

        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(0, 36000)
        self.spin_timeout.setValue(int(self.timeout))
        self.spin_timeout.valueChanged.connect(self._on_options_changed)

        self.spin_maxruntime = QSpinBox()
        self.spin_maxruntime.setRange(0, 36000)
        self.spin_maxruntime.setValue(int(self.maxruntime))
        self.spin_maxruntime.valueChanged.connect(self._on_options_changed)

        gl.addWidget(self.cb_2d, 0, 0)
        gl.addWidget(self.cb_3d, 0, 1)
        gl.addWidget(self.cb_fp, 1, 0)
        gl.addWidget(self.cb_convert3d, 1, 1)
        gl.addWidget(self.cb_remove_salt, 2, 0)
        gl.addWidget(self.cb_detect_arom, 2, 1)
        gl.addWidget(self.cb_nitro, 3, 0)
        gl.addWidget(self.cb_tauts, 3, 1)
        gl.addWidget(QLabel("Threads (0=default)"), 4, 0)
        gl.addWidget(self.spin_threads, 4, 1)
        gl.addWidget(QLabel("Subprocess timeout, s (0=unlimited)"), 5, 0)
        gl.addWidget(self.spin_timeout, 5, 1)
        gl.addWidget(QLabel("Per-molecule max runtime, s (0=unlimited)"), 6, 0)
        gl.addWidget(self.spin_maxruntime, 6, 1)
        gl.addWidget(self.cb_write_mols, 7, 0, 1, 2)
        root.layout().addWidget(opt)

        sel = QGroupBox("Descriptor selection")
        h = QHBoxLayout(sel)

        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.available_list.setMinimumWidth(160)
        self.available_list.setMinimumHeight(140)

        btn_col = QVBoxLayout()
        self.btn_refresh = QPushButton("Refresh list")
        self.btn_add = QPushButton("→ Add →")
        self.btn_rem = QPushButton("← Remove")
        self.btn_all = QPushButton("Add all")
        self.btn_none = QPushButton("Remove all")

        for b in (self.btn_refresh, self.btn_add, self.btn_rem, self.btn_all, self.btn_none):
            b.setMinimumWidth(90)

        self.btn_refresh.clicked.connect(self._refresh_catalog)
        self.btn_add.clicked.connect(self._on_add)
        self.btn_rem.clicked.connect(self._on_remove)
        self.btn_all.clicked.connect(self._on_add_all)
        self.btn_none.clicked.connect(self._on_remove_all)

        btn_col.addStretch(1)
        btn_col.addWidget(self.btn_refresh)
        btn_col.addSpacing(12)
        btn_col.addWidget(self.btn_add)
        btn_col.addWidget(self.btn_rem)
        btn_col.addSpacing(12)
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
        h.addWidget(self._wrap("Selected (empty = compute all available)", self.selected_list), 1)
        root.layout().addWidget(sel)

        runbox = QHBoxLayout()
        self.btn_compute = QPushButton("Compute PaDEL descriptors")
        self.btn_compute.clicked.connect(self.commit)
        self.cb_auto_run = QCheckBox("Auto-run")
        self.cb_auto_run.setChecked(bool(self.auto_run))
        self.cb_auto_run.toggled.connect(self._on_auto_run_toggled)
        runbox.addWidget(self.cb_auto_run)
        runbox.addStretch(1)
        runbox.addWidget(self.btn_compute)
        root.layout().addLayout(runbox)

    @staticmethod
    def _wrap(title: str, widget) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.addWidget(widget)
        return box

    def _set_status(self, text: str) -> None:
        self.lbl.setText(text)

    def _set_busy(self, busy: bool, text: str) -> None:
        widgets = [
            self.combo_preset,
            self.btn_refresh,
            self.btn_add,
            self.btn_rem,
            self.btn_all,
            self.btn_none,
            self.btn_compute,
        ]
        for widget in widgets:
            widget.setEnabled(not busy)
        self._set_status(text)
        if busy:
            self.progressBarInit()
        else:
            self.progressBarFinished()
            self._apply_dependency_state()

    def _refresh_dependency_state(self) -> None:
        self._dependency_ready, self._dependency_message = PadelDescriptorService.dependency_status()

    def _apply_dependency_state(self) -> None:
        enabled = bool(self._dependency_ready)
        for widget in (
            self.btn_refresh,
            self.btn_add,
            self.btn_rem,
            self.btn_all,
            self.btn_none,
            self.btn_compute,
            self.available_list,
            self.selected_list,
        ):
            widget.setEnabled(enabled)

        if not enabled:
            self._set_status(self._dependency_message)

    def _current_preset(self) -> PadelPreset:
        return self._preset_map.get(self.preset_key, self._preset_map["custom"])

    def _sync_preset_controls(self, *, initial: bool = False) -> None:
        preset = self._current_preset()
        desc = preset.description
        if preset.filename:
            desc += f" Using XML preset: {preset.filename}"
        self.lbl_preset_desc.setText(desc)

        preset_locked_scope = preset.key != "custom"
        for cb in (self.cb_2d, self.cb_3d, self.cb_fp):
            cb.setEnabled(not preset_locked_scope)

        if preset_locked_scope:
            self.cb_2d.blockSignals(True)
            self.cb_3d.blockSignals(True)
            self.cb_fp.blockSignals(True)
            try:
                self.calculate_2d = bool(preset.calculate_2d)
                self.calculate_3d = bool(preset.calculate_3d)
                self.fingerprints = bool(preset.fingerprints)
                self.cb_2d.setChecked(self.calculate_2d)
                self.cb_3d.setChecked(self.calculate_3d)
                self.cb_fp.setChecked(self.fingerprints)
            finally:
                self.cb_2d.blockSignals(False)
                self.cb_3d.blockSignals(False)
                self.cb_fp.blockSignals(False)
        elif initial:
            self.cb_2d.setChecked(bool(self.calculate_2d))
            self.cb_3d.setChecked(bool(self.calculate_3d))
            self.cb_fp.setChecked(bool(self.fingerprints))

    def _build_cfg(self) -> PadelComputeConfig:
        threads = -1 if int(self.threads) == 0 else int(self.threads)
        timeout = 0 if int(self.timeout) == 0 else int(self.timeout)
        maxruntime = -1 if int(self.maxruntime) == 0 else int(self.maxruntime)

        if self.preset_key != "custom":
            return PadelDescriptorService.config_from_preset(
                self.preset_key,
                convert_3d=bool(self.convert_3d),
                remove_salt=bool(self.remove_salt),
                detect_aromaticity=bool(self.detect_aromaticity),
                standardize_nitro=bool(self.standardize_nitro),
                standardize_tautomers=bool(self.standardize_tautomers),
                threads=threads,
                timeout=timeout,
                maxruntime=maxruntime,
            )

        return PadelComputeConfig(
            calculate_2d=bool(self.calculate_2d),
            calculate_3d=bool(self.calculate_3d),
            fingerprints=bool(self.fingerprints),
            convert_3d=bool(self.convert_3d),
            remove_salt=bool(self.remove_salt),
            detect_aromaticity=bool(self.detect_aromaticity),
            standardize_nitro=bool(self.standardize_nitro),
            standardize_tautomers=bool(self.standardize_tautomers),
            threads=threads,
            timeout=timeout,
            maxruntime=maxruntime,
            descriptor_types_path=None,
        )

    def _render_selected_list(self) -> None:
        self.selected_list.blockSignals(True)
        try:
            self.selected_list.clear()
            for name in self.selected_descriptors or []:
                self.selected_list.addItem(QListWidgetItem(name))
        finally:
            self.selected_list.blockSignals(False)

    def _render_available_list(self) -> None:
        selected = set(self._read_selected_names())
        self.available_list.clear()
        for name in self._available_names:
            if name not in selected:
                self.available_list.addItem(QListWidgetItem(name))

    def _read_selected_names(self) -> List[str]:
        return _list_widget_texts(self.selected_list)

    def _on_preset_changed(self) -> None:
        self.preset_key = str(self.combo_preset.currentData() or "custom")
        self._sync_preset_controls()
        self._service = PadelDescriptorService(self._build_cfg())
        self._refresh_dependency_state()
        self._autorun_after_catalog = bool(self.auto_run)
        self._refresh_catalog()

    def _on_options_changed(self) -> None:
        self.calculate_2d = bool(self.cb_2d.isChecked())
        self.calculate_3d = bool(self.cb_3d.isChecked())
        self.fingerprints = bool(self.cb_fp.isChecked())
        self.convert_3d = bool(self.cb_convert3d.isChecked())
        self.remove_salt = bool(self.cb_remove_salt.isChecked())
        self.detect_aromaticity = bool(self.cb_detect_arom.isChecked())
        self.standardize_nitro = bool(self.cb_nitro.isChecked())
        self.standardize_tautomers = bool(self.cb_tauts.isChecked())
        self.write_to_molecules = bool(self.cb_write_mols.isChecked())
        self.threads = int(self.spin_threads.value())
        self.timeout = int(self.spin_timeout.value())
        self.maxruntime = int(self.spin_maxruntime.value())
        self._service = PadelDescriptorService(self._build_cfg())
        self._refresh_dependency_state()
        self._autorun_after_catalog = bool(self.auto_run)
        self._refresh_catalog()

    def _on_add(self) -> None:
        items = self.available_list.selectedItems()
        if not items:
            return
        for item in items:
            row = self.available_list.row(item)
            self.available_list.takeItem(row)
            self.selected_list.addItem(QListWidgetItem(item.text()))
        self.selected_descriptors = self._read_selected_names()
        self._maybe_autorun()

    def _on_remove(self) -> None:
        items = self.selected_list.selectedItems()
        if not items:
            return
        for item in items:
            row = self.selected_list.row(item)
            self.selected_list.takeItem(row)
        self.selected_descriptors = self._read_selected_names()
        self._render_available_list()
        self._maybe_autorun()

    def _on_add_all(self) -> None:
        while self.available_list.count() > 0:
            item = self.available_list.takeItem(0)
            self.selected_list.addItem(QListWidgetItem(item.text()))
        self.selected_descriptors = self._read_selected_names()
        self._maybe_autorun()

    def _on_remove_all(self) -> None:
        self.selected_list.clear()
        self.selected_descriptors = []
        self._render_available_list()
        self._maybe_autorun()

    def _on_order_changed(self, *_args) -> None:
        self.selected_descriptors = self._read_selected_names()
        self._maybe_autorun()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        self.auto_run = bool(checked)
        self._maybe_autorun()

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._data = data
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
        extra = ""
        if self.calculate_3d and not self.convert_3d:
            extra = " | Note: 3D descriptors usually need 3D coordinates or 'Convert to 3D'."
        preset = self._current_preset().label
        return f"Input: Table rows={n_tab}, Molecules={n_mol} | Preset={preset}{extra}"

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and self._dependency_ready and (self._data is not None or self._molecules):
            self.commit()

    def _refresh_catalog(self) -> None:
        if not self._dependency_ready:
            self._available_names = []
            self._render_available_list()
            self._apply_dependency_state()
            return

        self._task_id += 1
        task_id = self._task_id
        self._set_busy(True, "Loading PaDEL descriptor catalog…")

        if self._future is not None:
            try:
                self._future.cancel()
            except RuntimeError as exc:
                logger.debug("Failed to cancel previous PaDEL catalog task: %s", exc)

        future = self.executor.submit(self._load_catalog_background, task_id, self._build_cfg())
        setattr(future, "_padel_task_id", task_id)
        self._future = future
        future.add_done_callback(self._on_done)

    def _load_catalog_background(self, task_id: int, cfg: PadelComputeConfig):
        try:
            service = PadelDescriptorService(cfg)
            names = service.list_descriptor_names(cfg=cfg)
            return ("catalog", task_id, names)
        except Exception as exc:
            return ("catalog_error", task_id, str(exc))

    def commit(self) -> None:
        self._refresh_dependency_state()
        if not self._dependency_ready:
            self.Outputs.data.send(None)
            self.Outputs.molecules.send([])
            self._apply_dependency_state()
            return

        if (self._data is None or len(self._data) == 0) and not self._molecules:
            self.Outputs.data.send(None)
            self.Outputs.molecules.send([])
            self._set_status("No input.")
            return

        self.selected_descriptors = self._read_selected_names()

        self._task_id += 1
        task_id = self._task_id
        self._set_busy(True, "Computing PaDEL descriptors (background)…")

        if self._future is not None:
            try:
                self._future.cancel()
            except RuntimeError as exc:
                logger.debug("Failed to cancel previous PaDEL compute task: %s", exc)

        future = self.executor.submit(
            self._compute_background,
            task_id,
            self._data,
            self._molecules,
            list(self.selected_descriptors),
            self._build_cfg(),
            bool(self.write_to_molecules),
        )
        setattr(future, "_padel_task_id", task_id)
        self._future = future
        future.add_done_callback(self._on_done)

    def _compute_background(
        self,
        task_id: int,
        data: Optional[Table],
        mols: Sequence[ChemMol],
        selected: Sequence[str],
        cfg: PadelComputeConfig,
        write_to_molecules: bool,
    ):
        try:
            service = PadelDescriptorService(cfg)
            base_table = data

            smiles: List[str] = []
            n_total = 0

            if base_table is not None and len(base_table) > 0:
                n_total = len(base_table)
                smi_var = _find_smiles_var(base_table)
                if smi_var is not None:
                    smiles = [str(row[smi_var]) if row[smi_var] is not None else "" for row in base_table]

            if (base_table is None or n_total == 0 or not any((s or "").strip() for s in smiles)) and mols:
                base_table = None
                smiles = service.chemmols_to_smiles(mols)
                n_total = len(smiles)

            if not smiles:
                return ("compute", task_id, base_table, list(mols), 0, 0, 0)

            df_full = service.compute(smiles, selected, cfg=cfg)
            selected_final = list(df_full.columns)
            out_table = self._attach_df_to_table(base_table, df_full, selected_final)

            out_mols = list(mols)
            if write_to_molecules and out_mols and len(out_mols) == len(df_full):
                for i, chem_mol in enumerate(out_mols):
                    if chem_mol is None:
                        continue
                    for col in selected_final:
                        value = df_full.iloc[i][col] if i < len(df_full) else np.nan
                        numeric_value = service.numeric_or_none(value)
                        if numeric_value is None:
                            continue
                        chem_mol.set_prop(col, numeric_value)

            invalid_count = sum(1 for s in smiles if not (s or "").strip())
            return ("compute", task_id, out_table, out_mols, n_total, len(selected_final), invalid_count)
        except Exception as exc:
            return ("compute_error", task_id, str(exc))

    def _attach_df_to_table(self, base: Optional[Table], df_full, selected: Sequence[str]) -> Table:
        selected = _unique_preserve_order([str(x) for x in selected if x])
        x_desc = df_full[selected].to_numpy(dtype=float, copy=False) if selected else np.zeros((len(df_full), 0), dtype=float)

        if base is None:
            taken: set[str] = set()
            desc_vars: List[ContinuousVariable] = []
            for name in selected:
                unique_name = _make_unique_name(name, taken)
                desc_vars.append(ContinuousVariable(unique_name))
            domain = Domain(desc_vars, metas=[])
            return Table.from_numpy(domain, X=x_desc, metas=np.zeros((len(x_desc), 0), dtype=object))

        domain0 = base.domain
        taken = {v.name for v in (list(domain0.attributes) + list(domain0.class_vars) + list(domain0.metas))}

        desc_vars: List[ContinuousVariable] = []
        for name in selected:
            unique_name = _make_unique_name(name, taken)
            desc_vars.append(ContinuousVariable(unique_name))

        attrs_out = list(domain0.attributes) + desc_vars
        domain_out = Domain(attrs_out, domain0.class_vars, metas=domain0.metas)

        x0 = base.X
        x_out = np.hstack([x0, x_desc]) if x_desc.size and x0.size else (x_desc if x_desc.size else x0)
        return Table.from_numpy(domain_out, X=x_out, Y=base.Y, metas=base.metas)

    def _on_done(self, future) -> None:
        try:
            result = future.result()
            kind = result[0]
            if kind == "catalog":
                _, task_id, names = result
                methodinvoke(self, "_apply_catalog", (int, object))(task_id, names)
            elif kind == "catalog_error":
                _, task_id, msg = result
                methodinvoke(self, "_apply_catalog_error", (int, str))(task_id, msg)
            elif kind == "compute_error":
                _, task_id, msg = result
                methodinvoke(self, "_apply_error", (int, str))(task_id, msg)
            else:
                _, task_id, table, mols, n_total, n_desc, invalid = result
                methodinvoke(self, "_apply_outputs", (int, object, object, int, int, int))(
                    task_id, table, mols, n_total, n_desc, invalid
                )
        except Exception as exc:
            task_id = getattr(future, "_padel_task_id", self._task_id)
            methodinvoke(self, "_apply_error", (int, str))(task_id, str(exc))

    @pyqtSlot(int, object)
    def _apply_catalog(self, task_id: int, names: object) -> None:
        if task_id != self._task_id:
            return
        self._available_names = list(names or [])
        available_set = set(self._available_names)
        self.selected_descriptors = [name for name in self._read_selected_names() if name in available_set]
        self._render_selected_list()
        self._render_available_list()
        msg = f"PaDEL catalog loaded: {len(self._available_names)} available columns."
        if self.calculate_3d and not self.convert_3d:
            msg += " 3D descriptors may be empty unless 3D coordinates are available."
        self._set_busy(False, msg)
        if self._autorun_after_catalog:
            self._autorun_after_catalog = False
            self._maybe_autorun()

    @pyqtSlot(int, str)
    def _apply_catalog_error(self, task_id: int, msg: str) -> None:
        if task_id != self._task_id:
            return
        self._available_names = []
        self._render_available_list()
        self._set_busy(False, f"Catalog unavailable: {msg}")

    @pyqtSlot(int, str)
    def _apply_error(self, task_id: int, msg: str) -> None:
        if task_id != self._task_id:
            return
        self._set_busy(False, f"Failed: {msg}")
        self.Outputs.data.send(None)
        self.Outputs.molecules.send([])

    @pyqtSlot(int, object, object, int, int, int)
    def _apply_outputs(self, task_id: int, table: object, mols: object, n_total: int, n_desc: int, invalid: int) -> None:
        if task_id != self._task_id:
            return
        self._set_busy(False, f"Done: rows={n_total}, descriptors={n_desc}, invalid={invalid}")
        self.Outputs.data.send(table)
        self.Outputs.molecules.send(mols)
