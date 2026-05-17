from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from AnyQt.QtCore import Qt, pyqtSignal, QThread, QTimer
from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table, Variable
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.from_orange import (
    TableMolConversionReport,
    table_to_chemmols_with_report,
)
from chem_inf_widgets.chemcore.descriptors.fingerprints import compute_fingerprints_from_smiles
from chem_inf_widgets.chemcore.services.rdkit_safe import (
    safe_canonical_smiles,
    safe_mol_from_smiles,
    safe_mol_to_inchikey,
)
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_error_status,
    format_table_report,
    set_widget_error,
    set_widget_warning,
)


# ---------------------------- helpers ----------------------------

def _unique_name(name: str, used: set[str]) -> str:
    """Return a unique variable name (Orange Domain requires uniqueness)."""
    if name not in used:
        used.add(name)
        return name
    i = 2
    while f"{name}_{i}" in used:
        i += 1
    out = f"{name}_{i}"
    used.add(out)
    return out


def _text_like_vars(data: Table) -> List[Variable]:
    """Return variables that can reasonably carry SMILES-like textual values."""
    candidates = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    return [v for v in candidates if isinstance(v, (StringVariable, DiscreteVariable))]


def _find_smiles_var(data: Table) -> Optional[Variable]:
    """Try to auto-detect a SMILES-like variable (meta preferred)."""
    wanted = {"smiles", "canonical_smiles", "smile"}
    candidates = _text_like_vars(data)

    for v in candidates:
        if v.name.strip().lower() in wanted:
            return v

    for v in candidates:
        if "smiles" in v.name.strip().lower() or "smi" == v.name.strip().lower():
            return v

    return None


def _col_as_str_list(data: Table, var: Variable) -> List[str]:
    out: List[str] = []
    for row in data:
        try:
            value = row[var]
        except Exception:
            value = None
        out.append("" if value is None else str(value).strip())
    return out


def _object_to_smiles(obj: Any) -> str:
    """Robustly obtain SMILES from ChemMol, RDKit Mol, or raw SMILES string."""
    if isinstance(obj, str):
        return obj.strip()
    try:
        from rdkit.Chem.rdchem import Mol  # type: ignore
        if isinstance(obj, Mol):
            return safe_canonical_smiles(obj, remove_hs=False, canonical=True, isomeric=True)
    except Exception:
        pass
    if isinstance(obj, ChemMol):
        return _chemmol_to_smiles(obj)
    return ""


def _copy_chemmol(cm: ChemMol) -> ChemMol:
    """Copy ChemMol so this widget does not mutate upstream workflow objects."""
    try:
        return cm.copy() if hasattr(cm, "copy") else copy.deepcopy(cm)
    except Exception:
        try:
            out = ChemMol.from_rdkit(cm.to_rdkit(), name=getattr(cm, "name", None))
            out.props.update(dict(getattr(cm, "props", {}) or {}))
            out.cache.update(dict(getattr(cm, "cache", {}) or {}))
            return out
        except Exception:
            return cm


def _chemmol_to_smiles(cm: ChemMol) -> str:
    """Robustly obtain SMILES from ChemMol."""
    if getattr(cm, "props", None):
        s = cm.props.get("SMILES") or cm.props.get("smiles")
        if isinstance(s, str) and s.strip():
            return s.strip()

    s2 = getattr(cm, "smiles", None)
    if isinstance(s2, str) and s2.strip():
        return s2.strip()

    try:
        return safe_canonical_smiles(cm.mol, remove_hs=False, canonical=True, isomeric=True) if cm.mol is not None else ""
    except Exception:
        return ""




def _inchikey_from_smiles(smiles: str) -> str:
    """Return a stable InChIKey for a SMILES string, or an empty string on failure."""
    parsed = safe_mol_from_smiles((smiles or "").strip(), sanitize=True, remove_hs=True)
    return safe_mol_to_inchikey(parsed.mol) if parsed.ok else ""

def _fp_bytes_from_row(bits01: np.ndarray, *, nbits: int) -> bytes:
    """
    Pack a 1D 0/1 array into bytes.

    Note: uses numpy.packbits. nbits is stored alongside bytes to disambiguate padding.
    """
    b = np.asarray(bits01, dtype=np.uint8).reshape(-1)
    packed = np.packbits(b, bitorder="little")
    # Ensure stable length for a given nbits
    need = (int(nbits) + 7) // 8
    if packed.size < need:
        packed = np.pad(packed, (0, need - packed.size), constant_values=0)
    elif packed.size > need:
        packed = packed[:need]
    return packed.tobytes()


def _attach_fp_props(
    cm: ChemMol,
    *,
    fp_type: str,
    nbits: int,
    radius: int,
    fp_bytes: bytes,
    onbits: Optional[List[int]] = None,
) -> None:
    """Attach a minimal, future-proof fingerprint schema into ChemMol.props['fp']."""
    if getattr(cm, "props", None) is None:
        cm.props = {}

    fp: Dict[str, Any] = {
        "type": str(fp_type),
        "nbits": int(nbits),
        "bytes": fp_bytes,
        "schema_version": "1.1",
    }
    if str(fp_type).lower() == "morgan":
        fp["radius"] = int(radius)
    if onbits is not None:
        fp["onbits"] = onbits

    cm.props["fp"] = fp


@dataclass(frozen=True)
class _FPJobSpec:
    fp_type: str
    bit_size: int
    radius: int
    sanitize: bool


# ---------------------------- worker ----------------------------

class _FPWorker(QThread):
    finished = pyqtSignal(object)  # (res, valid_smiles)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, smiles: List[str], spec: _FPJobSpec) -> None:
        super().__init__()
        self._smiles = smiles
        self._spec = spec

    def _emit_cancelled_if_requested(self) -> bool:
        if self.isInterruptionRequested():
            self.cancelled.emit()
            return True
        return False

    def run(self) -> None:
        try:
            if self._emit_cancelled_if_requested():
                return
            res = compute_fingerprints_from_smiles(
                self._smiles,
                fp_type=self._spec.fp_type,
                bit_size=int(self._spec.bit_size),
                radius=int(self._spec.radius),
                sanitize=bool(self._spec.sanitize),
                progress_cb=self.progress.emit,
                cancel_cb=self.isInterruptionRequested,
            )
            if self._emit_cancelled_if_requested():
                return
            valid_smiles = [self._smiles[i] for i in res.valid_indices]
            self.finished.emit((res, valid_smiles))
        except Exception as e:
            self.failed.emit(str(e))


# ---------------------------- widget ----------------------------

class OWFingerprintGenerator(OWWidget):
    name = "Fingerprint Generator"
    description = "Compute RDKit fingerprints (Morgan/RDKit/MACCS) from Table (SMILES) or Molecules."
    icon = "icons/descriptors/owmolfingerprintwidget.png"
    priority = 131

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        fingerprints = Output("Fingerprints", Table)
        molecules = Output("Molecules", list, auto_summary=False)

    # Settings
    fp_type_idx: int = Setting(0)  # 0 Morgan, 1 RDKit, 2 MACCS
    bit_size: int = Setting(1024)
    radius: int = Setting(2)
    sanitize: bool = Setting(True)

    smiles_var_name: str = Setting("")        # chosen column name (when using Table)
    keep_meta_names: list[str] = Setting([])  # which metas to keep (Table only)

    # NEW: optionally output ChemMol objects with joined props and fp schema
    output_molecules: bool = Setting(False)
    store_onbits: bool = Setting(False)

    # NEW: copy all original columns (metas + attrs) into output table (as metas)
    include_input_columns: bool = Setting(True)

    # NEW: append numeric descriptors as additional features (FP bits + descriptors) for QSAR
    append_numeric_descriptors: bool = Setting(False)

    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[Table] = None
        self.molecules: Optional[List[ChemMol]] = None
        self._table_report: Optional[TableMolConversionReport] = None

        self._worker: Optional[_FPWorker] = None
        self._pending_start = False

        self.mainArea.hide()

        root = QWidget(self.controlArea)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        self.controlArea.layout().addWidget(root)

        # --- info box ---
        info_box = QGroupBox("Status")
        info_l = QVBoxLayout(info_box)
        self.lbl_info = QLabel("Waiting for input…")
        self.lbl_info.setWordWrap(True)
        info_l.addWidget(self.lbl_info)
        layout.addWidget(info_box)

        # --- input selection ---
        inp_box = QGroupBox("Input")
        inp_form = QFormLayout(inp_box)
        inp_form.setLabelAlignment(Qt.AlignLeft)
        inp_form.setFormAlignment(Qt.AlignTop)

        self.cmb_smiles = QComboBox()
        self.cmb_smiles.currentTextChanged.connect(self._on_smiles_changed)
        inp_form.addRow("SMILES column (Table):", self.cmb_smiles)

        self.meta_list = QListWidget()
        self.meta_list.setSelectionMode(QListWidget.MultiSelection)
        self.meta_list.itemSelectionChanged.connect(self._on_meta_selection_changed)
        self.meta_list.setFixedHeight(120)
        inp_form.addRow("Keep meta columns:", self.meta_list)

        self.chk_output_molecules = QCheckBox("Also output Molecules (ChemMol) with joined props + fp schema")
        self.chk_output_molecules.setChecked(bool(self.output_molecules))
        self.chk_output_molecules.stateChanged.connect(self._on_output_molecules_changed)
        inp_form.addRow("", self.chk_output_molecules)

        self.chk_store_onbits = QCheckBox("Store fp.onbits (interpretable, larger)")
        self.chk_store_onbits.setChecked(bool(self.store_onbits))
        self.chk_store_onbits.stateChanged.connect(self._on_store_onbits_changed)
        inp_form.addRow("", self.chk_store_onbits)


        self.chk_include_input_cols = QCheckBox("Include input columns in output Table (kept as meta columns)")
        self.chk_include_input_cols.setChecked(bool(self.include_input_columns))
        self.chk_include_input_cols.stateChanged.connect(self._on_include_input_cols_changed)
        inp_form.addRow("", self.chk_include_input_cols)

        self.chk_append_numeric = QCheckBox("Append numeric descriptors as features (FP + descriptors) for QSAR")
        self.chk_append_numeric.setChecked(bool(self.append_numeric_descriptors))
        self.chk_append_numeric.stateChanged.connect(self._on_append_numeric_descriptors_changed)
        inp_form.addRow("", self.chk_append_numeric)

        self.lbl_hint = QLabel("Tip: If you connect Molecules input, SMILES column selection is ignored.")
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet("color: #666;")
        inp_form.addRow(self.lbl_hint)

        layout.addWidget(inp_box)

        # --- fingerprint params ---
        fp_box = QGroupBox("Fingerprint")
        fp_form = QFormLayout(fp_box)

        self.cmb_fp = QComboBox()
        self.cmb_fp.addItems(["Morgan", "RDKit", "MACCS"])
        self.cmb_fp.setCurrentIndex(int(self.fp_type_idx))
        self.cmb_fp.currentIndexChanged.connect(self._on_fp_changed)
        fp_form.addRow("Type:", self.cmb_fp)

        self.spin_bits = QSpinBox()
        self.spin_bits.setRange(128, 8192)
        self.spin_bits.setSingleStep(128)
        self.spin_bits.setValue(int(self.bit_size))
        self.spin_bits.valueChanged.connect(lambda v: setattr(self, "bit_size", int(v)))
        fp_form.addRow("Bit size:", self.spin_bits)

        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(1, 6)
        self.spin_radius.setSingleStep(1)
        self.spin_radius.setValue(int(self.radius))
        self.spin_radius.valueChanged.connect(lambda v: setattr(self, "radius", int(v)))
        fp_form.addRow("Radius (Morgan):", self.spin_radius)

        self.chk_sanitize = QCheckBox("Sanitize SMILES")
        self.chk_sanitize.setChecked(bool(self.sanitize))
        self.chk_sanitize.stateChanged.connect(self._on_sanitize_changed)
        fp_form.addRow("", self.chk_sanitize)

        layout.addWidget(fp_box)

        # --- run buttons ---
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Compute")
        self.btn_run.clicked.connect(self._start)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel)

        self.btn_run.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_cancel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        layout.addStretch(1)

        self._on_fp_changed()
        self._update_buttons()

    # ---------------- inputs ----------------

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self.data = data
        self._table_report = None
        self.Outputs.fingerprints.send(None)
        self.Outputs.molecules.send(None)

        if data is None:
            self.lbl_info.setText("No Table input.")
            self._fill_smiles_candidates([])
            self._fill_meta_list([])
            self._update_buttons()
            return

        auto = _find_smiles_var(data)
        smiles_candidates = self._string_vars_in_table(data)
        self._fill_smiles_candidates([v.name for v in smiles_candidates])

        if auto is not None:
            self.smiles_var_name = auto.name
        elif smiles_candidates:
            self.smiles_var_name = smiles_candidates[0].name

        if self.smiles_var_name:
            idx = self.cmb_smiles.findText(self.smiles_var_name)
            if idx >= 0:
                self.cmb_smiles.setCurrentIndex(idx)

        meta_names = [v.name for v in data.domain.metas]
        self._fill_meta_list(meta_names)
        if not self.keep_meta_names:
            self.keep_meta_names = meta_names
            self._select_meta_items(self.keep_meta_names)

        self._refresh_table_summary()
        self._update_buttons()

    @Inputs.molecules
    def set_molecules(self, mols: Optional[list]) -> None:
        self.molecules = mols if mols else None
        self.Outputs.fingerprints.send(None)
        self.Outputs.molecules.send(None)

        if self.molecules is None:
            self._refresh_table_summary()
        else:
            self.lbl_info.setText(f"Molecules input: {len(self.molecules)} molecules (Table selection ignored).")

        self._update_buttons()

    # ---------------- UI helpers ----------------

    def _string_vars_in_table(self, data: Table) -> List[Variable]:
        return _text_like_vars(data)

    def _fill_smiles_candidates(self, names: Sequence[str]) -> None:
        self.cmb_smiles.blockSignals(True)
        self.cmb_smiles.clear()
        self.cmb_smiles.addItems(list(names))
        if self.smiles_var_name:
            idx = self.cmb_smiles.findText(self.smiles_var_name)
            if idx >= 0:
                self.cmb_smiles.setCurrentIndex(idx)
        self.cmb_smiles.blockSignals(False)

    def _fill_meta_list(self, meta_names: Sequence[str]) -> None:
        self.meta_list.blockSignals(True)
        self.meta_list.clear()
        for name in meta_names:
            self.meta_list.addItem(QListWidgetItem(name))
        self._select_meta_items(self.keep_meta_names or [])
        self.meta_list.blockSignals(False)

    def _select_meta_items(self, names: Sequence[str]) -> None:
        wanted = set(names)
        for i in range(self.meta_list.count()):
            it = self.meta_list.item(i)
            it.setSelected(it.text() in wanted)

    def _on_smiles_changed(self, txt: str) -> None:
        self.smiles_var_name = txt
        self._refresh_table_summary()

    def _on_meta_selection_changed(self) -> None:
        self.keep_meta_names = [it.text() for it in self.meta_list.selectedItems()]

    def _on_output_molecules_changed(self, state: int) -> None:
        self.output_molecules = bool(state == Qt.Checked)

        # When outputting molecules with joined props, most users also expect
        # the output Table to keep the original columns for inspection/debugging.
        if self.output_molecules and not self.include_input_columns:
            self.include_input_columns = True
            self.chk_include_input_cols.setChecked(True)

        if not self.output_molecules:
            self.Outputs.molecules.send(None)

    def _on_store_onbits_changed(self, state: int) -> None:
        self.store_onbits = bool(state == Qt.Checked)

    def _on_include_input_cols_changed(self, state: int) -> None:
        self.include_input_columns = bool(state == Qt.Checked)

    def _on_append_numeric_descriptors_changed(self, state: int) -> None:
        self.append_numeric_descriptors = bool(state == Qt.Checked)

    def _on_sanitize_changed(self, state: int) -> None:
        self.sanitize = bool(state == Qt.Checked)
        self._refresh_table_summary()

    def _refresh_table_summary(self) -> None:
        if self.molecules is not None:
            self.lbl_info.setText(
                f"Molecules input: {len(self.molecules)} molecules (Table selection ignored)."
            )
            return

        if self.data is None:
            self._table_report = None
            self.lbl_info.setText("No Table input.")
            return

        try:
            _mols, report = table_to_chemmols_with_report(
                self.data,
                smiles_var=self.smiles_var_name or None,
                sanitize=bool(self.sanitize),
            )
            self._table_report = report
            self.lbl_info.setText(
                format_table_report(
                    report,
                    prefix="Table",
                    valid_label="valid SMILES",
                )
            )
        except Exception:
            self._table_report = None
            self.lbl_info.setText(f"Table input rows: {len(self.data)}")

    def _on_fp_changed(self) -> None:
        self.fp_type_idx = int(self.cmb_fp.currentIndex())
        self.spin_radius.setEnabled(self.fp_type_idx == 0)  # radius only relevant for Morgan
        self.spin_bits.setEnabled(self.fp_type_idx != 2)    # MACCS fixed length

    def _update_buttons(self) -> None:
        busy = self._worker is not None and self._worker.isRunning()
        has_input = (
            (self.molecules is not None and len(self.molecules) > 0)
            or (self.data is not None and len(self.data) > 0)
        )
        self.btn_run.setEnabled(has_input and not busy)
        self.btn_cancel.setEnabled(busy)

    # ---------------- run ----------------

    def _cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._pending_start = False
            self._worker.requestInterruption()
            self.lbl_info.setText("Cancelling fingerprint computation…")
            self._update_buttons()
            return
        self._worker = None
        self.progressBarFinished()
        self._update_buttons()

    def _start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._pending_start = True
            self._worker.requestInterruption()
            self.lbl_info.setText("Cancelling current fingerprint computation before restarting…")
            self._update_buttons()
            return

        clear_widget_messages(self)
        self.Outputs.fingerprints.send(None)
        self.Outputs.molecules.send(None)
        self._pending_start = False

        source_table: Optional[Table] = None
        source_molecules: Optional[List[ChemMol]] = None

        if self.molecules is not None and len(self.molecules) > 0:
            source_molecules = list(self.molecules)
            smiles = [_object_to_smiles(obj) for obj in source_molecules]
        else:
            if self.data is None or len(self.data) == 0:
                set_widget_warning(self, "No input.")
                return

            var = self._resolve_smiles_var(self.data, self.smiles_var_name)
            if var is None:
                set_widget_error(self, "Cannot find a valid SMILES column in Table.")
                return

            source_table = self.data
            smiles = _col_as_str_list(self.data, var)

        fp_map = {0: "morgan", 1: "rdkit", 2: "maccs"}
        fp_type = fp_map.get(int(self.fp_type_idx), "morgan")

        spec = _FPJobSpec(
            fp_type=fp_type,
            bit_size=int(self.spin_bits.value()),
            radius=int(self.spin_radius.value()),
            sanitize=bool(self.chk_sanitize.isChecked()),
        )

        self.progressBarInit()
        self.lbl_info.setText("Computing fingerprints…")
        self._update_buttons()

        self._worker = _FPWorker(smiles=smiles, spec=spec)
        worker = self._worker
        worker.finished.connect(
            lambda payload, worker=worker: self._on_finished(worker, payload, source_table, source_molecules, spec)
        )
        worker.failed.connect(lambda msg, worker=worker: self._on_failed(worker, msg))
        worker.cancelled.connect(lambda worker=worker: self._on_cancelled(worker))
        worker.progress.connect(self.progressBarSet)
        self._worker.start()

    def _resolve_smiles_var(self, data: Table, name: str) -> Optional[Variable]:
        if not name:
            return _find_smiles_var(data)
        for v in list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars):
            if isinstance(v, (StringVariable, DiscreteVariable)) and v.name == name:
                return v
        return _find_smiles_var(data)

    def _ensure_chemmol(self, obj: Any) -> Optional[ChemMol]:
        """Best-effort conversion to ChemMol (used when Molecules input is not strictly ChemMol)."""
        if isinstance(obj, ChemMol):
            return obj
        if isinstance(obj, str):
            try:
                return ChemMol.from_smiles(obj)
            except Exception:
                return None
        try:
            from rdkit.Chem.rdchem import Mol  # type: ignore
            if isinstance(obj, Mol):
                return ChemMol.from_rdkit(obj)
        except Exception:
            pass
        return None

    def _restart_if_pending(self) -> None:
        if self._pending_start:
            self._pending_start = False
            QTimer.singleShot(0, self._start)

    def _on_cancelled(self, worker: _FPWorker) -> None:
        if worker is not self._worker:
            return
        self.progressBarFinished()
        self._worker = None
        self.lbl_info.setText("Fingerprint computation cancelled.")
        self._update_buttons()
        self._restart_if_pending()

    def _on_failed(self, worker: _FPWorker, msg: str) -> None:
        if worker is not self._worker:
            return
        self.progressBarFinished()
        self._worker = None
        set_widget_error(self, f"Fingerprint computation failed: {msg}")
        self.lbl_info.setText(format_error_status(msg))
        self._update_buttons()
        self._restart_if_pending()

    def _on_finished(
        self,
        worker: _FPWorker,
        payload,
        source_table: Optional[Table],
        source_molecules: Optional[List[ChemMol]],
        spec: _FPJobSpec,
    ) -> None:
        if worker is not self._worker:
            return
        self.progressBarFinished()
        self._worker = None

        res, valid_smiles = payload

        if res.X.shape[0] == 0:
            set_widget_warning(self, "No valid molecules.")
            self.Outputs.fingerprints.send(None)
            self.Outputs.molecules.send(None)
            self.lbl_info.setText("No valid molecules.")
            self._update_buttons()
            return

        # ---------- build output Table ----------
        used: set[str] = set()

        names = list(res.bit_names) if getattr(res, "bit_names", None) else [f"FP_{i:04d}" for i in range(res.X.shape[1])]
        attrs = [ContinuousVariable(_unique_name(str(n), used)) for n in names]

        # Optional QSAR mode: append numeric descriptor columns as additional features
        desc_attrs: List[ContinuousVariable] = []
        desc_cols: List[np.ndarray] = []

        inchikeys = [_inchikey_from_smiles(smi) for smi in valid_smiles]
        metas: List[Any] = [
            StringVariable(_unique_name("SMILES", used)),
            StringVariable(_unique_name("inchikey", used)),
        ]
        meta_cols: List[np.ndarray] = [
            np.array(valid_smiles, dtype=object).reshape(-1, 1),
            np.array(inchikeys, dtype=object).reshape(-1, 1),
        ]

        class_vars = []
        Y = None

        if source_table is not None:
            valid_rows = source_table[res.valid_indices]

            smiles_name = (self.smiles_var_name or "").strip()
            desc_src_names: set[str] = set()
            if self.append_numeric_descriptors:
                for v in valid_rows.domain.attributes:
                    if smiles_name and v.name == smiles_name:
                        continue
                    if isinstance(v, ContinuousVariable):
                        out_v = ContinuousVariable(_unique_name(v.name, used))
                        desc_attrs.append(out_v)
                        col = np.asarray(valid_rows.get_column(v), dtype=np.float64).reshape(-1, 1)
                        desc_cols.append(col)
                        desc_src_names.add(v.name)

            def _clone_var_as_meta(var):
                name = _unique_name(var.name, used)
                if isinstance(var, StringVariable):
                    return StringVariable(name)
                if isinstance(var, ContinuousVariable):
                    return ContinuousVariable(name)
                if isinstance(var, DiscreteVariable):
                    return DiscreteVariable(name, values=list(var.values))
                # Fallback: keep it readable
                return StringVariable(name)

            keep = set(self.keep_meta_names or [])
            for v in valid_rows.domain.metas:
                if self.include_input_columns or (v.name in keep):
                    mv = _clone_var_as_meta(v)
                    metas.append(mv)
                    col = valid_rows.get_column(v)
                    meta_cols.append(np.array(col, dtype=object).reshape(-1, 1))

            # If requested: also carry over *all* original attributes (e.g., ChEMBL activity/descriptors)
            if self.include_input_columns:
                smiles_name = (self.smiles_var_name or "").strip()
                for v in valid_rows.domain.attributes:
                    # skip the SMILES column if it lives among attributes; we already output canonical SMILES
                    if smiles_name and v.name == smiles_name:
                        continue
                    # In QSAR mode, numeric descriptors are appended as features, so don't duplicate them as metas
                    if self.append_numeric_descriptors and v.name in desc_src_names:
                        continue
                    mv = _clone_var_as_meta(v)
                    metas.append(mv)
                    col = valid_rows.get_column(v)
                    meta_cols.append(np.array(col, dtype=object).reshape(-1, 1))

            # Preserve class variables (QSAR targets) if present
            if len(valid_rows.domain.class_vars) > 0:
                class_vars = list(valid_rows.domain.class_vars)
                Y = valid_rows.Y

        elif (source_molecules is not None) and (self.include_input_columns or self.append_numeric_descriptors):
            # Carry scalar ChemMol.props into the output table (as meta columns)
            valid_objs = [source_molecules[i] for i in res.valid_indices]
            chems: List[Optional[ChemMol]] = [
                obj if isinstance(obj, ChemMol) else self._ensure_chemmol(obj) for obj in valid_objs
            ]

            # Collect keys with scalar values
            keys: List[str] = []
            seen = set()
            rows_props: List[Dict[str, Any]] = []
            for cm in chems:
                props = dict(getattr(cm, "props", {}) or {}) if cm is not None else {}
                # Avoid duplicating/ballooning: skip stored fingerprints (bytes) if already present
                props.pop("fp", None)
                rows_props.append(props)
                for k, v in props.items():
                    if k in seen:
                        continue
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        seen.add(k)
                        keys.append(k)

            keys.sort()

            def _is_number(x: Any) -> bool:
                return isinstance(x, (int, float)) and not isinstance(x, bool)

            for k in keys:
                col_vals = [rp.get(k, None) for rp in rows_props]
                is_numeric_col = all((v is None) or _is_number(v) for v in col_vals)

                # QSAR mode: put numeric scalar props into X (features), not metas
                if is_numeric_col and self.append_numeric_descriptors:
                    desc_attrs.append(ContinuousVariable(_unique_name(k, used)))
                    col_f = np.array([np.nan if v is None else float(v) for v in col_vals], dtype=np.float64).reshape(-1, 1)
                    desc_cols.append(col_f)
                    continue

                if self.include_input_columns:
                    if is_numeric_col:
                        metas.append(ContinuousVariable(_unique_name(k, used)))
                    else:
                        metas.append(StringVariable(_unique_name(k, used)))
                    meta_cols.append(np.array(col_vals, dtype=object).reshape(-1, 1))

        X_fp = res.X.astype(np.float64, copy=False)
        if desc_cols:
            X_desc = np.hstack(desc_cols).astype(np.float64, copy=False)
            X = np.hstack([X_fp, X_desc])
        else:
            X = X_fp

        out_domain = Domain(attributes=attrs + desc_attrs, class_vars=class_vars, metas=metas)
        M = np.hstack(meta_cols) if meta_cols else None
        out = Table.from_numpy(out_domain, X=X, Y=Y, metas=M)

        self.Outputs.fingerprints.send(out)

        # ---------- optionally emit Molecules with joined props + fp schema ----------
        if self.output_molecules:
            nbits = int(res.X.shape[1])
            fp_type = str(res.fp_type).lower().strip()

            # create / filter molecules in the same order as fingerprint rows
            mols_out: List[ChemMol] = []

            if source_molecules is not None:
                for idx in res.valid_indices:
                    cm = source_molecules[idx]
                    if isinstance(cm, ChemMol):
                        mols_out.append(_copy_chemmol(cm))

            elif source_table is not None:
                valid_rows = source_table[res.valid_indices]
                try:
                    mols_out, mol_report = table_to_chemmols_with_report(
                        valid_rows,
                        smiles_var=self.smiles_var_name or None,
                        sanitize=bool(self.chk_sanitize.isChecked()),
                    )
                    if mol_report.n_invalid:
                        set_widget_warning(
                            self,
                            f"Molecules output skipped {mol_report.n_invalid} rows after fingerprint filtering.",
                        )
                except Exception as e:
                    set_widget_warning(self, f"Could not build Molecules output: {e}")
                    mols_out = []

            # Attach fingerprint schema
            n_attach = min(len(mols_out), int(res.X.shape[0]))
            if n_attach < int(res.X.shape[0]):
                set_widget_warning(
                    self,
                    "Molecules output length mismatch; attached fingerprints to available molecules only.",
                )

            for i in range(n_attach):
                row_bits = (res.X[i] > 0.5).astype(np.uint8)
                fp_bytes = _fp_bytes_from_row(row_bits, nbits=nbits)
                onbits = np.flatnonzero(row_bits).astype(int).tolist() if self.store_onbits else None
                _attach_fp_props(
                    mols_out[i],
                    fp_type=fp_type,
                    nbits=nbits,
                    radius=int(spec.radius),
                    fp_bytes=fp_bytes,
                    onbits=onbits,
                )

            self.Outputs.molecules.send(mols_out if mols_out else None)
        else:
            self.Outputs.molecules.send(None)

        if source_table is not None and self._table_report is not None:
            base_status = (
                f"Computed {out.X.shape[0]} fingerprints from table "
                f"({self._table_report.n_valid}/{self._table_report.n_rows} valid SMILES)"
            )
        elif source_molecules is not None:
            base_status = (
                f"Computed {out.X.shape[0]} fingerprints from molecules "
                f"(input: {len(source_molecules)})"
            )
        else:
            base_status = f"Computed {out.X.shape[0]} valid fingerprints"

        self.lbl_info.setText(
            base_status
            + f"; failed={len(res.failed_indices)}"
            + (f"; descriptors={len(desc_attrs)}" if desc_attrs else "")
            + ("; Molecules output=ON" if self.output_molecules else "")
        )
        self._update_buttons()
        self._restart_if_pending()

    def onDeleteWidget(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(2000)
        super().onDeleteWidget()
