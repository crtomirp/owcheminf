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
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table, Variable
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.rdkit_safe import (
    safe_canonical_smiles,
    safe_mol_from_smiles,
    safe_mol_to_inchikey,
)
from chem_inf_widgets.chemcore.descriptors.cyclic_registry_fingerprint import (
    BIT_SECTIONS,
    DEFAULT_N_BITS,
    compute_cyclic_registry_fingerprints_from_smiles,
)
from chem_inf_widgets.widgets.ui_helpers import clear_widget_messages, set_widget_error, set_widget_warning


# ---------------------------- helpers ----------------------------

def _unique_name(name: str, used: set[str]) -> str:
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
    candidates = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    return [v for v in candidates if isinstance(v, (StringVariable, DiscreteVariable))]


def _find_smiles_var(data: Table) -> Optional[Variable]:
    wanted = {"smiles", "canonical_smiles", "smile"}
    candidates = _text_like_vars(data)
    for v in candidates:
        if v.name.strip().lower() in wanted:
            return v
    return candidates[0] if candidates else None


def _find_name_var(data: Table, smiles_var: Optional[Variable]) -> Optional[Variable]:
    candidates = _text_like_vars(data)
    preferred = {
        "name",
        "title",
        "compound",
        "compound_name",
        "id",
        "ime",
        "ime spojine",
    }
    for v in candidates:
        if smiles_var is not None and v.name == smiles_var.name:
            continue
        if v.name.strip().lower() in preferred:
            return v
    for v in candidates:
        if smiles_var is None or v.name != smiles_var.name:
            return v
    return None


def _inchikey_from_smiles(smiles: str) -> str:
    """Return a stable InChIKey for a SMILES string, or an empty string on failure."""
    parsed = safe_mol_from_smiles((smiles or "").strip(), sanitize=True, remove_hs=True)
    return safe_mol_to_inchikey(parsed.mol) if parsed.ok else ""


def _col_as_str_list(data: Table, var: Variable) -> List[str]:
    out: List[str] = []
    for row in data:
        try:
            value = row[var]
        except Exception:
            value = None
        out.append("" if value is None else str(value).strip())
    return out


def _cell_to_python_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    text = str(value)
    if text == "?":
        return None
    return value


def _chemmol_to_smiles(cm: ChemMol) -> str:
    if getattr(cm, "props", None):
        s = cm.props.get("SMILES") or cm.props.get("smiles") or cm.props.get("canonical_smiles")
        if isinstance(s, str) and s.strip():
            return s.strip()
    s2 = getattr(cm, "smiles", None)
    if isinstance(s2, str) and s2.strip():
        return s2.strip()
    try:
        return safe_canonical_smiles(cm.mol, remove_hs=False, canonical=True, isomeric=True) if cm.mol is not None else ""
    except Exception:
        return ""


def _object_to_smiles(obj: Any) -> str:
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, ChemMol):
        return _chemmol_to_smiles(obj)
    try:
        from rdkit.Chem.rdchem import Mol  # type: ignore
        if isinstance(obj, Mol):
            return safe_canonical_smiles(obj, remove_hs=False, canonical=True, isomeric=True)
    except Exception:
        pass
    return ""


def _copy_chemmol(cm: ChemMol) -> ChemMol:
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


def _chemmol_from_table_row(
    row: Any,
    *,
    canonical_smiles: str,
    row_index: int,
    all_vars: Sequence[Variable],
    smiles_var: Optional[Variable],
    name_var: Optional[Variable],
) -> ChemMol:
    cm = ChemMol.from_smiles(canonical_smiles)
    props: Dict[str, Any] = {"SMILES": canonical_smiles, "source_row_index": row_index}
    for var in all_vars:
        if smiles_var is not None and var.name == smiles_var.name:
            continue
        if name_var is not None and var.name == name_var.name:
            continue
        try:
            value = _cell_to_python_value(row[var])
        except Exception:
            value = None
        if value is None:
            continue
        props[var.name] = value
    cm.props.update(props)
    if name_var is not None:
        try:
            raw_name = row[name_var]
        except Exception:
            raw_name = None
        if raw_name is not None:
            name = str(raw_name).strip()
            if name and name != "?":
                cm.name = name
    return cm


def _pack_bits(bits01: np.ndarray, nbits: int = DEFAULT_N_BITS) -> bytes:
    packed = np.packbits(np.asarray(bits01, dtype=np.uint8).reshape(-1), bitorder="little")
    need = (int(nbits) + 7) // 8
    if packed.size < need:
        packed = np.pad(packed, (0, need - packed.size), constant_values=0)
    elif packed.size > need:
        packed = packed[:need]
    return packed.tobytes()


def _attach_registry_fp(cm: ChemMol, row_bits: np.ndarray, *, registry_version: str, params: Dict[str, Any], store_onbits: bool) -> None:
    if getattr(cm, "props", None) is None:
        cm.props = {}
    onbits = np.flatnonzero(row_bits > 0.5).astype(int).tolist()
    cm.props["fp"] = {
        "type": "cyclic_registry_4096",
        "nbits": DEFAULT_N_BITS,
        "bytes": _pack_bits(row_bits, DEFAULT_N_BITS),
        "schema_version": "2.0",
        "fingerprint_version": params.get("fingerprint_version", "0.2.0"),
        "registry_version": registry_version,
        "sections": dict(BIT_SECTIONS),
        "onbits": onbits if store_onbits else None,
    }


@dataclass(frozen=True)
class _CRFPJobSpec:
    sanitize: bool
    include_morgan: bool
    include_atom_matches: bool
    max_registry_entries: int


class _CRFPWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, smiles: List[str], spec: _CRFPJobSpec) -> None:
        super().__init__()
        self._smiles = smiles
        self._spec = spec

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
            max_entries = int(self._spec.max_registry_entries)
            res = compute_cyclic_registry_fingerprints_from_smiles(
                self._smiles,
                include_morgan=bool(self._spec.include_morgan),
                sanitize=bool(self._spec.sanitize),
                max_registry_entries=max_entries if max_entries > 0 else None,
                include_atom_matches=bool(self._spec.include_atom_matches),
                progress_cb=self.progress.emit,
                cancel_cb=self.isInterruptionRequested,
            )
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
            self.finished.emit(res)
        except Exception as exc:
            self.failed.emit(str(exc))


class OWCyclicRegistryFingerprint(OWWidget):
    name = "Cyclic Registry Fingerprint"
    description = "Compute an interpretable 4096-bit cyclic/heterocycle registry fingerprint."
    icon = "icons/descriptors/owheterocyclicfingerprintwidget.png"
    priority = 132

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        fingerprints = Output("Fingerprints", Table)
        matches = Output("Matched Registry Entries", Table)
        molecules = Output("Molecules", list, auto_summary=False)

    smiles_var_name: str = Setting("")
    sanitize: bool = Setting(True)
    include_morgan: bool = Setting(True)
    include_atom_matches: bool = Setting(True)
    output_molecules: bool = Setting(False)
    store_onbits: bool = Setting(False)
    max_registry_entries: int = Setting(0)  # 0 = all

    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[Table] = None
        self.molecules_in: Optional[List[Any]] = None
        self._worker: Optional[_CRFPWorker] = None
        self._pending_start = False

        self.mainArea.hide()
        root = QWidget(self.controlArea)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        self.controlArea.layout().addWidget(root)

        info_box = QGroupBox("Status")
        info_l = QVBoxLayout(info_box)
        self.lbl_info = QLabel("No input.")
        self.lbl_info.setWordWrap(True)
        info_l.addWidget(self.lbl_info)
        layout.addWidget(info_box)

        input_box = QGroupBox("Input")
        input_form = QFormLayout(input_box)
        self.cmb_smiles = QComboBox()
        self.cmb_smiles.currentTextChanged.connect(self._on_smiles_changed)
        input_form.addRow("SMILES column", self.cmb_smiles)
        layout.addWidget(input_box)

        fp_box = QGroupBox("Cyclic Registry Fingerprint 4096")
        fp_form = QFormLayout(fp_box)
        self.chk_sanitize = QCheckBox("Sanitize SMILES")
        self.chk_sanitize.setChecked(bool(self.sanitize))
        self.chk_sanitize.stateChanged.connect(lambda s: setattr(self, "sanitize", bool(s == Qt.Checked)))
        fp_form.addRow("", self.chk_sanitize)

        self.chk_morgan = QCheckBox("Include general Morgan section, bits 0–2047")
        self.chk_morgan.setChecked(bool(self.include_morgan))
        self.chk_morgan.stateChanged.connect(lambda s: setattr(self, "include_morgan", bool(s == Qt.Checked)))
        fp_form.addRow("", self.chk_morgan)

        self.chk_atom_matches = QCheckBox("Store atom match indices in explanation table")
        self.chk_atom_matches.setChecked(bool(self.include_atom_matches))
        self.chk_atom_matches.stateChanged.connect(lambda s: setattr(self, "include_atom_matches", bool(s == Qt.Checked)))
        fp_form.addRow("", self.chk_atom_matches)

        self.spin_max_entries = QSpinBox()
        self.spin_max_entries.setRange(0, 100000)
        self.spin_max_entries.setValue(int(self.max_registry_entries))
        self.spin_max_entries.setSpecialValueText("All packaged registry entries")
        self.spin_max_entries.valueChanged.connect(lambda v: setattr(self, "max_registry_entries", int(v)))
        fp_form.addRow("Registry entry limit", self.spin_max_entries)

        layout.addWidget(fp_box)

        out_box = QGroupBox("Outputs")
        out_l = QVBoxLayout(out_box)
        self.chk_output_molecules = QCheckBox("Output Molecules with attached fp schema")
        self.chk_output_molecules.setChecked(bool(self.output_molecules))
        self.chk_output_molecules.stateChanged.connect(self._on_output_molecules_changed)
        out_l.addWidget(self.chk_output_molecules)
        self.chk_store_onbits = QCheckBox("Store onbits in Molecules output")
        self.chk_store_onbits.setChecked(bool(self.store_onbits))
        self.chk_store_onbits.stateChanged.connect(lambda s: setattr(self, "store_onbits", bool(s == Qt.Checked)))
        out_l.addWidget(self.chk_store_onbits)
        layout.addWidget(out_box)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Compute")
        self.btn_run.clicked.connect(self._start)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)
        layout.addStretch(1)
        self._update_buttons()

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self.data = data
        self.Outputs.fingerprints.send(None)
        self.Outputs.matches.send(None)
        self.Outputs.molecules.send(None)
        if data is None:
            self.lbl_info.setText("No Table input.")
            self._fill_smiles_candidates([])
        else:
            candidates = _text_like_vars(data)
            self._fill_smiles_candidates([v.name for v in candidates])
            auto = _find_smiles_var(data)
            if auto is not None:
                self.smiles_var_name = auto.name
                idx = self.cmb_smiles.findText(auto.name)
                if idx >= 0:
                    self.cmb_smiles.setCurrentIndex(idx)
            self.lbl_info.setText(f"Table input: {len(data)} rows. 4096-bit fingerprint ready.")
        self._update_buttons()

    @Inputs.molecules
    def set_molecules(self, mols: Optional[list]) -> None:
        self.molecules_in = mols if mols else None
        self.Outputs.fingerprints.send(None)
        self.Outputs.matches.send(None)
        self.Outputs.molecules.send(None)
        if self.molecules_in is None:
            if self.data is None:
                self.lbl_info.setText("No input.")
        else:
            self.lbl_info.setText(f"Molecules input: {len(self.molecules_in)} molecules. Table input ignored.")
        self._update_buttons()

    def _fill_smiles_candidates(self, names: Sequence[str]) -> None:
        self.cmb_smiles.blockSignals(True)
        self.cmb_smiles.clear()
        self.cmb_smiles.addItems(list(names))
        if self.smiles_var_name:
            idx = self.cmb_smiles.findText(self.smiles_var_name)
            if idx >= 0:
                self.cmb_smiles.setCurrentIndex(idx)
        self.cmb_smiles.blockSignals(False)

    def _on_smiles_changed(self, txt: str) -> None:
        self.smiles_var_name = txt

    def _on_output_molecules_changed(self, state: int) -> None:
        self.output_molecules = bool(state == Qt.Checked)
        if not self.output_molecules:
            self.Outputs.molecules.send(None)

    def _update_buttons(self) -> None:
        busy = self._worker is not None and self._worker.isRunning()
        has_input = (self.molecules_in is not None and len(self.molecules_in) > 0) or (self.data is not None and len(self.data) > 0)
        self.btn_run.setEnabled(has_input and not busy)
        self.btn_cancel.setEnabled(busy)

    def _resolve_smiles_var(self, data: Table, name: str) -> Optional[Variable]:
        if name:
            for v in list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars):
                if isinstance(v, (StringVariable, DiscreteVariable)) and v.name == name:
                    return v
        return _find_smiles_var(data)

    def _cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._pending_start = False
            self._worker.requestInterruption()
            self.lbl_info.setText("Cancelling cyclic registry fingerprint computation…")
            self._update_buttons()
            return
        self._worker = None
        self.progressBarFinished()
        self._update_buttons()

    def _start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._pending_start = True
            self._worker.requestInterruption()
            self.lbl_info.setText("Cancelling current computation before restart…")
            self._update_buttons()
            return

        clear_widget_messages(self)
        self.Outputs.fingerprints.send(None)
        self.Outputs.matches.send(None)
        self.Outputs.molecules.send(None)
        self._pending_start = False

        source_table: Optional[Table] = None
        source_molecules: Optional[List[Any]] = None
        if self.molecules_in is not None and len(self.molecules_in) > 0:
            source_molecules = list(self.molecules_in)
            smiles = [_object_to_smiles(obj) for obj in source_molecules]
        else:
            if self.data is None or len(self.data) == 0:
                set_widget_warning(self, "No input.")
                return
            var = self._resolve_smiles_var(self.data, self.smiles_var_name)
            if var is None:
                set_widget_error(self, "Cannot find a SMILES column.")
                return
            source_table = self.data
            smiles = _col_as_str_list(self.data, var)

        spec = _CRFPJobSpec(
            sanitize=bool(self.chk_sanitize.isChecked()),
            include_morgan=bool(self.chk_morgan.isChecked()),
            include_atom_matches=bool(self.chk_atom_matches.isChecked()),
            max_registry_entries=int(self.spin_max_entries.value()),
        )
        self.progressBarInit()
        self.lbl_info.setText("Computing 4096-bit cyclic registry fingerprints…")
        self._update_buttons()

        self._worker = _CRFPWorker(smiles, spec)
        worker = self._worker
        worker.finished.connect(lambda res, worker=worker: self._on_finished(worker, res, source_table, source_molecules))
        worker.failed.connect(lambda msg, worker=worker: self._on_failed(worker, msg))
        worker.cancelled.connect(lambda worker=worker: self._on_cancelled(worker))
        worker.progress.connect(self.progressBarSet)
        self._worker.start()

    def _restart_if_pending(self) -> None:
        if self._pending_start:
            self._pending_start = False
            QTimer.singleShot(0, self._start)

    def _on_cancelled(self, worker: _CRFPWorker) -> None:
        if worker is not self._worker:
            return
        self.progressBarFinished()
        self._worker = None
        self.lbl_info.setText("Cyclic registry fingerprint computation cancelled.")
        self._update_buttons()
        self._restart_if_pending()

    def _on_failed(self, worker: _CRFPWorker, msg: str) -> None:
        if worker is not self._worker:
            return
        self.progressBarFinished()
        self._worker = None
        set_widget_error(self, f"Cyclic registry fingerprint failed: {msg}")
        self.lbl_info.setText(f"Failed: {msg}")
        self._update_buttons()
        self._restart_if_pending()

    def _on_finished(self, worker: _CRFPWorker, res, source_table: Optional[Table], source_molecules: Optional[List[Any]]) -> None:
        if worker is not self._worker:
            return
        self.progressBarFinished()
        self._worker = None

        if res.X.shape[0] == 0:
            set_widget_warning(self, "No valid molecules.")
            self.Outputs.fingerprints.send(None)
            self.Outputs.matches.send(None)
            self.Outputs.molecules.send(None)
            self.lbl_info.setText("No valid molecules.")
            self._update_buttons()
            return

        fp_table = self._build_fingerprint_table(res, source_table)
        match_table = self._build_match_table(res)
        self.Outputs.fingerprints.send(fp_table)
        self.Outputs.matches.send(match_table)

        if self.output_molecules:
            self.Outputs.molecules.send(self._build_molecules_output(res, source_table, source_molecules))
        else:
            self.Outputs.molecules.send(None)

        n_matches = len(getattr(res, "matches", []) or [])
        err_suffix = f"; warnings/errors={len(res.errors)}" if getattr(res, "errors", None) else ""
        self.lbl_info.setText(
            f"Computed {res.X.shape[0]} × {res.X.shape[1]} cyclic registry fingerprints; "
            f"registry matches={n_matches}; failed molecules={len(res.failed_indices)}{err_suffix}."
        )
        if getattr(res, "errors", None):
            set_widget_warning(self, res.errors[0])
        self._update_buttons()
        self._restart_if_pending()

    def _build_fingerprint_table(self, res, source_table: Optional[Table]) -> Table:
        used: set[str] = set()
        attrs = [ContinuousVariable(_unique_name(str(n), used)) for n in res.bit_names]
        inchikeys = [_inchikey_from_smiles(smi) for smi in res.smiles]
        metas: List[Any] = [
            StringVariable(_unique_name("SMILES", used)),
            StringVariable(_unique_name("inchikey", used)),
        ]
        meta_cols: List[np.ndarray] = [
            np.array(res.smiles, dtype=object).reshape(-1, 1),
            np.array(inchikeys, dtype=object).reshape(-1, 1),
        ]
        class_vars = []
        Y = None

        if source_table is not None:
            valid_rows = source_table[res.valid_indices]
            smiles_name = (self.smiles_var_name or "").strip()

            def _clone(v: Variable) -> Variable:
                name = _unique_name(v.name, used)
                if isinstance(v, StringVariable):
                    return StringVariable(name)
                if isinstance(v, ContinuousVariable):
                    return ContinuousVariable(name)
                if isinstance(v, DiscreteVariable):
                    return DiscreteVariable(name, values=list(v.values))
                return StringVariable(name)

            # pass through all metas
            for v in valid_rows.domain.metas:
                metas.append(_clone(v))
                meta_cols.append(np.array(valid_rows.get_column(v), dtype=object).reshape(-1, 1))

            # pass through all attributes (skip the SMILES column itself)
            for v in valid_rows.domain.attributes:
                if smiles_name and v.name == smiles_name:
                    continue
                metas.append(_clone(v))
                meta_cols.append(np.array(valid_rows.get_column(v), dtype=object).reshape(-1, 1))

            # preserve class variables
            if len(valid_rows.domain.class_vars) > 0:
                class_vars = list(valid_rows.domain.class_vars)
                Y = valid_rows.Y

        domain = Domain(attrs, class_vars=class_vars, metas=metas)
        M = np.hstack(meta_cols)
        return Table.from_numpy(domain, X=res.X.astype(np.float64, copy=False), Y=Y, metas=M)

    def _build_match_table(self, res) -> Optional[Table]:
        matches = list(getattr(res, "matches", []) or [])
        if not matches:
            return None
        attrs = [
            ContinuousVariable("valid_row_index"),
            ContinuousVariable("source_row_index"),
            ContinuousVariable("bit"),
            ContinuousVariable("match_count"),
        ]
        metas = [
            StringVariable("SMILES"),
            StringVariable("inchikey"),
            StringVariable("entry_id"),
            StringVariable("name"),
            StringVariable("section"),
            StringVariable("family"),
            StringVariable("smarts"),
            StringVariable("atom_matches"),
        ]
        X = np.array(
            [
                [
                    m.row,
                    res.valid_indices[m.row] if 0 <= m.row < len(res.valid_indices) else -1,
                    m.bit,
                    m.match_count,
                ]
                for m in matches
            ],
            dtype=np.float64,
        )
        M = np.array(
            [
                [
                    res.smiles[m.row] if 0 <= m.row < len(res.smiles) else "",
                    _inchikey_from_smiles(res.smiles[m.row]) if 0 <= m.row < len(res.smiles) else "",
                    m.entry_id,
                    m.name,
                    m.section,
                    m.family,
                    m.smarts,
                    repr(m.atom_matches),
                ]
                for m in matches
            ],
            dtype=object,
        )
        return Table.from_numpy(Domain(attrs, metas=metas), X=X, metas=M)

    def _build_molecules_output(self, res, source_table: Optional[Table], source_molecules: Optional[List[Any]]) -> Optional[List[ChemMol]]:
        mols_out: List[ChemMol] = []
        if source_molecules is not None:
            for idx in res.valid_indices:
                obj = source_molecules[idx]
                if isinstance(obj, ChemMol):
                    mols_out.append(_copy_chemmol(obj))
                elif isinstance(obj, str):
                    try:
                        mols_out.append(ChemMol.from_smiles(obj))
                    except Exception:
                        pass
                else:
                    try:
                        from rdkit.Chem.rdchem import Mol  # type: ignore
                        if isinstance(obj, Mol):
                            mols_out.append(ChemMol.from_rdkit(obj))
                    except Exception:
                        pass
        elif source_table is not None:
            try:
                smiles_var = self._resolve_smiles_var(source_table, self.smiles_var_name)
                name_var = _find_name_var(source_table, smiles_var)
                all_vars = list(source_table.domain.metas) + list(source_table.domain.attributes) + list(source_table.domain.class_vars)
                for valid_pos, source_idx in enumerate(res.valid_indices):
                    row = source_table[source_idx]
                    mols_out.append(
                        _chemmol_from_table_row(
                            row,
                            canonical_smiles=res.smiles[valid_pos],
                            row_index=int(source_idx),
                            all_vars=all_vars,
                            smiles_var=smiles_var,
                            name_var=name_var,
                        )
                    )
            except Exception as exc:
                set_widget_warning(self, f"Could not build Molecules output: {exc}")
                mols_out = []
        if len(mols_out) != int(res.X.shape[0]):
            set_widget_warning(
                self,
                f"Molecules output alignment mismatch: molecules={len(mols_out)}, fingerprints={int(res.X.shape[0])}.",
            )
        n = min(len(mols_out), int(res.X.shape[0]))
        for i in range(n):
            params = dict(getattr(res, "params", {}) or {})
            params["fingerprint_version"] = getattr(res, "fingerprint_version", "0.2.0")
            _attach_registry_fp(
                mols_out[i],
                res.X[i],
                registry_version=str(getattr(res, "registry_version", "")),
                params=params,
                store_onbits=bool(self.store_onbits),
            )
        return mols_out if mols_out else None

    def onDeleteWidget(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(2000)
        super().onDeleteWidget()
