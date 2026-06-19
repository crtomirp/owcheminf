"""Orange widget: CmDock Docking.

Docks 3D ligands against a prepared CmDock receptor and outputs one row per
docked pose. The primary input is a ``Molecules`` list (3D ``ChemMol`` objects,
e.g. from SDF Reader); an optional ``Data`` table supplies extra per-ligand
columns that ride along onto every pose row. A table containing an SDF
text/path column can be docked on its own when no molecules are connected.

Outputs:
- ``Docked Poses`` — one row per pose. Every CmDock SDF property (``SCORE``,
  ``SCORE.INTER`` …) is expanded into its own column (numeric → continuous
  attribute, else string meta); the full pose mol block is kept in ``pose_sdf``.
- ``Molecules`` — the docked poses as ``ChemMol`` objects: the input molecule
  with its conformation **replaced by the docked pose geometry**, carrying the
  scores and pass-through data as properties.

The heavy lifting lives in the GUI-free service
:mod:`chem_inf_widgets.chemcore.services.cmdock_service`. CmDock must be
installed separately (https://gitlab.com/Jukic/cmdock).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QComboBox, QFileDialog, QPlainTextEdit
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.cmdock_service import (
    CORE_COLUMNS,
    DockingSettings,
    dock_dataframe,
    referenced_receptor_file,
)
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_error_status,
    set_widget_error,
    set_widget_warning,
)

# Fixed pose columns kept in the output table (the remaining CORE_COLUMNS are
# diagnostics or are superseded by the expanded SDF properties).
_OUT_NUMERIC = ("pose_index", "rank_by_score", "score", "cmdock_return_code")
_OUT_META = (
    "input_row_id",
    "ligand_name",
    "status",
    "score_field",
    "error_message",
    "pose_sdf",
    "result_sdf_path",
    "cmdock_command",
    "cmdock_stdout",
    "cmdock_stderr",
    "properties_json",
)
_INPUT_MODES = ("Auto", "SDF text", "SDF file path")


def _default_executable() -> str:
    root = os.environ.get("CMDOCK_ROOT")
    for candidate in ([os.path.join(root, "bin", "cmdock")] if root else []) + ["/opt/CmD/bin/cmdock"]:
        if os.path.isfile(candidate):
            return candidate
    return "cmdock"


def _default_library_dir() -> str:
    root = os.environ.get("CMDOCK_ROOT")
    for candidate in ([os.path.join(root, "lib")] if root else []) + ["/opt/CmD/lib"]:
        if os.path.isdir(candidate):
            return candidate
    return ""


def _is_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _unique(name: str, taken: set[str]) -> str:
    candidate, i = name, 2
    while candidate in taken:
        candidate, i = f"{name}_{i}", i + 1
    taken.add(candidate)
    return candidate


def build_poses_table(rows: List[dict], passthrough_names: List[str]) -> Optional[Table]:
    """Build an Orange Table from CmDock pose-row dicts.

    Numeric columns (fixed + numeric SDF properties + numeric pass-through) become
    continuous attributes; everything else becomes a string meta.
    """

    if not rows:
        return None

    # Union of SDF property keys, classified numeric vs. text.
    prop_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.get("_properties", {}):
            if key not in seen:
                seen.add(key)
                prop_keys.append(key)
    prop_keys.sort()

    def prop_is_numeric(key: str) -> bool:
        present = [r["_properties"][key] for r in rows if key in r.get("_properties", {})]
        return bool(present) and all(_is_number(v) for v in present)

    def col_is_numeric(name: str) -> bool:
        present = [r[name] for r in rows if r.get(name) not in (None, "")]
        return bool(present) and all(_is_number(v) for v in present)

    numeric_props = [k for k in prop_keys if prop_is_numeric(k)]
    text_props = [k for k in prop_keys if k not in numeric_props]
    numeric_pass = [n for n in passthrough_names if col_is_numeric(n)]
    text_pass = [n for n in passthrough_names if n not in numeric_pass]

    # The fixed "score" column mirrors the chosen ranking field; name it
    # explicitly (e.g. "ranking_score (SCORE.INTER)") so it is not confused with
    # the raw property column of the same value.
    score_field = next((r.get("score_field") for r in rows if r.get("score_field")), "") or "score"
    score_col = f"ranking_score ({score_field})"

    def _core_out(name: str) -> str:
        return score_col if name == "score" else name

    taken: set[str] = {_core_out(n) for n in _OUT_NUMERIC}  # core names are fixed; dedup the rest
    attr_specs = [(_core_out(n), "core", n) for n in _OUT_NUMERIC]
    attr_specs += [(_unique(k, taken), "prop", k) for k in numeric_props]
    attr_specs += [(_unique(n, taken), "pass", n) for n in numeric_pass]

    meta_specs = [(_unique(n, taken), "core", n) for n in _OUT_META]
    meta_specs += [(_unique(k, taken), "prop", k) for k in text_props]
    meta_specs += [(_unique(n, taken), "pass", n) for n in text_pass]

    def cell(row: dict, kind: str, key: str) -> Any:
        if kind == "core" or kind == "pass":
            return row.get(key)
        return row.get("_properties", {}).get(key)

    n = len(rows)
    X = np.full((n, len(attr_specs)), np.nan, dtype=float)
    for j, (_, kind, key) in enumerate(attr_specs):
        for i, row in enumerate(rows):
            X[i, j] = _as_float(cell(row, kind, key))

    M = np.empty((n, len(meta_specs)), dtype=object)
    for j, (_, kind, key) in enumerate(meta_specs):
        for i, row in enumerate(rows):
            value = cell(row, kind, key)
            M[i, j] = "" if value is None else str(value)

    domain = Domain(
        [ContinuousVariable(name) for name, _, _ in attr_specs],
        metas=[StringVariable(name) for name, _, _ in meta_specs],
    )
    table = Table.from_numpy(domain, X=X, metas=M)
    table.name = "Docked Poses"
    return table


def molecule_to_sdf_record(mol: Chem.Mol, name: Optional[str] = None) -> str:
    """Render a 3D RDKit mol as a single-record SDF (mol block + ``$$$$``)."""

    work = Chem.Mol(mol)
    if name:
        work.SetProp("_Name", str(name))
    block = Chem.MolToMolBlock(work)  # keeps the existing 3D conformer
    return block + "$$$$\n"


def has_3d_conformer(mol: Chem.Mol) -> bool:
    return mol is not None and mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()


def poses_to_chemmols(rows: List[dict], passthrough_names: List[str]) -> List[ChemMol]:
    """Build ``ChemMol`` poses: docked 3D geometry + docking + pass-through data.

    Each successful pose row becomes one molecule whose conformation is the docked
    pose (parsed from ``pose_sdf``) and whose ``props`` carry the scores, the raw
    CmDock SDF properties, and the per-ligand pass-through data.
    """

    out: List[ChemMol] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        block = row.get("pose_sdf") or ""
        mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
        if mol is None:
            mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
        if mol is None:
            continue

        props: dict[str, Any] = dict(row.get("_properties", {}))  # docked data (SCORE.* …)
        props.update(
            rank_by_score=row.get("rank_by_score"),
            pose_index=row.get("pose_index"),
            score=row.get("score"),
            score_field=row.get("score_field"),
        )
        for name in passthrough_names:  # original per-ligand data
            if name in row:
                props[name] = row[name]
        out.append(ChemMol(mol=mol, name=row.get("ligand_name") or None, props=props, cache={}))
    return out


class OWCmDockDocking(OWWidget, ConcurrentWidgetMixin):
    name = "CmDock Docking"
    description = "Dock SDF ligands against a CmDock receptor and output scored poses."
    icon = "icons/modeling/ow_cmdock_docking.png"
    priority = 720
    keywords = ["cmdock", "rdock", "docking", "virtual screening", "sdf", "poses", "score"]
    want_main_area = True

    class Inputs:
        molecules = Input("Molecules", list, auto_summary=False)
        data = Input("Data", Table)

    class Outputs:
        poses = Output("Docked Poses", Table)
        molecules = Output("Molecules", list, auto_summary=False)

    # Settings
    cmdock_executable: str = Setting(_default_executable())
    library_directory: str = Setting(_default_library_dir())
    receptor_prm: str = Setting("")
    protocol_file: str = Setting("dock.prm")
    n_docking_runs: int = Setting(100)
    n_best_poses: int = Setting(5)
    score_tag: str = Setting("SCORE.INTER")
    sdf_column: str = Setting("")
    input_mode: str = Setting("Auto")
    extra_flags: str = Setting("")
    keep_temporary_files: bool = Setting(False)
    fail_on_ligand_error: bool = Setting(False)
    use_gnu_parallel: bool = Setting(False)
    parallel_jobs: int = Setting(max(1, min(4, os.cpu_count() or 1)))

    def __init__(self) -> None:
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self._data: Optional[Table] = None
        self._molecules: List[ChemMol] = []

        rec = gui.widgetBox(self.controlArea, "Receptor")
        gui.lineEdit(rec, self, "receptor_prm", label="Receptor .prm", orientation=Qt.Vertical)
        gui.button(rec, self, "Browse .prm…", callback=self._browse_prm, autoDefault=False)

        exe = gui.widgetBox(self.controlArea, "CmDock")
        gui.lineEdit(exe, self, "cmdock_executable", label="cmdock executable", orientation=Qt.Vertical)
        gui.button(exe, self, "Browse executable…", callback=self._browse_exe, autoDefault=False)
        gui.lineEdit(exe, self, "library_directory", label="Library directory (optional)", orientation=Qt.Vertical)

        lig = gui.widgetBox(self.controlArea, "Ligands")
        gui.widgetLabel(lig, "Connect a 3D <b>Molecules</b> input to dock. The "
                             "options below are only used when docking an SDF "
                             "column from a <b>Data</b> table instead.")
        gui.widgetLabel(lig, "SDF column (table-only mode)")
        self._column_combo = QComboBox()
        self._column_combo.activated.connect(self._on_column_changed)
        lig.layout().addWidget(self._column_combo)
        gui.comboBox(lig, self, "input_mode", label="Input mode", items=_INPUT_MODES,
                     sendSelectedValue=True, orientation=Qt.Horizontal)

        params = gui.widgetBox(self.controlArea, "Docking parameters")
        gui.lineEdit(params, self, "protocol_file", label="Protocol (-p)")
        gui.spin(params, self, "n_docking_runs", 1, 100000, label="Runs (-n)")
        gui.spin(params, self, "n_best_poses", 0, 10000, label="Best poses (-b, 0 = off)")
        gui.lineEdit(params, self, "score_tag", label="Ranking score field")

        adv = gui.widgetBox(self.controlArea, "Advanced")
        gui.lineEdit(adv, self, "extra_flags", label="Extra cmdock flags")
        gui.checkBox(adv, self, "keep_temporary_files", "Keep temporary files")
        gui.checkBox(adv, self, "fail_on_ligand_error", "Fail on first ligand error")
        gui.checkBox(adv, self, "use_gnu_parallel", "Use GNU Parallel")
        gui.spin(adv, self, "parallel_jobs", 1, 256, label="Parallel jobs")

        self.dock_button = gui.button(self.controlArea, self, "Dock", callback=self._start_docking)

        status = gui.widgetBox(self.mainArea, "Status")
        self.lbl_status = gui.label(status, self, "Awaiting ligands…")
        self.lbl_status.setWordWrap(True)
        self._log = QPlainTextEdit(readOnly=True)
        self._log.setPlaceholderText("CmDock run details appear here.")
        gui.widgetBox(self.mainArea, "Log").layout().addWidget(self._log)

    # -- input -------------------------------------------------------------- #
    @Inputs.molecules
    def set_molecules(self, mols: Optional[list]) -> None:
        self._molecules = [m for m in (mols or []) if isinstance(m, ChemMol)]
        self._update_input_status()

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._data = data
        self._populate_columns()
        self._update_input_status()

    def _update_input_status(self) -> None:
        if self._molecules:
            extra = f" + {len(self._data)} data row(s)" if self._data is not None else ""
            self.lbl_status.setText(
                f"{len(self._molecules)} molecule(s){extra}. Set the receptor and press Dock."
            )
        elif self._data is not None:
            self.lbl_status.setText(
                f"{len(self._data)} table row(s) (table mode). Pick the SDF column and press Dock."
            )
        else:
            self.lbl_status.setText("Awaiting molecules…")

    def _string_columns(self) -> List[str]:
        if self._data is None:
            return []
        domain = self._data.domain
        return [v.name for v in list(domain.metas) + list(domain.attributes) if isinstance(v, StringVariable)]

    def _populate_columns(self) -> None:
        self._column_combo.clear()
        names = self._string_columns()
        self._column_combo.addItems(names)
        if names:
            if self.sdf_column not in names:
                # prefer a column that looks like SDF/molblock
                guess = next((n for n in names if n.lower() in ("sdf", "molblock", "mol_block", "molfile")), names[0])
                self.sdf_column = guess
            self._column_combo.setCurrentText(self.sdf_column)

    def _on_column_changed(self) -> None:
        self.sdf_column = self._column_combo.currentText()

    # -- browse helpers ----------------------------------------------------- #
    def _browse_prm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select receptor .prm", "", "CmDock prm (*.prm);;All files (*)")
        if path:
            self.receptor_prm = path

    def _browse_exe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select cmdock executable", "", "All files (*)")
        if path:
            self.cmdock_executable = path

    # -- run ---------------------------------------------------------------- #
    def _passthrough_map(self, columns: "dict[str, list]", sdf_column: str) -> dict[str, str]:
        """Map input column names to collision-safe output names."""

        reserved = set(CORE_COLUMNS) | {"_properties"}
        passthrough_map: dict[str, str] = {}
        used = set(reserved)
        for name in columns:
            if name == sdf_column:
                continue
            out = f"input_{name}" if name in reserved else name
            passthrough_map[name] = _unique(out, used)
        return passthrough_map

    def _data_columns(self, n_rows: int) -> "dict[str, list]":
        """Input ``Data`` columns as plain lists, padded/truncated to ``n_rows``."""

        if self._data is None:
            return {}
        domain = self._data.domain
        columns: dict[str, list] = {}
        for var in list(domain.attributes) + list(domain.class_vars) + list(domain.metas):
            if var.name in columns:
                continue
            col = list(self._data.get_column(var))
            columns[var.name] = (col + [None] * n_rows)[:n_rows]
        return columns

    @staticmethod
    def _molecule_prop_columns(mols: List[ChemMol]) -> "dict[str, list]":
        """Scalar ``ChemMol.props`` as aligned columns (used when no Data table)."""

        keys: list[str] = []
        seen: set[str] = set()
        for cm in mols:
            for key, value in (cm.props or {}).items():
                if key not in seen and isinstance(value, (str, int, float, bool)):
                    seen.add(key)
                    keys.append(key)

        def scalar(value: Any) -> Any:
            return value if isinstance(value, (str, int, float, bool)) else None

        return {key: [scalar((cm.props or {}).get(key)) for cm in mols] for key in keys}

    def _frame_from_molecules(self) -> tuple[pd.DataFrame, str, dict[str, str], List[str]]:
        mols = self._molecules
        sdf_texts = []
        for cm in mols:
            try:
                sdf_texts.append(molecule_to_sdf_record(cm.mol, cm.name))
            except Exception:
                sdf_texts.append("")  # becomes a failed ligand row downstream
        frame_data: dict[str, list] = {"__sdf__": sdf_texts}
        # Per-ligand data: from the Data table when connected, else the molecules' props.
        frame_data.update(
            self._data_columns(len(mols)) if self._data is not None
            else self._molecule_prop_columns(mols)
        )
        frame = pd.DataFrame(frame_data)
        passthrough_map = self._passthrough_map(frame_data, "__sdf__")
        return frame, "__sdf__", passthrough_map, list(passthrough_map.values())

    def _frame_from_table(self) -> tuple[pd.DataFrame, str, dict[str, str], List[str]]:
        columns = self._data_columns(len(self._data))
        frame = pd.DataFrame(columns)
        passthrough_map = self._passthrough_map(columns, self.sdf_column)
        return frame, self.sdf_column, passthrough_map, list(passthrough_map.values())

    def _start_docking(self) -> None:
        clear_widget_messages(self)
        if not self.receptor_prm.strip():
            set_widget_error(self, "Select a receptor .prm file.")
            return
        prm = Path(os.path.expanduser(self.receptor_prm.strip()))
        if not prm.is_file():
            set_widget_error(self, f"Receptor .prm not found: {prm}")
            return
        cavity = prm.with_suffix(".as")
        if not cavity.is_file():
            set_widget_error(
                self,
                f"Cavity file '{cavity.name}' not found next to the receptor. "
                f"Generate it first:  cmcavity -r {prm.name} -W",
            )
            return
        recfile = referenced_receptor_file(prm)
        if recfile and not (prm.parent / recfile).is_file():
            set_widget_error(
                self,
                f"The .prm references RECEPTOR_FILE '{recfile}', which is not in "
                f"{prm.parent}. Put that file there, or edit RECEPTOR_FILE in the .prm "
                f"to point to the actual receptor.",
            )
            return

        if self._molecules:
            not_3d = sum(1 for cm in self._molecules if not has_3d_conformer(cm.mol))
            if not_3d:
                set_widget_warning(self, f"{not_3d} molecule(s) lack a 3D conformer; docking may fail.")
            frame, sdf_col, passthrough_map, pass_names = self._frame_from_molecules()
        elif self._data is not None and self.sdf_column:
            frame, sdf_col, passthrough_map, pass_names = self._frame_from_table()
        else:
            set_widget_error(self, "Connect a Molecules input (or a Data table with an SDF column).")
            return

        settings = DockingSettings(
            cmdock_executable=self.cmdock_executable,
            receptor_prm=self.receptor_prm,
            library_directory=self.library_directory,
            protocol_file=self.protocol_file,
            n_docking_runs=self.n_docking_runs,
            n_best_poses=self.n_best_poses,
            score_tag=self.score_tag,
            input_mode=self.input_mode,
            extra_flags=self.extra_flags,
            keep_temporary_files=self.keep_temporary_files,
            fail_on_ligand_error=self.fail_on_ligand_error,
            use_gnu_parallel=self.use_gnu_parallel,
            parallel_jobs=self.parallel_jobs,
        )
        self.dock_button.setEnabled(False)
        self.progressBarInit()
        self.lbl_status.setText("Docking…")
        self.start(self._dock_task, frame, sdf_col, settings, passthrough_map, pass_names)

    @staticmethod
    def _dock_task(frame, sdf_column, settings, passthrough_map, pass_names, state: TaskState):
        def progress(fraction: float, message: str) -> None:
            state.set_progress_value(max(0.0, min(100.0, fraction * 100.0)))
            state.set_status(message)

        rows = dock_dataframe(frame, sdf_column, settings, passthrough_map, progress)
        return rows, pass_names

    def on_done(self, result) -> None:
        self.dock_button.setEnabled(True)
        self.progressBarFinished()
        rows, pass_names = result
        table = build_poses_table(rows, pass_names)
        pose_mols = poses_to_chemmols(rows, pass_names)
        n_failed = sum(1 for r in rows if r.get("status") == "failed")
        n_poses = sum(1 for r in rows if r.get("status") == "ok")

        self.Outputs.poses.send(table)
        self.Outputs.molecules.send(pose_mols or None)
        command = next((r.get("cmdock_command") for r in rows if r.get("cmdock_command")), "")
        first_fail = next((r for r in rows if r.get("status") == "failed"), None)
        diag = ""
        if first_fail is not None:
            output = (first_fail.get("cmdock_stdout") or "") + "\n" + (first_fail.get("cmdock_stderr") or "")
            diag = "\n".join(line for line in output.strip().splitlines() if line.strip())[-1500:]
        self._log.setPlainText(
            f"poses: {n_poses}\nfailures: {n_failed}\ncommand: {command}\n"
            + (f"first error: {first_fail.get('error_message')}\n" if first_fail else "")
            + (f"--- cmdock output ---\n{diag}\n" if diag else "")
        )
        if table is None:
            set_widget_warning(self, "CmDock produced no poses.")
            self.lbl_status.setText("No poses produced.")
            return
        if n_failed:
            set_widget_warning(self, f"{n_failed} ligand(s) failed — see status/error_message columns.")
        self.lbl_status.setText(f"Done: {n_poses} pose(s), {n_failed} failure(s).")

    def on_exception(self, ex: Exception) -> None:
        self.dock_button.setEnabled(True)
        self.progressBarFinished()
        set_widget_error(self, str(ex))
        self.lbl_status.setText(format_error_status(str(ex)))
        self._log.setPlainText(f"error: {ex}")
        self.Outputs.poses.send(None)
        self.Outputs.molecules.send(None)

    def onDeleteWidget(self) -> None:
        self.cancel()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWCmDockDocking).run()
