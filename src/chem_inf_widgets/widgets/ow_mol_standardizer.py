from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from AnyQt.QtCore import pyqtSlot
from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.from_orange import TableMolConversionReport, table_to_chemmols_with_report
from chem_inf_widgets.chemcore.services.curation_summary import (
    CURATION_BLOCKERS,
    CURATION_READY_FOR_DOCKING,
    CURATION_READY_FOR_QSAR,
    CURATION_RECOMMENDED_NEXT_STEP,
    CURATION_STAGE,
    CURATION_STATUS,
    CURATION_VERSION,
    CURATION_VERSION_FIELD,
    CURATION_WARNINGS,
    DOCKING_COMPATIBLE_STANDARDIZATION_PROFILES,
    QSAR_COMPATIBLE_STANDARDIZATION_PROFILES,
    annotate_curation_props,
    curation_summary_to_table,
    summary_from_standardization_rows,
)
from chem_inf_widgets.chemcore.services.mol_standardizer import (
    PROFILE_LABELS,
    MolStandardizer,
    StandardizeConfig,
    get_standardization_config,
)
from chem_inf_widgets.chemcore.services.report_table_utils import report_rows_to_table
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles
from chem_inf_widgets.widgets.ui_helpers import (
    format_done_status,
    format_failed_status,
    format_no_input_status,
    format_skip_warning,
    format_table_report,
    set_widget_warning,
)

logger = logging.getLogger(__name__)


def _find_smiles_var_in_table(data: Table) -> Optional[StringVariable]:
    if data is None:
        return None
    # Prefer meta named "SMILES"
    for v in data.domain.metas:
        if v.name.upper() == "SMILES":
            return v
    # fallback: any meta/attr with SMILES-ish name
    for v in list(data.domain.metas) + list(data.domain.attributes):
        if "SMILES" in v.name.upper():
            return v if isinstance(v, StringVariable) else None
    return None


def _extract_smiles_from_table(data: Table) -> List[str]:
    if data is None or len(data) == 0:
        return []
    v = _find_smiles_var_in_table(data)
    if v is None:
        return []
    out: List[str] = []
    for r in data:
        val = r[v]
        out.append("" if val is None else str(val).strip())
    return out


class OWMolStandardizer(OWWidget):
    name = "Mol Standardizer"
    description = "Standardize molecules (RDKit MolStandardize) from Table (SMILES) and/or ChemMol list."
    icon = "icons/standardization_filtering/owmolstandardizerwidget.png"
    priority = 113

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        modeling_data = Output("Modeling Data", Table)
        data = Output("Data", Table)
        molecules = Output("Molecules", list, auto_summary=False)
        qsar_ready_data = Output("QSAR-ready Data", Table)
        qsar_ready_molecules = Output("QSAR-ready Molecules", list, auto_summary=False)
        standardization_failed_data = Output("Standardization Failed Data", Table)
        standardization_failed_molecules = Output("Standardization Failed Molecules", list, auto_summary=False)
        standardization_report = Output("Standardization Report", Table)
        curation_summary = Output("Curation Summary", Table)

    # --- settings (persisted) ---
    standardization_profile: str = Setting("QSAR-ready")
    op_cleanup: bool = Setting(True)
    op_normalize: bool = Setting(True)
    op_metal_disconnect: bool = Setting(True)
    op_largest_fragment: bool = Setting(True)
    op_reionize: bool = Setting(True)
    op_uncharge: bool = Setting(True)

    sanitize_before: bool = Setting(True)
    sanitize_after: bool = Setting(True)

    overwrite_smiles: bool = Setting(False)
    keep_smiles_orig: bool = Setting(True)

    out_smiles_field: str = Setting("SMILES_STD")
    out_log_field: str = Setting("STD_LOG")

    def __init__(self) -> None:
        super().__init__()
        self.executor = ThreadExecutor(self)

        self._in_table: Optional[Table] = None
        self._in_molecules: List[ChemMol] = []
        self._table_report: Optional[TableMolConversionReport] = None
        self._applying_profile = False

        self._build_ui()
        self._update_status("Ready")

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        self.mainArea.hide()

        root = self.controlArea
        root.setMinimumWidth(340)

        box_profile = QGroupBox("Profile")
        h_profile = QHBoxLayout(box_profile)
        h_profile.addWidget(QLabel("Preset:"))
        self.cmb_profile = QComboBox()
        self.profile_items = ["Minimal", "QSAR-ready", "ChEMBL-like", "Docking-ready", "Custom"]
        self.cmb_profile.addItems(self.profile_items)
        if self.standardization_profile in self.profile_items:
            self.cmb_profile.setCurrentText(self.standardization_profile)
        else:
            self.cmb_profile.setCurrentText("QSAR-ready")
            self.standardization_profile = "QSAR-ready"
        self.cmb_profile.currentTextChanged.connect(self._on_profile_changed)
        h_profile.addWidget(self.cmb_profile, 1)
        root.layout().addWidget(box_profile)

        box_ops = QGroupBox("Standardization steps")
        g = QGridLayout(box_ops)

        self.cb_cleanup = QCheckBox("Cleanup")
        self.cb_cleanup.setChecked(self.op_cleanup)
        self.cb_normalize = QCheckBox("Normalize")
        self.cb_normalize.setChecked(self.op_normalize)
        self.cb_metal = QCheckBox("MetalDisconnector")
        self.cb_metal.setChecked(self.op_metal_disconnect)
        self.cb_largest = QCheckBox("LargestFragmentChooser")
        self.cb_largest.setChecked(self.op_largest_fragment)
        self.cb_reion = QCheckBox("Reionizer")
        self.cb_reion.setChecked(self.op_reionize)
        self.cb_unch = QCheckBox("Uncharger")
        self.cb_unch.setChecked(self.op_uncharge)

        cbs = [self.cb_cleanup, self.cb_normalize, self.cb_metal, self.cb_largest, self.cb_reion, self.cb_unch]
        for i, cb in enumerate(cbs):
            cb.stateChanged.connect(self._on_settings_changed)
            g.addWidget(cb, i // 2, i % 2)

        root.layout().addWidget(box_ops)

        box_san = QGroupBox("Sanitization")
        h = QHBoxLayout(box_san)
        self.cb_san_before = QCheckBox("Sanitize before")
        self.cb_san_before.setChecked(self.sanitize_before)
        self.cb_san_after = QCheckBox("Sanitize after")
        self.cb_san_after.setChecked(self.sanitize_after)
        self.cb_san_before.stateChanged.connect(self._on_settings_changed)
        self.cb_san_after.stateChanged.connect(self._on_settings_changed)
        h.addWidget(self.cb_san_before)
        h.addWidget(self.cb_san_after)
        root.layout().addWidget(box_san)

        box_out = QGroupBox("Output")
        v = QVBoxLayout(box_out)

        self.cb_overwrite = QCheckBox('Overwrite "SMILES" with standardized')
        self.cb_overwrite.setChecked(self.overwrite_smiles)
        self.cb_keep_orig = QCheckBox('Keep original as "SMILES_ORIG"')
        self.cb_keep_orig.setChecked(self.keep_smiles_orig)

        self.cb_overwrite.stateChanged.connect(self._on_settings_changed)
        self.cb_keep_orig.stateChanged.connect(self._on_settings_changed)

        v.addWidget(self.cb_overwrite)
        v.addWidget(self.cb_keep_orig)

        root.layout().addWidget(box_out)

        self.btn_run = QPushButton("Run standardization")
        self.btn_run.clicked.connect(self._on_run)
        root.layout().addWidget(self.btn_run)

        self._apply_profile_to_controls(self.standardization_profile)

        self.lbl = QLabel("Ready")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#475467;")
        root.layout().addWidget(self.lbl)

    def _update_status(self, text: str) -> None:
        self.lbl.setText(text)

    def _set_busy(self, busy: bool, text: str) -> None:
        self.btn_run.setEnabled(not busy)
        self._update_status(text)
        if busy:
            self.progressBarInit()
        else:
            self.progressBarFinished()

    def _on_profile_changed(self, profile: str) -> None:
        self.standardization_profile = str(profile or "Custom")
        self._apply_profile_to_controls(self.standardization_profile)

    def _profile_key(self, label: str) -> str:
        mapping = {v: k for k, v in PROFILE_LABELS.items()}
        return mapping.get(str(label or ""), "custom")

    def _apply_profile_to_controls(self, profile: str) -> None:
        key = self._profile_key(profile)
        if key == "custom":
            return
        cfg = get_standardization_config(key)
        self._applying_profile = True
        try:
            self.cb_cleanup.setChecked(cfg.cleanup)
            self.cb_normalize.setChecked(cfg.normalize)
            self.cb_metal.setChecked(cfg.metal_disconnect)
            self.cb_largest.setChecked(cfg.largest_fragment)
            self.cb_reion.setChecked(cfg.reionize)
            self.cb_unch.setChecked(cfg.uncharge)
            self.cb_san_before.setChecked(cfg.sanitize_before)
            self.cb_san_after.setChecked(cfg.sanitize_after)
        finally:
            self._applying_profile = False
        self._sync_settings_from_controls(mark_custom=False)

    def _sync_settings_from_controls(self, *, mark_custom: bool) -> None:
        self.op_cleanup = bool(self.cb_cleanup.isChecked())
        self.op_normalize = bool(self.cb_normalize.isChecked())
        self.op_metal_disconnect = bool(self.cb_metal.isChecked())
        self.op_largest_fragment = bool(self.cb_largest.isChecked())
        self.op_reionize = bool(self.cb_reion.isChecked())
        self.op_uncharge = bool(self.cb_unch.isChecked())

        self.sanitize_before = bool(self.cb_san_before.isChecked())
        self.sanitize_after = bool(self.cb_san_after.isChecked())

        self.overwrite_smiles = bool(self.cb_overwrite.isChecked())
        self.keep_smiles_orig = bool(self.cb_keep_orig.isChecked())

        if mark_custom and not self._applying_profile:
            self.standardization_profile = "Custom"
            self.cmb_profile.blockSignals(True)
            self.cmb_profile.setCurrentText("Custom")
            self.cmb_profile.blockSignals(False)

    def _on_settings_changed(self) -> None:
        self._sync_settings_from_controls(mark_custom=True)

    # ---------------- inputs ----------------

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._in_table = data
        self._table_report = None
        if data is not None and len(data) > 0:
            try:
                _mols, self._table_report = table_to_chemmols_with_report(data)
            except Exception:
                self._table_report = None
        self._update_status(self._input_summary())

    @Inputs.molecules
    def set_molecules(self, mols: Optional[list]) -> None:
        self._in_molecules = [m for m in (mols or []) if isinstance(m, ChemMol)]
        self._update_status(self._input_summary())

    def _input_summary(self) -> str:
        n_tab = 0 if self._in_table is None else len(self._in_table)
        n_mol = len(self._in_molecules)
        if self._table_report is not None:
            return format_table_report(self._table_report, prefix="Input") + f", Molecules={n_mol}"
        return f"Input: Table rows={n_tab}, Molecules={n_mol}"

    # ---------------- run ----------------

    def _make_config(self) -> StandardizeConfig:
        self._sync_settings_from_controls(mark_custom=False)
        return StandardizeConfig(
            cleanup=self.op_cleanup,
            normalize=self.op_normalize,
            metal_disconnect=self.op_metal_disconnect,
            largest_fragment=self.op_largest_fragment,
            reionize=self.op_reionize,
            uncharge=self.op_uncharge,
            sanitize_before=self.sanitize_before,
            sanitize_after=self.sanitize_after,
            canonical_smiles=True,
        )

    def _active_profile_key(self) -> str:
        return self._profile_key(self.standardization_profile)

    def _on_run(self) -> None:
        if (self._in_table is None or len(self._in_table) == 0) and not self._in_molecules:
            self._update_status(format_no_input_status())
            self.Outputs.modeling_data.send(None)
            self.Outputs.data.send(None)
            self.Outputs.molecules.send([])
            self.Outputs.qsar_ready_data.send(None)
            self.Outputs.qsar_ready_molecules.send([])
            self.Outputs.standardization_failed_data.send(None)
            self.Outputs.standardization_failed_molecules.send([])
            self.Outputs.standardization_report.send(None)
            self.Outputs.curation_summary.send(None)
            return

        cfg = self._make_config()
        set_widget_warning(
            self,
            format_skip_warning(
                0 if self._table_report is None else self._table_report.n_invalid,
                subject="invalid input rows",
                action="will produce empty standardized SMILES",
            ),
        )
        self._set_busy(True, "Standardizing…")

        fut = self.executor.submit(self._run_background, self._in_table, self._in_molecules, cfg, self._active_profile_key())
        fut.add_done_callback(self._on_done)

    def _run_background(
        self,
        data: Optional[Table],
        mols: Sequence[ChemMol],
        cfg: StandardizeConfig,
        profile_key: str,
    ) -> Tuple[Optional[Table], List[ChemMol], Optional[Table], Optional[Table], List[ChemMol], Optional[Table], List[ChemMol], Table, Table, int, int]:
        std = MolStandardizer(cfg, profile=profile_key)

        out_mols: List[ChemMol] = []
        out_table: Optional[Table] = None
        report_rows: List[dict] = []

        # 1) Standardize ChemMol list
        if mols:
            out_mols, mol_results = std.standardize_chemmols(
                mols,
                smiles_prop="SMILES",
                out_smiles_prop=self.out_smiles_field,
                out_log_prop=self.out_log_field,
                keep_original_smiles_prop=self.keep_smiles_orig,
                overwrite_smiles_prop=self.overwrite_smiles,
            )
            out_mols = self._annotate_standardization_curation(out_mols, mol_results, std.profile)
            for i, res in enumerate(mol_results, start=1):
                report_rows.append({
                    "row_index": i,
                    "source": "molecules",
                    "ok": int(bool(res.ok)),
                    "input_smiles": res.input_smiles,
                    "standardized_smiles": res.output_smiles,
                    "status": "ok" if res.ok else "failed",
                    "profile": std.profile,
                    "changed": int(bool(res.ok and res.input_smiles and res.output_smiles and res.input_smiles != res.output_smiles)),
                    "steps": res.log,
                    "log": res.log,
                })

        # 2) Standardize Table (SMILES)
        if data is not None and len(data) > 0:
            smiles = _extract_smiles_from_table(data)
            if not smiles:
                try:
                    out_table = self._append_std_columns_to_table(
                        data,
                        std_smiles=[""] * len(data),
                        logs=["No SMILES column found"] * len(data),
                        statuses=["failed"] * len(data),
                        profiles=[std.profile] * len(data),
                        changed=[0] * len(data),
                        input_smiles=[""] * len(data),
                        overwrite_smiles=False,
                    )
                except Exception as exc:
                    report_rows.append({
                        "row_index": "",
                        "source": "data_output",
                        "ok": 0,
                        "input_smiles": "",
                        "standardized_smiles": "",
                        "status": "failed",
                        "profile": std.profile,
                        "changed": 0,
                        "steps": "Audit Data construction failed",
                        "log": str(exc),
                    })
                    out_table = None
            else:
                std_smiles: List[str] = []
                logs: List[str] = []
                statuses: List[str] = []
                profiles: List[str] = []
                changed: List[int] = []
                input_smiles_values: List[str] = []
                for row_i, smi in enumerate(smiles, start=1):
                    res = std.standardize_smiles(smi)
                    std_smiles.append(res.output_smiles if res.ok else "")
                    logs.append(res.log)
                    statuses.append("ok" if res.ok else "failed")
                    profiles.append(std.profile)
                    changed.append(int(bool(res.ok and res.input_smiles and res.output_smiles and res.input_smiles != res.output_smiles)))
                    input_smiles_values.append(res.input_smiles)
                    report_rows.append({
                        "row_index": row_i,
                        "source": "table",
                        "ok": int(bool(res.ok)),
                        "input_smiles": res.input_smiles,
                        "standardized_smiles": res.output_smiles,
                        "status": "ok" if res.ok else "failed",
                        "profile": std.profile,
                        "changed": int(bool(res.ok and res.input_smiles and res.output_smiles and res.input_smiles != res.output_smiles)),
                        "steps": res.log,
                        "log": res.log,
                    })

                try:
                    out_table = self._append_std_columns_to_table(
                        data,
                        std_smiles=std_smiles,
                        logs=logs,
                        statuses=statuses,
                        profiles=profiles,
                        changed=changed,
                        input_smiles=input_smiles_values,
                        overwrite_smiles=self.overwrite_smiles,
                    )
                except Exception as exc:
                    report_rows.append({
                        "row_index": "",
                        "source": "data_output",
                        "ok": 0,
                        "input_smiles": "",
                        "standardized_smiles": "",
                        "status": "failed",
                        "profile": std.profile,
                        "changed": 0,
                        "steps": "Audit Data construction failed",
                        "log": str(exc),
                    })
                    out_table = None

                # If no input molecules were provided, also emit ChemMol list from standardized SMILES
                if not mols:
                    out_mols = self._chemmols_from_smiles(std_smiles, logs)
                    out_mols = self._annotate_standardization_curation_from_statuses(out_mols, ["ok"] * len(out_mols), std.profile)

        table_ok_indices = [int(r["row_index"]) - 1 for r in report_rows if r.get("source") == "table" and bool(r.get("ok"))]
        table_failed_indices = [int(r["row_index"]) - 1 for r in report_rows if r.get("source") == "table" and not bool(r.get("ok"))]
        modeling_table = None
        if data is not None and std.profile in QSAR_COMPATIBLE_STANDARDIZATION_PROFILES:
            try:
                modeling_table = self._modeling_table(
                    data,
                    table_ok_indices,
                    [
                        str(r.get("standardized_smiles", "") or "")
                        for r in report_rows
                        if r.get("source") == "table" and bool(r.get("ok"))
                    ],
                )
            except Exception as exc:
                # Standardization itself should not fail just because the slim
                # modelling table could not be constructed from an unusual
                # Orange domain.  The full audit Data and report still allow
                # the user to continue/debug.
                report_rows.append({
                    "row_index": "",
                    "source": "modeling_data",
                    "ok": 0,
                    "input_smiles": "",
                    "standardized_smiles": "",
                    "status": "failed",
                    "profile": std.profile,
                    "changed": 0,
                    "steps": "Modeling Data construction failed",
                    "log": str(exc),
                })
                modeling_table = None
        qsar_ready_table = modeling_table
        try:
            failed_table = self._subset_table(out_table, table_failed_indices) if out_table is not None else None
        except Exception as exc:
            report_rows.append({
                "row_index": "",
                "source": "failed_data",
                "ok": 0,
                "input_smiles": "",
                "standardized_smiles": "",
                "status": "failed",
                "profile": std.profile,
                "changed": 0,
                "steps": "Failed Data construction failed",
                "log": str(exc),
            })
            failed_table = None

        mol_ok = [i for i, r in enumerate(report_rows) if r.get("source") == "molecules" and bool(r.get("ok"))]
        mol_failed = [i for i, r in enumerate(report_rows) if r.get("source") == "molecules" and not bool(r.get("ok"))]
        qsar_ready_mols = [out_mols[i] for i in mol_ok if i < len(out_mols)] if std.profile in QSAR_COMPATIBLE_STANDARDIZATION_PROFILES else []
        failed_mols = [out_mols[i] for i in mol_failed if i < len(out_mols)]

        try:
            for row in report_rows:
                ok = bool(row.get("ok"))
                row.setdefault("qc_flags", "" if ok else "standardization_failed")
                row.setdefault("dropped_reason", "" if ok else "standardization_failed")
            report_table = self._standardization_report_to_table(report_rows)
        except Exception:
            logger.warning("Could not build standardization report table.", exc_info=True)
            report_table = None
        try:
            curation_table = curation_summary_to_table(summary_from_standardization_rows(report_rows, std.profile))
        except Exception:
            logger.warning("Could not build standardization curation summary table.", exc_info=True)
            curation_table = None
        return out_table, out_mols, modeling_table, qsar_ready_table, qsar_ready_mols, failed_table, failed_mols, report_table, curation_table, (0 if data is None else len(data)), len(out_mols)

    def _on_done(self, fut) -> None:
        try:
            # Send a single payload object back to the GUI thread.  Passing many
            # arguments through AnyQt/orangewidget.methodinvoke is fragile across
            # PyQt/PySide versions and can raise the opaque error:
            # "arguments did not match any overloaded call" even when the
            # standardization itself succeeded.
            payload = fut.result()
            methodinvoke(self, "_apply_outputs", (object,))(payload)
        except Exception as e:
            # Keep the GUI message short but include the original exception text.
            methodinvoke(self, "_apply_error", (str,))(str(e))

    @pyqtSlot(str)
    def _apply_error(self, msg: str) -> None:
        self._set_busy(False, format_failed_status(msg))
        self.Outputs.modeling_data.send(None)
        self.Outputs.data.send(None)
        self.Outputs.molecules.send([])
        self.Outputs.qsar_ready_data.send(None)
        self.Outputs.qsar_ready_molecules.send([])
        self.Outputs.standardization_failed_data.send(None)
        self.Outputs.standardization_failed_molecules.send([])
        self.Outputs.standardization_report.send(None)
        self.Outputs.curation_summary.send(None)

    @pyqtSlot(object)
    def _apply_outputs(self, payload: object) -> None:
        try:
            table, mols, modeling_table, qsar_table, qsar_mols, failed_table, failed_mols, report, curation, n_rows, n_mols = payload
        except Exception as exc:
            self._apply_error(f"Could not unpack standardization outputs: {exc}")
            return
        # table: Optional[Table], mols: List[ChemMol]
        if self._table_report is not None:
            status = format_done_status(
                f"Table rows={0 if table is None else len(table)}",
                f"valid input={self._table_report.n_valid}",
                f"invalid input={self._table_report.n_invalid}",
                f"Molecules={len(mols)}",
                prefix="Done",
            )
        else:
            status = format_done_status(
                f"Table rows={0 if table is None else len(table)}",
                f"Molecules={len(mols)}",
                prefix="Done",
            )
        self._set_busy(False, status)
        self.Outputs.modeling_data.send(modeling_table)
        self.Outputs.data.send(table)
        self.Outputs.molecules.send(mols)
        self.Outputs.qsar_ready_data.send(qsar_table)
        self.Outputs.qsar_ready_molecules.send(qsar_mols)
        self.Outputs.standardization_failed_data.send(failed_table)
        self.Outputs.standardization_failed_molecules.send(failed_mols)
        self.Outputs.standardization_report.send(report)
        self.Outputs.curation_summary.send(curation)

    # ---------------- table helpers ----------------

    @staticmethod
    def _as_class_y_or_none(data: Table, class_count: int):
        """Return a Y array accepted by Orange Table.from_numpy.

        Orange requires ``Y=None`` for domains without class variables.  For a
        single class variable, different Orange versions may expose ``data.Y``
        either as ``(n,)`` or ``(n, 1)``; both are normalized here.
        """
        if class_count <= 0:
            return None
        y = np.asarray(data.Y, dtype=float)
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        if y.shape[1] != class_count:
            y = y[:, :class_count] if y.shape[1] > class_count else np.resize(y, (len(data), class_count))
        return y

    def _append_std_columns_to_table(
        self,
        data: Table,
        std_smiles: List[str],
        logs: List[str],
        statuses: Optional[List[str]] = None,
        profiles: Optional[List[str]] = None,
        changed: Optional[List[int]] = None,
        input_smiles: Optional[List[str]] = None,
        overwrite_smiles: bool = False,
    ) -> Table:
        """
        Return a new Table:
        - keeps original attrs/class/metas
        - appends metas: SMILES_STD and STD_LOG (or configured names)
        - optionally overwrites existing "SMILES" meta (only if found exactly)
        """
        dom = data.domain
        metas = list(dom.metas)

        v_smiles = _find_smiles_var_in_table(data)
        idx_smiles_meta = None
        if v_smiles is not None:
            for i, v in enumerate(metas):
                if v is v_smiles:
                    idx_smiles_meta = i
                    break

        # Make all appended meta names unique.  Orange can behave poorly when a
        # domain contains duplicate variable names (for example after running
        # the widget twice, or when upstream widgets already created audit
        # columns).  Duplicate names were one source of the opaque error:
        # "arguments did not match any overloaded call".
        existing_names = {v.name for v in list(dom.attributes) + list(dom.class_vars) + list(dom.metas)}

        def new_meta(name: str) -> StringVariable:
            unique_name = self._unique_variable_name(name, existing_names)
            return StringVariable(unique_name)

        v_std = new_meta(self.out_smiles_field)
        v_std.attributes["format"] = "SMILES"
        v_log = new_meta(self.out_log_field)
        audit_vars = [
            new_meta("standardization_status"),
            new_meta("standardization_profile"),
            new_meta("standardization_changed"),
            new_meta("standardization_input_smiles"),
            new_meta("standardization_output_smiles"),
            new_meta("standardization_log"),
            new_meta("standardization_version"),
            new_meta(CURATION_STAGE),
            new_meta(CURATION_STATUS),
            new_meta(CURATION_READY_FOR_QSAR),
            new_meta(CURATION_READY_FOR_DOCKING),
            new_meta(CURATION_BLOCKERS),
            new_meta(CURATION_WARNINGS),
            new_meta(CURATION_RECOMMENDED_NEXT_STEP),
            new_meta(CURATION_VERSION_FIELD),
        ]

        metas_out = metas + [v_std, v_log] + audit_vars
        dom_out = Domain(list(dom.attributes), list(dom.class_vars), metas=metas_out)

        X = data.X
        # Orange Table.from_numpy expects Y=None when the output domain has
        # no class variables.  Passing data.Y from a class-less input table
        # can be an empty (n, 0) array, which raises the unhelpful runtime
        # error: "arguments did not match any overloaded call".
        Y = self._as_class_y_or_none(data, len(dom.class_vars))
        M = np.asarray(data.metas, dtype=object).copy()

        # ensure metas array exists
        if M is None or M.size == 0:
            M = np.zeros((len(data), len(metas)), dtype=object)

        M_out = np.zeros((len(data), len(metas_out)), dtype=object)
        if len(metas) > 0:
            M_out[:, : len(metas)] = M

        # overwrite SMILES meta if explicitly present and requested
        if overwrite_smiles and idx_smiles_meta is not None:
            for i in range(len(data)):
                if std_smiles[i]:
                    M_out[i, idx_smiles_meta] = std_smiles[i]

        statuses = statuses or ["ok" if smi else "failed" for smi in std_smiles]
        profiles = profiles or [self.standardization_profile] * len(data)
        changed = changed or [0] * len(data)
        input_smiles = input_smiles or ["" for _ in range(len(data))]

        # add std + audit columns
        for i in range(len(data)):
            std_value = std_smiles[i] if i < len(std_smiles) else ""
            log_value = logs[i] if i < len(logs) else ""
            M_out[i, len(metas) + 0] = std_value
            M_out[i, len(metas) + 1] = log_value
            audit_base = len(metas) + 2
            M_out[i, audit_base + 0] = statuses[i] if i < len(statuses) else "failed"
            M_out[i, audit_base + 1] = profiles[i] if i < len(profiles) else self.standardization_profile
            M_out[i, audit_base + 2] = str(changed[i] if i < len(changed) else 0)
            M_out[i, audit_base + 3] = input_smiles[i] if i < len(input_smiles) else ""
            M_out[i, audit_base + 4] = std_value
            M_out[i, audit_base + 5] = log_value
            M_out[i, audit_base + 6] = "phase2.5"
            c_status, ready_qsar, ready_docking, blockers, warnings, next_step = self._curation_values_for_standardization(
                statuses[i] if i < len(statuses) else "failed",
                profiles[i] if i < len(profiles) else self.standardization_profile,
            )
            M_out[i, audit_base + 7] = "standardization"
            M_out[i, audit_base + 8] = c_status
            M_out[i, audit_base + 9] = str(int(ready_qsar))
            M_out[i, audit_base + 10] = str(int(ready_docking))
            M_out[i, audit_base + 11] = blockers
            M_out[i, audit_base + 12] = warnings
            M_out[i, audit_base + 13] = next_step
            M_out[i, audit_base + 14] = CURATION_VERSION

        return Table.from_numpy(dom_out, X=X, Y=Y, metas=M_out)


    @staticmethod
    def _is_modeling_excluded_column(name: str) -> bool:
        """Columns not meant for slim downstream modelling tables."""
        key = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
        if key in {
            "smiles",
            "smile",
            "canonical_smiles",
            "can_smiles",
            "input_smiles",
            "original_smiles",
            "standardized_smiles",
            "smiles_orig",
            "smiles_std",
            "mol_smiles",
            "inchi",
            "inchikey",
            "inchi_key",
            "rdkit_mol",
            "mol",
        }:
            return True
        return key.startswith(("qc_", "standardization_", "curation_", "std_", "import_"))

    @staticmethod
    def _unique_variable_name(base: str, existing: set[str]) -> str:
        name = base
        counter = 2
        while name in existing:
            name = f"{base}_{counter}"
            counter += 1
        existing.add(name)
        return name

    @staticmethod
    def _inchikey_from_smiles(smiles: str) -> str:
        try:
            from rdkit import Chem

            mol = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol
            if mol is None:
                return ""
            return str(Chem.MolToInchiKey(mol) or "")
        except Exception:
            logger.debug("Could not derive InChIKey from SMILES during standardization.", exc_info=True)
            return ""

    @classmethod
    def _modeling_table(cls, data: Table, indices: Sequence[int], smiles_values: Sequence[str]) -> Table:
        """Return a slim table for downstream modelling.

        It preserves original non-structure/non-audit columns and appends exactly
        one structure column named ``SMILES`` plus ``inchikey`` as metas.
        """
        n_rows = len(data)
        pairs = []
        seen = set()
        for pos, raw_idx in enumerate(indices):
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            if idx < 0 or idx >= n_rows or idx in seen:
                continue
            smi = str((smiles_values[pos] if pos < len(smiles_values) else "") or "").strip()
            if not smi:
                continue
            pairs.append((idx, smi))
            seen.add(idx)

        keep_attrs_idx = [i for i, var in enumerate(data.domain.attributes) if not cls._is_modeling_excluded_column(var.name)]
        keep_class_idx = [i for i, var in enumerate(data.domain.class_vars) if not cls._is_modeling_excluded_column(var.name)]
        keep_meta_idx = [i for i, var in enumerate(data.domain.metas) if not cls._is_modeling_excluded_column(var.name)]

        attrs = [data.domain.attributes[i] for i in keep_attrs_idx]
        class_vars = [data.domain.class_vars[i] for i in keep_class_idx]
        metas = [data.domain.metas[i] for i in keep_meta_idx]
        existing = {var.name for var in attrs + class_vars + metas}
        smiles_name = cls._unique_variable_name("SMILES", existing)
        inchikey_name = cls._unique_variable_name("inchikey", existing)
        metas = metas + [StringVariable(smiles_name), StringVariable(inchikey_name)]
        domain = Domain(attrs, class_vars, metas=metas)

        if not pairs:
            return Table.from_numpy(
                domain,
                X=np.empty((0, len(attrs)), dtype=float),
                Y=np.empty((0, len(class_vars)), dtype=float) if class_vars else None,
                metas=np.empty((0, len(metas)), dtype=object),
            )

        rows = [idx for idx, _ in pairs]
        base_x = np.asarray(data.X, dtype=float)[:, keep_attrs_idx] if keep_attrs_idx else np.empty((len(data), 0), dtype=float)
        X = base_x[rows, :] if keep_attrs_idx else np.empty((len(rows), 0), dtype=float)

        if keep_class_idx:
            base_y = np.asarray(data.Y, dtype=float)
            if base_y.ndim == 1:
                base_y = base_y.reshape((-1, 1))
            if base_y.shape[1] <= max(keep_class_idx):
                Y = None
                class_vars = []
                domain = Domain(attrs, class_vars, metas=metas)
            else:
                Y = base_y[:, keep_class_idx][rows, :]
        else:
            Y = None

        base_metas = np.asarray(data.metas, dtype=object)[:, keep_meta_idx] if keep_meta_idx else np.empty((len(data), 0), dtype=object)
        extra = np.asarray([[smi, cls._inchikey_from_smiles(smi)] for _, smi in pairs], dtype=object)
        M = np.hstack([base_metas[rows, :], extra]) if keep_meta_idx else extra
        return Table.from_numpy(domain, X=X, Y=Y, metas=M)


    @staticmethod
    def _empty_like_table(data: Table) -> Table:
        """Return a zero-row table with the same domain.

        Orange's ``Table.from_numpy`` is picky about the Y argument: when the
        domain has no class variables, Y must be ``None``.  Passing an empty
        ``(0, 0)`` array can trigger the unhelpful Qt/Orange error
        "arguments did not match any overloaded call" in some environments.
        """
        return Table.from_numpy(
            data.domain,
            X=np.empty((0, len(data.domain.attributes)), dtype=float),
            Y=np.empty((0, len(data.domain.class_vars)), dtype=float) if len(data.domain.class_vars) else None,
            metas=np.empty((0, len(data.domain.metas)), dtype=object),
        )

    @staticmethod
    def _safe_subset_indices(data: Table, indices: Sequence[int]) -> List[int]:
        """Sanitize provenance indices before slicing an Orange Table."""
        n_rows = len(data)
        out: List[int] = []
        seen = set()
        for value in indices:
            try:
                idx = int(value)
            except Exception:
                continue
            if idx < 0 or idx >= n_rows or idx in seen:
                continue
            out.append(idx)
            seen.add(idx)
        return out

    @staticmethod
    def _subset_table(data: Optional[Table], indices: Sequence[int]) -> Optional[Table]:
        if data is None:
            return None
        valid = OWMolStandardizer._safe_subset_indices(data, indices)
        if not valid:
            return OWMolStandardizer._empty_like_table(data)
        return data[valid]

    @staticmethod
    def _curation_values_for_standardization(status: str, profile: str) -> tuple[str, bool, bool, str, str, str]:
        ok = str(status or "").strip().lower() == "ok"
        profile_key = str(profile or "").strip()
        if not ok:
            return ("Blocked", False, False, "Standardization failed", "", "Inspect Standardization Report")
        if profile_key in QSAR_COMPATIBLE_STANDARDIZATION_PROFILES:
            return ("QSAR-ready", True, False, "", "", "Descriptors / Fingerprints → QSAR Studio")
        if profile_key in DOCKING_COMPATIBLE_STANDARDIZATION_PROFILES:
            return ("Docking-ready", False, True, "", "", "Docking/pose workflow")
        return ("Standardized", False, False, "", "Profile is not marked as QSAR-ready", "Use QSAR-ready profile for QSAR workflows")

    @classmethod
    def _annotate_standardization_curation_from_statuses(cls, mols: Sequence[ChemMol], statuses: Sequence[str], profile: str) -> List[ChemMol]:
        annotated: List[ChemMol] = []
        for i, cm in enumerate(mols):
            status = statuses[i] if i < len(statuses) else "failed"
            c_status, ready_qsar, ready_docking, blockers, warnings, next_step = cls._curation_values_for_standardization(status, profile)
            annotated.extend(annotate_curation_props(
                [cm],
                stage="standardization",
                status=c_status,
                ready_for_qsar=bool(ready_qsar),
                ready_for_docking=bool(ready_docking),
                blockers=blockers,
                warnings=warnings,
                recommended_next_step=next_step,
            ))
        return annotated

    @classmethod
    def _annotate_standardization_curation(cls, mols: Sequence[ChemMol], results: Sequence[object], profile: str) -> List[ChemMol]:
        statuses = ["ok" if bool(getattr(res, "ok", False)) else "failed" for res in results]
        return cls._annotate_standardization_curation_from_statuses(mols, statuses, profile)


    def _standardization_report_to_table(self, rows: List[dict]) -> Table:
        """Build an Orange Table with one row per standardization attempt.

        The widget must never fail just because the report output is connected.
        Empty reports are represented as a zero-row table with a stable schema.
        """
        meta_columns = [
            "row_index",
            "source",
            "ok",
            "input_smiles",
            "standardized_smiles",
            "status",
            "profile",
            "changed",
            "steps",
            "log",
            "qc_flags",
            "dropped_reason",
        ]
        return report_rows_to_table(
            rows,
            meta_columns=meta_columns,
            name="Standardization Report",
        )

    def _chemmols_from_smiles(self, smiles: List[str], logs: List[str]) -> List[ChemMol]:
        from rdkit import Chem

        out: List[ChemMol] = []
        for i, smi in enumerate(smiles):
            smi = (smi or "").strip()
            if not smi:
                continue
            m = safe_mol_from_smiles(smi, sanitize=True, remove_hs=True).mol
            if m is None:
                continue
            cm = ChemMol.from_rdkit(m, name=f"mol_{i+1}")
            cm.set_prop("SMILES", smi)
            cm.set_prop(self.out_smiles_field, smi)
            log_value = logs[i] if i < len(logs) else ""
            cm.set_prop(self.out_log_field, log_value)
            cm.set_prop("standardization_status", "ok")
            cm.set_prop("standardization_profile", self.standardization_profile)
            cm.set_prop("standardization_changed", "")
            cm.set_prop("standardization_log", log_value)
            cm.set_prop("standardization_version", "phase2.5")
            try:
                from chem_inf_widgets.chemcore.molecule_contract import ensure_contract_props

                ensure_contract_props(cm, row_index=i + 1, input_smiles=smi)
            except Exception:
                pass
            out.append(cm)
        return out
