from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from AnyQt.QtCore import Qt, pyqtSlot as Slot
from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.admet.drug_filter_service import DrugRow, FilterConfig, filter_smiles
from chem_inf_widgets.chemcore.services.from_orange import TableMolConversionReport, table_to_chemmols_with_report
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_error_status,
    format_skip_warning,
    format_table_report,
    set_widget_error,
    set_widget_warning,
)


def _guess_smiles_column(table: Table) -> int:
    if table is None or table.domain is None:
        return 0
    metas = list(table.domain.metas or [])
    for i, v in enumerate(metas):
        if (v.name or "").strip().lower() == "smiles":
            return i
    return 0


class OWDrugFilter(OWWidget):
    name = "Drug Filter"
    description = "Full drug-likeness filter (Lipinski/Veber/QED + PAINS)."
    icon = "icons/standardization_filtering/owdrugfilterwidget.png"
    priority = 114
    want_main_area = False
    resizing_enabled = False

    class Inputs:
        data = Input("Input Table", Table)

    class Outputs:
        filtered = Output("Filtered Compounds", Table)

    filter_rule: str = Setting("Lipinski + Veber")
    selection_mode: str = Setting("Within Criteria")
    highlight_pains: bool = Setting(False)
    pains_json_path: str = Setting("smartspains.json")  # fallback only

    def __init__(self) -> None:
        super().__init__()
        self._data: Optional[Table] = None
        self._executor = ThreadExecutor(self)
        self._last_rows: List[DrugRow] = []
        self._table_report: Optional[TableMolConversionReport] = None

        self._build_ui()
        self._apply_style()
        self._set_status("Awaiting molecular data…", ok=True)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
        QLabel#HdrTitle { font-size: 15px; font-weight: 600; }
        QLabel#HdrSub { color: #6B7280; font-size: 12px; }
        QLabel#StatusChip {
            padding: 4px 10px;
            border: 1.5px solid #E5E7EB;
            border-radius: 10px;
            background: #F9FAFB;
            font-size: 12px;
        }
        """
        )

    def _build_ui(self) -> None:
        hdr = QWidget(self)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(6, 6, 6, 6)

        left = QVBoxLayout()
        left.addWidget(QLabel("Drug Filter", objectName="HdrTitle"))
        left.addWidget(QLabel("Lipinski / Veber / QED + PAINS • outputs filtered Orange Table", objectName="HdrSub"))
        hl.addLayout(left, 1)

        self.lbl_status = QLabel("Ready", objectName="StatusChip")
        self.lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hl.addWidget(self.lbl_status)

        self.controlArea.layout().addWidget(hdr)

        box = QGroupBox("Filtering")
        vl = QVBoxLayout(box)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Rule"))
        self.cmb_rule = QComboBox()
        self.cmb_rule.addItems(["Lipinski", "Veber", "Lipinski + Veber", "None"])
        self.cmb_rule.setCurrentText(self.filter_rule)
        self.cmb_rule.currentTextChanged.connect(self._on_rule_changed)
        row1.addWidget(self.cmb_rule, 1)
        vl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Selection"))
        self.cmb_sel = QComboBox()
        self.cmb_sel.addItems(["Forward All Molecules", "Within Criteria", "Out of Criteria"])
        self.cmb_sel.setCurrentText(self.selection_mode)
        self.cmb_sel.currentTextChanged.connect(self._on_sel_changed)
        row2.addWidget(self.cmb_sel, 1)
        vl.addLayout(row2)

        self.chk_pains = QCheckBox("Highlight PAINS substructures (store matched atom indices)")
        self.chk_pains.setChecked(bool(self.highlight_pains))
        self.chk_pains.toggled.connect(self._on_pains_toggled)
        vl.addWidget(self.chk_pains)

        self.btn_run = QPushButton("Filter Molecules")
        self.btn_run.clicked.connect(self._run)
        vl.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        vl.addWidget(self.progress)

        self.controlArea.layout().addWidget(box)
        self.controlArea.layout().addStretch(1)

    def _set_status(self, msg: str, ok: bool = True) -> None:
        self.lbl_status.setText(msg)
        if ok:
            self.lbl_status.setStyleSheet(
                "padding:4px 8px;border:1px solid #e1e1e1;border-radius:10px;background:#fafafa;"
            )
        else:
            self.lbl_status.setStyleSheet(
                "padding:4px 8px;border:1px solid #f2c2c2;border-radius:10px;background:#fff5f5;color:#a40000;"
            )

    def _set_busy(self, busy: bool, msg: str = "") -> None:
        self.btn_run.setEnabled(not busy)
        self.cmb_rule.setEnabled(not busy)
        self.cmb_sel.setEnabled(not busy)
        self.chk_pains.setEnabled(not busy)
        self.progress.setVisible(busy)
        if msg:
            self._set_status(msg, ok=True)

    def _resolve_pains_json_path(self) -> str:
        """Fallback only; service prefers the packaged chemcore/data resource."""
        here = Path(__file__).resolve()
        pkg_root = here.parents[1]  # chem_inf_widgets/
        candidate = pkg_root / "chemcore" / "data" / "smartspains.json"
        if candidate.exists():
            return str(candidate)

        return str(self.pains_json_path or "smartspains.json")

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        clear_widget_messages(self)
        self._data = data
        self._table_report = None
        if data is None or len(data) == 0:
            self._set_status("No valid data received.", ok=False)
            self.Outputs.filtered.send(None)
            return
        try:
            _mols, self._table_report = table_to_chemmols_with_report(data)
        except Exception:
            self._table_report = None
        if self._table_report is not None:
            self._set_status(format_table_report(self._table_report), ok=True)
        else:
            self._set_status(f"Input: {len(data)} rows.", ok=True)

    def _on_rule_changed(self, txt: str) -> None:
        self.filter_rule = txt

    def _on_sel_changed(self, txt: str) -> None:
        self.selection_mode = txt

    def _on_pains_toggled(self, state: bool) -> None:
        self.highlight_pains = bool(state)

    def _run(self) -> None:
        if self._data is None or len(self._data) == 0:
            self._set_status("No input data.", ok=False)
            return
        set_widget_warning(
            self,
            format_skip_warning(
                0 if self._table_report is None else self._table_report.n_invalid,
                subject="invalid input rows",
                action="will be skipped by drug filtering",
            ),
        )

        smiles_idx = _guess_smiles_column(self._data)
        smiles_col = self._data.metas[:, smiles_idx]

        cfg = FilterConfig(
            filter_rule=self.filter_rule,
            selection_mode=self.selection_mode,
            compute_qed=True,
            compute_pains=True,
            highlight_pains_atoms=bool(self.highlight_pains),
            pains_json_path=self._resolve_pains_json_path(),
        )

        self._set_busy(True, "Filtering…")

        def progress_cb(i: int, n: int) -> None:
            pct = int(100.0 * i / max(1, n))
            # IMPORTANT: invoke Qt slot (registered)
            methodinvoke(self, "_set_progress", (int,))(pct)

        fut = self._executor.submit(filter_smiles, list(smiles_col), cfg, progress_cb)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut) -> None:
        try:
            rows = fut.result()
            methodinvoke(self, "_finish", (object,))(rows)
        except Exception as e:
            methodinvoke(self, "_fail", (str,))(str(e))

    # ---- Qt slots needed for methodinvoke (fix for Windows/PyQt) ----

    @Slot(int)
    def _set_progress(self, pct: int) -> None:
        self.progress.setValue(int(pct))

    @Slot(str)
    def _fail(self, msg: str) -> None:
        self._set_busy(False)
        set_widget_error(self, msg)
        self._set_status(format_error_status(msg), ok=False)
        self.Outputs.filtered.send(None)

    @Slot(object)
    def _finish(self, rows: object) -> None:
        rows_typed: List[DrugRow] = rows or []
        self._last_rows = rows_typed
        table = self._rows_to_table(rows_typed, include_atoms=bool(self.highlight_pains))
        self.Outputs.filtered.send(table)
        self._set_busy(False)
        status = format_done_status(f"{len(rows_typed)} molecules forwarded")
        if self._table_report is not None:
            status += f" Valid input={self._table_report.n_valid}, invalid input={self._table_report.n_invalid}."
        self._set_status(status, ok=True)

    # ---------------------------------------------------------------

    def _rows_to_table(self, rows: List[DrugRow], include_atoms: bool) -> Optional[Table]:
        if not rows:
            return None

        features = [
            ContinuousVariable("QED Score"),
            ContinuousVariable("Lipinski Violations"),
            ContinuousVariable("MW"),
            ContinuousVariable("LogP"),
            ContinuousVariable("HBD"),
            ContinuousVariable("HBA"),
            ContinuousVariable("Rotatable Bonds"),
            ContinuousVariable("TPSA"),
            ContinuousVariable("PAINS Match"),
            ContinuousVariable("Veber Rule"),
            ContinuousVariable("Reactivity"),
            ContinuousVariable("Drug Score"),
        ]

        v_smiles = StringVariable("SMILES")
        v_smiles.attributes["format"] = "SMILES"
        v_can = StringVariable("Canonical SMILES")
        v_can.attributes["format"] = "SMILES"

        if include_atoms:
            meta_vars = [
                v_smiles,
                v_can,
                StringVariable("PAINS regID"),
                StringVariable("Criteria"),
                StringVariable("Highlighted Atoms"),
            ]
        else:
            meta_vars = [
                v_smiles,
                v_can,
                StringVariable("PAINS regID"),
                StringVariable("Criteria"),
            ]

        X = np.array(
            [
                [
                    r.qed_score,
                    r.lipinski_violations,
                    r.mw,
                    r.logp,
                    r.hbd,
                    r.hba,
                    r.rotatable_bonds,
                    r.tpsa,
                    r.pains_match,
                    r.veber_rule,
                    r.reactivity,
                    r.drug_score,
                ]
                for r in rows
            ],
            dtype=float,
        )

        if include_atoms:
            metas = np.array(
                [[r.smiles, r.canonical_smiles, r.pains_regid, r.criteria, r.highlighted_atoms or ""] for r in rows],
                dtype=object,
            )
        else:
            metas = np.array(
                [[r.smiles, r.canonical_smiles, r.pains_regid, r.criteria] for r in rows],
                dtype=object,
            )

        domain = Domain(features, metas=meta_vars)
        return Table.from_numpy(domain, X=X, metas=metas)

    def onDeleteWidget(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        super().onDeleteWidget()
