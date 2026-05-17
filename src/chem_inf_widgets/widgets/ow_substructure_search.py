from __future__ import annotations

from typing import List, Optional

import numpy as np
from AnyQt.QtCore import Qt, QTimer, pyqtSlot as Slot
from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.from_orange import TableMolConversionReport, table_to_chemmols_with_report
from chem_inf_widgets.chemcore.services.substructure_search_service import (
    SearchConfig,
    SearchHit,
    normalize_query_string,
    search_smiles,
)
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_error_status,
    format_skip_warning,
    format_table_report,
    set_widget_error,
    set_widget_warning,
)


def _find_smiles_meta_idx(table: Table) -> int:
    metas = list(table.domain.metas or [])
    for i, v in enumerate(metas):
        if (v.name or "").strip().lower() == "smiles":
            return i
    return 0


def _table_first_string(table: Table) -> str:
    if table is None or len(table) == 0:
        return ""
    # prefer meta "SMILES" if present
    metas = list(table.domain.metas or [])
    idx = None
    for i, v in enumerate(metas):
        if (v.name or "").strip().lower() == "smiles":
            idx = i
            break
    if idx is None:
        idx = 0 if metas else None
    if idx is None:
        return ""
    v = table[0].metas[idx]
    return (str(v) if v is not None else "").strip()


class OWSubstructureSearch(OWWidget):
    name = "Substructure & Similarity Search"
    description = "Query (string/ChemMol) + Compounds (Table) → filtered Table with highlight indices."
    icon = "icons/standardization_filtering/owsubstructurewidget.png"
    priority = 115
    want_main_area = False
    resizing_enabled = True

    class Inputs:
        query_text = Input("Query", str, auto_summary=False)
        query_table = Input("Query Data", Table)
        query_molecule = Input("Query Molecule", ChemMol, auto_summary=False)
        compounds = Input("Compounds", Table)

    class Outputs:
        filtered = Output("Filtered Compounds", Table)

    # settings
    search_type: str = Setting("substructure")
    similarity_threshold: float = Setting(0.3)
    fp_type: str = Setting("morgan")
    highlight_atoms: bool = Setting(True)

    _query: str = Setting("")

    def __init__(self) -> None:
        super().__init__()
        self._compounds: Optional[Table] = None
        self._executor = ThreadExecutor(self)
        self._compounds_report: Optional[TableMolConversionReport] = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._apply_search)

        self._build_ui()
        self._apply_style()
        self._set_status("Awaiting query + compounds…", ok=True)

    # ---------------- UI ----------------

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
        QLabel#HdrTitle { font-size: 15px; font-weight: 650; }
        QLabel#HdrSub { color: #6B7280; font-size: 12px; }
        QLabel#Chip {
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
        left.addWidget(QLabel(self.name, objectName="HdrTitle"))
        left.addWidget(QLabel("Input1: Query (str/ChemMol) • Input2: Compounds (Table)", objectName="HdrSub"))
        hl.addLayout(left, 1)

        self.lbl_status = QLabel("Ready", objectName="Chip")
        self.lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hl.addWidget(self.lbl_status)

        self.controlArea.layout().addWidget(hdr)

        box = QGroupBox("Search")
        vl = QVBoxLayout(box)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Type"))
        self.cmb_type = QComboBox()
        self.cmb_type.addItems(["substructure", "superstructure", "similarity", "exact"])
        self.cmb_type.setCurrentText(self.search_type)
        self.cmb_type.currentTextChanged.connect(self._on_type_changed)
        row1.addWidget(self.cmb_type, 1)
        vl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Query (SMILES/SMARTS)"))
        self.ed_query = QLineEdit()
        self.ed_query.setText(self._query)
        self.ed_query.textChanged.connect(self._on_query_edited)
        row2.addWidget(self.ed_query, 1)
        vl.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Similarity ≥"))
        self.spin_thr = QDoubleSpinBox()
        self.spin_thr.setRange(0.0, 1.0)
        self.spin_thr.setSingleStep(0.05)
        self.spin_thr.setDecimals(2)
        self.spin_thr.setValue(float(self.similarity_threshold))
        self.spin_thr.valueChanged.connect(self._on_thr_changed)
        row3.addWidget(self.spin_thr, 1)
        vl.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Fingerprint"))
        self.cmb_fp = QComboBox()
        self.cmb_fp.addItems(["morgan", "rdkit"])
        self.cmb_fp.setCurrentText(self.fp_type)
        self.cmb_fp.currentTextChanged.connect(self._on_fp_changed)
        row4.addWidget(self.cmb_fp, 1)
        vl.addLayout(row4)

        self.chk_highlight = QCheckBox("Return highlighted atoms (substructure)")
        self.chk_highlight.setChecked(bool(self.highlight_atoms))
        self.chk_highlight.toggled.connect(self._on_highlight_changed)
        vl.addWidget(self.chk_highlight)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self._apply_search)
        vl.addWidget(self.btn_apply)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        vl.addWidget(self.progress)

        self.controlArea.layout().addWidget(box)
        self.controlArea.layout().addStretch(1)

        self._update_controls_for_type()

    # ---------------- helpers ----------------

    def _update_controls_for_type(self) -> None:
        st = self.cmb_type.currentText()
        is_sim = (st == "similarity")
        self.spin_thr.setEnabled(is_sim)
        self.cmb_fp.setEnabled(is_sim)

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
        self.btn_apply.setEnabled(not busy)
        self.ed_query.setEnabled(not busy)
        self.cmb_type.setEnabled(not busy)
        self.spin_thr.setEnabled(not busy and self.cmb_type.currentText() == "similarity")
        self.cmb_fp.setEnabled(not busy and self.cmb_type.currentText() == "similarity")
        self.chk_highlight.setEnabled(not busy)
        self.progress.setVisible(busy)
        if msg:
            self._set_status(msg, ok=True)

    # ---------------- Inputs ----------------

    @Inputs.query_text
    def set_query_text(self, query: Optional[str]) -> None:
        q = (query or "").strip()
        if not q:
            return
        self._query = q
        if self.ed_query.text() != self._query:
            self.ed_query.setText(self._query)
        self._debounce.start(200)

    @Inputs.query_table
    def set_query_table(self, t: Optional[Table]) -> None:
        q = _table_first_string(t) if t is not None else ""
        if q:
            self.set_query_text(q)

    @Inputs.query_molecule
    def set_query_molecule(self, cm: Optional[ChemMol]) -> None:
        if cm is None or cm.mol is None:
            return
        try:
            q = cm.canonical_smiles(remove_hs=True, canonical=True, isomeric=True).strip()
        except Exception:
            try:
                q = cm.smiles().strip()
            except Exception:
                q = ""
        if q:
            self.set_query_text(q)

    @Inputs.compounds
    def set_compounds(self, data: Optional[Table]) -> None:
        clear_widget_messages(self)
        self._compounds = data
        self._compounds_report = None
        if data is None or len(data) == 0:
            self._set_status("No compounds input.", ok=False)
            self.Outputs.filtered.send(None)
            return
        try:
            _mols, self._compounds_report = table_to_chemmols_with_report(data)
        except Exception:
            self._compounds_report = None
        if self._compounds_report is not None:
            self._set_status(format_table_report(self._compounds_report, prefix="Compounds"), ok=True)
        else:
            self._set_status(f"Compounds: {len(data)} rows", ok=True)
        self._debounce.start(250)

    # ---------------- Events ----------------

    def _on_type_changed(self, txt: str) -> None:
        self.search_type = txt
        self._update_controls_for_type()
        self._debounce.start(250)

    def _on_query_edited(self, txt: str) -> None:
        self._query = (txt or "").strip()
        self._debounce.start(350)

    def _on_thr_changed(self, v: float) -> None:
        self.similarity_threshold = float(v)
        if self.search_type == "similarity":
            self._debounce.start(250)

    def _on_fp_changed(self, txt: str) -> None:
        self.fp_type = txt
        if self.search_type == "similarity":
            self._debounce.start(250)

    def _on_highlight_changed(self, v: bool) -> None:
        self.highlight_atoms = bool(v)
        if self.search_type == "substructure":
            self._debounce.start(250)

    # ---------------- Search (async) ----------------

    def _apply_search(self) -> None:
        if self._compounds is None or len(self._compounds) == 0:
            self._set_status("Awaiting compounds…", ok=False)
            self.Outputs.filtered.send(None)
            return

        if not self._query:
            self._set_status("Awaiting query…", ok=False)
            self.Outputs.filtered.send(None)
            return

        q = normalize_query_string(self._query, self.search_type)
        set_widget_warning(
            self,
            format_skip_warning(
                0 if self._compounds_report is None else self._compounds_report.n_invalid,
                subject="invalid compounds",
                action="will be skipped during search",
            ),
        )

        smiles_idx = _find_smiles_meta_idx(self._compounds)
        smiles_col = self._compounds.metas[:, smiles_idx]

        cfg = SearchConfig(
            search_type=self.search_type,
            similarity_threshold=float(self.similarity_threshold),
            fp_type=self.fp_type,
            return_highlight_atoms=bool(self.highlight_atoms),
        )

        self._set_busy(True, "Searching…")

        def progress_cb(i: int, n: int) -> None:
            pct = int(100.0 * i / max(1, n))
            methodinvoke(self, "_set_progress", (int,))(pct)

        fut = self._executor.submit(search_smiles, list(smiles_col), q, cfg, progress_cb)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut) -> None:
        try:
            hits = fut.result()
            methodinvoke(self, "_finish", (object,))(hits)
        except Exception as e:
            methodinvoke(self, "_fail", (str,))(str(e))

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
    def _finish(self, hits_obj: object) -> None:
        hits: List[SearchHit] = hits_obj or []
        if self._compounds is None:
            self._fail("Missing compounds input.")
            return

        if not hits:
            self._set_busy(False)
            status = "No matches."
            if self._compounds_report is not None:
                status += f" Valid compounds={self._compounds_report.n_valid}, invalid={self._compounds_report.n_invalid}."
            self._set_status(status, ok=True)
            self.Outputs.filtered.send(None)
            return

        out = self._build_output_table(self._compounds, hits)
        self.Outputs.filtered.send(out)

        self._set_busy(False)
        status = format_done_status(f"Matches={len(hits)}")
        if self._compounds_report is not None:
            status += f" | valid compounds={self._compounds_report.n_valid}, invalid={self._compounds_report.n_invalid}"
        self._set_status(status, ok=True)

    def _build_output_table(self, data: Table, hits: List[SearchHit]) -> Table:
        rows = [data[h.idx] for h in hits]

        v_hl = StringVariable("Highlighted Atoms")
        v_sim = ContinuousVariable("Similarity")

        metas_old = list(data.domain.metas)
        metas_new = metas_old + [v_hl, v_sim]
        domain = Domain(data.domain.attributes, data.domain.class_vars, metas=metas_new)

        X = np.array([r.x for r in rows], dtype=float)
        Y = np.array([r.y for r in rows], dtype=float) if data.domain.class_vars else None

        metas = []
        for r, h in zip(rows, hits):
            sim_val = h.similarity if np.isfinite(h.similarity) else np.nan
            metas.append(list(r.metas) + [h.highlighted_atoms_csv, sim_val])

        metas = np.array(metas, dtype=object)
        return Table.from_numpy(domain, X=X, Y=Y, metas=metas)

    def onDeleteWidget(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWSubstructureSearch).run()
