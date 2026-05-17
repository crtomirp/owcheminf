from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg

from AnyQt.QtCore import Qt, pyqtSlot as Slot
from AnyQt.QtGui import QImage, QPixmap
from AnyQt.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.atom_contribution_service import (
    AtomContributionResult,
    explain_molecule,
)

pg.setConfigOptions(antialias=True)

_C_POS = (34, 197, 94, 200)   # green — positive contribution
_C_NEG = (239, 68, 68, 200)   # red — negative contribution


# ── Orange ↔ numpy helpers ────────────────────────────────────────────────

def _table_to_arrays(data: Table):
    """Returns (X, feature_names, meta_dict) where meta_dict maps name→column."""
    n = len(data)
    attr_vars = list(data.domain.attributes)
    X = np.array(data.X, dtype=float) if attr_vars else np.empty((n, 0), dtype=float)
    feature_names = [v.name for v in attr_vars]

    meta_vars = list(data.domain.metas)
    metas: dict[str, np.ndarray] = {}
    if meta_vars and data.metas is not None and data.metas.size:
        M = data.metas
        for i, v in enumerate(meta_vars):
            col = M[:, i]
            metas[v.name] = np.array([str(x) if x is not None else "" for x in col])

    class_vars = list(data.domain.class_vars)
    if class_vars:
        Y = np.array(data.Y, dtype=float).reshape(n, -1)
        for i, v in enumerate(class_vars):
            pass  # not needed here

    return X, feature_names, metas


def _find_smiles_col(metas: dict[str, np.ndarray]) -> Optional[str]:
    for name in metas:
        if name.strip().lower() in {"smiles", "smi", "canonical_smiles"}:
            return name
    return None


def _find_id_col(metas: dict[str, np.ndarray], preferred: str) -> Optional[str]:
    if preferred and preferred in metas:
        return preferred
    for name in metas:
        if name.strip().lower() in {"compound_id", "chembl_id", "mol_id", "id", "name"}:
            return name
    return None


# ── Widget ────────────────────────────────────────────────────────────────

class OWAtomContributionMap(OWWidget):
    name = "Atom Contribution Map"
    description = (
        "Visualise per-atom contributions to QSAR/QSPR predictions using "
        "SHAP values and atom-level descriptor decomposition."
    )
    icon = "icons/modeling/ow_atom_contribution_map.png"
    priority = 146
    keywords = ["QSAR", "SHAP", "interpretability", "atom", "heatmap", "explainability"]
    want_main_area = True
    resizing_enabled = True

    class Inputs:
        model = Input("Model", object, auto_summary=False)
        data  = Input("Data", Table)

    class Outputs:
        contributions = Output("Contributions", Table)

    smiles_column: str = Setting("")
    id_column: str = Setting("")
    fp_radius: int = Setting(2)

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None
        self._data: Optional[Table] = None
        self._X: Optional[np.ndarray] = None
        self._feature_names: list[str] = []
        self._metas: dict = {}
        self._smiles_col: Optional[str] = None
        self._id_col: Optional[str] = None
        self._results: list[AtomContributionResult] = []
        self._executor = ThreadExecutor(self)

        self._build_control_area()
        self._build_main_area()
        self._set_status("Awaiting model + data…", ok=True)

    # ── Control area ──────────────────────────────────────────────────────

    def _build_control_area(self) -> None:
        ca = self.controlArea

        hdr = QWidget()
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(6, 6, 6, 6)
        left = QVBoxLayout()
        left.addWidget(QLabel("Atom Contribution Map", objectName="HdrTitle"))
        left.addWidget(QLabel("SHAP + per-atom importance visualisation", objectName="HdrSub"))
        hl.addLayout(left, 1)
        self._lbl_status = QLabel("Ready", objectName="StatusChip")
        self._lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hl.addWidget(self._lbl_status)
        ca.layout().addWidget(hdr)

        col_box = QGroupBox("Columns")
        col_vl = QVBoxLayout(col_box)

        row_s = QHBoxLayout()
        row_s.addWidget(QLabel("SMILES"))
        self._cmb_smiles = QComboBox()
        self._cmb_smiles.currentTextChanged.connect(self._on_col_changed)
        row_s.addWidget(self._cmb_smiles, 1)
        col_vl.addLayout(row_s)

        row_i = QHBoxLayout()
        row_i.addWidget(QLabel("ID"))
        self._cmb_id = QComboBox()
        self._cmb_id.currentTextChanged.connect(self._on_col_changed)
        row_i.addWidget(self._cmb_id, 1)
        col_vl.addLayout(row_i)

        ca.layout().addWidget(col_box)

        opt_box = QGroupBox("Options")
        opt_vl = QVBoxLayout(opt_box)
        row_r = QHBoxLayout()
        row_r.addWidget(QLabel("FP radius"))
        from AnyQt.QtWidgets import QSpinBox
        self._spin_r = QSpinBox()
        self._spin_r.setRange(1, 4)
        self._spin_r.setValue(int(self.fp_radius))
        row_r.addWidget(self._spin_r, 1)
        opt_vl.addLayout(row_r)

        self._btn_run = QPushButton("Explain all molecules")
        self._btn_run.clicked.connect(self._run)
        opt_vl.addWidget(self._btn_run)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        opt_vl.addWidget(self._progress)

        ca.layout().addWidget(opt_box)
        ca.layout().addStretch(1)

    # ── Main area ─────────────────────────────────────────────────────────

    def _build_main_area(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # Left: molecule list
        left_w = QWidget()
        left_vl = QVBoxLayout(left_w)
        left_vl.setContentsMargins(4, 4, 4, 4)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter molecules…")
        self._search.textChanged.connect(self._filter_list)
        left_vl.addWidget(self._search)
        self._mol_list = QListWidget()
        self._mol_list.currentRowChanged.connect(self._on_mol_selected)
        left_vl.addWidget(self._mol_list)
        splitter.addWidget(left_w)

        # Right: tabs
        self._tabs = QTabWidget()

        # Tab 1: Atom heatmap
        heatmap_w = QWidget()
        hm_vl = QVBoxLayout(heatmap_w)
        self._lbl_pred = QLabel("Select a molecule")
        self._lbl_pred.setStyleSheet("font-size:14px; font-weight:600; padding:6px;")
        self._lbl_pred.setAlignment(Qt.AlignCenter)
        hm_vl.addWidget(self._lbl_pred)
        self._lbl_mol_img = QLabel()
        self._lbl_mol_img.setAlignment(Qt.AlignCenter)
        self._lbl_mol_img.setMinimumSize(400, 260)
        hm_vl.addWidget(self._lbl_mol_img, 1)
        legend_row = QHBoxLayout()
        legend_row.addStretch(1)
        for color, text in (("#4ade80", "↑ Increases prediction"), ("#f87171", "↓ Decreases prediction")):
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{color}; font-size:16px;")
            legend_row.addWidget(dot)
            legend_row.addWidget(QLabel(text))
            legend_row.addSpacing(12)
        legend_row.addStretch(1)
        hm_vl.addLayout(legend_row)
        self._tabs.addTab(heatmap_w, "Atom Heatmap")

        # Tab 2: SHAP waterfall (horizontal bar chart)
        shap_w = QWidget()
        shap_vl = QVBoxLayout(shap_w)
        shap_vl.setContentsMargins(0, 0, 0, 0)
        self._pw_shap = pg.PlotWidget(background="w")
        self._pw_shap.setLabel("bottom", "SHAP value (impact on prediction)")
        self._pw_shap.showGrid(x=True, y=False, alpha=0.18)
        self._pw_shap.getAxis("bottom").setPen(pg.mkPen("#CBD5E1"))
        self._pw_shap.getAxis("left").setPen(pg.mkPen("#CBD5E1"))
        self._hover_shap = QLabel("  Hover a bar to see feature details")
        self._hover_shap.setStyleSheet("color:#64748B; font-size:11px;")
        shap_vl.addWidget(self._pw_shap)
        shap_vl.addWidget(self._hover_shap)
        self._tabs.addTab(shap_w, "SHAP Waterfall")

        # Tab 3: Global feature importance
        global_w = QWidget()
        global_vl = QVBoxLayout(global_w)
        global_vl.setContentsMargins(0, 0, 0, 0)
        self._pw_global = pg.PlotWidget(background="w")
        self._pw_global.setLabel("bottom", "Mean |SHAP value|")
        self._pw_global.showGrid(x=True, y=False, alpha=0.18)
        self._pw_global.getAxis("bottom").setPen(pg.mkPen("#CBD5E1"))
        self._pw_global.getAxis("left").setPen(pg.mkPen("#CBD5E1"))
        global_vl.addWidget(self._pw_global)
        self._tabs.addTab(global_w, "Global Importance")

        splitter.addWidget(self._tabs)
        splitter.setSizes([200, 600])
        self.mainArea.layout().addWidget(splitter)

    # ── Status ────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, ok: bool = True) -> None:
        self._lbl_status.setText(msg)
        self._lbl_status.setStyleSheet(
            "padding:4px 8px;border:1px solid #e1e1e1;border-radius:10px;background:#fafafa;"
            if ok else
            "padding:4px 8px;border:1px solid #f2c2c2;border-radius:10px;"
            "background:#fff5f5;color:#a40000;"
        )

    def _set_busy(self, busy: bool) -> None:
        self._btn_run.setEnabled(not busy)
        self._progress.setVisible(busy)

    # ── Inputs ────────────────────────────────────────────────────────────

    @Inputs.model
    def set_model(self, model) -> None:
        self._pipeline = model
        self._maybe_run()

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._data = data
        if data is None:
            self._X = None
            self._feature_names = []
            self._metas = {}
            self._mol_list.clear()
            self._cmb_smiles.clear()
            self._cmb_id.clear()
            self._set_status("No data.", ok=False)
            return

        self._X, self._feature_names, self._metas = _table_to_arrays(data)
        self._refresh_col_combos()
        self._set_status(f"{len(data)} rows, {len(self._feature_names)} features", ok=True)
        self._maybe_run()

    def _refresh_col_combos(self) -> None:
        meta_names = list(self._metas.keys())
        for cmb in (self._cmb_smiles, self._cmb_id):
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItem("")
            cmb.addItems(meta_names)
            cmb.blockSignals(False)
        # Auto-detect
        sc = _find_smiles_col(self._metas)
        if sc:
            idx = self._cmb_smiles.findText(sc)
            if idx >= 0:
                self._cmb_smiles.setCurrentIndex(idx)
        ic = _find_id_col(self._metas, self.id_column)
        if ic:
            idx = self._cmb_id.findText(ic)
            if idx >= 0:
                self._cmb_id.setCurrentIndex(idx)

    def _on_col_changed(self) -> None:
        self.smiles_column = self._cmb_smiles.currentText()
        self.id_column = self._cmb_id.currentText()

    def _maybe_run(self) -> None:
        if self._pipeline is not None and self._data is not None:
            self._run()

    # ── Explanation run ───────────────────────────────────────────────────

    def _run(self) -> None:
        if self._pipeline is None:
            self._set_status("No model connected.", ok=False)
            return
        if self._data is None or self._X is None or self._X.shape[1] == 0:
            self._set_status("No feature data.", ok=False)
            return

        smiles_col = self._cmb_smiles.currentText()
        if not smiles_col or smiles_col not in self._metas:
            self._set_status("Select SMILES column.", ok=False)
            return

        smiles_arr = self._metas[smiles_col]
        id_col = self._cmb_id.currentText()
        ids_arr = self._metas.get(id_col, np.array([str(i) for i in range(len(smiles_arr))]))
        X = self._X
        pipeline = self._pipeline
        feature_names = self._feature_names
        fp_radius = int(self._spin_r.value())

        self._set_busy(True)
        self._set_status("Computing SHAP…", ok=True)

        def _worker():
            results = []
            for i in range(len(smiles_arr)):
                smi = str(smiles_arr[i])
                cid = str(ids_arr[i])
                X_row = X[i]
                # Use all other rows as background (capped at 200 for speed)
                bg_idx = np.delete(np.arange(len(X)), i)
                if len(bg_idx) > 200:
                    rng = np.random.default_rng(42)
                    bg_idx = rng.choice(bg_idx, 200, replace=False)
                X_bg = X[bg_idx]
                try:
                    r = explain_molecule(smi, cid, pipeline, X_row, X_bg, feature_names, fp_radius)
                    results.append(r)
                except Exception:
                    results.append(None)
            return results

        fut = self._executor.submit(_worker)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut) -> None:
        try:
            results = fut.result()
            methodinvoke(self, "_finish", (object,))(results)
        except Exception as e:
            methodinvoke(self, "_fail", (str,))(str(e))

    @Slot(object)
    def _finish(self, results: object) -> None:
        self._results = [r for r in results if r is not None]
        self._set_busy(False)
        self._populate_list()
        self._populate_global_importance()
        self._send_contributions()
        n = len(self._results)
        self._set_status(f"Explained {n} molecules", ok=True)
        if self._results:
            self._mol_list.setCurrentRow(0)

    @Slot(str)
    def _fail(self, msg: str) -> None:
        self._set_busy(False)
        self._set_status("Error", ok=False)
        self._lbl_pred.setText(f"Error: {msg}")

    # ── Populate UI ───────────────────────────────────────────────────────

    def _populate_list(self) -> None:
        self._mol_list.clear()
        for r in self._results:
            item = QListWidgetItem(f"{r.compound_id}   pred={r.prediction:.3f}")
            self._mol_list.addItem(item)

    def _filter_list(self, text: str) -> None:
        for i in range(self._mol_list.count()):
            item = self._mol_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def _on_mol_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._results):
            return
        r = self._results[row]
        self._show_heatmap(r)
        self._show_shap_waterfall(r)

    def _show_heatmap(self, r: AtomContributionResult) -> None:
        unit_suffix = ""
        self._lbl_pred.setText(f"Predicted: {r.prediction:.4f}{unit_suffix}   (SHAP baseline: {r.baseline:.4f})")
        if r.svg_bytes:
            img = QImage.fromData(r.svg_bytes, "SVG")
            if not img.isNull():
                pix = QPixmap.fromImage(img)
                self._lbl_mol_img.setPixmap(
                    pix.scaled(self._lbl_mol_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                return
        self._lbl_mol_img.setText("Atom map not available for this molecule/descriptor type.")

    def _show_shap_waterfall(self, r: AtomContributionResult) -> None:
        self._pw_shap.clear()
        sv = r.shap_values
        names = r.feature_names
        if sv is None or len(sv) == 0:
            return

        # Show top N features sorted by |SHAP|
        n_show = min(25, len(sv))
        order = np.argsort(np.abs(sv))[-n_show:]  # top by magnitude
        vals = sv[order]
        feat_names = [names[i] for i in order]

        y_pos = np.arange(len(vals), dtype=float)
        colors = [_C_POS if v >= 0 else _C_NEG for v in vals]

        bars = pg.BarGraphItem(
            x0=np.zeros(len(vals)),
            x1=vals,
            y=y_pos,
            height=0.65,
            brushes=[pg.mkBrush(*c) for c in colors],
            pens=[pg.mkPen(None)] * len(vals),
        )
        self._pw_shap.addItem(bars)

        zero_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#CBD5E1", width=1))
        self._pw_shap.addItem(zero_line)

        ax = self._pw_shap.getAxis("left")
        ax.setTicks([list(zip(y_pos, feat_names))])

        # Hover overlay
        sc = pg.ScatterPlotItem(
            x=vals, y=y_pos, size=1,
            pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 0, 0),
            hoverable=True, hoverSize=10,
        )
        lbl = self._hover_shap

        def _on_hover(scatter, pts, ev):
            if pts:
                idx_pt = pts[0].index()
                if 0 <= idx_pt < len(feat_names):
                    lbl.setText(f"  {feat_names[idx_pt]}   SHAP = {vals[idx_pt]:.4f}")
            else:
                lbl.setText("  Hover a bar to see feature details")

        sc.sigHovered.connect(_on_hover)
        self._pw_shap.addItem(sc)

    def _populate_global_importance(self) -> None:
        self._pw_global.clear()
        if not self._results:
            return
        all_sv = np.array([r.shap_values for r in self._results if r.shap_values is not None])
        if all_sv.ndim != 2 or all_sv.shape[0] == 0:
            return
        mean_abs = np.mean(np.abs(all_sv), axis=0)
        names = self._results[0].feature_names

        n_show = min(30, len(mean_abs))
        order = np.argsort(mean_abs)[-n_show:]
        vals = mean_abs[order]
        feat_names = [names[i] for i in order]
        y_pos = np.arange(len(vals), dtype=float)

        bars = pg.BarGraphItem(
            x0=np.zeros(len(vals)), x1=vals,
            y=y_pos, height=0.65,
            brush=pg.mkBrush(37, 99, 235, 180),
            pen=pg.mkPen(None),
        )
        self._pw_global.addItem(bars)
        ax = self._pw_global.getAxis("left")
        ax.setTicks([list(zip(y_pos, feat_names))])

    # ── Output ────────────────────────────────────────────────────────────

    def _send_contributions(self) -> None:
        if not self._results:
            self.Outputs.contributions.send(None)
            return
        rows = []
        for r in self._results:
            row = {"compound_id": r.compound_id, "smiles": r.smiles, "prediction": r.prediction}
            for i, name in enumerate(r.feature_names):
                row[f"shap_{name}"] = float(r.shap_values[i]) if r.shap_values is not None else np.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        attrs, metas, X_cols, M_cols = [], [], [], []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                attrs.append(ContinuousVariable(str(col)))
                X_cols.append(col)
            else:
                metas.append(StringVariable(str(col)))
                M_cols.append(col)
        X = df[X_cols].to_numpy(dtype=float) if X_cols else np.empty((len(df), 0))
        M = df[M_cols].fillna("").astype(str).to_numpy(dtype=object) if M_cols else np.empty((len(df), 0), dtype=object)
        self.Outputs.contributions.send(Table.from_numpy(Domain(attrs, metas=metas), X=X, metas=M))

    def onDeleteWidget(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWAtomContributionMap).run()
