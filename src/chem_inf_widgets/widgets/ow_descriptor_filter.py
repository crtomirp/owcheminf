from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg

from AnyQt.QtCore import Qt, pyqtSlot as Slot
from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,

    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.orange_table_utils import looks_like_meta_key, safe_table_from_numpy
from chem_inf_widgets.chemcore.services.descriptor_filter_service import (
    DescriptorFilterConfig,
    DescriptorFilterResult,
    run_descriptor_filter,
)

pg.setConfigOptions(antialias=True)


# ── Orange ↔ pandas ────────────────────────────────────────────────────────

def _table_to_df(data: Table) -> pd.DataFrame:
    """Convert an Orange table to pandas without repeated full-matrix copies.

    The previous implementation converted ``data.X`` to a new numpy array once
    per descriptor column. With descriptor tables this can allocate hundreds or
    thousands of full copies and can make the OS kill Orange. Here we make at
    most one copy/view of X, one of Y, and then expose per-column views to
    pandas.
    """
    n = len(data)
    cols: dict = {}

    X = np.asarray(data.X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(n, -1)
    for i, v in enumerate(data.domain.attributes):
        cols[v.name] = X[:, i].copy() if i < X.shape[1] else np.full(n, np.nan, dtype=float)

    if data.domain.class_vars:
        Y = np.asarray(data.Y, dtype=float).reshape(n, -1)
        for i, v in enumerate(data.domain.class_vars):
            cols[v.name] = Y[:, i].copy() if i < Y.shape[1] else np.full(n, np.nan, dtype=float)

    if data.domain.metas is not None and data.metas is not None and data.metas.size:
        metas = np.asarray(data.metas, dtype=object)
        for i, v in enumerate(data.domain.metas):
            col = metas[:, i]
            if isinstance(v, StringVariable):
                cols[v.name] = ["" if x is None else str(x) for x in col]
            else:
                try:
                    cols[v.name] = pd.to_numeric(pd.Series(col), errors="coerce").to_numpy(dtype=float)
                except Exception:
                    cols[v.name] = ["" if x is None else str(x) for x in col]
    return pd.DataFrame(cols, index=range(n))


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with non-empty, unique string column names.

    Orange domains are much less forgiving than pandas. Duplicate names or
    blank names can trigger vague Qt/Orange errors when a widget sends output.
    """
    out = df.copy()
    used: dict[str, int] = {}
    names: list[str] = []
    for raw in out.columns:
        base = str(raw).strip() or "column"
        count = used.get(base, 0)
        used[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count + 1}")
    out.columns = names
    return out


def _df_to_table(df: pd.DataFrame, class_col: str = "", *, modeling_clean: bool = False) -> Table:
    """Convert a pandas table to Orange with stable modeling-friendly roles.

    In modeling-clean mode, obvious audit/provenance columns are removed, while
    ``SMILES`` and ``inchikey`` stay as metas and numeric descriptor/endpoint
    columns stay as attributes/class variables.
    """
    df = _normalise_column_names(df)
    class_col = str(class_col or "").strip()

    if modeling_clean:
        audit_prefixes = (
            "qc_", "standardization_", "curation_", "import_", "std_",
        )
        audit_names = {
            "smiles_orig", "smiles_std", "input_smiles", "canonical_smiles",
            "standardized_smiles", "inchi", "qc_issues", "qc_issue_codes",
        }
        keep_cols = []
        for col in df.columns:
            key = str(col).strip().lower()
            if key in {"smiles", "inchikey"} or key == class_col.lower():
                keep_cols.append(col)
            elif key in audit_names or key.startswith(audit_prefixes):
                continue
            else:
                keep_cols.append(col)
        df = df.loc[:, keep_cols].copy()

    attrs, class_vars, metas = [], [], []
    X_cols, Y_cols, M_cols = [], [], []
    for col in df.columns:
        key = str(col).strip() or "column"
        series = df[col]
        is_target = bool(class_col and key == class_col)
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # The selected target must remain a class variable even if its name
        # looks like metadata (for example activity_value or pActivity).
        if is_target:
            vals = pd.to_numeric(series, errors="coerce")
            class_vars.append(ContinuousVariable(key))
            Y_cols.append(col)
            df[col] = vals
        elif is_numeric and not looks_like_meta_key(key):
            df[col] = pd.to_numeric(series, errors="coerce")
            attrs.append(ContinuousVariable(key))
            X_cols.append(col)
        else:
            metas.append(StringVariable(key))
            M_cols.append(col)

    X = df[X_cols].to_numpy(dtype=float) if X_cols else np.empty((len(df), 0), dtype=float)
    Y = df[Y_cols].to_numpy(dtype=float) if Y_cols else None
    M = df[M_cols].fillna("").astype(str).to_numpy(dtype=object) if M_cols else None
    domain = Domain(attrs, class_vars=class_vars or None, metas=metas or None)
    return safe_table_from_numpy(domain, X=X, Y=Y, metas=M)


# ── Widget ────────────────────────────────────────────────────────────────

class OWDescriptorFilter(OWWidget):
    name = "Descriptor Pre-selector"
    description = (
        "Remove uninformative descriptors: high missing-value rate, "
        "near-zero variance, and high inter-feature correlation."
    )
    icon = "icons/modeling/ow_descriptor_filter.png"
    priority = 132
    keywords = ["descriptor", "feature selection", "variance", "correlation", "filter", "QSAR"]
    want_main_area = True
    resizing_enabled = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        filtered_data = Output("Filtered Data", Table)
        modeling_data = Output("Modeling Data", Table)
        report       = Output("Filter Report", Table)

    max_missing_fraction: float = Setting(0.20)
    min_variance: float         = Setting(0.01)
    max_correlation: float      = Setting(0.90)
    correlation_method: int     = Setting(0)   # 0=pearson, 1=spearman
    target_column: str          = Setting("")
    auto_run: bool              = Setting(True)
    max_correlation_features: int = Setting(1500)
    max_features_before_correlation: int = Setting(3000)
    max_output_features: int = Setting(1000)
    report_display_limit: int = Setting(1000)

    _CORR_METHODS = ["pearson", "spearman"]

    def __init__(self) -> None:
        super().__init__()
        self._data: Optional[Table] = None
        self._executor = ThreadExecutor(self)
        self._result: Optional[DescriptorFilterResult] = None

        self._build_control_area()
        self._build_main_area()
        self._set_status("Awaiting data…", ok=True)

    # ── Control area ──────────────────────────────────────────────────────

    def _build_control_area(self) -> None:
        ca = self.controlArea

        hdr = QWidget()
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(6, 6, 6, 6)
        lv = QVBoxLayout()
        lv.addWidget(QLabel("Descriptor Pre-selector", objectName="HdrTitle"))
        lv.addWidget(QLabel("Remove uninformative & redundant features", objectName="HdrSub"))
        hl.addLayout(lv, 1)
        self._lbl_status = QLabel("Ready", objectName="StatusChip")
        self._lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hl.addWidget(self._lbl_status)
        ca.layout().addWidget(hdr)

        # Filter thresholds
        flt = QGroupBox("Filters")
        flt_vl = QVBoxLayout(flt)

        row_miss = QHBoxLayout()
        row_miss.addWidget(QLabel("Max missing"))
        self._spin_miss = QDoubleSpinBox()
        self._spin_miss.setRange(0.0, 1.0)
        self._spin_miss.setSingleStep(0.05)
        self._spin_miss.setDecimals(2)
        self._spin_miss.setSuffix(" (fraction)")
        self._spin_miss.setValue(float(self.max_missing_fraction))
        self._spin_miss.valueChanged.connect(self._on_changed)
        row_miss.addWidget(self._spin_miss, 1)
        flt_vl.addLayout(row_miss)

        row_var = QHBoxLayout()
        row_var.addWidget(QLabel("Min variance"))
        self._spin_var = QDoubleSpinBox()
        self._spin_var.setRange(0.0, 100.0)
        self._spin_var.setSingleStep(0.001)
        self._spin_var.setDecimals(5)
        self._spin_var.setValue(float(self.min_variance))
        self._spin_var.valueChanged.connect(self._on_changed)
        row_var.addWidget(self._spin_var, 1)
        flt_vl.addLayout(row_var)

        row_corr = QHBoxLayout()
        row_corr.addWidget(QLabel("Max |correlation|"))
        self._spin_corr = QDoubleSpinBox()
        self._spin_corr.setRange(0.5, 1.0)
        self._spin_corr.setSingleStep(0.05)
        self._spin_corr.setDecimals(2)
        self._spin_corr.setValue(float(self.max_correlation))
        self._spin_corr.valueChanged.connect(self._on_changed)
        row_corr.addWidget(self._spin_corr, 1)
        flt_vl.addLayout(row_corr)

        row_cm = QHBoxLayout()
        row_cm.addWidget(QLabel("Corr method"))
        self._cmb_corr = QComboBox()
        self._cmb_corr.addItems(self._CORR_METHODS)
        self._cmb_corr.setCurrentIndex(int(self.correlation_method))
        self._cmb_corr.currentIndexChanged.connect(self._on_changed)
        row_cm.addWidget(self._cmb_corr, 1)
        flt_vl.addLayout(row_cm)

        row_limit = QHBoxLayout()
        row_limit.addWidget(QLabel("Full corr limit"))
        self._spin_corr_limit = QSpinBox()
        self._spin_corr_limit.setRange(100, 10000)
        self._spin_corr_limit.setSingleStep(100)
        self._spin_corr_limit.setToolTip(
            "Above this feature count, use memory-safe greedy correlation filtering "
            "instead of a full correlation matrix."
        )
        self._spin_corr_limit.setValue(int(self.max_correlation_features))
        self._spin_corr_limit.valueChanged.connect(self._on_changed)
        row_limit.addWidget(self._spin_corr_limit, 1)
        flt_vl.addLayout(row_limit)

        row_cap = QHBoxLayout()
        row_cap.addWidget(QLabel("Corr feature cap"))
        self._spin_corr_cap = QSpinBox()
        self._spin_corr_cap.setRange(0, 20000)
        self._spin_corr_cap.setSingleStep(250)
        self._spin_corr_cap.setToolTip(
            "Hard pre-correlation cap. If more features remain after missing/variance "
            "filters, only the strongest features are passed to correlation filtering. "
            "Use 0 to disable."
        )
        self._spin_corr_cap.setValue(int(self.max_features_before_correlation))
        self._spin_corr_cap.valueChanged.connect(self._on_changed)
        row_cap.addWidget(self._spin_corr_cap, 1)
        flt_vl.addLayout(row_cap)

        row_outcap = QHBoxLayout()
        row_outcap.addWidget(QLabel("Final feature cap"))
        self._spin_final_cap = QSpinBox()
        self._spin_final_cap.setRange(0, 50000)
        self._spin_final_cap.setSingleStep(250)
        self._spin_final_cap.setToolTip(
            "Maximum number of descriptor columns sent downstream after all filters. "
            "Use 0 to disable. Keep 500–2000 for stable QSAR workflows."
        )
        self._spin_final_cap.setValue(int(self.max_output_features))
        self._spin_final_cap.valueChanged.connect(self._on_changed)
        row_outcap.addWidget(self._spin_final_cap, 1)
        flt_vl.addLayout(row_outcap)

        row_tgt = QHBoxLayout()
        row_tgt.addWidget(QLabel("Target col"))
        self._cmb_target = QComboBox()
        self._cmb_target.setToolTip("Optional — guides representative selection in correlated clusters")
        self._cmb_target.currentIndexChanged.connect(self._on_changed)
        row_tgt.addWidget(self._cmb_target, 1)
        flt_vl.addLayout(row_tgt)

        ca.layout().addWidget(flt)

        run_box = QGroupBox("Run")
        run_vl = QVBoxLayout(run_box)
        self._chk_auto = QCheckBox("Auto-run")
        self._chk_auto.setChecked(bool(self.auto_run))
        self._chk_auto.toggled.connect(lambda s: setattr(self, "auto_run", bool(s)))
        run_vl.addWidget(self._chk_auto)
        self._btn_run = QPushButton("Apply filters")
        self._btn_run.clicked.connect(self.commit)
        run_vl.addWidget(self._btn_run)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        run_vl.addWidget(self._progress)
        ca.layout().addWidget(run_box)
        ca.layout().addStretch(1)

    # ── Main area ─────────────────────────────────────────────────────────

    def _build_main_area(self) -> None:
        self._tabs = QTabWidget()

        # Summary
        self._txt_summary = QTextBrowser()
        self._tabs.addTab(self._txt_summary, "Summary")

        # Variance distribution
        var_w = QWidget()
        var_vl = QVBoxLayout(var_w)
        var_vl.setContentsMargins(0, 0, 0, 0)
        self._pw_var = pg.PlotWidget(background="w")
        self._pw_var.setLabel("left", "Count")
        self._pw_var.setLabel("bottom", "Variance (log₁₀)")
        self._pw_var.showGrid(x=True, y=True, alpha=0.18)
        for ax in ("left", "bottom"):
            self._pw_var.getAxis(ax).setPen(pg.mkPen("#CBD5E1"))
        var_vl.addWidget(self._pw_var)
        self._tabs.addTab(var_w, "Variance")

        # Missing value distribution
        miss_w = QWidget()
        miss_vl = QVBoxLayout(miss_w)
        miss_vl.setContentsMargins(0, 0, 0, 0)
        self._pw_miss = pg.PlotWidget(background="w")
        self._pw_miss.setLabel("left", "Count")
        self._pw_miss.setLabel("bottom", "Missing fraction")
        self._pw_miss.showGrid(x=True, y=True, alpha=0.18)
        for ax in ("left", "bottom"):
            self._pw_miss.getAxis(ax).setPen(pg.mkPen("#CBD5E1"))
        miss_vl.addWidget(self._pw_miss)
        self._tabs.addTab(miss_w, "Missing Values")

        # Correlation clusters table
        self._tbl_corr = QTableWidget()
        self._tbl_corr.setColumnCount(4)
        self._tbl_corr.setHorizontalHeaderLabels(["Kept", "Removed features", "Max |r|", "Cluster size"])
        self._tbl_corr.horizontalHeader().setStretchLastSection(True)
        self._tbl_corr.setEditTriggers(QTableWidget.NoEditTriggers)
        self._tbl_corr.setAlternatingRowColors(True)
        self._tabs.addTab(self._tbl_corr, "Corr Clusters")

        # Full report table
        self._tbl_report = QTableWidget()
        self._tbl_report.setColumnCount(7)
        self._tbl_report.setHorizontalHeaderLabels([
            "Feature", "Family", "Missing %", "Variance", "Quality", "Imputed", "Status"
        ])
        self._tbl_report.horizontalHeader().setStretchLastSection(False)
        self._tbl_report.setEditTriggers(QTableWidget.NoEditTriggers)
        self._tbl_report.setAlternatingRowColors(True)
        self._tbl_report.setSortingEnabled(True)
        self._tabs.addTab(self._tbl_report, "Full Report")

        self.mainArea.layout().addWidget(self._tabs)

    # ── Helpers ───────────────────────────────────────────────────────────

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

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._data = data
        if data is None:
            self._set_status("No data.", ok=False)
            self._cmb_target.blockSignals(True)
            self._cmb_target.clear()
            self._cmb_target.blockSignals(False)
            self.Outputs.filtered_data.send(None)
            self.Outputs.modeling_data.send(None)
            self.Outputs.report.send(None)
            return

        all_vars = (list(data.domain.attributes)
                    + list(data.domain.class_vars)
                    + list(data.domain.metas))
        all_names = [v.name for v in all_vars]

        # Populate target combo
        self._cmb_target.blockSignals(True)
        self._cmb_target.clear()
        self._cmb_target.addItem("— none —")
        for n in all_names:
            self._cmb_target.addItem(n)

        # Auto-select: 1) saved setting, 2) Orange class variable, 3) name heuristic
        target = self.target_column
        if not target and data.domain.class_vars:
            target = data.domain.class_vars[0].name
        if not target:
            _HINTS = {"pactivity", "activity", "target", "y", "label"}
            for v in all_vars:
                if v.name.lower() in _HINTS:
                    target = v.name
                    break
        idx = self._cmb_target.findText(target) if target else -1
        self._cmb_target.setCurrentIndex(max(idx, 0))
        self.target_column = target if idx > 0 else ""
        self._cmb_target.blockSignals(False)

        n_feat = len(data.domain.attributes)
        n_meta = len(data.domain.metas)
        self._set_status(f"{len(data)} rows × {n_feat} descriptors + {n_meta} meta", ok=True)
        if self.auto_run:
            self.commit()

    # ── Settings ──────────────────────────────────────────────────────────

    def _on_changed(self) -> None:
        self.max_missing_fraction = float(self._spin_miss.value())
        self.min_variance = float(self._spin_var.value())
        self.max_correlation = float(self._spin_corr.value())
        self.correlation_method = int(self._cmb_corr.currentIndex())
        self.max_correlation_features = int(self._spin_corr_limit.value())
        self.max_features_before_correlation = int(self._spin_corr_cap.value())
        self.max_output_features = int(self._spin_final_cap.value())
        t = self._cmb_target.currentText()
        self.target_column = "" if t == "— none —" else t
        if self.auto_run and self._data is not None:
            self.commit()

    # ── Run ───────────────────────────────────────────────────────────────

    def commit(self) -> None:
        if self._data is None:
            return
        t = self._cmb_target.currentText()
        cfg = DescriptorFilterConfig(
            max_missing_fraction=float(self._spin_miss.value()),
            min_variance=float(self._spin_var.value()),
            max_correlation=float(self._spin_corr.value()),
            correlation_method=self._CORR_METHODS[int(self._cmb_corr.currentIndex())],
            target_column="" if t == "— none —" else t,
            max_correlation_features=int(self._spin_corr_limit.value()),
            max_features_before_correlation=int(self._spin_corr_cap.value()),
            max_output_features=int(self._spin_final_cap.value()),
        )
        data_snap = self._data
        self._set_busy(True)
        self._set_status("Filtering…", ok=True)

        def _worker():
            df = _table_to_df(data_snap)
            return run_descriptor_filter(df, cfg)

        fut = self._executor.submit(_worker)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut) -> None:
        try:
            result = fut.result()
            methodinvoke(self, "_finish", (object,))(result)
        except Exception as e:
            methodinvoke(self, "_fail", (str,))(str(e))

    @Slot(object)
    def _finish(self, result: object) -> None:
        filtered_df, res = result
        self._result = res
        self._set_busy(False)

        output_warnings: list[str] = []

        try:
            filtered_table = _df_to_table(filtered_df, class_col=self.target_column)
            self.Outputs.filtered_data.send(filtered_table)
        except Exception as exc:
            self.Outputs.filtered_data.send(None)
            output_warnings.append(f"Filtered Data output failed: {exc}")

        try:
            modeling_table = _df_to_table(filtered_df, class_col=self.target_column, modeling_clean=True)
            self.Outputs.modeling_data.send(modeling_table)
        except Exception as exc:
            self.Outputs.modeling_data.send(None)
            output_warnings.append(f"Modeling Data output failed: {exc}")

        try:
            report_tbl = _df_to_table(res.report_df)
            self.Outputs.report.send(report_tbl)
        except Exception as exc:
            self.Outputs.report.send(None)
            output_warnings.append(f"Filter Report output failed: {exc}")

        self._populate_summary(res)
        if output_warnings:
            current = self._txt_summary.toPlainText()
            self._txt_summary.setPlainText(
                current + "\n\nOutput warnings:\n" + "\n".join(f"- {w}" for w in output_warnings)
            )
        self._populate_variance_plot(res)
        self._populate_missing_plot(res)
        self._populate_corr_table(res)
        self._populate_report_table(res)

        pct = int(100 * res.n_output / res.n_input) if res.n_input else 0
        status_prefix = "Completed with output warning" if output_warnings else f"{res.n_output}/{res.n_input} features kept"
        self._set_status(
            f"{status_prefix} ({pct}%) — "
            f"empty={len(res.removed_empty)} miss={len(res.removed_missing)} "
            f"var={len(res.removed_low_variance)} "
            f"precap={len(getattr(res, 'removed_pre_correlation_cap', []))} "
            f"corr={len(res.removed_correlated)} "
            f"finalcap={len(getattr(res, 'removed_final_cap', []))}",
            ok=not output_warnings,
        )

    @Slot(str)
    def _fail(self, msg: str) -> None:
        self._set_busy(False)
        self._set_status("Error", ok=False)
        self._txt_summary.setPlainText(f"Failed:\n\n{msg}")

    # ── Populate ──────────────────────────────────────────────────────────

    def _populate_summary(self, r: DescriptorFilterResult) -> None:
        lines = [
            "Descriptor Pre-selector",
            "══════════════════════════════════════",
            f"  Input features  : {r.n_input}",
            f"  Output features : {r.n_output}",
            f"  Reduction       : {r.n_input - r.n_output} removed "
            f"({int(100*(r.n_input-r.n_output)/r.n_input) if r.n_input else 0}%)",
            "",
            "Step 1 — Empty descriptor filter",
            "──────────────────────────────────────",
            f"  Removed : {len(r.removed_empty)}",
            f"  Remaining: {r.n_after_empty}",
        ]
        if r.removed_empty:
            lines.append("  " + ", ".join(r.removed_empty[:10])
                         + (f"  … +{len(r.removed_empty)-10} more" if len(r.removed_empty) > 10 else ""))
        lines += [
            "",
            "Step 2 — Missing value filter",
            "──────────────────────────────────────",
            f"  Removed : {len(r.removed_missing)}",
            f"  Remaining: {r.n_after_missing}",
        ]
        if r.removed_missing:
            lines.append("  " + ", ".join(r.removed_missing[:10])
                         + (f"  … +{len(r.removed_missing)-10} more" if len(r.removed_missing) > 10 else ""))
        lines += [
            "",
            "Step 3 — Low-variance filter",
            "──────────────────────────────────────",
            f"  Removed : {len(r.removed_low_variance)}",
            f"  Remaining: {r.n_after_variance}",
        ]
        if r.removed_low_variance:
            lines.append("  " + ", ".join(r.removed_low_variance[:10])
                         + (f"  … +{len(r.removed_low_variance)-10} more" if len(r.removed_low_variance) > 10 else ""))
        removed_cap = getattr(r, "removed_pre_correlation_cap", [])
        lines += [
            "",
            "Step 4a — Pre-correlation feature cap",
            "──────────────────────────────────────",
            f"  Removed : {len(removed_cap)}",
            f"  Remaining: {getattr(r, 'n_after_pre_correlation_cap', r.n_after_variance)}",
        ]
        if removed_cap:
            lines.append("  " + ", ".join(removed_cap[:10])
                         + (f"  … +{len(removed_cap)-10} more" if len(removed_cap) > 10 else ""))
        lines += [
            "",
            "Step 4b — High-correlation filter",
            "──────────────────────────────────────",
            f"  Removed : {len(r.removed_correlated)}",
            f"  Remaining: {getattr(r, 'n_after_correlation', r.n_output)}",
            f"  Clusters: {len(r.corr_clusters)}",
        ]
        if r.corr_clusters:
            top = sorted(r.corr_clusters, key=lambda x: x["size"], reverse=True)[:5]
            for cl in top:
                lines.append(f"  kept={cl['kept']}  removed={len(cl['removed'])}  max|r|={cl['max_r']:.3f}")
        removed_final = getattr(r, "removed_final_cap", [])
        lines += [
            "",
            "Step 5 — Final modeling feature cap",
            "──────────────────────────────────────",
            f"  Removed : {len(removed_final)}",
            f"  Final output features: {r.n_output}",
        ]
        if removed_final:
            lines.append("  " + ", ".join(removed_final[:10])
                         + (f"  … +{len(removed_final)-10} more" if len(removed_final) > 10 else ""))
        if getattr(r, "quality_summary", None):
            qs = r.quality_summary or {}
            lines += ["", "Descriptor quality diagnostics", "──────────────────────────────────────"]
            lines.append(f"  Mean quality all : {qs.get('mean_quality_all', '—')}")
            lines.append(f"  Mean quality kept: {qs.get('mean_quality_kept', '—')}")
            top_fam = qs.get("top_removed_families") or []
            if top_fam:
                lines.append("  Top removed families:")
                for name, count in top_fam[:8]:
                    lines.append(f"    - {name}: {count}")
        if getattr(r, "notes", None):
            lines += ["", "Notes", "──────────────────────────────────────"]
            lines.extend(f"  {note}" for note in r.notes)
        self._txt_summary.setPlainText("\n".join(lines))

    def _populate_variance_plot(self, r: DescriptorFilterResult) -> None:
        self._pw_var.clear()
        vs = r.variance_series.dropna()
        vs = vs[vs > 0]
        if vs.empty:
            return
        log_v = np.log10(vs.values.astype(float))
        counts, edges = np.histogram(log_v, bins=40)
        # Kept vs removed colouring
        removed_set = set(r.removed_low_variance)
        log_kept    = np.log10(vs[~vs.index.isin(removed_set)].values.astype(float) + 1e-20)
        log_removed = np.log10(vs[vs.index.isin(removed_set)].values.astype(float) + 1e-20)
        ck, ek = np.histogram(log_kept,    bins=edges)
        cr, er = np.histogram(log_removed, bins=edges)
        w = float(edges[1] - edges[0])
        self._pw_var.addItem(pg.BarGraphItem(x=ek[:-1], height=ck, width=w, brush=pg.mkBrush(37,99,235,160), pen=pg.mkPen(None)))
        self._pw_var.addItem(pg.BarGraphItem(x=er[:-1], height=cr, width=w, brush=pg.mkBrush(239,68,68,160),  pen=pg.mkPen(None)))
        # Threshold line
        thresh = np.log10(max(float(self.min_variance), 1e-20))
        self._pw_var.addItem(pg.InfiniteLine(pos=thresh, angle=90, pen=pg.mkPen("#f59e0b", width=2, style=Qt.DashLine)))

    def _populate_missing_plot(self, r: DescriptorFilterResult) -> None:
        self._pw_miss.clear()
        ms = r.missing_series
        if ms.empty:
            return
        vals = ms.values.astype(float)
        removed_set = set(r.removed_missing)
        kept_vals    = ms[~ms.index.isin(removed_set)].values.astype(float)
        removed_vals = ms[ms.index.isin(removed_set)].values.astype(float)
        edges = np.linspace(0.0, 1.0, 21)
        ck, _ = np.histogram(kept_vals,    bins=edges)
        cr, _ = np.histogram(removed_vals, bins=edges)
        w = float(edges[1] - edges[0])
        self._pw_miss.addItem(pg.BarGraphItem(x=edges[:-1], height=ck, width=w, brush=pg.mkBrush(37,99,235,160), pen=pg.mkPen(None)))
        self._pw_miss.addItem(pg.BarGraphItem(x=edges[:-1], height=cr, width=w, brush=pg.mkBrush(239,68,68,160),  pen=pg.mkPen(None)))
        self._pw_miss.addItem(pg.InfiniteLine(pos=float(self.max_missing_fraction), angle=90, pen=pg.mkPen("#f59e0b", width=2, style=Qt.DashLine)))

    def _populate_corr_table(self, r: DescriptorFilterResult) -> None:
        clusters = sorted(r.corr_clusters, key=lambda x: x["size"], reverse=True)
        self._tbl_corr.setRowCount(len(clusters))
        for i, cl in enumerate(clusters):
            self._tbl_corr.setItem(i, 0, QTableWidgetItem(cl["kept"]))
            self._tbl_corr.setItem(i, 1, QTableWidgetItem(", ".join(cl["removed"][:8])
                                                            + (f" …+{len(cl['removed'])-8}" if len(cl["removed"]) > 8 else "")))
            self._tbl_corr.setItem(i, 2, QTableWidgetItem(f"{cl['max_r']:.4f}"))
            self._tbl_corr.setItem(i, 3, QTableWidgetItem(str(cl["size"])))
        self._tbl_corr.resizeColumnsToContents()
        self._tabs.setTabText(3, f"Corr Clusters ({len(clusters)})")

    def _populate_report_table(self, r: DescriptorFilterResult) -> None:
        df_full = r.report_df
        limit = max(100, int(getattr(self, "report_display_limit", 1000)))
        df = df_full.head(limit).copy()
        self._tbl_report.setSortingEnabled(False)
        self._tbl_report.setRowCount(len(df))
        _STATUS_COLOR = {
            "kept":          "#d1fae5",
            "high_missing":  "#fee2e2",
            "low_variance":  "#fef3c7",
            "pre_corr_cap":  "#e0f2fe",
            "correlated":    "#ede9fe",
            "final_cap":     "#fce7f3",
            "empty":         "#fecaca",
        }
        for i, row in df.iterrows():
            ri = int(i)
            status = str(row["status"])
            color = _STATUS_COLOR.get(status, "#ffffff")
            for ci, val in enumerate([
                str(row.get("feature", "")),
                str(row.get("family", "")),
                f"{float(row.get('missing_fraction', 0.0))*100:.1f}%",
                f"{row['variance']:.5g}" if pd.notna(row.get("variance")) else "—",
                f"{float(row.get('quality_score', 0.0)):.1f}",
                str(int(row.get("imputed_output_values", 0) or 0)),
                status,
            ]):
                item = QTableWidgetItem(val)
                item.setBackground(pg.mkColor(color))
                self._tbl_report.setItem(ri, ci, item)
        self._tbl_report.resizeColumnsToContents()
        self._tbl_report.setSortingEnabled(True)
        suffix = f"{len(df)}/{len(df_full)}" if len(df_full) > len(df) else str(len(df_full))
        self._tabs.setTabText(4, f"Report ({suffix})")

    def onDeleteWidget(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWDescriptorFilter).run()
