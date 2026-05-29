from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg

from AnyQt.QtCore import Qt, QTimer, pyqtSlot as Slot
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
    QSizePolicy,
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
    _CONTROL_PANEL_TARGET_WIDTH = 380
    _CONTROL_PANEL_MIN_WIDTH = 340
    _MAIN_PANEL_MIN_WIDTH = 760

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
        self._apply_left_right_ratio()
        QTimer.singleShot(0, self._apply_left_right_ratio)
        QTimer.singleShot(250, self._apply_left_right_ratio)
        self._set_status("Awaiting data…", ok=True)

    # ── Layout helpers ────────────────────────────────────────────────────

    def _apply_left_right_ratio(self) -> None:
        """Force a usable 30:70 layout: compact controls, wide report.

        Orange wraps ``controlArea``/``mainArea`` differently across Qt versions.
        The safest behaviour is therefore: keep the control side physically
        narrow, make the report side expandable, and only then set splitter
        sizes by locating the real children.
        """
        try:
            # Keep the visible control panel itself compact.
            self.controlArea.setMinimumWidth(self._CONTROL_PANEL_MIN_WIDTH)
            self.controlArea.setMaximumWidth(self._CONTROL_PANEL_TARGET_WIDTH)
            self.controlArea.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            self.mainArea.setMinimumWidth(self._MAIN_PANEL_MIN_WIDTH)
            self.mainArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass

        def _contains(parent, child) -> bool:
            try:
                w = child
                while w is not None:
                    if w is parent:
                        return True
                    w = w.parentWidget()
            except Exception:
                return False
            return False

        candidate_splitters = []
        for attr in ("splitter", "_splitter"):
            splitter = getattr(self, attr, None)
            if splitter is not None:
                candidate_splitters.append(splitter)
        try:
            candidate_splitters.extend(self.findChildren(QSplitter))
        except Exception:
            pass

        seen = set()
        for splitter in candidate_splitters:
            if splitter is None or id(splitter) in seen or not hasattr(splitter, "setSizes"):
                continue
            seen.add(id(splitter))
            try:
                if splitter.orientation() != Qt.Horizontal:
                    continue
                count = splitter.count()
                control_i = main_i = -1
                for i in range(count):
                    child = splitter.widget(i)
                    if child is self.controlArea or _contains(child, self.controlArea):
                        control_i = i
                    if child is self.mainArea or _contains(child, self.mainArea):
                        main_i = i
                if control_i < 0 or main_i < 0 or control_i == main_i:
                    continue

                control_widget = splitter.widget(control_i)
                main_widget = splitter.widget(main_i)
                for widget in (control_widget, self.controlArea):
                    if widget is None:
                        continue
                    try:
                        widget.setMinimumWidth(self._CONTROL_PANEL_MIN_WIDTH)
                        widget.setMaximumWidth(self._CONTROL_PANEL_TARGET_WIDTH)
                        widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
                    except Exception:
                        pass
                if main_widget is not None:
                    try:
                        main_widget.setMinimumWidth(self._MAIN_PANEL_MIN_WIDTH)
                        main_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                    except Exception:
                        pass

                sizes = [120] * count
                sizes[control_i] = self._CONTROL_PANEL_TARGET_WIDTH
                sizes[main_i] = max(
                    self._MAIN_PANEL_MIN_WIDTH,
                    int(self.width() * 0.72),
                )
                splitter.setSizes(sizes)
                splitter.setStretchFactor(control_i, 0)
                splitter.setStretchFactor(main_i, 1)
                splitter.setCollapsible(control_i, False)
                splitter.setCollapsible(main_i, False)
                break
            except Exception:
                pass

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._apply_left_right_ratio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._apply_left_right_ratio)

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

        # Summary / report
        self._txt_summary = QTextBrowser()
        self._txt_summary.setOpenExternalLinks(True)
        self._txt_summary.setStyleSheet("background:#ffffff; border:1px solid #E2E8F0; border-radius:8px;")
        self._tabs.addTab(self._txt_summary, "Report")

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
            import html as _html
            current = self._txt_summary.toHtml()
            warn_html = "<h2>Output warnings</h2><ul>" + "".join(
                f"<li>{_html.escape(w)}</li>" for w in output_warnings
            ) + "</ul>"
            self._txt_summary.setHtml(current.replace("</body>", warn_html + "</body>"))
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
        """Render a rich dashboard-style report in the Overview tab."""
        def pct(part: int | float, total: int | float) -> str:
            return f"{100.0 * float(part) / float(total):.1f}%" if total else "0.0%"

        def esc(x) -> str:
            import html
            return html.escape(str(x))

        def card(title: str, value: str, sub: str = "", accent: str = "#2563EB") -> str:
            return (
                f"<td class='kpi' style='border-top:4px solid {accent};'>"
                f"<div class='kpi-value' style='color:{accent};'>{esc(value)}</div>"
                f"<div class='kpi-title'>{esc(title)}</div>"
                f"<div class='kpi-sub'>{esc(sub)}</div>"
                f"</td>"
            )

        n_input = int(r.n_input)
        n_output = int(r.n_output)
        removed_total = max(0, n_input - n_output)
        removed_pct = pct(removed_total, n_input)
        removed_cap = getattr(r, "removed_pre_correlation_cap", []) or []
        removed_final = getattr(r, "removed_final_cap", []) or []
        n_after_corr = getattr(r, "n_after_correlation", n_output)
        n_after_precap = getattr(r, "n_after_pre_correlation_cap", r.n_after_variance)
        mean_missing = float(r.missing_series.mean()) if getattr(r, "missing_series", None) is not None and not r.missing_series.empty else 0.0
        median_var = float(r.variance_series.median()) if getattr(r, "variance_series", None) is not None and not r.variance_series.dropna().empty else 0.0
        median_abs_r = 0.0
        try:
            vals = [abs(float(cl.get("max_r", 0.0))) for cl in r.corr_clusters]
            median_abs_r = float(np.median(vals)) if vals else 0.0
        except Exception:
            pass

        cascade = [
            ("Input descriptors", n_input, "#2563EB"),
            (f"Missing filter<br><span class='muted'>(≤ {self.max_missing_fraction:.2f})</span>", -len(r.removed_missing), "#EF4444"),
            (f"Low variance<br><span class='muted'>(≥ {self.min_variance:.5g})</span>", -len(r.removed_low_variance), "#F97316"),
            ("Pre-corr cap", -len(removed_cap), "#0EA5E9"),
            (f"Correlation filter<br><span class='muted'>(|r| ≤ {self.max_correlation:.2f})</span>", -len(r.removed_correlated), "#EF4444"),
            ("Final cap", -len(removed_final), "#DB2777"),
            ("Final descriptors", n_output, "#16A34A"),
        ]
        max_abs = max([abs(v) for _, v, _ in cascade] + [1])
        cascade_rows = ""
        running = n_input
        for label, val, color in cascade:
            if label == "Input descriptors":
                remaining = n_input
                shown = n_input
            elif label == "Final descriptors":
                remaining = n_output
                shown = n_output
            else:
                running += val
                remaining = max(running, 0)
                shown = val
            width = max(6, int(100 * abs(shown) / max_abs))
            sign = "+" if shown > 0 and label not in {"Input descriptors", "Final descriptors"} else ""
            cascade_rows += (
                f"<tr><td>{label}</td><td class='num'>{sign}{shown}</td>"
                f"<td><div class='bar-bg'><div class='bar' style='width:{width}%; background:{color};'></div></div></td>"
                f"<td class='num'>{remaining}</td></tr>"
            )

        top_clusters = sorted(r.corr_clusters, key=lambda x: x["size"], reverse=True)[:10]
        cluster_rows = "".join(
            f"<tr><td>{i+1}</td><td>{int(cl.get('size', 0))}</td><td>{esc(cl.get('kept',''))}</td><td>{float(cl.get('max_r',0.0)):.3f}</td></tr>"
            for i, cl in enumerate(top_clusters)
        ) or "<tr><td colspan='4'>No highly correlated clusters were detected.</td></tr>"

        qs = getattr(r, "quality_summary", None) or {}
        top_fam = qs.get("top_removed_families") or []
        fam_rows = "".join(
            f"<tr><td>{esc(name)}</td><td>{count}</td></tr>" for name, count in top_fam[:8]
        ) or "<tr><td colspan='2'>No family-level removal summary available.</td></tr>"

        qflags = [
            ("High missing (> threshold)", len(r.removed_missing), "#EF4444"),
            ("Low variance", len(r.removed_low_variance), "#F97316"),
            ("Highly correlated", len(r.removed_correlated), "#8B5CF6"),
            ("Pre-correlation cap", len(removed_cap), "#0EA5E9"),
            ("Final cap", len(removed_final), "#DB2777"),
            ("Constant / empty", len(r.removed_empty), "#64748B"),
        ]
        qflag_rows = "".join(
            f"<tr><td><span class='dot' style='background:{color};'></span>{esc(name)}</td><td class='pill'>{count}</td></tr>"
            for name, count, color in qflags
        )

        next_steps = [
            f"{n_output} descriptors remain after pre-selection.",
            "Use <b>Modeling Data</b> as input for <b>QSAR/QSPR Model Hub</b>.",
            "Check multicollinearity/VIF again on the final descriptor set if using linear models.",
            "Validate with cross-validation and preferably an external test set.",
            "Keep the Filter Report table for the final QSAR report.",
        ]
        next_html = "".join(f"<li>{x}</li>" for x in next_steps)

        notes = "".join(f"<li>{esc(n)}</li>" for n in (getattr(r, "notes", None) or [])) or "<li>No additional notes.</li>"

        html = f"""
        <html><head><style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; color:#0F172A; margin:0; padding:18px; background:#FFFFFF; }}
        h1 {{ font-size:26px; margin:0 0 4px 0; font-weight:800; color:#0B1B3A; }}
        h2 {{ font-size:16px; margin:0 0 10px 0; color:#0B1B3A; }}
        .muted, .subtitle {{ color:#64748B; }}
        .subtitle {{ margin-bottom:14px; font-size:13px; }}
        table {{ border-collapse:collapse; width:100%; }}
        td, th {{ border:1px solid #E2E8F0; padding:7px 9px; font-size:12px; vertical-align:middle; }}
        th {{ background:#F8FAFC; color:#334155; text-align:left; }}
        .kpis td {{ width:16.6%; }}
        .kpi {{ background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:12px; }}
        .kpi-value {{ font-size:27px; font-weight:800; line-height:1.0; }}
        .kpi-title {{ font-weight:700; margin-top:6px; }}
        .kpi-sub {{ color:#64748B; font-size:11px; margin-top:3px; }}
        .grid {{ width:100%; border-spacing:10px; border-collapse:separate; }}
        .panel {{ border:1px solid #E2E8F0; border-radius:12px; background:#FFFFFF; padding:12px; }}
        .panel-blue {{ background:#F8FBFF; border-color:#BFDBFE; }}
        .num {{ text-align:right; font-weight:700; }}
        .bar-bg {{ height:12px; background:#F1F5F9; border-radius:8px; overflow:hidden; }}
        .bar {{ height:12px; border-radius:8px; }}
        .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; }}
        .pill {{ text-align:center; font-weight:700; border-radius:999px; color:#0F172A; background:#F8FAFC; }}
        .oklist li {{ margin:7px 0; font-size:12.5px; }}
        .oklist li::marker {{ color:#16A34A; }}
        .footer {{ color:#475569; font-size:12px; line-height:1.45; }}
        </style></head><body>
        <h1>Descriptor Pre-selection Report</h1>
        <div class="subtitle">Comprehensive quality assessment and filtering of molecular descriptors for QSAR/QSPR modeling.</div>

        <table class="kpis"><tr>
          {card('Input descriptors', str(n_input), '100.0%', '#2563EB')}
          {card('Kept descriptors', str(n_output), pct(n_output, n_input), '#16A34A')}
          {card('Removed descriptors', str(removed_total), pct(removed_total, n_input), '#F97316')}
          {card('Correlation clusters', str(len(r.corr_clusters)), f'{len(r.removed_correlated)} removed', '#8B5CF6')}
          {card('Redundancy pairs', str(len(r.removed_correlated)), f'|r| ≥ {self.max_correlation:.2f}', '#0EA5E9')}
          {card('Mean missing rate', f'{mean_missing*100:.1f}%', 'overall', '#D97706')}
        </tr></table>

        <table class="grid"><tr>
          <td class="panel" style="width:47%;"><h2>Filtering cascade</h2><table><tr><th>Step</th><th>Δ</th><th>Scale</th><th>Remaining</th></tr>{cascade_rows}</table></td>
          <td class="panel" style="width:30%;"><h2>Quality flags</h2><table>{qflag_rows}</table></td>
          <td class="panel panel-blue" style="width:23%;"><h2>Data summary</h2><table>
            <tr><td>Rows</td><td class="num">{len(self._data) if self._data is not None else '—'}</td></tr>
            <tr><td>Numeric descriptors</td><td class="num">{n_input}</td></tr>
            <tr><td>Median variance</td><td class="num">{median_var:.3g}</td></tr>
            <tr><td>Median cluster |r|</td><td class="num">{median_abs_r:.3f}</td></tr>
            <tr><td>Target column</td><td class="num">{esc(self.target_column or '—')}</td></tr>
          </table></td>
        </tr></table>

        <table class="grid"><tr>
          <td class="panel" style="width:34%;"><h2>Top 10 correlation clusters</h2><table><tr><th>ID</th><th>Size</th><th>Representative kept</th><th>Max |r|</th></tr>{cluster_rows}</table></td>
          <td class="panel" style="width:28%;"><h2>Descriptor families most affected</h2><table><tr><th>Family</th><th>Removed</th></tr>{fam_rows}</table></td>
          <td class="panel" style="width:38%;"><h2>Next steps for QSAR</h2><ul class="oklist">{next_html}</ul></td>
        </tr></table>

        <table class="grid"><tr>
          <td class="panel panel-blue" style="width:55%;"><h2>Interpretation</h2><div class="footer">Filtering removed <b>{removed_pct}</b> of descriptors. The remaining table is less redundant and should reduce overfitting risk, improve model interpretability, and speed up downstream model selection.</div></td>
          <td class="panel panel-blue" style="width:45%;"><h2>Report quality</h2><div class="footer">All filter stages completed. Notes:<ul>{notes}</ul></div></td>
        </tr></table>
        </body></html>
        """
        self._txt_summary.setHtml(html)

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
