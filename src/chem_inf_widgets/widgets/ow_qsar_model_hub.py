from __future__ import annotations

import traceback
from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from matplotlib.figure import Figure

from AnyQt.QtCore import Qt, pyqtSlot as Slot
from AnyQt.QtGui import QPixmap
from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

pg.setConfigOptions(antialias=True)

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services import qsar_regression_service as qsar_service
from chem_inf_widgets.chemcore.services.qsar_model_hub_service import (
    QSARModelHubConfig,
    QSARModelHubResult,
    available_model_keys,
    hpo_available,
    train_qsar_model_hub,
)
from chem_inf_widgets.chemcore.services.qsar_diagnostics_contract import (
    SELECTION_TOOL_OPTIONS,
    display_model_name as _display_model_name,
    residual_reference_levels as _residual_reference_levels,
)
from chem_inf_widgets.chemcore.services.qsar_target_contract import (
    DEFAULT_QSAR_TARGET_COLUMN,
    TARGET_COLUMN_CANDIDATES,
    preferred_target_name_from_table,
)
from chem_inf_widgets.chemcore.services.qsar_prediction_packager_service import (
    build_qsar_prediction_bundle,
)
from chem_inf_widgets.widgets import qsar_diagnostics_ui


# ── Orange ↔ pandas helpers ──────────────────────────────────────────────────

def _orange_table_to_dataframe(data: Table) -> pd.DataFrame:
    cols: dict = {}
    n = len(data)

    attr_vars = list(data.domain.attributes)
    if attr_vars:
        X = np.array(data.X, dtype=float)
        for i, var in enumerate(attr_vars):
            cols[var.name] = X[:, i] if X.ndim == 2 else X

    class_vars = list(data.domain.class_vars)
    if class_vars:
        Y = np.array(data.Y, dtype=float).reshape(n, -1)
        for i, var in enumerate(class_vars):
            cols[var.name] = Y[:, i]

    meta_vars = list(data.domain.metas)
    if meta_vars and data.metas is not None and data.metas.size:
        M = data.metas
        for i, var in enumerate(meta_vars):
            col = M[:, i]
            if isinstance(var, StringVariable):
                cols[var.name] = [str(v) if v is not None else "" for v in col]
            else:
                try:
                    cols[var.name] = col.astype(float)
                except Exception:
                    cols[var.name] = [str(v) for v in col]

    return pd.DataFrame(cols, index=range(n))


def _dataframe_to_orange(df: pd.DataFrame) -> Table:
    if df is None or df.empty:
        return Table.from_numpy(Domain([]), X=np.empty((0, 0)))
    attrs, metas, X_cols, M_cols = [], [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            attrs.append(ContinuousVariable(str(col)))
            X_cols.append(col)
        else:
            metas.append(StringVariable(str(col)))
            M_cols.append(col)
    X = df[X_cols].to_numpy(dtype=float) if X_cols else np.empty((len(df), 0), dtype=float)
    M = df[M_cols].fillna("").astype(str).to_numpy(dtype=object) if M_cols else np.empty((len(df), 0), dtype=object)
    return Table.from_numpy(Domain(attrs, metas=metas), X=X, metas=M)


def _guess_col(cols: list[str], candidates: list[str]) -> str:
    low = {c.strip().lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return ""


def _preferred_target_name(data: Table, candidates: list[str]) -> str:
    return preferred_target_name_from_table(data, candidates=candidates)


def _input_table_diagnostic(data: Table) -> str | None:
    names = {str(v.name).strip().lower() for v in list(data.domain.attributes) + list(data.domain.class_vars) + list(data.domain.metas)}
    if {"metric", "value"}.issubset(names):
        return (
            "Input looks like a summary/report table, not a modeling dataset. "
            "If this comes from QSAR Dataset Builder, connect the 'QSAR Ready Data' output."
        )
    if {"stage", "status", "note"}.issubset(names):
        return "Input looks like a workflow/report table, not a descriptor dataset for QSAR/QSPR modeling."
    return None


_MODEL_KEYS = available_model_keys()
_HPO_AVAILABLE = hpo_available()
_DEFAULT_MODEL_INDEX = _MODEL_KEYS.index("random_forest") if "random_forest" in _MODEL_KEYS else 0


# ── Widget ────────────────────────────────────────────────────────────────────

class OWQSARModelHub(OWWidget):
    name = "QSAR/QSPR Model Hub"
    description = "Train regression models for activity (QSAR) or physicochemical properties (QSPR) from descriptor tables."
    icon = "icons/modeling/ow_qsar_model_hub.png"
    priority = 144
    keywords = ["QSAR", "QSPR", "model", "regression", "property", "boiling point", "machine learning"]
    want_main_area = True
    resizing_enabled = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        model = Output("Model", object, auto_summary=False)
        predictions = Output("Predictions", Table)
        metrics = Output("Metrics", Table)
        model_summary = Output("Model Summary", Table)
        selected_compounds = Output("Selected Compounds", Table)

    target_column: str = Setting(DEFAULT_QSAR_TARGET_COLUMN)
    id_column: str = Setting("compound_id")
    target_unit: str = Setting("")
    model_index: int = Setting(_DEFAULT_MODEL_INDEX)
    test_size: float = Setting(0.25)
    cv_folds: int = Setting(5)
    random_state: int = Setting(42)
    auto_run: bool = Setting(True)
    use_hpo: bool = Setting(False)
    hpo_trials: int = Setting(50)
    hpo_sampler: str = Setting("tpe")
    hpo_pruner: str = Setting("median")
    use_feature_selection: bool = Setting(False)
    fs_max_features: int = Setting(50)
    ensemble_top_k: int = Setting(0)
    selection_tool: int = Setting(0)

    _SELECTION_TOOL_OPTIONS = list(SELECTION_TOOL_OPTIONS)

    # Known property column names for auto-detection
    _TARGET_CANDIDATES = list(TARGET_COLUMN_CANDIDATES)

    _MODEL_KEYS: list[str] = _MODEL_KEYS
    _HPO_AVAILABLE: bool = _HPO_AVAILABLE

    def __init__(self) -> None:
        super().__init__()
        self._data: Optional[Table] = None
        self._executor = ThreadExecutor(self)
        self._last_result: Optional[QSARModelHubResult] = None
        self._predictions_table: Optional[Table] = None
        self._diagnostic_selectors = {}
        self._diagnostic_context = None
        self._diagnostic_canvas = None
        self._diagnostic_fig = None
        self._last_model_name = "Random Forest"

        if not self._HPO_AVAILABLE:
            self.use_hpo = False
        self.model_index = min(max(int(self.model_index), 0), max(len(self._MODEL_KEYS) - 1, 0))
        if self._MODEL_KEYS and self._MODEL_KEYS[self.model_index] == "auto":
            self.model_index = _DEFAULT_MODEL_INDEX

        self._build_control_area()
        self._build_main_area()
        self._normalize_algorithm_selection()
        self._set_status("Awaiting data…", ok=True)

    # ── Control area ─────────────────────────────────────────────────────────

    def _build_control_area(self) -> None:
        ca = self.controlArea

        # Header
        hdr = QWidget()
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(6, 6, 6, 6)
        left = QVBoxLayout()
        left.addWidget(QLabel("QSAR/QSPR Model Hub", objectName="HdrTitle"))
        left.addWidget(QLabel("Activity (QSAR) or property (QSPR) regression from descriptors", objectName="HdrSub"))
        hl.addLayout(left, 1)
        self._lbl_status = QLabel("Ready", objectName="StatusChip")
        self._lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hl.addWidget(self._lbl_status)
        ca.layout().addWidget(hdr)

        # Columns
        col_box = QGroupBox("Columns")
        col_vl = QVBoxLayout(col_box)

        row_t = QHBoxLayout()
        row_t.addWidget(QLabel("Dependent variable (Y)"))
        self._ed_target = QLineEdit(self.target_column)
        self._ed_target.textChanged.connect(self._on_settings_changed)
        row_t.addWidget(self._ed_target, 1)
        col_vl.addLayout(row_t)

        row_i = QHBoxLayout()
        row_i.addWidget(QLabel("ID"))
        self._ed_id = QLineEdit(self.id_column)
        self._ed_id.textChanged.connect(self._on_settings_changed)
        row_i.addWidget(self._ed_id, 1)
        col_vl.addLayout(row_i)

        row_u = QHBoxLayout()
        row_u.addWidget(QLabel("Unit"))
        self._ed_unit = QLineEdit(self.target_unit)
        self._ed_unit.setPlaceholderText("e.g. °C, K, mg/L, log units")
        self._ed_unit.textChanged.connect(self._on_settings_changed)
        row_u.addWidget(self._ed_unit, 1)
        col_vl.addLayout(row_u)
        ca.layout().addWidget(col_box)

        # Model
        mdl_box = QGroupBox("Model")
        mdl_vl = QVBoxLayout(mdl_box)

        row_alg = QHBoxLayout()
        row_alg.addWidget(QLabel("Algorithm"))
        self._cmb_algo = QComboBox()
        self._cmb_algo.addItems(self._MODEL_KEYS)
        self._cmb_algo.setCurrentIndex(min(max(int(self.model_index), 0), max(len(self._MODEL_KEYS) - 1, 0)))
        self._cmb_algo.currentIndexChanged.connect(self._on_settings_changed)
        row_alg.addWidget(self._cmb_algo, 1)
        mdl_vl.addLayout(row_alg)

        row_ts = QHBoxLayout()
        row_ts.addWidget(QLabel("Test fraction"))
        self._spin_test = QDoubleSpinBox()
        self._spin_test.setRange(0.05, 0.80)
        self._spin_test.setSingleStep(0.05)
        self._spin_test.setDecimals(2)
        self._spin_test.setValue(float(self.test_size))
        self._spin_test.valueChanged.connect(self._on_settings_changed)
        row_ts.addWidget(self._spin_test, 1)
        mdl_vl.addLayout(row_ts)

        row_cv = QHBoxLayout()
        row_cv.addWidget(QLabel("CV folds"))
        self._spin_cv = QSpinBox()
        self._spin_cv.setRange(2, 20)
        self._spin_cv.setValue(int(self.cv_folds))
        self._spin_cv.valueChanged.connect(self._on_settings_changed)
        row_cv.addWidget(self._spin_cv, 1)
        mdl_vl.addLayout(row_cv)

        row_rs = QHBoxLayout()
        row_rs.addWidget(QLabel("Random seed"))
        self._spin_rs = QSpinBox()
        self._spin_rs.setRange(0, 999999)
        self._spin_rs.setValue(int(self.random_state))
        self._spin_rs.valueChanged.connect(self._on_settings_changed)
        row_rs.addWidget(self._spin_rs, 1)
        mdl_vl.addLayout(row_rs)

        self._chk_auto = QCheckBox("Auto-run")
        self._chk_auto.setChecked(bool(self.auto_run))
        self._chk_auto.toggled.connect(self._on_auto_toggled)
        mdl_vl.addWidget(self._chk_auto)

        row_sel = QHBoxLayout()
        row_sel.addWidget(QLabel("Selection"))
        self._cmb_selection_tool = QComboBox()
        self._cmb_selection_tool.addItems(self._SELECTION_TOOL_OPTIONS)
        self._cmb_selection_tool.setCurrentIndex(int(self.selection_tool))
        self._cmb_selection_tool.currentIndexChanged.connect(self._on_selection_tool_changed)
        row_sel.addWidget(self._cmb_selection_tool, 1)
        mdl_vl.addLayout(row_sel)

        self._btn_train = QPushButton("Train model")
        self._btn_train.clicked.connect(self.commit)
        mdl_vl.addWidget(self._btn_train)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        mdl_vl.addWidget(self._progress)

        ca.layout().addWidget(mdl_box)

        # HPO group
        hpo_box = QGroupBox("Hyperparameter Optimization (Optuna)")
        hpo_vl = QVBoxLayout(hpo_box)

        self._chk_hpo = QCheckBox("Enable HPO (overrides algorithm choice)")
        self._chk_hpo.setChecked(bool(self.use_hpo))
        self._chk_hpo.toggled.connect(self._on_hpo_toggled)
        hpo_vl.addWidget(self._chk_hpo)

        row_trials = QHBoxLayout()
        row_trials.addWidget(QLabel("Trials"))
        self._spin_trials = QSpinBox()
        self._spin_trials.setRange(5, 1000)
        self._spin_trials.setValue(int(self.hpo_trials))
        self._spin_trials.valueChanged.connect(self._on_settings_changed)
        row_trials.addWidget(self._spin_trials, 1)
        hpo_vl.addLayout(row_trials)

        row_samp = QHBoxLayout()
        row_samp.addWidget(QLabel("Sampler"))
        self._cmb_sampler = QComboBox()
        self._cmb_sampler.addItems(["tpe", "cmaes", "gp", "qmc", "random"])
        self._cmb_sampler.setCurrentText(self.hpo_sampler)
        self._cmb_sampler.currentTextChanged.connect(self._on_settings_changed)
        row_samp.addWidget(self._cmb_sampler, 1)
        hpo_vl.addLayout(row_samp)

        row_prun = QHBoxLayout()
        row_prun.addWidget(QLabel("Pruner"))
        self._cmb_pruner = QComboBox()
        self._cmb_pruner.addItems(["median", "none"])
        self._cmb_pruner.setCurrentText(self.hpo_pruner)
        self._cmb_pruner.currentTextChanged.connect(self._on_settings_changed)
        row_prun.addWidget(self._cmb_pruner, 1)
        hpo_vl.addLayout(row_prun)

        self._chk_fs = QCheckBox("Feature selection in HPO")
        self._chk_fs.setChecked(bool(self.use_feature_selection))
        self._chk_fs.toggled.connect(self._on_settings_changed)
        hpo_vl.addWidget(self._chk_fs)

        row_fs = QHBoxLayout()
        row_fs.addWidget(QLabel("Max features"))
        self._spin_fs = QSpinBox()
        self._spin_fs.setRange(2, 2000)
        self._spin_fs.setValue(int(self.fs_max_features))
        self._spin_fs.valueChanged.connect(self._on_settings_changed)
        row_fs.addWidget(self._spin_fs, 1)
        hpo_vl.addLayout(row_fs)

        row_ens = QHBoxLayout()
        row_ens.addWidget(QLabel("Ensemble top-K"))
        self._spin_ens = QSpinBox()
        self._spin_ens.setRange(0, 20)
        self._spin_ens.setValue(int(self.ensemble_top_k))
        self._spin_ens.setSpecialValueText("off")
        self._spin_ens.valueChanged.connect(self._on_settings_changed)
        row_ens.addWidget(self._spin_ens, 1)
        hpo_vl.addLayout(row_ens)

        algos_note = ", ".join(["RF", "ET", "GBM", "Ridge", "EN", "SVR"]
                               + (["LGB"] if "lightgbm" in self._MODEL_KEYS else [])
                               + (["XGB"] if "xgboost" in self._MODEL_KEYS else []))
        note = QLabel(f"Searches: {algos_note}")
        note.setWordWrap(True)
        note.setStyleSheet("color:#64748B; font-size:10px;")
        hpo_vl.addWidget(note)

        if not self._HPO_AVAILABLE:
            self._chk_hpo.setChecked(False)
            self._chk_hpo.setEnabled(False)
            self._spin_trials.setEnabled(False)
            self._cmb_sampler.setEnabled(False)
            self._cmb_pruner.setEnabled(False)
            self._chk_fs.setEnabled(False)
            self._spin_fs.setEnabled(False)
            self._spin_ens.setEnabled(False)
            missing_note = QLabel(
                "Optuna is not installed in this environment. HPO and the 'auto' model are disabled."
            )
            missing_note.setWordWrap(True)
            missing_note.setStyleSheet("color:#a40000; font-size:10px;")
            hpo_vl.addWidget(missing_note)

        ca.layout().addWidget(hpo_box)
        ca.layout().addStretch(1)

    # ── Main area ─────────────────────────────────────────────────────────────

    def _build_main_area(self) -> None:
        self._tabs = QTabWidget()

        # Summary tab
        self._txt_summary = QTextEdit()
        self._txt_summary.setReadOnly(True)
        self._txt_summary.setStyleSheet("font-family: monospace; font-size: 12px;")
        self._tabs.addTab(self._txt_summary, "Summary")

        # Diagnostics tab
        self._diagnostics_tab = QWidget()
        diag_vl = QVBoxLayout(self._diagnostics_tab)
        diag_vl.setContentsMargins(0, 0, 0, 0)
        diag_vl.setSpacing(6)
        self._diagnostics_hint = QLabel(
            "Predicted vs observed and residual diagnostics. Use rectangle or lasso selection to inspect compounds."
        )
        self._diagnostics_hint.setWordWrap(True)
        self._diagnostics_hint.setStyleSheet("color:#64748B; font-size:11px;")
        diag_vl.addWidget(self._diagnostics_hint)
        self._diagnostics_canvas_container = QWidget()
        self._diagnostics_canvas_layout = QVBoxLayout(self._diagnostics_canvas_container)
        self._diagnostics_canvas_layout.setContentsMargins(0, 0, 0, 0)
        diag_vl.addWidget(self._diagnostics_canvas_container, 4)

        self._selection_gallery_label = QLabel("Selected compounds")
        self._selection_gallery_label.setStyleSheet("font-weight:600; font-size:13px; color:#0F172A;")
        diag_vl.addWidget(self._selection_gallery_label)

        self._selection_gallery_scroll = QScrollArea()
        self._selection_gallery_scroll.setWidgetResizable(True)
        self._selection_gallery_scroll.setMinimumHeight(170)
        self._selection_gallery_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._selection_gallery_scroll.setStyleSheet(
            "QScrollArea { background: #ffffff; border: 1px solid #d7dee8; border-radius: 6px; }"
        )
        self._selection_gallery_container = QWidget()
        self._selection_gallery_layout = QHBoxLayout(self._selection_gallery_container)
        self._selection_gallery_layout.setContentsMargins(6, 6, 6, 6)
        self._selection_gallery_layout.setSpacing(10)
        self._selection_gallery_scroll.setWidget(self._selection_gallery_container)
        diag_vl.addWidget(self._selection_gallery_scroll, 2)
        self._tabs.addTab(self._diagnostics_tab, "Diagnostics")

        # Selected compounds tab
        self._selected_tab = QWidget()
        selected_vl = QVBoxLayout(self._selected_tab)
        selected_vl.setContentsMargins(0, 0, 0, 0)
        self._tbl_selected = QTableWidget()
        self._tbl_selected.setEditTriggers(QTableWidget.NoEditTriggers)
        self._tbl_selected.setAlternatingRowColors(True)
        self._tbl_selected.horizontalHeader().setStretchLastSection(True)
        selected_vl.addWidget(self._tbl_selected)
        self._tabs.addTab(self._selected_tab, "Selected")

        # Metrics tab
        self._tbl_metrics = QTableWidget()
        self._tbl_metrics.setColumnCount(3)
        self._tbl_metrics.setHorizontalHeaderLabels(["Split", "Metric", "Value"])
        self._tbl_metrics.horizontalHeader().setStretchLastSection(True)
        self._tbl_metrics.setEditTriggers(QTableWidget.NoEditTriggers)
        self._tbl_metrics.setAlternatingRowColors(True)
        self._tabs.addTab(self._tbl_metrics, "Metrics")

        # Features tab
        self._tbl_features = QTableWidget()
        self._tbl_features.setColumnCount(1)
        self._tbl_features.setHorizontalHeaderLabels(["Feature name"])
        self._tbl_features.horizontalHeader().setStretchLastSection(True)
        self._tbl_features.setEditTriggers(QTableWidget.NoEditTriggers)
        self._tbl_features.setAlternatingRowColors(True)
        self._tabs.addTab(self._tbl_features, "Features")

        # HPO History tab — sub-tabs for Opt History, Param Importance, Top Trials
        hpo_subtabs = QTabWidget()

        # Optimisation history plot
        hist_w = QWidget()
        hist_vl = QVBoxLayout(hist_w)
        hist_vl.setContentsMargins(0, 0, 0, 0)
        self._pw_hpo = pg.PlotWidget(background="w")
        self._pw_hpo.setLabel("left", "CV R²")
        self._pw_hpo.setLabel("bottom", "Trial")
        self._pw_hpo.showGrid(x=True, y=True, alpha=0.18)
        for ax in ("left", "bottom"):
            self._pw_hpo.getAxis(ax).setPen(pg.mkPen("#CBD5E1"))
        hist_vl.addWidget(self._pw_hpo)
        hpo_subtabs.addTab(hist_w, "Opt History")

        # Parameter importance plot
        pimp_w = QWidget()
        pimp_vl = QVBoxLayout(pimp_w)
        pimp_vl.setContentsMargins(0, 0, 0, 0)
        self._pw_pimp = pg.PlotWidget(background="w")
        self._pw_pimp.setLabel("bottom", "Relative importance (fANOVA)")
        self._pw_pimp.showGrid(x=True, y=False, alpha=0.18)
        self._pw_pimp.getAxis("bottom").setPen(pg.mkPen("#CBD5E1"))
        self._pw_pimp.getAxis("left").setPen(pg.mkPen("#CBD5E1"))
        pimp_vl.addWidget(self._pw_pimp)
        hpo_subtabs.addTab(pimp_w, "Param Importance")

        # Top trials table
        self._tbl_hpo = QTableWidget()
        self._tbl_hpo.setEditTriggers(QTableWidget.NoEditTriggers)
        self._tbl_hpo.setAlternatingRowColors(True)
        hpo_subtabs.addTab(self._tbl_hpo, "Top Trials")

        self._hpo_subtabs = hpo_subtabs
        self._tabs.addTab(hpo_subtabs, "HPO History")

        self.mainArea.layout().addWidget(self._tabs)
        self._show_selection_gallery_placeholder("Train a model, then select points on the diagnostics plot.")
        self._update_selected_table(None)

    # ── Status helpers ────────────────────────────────────────────────────────

    def _set_status(self, msg: str, ok: bool = True) -> None:
        self._lbl_status.setText(msg)
        if ok:
            self._lbl_status.setStyleSheet(
                "padding:4px 8px;border:1px solid #e1e1e1;border-radius:10px;background:#fafafa;"
            )
        else:
            self._lbl_status.setStyleSheet(
                "padding:4px 8px;border:1px solid #f2c2c2;border-radius:10px;"
                "background:#fff5f5;color:#a40000;"
            )

    def _set_busy(self, busy: bool) -> None:
        self._btn_train.setEnabled(not busy)
        self._cmb_algo.setEnabled(not busy)
        self._ed_target.setEnabled(not busy)
        self._ed_id.setEnabled(not busy)
        self._progress.setVisible(busy)

    def _current_model_key(self) -> str:
        if not self._MODEL_KEYS:
            return "random_forest"
        idx = min(max(int(self._cmb_algo.currentIndex()), 0), len(self._MODEL_KEYS) - 1)
        return str(self._MODEL_KEYS[idx])

    def _set_algorithm_without_signals(self, model_key: str) -> None:
        if model_key not in self._MODEL_KEYS:
            return
        idx = self._MODEL_KEYS.index(model_key)
        self._cmb_algo.blockSignals(True)
        self._cmb_algo.setCurrentIndex(idx)
        self._cmb_algo.blockSignals(False)
        self.model_index = idx

    def _normalize_algorithm_selection(self) -> None:
        current_key = self._current_model_key()
        if current_key == "auto" and not bool(self.use_hpo):
            fallback = "random_forest" if "random_forest" in self._MODEL_KEYS else self._MODEL_KEYS[0]
            self._set_algorithm_without_signals(fallback)
        elif current_key != "auto" and bool(self.use_hpo) and "auto" in self._MODEL_KEYS:
            self._set_algorithm_without_signals("auto")

    # ── Inputs ────────────────────────────────────────────────────────────────

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._data = data
        if data is None:
            self._set_status("No data.", ok=False)
            self._clear_outputs()
            return
        # Auto-detect column names
        all_names = (
            [v.name for v in data.domain.attributes]
            + [v.name for v in data.domain.class_vars]
            + [v.name for v in data.domain.metas]
        )
        t = _preferred_target_name(data, self._TARGET_CANDIDATES)
        i = _guess_col(all_names, ["compound_id", "chembl_id", "mol_id", "id", "name", "smiles_id"])
        self._ed_target.blockSignals(True)
        self._ed_id.blockSignals(True)
        try:
            if t and self._ed_target.text() != t:
                self._ed_target.setText(t)
            if i and self._ed_id.text() != i:
                self._ed_id.setText(i)
        finally:
            self._ed_target.blockSignals(False)
            self._ed_id.blockSignals(False)
        self.target_column = self._ed_target.text().strip()
        self.id_column = self._ed_id.text().strip()
        self._set_status(f"{len(data)} rows", ok=True)
        if self.auto_run:
            self.commit()

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_settings_changed(self) -> None:
        self.target_column = self._ed_target.text().strip()
        self.id_column = self._ed_id.text().strip()
        self.target_unit = self._ed_unit.text().strip()
        self.model_index = self._cmb_algo.currentIndex()
        self.test_size = float(self._spin_test.value())
        self.cv_folds = int(self._spin_cv.value())
        self.random_state = int(self._spin_rs.value())
        self.hpo_trials = int(self._spin_trials.value())
        self.hpo_sampler = self._cmb_sampler.currentText()
        self.hpo_pruner = self._cmb_pruner.currentText()
        self.use_feature_selection = self._chk_fs.isChecked()
        self.fs_max_features = int(self._spin_fs.value())
        self.ensemble_top_k = int(self._spin_ens.value())
        if self._current_model_key() == "auto" and self._HPO_AVAILABLE and not self._chk_hpo.isChecked():
            self._chk_hpo.blockSignals(True)
            self._chk_hpo.setChecked(True)
            self._chk_hpo.blockSignals(False)
            self.use_hpo = True
        self._normalize_algorithm_selection()
        if self.auto_run and self._data is not None:
            self.commit()

    def _on_auto_toggled(self, state: bool) -> None:
        self.auto_run = bool(state)

    def _on_selection_tool_changed(self, index: int) -> None:
        self.selection_tool = int(index)
        self._refresh_selector_modes()

    def _on_hpo_toggled(self, state: bool) -> None:
        if state and not self._HPO_AVAILABLE:
            self._chk_hpo.setChecked(False)
            self.use_hpo = False
            return
        self.use_hpo = bool(state)
        # When HPO active, set algo combo to "auto" visually
        if state and "auto" in self._MODEL_KEYS:
            self._cmb_algo.setCurrentIndex(self._MODEL_KEYS.index("auto"))
        elif (not state) and self._current_model_key() == "auto":
            fallback = "random_forest" if "random_forest" in self._MODEL_KEYS else self._MODEL_KEYS[0]
            self._set_algorithm_without_signals(fallback)
        if self.auto_run and self._data is not None:
            self.commit()

    # ── Training (async) ──────────────────────────────────────────────────────

    def commit(self) -> None:
        if self._data is None:
            self._set_status("No data.", ok=False)
            self._clear_outputs()
            return

        table_hint = _input_table_diagnostic(self._data)
        if table_hint:
            self._set_status("Error", ok=False)
            self._clear_outputs()
            self._txt_summary.setPlainText(table_hint)
            return

        target_column = self._ed_target.text().strip()
        if not target_column:
            self._set_status("Error", ok=False)
            self._clear_outputs()
            self._txt_summary.setPlainText(
                "No dependent variable (Y) was detected. "
                "Choose a numeric target column manually, or if the input comes from QSAR Dataset Builder "
                "connect the 'QSAR Ready Data' output."
            )
            return

        use_hpo = self._chk_hpo.isChecked()
        model_key = self._current_model_key()
        if model_key == "auto" and not use_hpo:
            fallback = "random_forest" if "random_forest" in self._MODEL_KEYS else self._MODEL_KEYS[0]
            self._set_algorithm_without_signals(fallback)
            model_key = fallback
        cfg = QSARModelHubConfig(
            target_column=target_column or DEFAULT_QSAR_TARGET_COLUMN,
            id_column=(self._ed_id.text().strip() or "compound_id"),
            model_key=model_key,
            test_size=float(self._spin_test.value()),
            cv_folds=int(self._spin_cv.value()),
            random_state=int(self._spin_rs.value()),
            use_hpo=use_hpo,
            hpo_trials=int(self._spin_trials.value()),
            hpo_sampler=self._cmb_sampler.currentText(),
            hpo_pruner=self._cmb_pruner.currentText(),
            use_feature_selection=self._chk_fs.isChecked(),
            fs_max_features=int(self._spin_fs.value()),
            ensemble_top_k=int(self._spin_ens.value()),
        )
        if use_hpo:
            ens_str = f", top-{cfg.ensemble_top_k} ensemble" if cfg.ensemble_top_k > 1 else ""
            fs_str = f", FS≤{cfg.fs_max_features}" if cfg.use_feature_selection else ""
            self._set_status(
                f"HPO {cfg.hpo_sampler}/{cfg.hpo_pruner} · {cfg.hpo_trials} trials{fs_str}{ens_str}…",
                ok=True,
            )

        self._set_busy(True)
        self._set_status("Training…", ok=True)
        self.Outputs.selected_compounds.send(None)
        self._update_selected_table(None)
        self._reset_diagnostics_view("Training in progress — diagnostics will appear when the model is ready.")

        data_snapshot = self._data

        def _run():
            df = _orange_table_to_dataframe(data_snapshot)
            return train_qsar_model_hub(df, cfg)

        fut = self._executor.submit(_run)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut) -> None:
        try:
            result = fut.result()
            methodinvoke(self, "_finish", (object,))(result)
        except Exception:
            methodinvoke(self, "_fail", (str,))(traceback.format_exc())

    @Slot(object)
    def _finish(self, result: object) -> None:
        r: QSARModelHubResult = result
        self._last_result = r
        self._last_model_name = _display_model_name(r.model_key)
        self._set_busy(False)

        self._predictions_table = _dataframe_to_orange(r.predictions)
        self.Outputs.model.send(
            build_qsar_prediction_bundle(
                r.pipeline,
                feature_names=list(r.feature_names),
                target_label=r.target_column,
                model_name=self._last_model_name,
                source_widget=self.name,
                training_rows=r.n_rows_used,
                selected_feature_names=list(r.feature_names),
            )
        )
        self.Outputs.predictions.send(self._predictions_table)
        self.Outputs.metrics.send(_dataframe_to_orange(r.metrics_table))
        self.Outputs.model_summary.send(_dataframe_to_orange(pd.DataFrame([r.summary])))
        self.Outputs.selected_compounds.send(None)

        self._populate_summary(r)
        self._populate_plots(r)
        self._populate_metrics(r)
        self._populate_features(r)
        self._populate_hpo(r)

        r2 = r.test_metrics.get("test_r2", float("nan"))
        r2_str = f"{r2:.3f}" if r2 == r2 else "n/a"
        self._set_status(
            f"{_display_model_name(r.model_key)} | R²={r2_str} | {r.n_rows_used} rows | {r.n_features_used} features",
            ok=True,
        )

    @Slot(str)
    def _fail(self, msg: str) -> None:
        self._set_busy(False)
        self._set_status("Error", ok=False)
        self._txt_summary.setPlainText(f"Training failed:\n\n{msg}")
        self._clear_outputs()

    # ── Output helpers ────────────────────────────────────────────────────────

    def _clear_outputs(self) -> None:
        self._last_result = None
        self._predictions_table = None
        self.Outputs.model.send(None)
        self.Outputs.predictions.send(None)
        self.Outputs.metrics.send(None)
        self.Outputs.model_summary.send(None)
        self.Outputs.selected_compounds.send(None)
        self._txt_summary.clear()
        self._pw_hpo.clear()
        self._pw_pimp.clear()
        self._tbl_hpo.setRowCount(0)
        self._tbl_metrics.setRowCount(0)
        self._tbl_features.setRowCount(0)
        self._update_selected_table(None)
        self._reset_diagnostics_view("Awaiting model results to render diagnostics.")

    def _populate_plots(self, r: QSARModelHubResult) -> None:
        self._populate_diagnostics(r)

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _reset_diagnostics_view(self, message: str) -> None:
        self._diagnostic_selectors = {}
        self._diagnostic_context = None
        self._diagnostic_canvas = None
        self._diagnostic_fig = None
        self._clear_layout(self._diagnostics_canvas_layout)
        self._diagnostics_hint.setText(message)
        self._show_selection_gallery_placeholder(message)

    def _populate_diagnostics(self, r: QSARModelHubResult) -> None:
        preds_df = r.predictions.copy()
        if preds_df.empty or self._predictions_table is None:
            self._reset_diagnostics_view("No predictions available for diagnostics.")
            return

        preds = preds_df["predicted"].to_numpy(dtype=float)
        actuals = preds_df["observed"].to_numpy(dtype=float)
        residuals = preds_df["residual"].to_numpy(dtype=float)
        levels = _residual_reference_levels(residuals)

        self._reset_diagnostics_view(
            "Drag a rectangle or lasso on either plot to select compounds and inspect them below."
        )
        self._update_selected_table(None)

        fig = Figure(figsize=(11.8, 5.6))
        ax_left = fig.add_subplot(121)
        ax_right = fig.add_subplot(122)

        palette = {
            "train": ("#2563EB", "Train"),
            "test": ("#EA580C", "Test"),
        }

        for split, (color, label) in palette.items():
            sub = preds_df[preds_df["split"].astype(str).str.lower() == split]
            if sub.empty:
                continue
            ax_left.scatter(
                sub["predicted"],
                sub["observed"],
                s=46,
                alpha=0.78,
                c=color,
                edgecolors="#0F172A",
                linewidths=0.35,
                label=label,
            )
            ax_right.scatter(
                sub["predicted"],
                sub["residual"],
                s=46,
                alpha=0.78,
                c=color,
                edgecolors="#0F172A",
                linewidths=0.35,
                label=label,
            )

        finite_left = np.isfinite(preds) & np.isfinite(actuals)
        if np.any(finite_left):
            left_min = float(min(np.min(preds[finite_left]), np.min(actuals[finite_left])))
            left_max = float(max(np.max(preds[finite_left]), np.max(actuals[finite_left])))
        else:
            left_min, left_max = 0.0, 1.0
        left_span = left_max - left_min
        left_pad = max(left_span * 0.07, 0.1 if left_span == 0 else 0.0)
        diag_min = left_min - left_pad
        diag_max = left_max + left_pad

        ax_left.plot(
            [diag_min, diag_max],
            [diag_min, diag_max],
            color="#475569",
            linestyle="-",
            linewidth=1.5,
            label="Ideal",
        )
        for offset, color, linestyle, linewidth, label in (
            (levels["plus_1std"], "#2563EB", "--", 1.25, "±1σ"),
            (levels["minus_1std"], "#2563EB", "--", 1.25, None),
            (levels["plus_2std"], "#EA580C", ":", 1.45, "±2σ"),
            (levels["minus_2std"], "#EA580C", ":", 1.45, None),
        ):
            ax_left.plot(
                [diag_min, diag_max],
                [diag_min + offset, diag_max + offset],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.95,
                label=label,
            )

        ax_left.set_xlim(diag_min, diag_max)
        ax_left.set_ylim(diag_min, diag_max)
        ax_left.set_aspect("equal", adjustable="box")
        ax_left.set_title("Predicted vs Observed")
        unit = self.target_unit.strip()
        axis_label = f"{r.target_column} ({unit})" if unit else r.target_column
        ax_left.set_xlabel(f"Predicted {axis_label}")
        ax_left.set_ylabel(f"Observed {axis_label}")
        ax_left.grid(alpha=0.28, linewidth=0.8)
        ax_left.legend(loc="best", frameon=True, framealpha=0.92, fontsize=8)

        ax_right.axhline(0.0, color="#475569", linestyle="-", linewidth=1.4, label="Zero residual")
        for value, color, linestyle, linewidth, label in (
            (levels["mean"], "#94A3B8", "--", 1.0, "Mean residual"),
            (levels["plus_1std"], "#2563EB", "--", 1.2, "±1σ"),
            (levels["minus_1std"], "#2563EB", "--", 1.2, None),
            (levels["plus_2std"], "#EA580C", ":", 1.4, "±2σ"),
            (levels["minus_2std"], "#EA580C", ":", 1.4, None),
        ):
            ax_right.axhline(value, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95, label=label)

        finite_right_x = np.isfinite(preds)
        if np.any(finite_right_x):
            x_min = float(np.min(preds[finite_right_x]))
            x_max = float(np.max(preds[finite_right_x]))
        else:
            x_min, x_max = 0.0, 1.0
        x_span = x_max - x_min
        x_pad = max(x_span * 0.07, 0.1 if x_span == 0 else 0.0)
        ax_right.set_xlim(x_min - x_pad, x_max + x_pad)

        finite_right_y = np.isfinite(residuals)
        if np.any(finite_right_y):
            y_min = float(np.min(residuals[finite_right_y]))
            y_max = float(np.max(residuals[finite_right_y]))
        else:
            y_min, y_max = -1.0, 1.0
        y_span = y_max - y_min
        y_pad = max(y_span * 0.10, max(abs(levels["plus_2std"]), abs(levels["minus_2std"]), 0.1) * 0.15)
        ax_right.set_ylim(min(y_min, levels["minus_2std"]) - y_pad, max(y_max, levels["plus_2std"]) + y_pad)
        ax_right.set_title("Residuals vs Predicted")
        ax_right.set_xlabel(f"Predicted {axis_label}")
        ax_right.set_ylabel(f"Residual ({unit})" if unit else "Residual")
        ax_right.grid(alpha=0.28, linewidth=0.8)
        ax_right.legend(loc="best", frameon=True, framealpha=0.92, fontsize=8)

        info_lines = [
            f"Model: {_display_model_name(r.model_key)}",
            f"Test R²: {r.test_metrics.get('test_r2', float('nan')):.3f}" if np.isfinite(r.test_metrics.get("test_r2", float("nan"))) else "Test R²: n/a",
            f"Test RMSE: {r.test_metrics.get('test_rmse', float('nan')):.3f}" if np.isfinite(r.test_metrics.get("test_rmse", float("nan"))) else "Test RMSE: n/a",
            f"Residual σ: {levels['std']:.3f}",
        ]
        ax_left.text(
            0.02,
            0.98,
            "\n".join(info_lines),
            transform=ax_left.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFFFF", "edgecolor": "#CBD5E1", "alpha": 0.95},
        )

        sel_left = ax_left.scatter([], [], s=110, facecolors="none", edgecolors="#F59E0B", linewidths=2.2, zorder=6)
        sel_right = ax_right.scatter([], [], s=110, facecolors="none", edgecolors="#F59E0B", linewidths=2.2, zorder=6)

        fig.tight_layout(pad=1.15, w_pad=1.25)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._attach_diagnostic_canvas(canvas, fig)
        self._install_point_selection(
            canvas=canvas,
            fig=fig,
            ax_left=ax_left,
            ax_right=ax_right,
            preds=preds,
            actuals=actuals,
            residuals=residuals,
            table=self._predictions_table,
            overlay_left=sel_left,
            overlay_right=sel_right,
        )

    def _attach_diagnostic_canvas(self, canvas, fig) -> None:
        self._clear_layout(self._diagnostics_canvas_layout)
        self._diagnostics_canvas_layout.addWidget(canvas)
        self._diagnostic_canvas = canvas
        self._diagnostic_fig = fig

    def _install_point_selection(
        self,
        *,
        canvas,
        fig,
        ax_left,
        ax_right,
        preds,
        actuals,
        residuals,
        table,
        overlay_left,
        overlay_right,
    ) -> None:
        self._diagnostic_context = qsar_diagnostics_ui.build_diagnostic_selection_context(
            canvas=canvas,
            figure=fig,
            preds=preds,
            y=actuals,
            residuals=residuals,
            table=table,
            overlay_left=overlay_left,
            overlay_right=overlay_right,
        )
        self._diagnostic_selectors = {
            "combined": qsar_diagnostics_ui.create_diagnostic_selectors(
                ax_left=ax_left,
                ax_right=ax_right,
                on_rect_left=lambda eclick, erelease: self._apply_plot_selection(eclick, erelease, left_plot=True),
                on_rect_right=lambda eclick, erelease: self._apply_plot_selection(eclick, erelease, left_plot=False),
                on_lasso_left=lambda verts: self._apply_lasso_selection(verts, left_plot=True),
                on_lasso_right=lambda verts: self._apply_lasso_selection(verts, left_plot=False),
            )
        }
        self._refresh_selector_modes()

    def _refresh_selector_modes(self) -> None:
        use_lasso = int(self.selection_tool) == 1
        for selectors in self._diagnostic_selectors.values():
            qsar_diagnostics_ui.set_selector_mode(selectors, use_lasso=use_lasso)

    def _apply_plot_selection(self, eclick, erelease, *, left_plot: bool) -> None:
        context = self._diagnostic_context
        if context is None or context.table is None:
            return
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        x0, x1 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y0, y1 = sorted([float(eclick.ydata), float(erelease.ydata)])
        preds, ys = qsar_diagnostics_ui.selection_plot_values(context, left_plot=left_plot)
        selected_idx = qsar_service.rectangle_selection_indices(preds, ys, x0, y0, x1, y1)
        self._publish_selection(selected_idx)

    def _apply_lasso_selection(self, vertices, *, left_plot: bool) -> None:
        context = self._diagnostic_context
        if context is None or context.table is None or not vertices:
            return
        preds, ys = qsar_diagnostics_ui.selection_plot_values(context, left_plot=left_plot)
        selected_idx = qsar_service.lasso_selection_indices(preds, ys, vertices)
        self._publish_selection(selected_idx)

    def _publish_selection(self, selected_idx) -> None:
        context = self._diagnostic_context
        if context is None or context.table is None:
            return
        self._update_selection_overlays(selected_idx)
        payload = qsar_service.build_selection_publish_payload(
            model_name=self._last_model_name,
            dataset_type="model hub",
            table=context.table,
            selected_idx=selected_idx,
        )
        self.Outputs.selected_compounds.send(payload.selected_table)
        self._update_selection_gallery(payload.gallery)
        self._update_selected_table(payload.selected_table)
        self._diagnostics_hint.setText(payload.status_text)

    def _update_selection_overlays(self, selected_idx) -> None:
        context = self._diagnostic_context
        if context is None:
            return
        qsar_diagnostics_ui.update_selection_overlays(context, selected_idx)

    def _clear_selection_gallery(self) -> None:
        while self._selection_gallery_layout.count():
            item = self._selection_gallery_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _show_selection_gallery_placeholder(self, text: str) -> None:
        self._clear_selection_gallery()
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color:#64748B; padding:6px;")
        self._selection_gallery_layout.addWidget(label)
        self._selection_gallery_layout.addStretch(1)

    def _update_selection_gallery(self, payload) -> None:
        if payload.placeholder_text:
            self._show_selection_gallery_placeholder(payload.placeholder_text)
            return
        self._clear_selection_gallery()
        for preview in payload.previews:
            pixmap = QPixmap()
            pixmap.loadFromData(preview.png_bytes, "PNG")

            card = QFrame()
            card.setFrameShape(QFrame.StyledPanel)
            card.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #d7dee8; border-radius: 6px; }")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(4, 4, 4, 4)
            card_layout.setSpacing(4)

            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setFixedSize(180, 135)
            img_label.setScaledContents(True)
            txt_label = QLabel(preview.title)
            txt_label.setWordWrap(True)
            txt_label.setStyleSheet("font-size: 11px; border: none; color: #334155;")

            card_layout.addWidget(img_label)
            card_layout.addWidget(txt_label)
            self._selection_gallery_layout.addWidget(card)

        if payload.more_count > 0:
            more_label = QLabel(f"+ {payload.more_count} more")
            more_label.setStyleSheet("color:#64748B; padding:12px;")
            self._selection_gallery_layout.addWidget(more_label)
        self._selection_gallery_layout.addStretch(1)

    def _update_selected_table(self, selected_table: Optional[Table]) -> None:
        selected_tab_index = self._tabs.indexOf(self._selected_tab)
        if selected_table is None or len(selected_table) == 0:
            self._tbl_selected.clearContents()
            self._tbl_selected.setRowCount(0)
            self._tbl_selected.setColumnCount(1)
            self._tbl_selected.setHorizontalHeaderLabels(["No selected compounds"])
            if selected_tab_index >= 0:
                self._tabs.setTabText(selected_tab_index, "Selected")
            return

        df = _orange_table_to_dataframe(selected_table)
        if df is None or df.empty:
            self._tbl_selected.clearContents()
            self._tbl_selected.setRowCount(0)
            self._tbl_selected.setColumnCount(1)
            self._tbl_selected.setHorizontalHeaderLabels(["No selected compounds"])
            if selected_tab_index >= 0:
                self._tabs.setTabText(selected_tab_index, "Selected")
            return

        cols = [str(col) for col in df.columns]
        self._tbl_selected.clearContents()
        self._tbl_selected.setColumnCount(len(cols))
        self._tbl_selected.setHorizontalHeaderLabels(cols)
        self._tbl_selected.setRowCount(len(df))
        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, col in enumerate(cols):
                value = row[col]
                if isinstance(value, (float, np.floating)):
                    text = f"{float(value):.4f}" if np.isfinite(float(value)) else ""
                else:
                    text = "" if pd.isna(value) else str(value)
                self._tbl_selected.setItem(row_idx, col_idx, QTableWidgetItem(text))
        self._tbl_selected.resizeColumnsToContents()
        if selected_tab_index >= 0:
            self._tabs.setTabText(selected_tab_index, f"Selected ({len(df)})")

    def _populate_summary(self, r: QSARModelHubResult) -> None:
        def _fmt(d: dict) -> str:
            return "  " + "\n  ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                      for k, v in d.items())

        note = ""
        if r.summary.get("auto_rdkit_descriptors"):
            note = (
                "\n⚠ No pre-computed descriptors — 12 compact RDKit descriptors\n"
                "  were auto-calculated from SMILES. Connect Mol Descriptors 2\n"
                "  first for better models.\n"
            )

        unit = self.target_unit.strip()
        target_str = f"{r.target_column} [{unit}]" if unit else r.target_column
        lines = [
            "QSAR/QSPR Model Hub",
            "══════════════════════════════════════",
            f"  Algorithm   : {_display_model_name(r.model_key)}",
            f"  Target      : {target_str}",
            f"  Rows used   : {r.n_rows_used} / {r.n_rows_input}",
            f"  Features    : {r.n_features_used} / {r.n_features_input}",
            f"  Scaled      : {r.summary.get('scale_features', False)}",
            "",
            "Train metrics",
            "──────────────────────────────────────",
            _fmt(r.train_metrics),
            "",
            "Test metrics",
            "──────────────────────────────────────",
            _fmt(r.test_metrics),
            "",
            "Cross-validation",
            "──────────────────────────────────────",
            _fmt(r.cv_metrics),
        ]
        if note:
            lines += ["", note]
        self._txt_summary.setPlainText("\n".join(lines))

    def _populate_metrics(self, r: QSARModelHubResult) -> None:
        rows = []
        for split, d in [("train", r.train_metrics), ("test", r.test_metrics), ("cv", r.cv_metrics)]:
            for k, v in d.items():
                rows.append((split, k, f"{v:.4f}" if isinstance(v, float) and v == v else str(v)))
        self._tbl_metrics.setRowCount(len(rows))
        for row_idx, (split, metric, value) in enumerate(rows):
            self._tbl_metrics.setItem(row_idx, 0, QTableWidgetItem(split))
            self._tbl_metrics.setItem(row_idx, 1, QTableWidgetItem(metric))
            self._tbl_metrics.setItem(row_idx, 2, QTableWidgetItem(value))
        self._tbl_metrics.resizeColumnsToContents()

    def _populate_features(self, r: QSARModelHubResult) -> None:
        names = r.feature_names
        self._tbl_features.setRowCount(len(names))
        for i, name in enumerate(names):
            self._tbl_features.setItem(i, 0, QTableWidgetItem(name))
        self._tabs.setTabText(4, f"Features ({len(names)})")

    def _populate_hpo(self, r: "QSARModelHubResult") -> None:
        self._pw_hpo.clear()
        self._pw_pimp.clear()
        self._tbl_hpo.setRowCount(0)
        # Parameter importance chart
        if r.param_importances:
            items = sorted(r.param_importances.items(), key=lambda x: x[1])
            names = [k for k, _ in items]
            vals  = np.array([v for _, v in items], dtype=float)
            y_pos = np.arange(len(vals), dtype=float)
            self._pw_pimp.addItem(pg.BarGraphItem(
                x0=np.zeros(len(vals)), x1=vals,
                y=y_pos, height=0.65,
                brush=pg.mkBrush(124, 58, 237, 180),
                pen=pg.mkPen(None),
            ))
            self._pw_pimp.getAxis("left").setTicks([list(zip(y_pos, names))])
        hdf = r.hpo_history
        if hdf is None or hdf.empty:
            self._tabs.setTabText(5, "HPO History")
            return
        trials = hdf["trial"].to_numpy(dtype=float)
        values = hdf["value_cv_r2"].to_numpy(dtype=float)
        # Running best line
        best_so_far = np.maximum.accumulate(np.where(np.isfinite(values), values, -np.inf))
        self._pw_hpo.addItem(pg.ScatterPlotItem(
            x=trials, y=values, size=6,
            pen=pg.mkPen(None), brush=pg.mkBrush(37, 99, 235, 160),
        ))
        self._pw_hpo.addItem(pg.PlotDataItem(
            trials, best_so_far,
            pen=pg.mkPen((234, 88, 12, 220), width=2),
        ))
        # Table of top 10 trials
        param_cols = [c for c in hdf.columns if c not in ("trial", "value_cv_r2")]
        all_cols = ["trial", "cv_r2"] + param_cols
        self._tbl_hpo.setColumnCount(len(all_cols))
        self._tbl_hpo.setHorizontalHeaderLabels(all_cols)
        top = hdf.nlargest(min(10, len(hdf)), "value_cv_r2")
        self._tbl_hpo.setRowCount(len(top))
        for ri, (_, row) in enumerate(top.iterrows()):
            self._tbl_hpo.setItem(ri, 0, QTableWidgetItem(str(int(row["trial"]))))
            self._tbl_hpo.setItem(ri, 1, QTableWidgetItem(f"{row['value_cv_r2']:.4f}"))
            for ci, pc in enumerate(param_cols, start=2):
                val = row.get(pc, "")
                self._tbl_hpo.setItem(ri, ci, QTableWidgetItem(str(val) if pd.notna(val) else ""))
        self._tbl_hpo.resizeColumnsToContents()
        self._tabs.setTabText(5, f"HPO History ({len(hdf)} trials)")

    def onDeleteWidget(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWQSARModelHub).run()
