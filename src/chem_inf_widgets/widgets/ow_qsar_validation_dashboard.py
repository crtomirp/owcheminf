from __future__ import annotations

import html
import re
from contextlib import suppress

import numpy as np
import pandas as pd
import pyqtgraph as pg
from AnyQt.QtCore import Qt
from AnyQt.QtCore import pyqtSlot as Slot
from AnyQt.QtGui import QPixmap
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from matplotlib.figure import Figure
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from chem_inf_widgets.chemcore.services import qsar_regression_service as qsar_service
from chem_inf_widgets.chemcore.services.qsar_diagnostics_contract import (
    SELECTION_TOOL_OPTIONS,
)
from chem_inf_widgets.chemcore.services.qsar_diagnostics_contract import (
    residual_reference_levels as _residual_reference_levels,
)
from chem_inf_widgets.chemcore.services.qsar_validation_dashboard_service import (
    QSARValidationConfig,
    validate_qsar_predictions,
)
from chem_inf_widgets.widgets import qsar_diagnostics_ui

pg.setConfigOptions(antialias=True)


_OBSERVED_CANDIDATES = (
    "observed",
    "y_true",
    "actual",
    "experimental",
    "measured",
    "reference",
    "target",
    "pactivity",
    "pic50",
)
_PREDICTED_CANDIDATES = (
    "predicted",
    "y_pred",
    "prediction",
    "pred",
    "fitted",
    "predicted_pactivity",
)
_SPLIT_CANDIDATES = (
    "split",
    "dataset",
    "partition",
    "subset",
    "fold_group",
    "set",
)
_ID_CANDIDATES = (
    "compound_id",
    "chembl_id",
    "molecule_id",
    "mol_id",
    "id",
    "name",
)

_METRIC_PLOT_SPECS = (
    ("r2", "R\u00b2"),
    ("rmse", "RMSE"),
    ("mae", "MAE"),
    ("ccc", "CCC"),
)

_METRIC_TABLE_COLUMNS = (
    ("group", "Split"),
    ("n", "N"),
    ("r2", "R\u00b2"),
    ("rmse", "RMSE"),
    ("mae", "MAE"),
    ("ccc", "CCC"),
    ("pearson_r", "Pearson r"),
    ("bias", "Bias"),
    ("residual_sd", "Residual SD"),
    ("median_abs_error", "Median AE"),
    ("p95_abs_residual", "P95 |Residual|"),
    ("max_abs_residual", "Max |Residual|"),
    ("slope", "Slope"),
    ("intercept", "Intercept"),
)


# ---------------------------------------------------------------------------
# Fast table → DataFrame conversion
# ---------------------------------------------------------------------------

def _table_to_df(data):
    if data is None:
        return None
    cols = {}
    n = len(data)
    for i, v in enumerate(data.domain.attributes):
        X = np.array(data.X, dtype=float)
        cols[v.name] = X[:, i] if X.ndim == 2 else X
    if data.domain.class_vars:
        Y = np.array(data.Y, dtype=float).reshape(n, -1)
        for i, v in enumerate(data.domain.class_vars):
            cols[v.name] = Y[:, i]
    if data.domain.metas and data.metas is not None and data.metas.size:
        M = data.metas
        for i, v in enumerate(data.domain.metas):
            col = M[:, i]
            if isinstance(v, StringVariable):
                cols[v.name] = [str(x) if x is not None else "" for x in col]
            else:
                try:
                    cols[v.name] = col.astype(float)
                except Exception:
                    cols[v.name] = [str(x) for x in col]
    return pd.DataFrame(cols, index=range(n))


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def _detect_column_name(df: pd.DataFrame, candidates: tuple[str, ...], *, numeric: bool | None = None) -> str | None:
    if df is None or df.empty:
        return None

    normalized = {_norm_key(col): str(col) for col in df.columns}
    for candidate in candidates:
        col = normalized.get(_norm_key(candidate))
        if col is None:
            continue
        if numeric is None:
            return col
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if is_numeric == numeric:
            return col

    for col in df.columns:
        col_name = str(col)
        key = _norm_key(col_name)
        if not key:
            continue
        if any(candidate in key for candidate in map(_norm_key, candidates)):
            if numeric is None:
                return col_name
            is_numeric = pd.api.types.is_numeric_dtype(df[col_name])
            if is_numeric == numeric:
                return col_name
    return None


# ---------------------------------------------------------------------------
# Orange Table output helper
# ---------------------------------------------------------------------------

def _dataframe_to_orange(df: pd.DataFrame) -> Table:
    if df is None or df.empty:
        return Table.from_numpy(Domain([]), X=np.empty((0, 0)))
    attrs = []
    metas = []
    X_cols = []
    M_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            attrs.append(ContinuousVariable(str(col)))
            X_cols.append(col)
        else:
            metas.append(StringVariable(str(col)))
            M_cols.append(col)
    X = df[X_cols].astype(float).to_numpy(dtype=float) if X_cols else np.empty((len(df), 0), dtype=float)
    M = (
        df[M_cols].fillna("").astype(str).to_numpy(dtype=object)
        if M_cols
        else np.empty((len(df), 0), dtype=object)
    )
    return Table.from_numpy(Domain(attrs, metas=metas), X=X, metas=M)


# ---------------------------------------------------------------------------
# Plot styling helper
# ---------------------------------------------------------------------------

def _style_plot(pw: pg.PlotWidget) -> None:
    pw.setBackground("#FFFFFF")
    pw.showGrid(x=True, y=True, alpha=0.18)
    for axis_name in ("left", "bottom", "right", "top"):
        ax = pw.getPlotItem().getAxis(axis_name)
        ax.setPen(pg.mkPen("#CBD5E1"))
        ax.setTextPen(pg.mkPen("#475569"))


def _count_card(title: str, value: str, subtitle: str) -> str:
    return (
        "<div style='display:inline-block; min-width:150px; margin:0 10px 10px 0; padding:12px 14px;"
        "border:1px solid #d7dee8; border-radius:8px; background:#ffffff;'>"
        f"<div style='font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:0.04em;'>{html.escape(title)}</div>"
        f"<div style='font-size:22px; font-weight:700; color:#0f172a; margin-top:4px;'>{html.escape(value)}</div>"
        f"<div style='font-size:11px; color:#475569; margin-top:4px;'>{html.escape(subtitle)}</div>"
        "</div>"
    )


def _render_validation_summary_html(
    result,
    *,
    observed_column: str,
    predicted_column: str,
    split_column: str,
    id_column: str,
) -> str:
    summary = result.summary or {}
    metrics = result.metrics if result.metrics is not None else pd.DataFrame()
    diagnostics = result.diagnostics if result.diagnostics is not None else pd.DataFrame()
    outliers = result.outliers if result.outliers is not None else pd.DataFrame()
    overall = summary.get("overall_metrics", {}) or {}
    split_groups = 0
    if split_column in diagnostics.columns:
        split_groups = int(diagnostics[split_column].dropna().astype(str).nunique())
    top_reasons = {}
    if "review_reason" in outliers.columns:
        top_reasons = outliers["review_reason"].astype(str).value_counts().head(3).to_dict()
    used_observed = str(summary.get("observed_column_used") or observed_column or "n/a")
    used_predicted = str(summary.get("predicted_column_used") or predicted_column or "n/a")
    used_split = summary.get("split_column_used")
    used_id = summary.get("id_column_used")

    cards = "".join(
        [
            _count_card("Rows", str(summary.get("n_rows", 0)), "validated prediction records"),
            _count_card("Outliers", str(summary.get("n_outliers", 0)), "rows flagged for review"),
            _count_card("R²", f"{float(overall.get('r2', float('nan'))):.3f}" if pd.notna(overall.get("r2")) else "n/a", "overall fit"),
            _count_card("RMSE", f"{float(overall.get('rmse', float('nan'))):.3f}" if pd.notna(overall.get("rmse")) else "n/a", "overall prediction error"),
            _count_card("MAE", f"{float(overall.get('mae', float('nan'))):.3f}" if pd.notna(overall.get("mae")) else "n/a", "overall absolute error"),
            _count_card("CCC", f"{float(overall.get('ccc', float('nan'))):.3f}" if pd.notna(overall.get("ccc")) else "n/a", "observed/predicted agreement"),
            _count_card("AD Coverage", f"{100.0 * float(summary.get('ad_coverage')):.1f}%" if pd.notna(summary.get("ad_coverage")) else "n/a", "inside applicability domain"),
        ]
    )

    notes = [
        f"Observed column: {used_observed}",
        f"Predicted column: {used_predicted}",
        f"Split column: {used_split if used_split is not None else 'not used'}",
        f"ID column: {used_id if used_id is not None else 'not found'}",
        f"Residual threshold: {float(summary.get('residual_threshold', float('nan'))):.3f}" if pd.notna(summary.get("residual_threshold")) else "Residual threshold: n/a",
        f"Z threshold: {float(summary.get('z_threshold', float('nan'))):.2f}" if pd.notna(summary.get("z_threshold")) else "Z threshold: n/a",
        f"Split groups: {split_groups}",
        f"Metrics rows: {len(metrics)}",
        f"Outlier rows: {len(outliers)}",
        f"Large residuals: {int(summary.get('n_large_residuals', 0))}",
        f"Residual z-outliers: {int(summary.get('n_z_outliers', 0))}",
        f"Outside AD: {int(summary.get('n_outside_ad', 0))}",
        f"Low AD confidence: {int(summary.get('n_low_ad_confidence', 0))}",
        f"Critical reviews: {int(summary.get('n_critical', 0))}",
        f"Warnings: {int(summary.get('n_warning', 0))}",
    ]
    notes_html = "".join(f"<li>{html.escape(line)}</li>" for line in notes)
    reasons_html = ""
    if top_reasons:
        reasons_list = "".join(
            f"<li>{html.escape(reason)}: {count}</li>" for reason, count in top_reasons.items()
        )
        reasons_html = (
            "<div style='margin-top: 14px; padding: 12px 14px; border:1px solid #d7dee8; border-radius:8px; background:#ffffff;'>"
            "<div style='font-size:13px; font-weight:600; margin-bottom:8px;'>Top review reasons</div>"
            f"<ul style='margin:0; padding-left:18px; color:#334155;'>{reasons_list}</ul>"
            "</div>"
        )

    return (
        "<html><body style='font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; "
        "font-size: 12px; color: #0f172a; background: #f8fafc;'>"
        "<h2 style='margin: 0 0 12px 0;'>Validation Summary</h2>"
        f"<div>{cards}</div>"
        "<div style='margin-top: 14px; padding: 12px 14px; border:1px solid #d7dee8; border-radius:8px; background:#ffffff;'>"
        "<div style='font-size:13px; font-weight:600; margin-bottom:8px;'>Configuration and coverage</div>"
        f"<ul style='margin:0; padding-left:18px; color:#334155;'>{notes_html}</ul>"
        "</div>"
        f"{reasons_html}"
        "<p style='margin-top:14px; color:#475569;'>"
        "Use the Diagnostics tab for visual selection and the Outliers tab for row-wise review of flagged compounds."
        "</p>"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class OWQSARValidationDashboard(OWWidget):
    name = "QSAR Validation Dashboard"
    description = "Evaluate QSAR predictions, residuals, split-level metrics, and outlier records."
    icon = "icons/modeling/ow_qsar_validation_dashboard.png"
    priority = 145
    keywords = ["QSAR", "validation", "metrics", "residuals"]

    want_main_area = True
    resizing_enabled = True

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    class Inputs:
        predictions = Input("Predictions", Table)

    class Outputs:
        validation_metrics = Output("Validation Metrics", Table)
        residual_diagnostics = Output("Residual Diagnostics", Table)
        outlier_records = Output("Outlier Records", Table)
        validation_summary = Output("Validation Summary", Table)
        selected_compounds = Output("Selected Compounds", Table)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    observed_column: str = Setting("observed")
    predicted_column: str = Setting("predicted")
    split_column: str = Setting("split")
    id_column: str = Setting("compound_id")
    residual_threshold: str = Setting("")
    z_threshold: float = Setting(3.0)
    auto_run: bool = Setting(True)
    selection_tool: int = Setting(0)

    _SELECTION_TOOL_OPTIONS = list(SELECTION_TOOL_OPTIONS)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self._predictions_data = None
        self._executor = ThreadExecutor()
        self._task = None
        self._diagnostics_table = None
        self._diagnostics_df = None
        self._outliers_df = None
        self._summary_result = None
        self._syncing_outliers_table = False
        self._diagnostic_selectors = {}
        self._diagnostic_context = None
        self._diagnostic_canvas = None
        self._diagnostic_fig = None

        self._build_control_area()
        self._build_main_area()

    # ------------------------------------------------------------------
    # Control area
    # ------------------------------------------------------------------

    def _build_control_area(self):
        ca = self.controlArea
        ca_layout = QVBoxLayout()
        ca.setLayout(ca_layout)

        # --- Header ---
        header_widget = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_widget.setLayout(header_layout)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        lbl_title = QLabel("QSAR Validation")
        lbl_title.setObjectName("HdrTitle")
        lbl_title.setStyleSheet("font-size:14px;font-weight:600;color:#1e293b;")
        lbl_sub = QLabel("Evaluate predictions \u00b7 residuals \u00b7 outliers")
        lbl_sub.setObjectName("HdrSub")
        lbl_sub.setStyleSheet("font-size:11px;color:#64748b;")
        title_col.addWidget(lbl_title)
        title_col.addWidget(lbl_sub)

        self._status_chip = QLabel("Idle")
        self._status_chip.setObjectName("StatusChip")
        self._set_chip_ok("Idle")
        self._status_chip.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header_layout.addLayout(title_col)
        header_layout.addStretch(1)
        header_layout.addWidget(self._status_chip)
        ca_layout.addWidget(header_widget)

        # --- Prediction columns group ---
        pred_box = QGroupBox("Prediction columns")
        pred_layout = QVBoxLayout()
        pred_layout.setSpacing(6)
        pred_box.setLayout(pred_layout)

        self._le_observed = self._make_row(pred_layout, "Observed", self.observed_column)
        self._le_predicted = self._make_row(pred_layout, "Predicted", self.predicted_column)
        self._le_split = self._make_row(pred_layout, "Split column", self.split_column)
        self._le_id = self._make_row(pred_layout, "ID column", self.id_column)

        self._le_observed.textChanged.connect(self._on_le_observed)
        self._le_predicted.textChanged.connect(self._on_le_predicted)
        self._le_split.textChanged.connect(self._on_le_split)
        self._le_id.textChanged.connect(self._on_le_id)

        ca_layout.addWidget(pred_box)

        # --- Outlier flags group ---
        out_box = QGroupBox("Outlier flags")
        out_layout = QVBoxLayout()
        out_layout.setSpacing(6)
        out_box.setLayout(out_layout)

        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z threshold"))
        self._spin_z = QDoubleSpinBox()
        self._spin_z.setRange(0.5, 10.0)
        self._spin_z.setSingleStep(0.5)
        self._spin_z.setValue(self.z_threshold)
        self._spin_z.valueChanged.connect(self._on_z_changed)
        z_row.addWidget(self._spin_z)
        out_layout.addLayout(z_row)

        self._chk_auto = QCheckBox("Auto-run")
        self._chk_auto.setChecked(bool(self.auto_run))
        self._chk_auto.stateChanged.connect(self._on_auto_changed)
        out_layout.addWidget(self._chk_auto)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Selection"))
        self._cmb_selection_tool = QComboBox()
        self._cmb_selection_tool.addItems(self._SELECTION_TOOL_OPTIONS)
        self._cmb_selection_tool.setCurrentIndex(int(self.selection_tool))
        self._cmb_selection_tool.currentIndexChanged.connect(self._on_selection_tool_changed)
        sel_row.addWidget(self._cmb_selection_tool)
        out_layout.addLayout(sel_row)

        self._btn_validate = QPushButton("Validate")
        self._btn_validate.clicked.connect(self.commit)
        out_layout.addWidget(self._btn_validate)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        out_layout.addWidget(self._progress)

        ca_layout.addWidget(out_box)
        ca_layout.addStretch(1)

    @staticmethod
    def _make_row(parent_layout: QVBoxLayout, label: str, default: str) -> QLineEdit:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(90)
        le = QLineEdit(default)
        row.addWidget(lbl)
        row.addWidget(le)
        parent_layout.addLayout(row)
        return le

    # Slot helpers for QLineEdit changes
    def _on_le_observed(self, txt):
        self.observed_column = txt
        self._settings_changed()

    def _on_le_predicted(self, txt):
        self.predicted_column = txt
        self._settings_changed()

    def _on_le_split(self, txt):
        self.split_column = txt
        self._settings_changed()

    def _on_le_id(self, txt):
        self.id_column = txt
        self._settings_changed()

    def _on_z_changed(self, val):
        self.z_threshold = val
        self._settings_changed()

    def _on_auto_changed(self, state):
        self.auto_run = bool(state)
        self._settings_changed()

    def _on_selection_tool_changed(self, index):
        self.selection_tool = int(index)
        self._refresh_selector_modes()

    @staticmethod
    def _set_line_edit_text(line_edit: QLineEdit, text: str) -> None:
        if line_edit.text() == text:
            return
        was_blocked = line_edit.blockSignals(True)
        line_edit.setText(text)
        line_edit.blockSignals(was_blocked)

    def _autodetect_prediction_columns(self, data: Table) -> None:
        df = _table_to_df(data)
        if df is None or df.empty:
            return

        detected_observed = _detect_column_name(df, _OBSERVED_CANDIDATES, numeric=True)
        detected_predicted = _detect_column_name(df, _PREDICTED_CANDIDATES, numeric=True)
        detected_split = _detect_column_name(df, _SPLIT_CANDIDATES, numeric=None)
        detected_id = _detect_column_name(df, _ID_CANDIDATES, numeric=None)

        defaults = {
            "observed_column": "observed",
            "predicted_column": "predicted",
            "split_column": "split",
            "id_column": "compound_id",
        }
        updates = [
            ("observed_column", detected_observed, self._le_observed),
            ("predicted_column", detected_predicted, self._le_predicted),
            ("split_column", detected_split, self._le_split),
            ("id_column", detected_id, self._le_id),
        ]
        available = {str(col) for col in df.columns}
        for attr_name, detected_value, line_edit in updates:
            current_value = getattr(self, attr_name, "").strip()
            should_replace = (
                (not current_value)
                or (current_value not in available)
                or (current_value == defaults[attr_name] and detected_value is not None and detected_value != current_value)
            )
            if detected_value and should_replace:
                setattr(self, attr_name, detected_value)
                self._set_line_edit_text(line_edit, detected_value)

    def _set_chip_ok(self, text: str):
        self._status_chip.setText(text)
        self._status_chip.setStyleSheet(
            "padding:4px 8px;border:1px solid #e1e1e1;border-radius:10px;"
            "background:#fafafa;"
        )

    def _set_chip_error(self, text: str):
        self._status_chip.setText(text)
        self._status_chip.setStyleSheet(
            "padding:4px 8px;border:1px solid #f2c2c2;border-radius:10px;"
            "background:#fff5f5;color:#a40000;"
        )

    # ------------------------------------------------------------------
    # Main area
    # ------------------------------------------------------------------

    def _build_main_area(self):
        ma = self.mainArea
        ma_layout = QVBoxLayout()
        ma_layout.setContentsMargins(4, 4, 4, 4)
        ma.setLayout(ma_layout)

        self._tabs = QTabWidget()
        ma_layout.addWidget(self._tabs)

        # Tab 1: Summary
        self._tab_summary = QWidget()
        summary_layout = QVBoxLayout()
        self._tab_summary.setLayout(summary_layout)
        self._summary_browser = QTextBrowser()
        self._summary_browser.setOpenExternalLinks(False)
        self._summary_browser.setHtml("<p><i>Awaiting validation results.</i></p>")
        summary_layout.addWidget(self._summary_browser)
        self._tabs.addTab(self._tab_summary, "Summary")

        # Tab 2: Diagnostics
        self._tab_diagnostics = QWidget()
        t1_layout = QVBoxLayout()
        self._tab_diagnostics.setLayout(t1_layout)
        self._diagnostics_hint = QLabel(
            "Predicted vs observed and residual diagnostics. Use rectangle or lasso selection to inspect compounds."
        )
        self._diagnostics_hint.setWordWrap(True)
        self._diagnostics_hint.setStyleSheet("color:#475569;font-size:11px;")
        t1_layout.addWidget(self._diagnostics_hint)

        self._diagnostics_canvas_container = QWidget()
        self._diagnostics_canvas_layout = QVBoxLayout(self._diagnostics_canvas_container)
        self._diagnostics_canvas_layout.setContentsMargins(0, 0, 0, 0)
        t1_layout.addWidget(self._diagnostics_canvas_container, 4)

        self._selection_gallery_label = QLabel("Selected compounds")
        self._selection_gallery_label.setStyleSheet("font-weight:600; font-size:13px; color:#0F172A;")
        t1_layout.addWidget(self._selection_gallery_label)

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
        t1_layout.addWidget(self._selection_gallery_scroll, 2)
        self._tabs.addTab(self._tab_diagnostics, "Diagnostics")

        # Tab 3: Selected compounds
        self._tab_selected = QWidget()
        t2_layout = QVBoxLayout()
        self._tab_selected.setLayout(t2_layout)
        self._selected_table = QTableWidget()
        self._selected_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._selected_table.setAlternatingRowColors(True)
        self._selected_table.horizontalHeader().setStretchLastSection(True)
        t2_layout.addWidget(self._selected_table)
        self._tabs.addTab(self._tab_selected, "Selected")

        # Tab 4: Metrics
        self._tab_metrics = QWidget()
        t3_layout = QVBoxLayout()
        self._tab_metrics.setLayout(t3_layout)
        metrics_grid = QGridLayout()
        metrics_grid.setContentsMargins(0, 0, 0, 0)
        metrics_grid.setHorizontalSpacing(8)
        metrics_grid.setVerticalSpacing(8)
        self._metric_plot_widgets = {}
        for index, (metric_col, metric_label) in enumerate(_METRIC_PLOT_SPECS):
            pw = pg.PlotWidget()
            _style_plot(pw)
            pw.setLabel("left", metric_label)
            pw.setMaximumHeight(180)
            pw.getPlotItem().setTitle(metric_label)
            self._metric_plot_widgets[metric_col] = pw
            metrics_grid.addWidget(pw, index // 2, index % 2)
        t3_layout.addLayout(metrics_grid)

        self._metrics_table_label = QLabel("Per-split validation metrics")
        self._metrics_table_label.setStyleSheet("font-weight:600; color:#0f172a;")
        t3_layout.addWidget(self._metrics_table_label)

        self._metrics_table = QTableWidget()
        self._metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._metrics_table.setAlternatingRowColors(True)
        self._metrics_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._metrics_table.horizontalHeader().setStretchLastSection(True)
        t3_layout.addWidget(self._metrics_table, 1)
        self._tabs.addTab(self._tab_metrics, "Metrics")

        # Tab 5: Outliers
        self._tab_outliers = QWidget()
        t4_layout = QVBoxLayout()
        self._tab_outliers.setLayout(t4_layout)
        self._outliers_table = QTableWidget()
        self._outliers_table.setColumnCount(8)
        self._outliers_table.setHorizontalHeaderLabels(
            ["ID", "Observed", "Predicted", "Residual", "Z-score", "AD", "Flag", "Reason"]
        )
        self._outliers_table.setAlternatingRowColors(True)
        self._outliers_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._outliers_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._outliers_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._outliers_table.horizontalHeader().setStretchLastSection(True)
        self._outliers_table.itemSelectionChanged.connect(self._on_outlier_selection_changed)
        t4_layout.addWidget(self._outliers_table)
        self._tabs.addTab(self._tab_outliers, "Outliers")
        self._show_selection_gallery_placeholder("Validate predictions, then select points on the diagnostics plot.")
        self._update_selected_table(None)

    # ------------------------------------------------------------------
    # Input handler
    # ------------------------------------------------------------------

    @Inputs.predictions
    def set_predictions(self, data):
        self._predictions_data = data
        if data is None:
            self._set_chip_ok("No data")
            self._summary_browser.setHtml("<p><i>Awaiting validation results.</i></p>")
        else:
            self._autodetect_prediction_columns(data)
            self._set_chip_ok(f"{len(data)} rows")
        if self.auto_run:
            self.commit()

    def _settings_changed(self):
        if self.auto_run and self._predictions_data is not None:
            self.commit()

    # ------------------------------------------------------------------
    # Async commit
    # ------------------------------------------------------------------

    def commit(self):
        if self._predictions_data is None:
            self._send_nones()
            self._set_chip_ok("No data")
            return

        # cancel any running task
        if self._task is not None:
            self._task.cancel()
            self._task = None

        self._btn_validate.setEnabled(False)
        self._progress.setVisible(True)
        self._set_chip_ok("Running\u2026")
        self.Outputs.selected_compounds.send(None)
        self._update_selected_table(None)
        self._reset_diagnostics_view("Validation in progress — diagnostics will appear when the run is ready.")
        self._summary_browser.setHtml("<p><i>Validation in progress...</i></p>")

        df = _table_to_df(self._predictions_data)
        threshold = None
        if str(self.residual_threshold).strip():
            with suppress(ValueError):
                threshold = float(str(self.residual_threshold).strip())

        cfg = QSARValidationConfig(
            observed_column=self.observed_column.strip() or "observed",
            predicted_column=self.predicted_column.strip() or "predicted",
            split_column=self.split_column.strip() or "split",
            id_column=self.id_column.strip() or "compound_id",
            residual_threshold=threshold,
            z_threshold=float(self.z_threshold),
        )

        def _run():
            return validate_qsar_predictions(df, cfg)

        self._task = self._executor.submit(
            _run,
            lambda result: methodinvoke(self, "_finish", (object,))(result),
            lambda exc: methodinvoke(self, "_fail", (str,))(str(exc)),
        )

    @Slot(object)
    def _finish(self, result):
        self._task = None
        self._btn_validate.setEnabled(True)
        self._progress.setVisible(False)

        summary = result.summary
        n_rows = summary.get("n_rows", "?")
        n_out = summary.get("n_outliers", "?")
        self._set_chip_ok(f"{n_rows} rows \u00b7 {n_out} flags")

        # Reconstruct predictions df from diagnostics (contains obs, pred, residual, id, split)
        diag_df = result.diagnostics
        self._diagnostics_df = diag_df.copy() if diag_df is not None else None
        self._outliers_df = result.outliers.copy() if result.outliers is not None else None
        self._summary_result = result
        self._diagnostics_table = _dataframe_to_orange(result.diagnostics)
        self._summary_browser.setHtml(
            _render_validation_summary_html(
                result,
                observed_column=self.observed_column.strip() or "observed",
                predicted_column=self.predicted_column.strip() or "predicted",
                split_column=self.split_column.strip() or "split",
                id_column=self.id_column.strip() or "compound_id",
            )
        )

        self._update_diagnostics(diag_df)
        self._update_metrics(result.metrics)
        self._update_outliers(result.outliers)

        self.Outputs.validation_metrics.send(_dataframe_to_orange(result.metrics))
        self.Outputs.residual_diagnostics.send(self._diagnostics_table)
        self.Outputs.outlier_records.send(_dataframe_to_orange(result.outliers))
        self.Outputs.validation_summary.send(
            _dataframe_to_orange(pd.DataFrame([result.summary]))
        )
        self.Outputs.selected_compounds.send(None)

    @Slot(str)
    def _fail(self, message: str):
        self._task = None
        self._btn_validate.setEnabled(True)
        self._progress.setVisible(False)
        self._set_chip_error("Error")
        self._send_nones()
        self._summary_browser.setHtml(f"<p><b>Validation failed.</b></p><p>{html.escape(message)}</p>")

    def _send_nones(self):
        self._diagnostics_table = None
        self._diagnostics_df = None
        self._outliers_df = None
        self._summary_result = None
        self.Outputs.validation_metrics.send(None)
        self.Outputs.residual_diagnostics.send(None)
        self.Outputs.outlier_records.send(None)
        self.Outputs.validation_summary.send(None)
        self.Outputs.selected_compounds.send(None)
        self._update_selected_table(None)
        self._reset_diagnostics_view("Awaiting validation results to render diagnostics.")
        self._summary_browser.setHtml("<p><i>Awaiting validation results.</i></p>")

    # ------------------------------------------------------------------
    # Plot update helpers
    # ------------------------------------------------------------------

    def _resolve_col(self, df: pd.DataFrame, setting_name: str, fallback: str) -> str:
        val = getattr(self, setting_name, "").strip() or fallback
        return val if val in df.columns else fallback

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

    def _update_diagnostics(self, df: pd.DataFrame):
        obs_col = self._resolve_col(df, "observed_column", "observed")
        pred_col = self._resolve_col(df, "predicted_column", "predicted")
        split_col = self._resolve_col(df, "split_column", "split")

        if obs_col not in df.columns or pred_col not in df.columns or self._diagnostics_table is None:
            self._reset_diagnostics_view("No diagnostics available.")
            return

        preds = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
        actuals = pd.to_numeric(df[obs_col], errors="coerce").to_numpy(dtype=float)
        residuals = actuals - preds
        levels = _residual_reference_levels(residuals)

        self._reset_diagnostics_view(
            "Drag a rectangle or lasso on either plot to select compounds and inspect them below."
        )
        self._update_selected_table(None)

        fig = Figure(figsize=(11.8, 5.6))
        ax_left = fig.add_subplot(121)
        ax_right = fig.add_subplot(122)

        _named_colors = {
            "train": "#2563EB",
            "test": "#EA580C",
        }
        _fallback_colors = ["#2563EB", "#EA580C", "#16A34A", "#9333EA", "#D97706", "#0891B2"]

        if split_col in df.columns:
            split_groups = [(str(v), df[df[split_col].astype(str) == str(v)]) for v in df[split_col].dropna().unique()]
        else:
            split_groups = [("All", df)]

        plotted_any = False
        for i, (label, sub) in enumerate(split_groups):
            if sub.empty:
                continue
            plotted_any = True
            color = _named_colors.get(label.lower(), _fallback_colors[i % len(_fallback_colors)])
            ax_left.scatter(
                pd.to_numeric(sub[pred_col], errors="coerce"),
                pd.to_numeric(sub[obs_col], errors="coerce"),
                s=46,
                alpha=0.78,
                c=color,
                edgecolors="#0F172A",
                linewidths=0.35,
                label=label,
            )
            ax_right.scatter(
                pd.to_numeric(sub[pred_col], errors="coerce"),
                pd.to_numeric(sub["residual"], errors="coerce"),
                s=46,
                alpha=0.78,
                c=color,
                edgecolors="#0F172A",
                linewidths=0.35,
                label=label,
            )

        if not plotted_any:
            self._reset_diagnostics_view("No diagnostics available.")
            return

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

        ax_left.plot([diag_min, diag_max], [diag_min, diag_max], color="#475569", linestyle="-", linewidth=1.5, label="Ideal")
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
        ax_left.set_xlabel("Predicted")
        ax_left.set_ylabel("Observed")
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
        ax_right.set_xlabel("Predicted")
        ax_right.set_ylabel("Residual")
        ax_right.grid(alpha=0.28, linewidth=0.8)
        ax_right.legend(loc="best", frameon=True, framealpha=0.92, fontsize=8)

        info_lines = [
            f"Rows: {len(df)}",
            f"Residual σ: {levels['std']:.3f}",
            f"Mean residual: {levels['mean']:.3f}",
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
            table=self._diagnostics_table,
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
    ):
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

    def _refresh_selector_modes(self):
        use_lasso = int(self.selection_tool) == 1
        for selectors in self._diagnostic_selectors.values():
            qsar_diagnostics_ui.set_selector_mode(selectors, use_lasso=use_lasso)

    def _apply_plot_selection(self, eclick, erelease, *, left_plot):
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

    def _apply_lasso_selection(self, vertices, *, left_plot):
        context = self._diagnostic_context
        if context is None or context.table is None or not vertices:
            return
        preds, ys = qsar_diagnostics_ui.selection_plot_values(context, left_plot=left_plot)
        selected_idx = qsar_service.lasso_selection_indices(preds, ys, vertices)
        self._publish_selection(selected_idx)

    def _publish_selection(self, selected_idx):
        context = self._diagnostic_context
        if context is None or context.table is None:
            return
        qsar_diagnostics_ui.update_selection_overlays(context, selected_idx)
        payload = qsar_service.build_selection_publish_payload(
            model_name="Validation Dashboard",
            dataset_type="validation",
            table=context.table,
            selected_idx=selected_idx,
        )
        self.Outputs.selected_compounds.send(payload.selected_table)
        self._update_selection_gallery(payload.gallery)
        self._update_selected_table(payload.selected_table)
        self._diagnostics_hint.setText(payload.status_text)
        if payload.selected_table is not None and len(payload.selected_table) > 0:
            self._tabs.setCurrentWidget(self._tab_selected)

    def _clear_selection_gallery(self):
        while self._selection_gallery_layout.count():
            item = self._selection_gallery_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _show_selection_gallery_placeholder(self, text):
        self._clear_selection_gallery()
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color:#64748B; padding:6px;")
        self._selection_gallery_layout.addWidget(label)
        self._selection_gallery_layout.addStretch(1)

    def _update_selection_gallery(self, payload):
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
            txt_label.setStyleSheet("font-size:11px; border:none; color:#334155;")
            card_layout.addWidget(img_label)
            card_layout.addWidget(txt_label)
            self._selection_gallery_layout.addWidget(card)

        if payload.more_count > 0:
            more_label = QLabel(f"+ {payload.more_count} more")
            more_label.setStyleSheet("color:#64748B; padding:12px;")
            self._selection_gallery_layout.addWidget(more_label)
        self._selection_gallery_layout.addStretch(1)

    def _update_selected_table(self, selected_table):
        selected_tab_index = self._tabs.indexOf(self._tab_selected)
        if selected_table is None or len(selected_table) == 0:
            self._selected_table.clearContents()
            self._selected_table.setRowCount(0)
            self._selected_table.setColumnCount(1)
            self._selected_table.setHorizontalHeaderLabels(["No selected compounds"])
            if selected_tab_index >= 0:
                self._tabs.setTabText(selected_tab_index, "Selected")
            return

        df = _table_to_df(selected_table)
        if df is None or df.empty:
            self._selected_table.clearContents()
            self._selected_table.setRowCount(0)
            self._selected_table.setColumnCount(1)
            self._selected_table.setHorizontalHeaderLabels(["No selected compounds"])
            if selected_tab_index >= 0:
                self._tabs.setTabText(selected_tab_index, "Selected")
            return

        cols = [str(col) for col in df.columns]
        self._selected_table.clearContents()
        self._selected_table.setColumnCount(len(cols))
        self._selected_table.setHorizontalHeaderLabels(cols)
        self._selected_table.setRowCount(len(df))
        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, col in enumerate(cols):
                value = row[col]
                if isinstance(value, (float, np.floating)):
                    text = f"{float(value):.4f}" if np.isfinite(float(value)) else ""
                else:
                    text = "" if pd.isna(value) else str(value)
                self._selected_table.setItem(row_idx, col_idx, QTableWidgetItem(text))
        self._selected_table.resizeColumnsToContents()
        if selected_tab_index >= 0:
            self._tabs.setTabText(selected_tab_index, f"Selected ({len(df)})")

    def _update_metrics(self, metrics_df: pd.DataFrame):
        for pw in self._metric_plot_widgets.values():
            pw.clear()
        self._metrics_table.clearContents()
        self._metrics_table.setRowCount(0)
        self._metrics_table.setColumnCount(0)

        if metrics_df is None or metrics_df.empty:
            return

        required = {"group"}
        if not required.issubset(set(metrics_df.columns)):
            return

        groups = metrics_df["group"].tolist()
        x_positions = np.arange(len(groups))

        colors = [(37, 99, 235, 210), (234, 88, 12, 210)]

        for metric_col, pw in self._metric_plot_widgets.items():
            if metric_col not in metrics_df.columns:
                continue
            for i, (_grp, row) in enumerate(metrics_df.iterrows()):
                if pd.isna(row[metric_col]):
                    continue
                val = float(row[metric_col])
                color = colors[i % len(colors)]
                bar = pg.BarGraphItem(
                    x=[x_positions[i]],
                    height=[val],
                    width=0.6,
                    brush=pg.mkBrush(*color),
                    pen=pg.mkPen(None),
                )
                pw.addItem(bar)

                text = pg.TextItem(
                    text=f"{val:.3f}",
                    color="#1e293b",
                    anchor=(0.5, 1.0),
                )
                text.setPos(x_positions[i], val)
                pw.addItem(text)

            ax = pw.getPlotItem().getAxis("bottom")
            ax.setTicks([list(zip(x_positions, [str(g) for g in groups]))])

        available_columns = [
            (column_name, header)
            for column_name, header in _METRIC_TABLE_COLUMNS
            if column_name in metrics_df.columns
        ]
        self._metrics_table.setColumnCount(len(available_columns))
        self._metrics_table.setHorizontalHeaderLabels([header for _, header in available_columns])
        self._metrics_table.setRowCount(len(metrics_df))
        for row_idx, (_, row) in enumerate(metrics_df.iterrows()):
            for col_idx, (column_name, _header) in enumerate(available_columns):
                value = row[column_name]
                if pd.isna(value):
                    text = ""
                elif isinstance(value, (int, np.integer)):
                    text = str(int(value))
                elif isinstance(value, (float, np.floating)):
                    text = f"{float(value):.4f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self._metrics_table.setItem(row_idx, col_idx, item)
        self._metrics_table.resizeColumnsToContents()

    def _resolved_outlier_id_column(self, outliers_df: pd.DataFrame) -> str | None:
        available = {str(column) for column in outliers_df.columns}
        summary = self._summary_result.summary if self._summary_result is not None else {}

        resolved = summary.get("id_column_used")
        if resolved in available:
            return str(resolved)
        if "compound_id" in available:
            return "compound_id"

        requested = self.id_column.strip()
        if requested in available:
            return requested

        for candidate in ("name", "molecule_id", "mol_id", "id"):
            if candidate in available:
                return candidate
        return None

    def _update_outliers(self, outliers_df: pd.DataFrame):
        tbl = self._outliers_table
        self._syncing_outliers_table = True
        tbl.blockSignals(True)
        tbl.clearSelection()
        tbl.setRowCount(0)

        if outliers_df is None or outliers_df.empty:
            self._outliers_df = None
            tbl.blockSignals(False)
            self._syncing_outliers_table = False
            return

        work_df = outliers_df.copy()
        if "abs_residual" in work_df.columns:
            work_df = work_df.sort_values("abs_residual", ascending=False)
        self._outliers_df = work_df

        id_column = self._resolved_outlier_id_column(work_df)
        col_map = {
            0: id_column,
            1: "observed",
            2: "predicted",
            3: "residual",
            4: "residual_z",
            5: "ad_flag",
            6: "validation_flag",
            7: "review_reason",
        }

        # Fall back gracefully for missing columns
        available = set(work_df.columns)

        tbl.setRowCount(len(work_df))
        for row_idx, (_, row) in enumerate(work_df.iterrows()):
            for col_idx, col_name in col_map.items():
                if col_name in available:
                    val = row[col_name]
                    text = f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
                else:
                    text = ""
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                tbl.setItem(row_idx, col_idx, item)

        tbl.resizeColumnsToContents()
        tbl.blockSignals(False)
        self._syncing_outliers_table = False

    def _on_outlier_selection_changed(self) -> None:
        if self._syncing_outliers_table or self._outliers_df is None or self._outliers_df.empty:
            return
        selected_rows = sorted({index.row() for index in self._outliers_table.selectionModel().selectedRows()})
        if not selected_rows:
            self._publish_selection(np.array([], dtype=int))
            return
        source_indices = self._outliers_df.iloc[selected_rows].index.to_numpy(dtype=int)
        self._publish_selection(source_indices)
