from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor, QPixmap
from AnyQt.QtWidgets import (
    QFrame, QLabel, QScrollArea, QSizePolicy, QSplitter,
    QTableWidget, QTableWidgetItem,
    QTextBrowser, QTabWidget, QWidget, QVBoxLayout,
)

from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output

from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector, RectangleSelector
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from chem_inf_widgets.chemcore.services import mlr_model_selection_service as mlr_service
from chem_inf_widgets.chemcore.services import qsar_regression_service as qsar_service
from chem_inf_widgets.chemcore.services.qsar_diagnostics_contract import (
    SELECTION_TOOL_OPTIONS,
    residual_reference_levels as _residual_reference_levels,
)
from chem_inf_widgets.chemcore.services.qsar_prediction_packager_service import (
    build_qsar_prediction_bundle,
)
from chem_inf_widgets.chemcore.qsar.mlr_selection import (
    ADConfig,
    SelectionConfig,
    filter_low_variance,
    filter_high_correlation,
    fit_mlr_with_selection,
    predict_with_ad,
    regression_metrics,
    external_validation_metrics,
    permutation_test_cv_q2,
)
from chem_inf_widgets.widgets import qsar_diagnostics_ui


@dataclass
class MLRModelBundle:
    """A small, serializable bundle to re-use the fitted model outside the widget."""
    y_name: str
    x_names_after_preprocess: List[str]
    selected_names: List[str]
    variance_kept_idx: np.ndarray
    corr_kept_idx: np.ndarray
    selected_idx_in_corr_space: np.ndarray
    imputer: SimpleImputer
    model: LinearRegression
    h_star: float

    def _transform(self, X_full: np.ndarray) -> np.ndarray:
        X = self.imputer.transform(X_full)
        X = X[:, self.variance_kept_idx]
        X = X[:, self.corr_kept_idx]
        X = X[:, self.selected_idx_in_corr_space]
        return X

    def predict(self, X_full: np.ndarray) -> np.ndarray:
        return self.model.predict(self._transform(X_full)).astype(float)


def _table_display_rows(table: Optional[Table]) -> tuple[list[str], list[list[str]]]:
    if table is None or len(table) == 0:
        return [], []
    variables = list(table.domain.attributes) + list(table.domain.class_vars) + list(table.domain.metas)
    columns = [var.name for var in variables]
    rows: list[list[str]] = []
    for row in table:
        values: list[str] = []
        for var in variables:
            value = row[var]
            if getattr(var, "is_continuous", False):
                try:
                    val = float(value)
                    values.append(f"{val:.4f}" if np.isfinite(val) else "")
                except Exception:
                    values.append("")
            else:
                values.append("" if value is None else str(value))
        rows.append(values)
    return columns, rows


# -----------------------------------------------------------------------------
# Widget
# -----------------------------------------------------------------------------

class OWMLRModelSelection(OWWidget):
    name = "MLR Model Selection"
    description = "Multiple Linear Regression with descriptor filtering + forward/backward/MC/GA selection and QSAR diagnostics."
    icon = "icons/modeling/qsar_regression.png"
    priority = 141

    class Inputs:
        data = Input("Data", Table)
        # NOTE: Do not use `optional=True` here; older Orange/Orange Widget
        # versions do not support that keyword and an unconnected input is
        # optional by default.
        test_data = Input("Test Data", Table)

    class Outputs:
        # Keep Outputs compatible across Orange versions (avoid extra kwargs).
        model = Output("Model", object, auto_summary=False)
        train_results = Output("Train Results", Table)
        test_results = Output("Test Results", Table)

        # New (2026-03): convenience outputs
        predictions = Output("Predictions", Table)   # alias for test/holdout results
        coefficients = Output("Coefficients", Table) # term/coef/se/t/p/vif table
        report_html = Output("Report HTML", str, auto_summary=False)     # full HTML report (incl. embedded plots)
        selected_compounds = Output("Selected Compounds", Table)

    # ---------------- settings ----------------
    y_name: str = Setting("")
    test_size: float = Setting(0.2)
    random_state: int = Setting(0)

    var_threshold: float = Setting(1e-12)
    corr_threshold: float = Setting(0.95)

    method: str = Setting("forward")   # forward/backward/montecarlo/genetic
    criterion: str = Setting("cv_r2")
    cv_folds: int = Setting(5)
    min_features: int = Setting(1)
    max_features: int = Setting(0)  # 0 = no explicit cap

    mc_iterations: int = Setting(2000)

    ga_population: int = Setting(80)
    ga_generations: int = Setting(60)
    ga_crossover: float = Setting(0.7)
    ga_mutation: float = Setting(0.02)
    ga_tournament: int = Setting(3)
    ga_elite: int = Setting(2)

    perm_n: int = Setting(100)

    # Applicability Domain (distance-based)
    ad_use_williams: bool = Setting(True)
    ad_use_knn: bool = Setting(False)
    ad_use_maha: bool = Setting(False)
    ad_combine_mode: str = Setting("and")
    ad_knn_k: int = Setting(5)
    ad_knn_quantile: float = Setting(0.95)
    ad_maha_alpha: float = Setting(0.95)
    ad_maha_use_chi2: bool = Setting(True)

    compute_external_metrics: bool = Setting(True)

    # View settings
    show_diagnostic_plots: bool = Setting(True)
    show_model_report: bool = Setting(True)
    auto_run: bool = Setting(True)
    selection_tool: int = Setting(0)
    selection_tool_options = list(SELECTION_TOOL_OPTIONS)

    def __init__(self):
        super().__init__()

        # Keep native Orange sizing. Avoid forced splitter sizes; these caused grey dead zones.
        if self.controlArea.layout() is not None:
            self.controlArea.layout().setSpacing(6)
        if self.mainArea.layout() is not None:
            self.mainArea.layout().setSpacing(6)

        self.data: Optional[Table] = None
        self.test_data: Optional[Table] = None
        self._selection_contexts: Dict[str, Dict[str, Any]] = {}
        self._selection_selectors: Dict[str, Dict[str, Any]] = {}
        self._latest_train_table_out: Optional[Table] = None
        self._latest_test_table_out: Optional[Table] = None

        # ------------- control panel -------------
        box = gui.widgetBox(self.controlArea, "Target / Split", spacing=8)
        self.y_combo = gui.comboBox(
            box, self, "y_name", label="Target (Y):", orientation=Qt.Horizontal,
            callback=self._deferred_apply, sendSelectedValue=True
        )
        gui.doubleSpin(
            box, self, "test_size", 0.05, 0.5, 0.05,
            label="Test size (if no Test Data):",
            orientation=Qt.Horizontal, callback=self._deferred_apply
        )
        gui.spin(
            box, self, "random_state", 0, 10_000, 1,
            label="Random seed:", orientation=Qt.Horizontal, callback=self._deferred_apply
        )

        pre = gui.widgetBox(self.controlArea, "Descriptor filtering", spacing=8)
        gui.doubleSpin(
            pre, self, "var_threshold", 0.0, 1e6, 1e-6,
            label="Low variance threshold:", orientation=Qt.Horizontal, callback=self._deferred_apply
        )
        gui.doubleSpin(
            pre, self, "corr_threshold", 0.5, 0.999, 0.01,
            label="High correlation |r| >= :", orientation=Qt.Horizontal, callback=self._deferred_apply
        )

        sel = gui.widgetBox(self.controlArea, "Selection method", spacing=8)
        gui.comboBox(
            sel, self, "method",
            items=["forward", "backward", "montecarlo", "genetic"],
            label="Method:", orientation=Qt.Horizontal,
            sendSelectedValue=True, callback=self._update_method_ui
        )
        gui.comboBox(
            sel, self, "criterion",
            items=["cv_r2", "cv_rmse", "adj_r2", "aic", "bic", "train_r2"],
            label="Criterion:", orientation=Qt.Horizontal,
            sendSelectedValue=True, callback=self._deferred_apply
        )
        gui.spin(sel, self, "cv_folds", 2, 20, 1, label="CV folds:", orientation=Qt.Horizontal, callback=self._deferred_apply)
        gui.spin(sel, self, "min_features", 1, 2000, 1, label="Min features:", orientation=Qt.Horizontal, callback=self._deferred_apply)
        gui.spin(sel, self, "max_features", 0, 2000, 1, label="Max features (0=all):", orientation=Qt.Horizontal, callback=self._deferred_apply)

        self.mc_box = gui.widgetBox(self.controlArea, "Monte-Carlo", spacing=8)
        gui.spin(self.mc_box, self, "mc_iterations", 100, 200_000, 100,
                 label="Iterations:", orientation=Qt.Horizontal, callback=self._deferred_apply)

        self.ga_box = gui.widgetBox(self.controlArea, "Genetic algorithm", spacing=8)
        gui.spin(self.ga_box, self, "ga_population", 10, 500, 1, label="Population:", orientation=Qt.Horizontal, callback=self._deferred_apply)
        gui.spin(self.ga_box, self, "ga_generations", 1, 500, 1, label="Generations:", orientation=Qt.Horizontal, callback=self._deferred_apply)
        gui.doubleSpin(self.ga_box, self, "ga_crossover", 0.0, 1.0, 0.05, label="Crossover:", orientation=Qt.Horizontal, callback=self._deferred_apply)
        gui.doubleSpin(self.ga_box, self, "ga_mutation", 0.0, 1.0, 0.01, label="Mutation:", orientation=Qt.Horizontal, callback=self._deferred_apply)
        gui.spin(self.ga_box, self, "ga_tournament", 2, 10, 1, label="Tournament:", orientation=Qt.Horizontal, callback=self._deferred_apply)
        gui.spin(self.ga_box, self, "ga_elite", 0, 20, 1, label="Elite:", orientation=Qt.Horizontal, callback=self._deferred_apply)

        ad = gui.widgetBox(self.controlArea, "Applicability Domain (AD)", spacing=8)
        gui.checkBox(ad, self, "ad_use_williams", "Williams (leverage)", callback=self._deferred_apply)
        gui.checkBox(ad, self, "ad_use_knn", "kNN distance", callback=self._deferred_apply)
        gui.spin(ad, self, "ad_knn_k", 1, 50, 1, label="kNN k:", orientation=Qt.Horizontal, callback=self._deferred_apply)
        gui.doubleSpin(ad, self, "ad_knn_quantile", 0.5, 0.999, 0.01, label="kNN quantile:", orientation=Qt.Horizontal, callback=self._deferred_apply)
        gui.checkBox(ad, self, "ad_use_maha", "Mahalanobis distance", callback=self._deferred_apply)
        gui.doubleSpin(ad, self, "ad_maha_alpha", 0.5, 0.999, 0.01, label="Mahalanobis α:", orientation=Qt.Horizontal, callback=self._deferred_apply)
        gui.checkBox(ad, self, "ad_maha_use_chi2", "Use chi-square threshold", callback=self._deferred_apply)
        gui.comboBox(ad, self, "ad_combine_mode", items=["and", "or"], label="Combine:", orientation=Qt.Horizontal, callback=self._deferred_apply)

        ext = gui.widgetBox(self.controlArea, "External validation", spacing=8)
        gui.checkBox(
            ext,
            self,
            "compute_external_metrics",
            "Compute CCC, r_m², Q²F1/F2/F3 (if test Y available)",
            callback=self._deferred_apply,
        )

        perm = gui.widgetBox(self.controlArea, "Y-randomization", spacing=8)
        gui.spin(perm, self, "perm_n", 0, 2000, 10, label="# permutations (0=off):", orientation=Qt.Horizontal, callback=self._deferred_apply)

        view = gui.widgetBox(self.controlArea, "View", spacing=6)
        gui.checkBox(
            view,
            self,
            "show_diagnostic_plots",
            "Show diagnostic plots",
            callback=self._apply_visibility_settings,
        )
        gui.checkBox(
            view,
            self,
            "show_model_report",
            "Show model report",
            callback=self._apply_visibility_settings,
        )
        gui.checkBox(
            view,
            self,
            "auto_run",
            "Auto-run",
            callback=self._deferred_apply,
        )
        self._cmb_selection_tool = gui.comboBox(
            view,
            self,
            "selection_tool",
            items=self.selection_tool_options,
            label="Selection:",
            orientation=Qt.Horizontal,
            callback=self._on_selection_tool_changed,
        )

        gui.button(self.controlArea, self, "Apply", callback=self.apply)

        # ------------- main area: clean modern layout -------------
        self.status_label = QLabel("No model fitted yet.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "QLabel {"
            "background: #f6f8fb;"
            "border: 1px solid #d7dee8;"
            "border-radius: 6px;"
            "padding: 8px 10px;"
            "font-weight: 600;"
            "color: #333;"
            "}"
        )
        self.mainArea.layout().addWidget(self.status_label, stretch=0)

        self.diagnostic_tabs = QTabWidget()
        self.diagnostic_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mainArea.layout().addWidget(self.diagnostic_tabs, stretch=4)

        self.summary = QTextBrowser()
        self.summary.setStyleSheet("background-color: #ffffff; padding: 8px;")
        self.diagnostic_tabs.addTab(self.summary, "Summary")

        self.fig_pred = Figure(figsize=(7.5, 5.2))
        self.canvas_pred = FigureCanvas(self.fig_pred)
        w1 = QWidget()
        l1 = QVBoxLayout()
        l1.setContentsMargins(2, 2, 2, 2)
        l1.addWidget(self.canvas_pred)
        w1.setLayout(l1)
        self.diagnostic_tabs.addTab(w1, "Predicted vs Real")

        self.fig_williams = Figure(figsize=(7.5, 5.2))
        self.canvas_williams = FigureCanvas(self.fig_williams)
        w2 = QWidget()
        l2 = QVBoxLayout()
        l2.setContentsMargins(2, 2, 2, 2)
        l2.addWidget(self.canvas_williams)
        w2.setLayout(l2)
        self.diagnostic_tabs.addTab(w2, "Williams AD")

        self.fig_perm = Figure(figsize=(7.5, 5.2))
        self.canvas_perm = FigureCanvas(self.fig_perm)
        w3 = QWidget()
        l3 = QVBoxLayout()
        l3.setContentsMargins(2, 2, 2, 2)
        l3.addWidget(self.canvas_perm)
        w3.setLayout(l3)
        self.diagnostic_tabs.addTab(w3, "Y-randomization")

        # Model Report tab — matches QSAR Regression layout
        self.report = QTextBrowser()
        self.report.setMinimumHeight(110)
        self.report.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.report.setStyleSheet(
            "background-color: #ffffff; padding: 12px; border: 1px solid #d7dee8; "
            "border-radius: 6px; font-family: Arial, sans-serif;"
        )
        self.diagnostic_tabs.addTab(self.report, "Model Report")

        # Features tab — same visual pattern as QSAR Regression
        self._features_tab = QWidget()
        self._features_layout = QVBoxLayout(self._features_tab)
        self._features_layout.setContentsMargins(6, 6, 6, 6)
        self._features_layout.setSpacing(6)
        self.diagnostic_tabs.addTab(self._features_tab, "Features")

        self._selected_tab = QWidget()
        self._selected_layout = QVBoxLayout(self._selected_tab)
        self._selected_layout.setContentsMargins(6, 6, 6, 6)
        self._selected_layout.setSpacing(6)
        self._selected_table = QTableWidget()
        self._selected_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._selected_table.setAlternatingRowColors(True)
        self._selected_table.horizontalHeader().setStretchLastSection(True)
        self._selected_layout.addWidget(self._selected_table)
        self.diagnostic_tabs.addTab(self._selected_tab, "Selected")

        self.flagged_label = QLabel("Flagged / selected compounds")
        self.flagged_label.setStyleSheet("font-weight: bold; margin-top: 6px;")
        self.mainArea.layout().addWidget(self.flagged_label, stretch=0)

        self.flagged_scroll = QScrollArea()
        self.flagged_scroll.setWidgetResizable(True)
        self.flagged_container = QWidget()
        self.flagged_layout = QVBoxLayout(self.flagged_container)
        self.flagged_layout.setContentsMargins(6, 6, 6, 6)
        self.flagged_layout.setSpacing(6)
        self.flagged_scroll.setWidget(self.flagged_container)
        self.flagged_scroll.setMinimumHeight(150)
        self.mainArea.layout().addWidget(self.flagged_scroll, stretch=2)
        self._show_flagged_placeholder("Run the model to see out-of-domain and high-residual compounds.")
        self._update_selected_table(None)

        self._update_method_ui()
        self._apply_visibility_settings()

    # ------------------------------------------------------------------

        # ------------- caching -------------
        # Cache expensive selection/fitting results keyed by (data signatures + settings).
        self._cache: Dict[Any, Any] = {}
        self._last_cache_key = None

    def _apply_visibility_settings(self):
        """Show/hide large diagnostics panel to free space for flagged compounds."""
        if hasattr(self, "diagnostic_tabs"):
            self.diagnostic_tabs.setVisible(bool(self.show_diagnostic_plots))

        if self.mainArea.layout() is not None and hasattr(self, "flagged_scroll"):
            plots_stretch = 5 if self.show_diagnostic_plots else 0
            gallery_stretch = 2
            if not self.show_diagnostic_plots:
                gallery_stretch += 5
            self.mainArea.layout().setStretchFactor(self.diagnostic_tabs, plots_stretch)
            self.mainArea.layout().setStretchFactor(self.flagged_scroll, gallery_stretch)
        self._update_flagged_label()

    def _update_status(self, html_or_text: str):
        if not hasattr(self, "status_label"):
            return
        text = str(html_or_text)
        # Status label supports rich text but should stay compact.
        self.status_label.setText(text)

    def _update_flagged_label(self):
        if not hasattr(self, "flagged_label"):
            return
        hidden = []
        if not bool(self.show_diagnostic_plots):
            hidden.append("diagnostic plots")
        if not bool(self.show_model_report):
            hidden.append("model report")
        if hidden:
            self.flagged_label.setText("Flagged / selected compounds — expanded view (" + ", ".join(hidden) + " hidden)")
        else:
            self.flagged_label.setText("Flagged / selected compounds")

    def _clear_flagged_layout(self):
        if not hasattr(self, "flagged_layout"):
            return
        while self.flagged_layout.count():
            item = self.flagged_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _show_flagged_placeholder(self, text: str):
        self._clear_flagged_layout()
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #666; padding: 6px;")
        self.flagged_layout.addWidget(label)
        self.flagged_layout.addStretch(1)

    def _reset_selection_state(self):
        self._selection_contexts = {}
        self._selection_selectors = {}
        self._update_selected_table(None)

    def _update_selected_table(self, selected_table: Optional[Table]):
        selected_tab_index = self.diagnostic_tabs.indexOf(self._selected_tab)
        if selected_table is None or len(selected_table) == 0:
            self._selected_table.clearContents()
            self._selected_table.setRowCount(0)
            self._selected_table.setColumnCount(1)
            self._selected_table.setHorizontalHeaderLabels(["No selected compounds"])
            if selected_tab_index >= 0:
                self.diagnostic_tabs.setTabText(selected_tab_index, "Selected")
            return

        columns, rows = _table_display_rows(selected_table)
        if not columns:
            self._selected_table.clearContents()
            self._selected_table.setRowCount(0)
            self._selected_table.setColumnCount(1)
            self._selected_table.setHorizontalHeaderLabels(["No selected compounds"])
            if selected_tab_index >= 0:
                self.diagnostic_tabs.setTabText(selected_tab_index, "Selected")
            return

        self._selected_table.clearContents()
        self._selected_table.setColumnCount(len(columns))
        self._selected_table.setHorizontalHeaderLabels(columns)
        self._selected_table.setRowCount(len(rows))
        for row_idx, values in enumerate(rows):
            for col_idx, value in enumerate(values):
                self._selected_table.setItem(row_idx, col_idx, QTableWidgetItem(value))
        self._selected_table.resizeColumnsToContents()
        if selected_tab_index >= 0:
            self.diagnostic_tabs.setTabText(selected_tab_index, f"Selected ({len(rows)})")

    def _combine_selection_tables(self, *tables: Optional[Table]) -> Optional[Table]:
        present = [table for table in tables if table is not None and len(table) > 0]
        if not present:
            return None
        if len(present) == 1:
            return present[0]
        try:
            return Table.concatenate(present, axis=0)
        except Exception:
            return Table.concatenate(present, axis=0, ignore_domains=True)

    def _meta_value_as_text(self, table: Table, row_idx: int, preferred: Sequence[str]) -> str:
        names = [var.name for var in table.domain.metas]
        for key in preferred:
            if key in names:
                j = names.index(key)
                val = table.metas[row_idx, j]
                if val is not None and str(val) != "":
                    return str(val)
        return f"row {row_idx + 1}"

    def _update_flagged_compounds(self, train_table_out: Optional[Table], test_table_out: Optional[Table]):
        """Show a compact list of out-of-domain or large-residual rows."""
        self._clear_flagged_layout()
        cards = []

        def collect(table: Optional[Table], label: str, max_items: int = 12):
            if table is None or len(table) == 0:
                return
            attr_names = [v.name for v in table.domain.attributes]
            ad_name = next((n for n in attr_names if n.endswith("_in_AD")), None)
            std_name = next((n for n in attr_names if n.endswith("_std_resid")), None)
            pred_name = next((n for n in attr_names if n.endswith("_y_pred")), None)
            y_name = next((n for n in attr_names if n.endswith("_y")), None)

            ad_col = table.get_column(ad_name) if ad_name else np.ones(len(table))
            std_col = table.get_column(std_name) if std_name else np.zeros(len(table))
            pred_col = table.get_column(pred_name) if pred_name else np.full(len(table), np.nan)
            y_col = table.get_column(y_name) if y_name else np.full(len(table), np.nan)

            mask = (ad_col < 0.5) | (np.abs(std_col) >= 2.5)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                # Show the strongest residuals anyway so the panel is useful.
                order = np.argsort(-np.abs(std_col))
                idx = order[: min(5, len(order))]
            for i in idx[:max_items]:
                title = self._meta_value_as_text(
                    table,
                    int(i),
                    ["Name", "name", "compound_id", "Compound ID", "SMILES", "smiles"],
                )
                cards.append(
                    (
                        label,
                        title,
                        float(y_col[i]) if np.isfinite(y_col[i]) else None,
                        float(pred_col[i]) if np.isfinite(pred_col[i]) else None,
                        float(std_col[i]) if np.isfinite(std_col[i]) else None,
                        bool(ad_col[i] >= 0.5),
                    )
                )

        collect(train_table_out, "train")
        collect(test_table_out, "test")

        if not cards:
            self._show_flagged_placeholder("No flagged compounds to show.")
            return

        for dataset_label, title, y_val, pred_val, std_val, in_ad in cards[:24]:
            card = QLabel()
            card.setWordWrap(True)
            ad_text = "inside AD" if in_ad else "outside AD"
            y_text = "NA" if y_val is None else f"{y_val:.4g}"
            pred_text = "NA" if pred_val is None else f"{pred_val:.4g}"
            std_text = "NA" if std_val is None else f"{std_val:.3g}"
            card.setText(
                f"<b>{dataset_label}</b> — {title}<br>"
                f"real: {y_text} &nbsp; predicted: {pred_text} &nbsp; "
                f"std. residual: {std_text} &nbsp; <b>{ad_text}</b>"
            )
            card.setStyleSheet(
                "padding: 8px; border: 1px solid #d0d0d0; border-radius: 6px; "
                "background: #ffffff;"
            )
            self.flagged_layout.addWidget(card)
        self.flagged_layout.addStretch(1)

    def _update_selection_gallery(self, payload) -> None:
        if payload.placeholder_text:
            self._show_flagged_placeholder(payload.placeholder_text)
            return

        self._clear_flagged_layout()
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
            self.flagged_layout.addWidget(card)

        if payload.more_count > 0:
            more_label = QLabel(f"+ {payload.more_count} more")
            more_label.setStyleSheet("color: #666; padding: 12px;")
            self.flagged_layout.addWidget(more_label)
        self.flagged_layout.addStretch(1)

    def _refresh_selector_modes(self):
        use_lasso = int(self.selection_tool) == 1
        for selectors in self._selection_selectors.values():
            selectors["rect"].set_active(not use_lasso)
            selectors["lasso"].set_active(use_lasso)

    def _install_plot_selection(self, *, plot_key: str, canvas, ax, x_values, y_values, table: Optional[Table]):
        if table is None or len(table) == 0:
            return
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        overlay = ax.scatter([], [], s=110, facecolors="none", edgecolors="#F59E0B", linewidths=2.2, zorder=6)
        self._selection_contexts[plot_key] = {
            "canvas": canvas,
            "x": x_arr,
            "y": y_arr,
            "table": table,
            "overlay": overlay,
        }
        rect = RectangleSelector(
            ax,
            lambda eclick, erelease: self._apply_plot_selection(plot_key, eclick, erelease),
            useblit=False,
            button=[1],
            minspanx=1e-9,
            minspany=1e-9,
            spancoords="data",
            interactive=False,
        )
        lasso = LassoSelector(ax, lambda verts: self._apply_lasso_selection(plot_key, verts))
        self._selection_selectors[plot_key] = {"rect": rect, "lasso": lasso}
        self._refresh_selector_modes()

    def _apply_plot_selection(self, plot_key: str, eclick, erelease):
        context = self._selection_contexts.get(plot_key)
        if context is None:
            return
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        x0, x1 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y0, y1 = sorted([float(eclick.ydata), float(erelease.ydata)])
        selected_idx = qsar_service.rectangle_selection_indices(context["x"], context["y"], x0, y0, x1, y1)
        self._publish_selection(plot_key, selected_idx)

    def _apply_lasso_selection(self, plot_key: str, vertices):
        context = self._selection_contexts.get(plot_key)
        if context is None or not vertices:
            return
        selected_idx = qsar_service.lasso_selection_indices(context["x"], context["y"], vertices)
        self._publish_selection(plot_key, selected_idx)

    def _clear_other_selection_overlays(self, active_plot_key: str):
        empty = np.empty((0, 2))
        for plot_key, context in self._selection_contexts.items():
            if plot_key == active_plot_key:
                continue
            context["overlay"].set_offsets(empty)
            context["canvas"].draw_idle()

    def _publish_selection(self, plot_key: str, selected_idx):
        context = self._selection_contexts.get(plot_key)
        if context is None:
            return
        self._clear_other_selection_overlays(plot_key)
        selected_idx_arr = np.asarray(selected_idx, dtype=int)
        if selected_idx_arr.size == 0:
            context["overlay"].set_offsets(np.empty((0, 2)))
        else:
            context["overlay"].set_offsets(np.column_stack([context["x"][selected_idx_arr], context["y"][selected_idx_arr]]))
        context["canvas"].draw_idle()
        payload = qsar_service.build_selection_publish_payload(
            model_name="MLR Model Selection",
            dataset_type=plot_key,
            table=context["table"],
            selected_idx=selected_idx_arr,
        )
        self.Outputs.selected_compounds.send(payload.selected_table)
        self._update_selection_gallery(payload.gallery)
        self._update_selected_table(payload.selected_table)
        self._update_status(payload.status_text)

    def _update_method_ui(self):
        # Guard against old int values saved before sendSelectedValue=True was set
        _METHOD_NAMES = ["forward", "backward", "montecarlo", "genetic"]
        if isinstance(self.method, int):
            self.method = _METHOD_NAMES[self.method] if 0 <= self.method < len(_METHOD_NAMES) else "forward"
        m = (self.method or "forward").lower()
        self.mc_box.setEnabled(m == "montecarlo")
        self.ga_box.setEnabled(m == "genetic")
        self._deferred_apply()

    def _on_selection_tool_changed(self):
        self._refresh_selector_modes()

    def _deferred_apply(self):
        # keep manual Apply available while allowing standalone/auto mode
        if bool(self.auto_run):
            self.apply()

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.data = data
        self._refresh_target_combo()
        self._deferred_apply()

    @Inputs.test_data
    def set_test_data(self, data: Optional[Table]):
        self.test_data = data
        self._deferred_apply()

    def _refresh_target_combo(self):
        self.y_combo.clear()
        if self.data is None or len(self.data) == 0:
            return
        candidates = mlr_service.continuous_candidates(self.data)
        if not candidates:
            return
        names = [v.name for v in candidates]
        self.y_combo.addItems(names)

        # pick default
        if self.y_name and self.y_name in names:
            self.y_combo.setCurrentIndex(names.index(self.y_name))
        else:
            # prefer continuous class var if present
            cv = self.data.domain.class_var
            if cv is not None and getattr(cv, "is_continuous", False):
                self.y_name = cv.name
                self.y_combo.setCurrentIndex(names.index(self.y_name))
            else:
                self.y_name = names[0]
                self.y_combo.setCurrentIndex(0)

    # ------------------------------------------------------------------

    def _make_cache_key(
        self,
        X_train_f: np.ndarray,
        y_train: np.ndarray,
        X_test_f: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        names: Sequence[str],
        cfg: SelectionConfig,
        ad_cfg: ADConfig,
    ) -> tuple:
        # Lightweight signatures to detect changes without hashing full arrays.
        def sig(X, y):
            if X is None:
                return None
            return (
                int(X.shape[0]),
                int(X.shape[1]),
                float(np.nanmean(X)),
                float(np.nanstd(X)),
                float(np.nanmean(y)) if y is not None else float('nan'),
                float(np.nanstd(y)) if y is not None else float('nan'),
            )

        return (
            sig(X_train_f, y_train),
            sig(X_test_f, y_test),
            tuple(names),
            self.y_name,
            float(self.test_size),
            int(self.random_state),
            float(self.var_threshold),
            float(self.corr_threshold),
            # selection + AD configs
            tuple(sorted(cfg.__dict__.items())),
            tuple(sorted(ad_cfg.__dict__.items())),
            bool(self.compute_external_metrics),
            int(self.perm_n),
        )

    def apply(self):
        # reset outputs
        self.Outputs.model.send(None)
        self.Outputs.train_results.send(None)
        self.Outputs.test_results.send(None)
        self.Outputs.predictions.send(None)
        self.Outputs.coefficients.send(None)
        self.Outputs.report_html.send(None)
        self.Outputs.selected_compounds.send(None)

        self.summary.setHtml("")
        self.report.setHtml("")
        self._clear_plots()
        self._reset_selection_state()
        self._show_flagged_placeholder("Run the model to see out-of-domain and high-residual compounds.")
        self._update_status("No model fitted yet.")
        self._latest_train_table_out = None
        self._latest_test_table_out = None

        if self.data is None or len(self.data) == 0:
            self._update_status("No input data.")
            return

        # resolve y var
        y_var = None
        for v in mlr_service.continuous_candidates(self.data):
            if v.name == self.y_name:
                y_var = v
                break
        if y_var is None:
            self.summary.setHtml("<b>Error:</b> No valid target selected.")
            self._update_status(self.summary.toPlainText() or "Error.")
            return

        # decide train/test tables
        if self.test_data is not None and len(self.test_data) > 0:
            train_table = self.data
            test_table = self.test_data
        else:
            idx = np.arange(len(self.data))
            try:
                train_idx, test_idx = train_test_split(
                    idx,
                    test_size=float(self.test_size),
                    random_state=int(self.random_state),
                    shuffle=True,
                )
            except Exception as e:
                self.summary.setHtml(f"<b>Error:</b> Cannot split data: {e}")
                self._update_status(self.summary.toPlainText() or "Error.")
                return
            train_table = self.data[train_idx]
            test_table = self.data[test_idx]

        # extract X/y from train
        try:
            X_train, y_train, x_vars = mlr_service.extract_xy(train_table, y_var)
        except Exception as e:
            self.summary.setHtml(f"<b>Error:</b> {e}")
            self._update_status(self.summary.toPlainText() or "Error.")
            return

        # extract X/y from test (y optional)
        X_test: Optional[np.ndarray]
        y_test: Optional[np.ndarray]
        try:
            X_test, y_test, _ = mlr_service.extract_xy(test_table, y_var)
        except Exception:
            X_test = mlr_service.extract_x_only(test_table, [v.name for v in x_vars])
            y_test = None

        # impute missing using training stats
        imputer = SimpleImputer(strategy="mean")
        X_train_imp = imputer.fit_transform(X_train).astype(float)
        X_test_imp = imputer.transform(X_test).astype(float) if X_test is not None else None

        # preprocessing: variance + correlation (fit on training)
        names0 = [v.name for v in x_vars]
        Xv, names_v, keep_var = filter_low_variance(
            X_train_imp, names0, threshold=float(self.var_threshold)
        )
        Xc, names_c, keep_corr = filter_high_correlation(
            Xv, names_v, threshold=float(self.corr_threshold)
        )

        # apply preprocessing to test
        Xt = None
        if X_test_imp is not None:
            Xt = X_test_imp[:, keep_var]
            Xt = Xt[:, keep_corr]

        # Migrate old int values from pre-sendSelectedValue sessions
        _METHOD_NAMES = ["forward", "backward", "montecarlo", "genetic"]
        _CRIT_NAMES   = ["cv_r2", "cv_rmse", "adj_r2", "aic", "bic", "train_r2"]
        method    = _METHOD_NAMES[self.method]    if isinstance(self.method, int)    else (self.method    or "forward")
        criterion = _CRIT_NAMES[self.criterion]   if isinstance(self.criterion, int) else (self.criterion or "cv_r2")

        # selection config
        cfg = SelectionConfig(
            method=method,
            criterion=criterion,
            cv_folds=int(self.cv_folds),
            random_state=int(self.random_state),
            max_features=int(self.max_features),
            min_features=int(self.min_features),
            mc_iterations=int(self.mc_iterations),
            ga_population=int(self.ga_population),
            ga_generations=int(self.ga_generations),
            ga_crossover=float(self.ga_crossover),
            ga_mutation=float(self.ga_mutation),
            ga_tournament=int(self.ga_tournament),
            ga_elite=int(self.ga_elite),
        )

        ad_cfg = ADConfig(
            use_williams=bool(self.ad_use_williams),
            use_knn=bool(self.ad_use_knn),
            use_mahalanobis=bool(self.ad_use_maha),
            combine_mode=str(self.ad_combine_mode),
            knn_k=int(self.ad_knn_k),
            knn_quantile=float(self.ad_knn_quantile),
            maha_alpha=float(self.ad_maha_alpha),
            maha_use_chi2=bool(self.ad_maha_use_chi2),
        )

        # caching: avoid re-running expensive selection if inputs+settings did not change
        cache_key = self._make_cache_key(Xc, y_train.astype(float), Xt, (None if y_test is None else y_test.astype(float)), names_c, cfg, ad_cfg)
        cached = self._cache.get(cache_key)
        if cached is not None:
            # restore outputs
            self.Outputs.train_results.send(cached["train_table_out"])
            self.Outputs.test_results.send(cached.get("test_table_out"))
            self.Outputs.predictions.send(cached.get("predictions_out"))
            self.Outputs.model.send(cached["bundle"])
            self.Outputs.coefficients.send(cached["coef_table_out"])
            self.Outputs.report_html.send(cached["report_html"])
            self.Outputs.selected_compounds.send(None)

            self.summary.setHtml(cached["summary_html"])
            self.report.setHtml(cached["report_html"])
            self._update_status(cached.get("status_text") or "MLR model ready.")
            self._latest_train_table_out = cached["train_table_out"]
            self._latest_test_table_out = cached.get("test_table_out")
            self._update_flagged_compounds(cached["train_table_out"], cached.get("test_table_out"))
            if "feat_names" in cached:
                self._render_features_tab(cached["feat_names"], cached["feat_coef_stats"], cached["feat_vifs"])
            self._apply_visibility_settings()

            # restore plots
            self._plot_pred_vs_real(
                cached["y_train"],
                cached["y_pred_train"],
                cached.get("y_test"),
                cached.get("y_pred_test"),
            )
            self._plot_williams(
                cached["lev_train"],
                cached["std_train"],
                cached["h_star"],
                cached.get("lev_test"),
                cached.get("std_test"),
            )
            self._plot_perm(cached.get("perm_info"))
            self._last_cache_key = cache_key
            return


        try:
            fit, sel = fit_mlr_with_selection(
                Xc, y_train.astype(float), names_c, cfg, ad_cfg=ad_cfg
            )
        except Exception as e:
            self.summary.setHtml(f"<b>Error during selection/fitting:</b> {e}")
            self._update_status(self.summary.toPlainText() or "Error.")
            return

        # predictions + AD (train)
        train_diag = predict_with_ad(fit, Xc, y_true=y_train.astype(float))
        y_pred_train = train_diag["y_pred"]
        lev_train = train_diag["leverage"]
        std_train = train_diag["std_resid"]
        in_ad_train = train_diag["in_ad"]

        train_metrics = fit.metrics_train
        cv_metrics = fit.cv_metrics

        # predictions + AD (test)
        test_metrics = None
        ext_metrics = None
        test_diag = None
        if Xt is not None:
            test_diag = predict_with_ad(
                fit, Xt, y_true=(None if y_test is None else y_test.astype(float))
            )
            if y_test is not None:
                test_metrics = regression_metrics(
                    y_test.astype(float), test_diag["y_pred"]
                )
                if bool(self.compute_external_metrics):
                    ext_metrics = external_validation_metrics(
                        y_train.astype(float),
                        y_test.astype(float),
                        test_diag["y_pred"].astype(float),
                    )

        # permutation test (optional)
        perm_info = None
        if int(self.perm_n) > 0:
            try:
                perm_info = permutation_test_cv_q2(
                    Xc,
                    y_train.astype(float),
                    fit.selected_idx,
                    n_permutations=int(self.perm_n),
                    cv=int(self.cv_folds),
                    random_state=int(self.random_state),
                )
            except Exception:
                perm_info = None

        # outputs: tables
        train_table_out = mlr_service.results_table(
            train_table,
            y_train,
            y_pred_train,
            lev_train,
            std_train,
            in_ad_train,
            prefix="train",
            in_ad_williams=(train_diag.get("in_ad_williams") if bool(self.ad_use_williams) else None),
            knn_dist=train_diag.get("knn_dist"),
            in_ad_knn=train_diag.get("in_ad_knn"),
            maha_d2=train_diag.get("maha_d2"),
            in_ad_maha=train_diag.get("in_ad_maha"),
        )
        self.Outputs.train_results.send(train_table_out)
        self._latest_train_table_out = train_table_out

        test_table_out = None
        if test_diag is not None:
            test_table_out = mlr_service.results_table(
                test_table,
                y_test,
                test_diag["y_pred"],
                test_diag["leverage"],
                test_diag["std_resid"],
                test_diag["in_ad"],
                prefix="test",
                in_ad_williams=(test_diag.get("in_ad_williams") if bool(self.ad_use_williams) else None),
                knn_dist=test_diag.get("knn_dist"),
                in_ad_knn=test_diag.get("in_ad_knn"),
                maha_d2=test_diag.get("maha_d2"),
                in_ad_maha=test_diag.get("in_ad_maha"),
            )
            self.Outputs.test_results.send(test_table_out)
        self._latest_test_table_out = test_table_out

        # output model bundle
        mlr_bundle = MLRModelBundle(
            y_name=y_var.name,
            x_names_after_preprocess=names_c,
            selected_names=fit.selected_names,
            variance_kept_idx=keep_var.astype(int),
            corr_kept_idx=keep_corr.astype(int),
            selected_idx_in_corr_space=fit.selected_idx.astype(int),
            imputer=imputer,
            model=fit.model,
            h_star=float(fit.h_star),
        )
        bundle = build_qsar_prediction_bundle(
            mlr_bundle,
            feature_names=list(names_c),
            target_label=y_var.name,
            recipe_kind="precomputed_table",
            model_name="Multiple Linear Regression",
            source_widget=self.name,
            training_rows=len(y_train),
            selected_feature_names=list(fit.selected_names),
        )
        self.Outputs.model.send(bundle)

        # UI: summary + plots
        summary_html = mlr_service.build_summary_html(
            y_var=y_var.name,
            n_train=len(y_train),
            n_test=(0 if test_diag is None else (len(y_test) if y_test is not None else len(test_diag["y_pred"]))),
            names_before=len(names0),
            names_after_pre=len(names_c),
            selected=fit.selected_names,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            ext_metrics=ext_metrics,
            cv_metrics=cv_metrics,
            h_star=float(fit.h_star),
            ad_cfg=fit.ad_cfg,
            knn_threshold=fit.knn_threshold,
            maha_threshold=fit.maha_threshold,
            coef_stats=fit.coef_stats,
            vifs=fit.vifs,
            perm_info=perm_info,
            method=self.method,
            criterion=self.criterion,
            cv_folds=self.cv_folds,
        )
        self.summary.setHtml(summary_html)
        self._plot_pred_vs_real(
            y_train,
            y_pred_train,
            y_test if (test_diag is not None and y_test is not None) else None,
            test_diag["y_pred"] if test_diag is not None else None,
        )
        self._plot_williams(
            lev_train,
            std_train,
            fit.h_star,
            test_diag["leverage"] if (test_diag is not None and test_diag["std_resid"] is not None) else None,
            test_diag["std_resid"] if (test_diag is not None and test_diag["std_resid"] is not None) else None,
        )
        self._plot_perm(perm_info)

        # coefficients output table
        coef_table_out = mlr_service.coefficients_table(fit.selected_names, fit.coef_stats, fit.vifs)
        self.Outputs.coefficients.send(coef_table_out)

        # predictions output (for convenience, mirrors Test Results if present)
        self.Outputs.predictions.send(test_table_out if test_diag is not None else None)
        self.Outputs.selected_compounds.send(None)

        # self-contained HTML report (summary + embedded plot images)
        report_html = mlr_service.build_report_html(summary_html, self.fig_pred, self.fig_williams, self.fig_perm)
        self.report.setHtml(report_html)
        self.Outputs.report_html.send(report_html)

        status_text = (
            f"MLR fitted: target <b>{y_var.name}</b>; "
            f"selected descriptors: <b>{len(fit.selected_names)}</b>; "
            f"train R²={train_metrics.get('r2', float('nan')):.3f}; "
            f"CV Q²={cv_metrics.get('q2', float('nan')):.3f}"
            + (f"; test R²={test_metrics.get('r2', float('nan')):.3f}" if test_metrics is not None else "")
        )
        self._update_status(status_text)
        self._render_features_tab(fit.selected_names, fit.coef_stats, fit.vifs)
        self._update_flagged_compounds(train_table_out, test_table_out)
        self._apply_visibility_settings()

        # store in cache
        self._cache[cache_key] = dict(
            train_table_out=train_table_out,
            test_table_out=(test_table_out if test_diag is not None else None),
            predictions_out=(test_table_out if test_diag is not None else None),
            bundle=bundle,
            coef_table_out=coef_table_out,
            summary_html=summary_html,
            report_html=report_html,
            y_train=y_train,
            y_pred_train=y_pred_train,
            y_test=(y_test if (test_diag is not None and y_test is not None) else None),
            y_pred_test=(test_diag["y_pred"] if test_diag is not None else None),
            lev_train=lev_train,
            std_train=std_train,
            lev_test=(test_diag["leverage"] if test_diag is not None else None),
            std_test=(test_diag["std_resid"] if test_diag is not None else None),
            h_star=float(fit.h_star),
            perm_info=perm_info,
            status_text=status_text,
            feat_names=fit.selected_names,
            feat_coef_stats=fit.coef_stats,
            feat_vifs=fit.vifs,
        )
        self._last_cache_key = cache_key

    def _make_subset_table(
        self,
        original: Table,
        X_subset: np.ndarray,
        y_subset: np.ndarray,
        x_vars: Sequence[ContinuousVariable],
        y_var: ContinuousVariable,
        X_subset_imp: np.ndarray,
        keep_var: np.ndarray,
        keep_corr: np.ndarray,
        y_pred: np.ndarray,
        leverage: np.ndarray,
        std_resid: np.ndarray,
        in_ad: np.ndarray,
        prefix: str,
    ) -> Table:
        # If subset is the full original table, we can just attach results.
        # Otherwise, make a minimal Table with the same metas (we skip exact row identity).
        base_table = original
        return mlr_service.results_table(base_table, y_subset, y_pred, leverage, std_resid, in_ad, prefix)

    def _make_test_table(
        self,
        base_table: Optional[Table],
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        x_vars: Sequence[ContinuousVariable],
        y_var: ContinuousVariable,
        y_pred: np.ndarray,
        leverage: np.ndarray,
        std_resid: Optional[np.ndarray],
        in_ad: np.ndarray,
        prefix: str,
    ) -> Table:
        if base_table is None:
            # create dummy table with no metas
            attrs = [ContinuousVariable(v.name) for v in x_vars]
            domain = Domain(attrs, (), ())
            base_table = Table.from_numpy(domain, X_test if X_test is not None else np.zeros((len(y_pred), len(attrs))))
        return mlr_service.results_table(base_table, y_test, y_pred, leverage, std_resid, in_ad, prefix)

    # ── Features tab (coefficients + statistics) ──────────────────────────

    def _render_features_tab(
        self,
        selected_names: Sequence[str],
        coef_stats: Dict[str, np.ndarray],
        vifs: np.ndarray,
    ) -> None:
        self._clear_features_tab()
        if not selected_names:
            self._features_layout.addWidget(QLabel("No descriptors selected."))
            return

        # coef_stats: intercept at index 0; descriptors start at 1
        betas = np.asarray(coef_stats["beta"])[1:]
        ses   = np.asarray(coef_stats["se"])[1:]
        ts    = np.asarray(coef_stats["t"])[1:]
        ps    = np.asarray(coef_stats["p"])[1:]
        vifs_a = np.asarray(vifs) if vifs is not None and len(vifs) == len(selected_names) else np.full(len(selected_names), np.nan)

        order   = np.argsort(np.abs(betas))[::-1]
        names_s = [selected_names[i] for i in order]
        betas_s = betas[order]
        ses_s   = ses[order]
        ts_s    = ts[order]
        ps_s    = ps[order]
        vifs_s  = vifs_a[order]

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        # ── Bar chart ──────────────────────────────────────────────────────
        top_n   = min(30, len(names_s))
        y_pos   = np.arange(top_n)
        colors  = ["#16a34a" if v >= 0 else "#dc2626" for v in betas_s[:top_n]]

        fig = Figure(figsize=(7, max(3, top_n * 0.30)))
        ax  = fig.add_subplot(111)
        ax.barh(y_pos, betas_s[:top_n], color=colors, edgecolor="none", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names_s[:top_n], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="#94a3b8", linewidth=0.8)
        ax.set_xlabel("Regression coefficient (β)", fontsize=9)
        ax.set_title(
            f"MLR — Coefficients  (top {top_n} of {len(names_s)} selected descriptors)",
            fontsize=9,
        )
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout(pad=1.0)

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chart_w = QWidget()
        chart_vl = QVBoxLayout(chart_w)
        chart_vl.setContentsMargins(0, 0, 0, 0)
        chart_vl.addWidget(canvas)
        legend = QLabel(
            "<span style='color:#16a34a'>■</span> positive β &nbsp;"
            "<span style='color:#dc2626'>■</span> negative β"
        )
        legend.setStyleSheet("font-size:10px; padding:2px 0;")
        chart_vl.addWidget(legend)
        splitter.addWidget(chart_w)

        # ── Stats table ────────────────────────────────────────────────────
        tbl_w  = QWidget()
        tbl_vl = QVBoxLayout(tbl_w)
        tbl_vl.setContentsMargins(0, 0, 0, 0)
        hdr = QLabel(f"All {len(names_s)} selected descriptors — sorted by |β|")
        hdr.setStyleSheet("font-size:10px; color:#6b7280; padding:2px 0;")
        tbl_vl.addWidget(hdr)

        tbl = QTableWidget(len(names_s), 7)
        tbl.setHorizontalHeaderLabels(["Rank", "Descriptor", "β", "SE", "t", "p-value", "VIF"])
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setAlternatingRowColors(True)
        tbl.setSortingEnabled(True)

        for i, (name, b, se, t, p, vif_val) in enumerate(
            zip(names_s, betas_s, ses_s, ts_s, ps_s, vifs_s)
        ):
            rank = QTableWidgetItem(); rank.setData(Qt.DisplayRole, i + 1)
            tbl.setItem(i, 0, rank)
            tbl.setItem(i, 1, QTableWidgetItem(name))

            b_item = QTableWidgetItem(); b_item.setData(Qt.DisplayRole, round(float(b), 5))
            b_item.setBackground(QColor("#dcfce7" if float(b) >= 0 else "#fee2e2"))
            tbl.setItem(i, 2, b_item)

            se_item = QTableWidgetItem(); se_item.setData(Qt.DisplayRole, round(float(se), 5))
            tbl.setItem(i, 3, se_item)

            t_item = QTableWidgetItem(); t_item.setData(Qt.DisplayRole, round(float(t), 4))
            tbl.setItem(i, 4, t_item)

            p_item = QTableWidgetItem(); p_item.setData(Qt.DisplayRole, round(float(p), 5))
            # p-value significance colouring
            if float(p) < 0.01:
                p_bg = "#dcfce7"   # green — highly significant
            elif float(p) < 0.05:
                p_bg = "#d1fae5"   # light green — significant
            elif float(p) < 0.10:
                p_bg = "#fef3c7"   # yellow — marginal
            else:
                p_bg = "#fee2e2"   # red — not significant
            p_item.setBackground(QColor(p_bg))
            tbl.setItem(i, 5, p_item)

            vif_item = QTableWidgetItem()
            if np.isfinite(vif_val):
                vif_item.setData(Qt.DisplayRole, round(float(vif_val), 3))
                vif_bg = "#dcfce7" if float(vif_val) < 5 else "#fef3c7" if float(vif_val) < 10 else "#fee2e2"
                vif_item.setBackground(QColor(vif_bg))
            else:
                vif_item.setText("—")
            tbl.setItem(i, 6, vif_item)

        tbl.resizeColumnToContents(0)
        tbl.resizeColumnToContents(1)
        tbl_vl.addWidget(tbl)
        splitter.addWidget(tbl_w)
        splitter.setSizes([420, 300])

        self._features_layout.addWidget(splitter)
        self.diagnostic_tabs.setTabText(
            self.diagnostic_tabs.indexOf(self._features_tab),
            f"Features ({len(names_s)})",
        )

    def _clear_features_tab(self) -> None:
        while self._features_layout.count():
            item = self._features_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.diagnostic_tabs.setTabText(
            self.diagnostic_tabs.indexOf(self._features_tab), "Features"
        )

    def _clear_plots(self):
        self._selection_contexts = {}
        self._selection_selectors = {}
        for fig in (self.fig_pred, self.fig_williams, self.fig_perm):
            fig.clear()
        self.canvas_pred.draw()
        self.canvas_williams.draw()
        self.canvas_perm.draw()

    def _plot_pred_vs_real(
        self,
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        y_test: Optional[np.ndarray],
        y_pred_test: Optional[np.ndarray],
    ):
        self.fig_pred.clear()
        ax = self.fig_pred.add_subplot(111)

        ax.scatter(y_train, y_pred_train, label="Train", alpha=0.75, edgecolors="k", linewidths=0.5)
        if y_test is not None and y_pred_test is not None:
            ax.scatter(y_test, y_pred_test, label="Test", alpha=0.75, edgecolors="k", linewidths=0.5)

        residual_arrays = [np.asarray(y_train, dtype=float) - np.asarray(y_pred_train, dtype=float)]
        if y_test is not None and y_pred_test is not None:
            residual_arrays.append(np.asarray(y_test, dtype=float) - np.asarray(y_pred_test, dtype=float))
        combined_residuals = np.concatenate([arr[np.isfinite(arr)] for arr in residual_arrays if np.size(arr)])
        levels = _residual_reference_levels(combined_residuals)

        arrays = [np.asarray(y_train, dtype=float), np.asarray(y_pred_train, dtype=float)]
        if y_test is not None:
            arrays.append(np.asarray(y_test, dtype=float))
        if y_pred_test is not None:
            arrays.append(np.asarray(y_pred_test, dtype=float))
        all_values = np.concatenate([a[np.isfinite(a)] for a in arrays if a is not None and np.size(a)])
        if all_values.size:
            mn, mx = float(np.min(all_values)), float(np.max(all_values))
        else:
            mn, mx = 0.0, 1.0
        span = mx - mn
        pad = max(span * 0.07, 0.1 if span == 0 else 0.0)
        mn -= pad
        mx += pad
        ax.plot([mn, mx], [mn, mx], color="#475569", linestyle="-", lw=1.6, label="Ideal")
        for offset, color, linestyle, linewidth, label in (
            (levels["plus_1std"], "#2563EB", "--", 1.2, "±1σ"),
            (levels["minus_1std"], "#2563EB", "--", 1.2, None),
            (levels["plus_2std"], "#EA580C", ":", 1.35, "±2σ"),
            (levels["minus_2std"], "#EA580C", ":", 1.35, None),
        ):
            ax.plot(
                [mn, mx],
                [mn - offset, mx - offset],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.95,
                label=label,
            )
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("Real")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Real")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        self.fig_pred.tight_layout(pad=1.1)

        x_values = np.asarray(y_train, dtype=float)
        y_values = np.asarray(y_pred_train, dtype=float)
        selection_table = self._latest_train_table_out
        if y_test is not None and y_pred_test is not None and self._latest_test_table_out is not None:
            x_values = np.concatenate([x_values, np.asarray(y_test, dtype=float)])
            y_values = np.concatenate([y_values, np.asarray(y_pred_test, dtype=float)])
            selection_table = self._combine_selection_tables(self._latest_train_table_out, self._latest_test_table_out)
        self._install_plot_selection(
            plot_key="predictions",
            canvas=self.canvas_pred,
            ax=ax,
            x_values=x_values,
            y_values=y_values,
            table=selection_table,
        )
        self.canvas_pred.draw()

    def _plot_williams(
        self,
        lev_train: np.ndarray,
        std_train: np.ndarray,
        h_star: float,
        lev_test: Optional[np.ndarray],
        std_test: Optional[np.ndarray],
    ):
        self.fig_williams.clear()
        ax = self.fig_williams.add_subplot(111)

        ax.scatter(lev_train, std_train, label="Train", alpha=0.75, edgecolors="k", linewidths=0.5)
        if lev_test is not None and std_test is not None:
            ax.scatter(lev_test, std_test, label="Test", alpha=0.75, edgecolors="k", linewidths=0.5)

        residual_arrays = [np.asarray(std_train, dtype=float)]
        if std_test is not None:
            residual_arrays.append(np.asarray(std_test, dtype=float))
        combined_std = np.concatenate([arr[np.isfinite(arr)] for arr in residual_arrays if np.size(arr)])
        levels = _residual_reference_levels(combined_std)

        ax.axvline(h_star, color="#475569", linestyle="--", lw=1.6, label=f"h* = {h_star:.3g}")
        ax.axhline(0.0, color="#475569", linestyle="-", lw=1.4, label="Zero residual")
        for value, color, linestyle, linewidth, label in (
            (levels["mean"], "#94A3B8", "--", 1.0, "Mean residual"),
            (levels["plus_1std"], "#2563EB", "--", 1.2, "±1σ"),
            (levels["minus_1std"], "#2563EB", "--", 1.2, None),
            (levels["plus_2std"], "#EA580C", ":", 1.35, "±2σ"),
            (levels["minus_2std"], "#EA580C", ":", 1.35, None),
        ):
            ax.axhline(value, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95, label=label)

        lev_arrays = [np.asarray(lev_train, dtype=float)]
        std_arrays = [np.asarray(std_train, dtype=float)]
        if lev_test is not None:
            lev_arrays.append(np.asarray(lev_test, dtype=float))
        if std_test is not None:
            std_arrays.append(np.asarray(std_test, dtype=float))
        lev_vals = np.concatenate([a[np.isfinite(a)] for a in lev_arrays if a is not None and np.size(a)])
        std_vals = np.concatenate([a[np.isfinite(a)] for a in std_arrays if a is not None and np.size(a)])

        x_max = max(float(np.max(lev_vals)) if lev_vals.size else 1.0, float(h_star))
        ax.set_xlim(0.0, x_max * 1.10 if x_max > 0 else 1.0)
        if std_vals.size:
            y_min = min(float(np.min(std_vals)), levels["minus_2std"])
            y_max = max(float(np.max(std_vals)), levels["plus_2std"])
        else:
            y_min, y_max = -2.0, 2.0
        y_span = y_max - y_min
        ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.08 * y_span)

        ax.set_xlabel("Leverage (h)")
        ax.set_ylabel("Standardized residual")
        ax.set_title("Williams plot / Applicability Domain")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        self.fig_williams.tight_layout(pad=1.1)

        x_values = np.asarray(lev_train, dtype=float)
        y_values = np.asarray(std_train, dtype=float)
        selection_table = self._latest_train_table_out
        if lev_test is not None and std_test is not None and self._latest_test_table_out is not None:
            x_values = np.concatenate([x_values, np.asarray(lev_test, dtype=float)])
            y_values = np.concatenate([y_values, np.asarray(std_test, dtype=float)])
            selection_table = self._combine_selection_tables(self._latest_train_table_out, self._latest_test_table_out)
        self._install_plot_selection(
            plot_key="williams",
            canvas=self.canvas_williams,
            ax=ax,
            x_values=x_values,
            y_values=y_values,
            table=selection_table,
        )
        self.canvas_williams.draw()

    def _plot_perm(self, perm_info: Optional[Dict[str, Any]]):
        self.fig_perm.clear()
        ax = self.fig_perm.add_subplot(111)

        if perm_info is None:
            ax.text(0.5, 0.5, "Permutation test disabled\n(set # permutations > 0)", ha="center", va="center")
            ax.set_axis_off()
            self.fig_perm.tight_layout(pad=1.1)
            self.canvas_perm.draw()
            return

        perm = perm_info["q2_perm"]
        obs = perm_info["q2_observed"]

        ax.hist(perm, bins=20, alpha=0.8)
        ax.axvline(obs, linestyle="--", lw=1.6, label=f"observed Q²={obs:.3f}")
        ax.set_xlabel("Permuted Q²")
        ax.set_ylabel("Count")
        ax.set_title("Y-randomization (CV Q²)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        self.fig_perm.tight_layout(pad=1.1)
        self.canvas_perm.draw()
