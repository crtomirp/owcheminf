import numpy as np
from AnyQt.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from AnyQt.QtCore import Qt as _Qt
from AnyQt.QtGui import QPixmap
from AnyQt.QtCore import QThread, pyqtSignal, QTimer
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
# Additional metrics for regression evaluation
from chem_inf_widgets.chemcore.services import qsar_regression_service as qsar_service
from chem_inf_widgets.chemcore.services.qsar_diagnostics_contract import (
    SELECTION_TOOL_OPTIONS,
    residual_reference_levels as _residual_reference_levels,
)
from chem_inf_widgets.chemcore.services.qsar_prediction_packager_service import (
    build_qsar_prediction_bundle,
)
from chem_inf_widgets.chemcore.services.qsar_target_contract import DEFAULT_QSAR_TARGET_COLUMN
from chem_inf_widgets.widgets import qsar_diagnostics_ui, qsar_features_ui

TORCH_AVAILABLE = qsar_service.TORCH_AVAILABLE


def _table_display_rows(table):
    if table is None or len(table) == 0:
        return [], []
    variables = list(table.domain.attributes) + list(table.domain.class_vars) + list(table.domain.metas)
    columns = [var.name for var in variables]
    rows = []
    for row in table:
        values = []
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


# ----------------------------------------------------------------------
class OWQSARRegression(OWWidget):
    name = "QSAR Regression"
    description = ("Build QSAR regression models with flexible settings. "
                   "Splits the data into training and test sets, supports an optional external set, "
                   "advanced hyperparameter tuning (Grid Search and Randomized Search), and provides diagnostic plots "
                   "with modern diagnostics, report tab, and compound selection preview.")
    icon = "icons/modeling/qsar_regression.png"
    priority = 142
    keywords = ["QSAR", "Regression", "SMILES"]

    class Inputs:
        data = Input("Data", Table)
        external_data = Input("External Data", Table)

    class Outputs:
        model = Output("Model", object, auto_summary=False)
        train_results = Output("Train Results", Table)
        test_results = Output("Test Results", Table)
        external_results = Output("External Results", Table)
        selected_compounds = Output("Selected Compounds", Table)
        descriptor_coefficients = Output("Descriptor Coefficients", Table)
        modeling_summary = Output("Modeling Summary", Table)
        applicability_domain = Output("Applicability Domain", Table)
        model_ranking = Output("Model Ranking", Table)

    want_main_area = True

    # Persistent settings
    selected_algorithm = Setting(0)
    normalization_method = Setting(0)
    imputation_method = Setting(1)
    cv_folds = Setting(5)
    test_size = Setting(0.3)
    tuning_method = Setting(0)
    n_iter = Setting(10)
    hyperparameters = Setting("")
    enable_feature_selection = Setting(False)
    num_features = Setting(10)
    max_model_features = Setting(1000)
    enable_applicability_domain = Setting(True)
    enable_auto_qsar = Setting(False)
    selection_tool = Setting(0)
    show_diagnostic_plots = Setting(True)
    show_model_report = Setting(True)
    auto_run = Setting(True)
    # Updated list of available algorithms:
    algorithms = qsar_service.available_algorithms()
    normalization_options = [
        "None",
        "Standard Scaler",
        "MinMax Scaler"
    ]
    imputation_options = [
        "None",
        "Mean",
        "Median",
        "Most Frequent"
    ]
    tuning_options = [
        "None",
        "Grid Search",
        "Randomized Search"
    ]
    selection_tool_options = list(SELECTION_TOOL_OPTIONS)

    def __init__(self):
        super().__init__()
        self.data = None
        self.external_data = None
        self.model = None
        self.worker = None
        self._pending_commit = False
        self.last_train_fig = None
        self.last_test_fig = None
        self.last_ext_fig = None
        self.last_model_name = ""
        self._diagnostic_selectors = {}
        self._diagnostic_contexts = {}
        self._latest_result = None

        # Keep the widget close to native Orange layout.
        # Do not force QSplitter sizes: it creates large grey dead zones on macOS/Qt.
        if self.controlArea.layout() is not None:
            self.controlArea.layout().setSpacing(6)
        if self.mainArea.layout() is not None:
            self.mainArea.layout().setSpacing(6)

        # --- Control Panel Setup ---
        settings_box = gui.widgetBox(self.controlArea, "Model Settings")
        self.algorithm_combo = gui.comboBox(settings_box, self, "selected_algorithm",
                                            label="Algorithm:",
                                            items=[name for name, _ in self.algorithms],
                                            callback=self.settings_changed)
        gui.comboBox(settings_box, self, "normalization_method",
                     label="Normalization:",
                     items=self.normalization_options,
                     callback=self.settings_changed)
        gui.comboBox(settings_box, self, "imputation_method",
                     label="Imputation:",
                     items=self.imputation_options,
                     callback=self.settings_changed)
        gui.spin(settings_box, self, "cv_folds", minv=2, maxv=20, step=1,
                 label="CV Folds:", callback=self.settings_changed)
        gui.doubleSpin(settings_box, self, "test_size", minv=0.1, maxv=0.9, step=0.05,
                       label="Test Set Fraction:", callback=self.settings_changed)
        gui.comboBox(settings_box, self, "tuning_method",
                     label="Hyperparameter Tuning:",
                     items=self.tuning_options,
                     callback=self.settings_changed)
        gui.spin(settings_box, self, "n_iter", minv=5, maxv=100, step=5,
                 label="Randomized Search Iterations:", callback=self.settings_changed)
        gui.lineEdit(settings_box, self, "hyperparameters",
                     label="Hyperparameters (JSON):", callback=self.settings_changed)
        gui.checkBox(settings_box, self, "enable_feature_selection",
                     "Enable Descriptor Subset Selection", callback=self.settings_changed)
        gui.spin(settings_box, self, "num_features", minv=1, maxv=100, step=1,
                 label="Number of Features:", callback=self.settings_changed)
        gui.spin(settings_box, self, "max_model_features", minv=0, maxv=10000, step=100,
                 label="QSAR feature cap (0=off):", callback=self.settings_changed)
        gui.checkBox(settings_box, self, "enable_applicability_domain",
                     "Enable Applicability Domain", callback=self.settings_changed)
        self.auto_qsar_checkbox = gui.checkBox(settings_box, self, "enable_auto_qsar",
                                               "Auto QSAR model selection", callback=self.settings_changed)
        self.auto_qsar_hint = QLabel()
        self.auto_qsar_hint.setWordWrap(True)
        self.auto_qsar_hint.setStyleSheet("font-size: 11px; color: #64748b; margin-top: 2px;")
        settings_box.layout().addWidget(self.auto_qsar_hint)
        self.selection_tool_combo = QComboBox()
        self.selection_tool_combo.addItems(self.selection_tool_options)
        self.selection_tool_combo.setCurrentIndex(int(self.selection_tool))
        self.selection_tool_combo.currentIndexChanged.connect(self._on_selection_tool_changed)
        settings_box.layout().addWidget(QLabel("Selection Tool:"))
        settings_box.layout().addWidget(self.selection_tool_combo)

        view_box = gui.widgetBox(self.controlArea, "View")
        gui.checkBox(
            view_box,
            self,
            "show_diagnostic_plots",
            "Show diagnostic plots",
            callback=self._apply_visibility_settings,
        )
        gui.checkBox(
            view_box,
            self,
            "show_model_report",
            "Show model report",
            callback=self._apply_visibility_settings,
        )
        gui.checkBox(
            view_box,
            self,
            "auto_run",
            "Auto-run",
            callback=self.settings_changed,
        )

        gui.button(settings_box, self, "Commit", callback=self.commit)
        gui.button(settings_box, self, "Export PDF", callback=self.export_pdf)

        startup_message = "No model trained yet."
        if not TORCH_AVAILABLE:
            startup_message += " Deep Learning Regression is hidden because 'torch' is not installed."
        # Status is shown in the main status banner; keep this QLabel for existing code paths.
        self.info_label = QLabel(startup_message)
        self.info_label.hide()

        # --- Main status banner ---
        self.status_banner = QWidget()
        self.status_banner_layout = QHBoxLayout(self.status_banner)
        self.status_banner_layout.setContentsMargins(0, 0, 0, 0)
        self.status_banner_layout.setSpacing(8)

        self.main_status_label = QLabel(startup_message)
        self.main_status_label.setWordWrap(True)
        self.main_status_label.setStyleSheet(
            "QLabel {"
            "background: #f6f8fb;"
            "border: 1px solid #d7dee8;"
            "border-radius: 6px;"
            "padding: 8px 10px;"
            "font-weight: 600;"
            "color: #333;"
            "}"
        )
        self.mode_chip_label = QLabel()
        self.mode_chip_label.setAlignment(_Qt.AlignCenter)
        self.mode_chip_label.setMinimumWidth(118)

        self.status_banner_layout.addWidget(self.main_status_label, stretch=1)
        self.status_banner_layout.addWidget(self.mode_chip_label, stretch=0, alignment=_Qt.AlignTop)
        self.mainArea.layout().addWidget(self.status_banner, stretch=0)

        # --- Diagnostic Plots and report tabs ---
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mainArea.layout().addWidget(self.tabs, stretch=4)
        self.train_tab = QWidget()
        self.test_tab = QWidget()
        self.ext_tab = QWidget()
        self.report_tab = QWidget()
        self.features_tab  = QWidget()
        self.selected_tab = QWidget()
        self.train_layout    = QVBoxLayout(self.train_tab)
        self.test_layout     = QVBoxLayout(self.test_tab)
        self.ext_layout      = QVBoxLayout(self.ext_tab)
        self.report_layout   = QVBoxLayout(self.report_tab)
        self.features_layout = QVBoxLayout(self.features_tab)
        self.selected_layout = QVBoxLayout(self.selected_tab)
        for layout in (self.train_layout, self.test_layout, self.ext_layout,
                       self.report_layout, self.features_layout, self.selected_layout):
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(6)

        self.tabs.addTab(self.train_tab,   "Training")
        self.tabs.addTab(self.test_tab,    "Test")
        self.tabs.addTab(self.ext_tab,     "External")
        self.tabs.addTab(self.selected_tab, "Selected")
        self.tabs.addTab(self.report_tab,  "Model Report")
        self.tabs.addTab(self.features_tab, "Features")

        # --- HTML5 Report Widget as a tab, not as a permanently visible lower panel ---
        self.report_browser = QTextBrowser()
        self.report_browser.setMinimumHeight(110)
        self.report_browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.report_browser.setStyleSheet(
            "background-color: #ffffff; padding: 12px; border: 1px solid #d7dee8; "
            "border-radius: 6px; font-family: Arial, sans-serif;"
        )
        self.report_layout.addWidget(self.report_browser)

        self.selected_table = QTableWidget()
        self.selected_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.selected_table.setAlternatingRowColors(True)
        self.selected_table.horizontalHeader().setStretchLastSection(True)
        self.selected_layout.addWidget(self.selected_table)

        self.selection_gallery_label = QLabel("Selected compounds")
        self.selection_gallery_label.setStyleSheet("font-weight: bold; margin-top: 6px; font-size: 13px;")
        self.mainArea.layout().addWidget(self.selection_gallery_label, stretch=0)

        self.selection_gallery_scroll = QScrollArea()
        self.selection_gallery_scroll.setWidgetResizable(True)
        self.selection_gallery_container = QWidget()
        self.selection_gallery_layout = QHBoxLayout(self.selection_gallery_container)
        self.selection_gallery_layout.setContentsMargins(6, 6, 6, 6)
        self.selection_gallery_layout.setSpacing(10)
        self.selection_gallery_scroll.setWidget(self.selection_gallery_container)
        self.selection_gallery_scroll.setMinimumHeight(170)
        self.selection_gallery_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.selection_gallery_scroll.setStyleSheet(
            "QScrollArea { background: #ffffff; border: 1px solid #d7dee8; border-radius: 6px; }"
        )
        self.mainArea.layout().addWidget(self.selection_gallery_scroll, stretch=2)
        self._show_selection_gallery_placeholder("Select points on a diagnostic plot to preview compounds.")
        self._update_auto_qsar_state()
        self._apply_visibility_settings()

    @Inputs.data
    def set_data(self, dataset):
        self.data = dataset
        self._maybe_autorun()

    @Inputs.external_data
    def set_external_data(self, dataset):
        self.external_data = dataset
        self._maybe_autorun()

    def _set_status(self, text):
        if hasattr(self, "info_label"):
            self.info_label.setText(text)
        if hasattr(self, "main_status_label"):
            self.main_status_label.setText(text)

    def _tab_index(self, widget):
        return self.tabs.indexOf(widget) if hasattr(self, "tabs") else -1

    def _ensure_tab(self, widget, title, position):
        if self._tab_index(widget) < 0:
            self.tabs.insertTab(min(position, self.tabs.count()), widget, title)

    def _remove_tab(self, widget):
        idx = self._tab_index(widget)
        if idx >= 0:
            self.tabs.removeTab(idx)

    def _apply_visibility_settings(self):
        """Show/hide large panels so the selected-compound gallery can use the space."""
        if not hasattr(self, "tabs"):
            return

        diagnostic_tabs = (
            (self.train_tab, "Training", 0),
            (self.test_tab, "Test", 1),
            (self.ext_tab, "External", 2),
        )
        for widget, title, position in diagnostic_tabs:
            if bool(self.show_diagnostic_plots):
                self._ensure_tab(widget, title, position)
            else:
                self._remove_tab(widget)

        if bool(self.show_model_report):
            self._ensure_tab(self.report_tab, "Model Report", 4)
        else:
            self._remove_tab(self.report_tab)

        self.tabs.setVisible(self.tabs.count() > 0)

        if self.mainArea.layout() is not None and hasattr(self, "selection_gallery_scroll"):
            tabs_stretch = 5 if self.tabs.isVisible() else 0
            gallery_stretch = 2
            if not self.show_diagnostic_plots:
                gallery_stretch += 3
            if not self.show_model_report:
                gallery_stretch += 1
            if not self.tabs.isVisible():
                gallery_stretch += 3
            self.mainArea.layout().setStretchFactor(self.status_banner, 0)
            self.mainArea.layout().setStretchFactor(self.tabs, tabs_stretch)
            self.mainArea.layout().setStretchFactor(self.selection_gallery_label, 0)
            self.mainArea.layout().setStretchFactor(self.selection_gallery_scroll, gallery_stretch)

        self._update_view_hint()

    def _update_view_hint(self):
        if not hasattr(self, "selection_gallery_label"):
            return
        hidden = []
        if not bool(self.show_diagnostic_plots):
            hidden.append("diagnostic plots")
        if not bool(self.show_model_report):
            hidden.append("model report")
        if hidden:
            self.selection_gallery_label.setText(
                "Selected compounds — expanded view (" + ", ".join(hidden) + " hidden)"
            )
        else:
            self.selection_gallery_label.setText("Selected compounds")

    def settings_changed(self):
        self._update_auto_qsar_state()
        self._maybe_autorun()

    def _update_auto_qsar_state(self):
        auto_enabled = bool(self.enable_auto_qsar)
        if hasattr(self, "algorithm_combo"):
            self.algorithm_combo.setEnabled(not auto_enabled)
        if hasattr(self, "auto_qsar_hint"):
            if auto_enabled:
                self.auto_qsar_hint.setText(
                    "Auto QSAR is enabled. Manual algorithm choice is ignored and the best candidate model is selected automatically."
                )
            else:
                self.auto_qsar_hint.setText("Manual algorithm choice is active.")
        if hasattr(self, "mode_chip_label"):
            if auto_enabled:
                self.mode_chip_label.setText("Auto QSAR mode")
                self.mode_chip_label.setStyleSheet(
                    "QLabel {"
                    "background: #eff6ff;"
                    "color: #1d4ed8;"
                    "border: 1px solid #bfdbfe;"
                    "border-radius: 12px;"
                    "padding: 6px 10px;"
                    "font-weight: 700;"
                    "}"
                )
            else:
                self.mode_chip_label.setText("Manual mode")
                self.mode_chip_label.setStyleSheet(
                    "QLabel {"
                    "background: #ecfdf5;"
                    "color: #047857;"
                    "border: 1px solid #a7f3d0;"
                    "border-radius: 12px;"
                    "padding: 6px 10px;"
                    "font-weight: 700;"
                    "}"
                )

    def _maybe_autorun(self):
        if bool(self.auto_run) and self.data is not None:
            self.commit()

    def _on_selection_tool_changed(self, index):
        self.selection_tool = int(index)
        self._refresh_selector_modes()

    def commit(self):
        if self.data is None:
            self._latest_result = None
            self._set_status("No main data provided.")
            self.Outputs.selected_compounds.send(None)
            self.Outputs.descriptor_coefficients.send(None)
            self.Outputs.modeling_summary.send(None)
            self.Outputs.applicability_domain.send(None)
            self.Outputs.model_ranking.send(None)
            self._update_selected_table(None)
            self._show_selection_gallery_placeholder("No compounds selected.")
            return

        if self.selected_algorithm < 0 or self.selected_algorithm >= len(self.algorithms):
            self.selected_algorithm = 0

        if self.worker is not None and self.worker.isRunning():
            self._pending_commit = True
            self.worker.requestInterruption()
            self._set_status("Cancelling previous calculation before restarting…")
            return
        self._reset_diagnostic_views()
        self._latest_result = None
        self._pending_commit = False
        self._update_selected_table(None)

        requested_model_name = self.algorithms[self.selected_algorithm][0]
        self.last_model_name = "Auto QSAR" if self.enable_auto_qsar else requested_model_name
        waiting_label = self.last_model_name
        self._set_status(qsar_service.build_waiting_status_text(waiting_label))
        self.report_browser.setHtml(qsar_service.build_waiting_report_html())

        config = qsar_service.build_run_config(
            selected_algorithm=self.selected_algorithm,
            normalization_method=self.normalization_method,
            imputation_method=self.imputation_method,
            cv_folds=self.cv_folds,
            test_size=self.test_size,
            tuning_method=self.tuning_method,
            n_iter=self.n_iter,
            hyperparameters=self.hyperparameters,
            enable_feature_selection=self.enable_feature_selection,
            num_features=self.num_features,
            algorithms=self.algorithms,
            max_model_features=self.max_model_features,
            enable_applicability_domain=self.enable_applicability_domain,
            enable_auto_qsar=self.enable_auto_qsar,
        )
        self.worker = QSARWorker(self.data, self.external_data, config)
        worker = self.worker
        worker.finished_signal.connect(lambda result, worker=worker: self._on_worker_results(worker, result))
        worker.error_signal.connect(lambda error_msg, worker=worker: self._on_worker_error(worker, error_msg))
        worker.cancelled_signal.connect(lambda worker=worker: self._on_worker_cancelled(worker))
        worker.start()

    def _restart_if_pending(self):
        if self._pending_commit:
            self._pending_commit = False
            QTimer.singleShot(0, self.commit)

    def _on_worker_results(self, worker, result):
        if worker is not self.worker:
            return
        self.worker = None
        self.handle_results(result)
        self._restart_if_pending()

    def _on_worker_error(self, worker, error_msg):
        if worker is not self.worker:
            return
        self.worker = None
        self.handle_error(error_msg)
        self._restart_if_pending()

    def _on_worker_cancelled(self, worker):
        if worker is not self.worker:
            return
        self.worker = None
        self._set_status(qsar_service.build_cancelled_status_text())
        self._restart_if_pending()

    def handle_results(self, result):
        self._latest_result = result
        self.model = result["model"]
        self.last_model_name = result.get("model_name", self.last_model_name)
        self._send_primary_outputs(result)
        self._show_selection_gallery_placeholder("Select points on a diagnostic plot to preview compounds.")
        self._set_status(qsar_service.build_completed_status_text(self.last_model_name, result["performance_text"]))

        self._render_diagnostics_from_result(result)
        self._render_features_tab(result)
        self.update_report_browser(result)
        self._apply_visibility_settings()

    def handle_error(self, error_msg):
        self._latest_result = None
        self._set_status(qsar_service.build_error_status_text(error_msg))

    def onDeleteWidget(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(2000)
        super().onDeleteWidget()

    def update_diagnostics(self, dataset_type, X, y, pipeline, is_classification=False, result_table=None):
        diagnostic = qsar_service.prepare_diagnostic_plot_data(
            X,
            y,
            pipeline,
            is_classification=is_classification,
        )
        plot_spec = qsar_service.build_diagnostic_plot_spec(diagnostic)
        preds = np.asarray(diagnostic.preds, dtype=float)
        actuals = np.asarray(diagnostic.actuals, dtype=float)
        residuals = np.asarray(diagnostic.residuals, dtype=float)
        levels = _residual_reference_levels(residuals)
        fig = Figure(figsize=(11.5, 5.2))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        for series in plot_spec.left_series:
            ax1.scatter(series.x, series.y, alpha=0.75, c=series.color, edgecolors="k", linewidths=0.5, label=series.label)

        # Use data-driven limits instead of forcing the axes to start at zero.
        finite_left = np.isfinite(preds) & np.isfinite(actuals)
        if np.any(finite_left):
            left_min = float(min(np.min(preds[finite_left]), np.min(actuals[finite_left])))
            left_max = float(max(np.max(preds[finite_left]), np.max(actuals[finite_left])))
        else:
            left_min, left_max = 0.0, 1.0
        span = left_max - left_min
        pad = max(span * 0.07, 0.1 if span == 0 else 0.0)
        diag_min = left_min - pad
        diag_max = left_max + pad
        ax1.plot([diag_min, diag_max], [diag_min, diag_max], color="#475569", linestyle="-", lw=1.6, label="Ideal")
        for offset, color, linestyle, linewidth, label in (
            (levels["plus_1std"], "#2563EB", "--", 1.2, "±1σ"),
            (levels["minus_1std"], "#2563EB", "--", 1.2, None),
            (levels["plus_2std"], "#EA580C", ":", 1.35, "±2σ"),
            (levels["minus_2std"], "#EA580C", ":", 1.35, None),
        ):
            ax1.plot(
                [diag_min, diag_max],
                [diag_min + offset, diag_max + offset],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.95,
                label=label,
            )
        ax1.set_xlim(diag_min, diag_max)
        ax1.set_ylim(diag_min, diag_max)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_title(plot_spec.left_title)
        ax1.set_xlabel(plot_spec.left_xlabel)
        ax1.set_ylabel(plot_spec.left_ylabel)
        ax1.grid(alpha=0.3)
        if plot_spec.show_legends:
            ax1.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8)

        for series in plot_spec.right_series:
            ax2.scatter(series.x, series.y, alpha=0.75, c=series.color, edgecolors="k", linewidths=0.5, label=series.label)
        ax2.axhline(0, color="#475569", linestyle="-", lw=1.4, label="Zero residual")
        for value, color, linestyle, linewidth, label in (
            (levels["mean"], "#94A3B8", "--", 1.0, "Mean residual"),
            (levels["plus_1std"], "#2563EB", "--", 1.2, "±1σ"),
            (levels["minus_1std"], "#2563EB", "--", 1.2, None),
            (levels["plus_2std"], "#EA580C", ":", 1.35, "±2σ"),
            (levels["minus_2std"], "#EA580C", ":", 1.35, None),
        ):
            ax2.axhline(value, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95, label=label)
        ax2.set_title(plot_spec.right_title)
        ax2.set_xlabel(plot_spec.right_xlabel)
        ax2.set_ylabel(plot_spec.right_ylabel)
        if not diagnostic.is_classification:
            finite_right_x = np.isfinite(preds)
            if np.any(finite_right_x):
                x_min = float(np.min(preds[finite_right_x]))
                x_max = float(np.max(preds[finite_right_x]))
            else:
                x_min, x_max = 0.0, 1.0
            x_span = x_max - x_min
            x_pad = max(x_span * 0.07, 0.1 if x_span == 0 else 0.0)
            ax2.set_xlim(x_min - x_pad, x_max + x_pad)

            finite_right_y = np.isfinite(residuals)
            if np.any(finite_right_y):
                y_min = float(np.min(residuals[finite_right_y]))
                y_max = float(np.max(residuals[finite_right_y]))
            else:
                y_min, y_max = -1.0, 1.0
            y_span = y_max - y_min
            y_pad = max(
                y_span * 0.10,
                max(abs(levels["plus_2std"]), abs(levels["minus_2std"]), 0.1) * 0.15,
            )
            ax2.set_ylim(min(y_min, levels["minus_2std"]) - y_pad, max(y_max, levels["plus_2std"]) + y_pad)
            ax2.grid(alpha=0.3)
        if plot_spec.show_legends:
            ax2.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8)

        sel1 = ax1.scatter([], [], s=90, facecolors="none", edgecolors="#ff8c00", linewidths=2.0, zorder=5)
        sel2 = ax2.scatter([], [], s=90, facecolors="none", edgecolors="#ff8c00", linewidths=2.0, zorder=5)

        fig.tight_layout(pad=1.1, w_pad=1.2)
        canvas = FigureCanvas(fig)
        self._attach_diagnostic_canvas(dataset_type, canvas, fig)

        self._install_point_selection(
            dataset_type=dataset_type,
            canvas=canvas,
            fig=fig,
            ax_left=ax1,
            ax_right=ax2,
            preds=preds,
            y=actuals,
            residuals=residuals,
            result_table=result_table,
            overlay_left=sel1,
            overlay_right=sel2,
        )

    def _attach_diagnostic_canvas(self, dataset_type, canvas, fig):
        targets = {
            "train": (self.train_layout, "last_train_fig"),
            "test": (self.test_layout, "last_test_fig"),
            "external": (self.ext_layout, "last_ext_fig"),
        }
        if dataset_type not in targets:
            return

        layout, fig_attr = targets[dataset_type]
        self.clear_layout(layout)
        layout.addWidget(canvas)
        setattr(self, fig_attr, fig)

    def _install_point_selection(
        self,
        *,
        dataset_type,
        canvas,
        fig,
        ax_left,
        ax_right,
        preds,
        y,
        residuals,
        result_table,
        overlay_left,
        overlay_right,
    ):
        self._diagnostic_contexts[dataset_type] = qsar_diagnostics_ui.build_diagnostic_selection_context(
            canvas=canvas,
            figure=fig,
            preds=preds,
            y=y,
            residuals=residuals,
            table=result_table,
            overlay_left=overlay_left,
            overlay_right=overlay_right,
        )

        self._diagnostic_selectors[dataset_type] = qsar_diagnostics_ui.create_diagnostic_selectors(
            ax_left=ax_left,
            ax_right=ax_right,
            on_rect_left=lambda eclick, erelease: self._apply_plot_selection(
                dataset_type, eclick, erelease, left_plot=True
            ),
            on_rect_right=lambda eclick, erelease: self._apply_plot_selection(
                dataset_type, eclick, erelease, left_plot=False
            ),
            on_lasso_left=lambda verts: self._apply_lasso_selection(dataset_type, verts, left_plot=True),
            on_lasso_right=lambda verts: self._apply_lasso_selection(dataset_type, verts, left_plot=False),
        )
        self._refresh_selector_modes()

    def _refresh_selector_modes(self):
        use_lasso = int(self.selection_tool) == 1
        for selectors in self._diagnostic_selectors.values():
            qsar_diagnostics_ui.set_selector_mode(selectors, use_lasso=use_lasso)

    def _apply_plot_selection(self, dataset_type, eclick, erelease, *, left_plot):
        context = self._diagnostic_contexts.get(dataset_type)
        if context is None or context.table is None:
            return

        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return

        x0, x1 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y0, y1 = sorted([float(eclick.ydata), float(erelease.ydata)])

        preds, ys = qsar_diagnostics_ui.selection_plot_values(context, left_plot=left_plot)
        selected_idx = qsar_service.rectangle_selection_indices(preds, ys, x0, y0, x1, y1)
        self._publish_selection(dataset_type, selected_idx)

    def _apply_lasso_selection(self, dataset_type, vertices, *, left_plot):
        context = self._diagnostic_contexts.get(dataset_type)
        if context is None or context.table is None or not vertices:
            return

        preds, ys = qsar_diagnostics_ui.selection_plot_values(context, left_plot=left_plot)
        selected_idx = qsar_service.lasso_selection_indices(preds, ys, vertices)
        self._publish_selection(dataset_type, selected_idx)

    def _publish_selection(self, dataset_type, selected_idx):
        self._clear_other_selection_overlays(dataset_type)
        self._update_selection_overlays(dataset_type, selected_idx)
        context = self._diagnostic_contexts[dataset_type]
        payload = qsar_service.build_selection_publish_payload(
            model_name=self.last_model_name,
            dataset_type=dataset_type,
            table=context.table,
            selected_idx=selected_idx,
        )
        self.Outputs.selected_compounds.send(payload.selected_table)
        self._update_selection_gallery(payload.gallery)
        self._update_selected_table(payload.selected_table)
        self._set_status(payload.status_text)

    def _update_selection_overlays(self, dataset_type, selected_idx):
        context = self._diagnostic_contexts.get(dataset_type)
        if context is None:
            return
        qsar_diagnostics_ui.update_selection_overlays(context, selected_idx)

    def _clear_other_selection_overlays(self, active_dataset_type):
        for dataset_type, context in self._diagnostic_contexts.items():
            if dataset_type == active_dataset_type:
                continue
            qsar_diagnostics_ui.clear_selection_overlays(context)

    def _clear_selection_gallery(self):
        while self.selection_gallery_layout.count():
            item = self.selection_gallery_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _show_selection_gallery_placeholder(self, text):
        self._clear_selection_gallery()
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #666; padding: 6px;")
        self.selection_gallery_layout.addWidget(label)
        self.selection_gallery_layout.addStretch(1)

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
            txt_label.setStyleSheet("font-size: 11px; border: none; color: #333;")
            card_layout.addWidget(img_label)
            card_layout.addWidget(txt_label)
            self.selection_gallery_layout.addWidget(card)

        if payload.more_count > 0:
            more_label = QLabel(f"+ {payload.more_count} more")
            more_label.setStyleSheet("color: #666; padding: 12px;")
            self.selection_gallery_layout.addWidget(more_label)
        self.selection_gallery_layout.addStretch(1)

    def _update_selected_table(self, selected_table):
        selected_tab_index = self.tabs.indexOf(self.selected_tab)
        if selected_table is None or len(selected_table) == 0:
            self.selected_table.clearContents()
            self.selected_table.setRowCount(0)
            self.selected_table.setColumnCount(1)
            self.selected_table.setHorizontalHeaderLabels(["No selected compounds"])
            if selected_tab_index >= 0:
                self.tabs.setTabText(selected_tab_index, "Selected")
            return

        columns, rows = _table_display_rows(selected_table)
        if not columns:
            self.selected_table.clearContents()
            self.selected_table.setRowCount(0)
            self.selected_table.setColumnCount(1)
            self.selected_table.setHorizontalHeaderLabels(["No selected compounds"])
            if selected_tab_index >= 0:
                self.tabs.setTabText(selected_tab_index, "Selected")
            return

        self.selected_table.clearContents()
        self.selected_table.setColumnCount(len(columns))
        self.selected_table.setHorizontalHeaderLabels(columns)
        self.selected_table.setRowCount(len(rows))
        for row_idx, values in enumerate(rows):
            for col_idx, value in enumerate(values):
                self.selected_table.setItem(row_idx, col_idx, QTableWidgetItem(value))
        self.selected_table.resizeColumnsToContents()
        if selected_tab_index >= 0:
            self.tabs.setTabText(selected_tab_index, f"Selected ({len(rows)})")

    def update_report_browser(self, result):
        html = qsar_service.build_report_html_from_context(self._build_report_context(result))
        self.report_browser.setHtml(html)

    def _build_report_context(self, result):
        total_desc = len(self.data.domain.attributes) if self.data is not None else 0
        used_desc = self.num_features if self.enable_feature_selection else total_desc
        return qsar_service.build_report_context(
            model_name=self.last_model_name,
            total_descriptors=total_desc,
            descriptors_used=used_desc,
            cv_score=result.get("cv_score"),
            train_metrics=result.get("train_metrics", {}),
            test_metrics=result.get("test_metrics", {}),
            external_metrics=result.get("external_metrics", {}),
        )

    # ── Williams AD tab ───────────────────────────────────────────────────

    # ── Features tab ──────────────────────────────────────────────────────

    def _render_features_tab(self, result: dict) -> None:
        self.clear_layout(self.features_layout)
        payload = qsar_service.build_feature_inspection_payload(result, model_name=self.last_model_name)

        if not payload.available:
            self.features_layout.addWidget(
                qsar_features_ui.build_feature_message_label(
                    payload.message_html or "No feature information available."
                )
            )
            self.tabs.setTabText(self.tabs.indexOf(self.features_tab), payload.tab_title)
            return

        if payload.values is None:
            self.features_layout.addWidget(qsar_features_ui.build_feature_message_label(payload.message_html))
            self._render_features_table(list(payload.names), None, payload.value_label)
            self.tabs.setTabText(self.tabs.indexOf(self.features_tab), payload.tab_title)
            return

        splitter = QSplitter(_Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        fig = qsar_features_ui.build_feature_chart_figure(
            payload.chart_names,
            payload.chart_values,
            payload.chart_colors,
            value_label=payload.value_label,
            chart_title=payload.chart_title,
        )
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chart_container = QWidget()
        chart_vl = QVBoxLayout(chart_container)
        chart_vl.setContentsMargins(0, 0, 0, 0)
        chart_vl.addWidget(canvas)

        if payload.value_label == "Coefficient":
            legend_lbl = QLabel(
                "<span style='color:#16a34a'>■</span> positive &nbsp;"
                "<span style='color:#dc2626'>■</span> negative"
            )
            legend_lbl.setStyleSheet("font-size:10px; padding:2px 0;")
            chart_vl.addWidget(legend_lbl)

        splitter.addWidget(chart_container)

        tbl_container = QWidget()
        tbl_vl = QVBoxLayout(tbl_container)
        tbl_vl.setContentsMargins(0, 0, 0, 0)
        tbl_vl.addWidget(qsar_features_ui.build_feature_subtitle_label(payload.subtitle))
        tbl = qsar_features_ui.build_features_table(
            list(payload.names),
            payload.values,
            payload.value_label,
            ses=payload.ses,
            ts=payload.ts,
            ps=payload.ps,
            vifs=payload.vifs,
        )
        tbl_vl.addWidget(tbl)
        splitter.addWidget(tbl_container)
        splitter.setSizes([420, 280])

        self.features_layout.addWidget(splitter)
        self.tabs.setTabText(self.tabs.indexOf(self.features_tab), payload.tab_title)

    def _render_features_table(self, names, values, col_label: str) -> None:
        """Render just the table (no chart) when values are unavailable."""
        self.features_layout.addWidget(
            qsar_features_ui.build_feature_subtitle_label(
                f"{len(names)} selected descriptors (no {col_label.lower()} values for this model type)"
            )
        )
        if values is None:
            values = [float("nan")] * len(names)
        tbl = qsar_features_ui.build_features_table(names, values, col_label)
        self.features_layout.addWidget(tbl)

    def _reset_diagnostic_views(self):
        self.clear_layout(self.train_layout)
        self.clear_layout(self.test_layout)
        self.clear_layout(self.ext_layout)
        self.clear_layout(self.features_layout)
        self.tabs.setTabText(self.tabs.indexOf(self.features_tab), "Features")
        self._diagnostic_selectors = {}
        self._diagnostic_contexts = {}
        self._update_selected_table(None)
        self._show_selection_gallery_placeholder("Select points on a diagnostic plot to preview compounds.")

    def _send_primary_outputs(self, result):
        self.Outputs.model.send(
            build_qsar_prediction_bundle(
                self.model,
                feature_names=list(result.get("feature_names", [])),
                target_label=str(result.get("target_column", "") or DEFAULT_QSAR_TARGET_COLUMN),
                recipe_kind="rdkit_compact" if result.get("generated_descriptors") else None,
                model_name=self.last_model_name,
                source_widget=self.name,
                training_rows=len(result["train_table"]) if result.get("train_table") is not None else None,
                selected_feature_names=list(result.get("feature_names", [])),
            )
        )
        self.Outputs.train_results.send(result["train_table"])
        self.Outputs.test_results.send(result["test_table"])
        if result.get("external_table") is not None:
            self.Outputs.external_results.send(result["external_table"])
        self.Outputs.selected_compounds.send(None)
        coef_table = qsar_service.extract_descriptor_coefficients(
            result["pipeline"], result["feature_names"]
        )
        self.Outputs.descriptor_coefficients.send(coef_table)
        self.Outputs.modeling_summary.send(result.get("modeling_summary_table"))
        self.Outputs.applicability_domain.send(result.get("applicability_domain_table"))
        self.Outputs.model_ranking.send(result.get("model_ranking_table"))

    def _render_diagnostics_from_result(self, result):
        for payload in qsar_service.diagnostic_payloads_from_result(
            result,
            include_external=self.external_data is not None,
        ):
            self.update_diagnostics(
                payload.dataset_type,
                payload.X,
                payload.y,
                payload.pipeline,
                payload.is_classification,
                payload.result_table,
            )

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def create_pdf_report_figure(self, result):
        return qsar_service.build_pdf_report_figure_from_context(self._build_report_context(result))

    def export_pdf(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export PDF", "", "PDF Files (*.pdf)")
        if filename:
            try:
                report_fig = self.create_pdf_report_figure(self._latest_result) if self._latest_result is not None else None
                figures = qsar_service.collect_pdf_export_figures(
                    report_fig,
                    self.last_train_fig,
                    self.last_test_fig,
                    self.last_ext_fig,
                )
                if not figures:
                    self._set_status(qsar_service.build_pdf_export_empty_status_text())
                    return
                with PdfPages(filename) as pdf:
                    for fig in figures:
                        pdf.savefig(fig)
                self._set_status(qsar_service.build_pdf_export_success_status_text())
            except Exception as e:
                self._set_status(qsar_service.build_pdf_export_error_status_text(e))

    def send_report(self):
        if self.model is not None:
            self.report_plot()
            self.report_caption("QSAR Regression Model\n" + self.info_label.text())

# ----------------------------------------------------------------------
class QSARWorker(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()

    def __init__(self, data, external_data, config: qsar_service.QSARRunConfig, parent=None):
        super().__init__(parent)
        self.data = data
        self.external_data = external_data
        self.config = config
        self.last_result = None

    def _emit_cancelled_if_requested(self) -> bool:
        if self.isInterruptionRequested():
            self.cancelled_signal.emit()
            return True
        return False

    def run(self):
        try:
            result = qsar_service.run_qsar_regression(
                self.data,
                self.external_data,
                self.config,
                interruption_requested=self.isInterruptionRequested,
            )
            if result is None:
                self.cancelled_signal.emit()
                return
            self.last_result = result
            if self._emit_cancelled_if_requested():
                return
            self.finished_signal.emit(result)
        except Exception as ex:
            self.error_signal.emit(str(ex))

if __name__ == "__main__":
    app = QApplication([])
    ow = OWQSARRegression()
    ow.show()
    app.exec_()
