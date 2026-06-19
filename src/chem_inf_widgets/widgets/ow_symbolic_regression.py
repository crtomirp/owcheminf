from __future__ import annotations

from html import escape
from typing import Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from chem_inf_widgets.chemcore.services.symbolic_regression_service import (
    SymbolicRegressionConfig,
    SymbolicRegressionResult,
    continuous_target_candidates,
    fit_symbolic_regression,
    preferred_target_name,
)
from chem_inf_widgets.widgets.ui_helpers import (
    format_done_status,
    format_failed_status,
    format_no_input_status,
)


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
                    num = float(value)
                    values.append(f"{num:.5g}" if np.isfinite(num) else "")
                except Exception:
                    values.append("")
            else:
                values.append("" if value is None else str(value))
        rows.append(values)
    return columns, rows


class OWSymbolicRegression(OWWidget):
    name = "Symbolic Regression"
    description = "Experimental symbolic regression using sparse basis-function search over descriptor columns."
    icon = "icons/modeling/owsymbolicregressionwidget.svg"
    priority = 143
    keywords = ["symbolic", "regression", "experimental", "qsar"]

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        model = Output("Model", object, auto_summary=False)
        predictions = Output("Predictions", Table, default=True)
        term_table = Output("Term Table", Table)
        modeling_summary = Output("Modeling Summary", Table)
        expression = Output("Expression", str, auto_summary=False)

    want_main_area = True

    target_name: str = Setting("")
    max_features: int = Setting(6)
    max_terms: int = Setting(4)
    cv_folds: int = Setting(5)
    include_square: bool = Setting(True)
    include_cube: bool = Setting(False)
    include_log: bool = Setting(True)
    include_sqrt: bool = Setting(True)
    include_inverse: bool = Setting(True)
    include_interactions: bool = Setting(True)
    auto_run: bool = Setting(True)

    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[Table] = None
        self._result: Optional[SymbolicRegressionResult] = None

        settings_box = gui.widgetBox(self.controlArea, "Experimental Search")
        self._target_combo = gui.comboBox(
            settings_box,
            self,
            "target_name",
            label="Target:",
            orientation=Qt.Horizontal,
            callback=self._on_settings_changed,
            sendSelectedValue=True,
        )
        gui.spin(
            settings_box,
            self,
            "max_features",
            minv=2,
            maxv=32,
            step=1,
            label="Top descriptors:",
            callback=self._on_settings_changed,
        )
        gui.spin(
            settings_box,
            self,
            "max_terms",
            minv=1,
            maxv=12,
            step=1,
            label="Max expression terms:",
            callback=self._on_settings_changed,
        )
        gui.spin(
            settings_box,
            self,
            "cv_folds",
            minv=2,
            maxv=10,
            step=1,
            label="CV folds:",
            callback=self._on_settings_changed,
        )

        basis_box = gui.widgetBox(self.controlArea, "Basis Library")
        gui.checkBox(basis_box, self, "include_square", "Include squares", callback=self._on_settings_changed)
        gui.checkBox(basis_box, self, "include_cube", "Include cubes", callback=self._on_settings_changed)
        gui.checkBox(basis_box, self, "include_log", "Include sign(x) * log1p(abs(x))", callback=self._on_settings_changed)
        gui.checkBox(basis_box, self, "include_sqrt", "Include sign(x) * sqrt(abs(x))", callback=self._on_settings_changed)
        gui.checkBox(basis_box, self, "include_inverse", "Include 1 / (1 + abs(x))", callback=self._on_settings_changed)
        gui.checkBox(basis_box, self, "include_interactions", "Include pairwise interactions", callback=self._on_settings_changed)

        run_box = gui.widgetBox(self.controlArea, "Run")
        gui.checkBox(run_box, self, "auto_run", "Auto-run on changes", callback=self._on_settings_changed)
        gui.button(run_box, self, "Fit symbolic regression", callback=self.commit)

        self._status = QLabel("Waiting for data.")
        self._status.setWordWrap(True)
        self.controlArea.layout().addWidget(self._status)

        splitter = QSplitter(Qt.Vertical)
        self.mainArea.layout().addWidget(splitter)

        self._expression_browser = QTextBrowser()
        self._expression_browser.setOpenExternalLinks(False)
        splitter.addWidget(self._expression_browser)

        tabs = QTabWidget()
        splitter.addWidget(tabs)
        splitter.setSizes([180, 520])

        self._summary_browser = QTextBrowser()
        tabs.addTab(self._summary_browser, "Summary")

        self._terms_table = QTableWidget()
        tabs.addTab(self._terms_table, "Terms")

        self._predictions_table = QTableWidget()
        tabs.addTab(self._predictions_table, "Predictions")

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure(figsize=(5.0, 4.0))
        self._canvas = FigureCanvas(self._figure)
        plot_layout.addWidget(self._canvas)
        tabs.addTab(plot_widget, "Observed vs Predicted")

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self.data = data
        self._refresh_target_combo()
        if self.auto_run:
            self.commit()
        else:
            self._clear_outputs(keep_status=True)
            self._status.setText("Ready to fit symbolic regression.")

    def _refresh_target_combo(self) -> None:
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        if self.data is None:
            self.target_name = ""
            self._target_combo.blockSignals(False)
            return
        target_vars = continuous_target_candidates(self.data)
        for variable in target_vars:
            self._target_combo.addItem(variable.name)
        target_names = [variable.name for variable in target_vars]
        if self.target_name not in target_names:
            self.target_name = preferred_target_name(self.data)
        if self.target_name:
            self._target_combo.setCurrentText(self.target_name)
        self._target_combo.blockSignals(False)

    def _config(self) -> SymbolicRegressionConfig:
        return SymbolicRegressionConfig(
            max_features=int(self.max_features),
            max_terms=int(self.max_terms),
            cv_folds=int(self.cv_folds),
            include_square=bool(self.include_square),
            include_cube=bool(self.include_cube),
            include_log=bool(self.include_log),
            include_sqrt=bool(self.include_sqrt),
            include_inverse=bool(self.include_inverse),
            include_interactions=bool(self.include_interactions),
        )

    def _on_settings_changed(self) -> None:
        if self.auto_run and self.data is not None:
            self.commit()

    def commit(self) -> None:
        if self.data is None or len(self.data) == 0:
            self._clear_outputs()
            self._status.setText(format_no_input_status())
            return
        try:
            result = fit_symbolic_regression(
                self.data,
                target_name=self.target_name or preferred_target_name(self.data),
                config=self._config(),
            )
        except Exception as exc:
            self._result = None
            self._clear_outputs()
            self._status.setText(format_failed_status(str(exc)))
            return

        self._result = result
        self._update_view(result)
        self._status.setText(
            format_done_status(
                f"rows={len(self.data)}",
                f"candidate terms={len(result.candidate_terms)}",
                f"selected terms={len(result.selected_terms)}",
                prefix="Fitted",
            )
        )
        self.Outputs.model.send(result.model)
        self.Outputs.predictions.send(result.predictions_table)
        self.Outputs.term_table.send(result.term_table)
        self.Outputs.modeling_summary.send(result.summary_table)
        self.Outputs.expression.send(result.expression)

    def _update_view(self, result: SymbolicRegressionResult) -> None:
        self._expression_browser.setHtml(
            "<h3>Expression</h3>"
            f"<p><code>{escape(result.expression)}</code></p>"
        )
        self._summary_browser.setHtml(
            "<h3>Training Summary</h3>"
            f"<p><b>Train R²:</b> {result.train_metrics.get('r2', float('nan')):.4f}<br>"
            f"<b>Train RMSE:</b> {result.train_metrics.get('rmse', float('nan')):.4f}<br>"
            f"<b>CV R²:</b> {result.cv_metrics.get('r2', float('nan')):.4f}<br>"
            f"<b>CV RMSE:</b> {result.cv_metrics.get('rmse', float('nan')):.4f}</p>"
            "<h4>Selected Terms</h4>"
            "<ul>"
            + "".join(f"<li>{escape(term.label)}</li>" for term in result.selected_terms)
            + "</ul>"
        )
        self._populate_table_widget(self._terms_table, result.term_table)
        self._populate_table_widget(self._predictions_table, result.predictions_table)
        self._update_plot(result)

    def _update_plot(self, result: SymbolicRegressionResult) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        y_true = next(
            (
                self.data.get_column(var).astype(float)
                for var in list(self.data.domain.class_vars) + list(self.data.domain.attributes)
                if getattr(var, "is_continuous", False) and var.name == result.target_name
            ),
            np.asarray([], dtype=float),
        )
        y_pred = np.asarray(result.predictions, dtype=float)
        ax.scatter(y_true, y_pred, s=28, alpha=0.8, color="#2563eb", edgecolor="white", linewidth=0.4)
        if len(y_true):
            lo = float(min(np.min(y_true), np.min(y_pred)))
            hi = float(max(np.max(y_true), np.max(y_pred)))
            ax.plot([lo, hi], [lo, hi], color="#94a3b8", linewidth=1.0, linestyle="--")
        ax.set_xlabel(f"Observed {result.target_name}")
        ax.set_ylabel("Predicted")
        ax.set_title("Observed vs Predicted")
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def _populate_table_widget(self, widget: QTableWidget, table: Optional[Table]) -> None:
        columns, rows = _table_display_rows(table)
        widget.clear()
        widget.setColumnCount(len(columns))
        widget.setRowCount(len(rows))
        widget.setHorizontalHeaderLabels(columns)
        for row_idx, row in enumerate(rows):
            for col_idx, value in enumerate(row):
                widget.setItem(row_idx, col_idx, QTableWidgetItem(value))
        widget.resizeColumnsToContents()

    def _clear_outputs(self, *, keep_status: bool = False) -> None:
        self._result = None
        self._expression_browser.clear()
        self._summary_browser.clear()
        self._terms_table.clear()
        self._terms_table.setRowCount(0)
        self._terms_table.setColumnCount(0)
        self._predictions_table.clear()
        self._predictions_table.setRowCount(0)
        self._predictions_table.setColumnCount(0)
        self._figure.clear()
        self._canvas.draw_idle()
        self.Outputs.model.send(None)
        self.Outputs.predictions.send(None)
        self.Outputs.term_table.send(None)
        self.Outputs.modeling_summary.send(None)
        self.Outputs.expression.send("")
        if not keep_status:
            self._status.setText("Waiting for data.")
