from __future__ import annotations

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, pyqtSlot as Slot
from AnyQt.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.qsar_descriptor_explorer_service import (
    QSARDescriptorExplorerConfig,
    explore_qsar_descriptors,
)


# ---------------------------------------------------------------------------
# Orange ↔ pandas helpers
# ---------------------------------------------------------------------------

def _table_to_df(data: Table | None) -> pd.DataFrame | None:
    if data is None:
        return None
    cols: dict[str, object] = {}
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


def _df_to_table(df: pd.DataFrame | None) -> Table:
    if df is None or df.empty:
        return Table.from_numpy(Domain([]), X=np.empty((0, 0)))
    attrs, metas, x_cols, m_cols = [], [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            attrs.append(ContinuousVariable(str(col)))
            x_cols.append(col)
        else:
            metas.append(StringVariable(str(col)))
            m_cols.append(col)
    X = df[x_cols].astype(float).to_numpy(dtype=float) if x_cols else np.empty((len(df), 0), dtype=float)
    M = df[m_cols].fillna("").astype(str).to_numpy(dtype=object) if m_cols else np.empty((len(df), 0), dtype=object)
    return Table.from_numpy(Domain(attrs, metas=metas), X=X, metas=M)


class _Worker:
    def __init__(self, df: pd.DataFrame, config: QSARDescriptorExplorerConfig):
        self.df = df
        self.config = config

    def __call__(self):
        return explore_qsar_descriptors(self.df, self.config)


class OWDescriptorExplorer(OWWidget):
    name = "QSAR Descriptor Explorer"
    description = "Analyze QSAR descriptor quality, categories, missingness, variance, and redundancy."
    icon = "icons/descriptors/owmoldescriptorwidget.png"
    priority = 136
    keywords = ["QSAR", "descriptors", "descriptor quality", "correlation", "variance"]

    want_main_area = True
    resizing_enabled = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        filtered_data = Output("Filtered Data", Table)
        descriptor_summary = Output("Descriptor Summary", Table)
        category_summary = Output("Category Summary", Table)
        correlation_pairs = Output("Correlation Pairs", Table)
        quality_report = Output("Quality Report", Table)
        report_html = Output("Report HTML", str, auto_summary=False)
        report_markdown = Output("Report Markdown", str, auto_summary=False)

    target_column: str = Setting("")
    id_column: str = Setting("compound_id")
    missing_threshold: float = Setting(0.20)
    low_variance_threshold: float = Setting(1.0e-12)
    high_correlation_threshold: float = Setting(0.95)
    auto_run: bool = Setting(True)

    def __init__(self) -> None:
        super().__init__()
        self._data: Table | None = None
        self._executor = ThreadExecutor(self)
        self._task = None
        self._last_result = None

        self._build_control_area()
        self._build_main_area()
        self._set_status("Awaiting descriptor table…", ok=True)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_control_area(self) -> None:
        ca = self.controlArea

        hdr = QWidget()
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(6, 6, 6, 6)
        left = QVBoxLayout()
        title = QLabel("Descriptor Explorer")
        title.setStyleSheet("font-size:14px;font-weight:600;color:#1e293b;")
        sub = QLabel("Quality · categories · redundancy")
        sub.setStyleSheet("font-size:11px;color:#64748b;")
        left.addWidget(title)
        left.addWidget(sub)
        hl.addLayout(left, 1)
        self._lbl_status = QLabel("Ready")
        self._lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hl.addWidget(self._lbl_status)
        ca.layout().addWidget(hdr)

        col_box = QGroupBox("Columns")
        col_vl = QVBoxLayout(col_box)
        row_t = QHBoxLayout()
        row_t.addWidget(QLabel("Target/Y"))
        self._ed_target = QLineEdit(self.target_column)
        self._ed_target.setPlaceholderText("optional, e.g. pActivity")
        self._ed_target.textChanged.connect(self._settings_changed)
        row_t.addWidget(self._ed_target, 1)
        col_vl.addLayout(row_t)
        row_i = QHBoxLayout()
        row_i.addWidget(QLabel("ID column"))
        self._ed_id = QLineEdit(self.id_column)
        self._ed_id.textChanged.connect(self._settings_changed)
        row_i.addWidget(self._ed_id, 1)
        col_vl.addLayout(row_i)
        ca.layout().addWidget(col_box)

        filt_box = QGroupBox("Descriptor filters")
        filt_vl = QVBoxLayout(filt_box)
        self._spin_missing = self._double_row(filt_vl, "Max missing fraction", self.missing_threshold, 0.0, 1.0, 0.05)
        self._spin_var = self._double_row(filt_vl, "Low variance cutoff", self.low_variance_threshold, 0.0, 1.0, 1.0e-6, decimals=8)
        self._spin_corr = self._double_row(filt_vl, "High correlation |r|", self.high_correlation_threshold, 0.50, 1.0, 0.01)

        self._chk_auto = QCheckBox("Auto-run")
        self._chk_auto.setChecked(bool(self.auto_run))
        self._chk_auto.toggled.connect(self._on_auto_toggled)
        filt_vl.addWidget(self._chk_auto)

        self._btn_run = QPushButton("Explore descriptors")
        self._btn_run.clicked.connect(self.commit)
        filt_vl.addWidget(self._btn_run)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        filt_vl.addWidget(self._progress)
        ca.layout().addWidget(filt_box)
        ca.layout().addStretch(1)

    def _double_row(self, parent, label, value, minv, maxv, step, decimals=3):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(float(minv), float(maxv))
        spin.setSingleStep(float(step))
        spin.setDecimals(int(decimals))
        spin.setValue(float(value))
        spin.valueChanged.connect(self._settings_changed)
        row.addWidget(spin, 1)
        parent.addLayout(row)
        return spin

    def _build_main_area(self) -> None:
        ma = self.mainArea
        layout = QVBoxLayout(ma)
        layout.setContentsMargins(4, 4, 4, 4)
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        self._tabs.addTab(self._browser, "Report")

        self._tbl_quality = QTableWidget()
        self._tabs.addTab(self._tbl_quality, "Quality")
        self._tbl_categories = QTableWidget()
        self._tabs.addTab(self._tbl_categories, "Categories")
        self._tbl_descriptors = QTableWidget()
        self._tabs.addTab(self._tbl_descriptors, "Descriptors")
        self._tbl_corr = QTableWidget()
        self._tabs.addTab(self._tbl_corr, "Correlations")

    # ------------------------------------------------------------------
    # Settings/input
    # ------------------------------------------------------------------

    @Inputs.data
    def set_data(self, data: Table | None) -> None:
        self._data = data
        if data is None:
            self._set_status("No data", ok=True)
        else:
            self._set_status(f"{len(data)} rows", ok=True)
        self._maybe_autorun()

    def _settings_changed(self, *_) -> None:
        self.target_column = self._ed_target.text().strip()
        self.id_column = self._ed_id.text().strip()
        self.missing_threshold = float(self._spin_missing.value())
        self.low_variance_threshold = float(self._spin_var.value())
        self.high_correlation_threshold = float(self._spin_corr.value())
        self._maybe_autorun()

    def _on_auto_toggled(self, value: bool) -> None:
        self.auto_run = bool(value)
        self._maybe_autorun()

    def _maybe_autorun(self) -> None:
        if self.auto_run:
            self.commit()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def commit(self) -> None:
        if self._data is None:
            self._clear_outputs()
            self._browser.setHtml("<p>Connect a descriptor table.</p>")
            return
        df = _table_to_df(self._data)
        if df is None or df.empty:
            self._clear_outputs()
            self._browser.setHtml("<p>Input table is empty.</p>")
            return

        config = QSARDescriptorExplorerConfig(
            target_column=self.target_column,
            id_column=self.id_column,
            missing_threshold=float(self.missing_threshold),
            low_variance_threshold=float(self.low_variance_threshold),
            high_correlation_threshold=float(self.high_correlation_threshold),
        )
        self._set_busy(True)
        self._set_status("Analyzing…", ok=True)
        self._task = self._executor.submit(_Worker(df, config))
        self._task.add_done_callback(methodinvoke(self, "_on_done", (object,)))

    @Slot(object)
    def _on_done(self, future) -> None:
        self._set_busy(False)
        try:
            result = future.result()
        except Exception as exc:
            self._set_status(f"Error: {exc}", ok=False)
            self._browser.setHtml(f"<p><b>Error:</b> {exc}</p>")
            self._clear_outputs()
            return
        self._last_result = result
        self._browser.setHtml(result.html_report)
        self._fill_table(self._tbl_quality, result.quality_report)
        self._fill_table(self._tbl_categories, result.category_summary)
        self._fill_table(self._tbl_descriptors, result.descriptor_summary)
        self._fill_table(self._tbl_corr, result.correlation_pairs)
        self.Outputs.filtered_data.send(_df_to_table(result.filtered_data))
        self.Outputs.descriptor_summary.send(_df_to_table(result.descriptor_summary))
        self.Outputs.category_summary.send(_df_to_table(result.category_summary))
        self.Outputs.correlation_pairs.send(_df_to_table(result.correlation_pairs))
        self.Outputs.quality_report.send(_df_to_table(result.quality_report))
        self.Outputs.report_html.send(result.html_report)
        self.Outputs.report_markdown.send(result.markdown_report)
        n_final = 0
        if result.quality_report is not None and not result.quality_report.empty:
            m = result.quality_report.set_index("metric")["value"].to_dict()
            n_final = int(m.get("recommended_after_redundancy_filter", 0) or 0)
        self._set_status(f"Done · {n_final} recommended descriptors", ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        self._progress.setVisible(bool(busy))
        self._btn_run.setEnabled(not busy)

    def _set_status(self, text: str, ok: bool = True) -> None:
        self._lbl_status.setText(text)
        if ok:
            self._lbl_status.setStyleSheet("padding:4px 8px;border:1px solid #e1e1e1;border-radius:10px;background:#fafafa;")
        else:
            self._lbl_status.setStyleSheet("padding:4px 8px;border:1px solid #f2c2c2;border-radius:10px;background:#fff5f5;color:#a40000;")

    def _clear_outputs(self) -> None:
        empty = Table.from_numpy(Domain([]), X=np.empty((0, 0)))
        self.Outputs.filtered_data.send(empty)
        self.Outputs.descriptor_summary.send(empty)
        self.Outputs.category_summary.send(empty)
        self.Outputs.correlation_pairs.send(empty)
        self.Outputs.quality_report.send(empty)
        self.Outputs.report_html.send("")
        self.Outputs.report_markdown.send("")

    @staticmethod
    def _fill_table(widget: QTableWidget, df: pd.DataFrame | None, max_rows: int = 300) -> None:
        widget.clear()
        if df is None or df.empty:
            widget.setRowCount(0)
            widget.setColumnCount(0)
            return
        show = df.head(max_rows).copy()
        widget.setColumnCount(len(show.columns))
        widget.setRowCount(len(show))
        widget.setHorizontalHeaderLabels([str(c) for c in show.columns])
        for i, (_, row) in enumerate(show.iterrows()):
            for j, col in enumerate(show.columns):
                value = row[col]
                if isinstance(value, float):
                    text = f"{value:.5g}"
                else:
                    text = "" if pd.isna(value) else str(value)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                widget.setItem(i, j, item)
        widget.resizeColumnsToContents()
