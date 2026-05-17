from __future__ import annotations

import concurrent.futures
import traceback

import numpy as np
import pandas as pd
import pyqtgraph as pg
from AnyQt.QtCore import Qt, QThread, pyqtSignal, pyqtSlot as Slot
from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.model_explanation_service import (
    ModelExplanationConfig,
    explain_qsar_model,
)

pg.setConfigOptions(antialias=True)

# ---------------------------------------------------------------------------
# Fast Orange <-> DataFrame helpers
# ---------------------------------------------------------------------------

def _table_to_df(data: Table | None) -> pd.DataFrame | None:
    if data is None:
        return None
    cols: dict = {}
    n = len(data)
    if list(data.domain.attributes):
        X = np.array(data.X, dtype=float)
        for i, v in enumerate(data.domain.attributes):
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
                cols[v.name] = col.astype(float)
    return pd.DataFrame(cols, index=range(n))


def _df_to_table(df: pd.DataFrame | None) -> Table:
    if df is None or df.empty:
        return Table.from_numpy(Domain([]), X=np.empty((0, 0)))
    attrs, metas, x_cols, m_cols = [], [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            attrs.append(ContinuousVariable(str(col)))
            x_cols.append(col)
        else:
            metas.append(StringVariable(str(col)))
            m_cols.append(col)
    X = (
        df[x_cols].to_numpy(dtype=float)
        if x_cols
        else np.empty((len(df), 0), dtype=float)
    )
    M = (
        df[m_cols].fillna("").astype(str).to_numpy(dtype=object)
        if m_cols
        else np.empty((len(df), 0), dtype=object)
    )
    return Table.from_numpy(Domain(attrs, metas=metas), X=X, metas=M)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _Worker(QThread):
    finished = pyqtSignal(object)   # result or Exception
    failed = pyqtSignal(str)

    def __init__(self, df, model, cfg, parent=None):
        super().__init__(parent)
        self._df = df
        self._model = model
        self._cfg = cfg

    def run(self):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(explain_qsar_model, self._df, self._model, self._cfg)
                result = future.result()
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class OWModelExplanation(OWWidget):
    name = "Model Explanation"
    description = "Explain QSAR models and feature contributions for descriptor/fingerprint tables."
    icon = "icons/modeling/ow_model_explanation.png"
    priority = 146
    keywords = ["QSAR", "explain", "feature importance", "interpretability"]

    want_main_area = True
    resizing_enabled = True

    methods: list[str] = ["auto", "permutation", "univariate"]

    # ------------------------------------------------------------------
    # Inputs / Outputs
    # ------------------------------------------------------------------
    class Inputs:
        data = Input("Data", Table)
        model = Input("Model", object, auto_summary=False)

    class Outputs:
        feature_importance = Output("Feature Importance", Table)
        local_contributions = Output("Local Contributions", Table)
        feature_summary = Output("Feature Summary", Table)
        explanation_summary = Output("Explanation Summary", Table)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    target_column: str = Setting("pActivity")
    id_column: str = Setting("compound_id")
    method_index: int = Setting(0)
    max_features: int = Setting(50)
    auto_run: bool = Setting(False)

    # ------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self._data: Table | None = None
        self._model = None
        self._worker: _Worker | None = None

        self._build_control_area()
        self._build_main_area()

    # ------------------------------------------------------------------
    # Control area
    # ------------------------------------------------------------------
    def _build_control_area(self):
        ca = self.controlArea
        ca_layout = QVBoxLayout()
        ca_layout.setContentsMargins(8, 8, 8, 8)
        ca_layout.setSpacing(6)
        ca.setLayout(ca_layout)

        # --- Header ---
        hdr_widget = QWidget()
        hdr_layout = QVBoxLayout(hdr_widget)
        hdr_layout.setContentsMargins(0, 0, 0, 0)
        hdr_layout.setSpacing(2)

        title_row = QWidget()
        title_row_layout = QHBoxLayout(title_row)
        title_row_layout.setContentsMargins(0, 0, 0, 0)

        lbl_title = QLabel("Model Explanation")
        lbl_title.setObjectName("HdrTitle")
        lbl_title.setStyleSheet("font-weight:600;font-size:13px;")
        title_row_layout.addWidget(lbl_title)
        title_row_layout.addStretch(1)

        self._status_chip = QLabel("Ready")
        self._status_chip.setObjectName("StatusChip")
        self._status_chip.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._status_chip.setStyleSheet(
            "padding:4px 8px;border:1px solid #e1e1e1;"
            "border-radius:10px;background:#fafafa;"
        )
        title_row_layout.addWidget(self._status_chip)
        hdr_layout.addWidget(title_row)

        lbl_sub = QLabel("Feature importance & local contributions")
        lbl_sub.setObjectName("HdrSub")
        lbl_sub.setStyleSheet("color:#666;font-size:11px;")
        hdr_layout.addWidget(lbl_sub)

        ca_layout.addWidget(hdr_widget)

        # --- Input columns group ---
        grp_input = QGroupBox("Input columns")
        grp_input_layout = QVBoxLayout(grp_input)
        grp_input_layout.setSpacing(4)

        grp_input_layout.addWidget(QLabel("Target column:"))
        self._target_edit = QLineEdit(self.target_column)
        self._target_edit.textChanged.connect(self._on_target_changed)
        grp_input_layout.addWidget(self._target_edit)

        grp_input_layout.addWidget(QLabel("ID column:"))
        self._id_edit = QLineEdit(self.id_column)
        self._id_edit.textChanged.connect(self._on_id_changed)
        grp_input_layout.addWidget(self._id_edit)

        ca_layout.addWidget(grp_input)

        # --- Explanation group ---
        grp_expl = QGroupBox("Explanation")
        grp_expl_layout = QVBoxLayout(grp_expl)
        grp_expl_layout.setSpacing(4)

        grp_expl_layout.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(self.methods)
        self._method_combo.setCurrentIndex(self.method_index)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        grp_expl_layout.addWidget(self._method_combo)

        grp_expl_layout.addWidget(QLabel("Max features:"))
        self._max_features_spin = QSpinBox()
        self._max_features_spin.setRange(5, 500)
        self._max_features_spin.setSingleStep(5)
        self._max_features_spin.setValue(self.max_features)
        self._max_features_spin.valueChanged.connect(self._on_max_features_changed)
        grp_expl_layout.addWidget(self._max_features_spin)

        self._auto_run_cb = QCheckBox("Auto-run")
        self._auto_run_cb.setChecked(self.auto_run)
        self._auto_run_cb.stateChanged.connect(self._on_auto_run_changed)
        grp_expl_layout.addWidget(self._auto_run_cb)

        self._run_btn = QPushButton("Explain model")
        self._run_btn.clicked.connect(self.commit)
        grp_expl_layout.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        grp_expl_layout.addWidget(self._progress)

        ca_layout.addWidget(grp_expl)
        ca_layout.addStretch(1)

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

        # --- Tab 1: Feature Importance ---
        fi_widget = QWidget()
        fi_layout = QVBoxLayout(fi_widget)
        fi_layout.setContentsMargins(4, 4, 4, 4)
        fi_layout.setSpacing(4)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=False, alpha=0.15)
        self._plot_widget.getAxis("left").setStyle(tickLength=0)
        fi_layout.addWidget(self._plot_widget, 1)

        self._hover_label = QLabel("Top feature: —")
        self._hover_label.setStyleSheet("font-size:11px;color:#444;padding:2px 4px;")
        fi_layout.addWidget(self._hover_label)

        self._tabs.addTab(fi_widget, "Feature Importance")

        # --- Tab 2: Local Contributions ---
        self._local_browser = QTextBrowser()
        self._local_browser.setOpenExternalLinks(False)
        self._tabs.addTab(self._local_browser, "Local Contributions")

        # --- Tab 3: Summary ---
        self._summary_browser = QTextBrowser()
        self._summary_browser.setOpenExternalLinks(False)
        self._tabs.addTab(self._summary_browser, "Summary")

    # ------------------------------------------------------------------
    # Settings change slots
    # ------------------------------------------------------------------
    @Slot(str)
    def _on_target_changed(self, text: str):
        self.target_column = text
        self._maybe_autorun()

    @Slot(str)
    def _on_id_changed(self, text: str):
        self.id_column = text
        self._maybe_autorun()

    @Slot(int)
    def _on_method_changed(self, idx: int):
        self.method_index = idx
        self._maybe_autorun()

    @Slot(int)
    def _on_max_features_changed(self, val: int):
        self.max_features = val
        self._maybe_autorun()

    @Slot(int)
    def _on_auto_run_changed(self, state: int):
        self.auto_run = bool(state)
        self._maybe_autorun()

    def _maybe_autorun(self):
        if self.auto_run and self._data is not None and self._model is not None:
            self.commit()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    @Inputs.data
    def set_data(self, data: Table | None):
        self._data = data
        if data is not None:
            self._auto_detect_columns(data)
        if self.auto_run and self._model is not None:
            self.commit()

    @Inputs.model
    def set_model(self, model):
        self._model = model
        if self.auto_run and self._data is not None:
            self.commit()

    # ------------------------------------------------------------------
    # Auto-detect column names from domain
    # ------------------------------------------------------------------
    def _auto_detect_columns(self, data: Table):
        all_vars = (
            list(data.domain.attributes)
            + list(data.domain.class_vars)
            + list(data.domain.metas)
        )
        names_lower = [v.name.lower() for v in all_vars]
        names = [v.name for v in all_vars]

        # target
        if not self._target_edit.text().strip():
            for kw in ("pactivity", "activity", "target"):
                for i, nl in enumerate(names_lower):
                    if kw in nl:
                        self._target_edit.setText(names[i])
                        break
                else:
                    continue
                break

        # id
        if not self._id_edit.text().strip():
            for kw in ("compound_id", "id", "name"):
                for i, nl in enumerate(names_lower):
                    if kw in nl:
                        self._id_edit.setText(names[i])
                        break
                else:
                    continue
                break

    # ------------------------------------------------------------------
    # Commit / run
    # ------------------------------------------------------------------
    def commit(self):
        if self._data is None or self._model is None:
            self._send_none()
            self._set_status("No data/model", error=True)
            return

        if self._worker is not None and self._worker.isRunning():
            return  # already running

        df = _table_to_df(self._data)
        cfg = ModelExplanationConfig(
            target_column=self._target_edit.text().strip() or "pActivity",
            id_column=self._id_edit.text().strip() or "compound_id",
            method=self.methods[int(self.method_index)],
            max_features=int(self._max_features_spin.value()),
        )

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._set_status("Running…")

        self._worker = _Worker(df, self._model, cfg, parent=self)
        self._worker.finished.connect(self._finish)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------
    @Slot(object)
    def _finish(self, result):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)

        # Send outputs
        self.Outputs.feature_importance.send(_df_to_table(result.feature_importance))
        self.Outputs.local_contributions.send(_df_to_table(result.local_contributions))
        self.Outputs.feature_summary.send(_df_to_table(result.feature_summary))
        self.Outputs.explanation_summary.send(_df_to_table(result.explanation_summary))

        sd = result.summary_dict
        method_used = sd.get("method_used", "?")
        n_features = sd.get("features_reported", 0)
        self._set_status(f"{method_used} · {n_features} features")

        self._populate_importance_plot(result.feature_importance)
        self._populate_local_contributions(result.local_contributions)
        self._populate_summary(result)

    @Slot(str)
    def _on_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._send_none()
        self._set_status("Error", error=True)
        self._summary_browser.setPlainText(f"Model Explanation failed:\n\n{msg}")

    # ------------------------------------------------------------------
    # Tab population
    # ------------------------------------------------------------------
    def _populate_importance_plot(self, fi_df: pd.DataFrame | None):
        pw = self._plot_widget
        pw.clear()

        if fi_df is None or fi_df.empty:
            self._hover_label.setText("Top feature: —")
            return

        top_n = min(25, len(fi_df))
        df = fi_df.sort_values("importance", ascending=False).head(top_n)
        # reverse so rank-1 is at top of chart
        df = df.iloc[::-1].reset_index(drop=True)

        values = df["importance"].to_numpy(dtype=float)
        feature_names = df["feature"].tolist()
        n = len(values)
        y_pos = np.arange(n, dtype=float)

        bars = pg.BarGraphItem(
            x1=np.zeros(n),
            x0=values,
            y=y_pos,
            height=0.65,
            brush=pg.mkBrush(37, 99, 235, 200),
            pen=pg.mkPen(None),
        )
        pw.addItem(bars)

        # Y-axis ticks: feature names
        ticks = [(float(i), name) for i, name in enumerate(feature_names)]
        pw.getAxis("left").setTicks([ticks])
        pw.getAxis("left").setStyle(tickLength=0, tickTextOffset=4)
        pw.setYRange(-0.5, n - 0.5, padding=0.02)

        # Scatter overlay for hover
        scatter = pg.ScatterPlotItem(
            x=values,
            y=y_pos,
            data=feature_names,
            size=1,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 0, 0, 0),
            hoverable=True,
        )

        def _update_hover(pts, _ev=None):
            if pts:
                name = pts[0].data()
                score = pts[0].pos().x()
                self._hover_label.setText(f"Feature: {name}  (score={score:.4g})")
            else:
                # show top feature as default
                top_name = feature_names[-1] if feature_names else "—"
                top_score = values[-1] if len(values) else 0.0
                self._hover_label.setText(
                    f"Top feature: {top_name}  (score={top_score:.4g})"
                )

        scatter.sigHovered.connect(_update_hover)
        pw.addItem(scatter)

        # Default info label
        top_name = feature_names[-1] if feature_names else "—"
        top_score = float(values[-1]) if len(values) else 0.0
        self._hover_label.setText(
            f"Top feature: {top_name}  (score={top_score:.4g})"
        )

    def _populate_local_contributions(self, lc_df: pd.DataFrame | None):
        if lc_df is None or lc_df.empty:
            self._local_browser.setHtml("<p><i>No local contributions available.</i></p>")
            return

        id_col = self._id_edit.text().strip() or "compound_id"
        # find all feature columns (non-id)
        feat_cols = [c for c in lc_df.columns if c != id_col]

        html_parts = [
            "<html><body style='font-family:monospace;font-size:12px;'>",
            "<h3>Local Contributions (top 5 features per compound)</h3>",
        ]

        max_compounds = min(50, len(lc_df))
        for _, row in lc_df.head(max_compounds).iterrows():
            cid = row[id_col] if id_col in row.index else "?"
            html_parts.append(f"<b>Compound:</b> {cid}<br>")
            html_parts.append(
                "<table border='0' cellspacing='2' style='margin-left:12px;'>"
                "<tr><th align='left'>Feature</th><th align='right'>Contribution</th></tr>"
            )
            # sort feature cols by abs value
            vals = {c: row[c] for c in feat_cols if pd.notna(row.get(c, None))}
            sorted_feats = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for fname, fval in sorted_feats:
                color = "#1a6e1a" if fval >= 0 else "#a40000"
                html_parts.append(
                    f"<tr><td>{fname}</td>"
                    f"<td align='right' style='color:{color};'>{fval:+.4g}</td></tr>"
                )
            html_parts.append("</table><hr>")

        html_parts.append("</body></html>")
        self._local_browser.setHtml("".join(html_parts))

    def _populate_summary(self, result):
        sd = result.summary_dict
        lines = [
            "Model Explanation Summary",
            "=========================",
            f"Method:         {sd.get('method_used', '?')}",
            f"Rows used:      {sd.get('n_rows_used', '?')}",
            f"Features used:  {sd.get('n_features_used', '?')}",
            f"Features reported: {sd.get('features_reported', '?')}",
            "",
            "Top features:",
        ]
        for entry in sd.get("top_features", [])[:20]:
            score = entry.get("normalized_importance", entry.get("importance", 0))
            lines.append(f"  {entry['rank']:>3}. {entry['feature']}: {score:.4g}")
        self._summary_browser.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _send_none(self):
        self.Outputs.feature_importance.send(None)
        self.Outputs.local_contributions.send(None)
        self.Outputs.feature_summary.send(None)
        self.Outputs.explanation_summary.send(None)

    def _set_status(self, text: str, *, error: bool = False):
        self._status_chip.setText(text)
        if error:
            self._status_chip.setStyleSheet(
                "padding:4px 8px;border:1px solid #f2c2c2;"
                "border-radius:10px;background:#fff5f5;color:#a40000;"
            )
        else:
            self._status_chip.setStyleSheet(
                "padding:4px 8px;border:1px solid #e1e1e1;"
                "border-radius:10px;background:#fafafa;"
            )
