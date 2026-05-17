from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyqtgraph as pg
from AnyQt.QtCore import Qt, pyqtSignal, pyqtSlot
from AnyQt.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.qsar_prediction_packager_service import (
    QSARPredictionPackagerConfig,
    QSARPredictionModelBundle,
    predict_with_qsar_model,
)
from chem_inf_widgets.chemcore.services.qsar_target_contract import (
    DEFAULT_QSAR_TARGET_COLUMN,
    infer_target_label_from_model,
    prediction_column_name_for_target,
)

pg.setConfigOptions(antialias=True)

_CHIP_OK = (
    "padding:4px 8px;"
    "border:1px solid #e1e1e1;"
    "border-radius:10px;"
    "background:#fafafa;"
)
_CHIP_ERR = (
    "padding:4px 8px;"
    "border:1px solid #f2c2c2;"
    "border-radius:10px;"
    "background:#fff5f5;"
    "color:#a40000;"
)

_AXIS_PEN = pg.mkPen(color="#CBD5E1", width=1)


# ---------------------------------------------------------------------------
# Fast Orange ↔ pandas helpers
# ---------------------------------------------------------------------------

def _table_to_df(data: Table | None) -> pd.DataFrame | None:
    if data is None:
        return None
    cols: dict = {}
    n = len(data)
    attr_vars = list(data.domain.attributes)
    if attr_vars:
        X = np.array(data.X, dtype=float)
        for i, v in enumerate(attr_vars):
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
# Worker
# ---------------------------------------------------------------------------

class _Worker:
    """Callable executed in a thread pool."""

    def __init__(self, model, df: pd.DataFrame, config: QSARPredictionPackagerConfig):
        self._model = model
        self._df = df
        self._config = config

    def __call__(self):
        return predict_with_qsar_model(self._model, self._df, self._config)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class OWQSARPredictionPackager(OWWidget):
    name = "QSAR Prediction Packager"
    description = "Apply a trained QSAR model to query compounds and create a prediction package."
    icon = "icons/modeling/ow_qsar_prediction_packager.png"
    priority = 150
    keywords = ["QSAR", "prediction", "deployment", "external set"]

    want_main_area = True
    resizing_enabled = True

    # --- Signals (thread → main thread) ---
    _sig_done = pyqtSignal(object)   # result object
    _sig_err  = pyqtSignal(str)      # error message

    # --- Orange I/O ---
    class Inputs:
        model = Input("Model", object, auto_summary=False)
        query_data = Input("Query Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        predictions     = Output("Predictions", Table)
        feature_report  = Output("Feature Report", Table)
        package_manifest = Output("Package Manifest", Table)
        failed_records  = Output("Failed Records", Table)

    # --- Settings ---
    id_column             = Setting("compound_id")
    target_label          = Setting(DEFAULT_QSAR_TARGET_COLUMN)
    prediction_column     = Setting(prediction_column_name_for_target(DEFAULT_QSAR_TARGET_COLUMN))
    include_input_columns = Setting(True)
    auto_run              = Setting(False)

    # -----------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self._model = None
        self._query_data = None
        self._molecules = None
        self._target_label_autofill = True
        self._prediction_column_autofill = True
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None

        self._sig_done.connect(self._finish)
        self._sig_err.connect(self._on_error)

        self._build_control_area()
        self._build_main_area()

    # -----------------------------------------------------------------------
    # Control area
    # -----------------------------------------------------------------------

    def _build_control_area(self):
        ca = self.controlArea

        # --- Header ---
        hdr_widget = QWidget()
        hdr_layout = QVBoxLayout(hdr_widget)
        hdr_layout.setContentsMargins(0, 0, 0, 4)

        title_row = QWidget()
        title_row_layout = QHBoxLayout(title_row)
        title_row_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("QSAR Prediction Packager")
        title.setObjectName("HdrTitle")
        title.setStyleSheet("font-weight:bold;font-size:13px;")
        title_row_layout.addWidget(title)
        title_row_layout.addStretch(1)

        self._status_chip = QLabel("Ready")
        self._status_chip.setObjectName("StatusChip")
        self._status_chip.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._status_chip.setStyleSheet(_CHIP_OK)
        title_row_layout.addWidget(self._status_chip)

        hdr_layout.addWidget(title_row)

        subtitle = QLabel("Apply trained model to new compounds")
        subtitle.setObjectName("HdrSub")
        subtitle.setStyleSheet("color:#666;font-size:11px;")
        hdr_layout.addWidget(subtitle)

        ca.layout().addWidget(hdr_widget)

        # --- Group box ---
        grp = QGroupBox("Prediction settings")
        grp_layout = QVBoxLayout(grp)
        grp_layout.setSpacing(6)

        def _row(label_text: str, attr: str) -> QLineEdit:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(120)
            edit = QLineEdit(getattr(self, attr))
            edit.textChanged.connect(lambda text, a=attr: self._on_line_edit(a, text))
            row_layout.addWidget(lbl)
            row_layout.addWidget(edit)
            grp_layout.addWidget(row)
            return edit

        self._edit_id_column         = _row("ID column:",        "id_column")
        self._edit_target_label      = _row("Dependent variable (Y):",     "target_label")
        self._edit_prediction_column = _row("Prediction column:", "prediction_column")

        self._cb_include = QCheckBox("Include input columns")
        self._cb_include.setChecked(bool(self.include_input_columns))
        self._cb_include.stateChanged.connect(self._on_cb_include)
        grp_layout.addWidget(self._cb_include)

        self._cb_auto = QCheckBox("Auto-run")
        self._cb_auto.setChecked(bool(self.auto_run))
        self._cb_auto.stateChanged.connect(self._on_cb_auto)
        grp_layout.addWidget(self._cb_auto)

        self._btn_predict = QPushButton("Predict")
        self._btn_predict.clicked.connect(self.commit)
        grp_layout.addWidget(self._btn_predict)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        grp_layout.addWidget(self._progress)

        ca.layout().addWidget(grp)
        ca.layout().addStretch(1)

    # -----------------------------------------------------------------------
    # Main area
    # -----------------------------------------------------------------------

    def _build_main_area(self):
        tabs = QTabWidget()
        self.mainArea.layout().addWidget(tabs)

        # Tab 1 — Summary
        self._summary_browser = QTextBrowser()
        tabs.addTab(self._summary_browser, "Summary")

        # Tab 2 — Distribution
        dist_widget = QWidget()
        dist_layout = QVBoxLayout(dist_widget)
        dist_layout.setContentsMargins(4, 4, 4, 4)
        self._plot = pg.PlotWidget(title="Predicted value distribution")
        self._plot.setBackground("#FFFFFF")
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        self._plot.getAxis("bottom").setPen(_AXIS_PEN)
        self._plot.getAxis("left").setPen(_AXIS_PEN)
        self._plot.getAxis("bottom").setLabel("Predicted value")
        self._plot.getAxis("left").setLabel("Count")
        self._bar_item: pg.BarGraphItem | None = None
        dist_layout.addWidget(self._plot)
        tabs.addTab(dist_widget, "Distribution")

        # Tab 3 — Manifest
        self._manifest_table = QTableWidget(0, 2)
        self._manifest_table.setHorizontalHeaderLabels(["Property", "Value"])
        self._manifest_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._manifest_table.horizontalHeader().setStretchLastSection(True)
        self._manifest_table.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        tabs.addTab(self._manifest_table, "Manifest")

    # -----------------------------------------------------------------------
    # Slot helpers
    # -----------------------------------------------------------------------

    def _on_line_edit(self, attr: str, text: str):
        if attr == "target_label":
            self._target_label_autofill = False
        elif attr == "prediction_column":
            self._prediction_column_autofill = False
        setattr(self, attr, text)
        self._maybe_commit()

    def _on_cb_include(self, state: int):
        self.include_input_columns = bool(state)
        self._maybe_commit()

    def _on_cb_auto(self, state: int):
        self.auto_run = bool(state)
        self._maybe_commit()

    def _maybe_commit(self):
        if self.auto_run:
            self.commit()

    # -----------------------------------------------------------------------
    # Orange inputs
    # -----------------------------------------------------------------------

    @Inputs.model
    def set_model(self, model):
        self._model = model
        self._sync_labels_from_model(model)
        self._maybe_commit()

    @Inputs.query_data
    def set_query_data(self, data: Table | None):
        self._query_data = data
        self._maybe_commit()

    @Inputs.molecules
    def set_molecules(self, molecules):
        self._molecules = molecules
        self._maybe_commit()

    # -----------------------------------------------------------------------
    # Prediction logic
    # -----------------------------------------------------------------------

    def commit(self):
        df, source_mode = self._build_input_dataframe()
        if self._model is None or df is None or df.empty:
            self._send_none_outputs()
            self._set_status("Need Model + Query Data/Molecules", error=True)
            return

        # Debounce: cancel any running job (best effort)
        if self._future is not None and not self._future.done():
            self._future.cancel()

        config = QSARPredictionPackagerConfig(
            id_column=self.id_column.strip() or "compound_id",
            target_label=self.target_label.strip() or DEFAULT_QSAR_TARGET_COLUMN,
            prediction_column=self.prediction_column.strip() or prediction_column_name_for_target(self.target_label),
            include_input_columns=bool(self.include_input_columns),
            source_mode=source_mode,
        )

        worker = _Worker(self._model, df, config)

        self._progress.setVisible(True)
        self._btn_predict.setEnabled(False)
        self._set_status("Running…")

        sig_done = self._sig_done
        sig_err  = self._sig_err

        def _run():
            try:
                result = worker()
                sig_done.emit(result)
            except Exception:
                sig_err.emit(traceback.format_exc())

        self._future = self._executor.submit(_run)

    # -----------------------------------------------------------------------
    # Thread callbacks (already on the main thread via signal)
    # -----------------------------------------------------------------------

    @pyqtSlot(object)
    def _finish(self, result):
        self._progress.setVisible(False)
        self._btn_predict.setEnabled(True)

        # Send outputs
        self.Outputs.predictions.send(_df_to_table(result.predictions))
        self.Outputs.feature_report.send(_df_to_table(result.feature_report))
        self.Outputs.package_manifest.send(
            _df_to_table(pd.DataFrame([result.package_manifest]))
        )
        self.Outputs.failed_records.send(_df_to_table(result.failed_records))

        manifest = result.package_manifest
        n_predicted = len(result.predictions) if result.predictions is not None else 0
        n_failed = len(result.failed_records) if result.failed_records is not None else 0

        self._set_status(f"Done — {n_predicted} predicted")

        # --- Summary tab ---
        feature_names = manifest.get("feature_names", []) or []
        if isinstance(feature_names, str):
            feature_names = [s.strip() for s in feature_names.split(",") if s.strip()]
        preview = feature_names[:10]
        more = len(feature_names) - len(preview)
        feature_preview = ", ".join(preview) + (f" … (+{more} more)" if more > 0 else "")

        summary_lines = [
            "<b>QSAR Prediction Package Summary</b><hr>",
            f"<b>Model name:</b> {manifest.get('bundle_model_name', manifest.get('model_type', 'N/A'))}",
            f"<b>Model type:</b> {manifest.get('model_type', 'N/A')}",
            f"<b>Source widget:</b> {manifest.get('bundle_source_widget', 'N/A')}",
            f"<b>Predicted records:</b> {n_predicted}",
            f"<b>Failed records:</b> {n_failed}",
            f"<b>Input mode:</b> {manifest.get('source_mode', 'N/A')}",
            f"<b>Training rows:</b> {manifest.get('bundle_training_rows', 'N/A')}",
            f"<b>Features used:</b> {manifest.get('features_used', 'N/A')}",
            f"<b>Selected features:</b> {manifest.get('bundle_selected_feature_count', 'N/A')}",
            f"<b>Feature recipe:</b> {manifest.get('recipe_description', 'N/A')}",
            f"<b>Feature names (first 10):</b> {feature_preview or 'N/A'}",
        ]
        self._summary_browser.setHtml("<br>".join(summary_lines))

        # --- Distribution tab ---
        self._update_histogram(result.predictions, self.prediction_column.strip() or prediction_column_name_for_target(self.target_label))

        # --- Manifest tab ---
        self._populate_manifest(manifest)

    @pyqtSlot(str)
    def _on_error(self, tb: str):
        self._progress.setVisible(False)
        self._btn_predict.setEnabled(True)
        self._send_none_outputs()
        self._set_status("Error", error=True)
        self._summary_browser.setPlainText(f"QSAR Prediction Packager failed:\n\n{tb}")

    # -----------------------------------------------------------------------
    # UI update helpers
    # -----------------------------------------------------------------------

    def _set_status(self, text: str, *, error: bool = False):
        self._status_chip.setText(text)
        self._status_chip.setStyleSheet(_CHIP_ERR if error else _CHIP_OK)

    def _send_none_outputs(self):
        self.Outputs.predictions.send(None)
        self.Outputs.feature_report.send(None)
        self.Outputs.package_manifest.send(None)
        self.Outputs.failed_records.send(None)

    def _sync_labels_from_model(self, model) -> None:
        target_name = infer_target_label_from_model(model, fallback=DEFAULT_QSAR_TARGET_COLUMN)
        if not target_name:
            return
        if self._target_label_autofill:
            self.target_label = target_name
            self._edit_target_label.blockSignals(True)
            self._edit_target_label.setText(target_name)
            self._edit_target_label.blockSignals(False)
        if self._prediction_column_autofill:
            predicted_name = prediction_column_name_for_target(target_name)
            self.prediction_column = predicted_name
            self._edit_prediction_column.blockSignals(True)
            self._edit_prediction_column.setText(predicted_name)
            self._edit_prediction_column.blockSignals(False)
        self._plot.getAxis("bottom").setLabel(f"Predicted {target_name}")

    def _model_target_name(self, model) -> str:
        return infer_target_label_from_model(model, fallback="")

    def _build_input_dataframe(self) -> tuple[pd.DataFrame | None, str | None]:
        if self._query_data is not None:
            return _table_to_df(self._query_data), "query_table"
        if self._molecules:
            return self._molecules_to_df(self._molecules), "molecules_input"
        return None, None

    def _molecules_to_df(self, molecules) -> pd.DataFrame:
        rows: list[dict] = []
        for i, item in enumerate(list(molecules or []), start=1):
            if isinstance(item, ChemMol):
                smiles = item.canonical_smiles()
                compound_id = (
                    item.get_prop("compound_id")
                    or item.get_prop("row_id")
                    or item.name
                    or f"compound_{i:04d}"
                )
                row = {
                    "compound_id": str(compound_id),
                    "canonical_smiles": str(smiles or ""),
                    "name": str(item.name or ""),
                }
                for key, value in (item.props or {}).items():
                    if key in row or isinstance(value, (dict, list, tuple, set)):
                        continue
                    row[str(key)] = value
            else:
                row = {
                    "compound_id": f"compound_{i:04d}",
                    "canonical_smiles": str(item or ""),
                    "name": "",
                }
            rows.append(row)
        return pd.DataFrame(rows)

    def _update_histogram(self, predictions: pd.DataFrame | None, col: str):
        self._plot.clear()
        if predictions is None or col not in predictions.columns:
            return
        values = pd.to_numeric(predictions[col], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            return
        counts, edges = np.histogram(values, bins=min(30, max(5, values.size // 10 + 1)))
        bar = pg.BarGraphItem(
            x=edges[:-1],
            height=counts,
            width=np.diff(edges) * 0.9,
            brush=pg.mkBrush(37, 99, 235, 180),
            pen=pg.mkPen(None),
        )
        self._plot.addItem(bar)

    def _populate_manifest(self, manifest: dict):
        self._manifest_table.setRowCount(0)
        for key, value in manifest.items():
            row = self._manifest_table.rowCount()
            self._manifest_table.insertRow(row)
            self._manifest_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self._manifest_table.setItem(row, 1, QTableWidgetItem(str(value)))

    # -----------------------------------------------------------------------
    def onDeleteWidget(self):
        self._executor.shutdown(wait=False)
        super().onDeleteWidget()
