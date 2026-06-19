from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import pyqtgraph as pg
from AnyQt.QtCore import QTimer, Qt, pyqtSlot as Slot
from AnyQt.QtGui import QColor, QFont
from AnyQt.QtPrintSupport import QPrinter
from AnyQt.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.qsar_report_generator_service import (
    QSARReportConfig,
    generate_qsar_report,
)

pg.setConfigOptions(antialias=True)

# ── colour palette ────────────────────────────────────────────────────────────
_C_TRAIN   = (37,  99, 235, 210)   # blue
_C_TEST    = (234, 88,  12, 210)   # orange
_C_CV      = (22, 163,  74, 210)   # green
_C_GRID    = (200, 200, 200, 60)
_C_DIAG    = (148, 163, 184, 200)  # slate


# ── Orange → pandas (fast numpy path) ────────────────────────────────────────

def _table_to_df(data: Optional[Table]) -> Optional[pd.DataFrame]:
    if data is None:
        return None
    cols: dict = {}
    n = len(data)
    if data.domain.attributes:
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
                try:
                    cols[v.name] = col.astype(float)
                except Exception:
                    cols[v.name] = [str(x) for x in col]
    return pd.DataFrame(cols, index=range(n))


def _df_to_table(df: Optional[pd.DataFrame]) -> Table:
    if df is None or df.empty:
        return Table.from_numpy(Domain([]), X=np.empty((0, 0)))
    attrs, metas, xc, mc = [], [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            attrs.append(ContinuousVariable(str(col))); xc.append(col)
        else:
            metas.append(StringVariable(str(col))); mc.append(col)
    X = df[xc].to_numpy(dtype=float) if xc else np.empty((len(df), 0), dtype=float)
    M = df[mc].fillna("").astype(str).to_numpy(dtype=object) if mc else np.empty((len(df), 0), dtype=object)
    return Table.from_numpy(Domain(attrs, metas=metas), X=X, metas=M)


def _norm_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _detect_column(df: Optional[pd.DataFrame], candidates: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    by_key = {_norm_key(col): col for col in df.columns}
    for cand in candidates:
        key = _norm_key(cand)
        if key in by_key:
            return by_key[key]
    candidate_keys = [_norm_key(cand) for cand in candidates]
    for key, col in by_key.items():
        if any(cand in key for cand in candidate_keys):
            return col
    return None


def _prediction_columns(pred_df: Optional[pd.DataFrame]) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    obs_col = _detect_column(pred_df, ["observed", "y_true", "actual", "experimental", "measured", "reference", "pActivity"])
    pred_col = _detect_column(pred_df, ["predicted", "y_pred", "prediction", "predicted_pActivity", "estimate"])
    split_col = _detect_column(pred_df, ["split", "dataset", "partition", "group", "subset"])
    residual_col = _detect_column(pred_df, ["residual", "error", "prediction_error", "obs_minus_pred"])
    id_col = _detect_column(pred_df, ["compound_id", "id", "name", "molecule_id"])
    return obs_col, pred_col, split_col, residual_col, id_col


def _normalize_split_value(value: object) -> str:
    key = _norm_key(value)
    if key in {"cv", "cross_validation", "cross-validation", "validation"}:
        return "cross_validation"
    if key in {"external", "holdout"}:
        return "test"
    if key.startswith("train"):
        return "train"
    if key.startswith("test"):
        return "test"
    return key


def _split_groups(pred_df: pd.DataFrame, split_col: Optional[str]) -> list[tuple[str, pd.DataFrame]]:
    if split_col is None or split_col not in pred_df.columns:
        return [("all", pred_df)]
    frame = pred_df.copy()
    frame["_split_norm"] = frame[split_col].map(_normalize_split_value)
    groups: list[tuple[str, pd.DataFrame]] = []
    for split_name in ("train", "test", "cross_validation"):
        sub = frame[frame["_split_norm"] == split_name].drop(columns=["_split_norm"])
        if not sub.empty:
            groups.append((split_name, sub))
    if groups:
        return groups
    return [("all", frame.drop(columns=["_split_norm"]))]


def _metric_lookup(metrics: Optional[pd.DataFrame]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    if metrics is None or metrics.empty:
        return out
    cols = {_norm_key(c): c for c in metrics.columns}
    metric_col = cols.get("metric") or cols.get("name")
    value_col = cols.get("value") or cols.get("score")
    split_col = cols.get("group") or cols.get("split") or cols.get("dataset") or cols.get("partition")
    if metric_col and value_col:
        for _, row in metrics.iterrows():
            try:
                value = float(row[value_col])
            except Exception:
                continue
            split = _normalize_split_value(row[split_col]) if split_col else ""
            metric = _norm_key(row[metric_col])
            out[(split, metric)] = value
            out.setdefault(("", metric), value)
        return out
    for col in metrics.columns:
        try:
            value = float(metrics[col].iloc[0])
        except Exception:
            continue
        key = _norm_key(col)
        split = ""
        metric = key
        for split_name in ("train", "test", "cv", "cross_validation", "external", "validation"):
            if key.startswith(split_name + "_"):
                split = _normalize_split_value(split_name)
                metric = key[len(split_name) + 1:]
                break
            if key.endswith("_" + split_name):
                split = _normalize_split_value(split_name)
                metric = key[: -(len(split_name) + 1)]
                break
        out[(split, metric)] = value
        out.setdefault(("", metric), value)
    return out


def _first_metric(lookup: dict[tuple[str, str], float], names: list[str], splits: list[str]) -> Optional[float]:
    aliases: list[str] = []
    for name in names:
        key = _norm_key(name)
        aliases.extend([key, key.replace("2", "_2"), key.replace("_", "")])
    for split in splits:
        split_key = _normalize_split_value(split)
        for alias in aliases:
            if (split_key, alias) in lookup:
                return lookup[(split_key, alias)]
    for alias in aliases:
        if ("", alias) in lookup:
            return lookup[("", alias)]
    return None


def _importance_frame(expl_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if expl_df is None or expl_df.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    feat_col = _detect_column(expl_df, ["feature", "feature_name", "descriptor", "name"])
    imp_col = _detect_column(expl_df, ["importance", "score", "mean_abs_shap", "mean_importance", "coefficient", "normalized_importance", "value"])
    if feat_col and imp_col:
        df = expl_df[[feat_col, imp_col]].copy()
        df.columns = ["feature", "importance"]
        df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
        return df.dropna(subset=["importance"])

    pairs_col = _detect_column(expl_df, ["top_feature_pairs", "top_features"])
    if not pairs_col:
        return pd.DataFrame(columns=["feature", "importance"])
    for raw_value in expl_df[pairs_col].dropna():
        try:
            parsed = ast.literal_eval(str(raw_value))
        except Exception:
            continue
        if isinstance(parsed, list) and parsed:
            rows = []
            for item in parsed:
                if isinstance(item, dict) and "feature" in item:
                    value = item.get("importance", item.get("normalized_importance"))
                    rows.append({"feature": item["feature"], "importance": value})
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    rows.append({"feature": item[0], "importance": item[1]})
            if rows:
                df = pd.DataFrame(rows)
                df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
                return df.dropna(subset=["importance"])
    return pd.DataFrame(columns=["feature", "importance"])


# ── pyqtgraph helpers ─────────────────────────────────────────────────────────

def _styled_plot(title: str = "") -> pg.PlotWidget:
    pw = pg.PlotWidget(title=title)
    pw.setBackground("#FFFFFF")
    pw.getPlotItem().getAxis("left").setPen(pg.mkPen("#CBD5E1"))
    pw.getPlotItem().getAxis("bottom").setPen(pg.mkPen("#CBD5E1"))
    pw.getPlotItem().getAxis("left").setTextPen(pg.mkPen("#475569"))
    pw.getPlotItem().getAxis("bottom").setTextPen(pg.mkPen("#475569"))
    pw.showGrid(x=True, y=True, alpha=0.18)
    return pw


def _scatter(x, y, rgba, labels=None, size=9) -> pg.ScatterPlotItem:
    brush = pg.mkBrush(*rgba)
    kw = dict(x=x, y=y, size=size, pen=pg.mkPen(None), brush=brush, hoverable=True,
              hoverSize=13, hoverBrush=pg.mkBrush(255, 220, 0, 230))
    if labels is not None:
        kw["data"] = labels
    return pg.ScatterPlotItem(**kw)


def _set_numeric_plot_range(plot: pg.PlotWidget, x_values, y_values, *, x_pad_ratio: float = 0.06, y_pad_ratio: float = 0.08) -> None:
    x_arr = np.asarray(x_values, dtype=float).ravel()
    y_arr = np.asarray(y_values, dtype=float).ravel()
    x_arr = x_arr[np.isfinite(x_arr)]
    y_arr = y_arr[np.isfinite(y_arr)]
    if x_arr.size == 0 or y_arr.size == 0:
        return

    x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
    y_min, y_max = float(np.min(y_arr)), float(np.max(y_arr))
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = max(x_span * x_pad_ratio, 0.15 if x_span == 0 else 0.0)
    y_pad = max(y_span * y_pad_ratio, 0.15 if y_span == 0 else 0.0)
    plot.setXRange(x_min - x_pad, x_max + x_pad, padding=0.0)
    plot.setYRange(y_min - y_pad, y_max + y_pad, padding=0.0)


# ── hover info label ──────────────────────────────────────────────────────────

class _HoverLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setStyleSheet(
            "font-size:11px; color:#475569; background:#F8FAFC; "
            "border:1px solid #E2E8F0; border-radius:5px; padding:3px 8px;"
        )
        self.setFixedHeight(24)
        self.setText("Hover over a point for details")

    def update_point(self, label, x, y):
        self.setText(f"  {label}   obs={x:.3f}   pred={y:.3f}   res={y-x:.3f}")

    def clear_point(self):
        self.setText("Hover over a point for details")


# ── widget ────────────────────────────────────────────────────────────────────

class OWQSARReportGenerator(OWWidget):
    name = "QSAR Report Generator"
    description = "Interactive QSAR diagnostics: observed vs predicted, residuals, metrics and feature importance."
    icon = "icons/modeling/ow_qsar_report_generator.png"
    priority = 149
    keywords = ["QSAR", "report", "diagnostics", "graphs", "interactive"]
    want_main_area = True
    resizing_enabled = True

    class Inputs:
        dataset            = Input("Dataset",            Table)
        metrics            = Input("Metrics",            Table)
        predictions        = Input("Predictions",        Table)
        validation_summary = Input("Validation Summary", Table)
        feature_importance = Input("Feature Importance", Table)
        # Compatibility input for QSAR/QSPR Model Hub.
        # Model Hub emits "Model Summary", while the report service historically
        # expected a generic "Validation Summary" table.  Both now feed the same
        # report section; Validation Summary takes precedence if both are connected.
        model_summary      = Input("Model Summary",      Table)
        ad_summary         = Input("AD Summary",         Table)
        explanation_summary= Input("Explanation Summary",Table)

    class Outputs:
        report_markdown  = Output("Report Markdown",  str,   auto_summary=False)
        report_html      = Output("Report HTML",      str,   auto_summary=False)
        report_pdf_path  = Output("Report PDF Path",  str,   auto_summary=False)
        report_sections  = Output("Report Sections",  Table)
        report_summary   = Output("Report Summary",   Table)

    title_text:       str  = Setting("QSAR Studio Report")
    project_name:     str  = Setting("QSAR project")
    author:           str  = Setting("")
    max_preview_rows: int  = Setting(12)
    auto_run:         bool = Setting(True)

    def __init__(self) -> None:
        super().__init__()
        self._data: dict = {k: None for k in
            ("dataset", "metrics", "predictions",
             "validation_summary", "feature_importance", "model_summary",
             "ad_summary", "explanation_summary")}
        self._executor = ThreadExecutor(self)
        self._last_report_html: str = ""
        self._last_report_markdown: str = ""
        self._latest_plot_frames: dict[str, Optional[pd.DataFrame]] = {
            "predictions": None,
            "metrics": None,
            "importance": None,
        }

        self._build_control_area()
        self._build_main_area()
        self._set_status("Connect QSAR output tables.", ok=True)

    # ── control area ──────────────────────────────────────────────────────────

    def _build_control_area(self) -> None:
        ca = self.controlArea

        hdr = QWidget()
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(6, 6, 6, 6)
        left = QVBoxLayout()
        left.addWidget(QLabel("QSAR Report", objectName="HdrTitle"))
        left.addWidget(QLabel("Interactive diagnostics & reproducible report", objectName="HdrSub"))
        hl.addLayout(left, 1)
        self._lbl_status = QLabel("Ready", objectName="StatusChip")
        self._lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hl.addWidget(self._lbl_status)
        ca.layout().addWidget(hdr)

        box = QGroupBox("Report metadata")
        vl = QVBoxLayout(box)
        for attr, label in [("title_text", "Title"), ("project_name", "Project"), ("author", "Author")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            ed = QLineEdit(getattr(self, attr))
            ed.textChanged.connect(lambda t, a=attr: (setattr(self, a, t), self._maybe_commit()))
            row.addWidget(ed, 1)
            vl.addLayout(row)

        row_p = QHBoxLayout()
        row_p.addWidget(QLabel("Preview rows"))
        sp = QSpinBox(); sp.setRange(3, 50); sp.setValue(self.max_preview_rows)
        sp.valueChanged.connect(lambda v: setattr(self, "max_preview_rows", v))
        row_p.addWidget(sp, 1)
        vl.addLayout(row_p)

        self._chk_auto = QCheckBox("Auto-run")
        self._chk_auto.setChecked(bool(self.auto_run))
        self._chk_auto.toggled.connect(lambda v: setattr(self, "auto_run", bool(v)))
        vl.addWidget(self._chk_auto)

        self._btn_run = QPushButton("Generate report")
        self._btn_run.clicked.connect(self.commit)
        vl.addWidget(self._btn_run)

        self._btn_pdf = QPushButton("Export PDF")
        self._btn_pdf.setEnabled(False)
        self._btn_pdf.clicked.connect(self.export_pdf)
        vl.addWidget(self._btn_pdf)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        vl.addWidget(self._progress)

        ca.layout().addWidget(box)
        ca.layout().addStretch(1)

    # ── main area ─────────────────────────────────────────────────────────────

    def _build_main_area(self) -> None:
        self._tabs = QTabWidget()
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Tab 0 – HTML report
        self._report_browser = QTextBrowser()
        self._report_browser.setOpenExternalLinks(True)
        self._tabs.addTab(self._report_browser, "Report")

        # Tab 1 – Observed vs Predicted
        self._tabs.addTab(self._build_obs_pred_tab(), "Obs vs Pred")

        # Tab 2 – Residuals
        self._tabs.addTab(self._build_residuals_tab(), "Residuals")

        # Tab 3 – Metrics
        self._tabs.addTab(self._build_metrics_tab(), "Metrics")

        # Tab 4 – Feature importance
        self._tabs.addTab(self._build_importance_tab(), "Feature Importance")

        self.mainArea.layout().addWidget(self._tabs)

    def _build_obs_pred_tab(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)
        self._op_plot = _styled_plot()
        self._op_plot.setLabel("left",   "Predicted")
        self._op_plot.setLabel("bottom", "Observed")
        self._op_hover = _HoverLabel()

        legend = self._make_legend([
            ("Train", _C_TRAIN), ("Test", _C_TEST), ("CV", _C_CV)
        ])
        vl.addWidget(legend)
        vl.addWidget(self._op_plot, 1)
        vl.addWidget(self._op_hover)
        return w

    def _build_residuals_tab(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)
        self._res_plot = _styled_plot()
        self._res_plot.setLabel("left",   "Residual (obs − pred)")
        self._res_plot.setLabel("bottom", "Predicted")
        self._res_hover = _HoverLabel()

        legend = self._make_legend([("Train", _C_TRAIN), ("Test", _C_TEST), ("CV", _C_CV)])
        vl.addWidget(legend)
        vl.addWidget(self._res_plot, 1)
        vl.addWidget(self._res_hover)
        return w

    def _build_metrics_tab(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)
        self._met_r2   = _styled_plot("R²")
        self._met_rmse = _styled_plot("RMSE")
        self._met_mae  = _styled_plot("MAE")
        for p in (self._met_r2, self._met_rmse, self._met_mae):
            p.setMaximumHeight(200)
        vl.addWidget(self._met_r2)
        vl.addWidget(self._met_rmse)
        vl.addWidget(self._met_mae)
        return w

    def _build_importance_tab(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)
        self._imp_plot = _styled_plot("Feature Importance")
        self._imp_plot.setLabel("left",   "Feature")
        self._imp_plot.setLabel("bottom", "Importance score")
        vl.addWidget(self._imp_plot, 1)
        self._imp_placeholder = QLabel(
            "Connect Explanation Summary to see feature importance.")
        self._imp_placeholder.setAlignment(Qt.AlignCenter)
        self._imp_placeholder.setStyleSheet("color:#94A3B8; font-size:13px; padding:30px;")
        vl.addWidget(self._imp_placeholder)
        return w

    @staticmethod
    def _make_legend(items: list) -> QWidget:
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(6, 2, 6, 2)
        hl.setSpacing(18)
        for label, rgba in items:
            dot = QLabel("●")
            r, g, b, _ = rgba
            dot.setStyleSheet(f"color: rgb({r},{g},{b}); font-size:18px;")
            txt = QLabel(label)
            txt.setStyleSheet("font-size:12px; color:#475569;")
            hl.addWidget(dot)
            hl.addWidget(txt)
        hl.addStretch(1)
        return w

    # ── status ────────────────────────────────────────────────────────────────

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
        self._btn_pdf.setEnabled((not busy) and bool(self._last_report_html.strip()))
        self._progress.setVisible(busy)

    # ── inputs ────────────────────────────────────────────────────────────────

    @Inputs.dataset
    def set_dataset(self, d): self._data["dataset"] = d; self._maybe_commit()

    @Inputs.metrics
    def set_metrics(self, d): self._data["metrics"] = d; self._maybe_commit()

    @Inputs.predictions
    def set_predictions(self, d): self._data["predictions"] = d; self._maybe_commit()

    @Inputs.validation_summary
    def set_validation_summary(self, d): self._data["validation_summary"] = d; self._maybe_commit()

    @Inputs.feature_importance
    def set_feature_importance(self, d): self._data["feature_importance"] = d; self._maybe_commit()

    @Inputs.model_summary
    def set_model_summary(self, d): self._data["model_summary"] = d; self._maybe_commit()

    @Inputs.ad_summary
    def set_ad_summary(self, d): self._data["ad_summary"] = d; self._maybe_commit()

    @Inputs.explanation_summary
    def set_explanation_summary(self, d): self._data["explanation_summary"] = d; self._maybe_commit()

    def _maybe_commit(self):
        if self.auto_run:
            self.commit()

    # ── commit (async) ────────────────────────────────────────────────────────

    def commit(self) -> None:
        snapshot = {k: _table_to_df(v) for k, v in self._data.items()}
        # The report generator supports both the generic Validation Summary
        # and the QSAR/QSPR Model Hub specific Model Summary.  This avoids a
        # dead-end workflow where Model Hub outputs cannot be connected to a
        # matching report input.
        if snapshot.get("validation_summary") is None:
            snapshot["validation_summary"] = snapshot.get("model_summary")
        if snapshot.get("explanation_summary") is None:
            snapshot["explanation_summary"] = snapshot.get("feature_importance")
        cfg = QSARReportConfig(
            title=self.title_text.strip() or "QSAR Studio Report",
            project_name=self.project_name.strip() or "QSAR project",
            author=self.author.strip(),
            max_preview_rows=int(self.max_preview_rows),
        )
        self._set_busy(True)
        self._set_status("Generating…", ok=True)

        def _run():
            result = generate_qsar_report(
                dataset=snapshot["dataset"],
                metrics=snapshot["metrics"],
                predictions=snapshot["predictions"],
                validation_summary=snapshot["validation_summary"],
                ad_summary=snapshot["ad_summary"],
                explanation_summary=snapshot["explanation_summary"],
                config=cfg,
            )
            return result, snapshot

        fut = self._executor.submit(_run)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut) -> None:
        try:
            result, snapshot = fut.result()
            methodinvoke(self, "_finish", (object,))((result, snapshot))
        except Exception as e:
            methodinvoke(self, "_fail", (str,))(str(e))

    @Slot(object)
    def _finish(self, payload: object) -> None:
        result, snapshot = payload
        self._set_busy(False)
        self._last_report_markdown = result.markdown or ""
        self._last_report_html = result.html or f"<pre>{result.markdown}</pre>"

        self.Outputs.report_markdown.send(self._last_report_markdown)
        self.Outputs.report_html.send(self._last_report_html)
        self.Outputs.report_sections.send(_df_to_table(result.sections))
        self.Outputs.report_summary.send(_df_to_table(pd.DataFrame([result.summary])))
        self.Outputs.report_pdf_path.send(None)

        # Render HTML report
        self._report_browser.setHtml(self._last_report_html)

        # Update plots
        pred_df  = snapshot.get("predictions")
        met_df   = snapshot.get("metrics")
        expl_df  = snapshot.get("explanation_summary")
        self._latest_plot_frames["predictions"] = pred_df
        self._latest_plot_frames["metrics"] = met_df
        self._latest_plot_frames["importance"] = expl_df

        self._update_obs_pred(pred_df)
        self._update_residuals(pred_df)
        self._update_metrics(met_df)
        self._update_importance(expl_df)
        QTimer.singleShot(0, self._refresh_current_tab)

        n_sec = result.summary.get("sections_created", "?")
        missing = result.summary.get("sections_missing", 0)
        status = f"{n_sec} sections · report ready"
        if missing:
            status += f" · {missing} missing"
        self._set_status(status, ok=True)

    @Slot(str)
    def _fail(self, msg: str) -> None:
        self._set_busy(False)
        self._set_status("Error", ok=False)
        self._btn_pdf.setEnabled(bool(self._last_report_html.strip()))
        self._report_browser.setPlainText(f"Report generation failed:\n\n{msg}")

    def _on_tab_changed(self, index: int) -> None:
        QTimer.singleShot(0, self._refresh_current_tab)

    def _refresh_current_tab(self) -> None:
        idx = int(self._tabs.currentIndex())
        pred_df = self._latest_plot_frames.get("predictions")
        met_df = self._latest_plot_frames.get("metrics")
        expl_df = self._latest_plot_frames.get("importance")
        if idx == 1:
            self._update_obs_pred(pred_df)
        elif idx == 2:
            self._update_residuals(pred_df)
        elif idx == 3:
            self._update_metrics(met_df)
        elif idx == 4:
            self._update_importance(expl_df)

    def _default_pdf_name(self) -> str:
        title = self.title_text.strip() or "qsar_report"
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in title).strip("_")
        return f"{safe or 'qsar_report'}.pdf"

    def _save_report_pdf(self, filename: str) -> str:
        html = self._last_report_html.strip()
        if not html:
            raise ValueError("No report available for PDF export.")
        out_path = Path(filename)
        if out_path.suffix.lower() != ".pdf":
            out_path = out_path.with_suffix(".pdf")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(str(out_path))
        printer.setPageMargins(12, 12, 12, 12, QPrinter.Millimeter)
        self._report_browser.document().print_(printer)
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise IOError(f"PDF export failed for {out_path}")
        return str(out_path)

    def export_pdf(self) -> None:
        if not self._last_report_html.strip():
            self._set_status("No report available for PDF export.", ok=False)
            self.Outputs.report_pdf_path.send(None)
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export PDF",
            self._default_pdf_name(),
            "PDF Files (*.pdf)",
        )
        if not filename:
            return
        try:
            saved_path = self._save_report_pdf(filename)
            self.Outputs.report_pdf_path.send(saved_path)
            self._set_status(f"PDF exported · {Path(saved_path).name}", ok=True)
        except Exception as exc:
            self.Outputs.report_pdf_path.send(None)
            self._set_status(f"Error exporting PDF: {exc}", ok=False)

    # ── plot updaters ─────────────────────────────────────────────────────────

    def _update_obs_pred(self, pred_df: Optional[pd.DataFrame]) -> None:
        self._op_plot.clear()
        if pred_df is None or pred_df.empty:
            return
        obs_col, pred_col, split_col, _residual_col, id_col = _prediction_columns(pred_df)
        if obs_col is None or pred_col is None:
            return

        split_colors = {
            "train": _C_TRAIN,
            "test": _C_TEST,
            "cross_validation": _C_CV,
            "all": _C_TEST,
        }
        numeric = pred_df.copy()
        numeric[obs_col] = pd.to_numeric(numeric[obs_col], errors="coerce")
        numeric[pred_col] = pd.to_numeric(numeric[pred_col], errors="coerce")

        for split, sub in _split_groups(numeric, split_col):
            rgba = split_colors.get(split, _C_TEST)
            sub = sub.dropna(subset=[obs_col, pred_col])
            if sub.empty:
                continue
            obs = sub[obs_col].to_numpy(dtype=float)
            pred = sub[pred_col].to_numpy(dtype=float)
            ids = (
                sub[id_col].astype(str).to_numpy()
                if id_col and id_col in sub.columns
                else np.arange(len(sub)).astype(str)
            )
            sc = _scatter(obs, pred, rgba, labels=ids)
            sc.sigHovered.connect(
                lambda _item, points, ev, hover_lbl=self._op_hover: self._on_hover_op(points, ev, hover_lbl)
            )
            self._op_plot.addItem(sc)

        # diagonal y=x line
        valid = numeric[[obs_col, pred_col]].dropna()
        if valid.empty:
            return
        all_vals = np.concatenate([valid[obs_col].to_numpy(dtype=float), valid[pred_col].to_numpy(dtype=float)])
        mn, mx = np.nanmin(all_vals), np.nanmax(all_vals)
        pad = (mx - mn) * 0.05 if mx > mn else 0.5
        diag = pg.PlotDataItem([mn - pad, mx + pad], [mn - pad, mx + pad],
                                pen=pg.mkPen(_C_DIAG, width=1.5, style=Qt.DashLine))
        self._op_plot.addItem(diag)
        self._op_plot.setXRange(mn - pad, mx + pad, padding=0.0)
        self._op_plot.setYRange(mn - pad, mx + pad, padding=0.0)

    def _update_residuals(self, pred_df: Optional[pd.DataFrame]) -> None:
        self._res_plot.clear()
        if pred_df is None or pred_df.empty:
            return
        obs_col, pred_col, split_col, residual_col, id_col = _prediction_columns(pred_df)
        if obs_col is None or pred_col is None:
            return

        split_colors = {
            "train": _C_TRAIN,
            "test": _C_TEST,
            "cross_validation": _C_CV,
            "all": _C_TEST,
        }
        numeric = pred_df.copy()
        numeric[pred_col] = pd.to_numeric(numeric[pred_col], errors="coerce")
        if residual_col and residual_col in numeric.columns:
            numeric[residual_col] = pd.to_numeric(numeric[residual_col], errors="coerce")
        else:
            numeric[obs_col] = pd.to_numeric(numeric[obs_col], errors="coerce")
            residual_col = "__residual__"
            numeric[residual_col] = numeric[obs_col] - numeric[pred_col]

        for split, sub in _split_groups(numeric, split_col):
            rgba = split_colors.get(split, _C_TEST)
            sub = sub.dropna(subset=[pred_col, residual_col])
            if sub.empty:
                continue
            pred = sub[pred_col].to_numpy(dtype=float)
            res = sub[residual_col].to_numpy(dtype=float)
            ids = (
                sub[id_col].astype(str).to_numpy()
                if id_col and id_col in sub.columns
                else np.arange(len(sub)).astype(str)
            )
            sc = _scatter(pred, res, rgba, labels=ids)
            sc.sigHovered.connect(
                lambda _item, points, ev, hl=self._res_hover: self._on_hover_res(points, ev, hl)
            )
            self._res_plot.addItem(sc)

        # zero line
        valid_pred = numeric[pred_col].dropna()
        if valid_pred.empty:
            return
        residual_vals = numeric[residual_col].dropna()
        if residual_vals.empty:
            return
        mn, mx = float(valid_pred.min()), float(valid_pred.max())
        self._res_plot.addItem(
            pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(_C_DIAG, width=1, style=Qt.DashLine))
        )
        _set_numeric_plot_range(
            self._res_plot,
            valid_pred.to_numpy(dtype=float),
            residual_vals.to_numpy(dtype=float),
        )

    def _update_metrics(self, met_df: Optional[pd.DataFrame]) -> None:
        for plot in (self._met_r2, self._met_rmse, self._met_mae):
            plot.clear()
        if met_df is None or met_df.empty:
            return
        lookup = _metric_lookup(met_df)
        if not lookup:
            return

        groups = ["train", "test", "cross_validation"]
        labels = ["Train", "Test", "CV"]
        colors = [_C_TRAIN, _C_TEST, _C_CV]

        metric_aliases = {
            "r2": ["r2", "q2", "r_2", "q_2", "cv_r2"],
            "rmse": ["rmse"],
            "mae": ["mae"],
        }
        for plot, key in [(self._met_r2, "r2"), (self._met_rmse, "rmse"), (self._met_mae, "mae")]:
            used_values: list[float] = []
            plot.getPlotItem().getAxis("bottom").setTicks([
                [(i, lbl) for i, lbl in enumerate(labels)]
            ])
            for i, (grp, _lbl, rgba) in enumerate(zip(groups, labels, colors)):
                val = _first_metric(lookup, metric_aliases[key], [grp])
                if val is None:
                    continue
                if not np.isfinite(val):
                    continue
                used_values.append(float(val))
                bar = pg.BarGraphItem(x=[i], height=[val], width=0.5,
                                      brush=pg.mkBrush(*rgba),
                                      pen=pg.mkPen(None))
                plot.addItem(bar)
                # value label
                txt = pg.TextItem(f"{val:.3f}", anchor=(0.5, 0), color="#0F172A")
                txt.setFont(QFont("Arial", 9))
                txt.setPos(i, val)
                plot.addItem(txt)
            if used_values:
                y_min = min(0.0, float(min(used_values)))
                y_max = float(max(used_values))
                y_pad = max((y_max - y_min) * 0.12, 0.1)
                plot.setXRange(-0.6, len(labels) - 0.4, padding=0.0)
                plot.setYRange(y_min - y_pad * 0.25, y_max + y_pad, padding=0.0)

    def _update_importance(self, expl_df: Optional[pd.DataFrame]) -> None:
        self._imp_plot.clear()
        if expl_df is None or expl_df.empty:
            self._imp_placeholder.setText("Connect Explanation Summary or Feature Importance to see top descriptors.")
            self._imp_placeholder.setVisible(True)
            return
        df = _importance_frame(expl_df)
        if df.empty:
            self._imp_placeholder.setText("Could not detect feature importance rows in the supplied explanation table.")
            self._imp_placeholder.setVisible(True)
            return
        self._imp_placeholder.setVisible(False)
        df = df.sort_values("importance", key=lambda s: s.abs(), ascending=False).head(25)

        names = df["feature"].astype(str).tolist()
        values = df["importance"].to_numpy(dtype=float)
        n = len(names)

        # horizontal bars
        bar = pg.BarGraphItem(x1=np.zeros(n), x0=values[::-1],
                               y=np.arange(n), height=0.65,
                               brush=pg.mkBrush(*_C_TRAIN),
                               pen=pg.mkPen(None))
        self._imp_plot.addItem(bar)
        self._imp_plot.getPlotItem().getAxis("left").setTicks(
            [[(i, nm) for i, nm in enumerate(names[::-1])]]
        )
        self._imp_plot.setYRange(-0.5, n - 0.5)
        x_abs_max = float(np.max(np.abs(values))) if len(values) else 0.0
        x_pad = max(x_abs_max * 0.08, 0.1)
        self._imp_plot.setXRange(min(0.0, float(np.min(values))) - x_pad, max(0.0, float(np.max(values))) + x_pad, padding=0.0)

    # ── hover handlers ────────────────────────────────────────────────────────

    @staticmethod
    def _on_hover_op(points, ev, hover_lbl: _HoverLabel) -> None:
        if points:
            p = points[0]
            lbl = str(p.data()) if p.data() is not None else "?"
            hover_lbl.update_point(lbl, p.pos().x(), p.pos().y())
        else:
            hover_lbl.clear_point()

    @staticmethod
    def _on_hover_res(points, ev, hover_lbl: _HoverLabel) -> None:
        if points:
            p = points[0]
            lbl = str(p.data()) if p.data() is not None else "?"
            x, y = p.pos().x(), p.pos().y()
            hover_lbl.setText(f"  {lbl}   pred={x:.3f}   residual={y:.3f}")
        else:
            hover_lbl.clear_point()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._refresh_current_tab)

    def onDeleteWidget(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWQSARReportGenerator).run()
