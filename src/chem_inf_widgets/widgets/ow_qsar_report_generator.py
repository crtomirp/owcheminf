from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import pyqtgraph as pg
from AnyQt.QtCore import Qt, pyqtSlot as Slot
from AnyQt.QtGui import QColor, QFont
from AnyQt.QtWidgets import (
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
        ad_summary         = Input("AD Summary",         Table)
        explanation_summary= Input("Explanation Summary",Table)

    class Outputs:
        report_markdown  = Output("Report Markdown",  str,   auto_summary=False)
        report_html      = Output("Report HTML",      str,   auto_summary=False)
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
             "validation_summary", "ad_summary", "explanation_summary")}
        self._executor = ThreadExecutor(self)

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

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        vl.addWidget(self._progress)

        ca.layout().addWidget(box)
        ca.layout().addStretch(1)

    # ── main area ─────────────────────────────────────────────────────────────

    def _build_main_area(self) -> None:
        self._tabs = QTabWidget()

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
            ("Train", _C_TRAIN), ("Test", _C_TEST)
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

        legend = self._make_legend([("Train", _C_TRAIN), ("Test", _C_TEST)])
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

        self.Outputs.report_markdown.send(result.markdown)
        self.Outputs.report_html.send(result.html)
        self.Outputs.report_sections.send(_df_to_table(result.sections))
        self.Outputs.report_summary.send(_df_to_table(pd.DataFrame([result.summary])))

        # Render HTML report
        self._report_browser.setHtml(result.html or f"<pre>{result.markdown}</pre>")

        # Update plots
        pred_df  = snapshot.get("predictions")
        met_df   = snapshot.get("metrics")
        expl_df  = snapshot.get("explanation_summary")

        self._update_obs_pred(pred_df)
        self._update_residuals(pred_df)
        self._update_metrics(met_df)
        self._update_importance(expl_df)

        n_sec = result.summary.get("sections_created", "?")
        self._set_status(f"{n_sec} sections · report ready", ok=True)

    @Slot(str)
    def _fail(self, msg: str) -> None:
        self._set_busy(False)
        self._set_status("Error", ok=False)
        self._report_browser.setPlainText(f"Report generation failed:\n\n{msg}")

    # ── plot updaters ─────────────────────────────────────────────────────────

    def _update_obs_pred(self, pred_df: Optional[pd.DataFrame]) -> None:
        self._op_plot.clear()
        if pred_df is None or pred_df.empty:
            return
        if not {"observed", "predicted"}.issubset(pred_df.columns):
            return

        for split, rgba in [("train", _C_TRAIN), ("test", _C_TEST)]:
            sub = pred_df[pred_df.get("split", pd.Series(["test"] * len(pred_df))) == split] \
                if "split" in pred_df.columns else pred_df
            if sub.empty:
                continue
            obs  = sub["observed"].values.astype(float)
            pred = sub["predicted"].values.astype(float)
            ids  = sub["compound_id"].values if "compound_id" in sub.columns else np.arange(len(sub)).astype(str)
            sc = _scatter(obs, pred, rgba, labels=ids)
            sc.sigHovered.connect(
                lambda pts, ev, hover_lbl=self._op_hover: self._on_hover_op(pts, ev, hover_lbl)
            )
            self._op_plot.addItem(sc)

        # diagonal y=x line
        all_vals = np.concatenate([pred_df["observed"].values, pred_df["predicted"].values])
        mn, mx = np.nanmin(all_vals), np.nanmax(all_vals)
        pad = (mx - mn) * 0.05 if mx > mn else 0.5
        diag = pg.PlotDataItem([mn - pad, mx + pad], [mn - pad, mx + pad],
                                pen=pg.mkPen(_C_DIAG, width=1.5, style=Qt.DashLine))
        self._op_plot.addItem(diag)

    def _update_residuals(self, pred_df: Optional[pd.DataFrame]) -> None:
        self._res_plot.clear()
        if pred_df is None or pred_df.empty:
            return
        if not {"observed", "predicted"}.issubset(pred_df.columns):
            return

        for split, rgba in [("train", _C_TRAIN), ("test", _C_TEST)]:
            sub = pred_df[pred_df["split"] == split] if "split" in pred_df.columns else pred_df
            if sub.empty:
                continue
            pred = sub["predicted"].values.astype(float)
            res  = (sub["observed"].values - sub["predicted"].values).astype(float)
            ids  = sub["compound_id"].values if "compound_id" in sub.columns else np.arange(len(sub)).astype(str)
            sc = _scatter(pred, res, rgba, labels=ids)
            sc.sigHovered.connect(
                lambda pts, ev, hl=self._res_hover: self._on_hover_res(pts, ev, hl)
            )
            self._res_plot.addItem(sc)

        # zero line
        mn, mx = pred_df["predicted"].min(), pred_df["predicted"].max()
        self._res_plot.addItem(
            pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(_C_DIAG, width=1, style=Qt.DashLine))
        )

    def _update_metrics(self, met_df: Optional[pd.DataFrame]) -> None:
        for plot in (self._met_r2, self._met_rmse, self._met_mae):
            plot.clear()
        if met_df is None or met_df.empty:
            return
        if not {"group", "metric", "value"}.issubset(met_df.columns):
            return

        groups = ["train", "test", "cross_validation"]
        labels = ["Train", "Test", "CV"]
        colors = [_C_TRAIN, _C_TEST, _C_CV]

        for plot, key in [(self._met_r2, "r2"), (self._met_rmse, "rmse"), (self._met_mae, "mae")]:
            plot.getPlotItem().getAxis("bottom").setTicks([
                [(i, lbl) for i, lbl in enumerate(labels)]
            ])
            for i, (grp, lbl, rgba) in enumerate(zip(groups, labels, colors)):
                rows = met_df[met_df["group"] == grp]
                # find any row whose metric name ends with key
                row = rows[rows["metric"].str.endswith(key)]
                if row.empty:
                    continue
                val = float(row.iloc[0]["value"])
                if not np.isfinite(val):
                    continue
                bar = pg.BarGraphItem(x=[i], height=[val], width=0.5,
                                      brush=pg.mkBrush(*rgba),
                                      pen=pg.mkPen(None))
                plot.addItem(bar)
                # value label
                txt = pg.TextItem(f"{val:.3f}", anchor=(0.5, 0), color="#0F172A")
                txt.setFont(QFont("Arial", 9))
                txt.setPos(i, val)
                plot.addItem(txt)

    def _update_importance(self, expl_df: Optional[pd.DataFrame]) -> None:
        self._imp_plot.clear()
        if expl_df is None or expl_df.empty:
            self._imp_placeholder.setVisible(True)
            return
        self._imp_placeholder.setVisible(False)

        # Try common column names for feature / importance
        feat_col = next((c for c in expl_df.columns
                         if c.lower() in {"feature", "feature_name", "descriptor"}), None)
        imp_col  = next((c for c in expl_df.columns
                         if c.lower() in {"importance", "score", "mean_abs_shap",
                                          "mean_importance", "coefficient"}), None)
        if feat_col is None or imp_col is None:
            self._imp_placeholder.setText("Could not detect feature/importance columns.")
            self._imp_placeholder.setVisible(True)
            return

        df = expl_df[[feat_col, imp_col]].dropna().copy()
        df[imp_col] = pd.to_numeric(df[imp_col], errors="coerce")
        df = df.dropna().sort_values(imp_col, ascending=False).head(25)
        if df.empty:
            return

        names  = df[feat_col].tolist()
        values = df[imp_col].values.astype(float)
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

    def onDeleteWidget(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWQSARReportGenerator).run()
