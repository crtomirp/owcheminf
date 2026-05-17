from __future__ import annotations

import numpy as np
import pandas as pd
from AnyQt.QtWidgets import QTextEdit
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.ad_workbench_service import (
    ADWorkbenchConfig,
    evaluate_applicability_domain_workbench,
)


def _orange_table_to_dataframe(data: Table) -> pd.DataFrame:
    rows = []
    variables = list(data.domain.attributes) + list(data.domain.class_vars) + list(data.domain.metas)
    for inst in data:
        row = {}
        for var in variables:
            val = inst[var]
            try:
                if isinstance(var, StringVariable):
                    row[var.name] = str(val)
                else:
                    row[var.name] = float(val) if not np.isnan(float(val)) else np.nan
            except Exception:
                row[var.name] = str(val)
        rows.append(row)
    return pd.DataFrame(rows)


def _dataframe_to_orange(df: pd.DataFrame) -> Table:
    if df is None or df.empty:
        return Table.from_numpy(Domain([]), X=np.empty((0, 0)))
    attrs = []
    metas = []
    x_cols = []
    m_cols = []
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            metas.append(StringVariable(str(col)))
            m_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            attrs.append(ContinuousVariable(str(col)))
            x_cols.append(col)
        else:
            metas.append(StringVariable(str(col)))
            m_cols.append(col)
    X = df[x_cols].to_numpy(dtype=float) if x_cols else np.empty((len(df), 0), dtype=float)
    M = df[m_cols].fillna("").astype(str).to_numpy(dtype=object) if m_cols else np.empty((len(df), 0), dtype=object)
    return Table.from_numpy(Domain(attrs, metas=metas), X=X, metas=M)


class OWADWorkbench(OWWidget):
    name = "Applicability Domain Workbench"
    description = "Professional QSAR applicability domain analysis with Williams, kNN, and Mahalanobis diagnostics."
    icon = "icons/analysis/owapplicabilitydomainwidget.svg"
    priority = 145
    keywords = ["QSAR", "applicability domain", "AD", "Williams", "kNN", "Mahalanobis"]

    class Inputs:
        data = Input("Query Data", Table)
        reference_data = Input("Reference Data", Table)

    class Outputs:
        data_results = Output("AD Results", Table)
        reference_results = Output("Reference Results", Table)
        out_of_domain = Output("Out-of-Domain Records", Table)
        summary = Output("AD Summary", Table)
        method_details = Output("Method Details", Table)

    want_main_area = True

    id_column = Setting("compound_id")
    use_williams = Setting(True)
    use_knn = Setting(True)
    use_mahalanobis = Setting(False)
    combine_mode_index = Setting(0)
    knn_k = Setting(5)
    knn_quantile = Setting(0.95)
    maha_alpha = Setting(0.95)
    auto_run = Setting(False)

    def __init__(self):
        super().__init__()
        self.data = None
        self.reference_data = None

        box = gui.widgetBox(self.controlArea, "Input")
        gui.lineEdit(box, self, "id_column", label="ID column:", callback=self._settings_changed)

        method_box = gui.widgetBox(self.controlArea, "Domain methods")
        gui.checkBox(method_box, self, "use_williams", "Williams leverage", callback=self._settings_changed)
        gui.checkBox(method_box, self, "use_knn", "kNN distance", callback=self._settings_changed)
        gui.spin(method_box, self, "knn_k", minv=1, maxv=50, step=1, label="kNN k:", callback=self._settings_changed)
        gui.doubleSpin(method_box, self, "knn_quantile", minv=0.50, maxv=0.999, step=0.01, label="kNN quantile:", callback=self._settings_changed)
        gui.checkBox(method_box, self, "use_mahalanobis", "Mahalanobis distance", callback=self._settings_changed)
        gui.doubleSpin(method_box, self, "maha_alpha", minv=0.50, maxv=0.999, step=0.01, label="Mahalanobis alpha:", callback=self._settings_changed)
        gui.comboBox(method_box, self, "combine_mode_index", label="Combine methods:", items=["and", "or"], callback=self._settings_changed)
        gui.checkBox(method_box, self, "auto_run", "Auto-run", callback=self._settings_changed)
        gui.button(method_box, self, "Evaluate AD", callback=self.commit)

        self.info_label = gui.label(self.controlArea, self, "No data.")
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(280)
        self.mainArea.layout().addWidget(self.report)

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.info_label.setText("Query data received." if data is not None else "No query data.")
        if self.auto_run:
            self.commit()

    @Inputs.reference_data
    def set_reference_data(self, data):
        self.reference_data = data
        if self.auto_run:
            self.commit()

    def _settings_changed(self):
        if self.auto_run and self.data is not None:
            self.commit()

    def _send_none(self):
        self.Outputs.data_results.send(None)
        self.Outputs.reference_results.send(None)
        self.Outputs.out_of_domain.send(None)
        self.Outputs.summary.send(None)
        self.Outputs.method_details.send(None)

    def commit(self):
        if self.data is None and self.reference_data is None:
            self._send_none()
            self.report.setPlainText("No input data.")
            return
        try:
            query_table = self.data if self.data is not None else self.reference_data
            ref_table = self.reference_data if self.reference_data is not None else self.data
            query_df = _orange_table_to_dataframe(query_table)
            ref_df = _orange_table_to_dataframe(ref_table)
            cfg = ADWorkbenchConfig(
                id_column=self.id_column.strip() or "compound_id",
                combine_mode=["and", "or"][int(self.combine_mode_index)],
                use_williams=bool(self.use_williams),
                use_knn=bool(self.use_knn),
                use_mahalanobis=bool(self.use_mahalanobis),
                knn_k=int(self.knn_k),
                knn_quantile=float(self.knn_quantile),
                maha_alpha=float(self.maha_alpha),
            )
            result = evaluate_applicability_domain_workbench(ref_df, query_df, cfg)
            self.Outputs.data_results.send(_dataframe_to_orange(result.scored_query))
            self.Outputs.reference_results.send(_dataframe_to_orange(result.scored_reference))
            self.Outputs.out_of_domain.send(_dataframe_to_orange(result.out_of_domain))
            self.Outputs.summary.send(_dataframe_to_orange(result.summary))
            self.Outputs.method_details.send(_dataframe_to_orange(result.method_details))
            self.info_label.setText(
                f"AD complete: {result.summary_dict['query_in_domain']} in-domain, {result.summary_dict['query_out_of_domain']} out-of-domain"
            )
            self.report.setPlainText(self._format_report(result))
        except Exception as exc:
            self._send_none()
            self.info_label.setText(f"Failed: {exc}")
            self.report.setPlainText(f"Applicability Domain Workbench failed:\n\n{exc}")

    def _format_report(self, result) -> str:
        lines = [
            "Applicability Domain Workbench Report",
            "======================================",
            f"Reference rows: {result.summary_dict['reference_rows']}",
            f"Query rows: {result.summary_dict['query_rows']}",
            f"Features: {result.summary_dict['feature_count']}",
            f"Query in domain: {result.summary_dict['query_in_domain']}",
            f"Query out of domain: {result.summary_dict['query_out_of_domain']}",
            f"Williams h*: {result.summary_dict['h_star']:.4g}",
            "",
            "Feature preview:",
        ]
        lines.extend([f"  - {f}" for f in result.feature_names[:25]])
        return "\n".join(lines)
