from __future__ import annotations

from typing import List, Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLabel, QVBoxLayout, QWidget
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.qsar.mlr_selection import ADConfig
from chem_inf_widgets.chemcore.services.applicability_domain_service import (
    ApplicabilityDomainFit,
    ApplicabilityDomainPrediction,
    fit_applicability_domain,
    score_applicability_domain,
)
from chem_inf_widgets.widgets.ui_helpers import format_error_status, set_widget_error


def _continuous_feature_names(data: Table) -> List[str]:
    return [
        var.name
        for var in data.domain.attributes
        if getattr(var, "is_continuous", False)
    ]


def _common_continuous_feature_names(reference: Table, query: Table) -> List[str]:
    query_names = set(_continuous_feature_names(query))
    return [name for name in _continuous_feature_names(reference) if name in query_names]


def _extract_feature_matrix(data: Table, feature_names: List[str]) -> np.ndarray:
    columns = []
    for name in feature_names:
        variable = next((var for var in data.domain.attributes if var.name == name), None)
        if variable is None:
            raise ValueError(f"Feature '{name}' is missing from the input data.")
        column = data.get_column(variable)
        columns.append(column.astype(float))
    if not columns:
        raise ValueError("No shared continuous descriptor columns found in attributes.")
    return np.column_stack(columns).astype(float)


def _prediction_table(
    data: Table,
    prediction: ApplicabilityDomainPrediction,
    *,
    include_williams: bool,
    include_knn: bool,
    include_maha: bool,
) -> Table:
    used = {var.name for var in data.domain.attributes} | {var.name for var in data.domain.class_vars} | {var.name for var in data.domain.metas}

    def unique_name(name: str) -> str:
        if name not in used:
            used.add(name)
            return name
        i = 2
        while f"{name}_{i}" in used:
            i += 1
        out = f"{name}_{i}"
        used.add(out)
        return out

    attrs = list(data.domain.attributes)
    attrs.append(ContinuousVariable(unique_name("AD_leverage")))
    if include_williams:
        attrs.append(DiscreteVariable(unique_name("AD_in_williams"), values=("False", "True")))
    if include_knn:
        attrs.append(ContinuousVariable(unique_name("AD_knn_dist")))
        attrs.append(DiscreteVariable(unique_name("AD_in_knn"), values=("False", "True")))
    if include_maha:
        attrs.append(ContinuousVariable(unique_name("AD_maha_d2")))
        attrs.append(DiscreteVariable(unique_name("AD_in_maha"), values=("False", "True")))
    attrs.append(DiscreteVariable(unique_name("AD_in_domain"), values=("False", "True")))

    domain = Domain(attrs, data.domain.class_vars, data.domain.metas)

    columns = [data.X.astype(float)]
    columns.append(prediction.leverage.reshape(-1, 1))
    if include_williams:
        columns.append(prediction.in_ad_williams.astype(int).reshape(-1, 1))
    if include_knn:
        knn_dist = prediction.knn_dist if prediction.knn_dist is not None else np.full(len(data), np.nan, dtype=float)
        columns.append(knn_dist.reshape(-1, 1))
        columns.append(prediction.in_ad_knn.astype(int).reshape(-1, 1))
    if include_maha:
        maha_d2 = prediction.maha_d2 if prediction.maha_d2 is not None else np.full(len(data), np.nan, dtype=float)
        columns.append(maha_d2.reshape(-1, 1))
        columns.append(prediction.in_ad_maha.astype(int).reshape(-1, 1))
    columns.append(prediction.in_ad.astype(int).reshape(-1, 1))

    X_out = np.hstack(columns).astype(float)
    return Table.from_numpy(domain, X=X_out, Y=data.Y, metas=data.metas)


def _summary_table(
    fit: ApplicabilityDomainFit,
    ref_prediction: ApplicabilityDomainPrediction,
    query_prediction: ApplicabilityDomainPrediction,
) -> Table:
    rows = [
        ("reference_rows", "reference", float(fit.ref_row_count)),
        ("query_rows", "query", float(len(query_prediction.in_ad))),
        ("features", "shared", float(len(fit.feature_names))),
        ("h_star", "williams", float(fit.h_star)),
        ("reference_in_domain", "reference", float(np.sum(ref_prediction.in_ad))),
        ("query_in_domain", "query", float(np.sum(query_prediction.in_ad))),
    ]
    if fit.knn_threshold is not None:
        rows.append(("knn_threshold", "knn", float(fit.knn_threshold)))
    if fit.maha_threshold is not None:
        rows.append(("maha_threshold", "mahalanobis", float(fit.maha_threshold)))

    domain = Domain([ContinuousVariable("Value")], metas=[StringVariable("Metric"), StringVariable("Method")])
    X = np.array([[value] for _metric, _method, value in rows], dtype=float)
    metas = np.array([[metric, method] for metric, method, _value in rows], dtype=object)
    return Table.from_numpy(domain, X=X, metas=metas)


class OWApplicabilityDomain(OWWidget):
    name = "Applicability Domain"
    description = "Evaluate whether compounds lie inside the descriptor-space domain of a reference set."
    icon = "icons/analysis/owapplicabilitydomainwidget.svg"
    priority = 136

    class Inputs:
        data = Input("Data", Table)
        reference_data = Input("Reference Data", Table)

    class Outputs:
        data_results = Output("Data Results", Table)
        reference_results = Output("Reference Results", Table)
        summary = Output("AD Summary", Table)

    use_williams: bool = Setting(True)
    use_knn: bool = Setting(True)
    use_maha: bool = Setting(False)
    combine_mode: str = Setting("and")
    knn_k: int = Setting(5)
    knn_quantile: float = Setting(0.95)
    maha_alpha: float = Setting(0.95)
    maha_use_chi2: bool = Setting(True)
    auto_run: bool = Setting(False)

    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[Table] = None
        self.reference_data: Optional[Table] = None

        self.mainArea.hide()

        box = gui.widgetBox(self.controlArea, "Applicability Domain", spacing=8)
        gui.checkBox(box, self, "use_williams", "Williams leverage", callback=self._update_status_only)
        gui.checkBox(box, self, "use_knn", "kNN distance", callback=self._update_status_only)
        gui.spin(box, self, "knn_k", 1, 50, 1, label="kNN k:", orientation=Qt.Horizontal, callback=self._update_status_only)
        gui.doubleSpin(box, self, "knn_quantile", 0.5, 0.999, 0.01, label="kNN quantile:", orientation=Qt.Horizontal, callback=self._update_status_only)
        gui.checkBox(box, self, "use_maha", "Mahalanobis distance", callback=self._update_status_only)
        gui.doubleSpin(box, self, "maha_alpha", 0.5, 0.999, 0.01, label="Mahalanobis α:", orientation=Qt.Horizontal, callback=self._update_status_only)
        gui.checkBox(box, self, "maha_use_chi2", "Use chi-square threshold", callback=self._update_status_only)
        gui.comboBox(box, self, "combine_mode", items=["and", "or"], label="Combine:", orientation=Qt.Horizontal, callback=self._update_status_only)
        gui.checkBox(box, self, "auto_run", "Auto-run", callback=self._update_status_only)

        gui.button(self.controlArea, self, "Evaluate AD", callback=self.commit)

        status_root = QWidget(self.controlArea)
        status_layout = QVBoxLayout(status_root)
        status_layout.setContentsMargins(6, 6, 6, 6)
        self.status_label = QLabel("Waiting for input…")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        self.controlArea.layout().addWidget(status_root)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _update_status_only(self) -> None:
        self._set_status(self._input_summary())
        if bool(self.auto_run) and self.data is not None:
            self.commit()

    def _input_summary(self) -> str:
        query_rows = 0 if self.data is None else len(self.data)
        ref_rows = 0 if self.reference_data is None else len(self.reference_data)
        return f"Input: Data rows={query_rows}, Reference rows={ref_rows or query_rows}"

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self.data = data
        self._set_status(self._input_summary())
        if bool(self.auto_run) and self.data is not None:
            self.commit()

    @Inputs.reference_data
    def set_reference_data(self, data: Optional[Table]) -> None:
        self.reference_data = data
        self._set_status(self._input_summary())
        if bool(self.auto_run) and self.data is not None:
            self.commit()

    def _ad_config(self) -> ADConfig:
        return ADConfig(
            use_williams=bool(self.use_williams),
            use_knn=bool(self.use_knn),
            use_mahalanobis=bool(self.use_maha),
            combine_mode=str(self.combine_mode or "and"),
            knn_k=int(self.knn_k),
            knn_quantile=float(self.knn_quantile),
            maha_alpha=float(self.maha_alpha),
            maha_use_chi2=bool(self.maha_use_chi2),
        )

    def commit(self) -> None:
        self.clear_messages()
        if self.data is None:
            self._set_status("No data input.")
            self.Outputs.data_results.send(None)
            self.Outputs.reference_results.send(None)
            self.Outputs.summary.send(None)
            return

        reference = self.reference_data if self.reference_data is not None else self.data
        query = self.data

        try:
            feature_names = _common_continuous_feature_names(reference, query)
            if not feature_names:
                raise ValueError("No shared continuous descriptor columns found in attributes.")

            X_ref = _extract_feature_matrix(reference, feature_names)
            X_query = _extract_feature_matrix(query, feature_names)

            fit = fit_applicability_domain(X_ref, feature_names, ad_cfg=self._ad_config())
            ref_prediction = score_applicability_domain(fit, X_ref)
            query_prediction = score_applicability_domain(fit, X_query)

            ref_table = _prediction_table(
                reference,
                ref_prediction,
                include_williams=bool(self.use_williams),
                include_knn=bool(self.use_knn),
                include_maha=bool(self.use_maha),
            )
            query_table = _prediction_table(
                query,
                query_prediction,
                include_williams=bool(self.use_williams),
                include_knn=bool(self.use_knn),
                include_maha=bool(self.use_maha),
            )
            summary = _summary_table(fit, ref_prediction, query_prediction)
        except ValueError as exc:
            set_widget_error(self, str(exc))
            self._set_status(format_error_status(str(exc)))
            self.Outputs.data_results.send(None)
            self.Outputs.reference_results.send(None)
            self.Outputs.summary.send(None)
            return

        self.Outputs.data_results.send(query_table)
        self.Outputs.reference_results.send(ref_table)
        self.Outputs.summary.send(summary)

        parts = [f"Features={len(feature_names)}", f"h*={fit.h_star:.4g}"]
        if fit.knn_threshold is not None:
            parts.append(f"kNN thr={fit.knn_threshold:.4g}")
        if fit.maha_threshold is not None:
            parts.append(f"Mahalanobis thr={fit.maha_threshold:.4g}")
        self._set_status(
            f"Reference in-domain={int(np.sum(ref_prediction.in_ad))}/{len(ref_prediction.in_ad)}, "
            f"Data in-domain={int(np.sum(query_prediction.in_ad))}/{len(query_prediction.in_ad)}. "
            + ", ".join(parts)
        )


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWApplicabilityDomain).run()
