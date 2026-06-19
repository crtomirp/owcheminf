from __future__ import annotations

from html import escape

from AnyQt.QtWidgets import (
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
)
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.dataset_profiler_service import (
    DatasetProfilerConfig,
    DatasetProfilerResult,
    dataset_profile_summary_as_rows,
    descriptor_summary_as_rows,
    problematic_compounds_as_dicts,
    problematic_compounds_table,
    profiled_records_table,
    run_dataset_profiler,
)
from chem_inf_widgets.chemcore.services.dataset_profiler_service import (
    summary_table as dataset_profile_summary_table,
)
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_loaded_status,
    format_no_input_status,
)
from chem_inf_widgets.widgets.utils import send_output_values, show_service_issues


def _count_card(label: str, value: int) -> str:
    return (
        "<div style='display:inline-block; min-width:150px; margin:6px; padding:10px 12px; "
        "border:1px solid #D0D5DD; border-radius:10px; background:#F8FAFC;'>"
        f"<div style='font-size:12px; color:#475467;'>{escape(label)}</div>"
        f"<div style='font-size:24px; font-weight:600; color:#101828;'>{int(value)}</div>"
        "</div>"
    )


def _render_report_html(result: DatasetProfilerResult) -> str:
    summary = result.summary
    missing_rows = sorted(
        dataset_profile_summary_as_rows(summary),
        key=lambda row: str(row.get("metric", "")),
    )
    top_missing = [
        row for row in missing_rows if str(row.get("metric", "")).startswith("missing_")
    ][:5]
    issue_rows = [
        row for row in missing_rows if str(row.get("metric", "")).startswith("issue_")
    ][:5]

    descriptor_rows = descriptor_summary_as_rows(result.descriptor_summary)
    descriptor_items = "".join(
        (
            f"<li>{escape(str(row['descriptor']))}: count={int(row['count'])}, "
            f"mean={row['mean']:.3f}</li>"
        )
        if row["count"]
        else f"<li>{escape(str(row['descriptor']))}: no valid values</li>"
        for row in descriptor_rows
    )

    top_missing_html = "".join(
        f"<li>{escape(str(row['metric']).removeprefix('missing_'))}: {int(row['value'])}</li>"
        for row in top_missing
    ) or "<li>No missing values detected.</li>"
    issue_html = "".join(
        f"<li>{escape(str(row['metric']).removeprefix('issue_'))}: {int(row['value'])}</li>"
        for row in issue_rows
    ) or "<li>No service issues recorded.</li>"

    cards = "".join(
        [
            _count_card("Rows", summary.n_rows),
            _count_card("Valid molecules", summary.n_valid_molecules),
            _count_card("Invalid molecules", summary.n_invalid_molecules),
            _count_card("Duplicate rows", summary.duplicate_smiles_count),
            _count_card("Lipinski fails", summary.n_lipinski_fail),
            _count_card("PAINS matches", summary.n_pains_matches),
        ]
    )

    return (
        "<html><body style='font-family: sans-serif;'>"
        "<h2>Dataset Profile</h2>"
        f"{cards}"
        "<h3>Missing values</h3>"
        f"<ul>{top_missing_html}</ul>"
        "<h3>Service issues</h3>"
        f"<ul>{issue_html}</ul>"
        "<h3>Descriptor summary</h3>"
        f"<ul>{descriptor_items}</ul>"
        "</body></html>"
    )


def _set_table_rows(widget: QTableWidget, rows: list[dict[str, object]]) -> None:
    if not rows:
        widget.clear()
        widget.setRowCount(0)
        widget.setColumnCount(0)
        return

    headers = list(rows[0].keys())
    widget.setColumnCount(len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setRowCount(len(rows))
    for row_idx, row in enumerate(rows):
        for col_idx, header in enumerate(headers):
            value = row.get(header, "")
            item = QTableWidgetItem("" if value is None else str(value))
            widget.setItem(row_idx, col_idx, item)
    widget.resizeColumnsToContents()


class OWDatasetProfiler(OWWidget):
    name = "Dataset Profiler"
    description = "Profile molecular datasets for validity, duplicates, missing values, and basic drug-likeness signals."
    icon = "icons/modeling/qsar_regression.png"
    priority = 144
    keywords = ["dataset", "profiler", "duplicates", "lipinski", "PAINS", "report"]
    want_main_area = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        profiled_table = Output("Profiled Table", Table, default=True)
        problematic_compounds = Output("Problematic Compounds", Table)
        summary_table = Output("Summary Table", Table)

    auto_run = Setting(True)
    compute_pains = Setting(True)

    def __init__(self) -> None:
        super().__init__()
        self.data: Table | None = None
        self._result: DatasetProfilerResult | None = None
        self._build_ui()
        self._set_status("Waiting for data.", ok=True)

    def _build_ui(self) -> None:
        options_box = gui.widgetBox(self.controlArea, "Options")
        gui.checkBox(
            options_box,
            self,
            "compute_pains",
            "Compute PAINS alerts",
            callback=self._on_settings_changed,
        )
        gui.checkBox(
            options_box,
            self,
            "auto_run",
            "Auto-run on input changes",
            callback=self._on_settings_changed,
        )
        gui.button(options_box, self, "Run profile", callback=self.commit)

        self._status_label = QLabel("Waiting for data.")
        self._status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self._status_label)

        tabs = QTabWidget()
        self.mainArea.layout().addWidget(tabs)

        self._report_browser = QTextBrowser()
        self._report_browser.setOpenExternalLinks(False)
        tabs.addTab(self._report_browser, "Summary")

        self._descriptor_table = QTableWidget()
        tabs.addTab(self._descriptor_table, "Descriptors")

        self._problem_table = QTableWidget()
        tabs.addTab(self._problem_table, "Problematic Rows")

    def _set_status(self, text: str, ok: bool = False) -> None:
        color = "#027A48" if ok else "#475467"
        self._status_label.setStyleSheet(f"color:{color};")
        self._status_label.setText(text)

    def _on_settings_changed(self) -> None:
        if self.auto_run and self.data is not None:
            self.commit()

    def _config(self) -> DatasetProfilerConfig:
        return DatasetProfilerConfig(compute_pains=bool(self.compute_pains))

    @Inputs.data
    def set_data(self, data: Table | None) -> None:
        self.data = data
        if data is None:
            self._send_empty()
            self._set_status(format_no_input_status(), ok=False)
            return

        self._set_status(format_loaded_status(len(data)), ok=True)
        if self.auto_run:
            self.commit()

    def _send_empty(self) -> None:
        self._result = None
        clear_widget_messages(self)
        self._report_browser.clear()
        self._descriptor_table.clear()
        self._descriptor_table.setRowCount(0)
        self._descriptor_table.setColumnCount(0)
        self._problem_table.clear()
        self._problem_table.setRowCount(0)
        self._problem_table.setColumnCount(0)
        send_output_values(
            (self.Outputs.profiled_table, None),
            (self.Outputs.problematic_compounds, None),
            (self.Outputs.summary_table, None),
        )

    def commit(self) -> None:
        if self.data is None:
            self._send_empty()
            self._set_status(format_no_input_status(), ok=False)
            return

        clear_widget_messages(self)
        result = run_dataset_profiler(self.data, self._config())
        self._result = result

        profiled = profiled_records_table(result)
        problematic = problematic_compounds_table(result)
        summary = dataset_profile_summary_table(result)
        send_output_values(
            (self.Outputs.profiled_table, profiled),
            (self.Outputs.problematic_compounds, problematic),
            (self.Outputs.summary_table, summary),
        )

        self._report_browser.setHtml(_render_report_html(result))
        _set_table_rows(self._descriptor_table, descriptor_summary_as_rows(result.descriptor_summary))
        _set_table_rows(self._problem_table, problematic_compounds_as_dicts(result.records)[:100])

        show_service_issues(self, result.issues, subject="dataset profiler", issue_label="issue")
        self._set_status(
            format_done_status(
                f"rows={result.summary.n_rows}",
                f"valid={result.summary.n_valid_molecules}",
                f"invalid={result.summary.n_invalid_molecules}",
            ),
            ok=not any(issue.severity == "error" for issue in result.issues),
        )


__all__ = ["OWDatasetProfiler"]
