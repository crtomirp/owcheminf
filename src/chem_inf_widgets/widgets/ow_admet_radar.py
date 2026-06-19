from __future__ import annotations

import math
from html import escape

import pyqtgraph as pg
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QLabel,
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

from chem_inf_widgets.chemcore.services.admet_radar_service import (
    AdmetRadarConfig,
    AdmetRadarResult,
    admet_flagged_records_as_dicts,
    admet_flagged_table,
    admet_radar_records_table,
    admet_radar_summary_table,
    admet_summary_as_rows,
    run_admet_radar,
)
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_loaded_status,
    format_no_input_status,
)
from chem_inf_widgets.widgets.utils import send_output_values, show_service_issues

_PROFILE_METRICS = [
    ("Lipinski", lambda record: 1.0 if record.lipinski_pass else 0.0),
    ("Veber", lambda record: 1.0 if record.veber_pass else 0.0),
    ("Ghose", lambda record: 1.0 if record.ghose_pass else 0.0),
    ("Egan", lambda record: 1.0 if record.egan_pass else 0.0),
    ("Muegge", lambda record: 1.0 if record.muegge_pass else 0.0),
    ("PAINS clean", lambda record: 0.0 if record.pains_match else 1.0),
    ("Brenk clean", lambda record: 0.0 if record.brenk_match else 1.0),
    ("QED", lambda record: max(0.0, min(float(record.qed_score), 1.0))),
]


def _count_card(label: str, value: int) -> str:
    return (
        "<div style='display:inline-block; min-width:150px; margin:6px; padding:10px 12px; "
        "border:1px solid #D0D5DD; border-radius:10px; background:#F8FAFC;'>"
        f"<div style='font-size:12px; color:#475467;'>{escape(label)}</div>"
        f"<div style='font-size:24px; font-weight:600; color:#101828;'>{int(value)}</div>"
        "</div>"
    )


def _render_report_html(result: AdmetRadarResult) -> str:
    summary = result.summary
    summary_rows = admet_summary_as_rows(result)
    issue_rows = [
        row
        for row in summary_rows
        if str(row.get("metric", "")).startswith("issue_")
    ][:5]
    flagged_rows = admet_flagged_records_as_dicts(result)[:5]

    cards = "".join(
        [
            _count_card("Rows", summary.n_rows),
            _count_card("Valid molecules", summary.n_valid_molecules),
            _count_card("Invalid molecules", summary.n_invalid_molecules),
            _count_card("Lipinski pass", summary.n_lipinski_pass),
            _count_card("Veber pass", summary.n_veber_pass),
            _count_card("QED / alerts flagged", summary.n_pains_matches + summary.n_brenk_matches),
        ]
    )

    issue_html = "".join(
        f"<li>{escape(str(row['description']))}</li>"
        for row in issue_rows
    ) or "<li>No service issues recorded.</li>"

    flagged_html = "".join(
        "<li>"
        f"{escape(str(row.get('name') or row.get('input_smiles') or 'compound'))}: "
        f"Lipinski={int(row.get('lipinski_pass', 0))}, "
        f"Veber={int(row.get('veber_pass', 0))}, "
        f"PAINS={int(row.get('pains_match', 0))}, "
        f"Brenk={int(row.get('brenk_match', 0))}"
        "</li>"
        for row in flagged_rows
    ) or "<li>No flagged compounds detected.</li>"

    return (
        "<html><body style='font-family: sans-serif;'>"
        "<h2>ADMET Radar</h2>"
        f"{cards}"
        "<h3>Service issues</h3>"
        f"<ul>{issue_html}</ul>"
        "<h3>Flagged compounds preview</h3>"
        f"<ul>{flagged_html}</ul>"
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
    for row_index, row in enumerate(rows):
        for column_index, header in enumerate(headers):
            item = QTableWidgetItem("" if row.get(header) is None else str(row.get(header)))
            widget.setItem(row_index, column_index, item)
    widget.resizeColumnsToContents()


def _styled_plot(title: str = "") -> pg.PlotWidget:
    plot_widget = pg.PlotWidget(title=title)
    plot_widget.setBackground("#FFFFFF")
    plot_widget.hideAxis("left")
    plot_widget.hideAxis("bottom")
    plot_widget.setMenuEnabled(False)
    plot_widget.setMouseEnabled(x=False, y=False)
    return plot_widget


def _profile_values(record) -> list[float]:
    return [getter(record) for _label, getter in _PROFILE_METRICS]


def _profile_mean(records) -> list[float]:
    if not records:
        return [0.0] * len(_PROFILE_METRICS)
    totals = [0.0] * len(_PROFILE_METRICS)
    for record in records:
        for index, value in enumerate(_profile_values(record)):
            totals[index] += float(value)
    return [value / len(records) for value in totals]


def _radar_points(values: list[float], *, radius: float = 1.0) -> tuple[list[float], list[float]]:
    if not values:
        return [], []

    xs: list[float] = []
    ys: list[float] = []
    n_values = len(values)
    for index, value in enumerate(values):
        angle = (2.0 * 3.141592653589793 * index / n_values) - (3.141592653589793 / 2.0)
        score = max(0.0, min(float(value), 1.0)) * radius
        xs.append(score * math.cos(angle))
        ys.append(score * math.sin(angle))
    xs.append(xs[0])
    ys.append(ys[0])
    return xs, ys


def _profile_name(record) -> str:
    if record.name:
        return str(record.name)
    if record.input_smiles:
        return str(record.input_smiles)
    return f"row {record.row_index}"


class OWAdmetRadar(OWWidget):
    name = "ADMET Radar"
    description = "Summarize drug-likeness rules and structural alerts for molecular datasets."
    icon = "icons/standardization_filtering/owdrugfilterwidget.png"
    priority = 146
    keywords = ["admet", "lipinski", "veber", "muegge", "pains", "brenk", "qed"]
    want_main_area = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        admet_table = Output("ADMET Table", Table, default=True)
        flagged_compounds = Output("Flagged Compounds", Table)
        summary_table = Output("Summary Table", Table)

    auto_run = Setting(True)
    compute_pains = Setting(True)
    compute_brenk = Setting(True)

    def __init__(self) -> None:
        super().__init__()
        self.data: Table | None = None
        self._result: AdmetRadarResult | None = None
        self._flagged_records = []
        self._profile_record = None
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
            "compute_brenk",
            "Compute Brenk alerts",
            callback=self._on_settings_changed,
        )
        gui.checkBox(
            options_box,
            self,
            "auto_run",
            "Auto-run on input changes",
            callback=self._on_settings_changed,
        )
        gui.button(options_box, self, "Run ADMET radar", callback=self.commit)

        self._status_label = QLabel("Waiting for data.")
        self._status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self._status_label)

        tabs = QTabWidget()
        self.mainArea.layout().addWidget(tabs)

        self._report_browser = QTextBrowser()
        self._report_browser.setOpenExternalLinks(False)
        tabs.addTab(self._report_browser, "Summary")

        self._summary_table_widget = QTableWidget()
        tabs.addTab(self._summary_table_widget, "Rule Summary")

        self._flagged_table_widget = QTableWidget()
        self._flagged_table_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self._flagged_table_widget.setSelectionMode(QTableWidget.SingleSelection)
        self._flagged_table_widget.itemSelectionChanged.connect(self._on_flagged_selection_changed)
        tabs.addTab(self._flagged_table_widget, "Flagged Compounds")

        profile_page = QWidget()
        profile_layout = QVBoxLayout(profile_page)
        self._profile_label = QLabel("Select a flagged compound to inspect its ADMET rule profile.")
        self._profile_label.setWordWrap(True)
        profile_layout.addWidget(self._profile_label)
        self._profile_plot = _styled_plot("Rule profile (0-1)")
        profile_layout.addWidget(self._profile_plot)
        tabs.addTab(profile_page, "Profile Plot")

    def _set_status(self, text: str, ok: bool = False) -> None:
        color = "#027A48" if ok else "#475467"
        self._status_label.setStyleSheet(f"color:{color};")
        self._status_label.setText(text)

    def _config(self) -> AdmetRadarConfig:
        return AdmetRadarConfig(
            compute_pains=bool(self.compute_pains),
            compute_brenk=bool(self.compute_brenk),
        )

    def _on_settings_changed(self) -> None:
        if self.auto_run and self.data is not None:
            self.commit()

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
        self._flagged_records = []
        self._profile_record = None
        clear_widget_messages(self)
        self._report_browser.clear()
        self._summary_table_widget.clear()
        self._summary_table_widget.setRowCount(0)
        self._summary_table_widget.setColumnCount(0)
        self._flagged_table_widget.clear()
        self._flagged_table_widget.setRowCount(0)
        self._flagged_table_widget.setColumnCount(0)
        self._profile_plot.clear()
        self._profile_label.setText("Select a flagged compound to inspect its ADMET rule profile.")
        send_output_values(
            (self.Outputs.admet_table, None),
            (self.Outputs.flagged_compounds, None),
            (self.Outputs.summary_table, None),
        )

    def _render_profile_plot(self) -> None:
        self._profile_plot.clear()
        if self._result is None or self._profile_record is None:
            self._profile_label.setText("Select a flagged compound to inspect its ADMET rule profile.")
            return

        plot_item = self._profile_plot.getPlotItem()
        grid_pen = pg.mkPen("#CBD5E1", width=1)
        axis_pen = pg.mkPen("#94A3B8", width=1)
        baseline_pen = pg.mkPen("#64748B", width=2, style=Qt.DashLine)
        selected_pen = pg.mkPen("#0F766E", width=3)
        selected_symbol_brush = pg.mkBrush("#0F766E")

        for level in (0.25, 0.5, 0.75, 1.0):
            x_grid, y_grid = _radar_points([level] * len(_PROFILE_METRICS))
            plot_item.addItem(pg.PlotDataItem(x_grid, y_grid, pen=grid_pen))

        axis_x, axis_y = _radar_points([1.0] * len(_PROFILE_METRICS))
        for index in range(len(_PROFILE_METRICS)):
            plot_item.addItem(
                pg.PlotDataItem([0.0, axis_x[index]], [0.0, axis_y[index]], pen=axis_pen)
            )

        for index, (label, _getter) in enumerate(_PROFILE_METRICS):
            label_x = axis_x[index] * 1.14
            label_y = axis_y[index] * 1.14
            text = pg.TextItem(label, color="#334155", anchor=(0.5, 0.5))
            text.setPos(label_x, label_y)
            plot_item.addItem(text)

        valid_records = [record for record in self._result.records if record.valid_molecule]
        mean_values = _profile_mean(valid_records)
        selected_values = _profile_values(self._profile_record)
        mean_x, mean_y = _radar_points(mean_values)
        selected_x, selected_y = _radar_points(selected_values)

        plot_item.addItem(pg.PlotDataItem(mean_x, mean_y, pen=baseline_pen))
        plot_item.addItem(
            pg.PlotDataItem(
                selected_x,
                selected_y,
                pen=selected_pen,
                symbol="o",
                symbolSize=8,
                symbolBrush=selected_symbol_brush,
                symbolPen=selected_pen,
            )
        )
        plot_item.setXRange(-1.35, 1.35, padding=0.0)
        plot_item.setYRange(-1.25, 1.25, padding=0.0)

        record = self._profile_record
        self._profile_label.setText(
            "Selected: "
            f"{_profile_name(record)} | "
            f"QED={record.qed_score:.2f}, "
            f"MW={record.molecular_weight:.1f}, "
            f"LogP={record.logp:.2f}, "
            f"PAINS={'yes' if record.pains_match else 'no'}, "
            f"Brenk={'yes' if record.brenk_match else 'no'}"
        )

    def _on_flagged_selection_changed(self) -> None:
        if not self._flagged_records:
            return
        row_index = self._flagged_table_widget.currentRow()
        if row_index < 0:
            selected_ranges = self._flagged_table_widget.selectedRanges()
            if not selected_ranges:
                return
            row_index = selected_ranges[0].topRow()
        if 0 <= row_index < len(self._flagged_records):
            self._profile_record = self._flagged_records[row_index]
            self._render_profile_plot()

    def commit(self) -> None:
        if self.data is None:
            self._send_empty()
            self._set_status(format_no_input_status(), ok=False)
            return

        clear_widget_messages(self)
        result = run_admet_radar(self.data, self._config())
        self._result = result

        admet_table = admet_radar_records_table(result)
        flagged_table = admet_flagged_table(result)
        summary = admet_radar_summary_table(result)
        send_output_values(
            (self.Outputs.admet_table, admet_table),
            (self.Outputs.flagged_compounds, flagged_table),
            (self.Outputs.summary_table, summary),
        )

        self._report_browser.setHtml(_render_report_html(result))
        _set_table_rows(self._summary_table_widget, admet_summary_as_rows(result))
        flagged_rows = admet_flagged_records_as_dicts(result)[:100]
        _set_table_rows(self._flagged_table_widget, flagged_rows)
        self._flagged_records = [
            record
            for record in result.records
            if (
                not record.valid_molecule
                or not record.lipinski_pass
                or not record.veber_pass
                or not record.ghose_pass
                or not record.egan_pass
                or not record.muegge_pass
                or record.pains_match
                or record.brenk_match
                or bool(record.issue_codes)
            )
        ][:100]
        if self._flagged_records:
            self._flagged_table_widget.selectRow(0)
            self._profile_record = self._flagged_records[0]
        else:
            valid_records = [record for record in result.records if record.valid_molecule]
            self._profile_record = valid_records[0] if valid_records else None
        self._render_profile_plot()

        show_service_issues(self, result.issues, subject="admet radar", issue_label="issue")
        self._set_status(
            format_done_status(
                f"rows={result.summary.n_rows}",
                f"valid={result.summary.n_valid_molecules}",
                f"flagged={len(admet_flagged_records_as_dicts(result))}",
            ),
            ok=not any(issue.severity == "error" for issue in result.issues),
        )


__all__ = ["OWAdmetRadar"]
