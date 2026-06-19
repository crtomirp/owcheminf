from __future__ import annotations

import math
from html import escape

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QComboBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from Orange.data import ContinuousVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.activity_cliff_service import find_activity_cliffs
from chem_inf_widgets.chemcore.services.chemical_series_service import (
    ChemicalSeriesConfig,
    ChemicalSeriesResult,
    chemical_series_members_as_dicts,
    chemical_series_members_table,
    chemical_series_summary_as_rows,
    chemical_series_summary_table,
    chemical_series_table,
    run_chemical_series_explorer,
    series_rows_as_dicts,
)
from chem_inf_widgets.chemcore.services.matched_pair_service import find_matched_pairs
from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table
from chem_inf_widgets.chemcore.services.rgroup_decomposition_service import decompose_rgroups
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_loaded_status,
    format_no_input_status,
    set_widget_warning,
)
from chem_inf_widgets.widgets.utils import (
    combine_messages,
    send_output_values,
    summarize_service_issues,
)


def _count_card(label: str, value: int | str) -> str:
    return (
        "<div style='display:inline-block; min-width:150px; margin:6px; padding:10px 12px; "
        "border:1px solid #D0D5DD; border-radius:10px; background:#F8FAFC;'>"
        f"<div style='font-size:12px; color:#475467;'>{escape(label)}</div>"
        f"<div style='font-size:24px; font-weight:600; color:#101828;'>{escape(str(value))}</div>"
        "</div>"
    )


def _render_report_html(result: ChemicalSeriesResult) -> str:
    summary = result.summary
    top_series = series_rows_as_dicts(result)[:5]
    issue_rows = [
        row for row in chemical_series_summary_as_rows(result) if str(row.get("metric", "")).startswith("issue_")
    ][:5]

    cards = "".join(
        [
            _count_card("Rows", summary.n_rows),
            _count_card("Valid molecules", summary.n_valid_molecules),
            _count_card("Invalid molecules", summary.n_invalid_molecules),
            _count_card("Series", summary.n_series),
            _count_card("Singleton series", summary.n_singleton_series),
            _count_card("Acyclic rows", summary.n_acyclic_rows),
        ]
    )

    top_series_html = "".join(
        (
            f"<li>{escape(str(row['scaffold']))}: count={int(row['count'])}, "
            f"fraction={float(row['fraction']) * 100:.1f}%, "
            f"mean_activity={row['mean_activity']}</li>"
        )
        for row in top_series
    ) or "<li>No scaffold series detected.</li>"

    issue_html = "".join(
        f"<li>{escape(str(row['description']))}</li>"
        for row in issue_rows
    ) or "<li>No service issues recorded.</li>"

    target_html = (
        f"<p><b>Activity column:</b> {escape(summary.target_column)}</p>"
        if summary.target_column
        else "<p><b>Activity column:</b> not used</p>"
    )

    return (
        "<html><body style='font-family: sans-serif;'>"
        "<h2>Chemical Series Explorer</h2>"
        f"{cards}"
        f"{target_html}"
        "<h3>Top series</h3>"
        f"<ul>{top_series_html}</ul>"
        "<h3>Service issues</h3>"
        f"<ul>{issue_html}</ul>"
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


def _numeric_target_names(data: Table | None) -> list[str]:
    if data is None:
        return []
    variables = list(data.domain.class_vars) + list(data.domain.attributes) + list(data.domain.metas)
    return [var.name for var in variables if isinstance(var, ContinuousVariable)]


def _rgroup_rows(result) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in result.rows:
        item = {
            "row_index": int(row.index) + 1,
            "core": row.core,
        }
        for label in result.group_labels:
            item[label] = row.groups.get(label, "")
        rows.append(item)
    return rows


def _rgroup_table(result) -> Table | None:
    rows = _rgroup_rows(result)
    meta_columns = ["row_index", "core", *list(result.group_labels)]
    return records_to_orange_table(rows, meta_columns=meta_columns, name="Chemical Series R-Group Table")


def _matched_pair_rows(rows, selected_records) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        record_a = selected_records[row.index_a]
        record_b = selected_records[row.index_b]
        out.append(
            {
                "row_index_a": record_a.row_index,
                "row_index_b": record_b.row_index,
                "name_a": record_a.name,
                "name_b": record_b.name,
                "smiles_a": row.smiles_a,
                "smiles_b": row.smiles_b,
                "transformation": row.transformation,
                "shared_heavy_atoms": row.shared_heavy_atoms,
                "delta_activity": row.delta_property,
            }
        )
    return out


def _matched_pair_table(rows, selected_records) -> Table | None:
    return records_to_orange_table(
        _matched_pair_rows(rows, selected_records),
        meta_columns=[
            "row_index_a",
            "row_index_b",
            "name_a",
            "name_b",
            "smiles_a",
            "smiles_b",
            "transformation",
        ],
        name="Chemical Series Matched Pairs",
    )


def _optional_finite_float(value) -> float | None:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def _activity_cliff_rows(result, selected_records) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for pair in result.pairs:
        record_a = selected_records[pair.index_a]
        record_b = selected_records[pair.index_b]
        out.append(
            {
                "row_index_a": record_a.row_index,
                "row_index_b": record_b.row_index,
                "name_a": pair.name_a,
                "name_b": pair.name_b,
                "smiles_a": pair.smiles_a,
                "smiles_b": pair.smiles_b,
                "similarity": pair.similarity,
                "activity_ratio": pair.activity_ratio,
                "cliff_score": pair.cliff_score,
                "activity_a": pair.activity_a,
                "activity_b": pair.activity_b,
                "higher_active": pair.higher_active,
            }
        )
    return out


def _activity_cliff_table(result, selected_records) -> Table | None:
    return records_to_orange_table(
        _activity_cliff_rows(result, selected_records),
        meta_columns=[
            "row_index_a",
            "row_index_b",
            "name_a",
            "name_b",
            "smiles_a",
            "smiles_b",
            "higher_active",
        ],
        name="Chemical Series Activity Cliffs",
    )


class OWChemicalSeriesExplorer(OWWidget):
    name = "Chemical Series Explorer"
    description = "Group compounds into scaffold-defined series and summarize per-series activity trends."
    icon = "icons/analysis/owscaffoldanalysiswidget.svg"
    priority = 147
    keywords = ["series", "scaffold", "sar", "murcko", "activity"]
    want_main_area = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        series_table = Output("Series Table", Table, default=True)
        members_table = Output("Members Table", Table)
        summary_table = Output("Summary Table", Table)
        selected_data = Output("Selected Data", Table)
        rgroup_table = Output("R-Group Table", Table)
        matched_pair_table = Output("Matched Pair Table", Table)
        activity_cliff_table = Output("Activity Cliff Table", Table)

    auto_run = Setting(True)
    scaffold_kind_idx = Setting(0)
    activity_log_scale = Setting(False)
    target_var_name = Setting("")

    _KINDS = [("Murcko", "murcko"), ("Generic Murcko", "generic")]

    def __init__(self) -> None:
        super().__init__()
        self.data: Table | None = None
        self._result: ChemicalSeriesResult | None = None
        self._series_scaffolds: list[str] = []
        self._selected_scaffold = ""
        self._service_warning_message = ""
        self._rgroup_warning_message = ""
        self._mmp_warning_message = ""
        self._cliff_warning_message = ""
        self._build_ui()
        self._set_status("Waiting for data.", ok=True)
        self._refresh_target_combo()

    def _build_ui(self) -> None:
        options_box = gui.widgetBox(self.controlArea, "Options")
        gui.comboBox(
            options_box,
            self,
            "scaffold_kind_idx",
            label="Scaffold type",
            items=[label for label, _kind in self._KINDS],
            callback=self._on_settings_changed,
        )
        self._target_combo = QComboBox()
        self._target_combo.currentTextChanged.connect(self._on_target_changed)
        options_box.layout().addWidget(QLabel("Activity column"))
        options_box.layout().addWidget(self._target_combo)
        gui.checkBox(
            options_box,
            self,
            "activity_log_scale",
            "Treat activity as log scale",
            callback=self._on_settings_changed,
        )
        gui.checkBox(
            options_box,
            self,
            "auto_run",
            "Auto-run on input changes",
            callback=self._on_settings_changed,
        )
        gui.button(options_box, self, "Run series explorer", callback=self.commit)

        self._status_label = QLabel("Waiting for data.")
        self._status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self._status_label)

        tabs = QTabWidget()
        self.mainArea.layout().addWidget(tabs)

        self._report_browser = QTextBrowser()
        self._report_browser.setOpenExternalLinks(False)
        tabs.addTab(self._report_browser, "Summary")

        self._series_table_widget = QTableWidget()
        self._series_table_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self._series_table_widget.setSelectionMode(QTableWidget.SingleSelection)
        self._series_table_widget.itemSelectionChanged.connect(self._on_series_selection_changed)
        tabs.addTab(self._series_table_widget, "Series")

        self._members_table_widget = QTableWidget()
        tabs.addTab(self._members_table_widget, "Members")

        self._rgroup_status_label = QLabel("Select a series with at least two valid molecules.")
        self._rgroup_status_label.setWordWrap(True)
        self._rgroup_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._rgroup_table_widget = QTableWidget()
        rgroup_page = QWidget()
        rgroup_layout = QVBoxLayout(rgroup_page)
        rgroup_layout.addWidget(self._rgroup_status_label)
        rgroup_layout.addWidget(self._rgroup_table_widget)
        tabs.addTab(rgroup_page, "R-Groups")

        self._mmp_status_label = QLabel("Select a series with at least two valid molecules.")
        self._mmp_status_label.setWordWrap(True)
        self._mmp_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._mmp_table_widget = QTableWidget()
        mmp_page = QWidget()
        mmp_layout = QVBoxLayout(mmp_page)
        mmp_layout.addWidget(self._mmp_status_label)
        mmp_layout.addWidget(self._mmp_table_widget)
        tabs.addTab(mmp_page, "Matched Pairs")

        self._cliff_status_label = QLabel("Select a series with at least two valid molecules and activities.")
        self._cliff_status_label.setWordWrap(True)
        self._cliff_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._cliff_table_widget = QTableWidget()
        cliff_page = QWidget()
        cliff_layout = QVBoxLayout(cliff_page)
        cliff_layout.addWidget(self._cliff_status_label)
        cliff_layout.addWidget(self._cliff_table_widget)
        tabs.addTab(cliff_page, "Activity Cliffs")

    def _set_status(self, text: str, ok: bool = False) -> None:
        color = "#027A48" if ok else "#475467"
        self._status_label.setStyleSheet(f"color:{color};")
        self._status_label.setText(text)

    def _refresh_target_combo(self) -> None:
        names = _numeric_target_names(self.data)
        current = self.target_var_name if self.target_var_name in names else ""
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        self._target_combo.addItem("Auto")
        for name in names:
            self._target_combo.addItem(name)
        if current:
            self._target_combo.setCurrentText(current)
        else:
            self._target_combo.setCurrentIndex(0)
            self.target_var_name = ""
        self._target_combo.setEnabled(bool(names))
        self._target_combo.blockSignals(False)

    def _on_target_changed(self, text: str) -> None:
        self.target_var_name = "" if text == "Auto" else str(text)
        self._on_settings_changed()

    def _on_settings_changed(self) -> None:
        if self.auto_run and self.data is not None:
            self.commit()

    def _config(self) -> ChemicalSeriesConfig:
        return ChemicalSeriesConfig(
            scaffold_kind=self._KINDS[int(self.scaffold_kind_idx)][1],
            target_column=self.target_var_name or None,
            activity_log_scale=bool(self.activity_log_scale),
        )

    @Inputs.data
    def set_data(self, data: Table | None) -> None:
        self.data = data
        self._refresh_target_combo()
        if data is None:
            self._send_empty()
            self._set_status(format_no_input_status(), ok=False)
            return

        self._set_status(format_loaded_status(len(data)), ok=True)
        if self.auto_run:
            self.commit()

    def _send_empty(self) -> None:
        self._result = None
        self._series_scaffolds = []
        self._selected_scaffold = ""
        self._service_warning_message = ""
        self._rgroup_warning_message = ""
        self._mmp_warning_message = ""
        self._cliff_warning_message = ""
        clear_widget_messages(self)
        self._report_browser.clear()
        self._series_table_widget.clear()
        self._series_table_widget.setRowCount(0)
        self._series_table_widget.setColumnCount(0)
        self._members_table_widget.clear()
        self._members_table_widget.setRowCount(0)
        self._members_table_widget.setColumnCount(0)
        self._rgroup_status_label.setText("Select a series with at least two valid molecules.")
        self._rgroup_table_widget.clear()
        self._rgroup_table_widget.setRowCount(0)
        self._rgroup_table_widget.setColumnCount(0)
        self._mmp_status_label.setText("Select a series with at least two valid molecules.")
        self._mmp_table_widget.clear()
        self._mmp_table_widget.setRowCount(0)
        self._mmp_table_widget.setColumnCount(0)
        self._cliff_status_label.setText("Select a series with at least two valid molecules and activities.")
        self._cliff_table_widget.clear()
        self._cliff_table_widget.setRowCount(0)
        self._cliff_table_widget.setColumnCount(0)
        send_output_values(
            (self.Outputs.series_table, None),
            (self.Outputs.members_table, None),
            (self.Outputs.summary_table, None),
            (self.Outputs.selected_data, None),
            (self.Outputs.rgroup_table, None),
            (self.Outputs.matched_pair_table, None),
            (self.Outputs.activity_cliff_table, None),
        )

    def _send_selected_data(self) -> None:
        if self.data is None or self._result is None or not self._selected_scaffold:
            self.Outputs.selected_data.send(None)
            return

        row_indices = [
            int(record.row_index) - 1
            for record in self._result.members
            if record.series_scaffold == self._selected_scaffold and record.valid_molecule
        ]
        if not row_indices:
            self.Outputs.selected_data.send(None)
            return

        self.Outputs.selected_data.send(self.data[row_indices])

    def _selected_member_records(self):
        if self._result is None or not self._selected_scaffold:
            return []
        return [
            record
            for record in self._result.members
            if record.series_scaffold == self._selected_scaffold and record.valid_molecule
        ]

    def _apply_warning_state(self) -> None:
        set_widget_warning(
            self,
            combine_messages(
                self._service_warning_message,
                self._rgroup_warning_message,
                self._mmp_warning_message,
                self._cliff_warning_message,
            ),
        )

    def _update_rgroup_preview(self) -> None:
        selected_records = self._selected_member_records()
        self._rgroup_warning_message = ""
        if len(selected_records) < 2:
            self._rgroup_status_label.setText("Select a series with at least two valid molecules.")
            self._rgroup_table_widget.clear()
            self._rgroup_table_widget.setRowCount(0)
            self._rgroup_table_widget.setColumnCount(0)
            self.Outputs.rgroup_table.send(None)
            self._apply_warning_state()
            return

        smiles_values = [record.input_smiles for record in selected_records if str(record.input_smiles).strip()]
        if len(smiles_values) < 2:
            self._rgroup_status_label.setText("Selected series does not contain enough valid SMILES for R-group decomposition.")
            self._rgroup_table_widget.clear()
            self._rgroup_table_widget.setRowCount(0)
            self._rgroup_table_widget.setColumnCount(0)
            self.Outputs.rgroup_table.send(None)
            self._apply_warning_state()
            return

        try:
            decomposition = decompose_rgroups(smiles_values)
        except ValueError as exc:
            self._rgroup_warning_message = f"R-group preview unavailable: {exc}"
            self._rgroup_status_label.setText("R-group preview unavailable for the selected series.")
            self._rgroup_table_widget.clear()
            self._rgroup_table_widget.setRowCount(0)
            self._rgroup_table_widget.setColumnCount(0)
            self.Outputs.rgroup_table.send(None)
            self._apply_warning_state()
            return
        except Exception as exc:
            self._rgroup_warning_message = f"R-group preview failed: {exc}"
            self._rgroup_status_label.setText("R-group preview failed for the selected series.")
            self._rgroup_table_widget.clear()
            self._rgroup_table_widget.setRowCount(0)
            self._rgroup_table_widget.setColumnCount(0)
            self.Outputs.rgroup_table.send(None)
            self._apply_warning_state()
            return

        self._rgroup_status_label.setText(
            f"Core: {decomposition.core} | matched={len(decomposition.matched_indices)}, "
            f"unmatched={len(decomposition.unmatched_indices)}"
        )
        _set_table_rows(self._rgroup_table_widget, _rgroup_rows(decomposition))
        self.Outputs.rgroup_table.send(_rgroup_table(decomposition))
        self._apply_warning_state()

    def _update_matched_pair_preview(self) -> None:
        selected_records = self._selected_member_records()
        self._mmp_warning_message = ""
        if len(selected_records) < 2:
            self._mmp_status_label.setText("Select a series with at least two valid molecules.")
            self._mmp_table_widget.clear()
            self._mmp_table_widget.setRowCount(0)
            self._mmp_table_widget.setColumnCount(0)
            self.Outputs.matched_pair_table.send(None)
            self._apply_warning_state()
            return

        smiles_values = [record.input_smiles for record in selected_records if str(record.input_smiles).strip()]
        if len(smiles_values) < 2:
            self._mmp_status_label.setText("Selected series does not contain enough valid SMILES for matched-pair analysis.")
            self._mmp_table_widget.clear()
            self._mmp_table_widget.setRowCount(0)
            self._mmp_table_widget.setColumnCount(0)
            self.Outputs.matched_pair_table.send(None)
            self._apply_warning_state()
            return

        property_values = [_optional_finite_float(record.activity) for record in selected_records]
        try:
            rows = find_matched_pairs(
                smiles_values,
                property_values,
                min_shared_atoms=4,
                max_pairs=100,
            )
        except Exception as exc:
            self._mmp_warning_message = f"Matched-pair preview failed: {exc}"
            self._mmp_status_label.setText("Matched-pair preview failed for the selected series.")
            self._mmp_table_widget.clear()
            self._mmp_table_widget.setRowCount(0)
            self._mmp_table_widget.setColumnCount(0)
            self.Outputs.matched_pair_table.send(None)
            self._apply_warning_state()
            return

        if not rows:
            self._mmp_status_label.setText("No matched pairs found for the selected series.")
            self._mmp_table_widget.clear()
            self._mmp_table_widget.setRowCount(0)
            self._mmp_table_widget.setColumnCount(0)
            self.Outputs.matched_pair_table.send(None)
            self._apply_warning_state()
            return

        self._mmp_status_label.setText(f"Matched pairs: {len(rows)}")
        _set_table_rows(self._mmp_table_widget, _matched_pair_rows(rows, selected_records))
        self.Outputs.matched_pair_table.send(_matched_pair_table(rows, selected_records))
        self._apply_warning_state()

    def _update_activity_cliff_preview(self) -> None:
        selected_records = self._selected_member_records()
        self._cliff_warning_message = ""
        if len(selected_records) < 2:
            self._cliff_status_label.setText("Select a series with at least two valid molecules and activities.")
            self._cliff_table_widget.clear()
            self._cliff_table_widget.setRowCount(0)
            self._cliff_table_widget.setColumnCount(0)
            self.Outputs.activity_cliff_table.send(None)
            self._apply_warning_state()
            return

        eligible_records = [
            record
            for record in selected_records
            if str(record.input_smiles).strip() and _optional_finite_float(record.activity) is not None
        ]
        if len(eligible_records) < 2:
            self._cliff_status_label.setText("Selected series needs at least two valid activity values for cliff analysis.")
            self._cliff_table_widget.clear()
            self._cliff_table_widget.setRowCount(0)
            self._cliff_table_widget.setColumnCount(0)
            self.Outputs.activity_cliff_table.send(None)
            self._apply_warning_state()
            return

        smiles_values = [record.input_smiles for record in eligible_records]
        activity_values = [float(_optional_finite_float(record.activity)) for record in eligible_records]
        names = [record.name for record in eligible_records]

        try:
            result = find_activity_cliffs(
                smiles_values,
                activity_values,
                names=names,
                similarity_threshold=0.25,
                activity_fold_threshold=10.0,
                activity_log_scale=bool(self.activity_log_scale),
                max_pairs=100,
            )
        except Exception as exc:
            self._cliff_warning_message = f"Activity-cliff preview failed: {exc}"
            self._cliff_status_label.setText("Activity-cliff preview failed for the selected series.")
            self._cliff_table_widget.clear()
            self._cliff_table_widget.setRowCount(0)
            self._cliff_table_widget.setColumnCount(0)
            self.Outputs.activity_cliff_table.send(None)
            self._apply_warning_state()
            return

        if not result.pairs:
            self._cliff_status_label.setText("No activity cliffs found for the selected series.")
            self._cliff_table_widget.clear()
            self._cliff_table_widget.setRowCount(0)
            self._cliff_table_widget.setColumnCount(0)
            self.Outputs.activity_cliff_table.send(None)
            self._apply_warning_state()
            return

        self._cliff_status_label.setText(
            f"Activity cliffs: {len(result.pairs)} | compounds in cliffs: {len(result.unique_cliff_indices)}"
        )
        _set_table_rows(self._cliff_table_widget, _activity_cliff_rows(result, eligible_records))
        self.Outputs.activity_cliff_table.send(_activity_cliff_table(result, eligible_records))
        self._apply_warning_state()

    def _update_selected_outputs(self) -> None:
        self._send_selected_data()
        self._update_rgroup_preview()
        self._update_matched_pair_preview()
        self._update_activity_cliff_preview()

    def _on_series_selection_changed(self) -> None:
        if not self._series_scaffolds:
            self._selected_scaffold = ""
            self._update_selected_outputs()
            return

        row_index = self._series_table_widget.currentRow()
        if row_index < 0:
            selected_ranges = self._series_table_widget.selectedRanges()
            if not selected_ranges:
                self._selected_scaffold = ""
                self._update_selected_outputs()
                return
            row_index = selected_ranges[0].topRow()

        if 0 <= row_index < len(self._series_scaffolds):
            self._selected_scaffold = self._series_scaffolds[row_index]
        else:
            self._selected_scaffold = ""
        self._update_selected_outputs()

    def commit(self) -> None:
        if self.data is None:
            self._send_empty()
            self._set_status(format_no_input_status(), ok=False)
            return

        clear_widget_messages(self)
        result = run_chemical_series_explorer(self.data, self._config())
        self._result = result
        self._service_warning_message = summarize_service_issues(
            result.issues,
            subject="chemical series explorer",
            issue_label="issue",
        )

        series_table = chemical_series_table(result)
        members_table = chemical_series_members_table(result)
        summary = chemical_series_summary_table(result)
        send_output_values(
            (self.Outputs.series_table, series_table),
            (self.Outputs.members_table, members_table),
            (self.Outputs.summary_table, summary),
        )

        self._report_browser.setHtml(_render_report_html(result))
        series_rows = series_rows_as_dicts(result)[:100]
        _set_table_rows(self._series_table_widget, series_rows)
        _set_table_rows(self._members_table_widget, chemical_series_members_as_dicts(result)[:200])
        self._series_scaffolds = [str(row.get("scaffold", "")) for row in series_rows]
        self._series_table_widget.blockSignals(True)
        if self._series_scaffolds:
            self._series_table_widget.setCurrentCell(0, 0)
            self._series_table_widget.selectRow(0)
            self._selected_scaffold = self._series_scaffolds[0]
        else:
            self._selected_scaffold = ""
        self._series_table_widget.blockSignals(False)
        self._update_selected_outputs()

        self._set_status(
            format_done_status(
                f"rows={result.summary.n_rows}",
                f"series={result.summary.n_series}",
                f"valid={result.summary.n_valid_molecules}",
            ),
            ok=not any(issue.severity == "error" for issue in result.issues),
        )


__all__ = ["OWChemicalSeriesExplorer"]
