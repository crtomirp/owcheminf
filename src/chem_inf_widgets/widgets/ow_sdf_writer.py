from __future__ import annotations

import os
from typing import List, Optional

from AnyQt.QtWidgets import QFileDialog, QListWidget, QListWidgetItem
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState

from chem_inf_widgets.chemcore.io.sdf import write_sdf
from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.from_orange import (
    TableMolConversionReport,
    table_to_chemmols_with_report,
)
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_no_input_status,
    format_skip_warning,
    format_table_report,
    set_widget_error,
    set_widget_information,
    set_widget_warning,
)


class OWSdfWriter(OWWidget, ConcurrentWidgetMixin):
    name = "SDF Writer"
    description = "Write SDF from Table or Molecule objects."
    icon = "icons/input_output/owsdfwriterwidget.png"
    priority = 102

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    sdf_path: str = Setting("out.sdf")
    write_props: bool = Setting(True)
    selected_props: List[str] = Setting([])

    def __init__(self) -> None:
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self._data: Optional[Table] = None
        self._mols: Optional[List[ChemMol]] = None
        self._available_props: List[str] = []
        self._table_report: Optional[TableMolConversionReport] = None
        self._last_save_attempt_count: int = 0

        box = gui.widgetBox(self.controlArea, "Output file")
        gui.lineEdit(box, self, "sdf_path", label="SDF path", orientation="vertical")
        gui.button(box, self, "Browse...", callback=self._browse)

        opts = gui.widgetBox(self.controlArea, "Options")
        gui.checkBox(opts, self, "write_props", "Write properties")

        prop_box = gui.widgetBox(self.controlArea, "Properties to write")
        self.prop_list = QListWidget()
        self.prop_list.setSelectionMode(QListWidget.MultiSelection)
        prop_box.layout().addWidget(self.prop_list)

        row = gui.hBox(self.controlArea)
        gui.button(row, self, "Refresh properties", callback=self._refresh_props)
        gui.button(self.controlArea, self, "Save", callback=self._start_save)

        status_box = gui.widgetBox(self.controlArea, "Status")
        self.lbl_status = gui.label(status_box, self, format_no_input_status())
        self.lbl_status.setWordWrap(True)

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._data = data
        self._table_report = None
        if data is not None:
            self._mols = None
        self._refresh_props()
        self._refresh_status()

    @Inputs.molecules
    def set_molecules(self, mols: Optional[list]) -> None:
        # accept list[ChemMol]
        self._mols = mols
        self._table_report = None
        if mols is not None:
            self._data = None
        self._refresh_props()
        self._refresh_status()

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save SDF file", self.sdf_path or "out.sdf", "SDF files (*.sdf *.sd);;All files (*)"
        )
        if path:
            self.sdf_path = path

    def _refresh_props(self) -> None:
        self.prop_list.clear()
        self._available_props.clear()

        if self._mols:
            keys = set()
            for cm in self._mols:
                keys.update(cm.props.keys())
            self._available_props = sorted(keys)

        elif self._data is not None:
            try:
                _mols, self._table_report = table_to_chemmols_with_report(self._data)
            except Exception:
                self._table_report = None
            # candidates = all metas/attrs except Name/SMILES
            all_vars = list(self._data.domain.metas) + list(self._data.domain.attributes) + list(self._data.domain.class_vars)
            exclude = {"name", "smiles"}
            self._available_props = [v.name for v in all_vars if v.name.lower() not in exclude]

        else:
            return

        for k in self._available_props:
            it = QListWidgetItem(k)
            it.setSelected(k in set(self.selected_props))
            self.prop_list.addItem(it)

    def _refresh_status(self) -> None:
        if self._mols is not None:
            self.lbl_status.setText(f"Input: {len(self._mols)} molecules ready for SDF export.")
            return
        if self._data is not None:
            if self._table_report is not None:
                self.lbl_status.setText(
                    format_table_report(
                        self._table_report,
                        prefix="Input table",
                        valid_label="valid SMILES",
                    )
                )
            else:
                self.lbl_status.setText(f"Input table: rows={len(self._data)}")
            return
        self.lbl_status.setText(format_no_input_status())

    def _start_save(self) -> None:
        clear_widget_messages(self, information=True)
        if not self.sdf_path:
            set_widget_error(self, "No output path selected.")
            self.lbl_status.setText("No output path selected.")
            return

        out_dir = os.path.dirname(os.path.abspath(self.sdf_path)) or "."
        if not os.path.exists(out_dir):
            set_widget_error(self, f"Output directory does not exist:\n{out_dir}")
            self.lbl_status.setText("Output directory does not exist.")
            return

        self.selected_props = [i.text() for i in self.prop_list.selectedItems()]
        prop_cols = self.selected_props if (self.write_props and self.selected_props) else None

        # normalize to list[ChemMol]
        if self._mols is not None:
            mols = self._mols
            self._last_save_attempt_count = len(mols)
            set_widget_warning(self, "")
        elif self._data is not None:
            mols, report = table_to_chemmols_with_report(self._data)
            self._table_report = report
            self._last_save_attempt_count = report.n_valid
            set_widget_warning(self, format_skip_warning(report.n_invalid, action="were skipped before SDF export"))
        else:
            set_widget_error(self, "No input data.")
            self.lbl_status.setText("No input data.")
            return

        # Filtriranje lastnosti se zgodi tukaj (če write_sdf to podpira)
        include_props = False if not self.write_props else (prop_cols if prop_cols is not None else True)
        self.lbl_status.setText(f"Writing SDF to {os.path.basename(self.sdf_path)}...")

        self.start(self._task_write, mols, self.sdf_path, include_props)

    def _task_write(self, mols: List[ChemMol], path: str, include_props, state: TaskState) -> int:
        if hasattr(state, "is_interruption_requested") and state.is_interruption_requested():
            return 0
        return write_sdf(mols, path, include_props=include_props)

    def on_done(self, result) -> None:
        set_widget_information(self, f"Wrote {result} molecules to:\n{self.sdf_path}")
        self.lbl_status.setText(
            format_done_status(
                f"wrote {result}/{self._last_save_attempt_count} molecules",
                f"file={os.path.basename(self.sdf_path)}",
            )
        )

    def on_exception(self, ex: BaseException) -> None:
        set_widget_error(self, str(ex))
        self.lbl_status.setText("SDF export failed.")
