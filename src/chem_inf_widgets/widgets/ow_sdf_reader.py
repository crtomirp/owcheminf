from __future__ import annotations

import os
from typing import List, Optional

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFileDialog, QListWidget, QListWidgetItem
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Output
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState

from chem_inf_widgets.chemcore.io.sdf import read_sdf
from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.from_orange import chemmols_to_table
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_error_status,
    format_no_input_status,
    format_skip_warning,
    set_widget_error,
    set_widget_warning,
)


class OWSdfReader(OWWidget, ConcurrentWidgetMixin):
    name = "SDF Reader"
    description = "Read SDF and output Table and/or Molecules."
    icon = "icons/input_output/owsdfreaderwidget.png"
    priority = 101

    class Outputs:
        data = Output("Data", Table)
        molecules = Output("Molecules", list, auto_summary=False)

    sdf_path: str = Setting("")
    sanitize: bool = Setting(True)
    remove_hs: bool = Setting(True)
    selected_props: List[str] = Setting([])

    def __init__(self) -> None:
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self._available_props: List[str] = []
        self._last_result = None

        box = gui.widgetBox(self.controlArea, "Input file")
        gui.lineEdit(box, self, "sdf_path", label="SDF path", orientation=Qt.Vertical)
        gui.button(box, self, "Browse...", callback=self._browse)

        opts = gui.widgetBox(self.controlArea, "Options")
        gui.checkBox(opts, self, "sanitize", "Sanitize molecules")
        gui.checkBox(opts, self, "remove_hs", "Remove hydrogens")

        prop_box = gui.widgetBox(self.controlArea, "SDF properties to keep")
        self.prop_list = QListWidget()
        self.prop_list.setSelectionMode(QListWidget.MultiSelection)
        prop_box.layout().addWidget(self.prop_list)

        row = gui.hBox(self.controlArea)
        gui.button(row, self, "Refresh properties", callback=self._scan_properties)
        gui.button(self.controlArea, self, "Load", callback=self._start_load)

        status_box = gui.widgetBox(self.controlArea, "Status")
        self.lbl_status = gui.label(status_box, self, "No SDF loaded.")
        self.lbl_status.setWordWrap(True)

        if self.sdf_path:
            self._scan_properties()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SDF file", "", "SDF files (*.sdf *.sd);;All files (*)"
        )
        if path:
            self.sdf_path = path
            self._scan_properties()

    def _scan_properties(self) -> None:
        self.prop_list.clear()
        self._available_props.clear()
        self._last_result = None

        if not self.sdf_path:
            self.lbl_status.setText("No SDF selected.")
            return
        if not os.path.exists(self.sdf_path):
            set_widget_error(self, f"File does not exist:\n{self.sdf_path}")
            self.lbl_status.setText("Selected SDF file does not exist.")
            return

        res = read_sdf(self.sdf_path, sanitize=False, remove_hs=False, max_mols=50)
        keys = set()
        for cm in res.mols:
            keys.update(cm.props.keys())

        self._available_props = sorted(keys)
        for k in self._available_props:
            it = QListWidgetItem(k)
            it.setSelected(k in set(self.selected_props))
            self.prop_list.addItem(it)

        clear_widget_messages(self)
        self.lbl_status.setText(
            f"Scanned preview: {res.n_total} records, {len(res.mols)} readable, {res.n_failed} failed."
        )

    def _start_load(self) -> None:
        if not self.sdf_path:
            set_widget_error(self, "No file selected.")
            self.lbl_status.setText(format_no_input_status("SDF file"))
            return

        self.selected_props = [i.text() for i in self.prop_list.selectedItems()]
        keep = self.selected_props if self.selected_props else None
        self.lbl_status.setText("Loading SDF...")

        self.start(self._task_load, self.sdf_path, bool(self.sanitize), bool(self.remove_hs), keep)

    def _task_load(
        self,
        path: str,
        sanitize: bool,
        remove_hs: bool,
        keep_props: Optional[List[str]],
        state: TaskState,
    ):
        if hasattr(state, "is_interruption_requested") and state.is_interruption_requested():
            return None
        return read_sdf(path, sanitize=sanitize, remove_hs=remove_hs, keep_props=keep_props)

    def on_done(self, result) -> None:
        if result is None:
            self.Outputs.data.send(None)
            self.Outputs.molecules.send(None)
            self.lbl_status.setText("SDF load cancelled.")
            return

        self._last_result = result
        mols = result.mols

        # output molecules
        self.Outputs.molecules.send(mols)

        # output table
        table = chemmols_to_table(mols, prop_keys=self.selected_props or None)
        self.Outputs.data.send(table)
        set_widget_warning(self, format_skip_warning(result.n_failed, subject="SDF records"))
        kept_props = len(self.selected_props or self._available_props)
        self.lbl_status.setText(
            format_done_status(
                f"loaded {len(mols)}/{result.n_total} molecules",
                f"failed={result.n_failed}",
                f"properties kept={kept_props}",
            )
        )

    def on_exception(self, ex: BaseException) -> None:
        set_widget_error(self, str(ex))
        self.Outputs.data.send(None)
        self.Outputs.molecules.send(None)
        self.lbl_status.setText(format_error_status("SDF load failed"))
