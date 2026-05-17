from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from AnyQt.QtCore import Qt, QTimer, pyqtSlot
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QSplitter,
)

from Orange.data import Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import OWWidget, Output

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.chembl_target_service import ChemBLTargetService
from chem_inf_widgets.chemcore.services.chembl_assay_service import ChemBLAssayService
from chem_inf_widgets.chemcore.services.chembl_bioactivity_service import ChemBLBioactivityService
from chem_inf_widgets.chemcore.services.chembl_molecule_service import ChemBLMoleculeService, ChemBLMoleculePropsRecord
from chem_inf_widgets.chemcore.services.chembl_browser_service import (
    SummaryRow,
    compile_user_pattern,
    filter_targets,
    format_number,
    format_output_summary,
    query_needs_postfilter,
    summarize_activity_types,
)
from chem_inf_widgets.chemcore.services.chembl_models import (
    ChemBLTargetRecord,
    ChemBLAssayRecord,
    ChemBLBioactivityRecord,
    ChemBLMoleculeRecord,
)
from chem_inf_widgets.chemcore.services.chembl_output_builder import (
    aggregate_bio_by_molecule,
    build_bioactivity_outputs,
    build_molecule_outputs,
    derive_prop_keys_from_records,
)
from chem_inf_widgets.widgets.ui_helpers import (
    format_done_status,
    format_error_status,
    format_no_input_status,
)

try:
    from chem_inf_widgets.chemcore.services.disk_cache import CachePolicy
except Exception:  # pragma: no cover
    CachePolicy = None  # type: ignore

logger = logging.getLogger(__name__)


def _safe_fetch_props_by_id(
    svc_mols: ChemBLMoleculeService,
    ids: Sequence[str],
    prop_keys: Sequence[str],
) -> tuple[Dict[str, ChemBLMoleculePropsRecord], str]:
    normalized_ids = [str(value or "").strip().upper() for value in ids]
    normalized_ids = [value for value in normalized_ids if value]
    if not prop_keys or not normalized_ids:
        return {}, ""
    try:
        props = svc_mols.fetch_molecules_with_properties(list(dict.fromkeys(normalized_ids))[:1500], list(prop_keys))
        return {p.chembl_id.strip().upper(): p for p in props}, ""
    except Exception as exc:
        logger.warning("Could not enrich ChEMBL bioactivity outputs with molecule properties.", exc_info=True)
        return {}, f"Property enrichment skipped: {exc}"

def _selected_row_indices(table: QTableWidget) -> List[int]:
    selection_model = table.selectionModel()
    if selection_model is None:
        return []
    return [row.row() for row in selection_model.selectedRows() if row.row() >= 0]


class OWChemBLBrowser(OWWidget):
    name = "ChEMBL Browser"
    description = "ChEMBL browser with exportable properties + bioactivities for QSAR/clustering."
    icon = "icons/data_retrieval/owchemblbrowserwidget.png"
    priority = 104

    class Outputs:
        data = Output("Data", Table)
        molecules = Output("Molecules", list, auto_summary=False)
        selected_data = Output("Selected Data", Table)
        selected_molecules = Output("Selected Molecules", list, auto_summary=False)

    # persisted settings
    target_query: str = Setting("EGFR")
    target_limit: int = Setting(50)
    target_filter: str = Setting("")

    assay_min_conf: int = Setting(7)
    assay_type: str = Setting("ANY")

    activity_standard_type: str = Setting("IC50")
    activity_limit: int = Setting(1000)

    mol_from_target_standard_type: str = Setting("ANY")
    mol_from_target_limit: int = Setting(999)

    debounce_ms: int = Setting(450)
    auto_search_min_chars: int = Setting(2)

    auto_fetch_bio_on_load: bool = Setting(True)

    include_props_in_table: bool = Setting(True)
    include_props_in_molecules: bool = Setting(True)

    selected_prop_keys: List[str] = Setting([])
    selected_bio_fields: List[str] = Setting([])

    # bio field picker (what can be joined into the Molecules table as aggregated values)
    BIO_FIELD_SPECS: List[Tuple[str, str]] = [
        ("pChEMBL", "num"),
        ("standard_value", "num"),
        ("IC50_nM", "num"),
        ("standard_type", "meta"),
        ("standard_units", "meta"),
        ("assay_chembl_id", "meta"),
        ("target_chembl_id", "meta"),
        ("molecule_chembl_id", "meta"),
        ("pref_name", "meta"),
        ("SMILES", "smiles"),
    ]

    def __init__(self):
        super().__init__()
        self.executor = ThreadExecutor(self)

        cache_policy = None
        if CachePolicy is not None:
            cache_policy = CachePolicy(enabled=True, ttl_s=6 * 3600)

        if cache_policy is not None:
            self.svc_targets = ChemBLTargetService(timeout_s=60, retries=3, cache_policy=cache_policy)
            self.svc_assays = ChemBLAssayService(timeout_s=60, retries=3, cache_policy=cache_policy)
        else:
            self.svc_targets = ChemBLTargetService(timeout_s=60, retries=3)
            self.svc_assays = ChemBLAssayService(timeout_s=60, retries=3)

        self.svc_bio = ChemBLBioactivityService(timeout_s=60, retries=3)
        self.svc_mols = ChemBLMoleculeService(timeout_s=60, retries=3)

        # state
        self._targets_raw: List[ChemBLTargetRecord] = []
        self._targets: List[ChemBLTargetRecord] = []
        self._assays: List[ChemBLAssayRecord] = []
        self._bio: List[ChemBLBioactivityRecord] = []
        self._molecules: List[ChemBLMoleculeRecord] = []

        self._selected_target: Optional[ChemBLTargetRecord] = None

        self._last_table: Optional[Table] = None
        self._last_molecules: List[ChemMol] = []

        self._available_prop_keys: List[str] = []
        self._mol_props_by_id: Dict[str, ChemBLMoleculePropsRecord] = {}

        # coordination for auto-bio fetch on Molecules load
        self._auto_bio_inflight: bool = False
        self._auto_bio_ready: bool = False
        self._mol_props_ready: bool = False
        self._molecule_outputs_built: bool = False

        # defaults
        if not self.selected_bio_fields:
            self.selected_bio_fields = [
                "pChEMBL",
                "standard_value",
                "standard_type",
                "standard_units",
                "assay_chembl_id",
                "target_chembl_id",
                "molecule_chembl_id",
            ]

        # debounce for target search
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._debounced_target_search)

        self._build_ui()
        self._apply_styles()

        self._refresh_property_keys_background(sample_ids=None)
        self._set_busy(False, "Ready")

    # ---------------- UI / style ----------------

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
        QLabel#HeaderTitle { font-size: 16px; font-weight: 650; }
        QLabel#HeaderSub { color: #667085; font-size: 12px; }
        """
        )

    def _build_ui(self) -> None:
        self.mainArea.hide()
        root = self.controlArea
        root.setMinimumWidth(760)

        header = QWidget(self)
        hb = QHBoxLayout(header)
        hb.setContentsMargins(6, 6, 6, 6)

        vb = QVBoxLayout()
        vb.addWidget(QLabel("ChEMBL Browser", objectName="HeaderTitle"))
        vb.addWidget(QLabel("Targets → Assays → Bioactivities/Molecules. Exports as Table + ChemMol.", objectName="HeaderSub"))
        hb.addLayout(vb, 1)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hb.addWidget(self.lbl_status)

        root.layout().addWidget(header)

        search_box = QGroupBox("Targets search")
        g = QGridLayout(search_box)

        self.target_edit = QLineEdit(self.target_query)
        self.target_edit.setPlaceholderText("Server query (EGFR, DRD2, kinase, CHEMBL...)")
        self.target_edit.textChanged.connect(self._on_target_text_changed)

        self.spin_target_limit = QSpinBox()
        self.spin_target_limit.setRange(1, 500)
        self.spin_target_limit.setValue(int(self.target_limit))

        self.btn_search_targets = QPushButton("Search")
        self.btn_search_targets.clicked.connect(self._on_search_targets)

        self.target_filter_edit = QLineEdit(self.target_filter)
        self.target_filter_edit.setPlaceholderText("Local filter: CYP*  /kinase$/  etc.")
        self.target_filter_edit.textChanged.connect(self._on_target_filter_changed)

        self.btn_export_csv = QPushButton("Export CSV (Data)")
        self.btn_export_csv.clicked.connect(self._on_export_csv)

        g.addWidget(QLabel("Query"), 0, 0)
        g.addWidget(self.target_edit, 0, 1, 1, 3)
        g.addWidget(QLabel("Limit"), 0, 4)
        g.addWidget(self.spin_target_limit, 0, 5)
        g.addWidget(self.btn_search_targets, 0, 6)

        g.addWidget(QLabel("Filter"), 1, 0)
        g.addWidget(self.target_filter_edit, 1, 1, 1, 5)
        g.addWidget(self.btn_export_csv, 1, 6)

        root.layout().addWidget(search_box)

        self.tabs = QTabWidget()
        root.layout().addWidget(self.tabs)

        self.tab_targets = QWidget()
        self.tab_assays = QWidget()
        self.tab_bio = QWidget()
        self.tab_mols = QWidget()

        self.tabs.addTab(self.tab_targets, "Targets")
        self.tabs.addTab(self.tab_assays, "Assays")
        self.tabs.addTab(self.tab_bio, "Bioactivities")
        self.tabs.addTab(self.tab_mols, "Molecules")

        self._build_targets_tab()
        self._build_assays_tab()
        self._build_bio_tab()
        self._build_mols_tab()

    def _build_targets_tab(self) -> None:
        layout = QVBoxLayout(self.tab_targets)
        self.tbl_targets = QTableWidget(0, 4)
        self.tbl_targets.setHorizontalHeaderLabels(["ChEMBL ID", "Name", "Organism", "Type"])
        self.tbl_targets.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_targets.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_targets.itemSelectionChanged.connect(self._on_target_selected)
        layout.addWidget(self.tbl_targets)

    def _build_assays_tab(self) -> None:
        layout = QVBoxLayout(self.tab_assays)

        filters = QGroupBox("Assay filters")
        fl = QHBoxLayout(filters)

        self.spin_conf = QSpinBox()
        self.spin_conf.setRange(0, 9)
        self.spin_conf.setValue(int(self.assay_min_conf))

        self.cmb_assay_type = QComboBox()
        self.cmb_assay_type.addItems(["ANY", "B", "F", "A"])
        self.cmb_assay_type.setCurrentText(self.assay_type)

        self.btn_load_assays = QPushButton("Load Assays for Selected Target")
        self.btn_load_assays.clicked.connect(self._on_load_assays)

        fl.addWidget(QLabel("Min confidence"))
        fl.addWidget(self.spin_conf)
        fl.addSpacing(10)
        fl.addWidget(QLabel("Assay type"))
        fl.addWidget(self.cmb_assay_type)
        fl.addStretch(1)
        fl.addWidget(self.btn_load_assays)

        layout.addWidget(filters)

        self.tbl_assays = QTableWidget(0, 5)
        self.tbl_assays.setHorizontalHeaderLabels(["Assay ID", "Type", "Conf", "Organism", "Description"])
        self.tbl_assays.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_assays.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.tbl_assays)

    def _build_bio_tab(self) -> None:
        layout = QVBoxLayout(self.tab_bio)

        ctl = QGroupBox("Bioactivity fetch")
        hb = QHBoxLayout(ctl)

        self.edit_std_type = QLineEdit(self.activity_standard_type)
        self.edit_std_type.setPlaceholderText("Standard type (IC50, Ki, EC50, Kd, Inhibition...)")

        self.spin_act_limit = QSpinBox()
        self.spin_act_limit.setRange(1, 5000)
        self.spin_act_limit.setValue(int(self.activity_limit))

        self.btn_fetch_bio = QPushButton("Fetch Bioactivities")
        self.btn_fetch_bio.clicked.connect(self._on_fetch_bio)

        hb.addWidget(QLabel("standard_type"))
        hb.addWidget(self.edit_std_type, 2)
        hb.addWidget(QLabel("limit"))
        hb.addWidget(self.spin_act_limit)
        hb.addWidget(self.btn_fetch_bio)

        layout.addWidget(ctl)

        rep = QGroupBox("Activity type summary")
        rv = QVBoxLayout(rep)
        self.tbl_report = QTableWidget(0, 3)
        self.tbl_report.setHorizontalHeaderLabels(["Type", "Count", "%"])
        rv.addWidget(self.tbl_report)
        layout.addWidget(rep)

        self.tbl_bio = QTableWidget(0, 6)
        self.tbl_bio.setHorizontalHeaderLabels(["Molecule", "SMILES", "Type", "Value", "Units", "pChEMBL"])
        self.tbl_bio.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_bio.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.tbl_bio)

        row = QWidget()
        hr = QHBoxLayout(row)
        hr.setContentsMargins(0, 0, 0, 0)

        self.btn_send_selected_bio = QPushButton("Send selected rows only (Bioactivities)")
        self.btn_send_selected_bio.clicked.connect(self._on_send_selected_bio)

        hr.addStretch(1)
        hr.addWidget(self.btn_send_selected_bio)
        layout.addWidget(row)

    def _build_mols_tab(self) -> None:
        layout = QVBoxLayout(self.tab_mols)
        splitter = QSplitter(Qt.Horizontal)

        # left panel: settings / pickers
        left = QWidget()
        lv = QVBoxLayout(left)

        load_box = QGroupBox("Load molecules from selected target")
        g = QGridLayout(load_box)

        self.edit_selected_target = QLineEdit()
        self.edit_selected_target.setReadOnly(True)
        self.edit_selected_target.setPlaceholderText("Select a target in Targets tab…")

        self.cmb_mol_standard_type = QComboBox()
        self.cmb_mol_standard_type.setEditable(True)
        self.cmb_mol_standard_type.addItems(["ANY", "IC50", "Ki", "EC50", "Kd", "Inhibition", "Activity"])
        self.cmb_mol_standard_type.setCurrentText(self.mol_from_target_standard_type or "ANY")

        self.spin_mol_limit = QSpinBox()
        self.spin_mol_limit.setRange(1, 5000)
        self.spin_mol_limit.setValue(int(self.mol_from_target_limit))

        self.btn_load_mols = QPushButton("Load")
        self.btn_load_mols.clicked.connect(self._on_load_molecules_from_target)

        self.chk_auto_bio = QCheckBox("Auto-fetch bioactivities (for aggregated bio)")
        self.chk_auto_bio.setChecked(bool(self.auto_fetch_bio_on_load))
        self.chk_auto_bio.toggled.connect(self._on_auto_bio_toggled)

        g.addWidget(QLabel("Target"), 0, 0)
        g.addWidget(self.edit_selected_target, 0, 1, 1, 3)

        g.addWidget(QLabel("standard_type"), 1, 0)
        g.addWidget(self.cmb_mol_standard_type, 1, 1)
        g.addWidget(QLabel("limit"), 1, 2)
        g.addWidget(self.spin_mol_limit, 1, 3)
        g.addWidget(self.btn_load_mols, 1, 4)

        g.addWidget(self.chk_auto_bio, 2, 1, 1, 4)

        lv.addWidget(load_box)
        lv.addWidget(self._build_pickers_box())
        lv.addStretch(1)

        # right panel: table
        right = QWidget()
        rv = QVBoxLayout(right)

        table_box = QGroupBox("Molecules table (properties + aggregated bio)")
        tv = QVBoxLayout(table_box)

        self.tbl_mols = QTableWidget(0, 3)
        self.tbl_mols.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_mols.setSelectionMode(QAbstractItemView.ExtendedSelection)
        tv.addWidget(self.tbl_mols)

        actions = QWidget()
        ah = QHBoxLayout(actions)
        ah.setContentsMargins(0, 0, 0, 0)

        self.btn_send_selected_mols = QPushButton("Send selected rows only (Molecules)")
        self.btn_send_selected_mols.clicked.connect(self._on_send_selected_molecules)

        self.btn_export_sdf = QPushButton("Export SDF (selected if any, else all)")
        self.btn_export_sdf.clicked.connect(self._on_export_sdf)

        ah.addWidget(self.btn_export_sdf)
        ah.addStretch(1)
        ah.addWidget(self.btn_send_selected_mols)
        tv.addWidget(actions)

        rv.addWidget(table_box)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([340, 900])

        layout.addWidget(splitter)
        self._render_molecules_table()

    def _build_pickers_box(self) -> QGroupBox:
        box = QGroupBox("Export fields")
        v = QVBoxLayout(box)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)

        self.btn_refresh_props = QPushButton("Refresh property keys")
        self.btn_refresh_props.clicked.connect(self._on_refresh_prop_keys_clicked)

        self.btn_props_in_table = QPushButton("Props in Table: ON" if self.include_props_in_table else "Props in Table: OFF")
        self.btn_props_in_table.setCheckable(True)
        self.btn_props_in_table.setChecked(bool(self.include_props_in_table))
        self.btn_props_in_table.clicked.connect(self._on_flags_changed)

        self.btn_props_in_mols = QPushButton("Props in Molecules: ON" if self.include_props_in_molecules else "Props in Molecules: OFF")
        self.btn_props_in_mols.setCheckable(True)
        self.btn_props_in_mols.setChecked(bool(self.include_props_in_molecules))
        self.btn_props_in_mols.clicked.connect(self._on_flags_changed)

        h.addWidget(self.btn_props_in_table)
        h.addWidget(self.btn_props_in_mols)
        h.addStretch(1)
        h.addWidget(self.btn_refresh_props)
        v.addWidget(row)

        v.addWidget(QLabel("Bio fields (aggregated per molecule):"))
        self.list_bio_fields = QListWidget()
        self.list_bio_fields.setSelectionMode(QAbstractItemView.NoSelection)
        self.list_bio_fields.itemChanged.connect(self._on_bio_fields_changed)
        v.addWidget(self.list_bio_fields)

        v.addWidget(QLabel("Molecule properties (molecule_properties):"))
        self.list_props = QListWidget()
        self.list_props.setSelectionMode(QAbstractItemView.NoSelection)
        self.list_props.itemChanged.connect(self._on_prop_keys_changed)
        v.addWidget(self.list_props)

        self._render_bio_fields_list()
        self._render_props_list()

        return box

    def _on_auto_bio_toggled(self, checked: bool) -> None:
        self.auto_fetch_bio_on_load = bool(checked)

    # ---------------- status / busy ----------------

    @pyqtSlot(bool, str)
    def _set_busy(self, busy: bool, msg: str = "") -> None:
        for w in [
            self.btn_search_targets,
            self.btn_load_assays,
            self.btn_fetch_bio,
            self.btn_export_csv,
            self.btn_load_mols,
            self.btn_send_selected_bio,
            self.btn_send_selected_mols,
            self.btn_export_sdf,
            self.btn_refresh_props,
        ]:
            w.setEnabled(not busy)
        if msg:
            self.lbl_status.setText(msg)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._set_busy(False, format_error_status(msg))

    def _invoke_slot(self, slot_name: str, slot_types: Tuple[type, ...], *args: Any) -> None:
        methodinvoke(self, slot_name, slot_types)(*args)

    def _dispatch_future_result(
        self,
        fut,
        slot_name: str,
        slot_types: Tuple[type, ...],
        error_prefix: str,
        *slot_args: Any,
        fallback_result: Any = None,
        use_fallback: bool = False,
    ) -> None:
        try:
            result = fut.result()
        except Exception as exc:
            if use_fallback:
                self._invoke_slot(slot_name, slot_types, fallback_result, *slot_args)
                return
            self._invoke_slot("_on_error", (str,), f"{error_prefix}: {exc}")
            return
        self._invoke_slot(slot_name, slot_types, result, *slot_args)

    # ---------------- debounce targets ----------------

    def _on_target_text_changed(self, _txt: str) -> None:
        self._debounce_timer.start(int(self.debounce_ms))

    def _debounced_target_search(self) -> None:
        q = self.target_edit.text().strip()
        if len(q) < int(self.auto_search_min_chars):
            return
        self._on_search_targets()

    def _on_target_filter_changed(self, text: str) -> None:
        self.target_filter = text
        self._apply_target_filter()

    # ---------------- targets ----------------

    def _on_search_targets(self) -> None:
        q = self.target_edit.text().strip()
        self.target_query = q
        self.target_limit = int(self.spin_target_limit.value())
        if not q:
            self.lbl_status.setText("Enter a query.")
            return

        broad_q = q
        if (q.startswith("/") and q.endswith("/")) or ("*" in q) or ("?" in q):
            broad_q = re.sub(r"[*?/]+", " ", q).strip() or "kinase"

        self._set_busy(True, f"Searching targets for '{broad_q}' …")
        t0 = time.perf_counter()
        fut = self.executor.submit(self.svc_targets.search, broad_q, int(self.target_limit))
        fut.add_done_callback(lambda f: self._on_targets_ready(f, t0, q))

    def _on_targets_ready(self, fut, t0: float, original_query: str) -> None:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        self._dispatch_future_result(
            fut,
            "_update_targets",
            (object, int, str),
            "Target search failed",
            dt_ms,
            original_query,
        )

    @pyqtSlot(object, int, str)
    def _update_targets(self, targets, dt_ms: int, original_query: str) -> None:
        self._targets_raw = list(targets or [])

        auto_pat = compile_user_pattern(original_query)
        if auto_pat is not None and query_needs_postfilter(original_query):
            self._targets_raw = filter_targets(self._targets_raw, auto_pat)

        self._apply_target_filter()

        self._selected_target = None
        self._assays = []
        self._bio = []
        self._molecules = []
        self._mol_props_by_id = {}

        self.edit_selected_target.setText("")
        self._fill_assays([])
        self._fill_bio([])
        self._fill_report([])
        self._render_molecules_table()

        self.Outputs.data.send(None)
        self.Outputs.molecules.send([])
        self.Outputs.selected_data.send(None)
        self.Outputs.selected_molecules.send([])

        self._set_busy(False, f"Found {len(self._targets_raw)} targets in {dt_ms} ms (filtered: {len(self._targets)}).")
        self.tabs.setCurrentWidget(self.tab_targets)

    def _apply_target_filter(self) -> None:
        self._targets = filter_targets(self._targets_raw, compile_user_pattern(self.target_filter))
        self._fill_targets()

    def _fill_targets(self) -> None:
        self.tbl_targets.setRowCount(0)
        for t in self._targets:
            r = self.tbl_targets.rowCount()
            self.tbl_targets.insertRow(r)
            self.tbl_targets.setItem(r, 0, QTableWidgetItem(t.chembl_id))
            self.tbl_targets.setItem(r, 1, QTableWidgetItem(t.pref_name))
            self.tbl_targets.setItem(r, 2, QTableWidgetItem(t.organism))
            self.tbl_targets.setItem(r, 3, QTableWidgetItem(t.target_type))
        self.tbl_targets.resizeColumnsToContents()

    def _on_target_selected(self) -> None:
        rows = _selected_row_indices(self.tbl_targets)
        if not rows:
            self._selected_target = None
            self.edit_selected_target.setText("")
            return

        idx = rows[0]
        if 0 <= idx < len(self._targets):
            self._selected_target = self._targets[idx]
            self.edit_selected_target.setText(f"{self._selected_target.chembl_id} • {self._selected_target.pref_name}")
            self.lbl_status.setText(f"Selected target: {self._selected_target.chembl_id}")

    # ---------------- assays ----------------

    def _on_load_assays(self) -> None:
        if self._selected_target is None:
            self.lbl_status.setText("Select a target first.")
            return

        self.assay_min_conf = int(self.spin_conf.value())
        self.assay_type = self.cmb_assay_type.currentText().strip()

        self._set_busy(True, f"Loading assays for {self._selected_target.chembl_id} …")
        t0 = time.perf_counter()
        fut = self.executor.submit(
            self.svc_assays.fetch_for_target,
            self._selected_target.chembl_id,
            int(self.assay_min_conf),
            self.assay_type,
        )
        fut.add_done_callback(lambda f: self._on_assays_ready(f, t0))

    def _on_assays_ready(self, fut, t0: float) -> None:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        self._dispatch_future_result(
            fut,
            "_update_assays",
            (object, int),
            "Assay load failed",
            dt_ms,
        )

    @pyqtSlot(object, int)
    def _update_assays(self, assays, dt_ms: int) -> None:
        self._assays = list(assays or [])
        self._fill_assays(self._assays)
        self._set_busy(False, f"Loaded {len(self._assays)} assays in {dt_ms} ms.")
        self.tabs.setCurrentWidget(self.tab_assays)

    def _fill_assays(self, assays: List[ChemBLAssayRecord]) -> None:
        self.tbl_assays.setRowCount(0)
        for a in assays:
            r = self.tbl_assays.rowCount()
            self.tbl_assays.insertRow(r)
            self.tbl_assays.setItem(r, 0, QTableWidgetItem(a.assay_chembl_id))
            self.tbl_assays.setItem(r, 1, QTableWidgetItem(a.assay_type))
            self.tbl_assays.setItem(r, 2, QTableWidgetItem("" if a.confidence_score is None else str(a.confidence_score)))
            self.tbl_assays.setItem(r, 3, QTableWidgetItem(a.organism))
            self.tbl_assays.setItem(r, 4, QTableWidgetItem(a.description))
        self.tbl_assays.resizeColumnsToContents()

    # ---------------- bioactivities ----------------

    def _on_fetch_bio(self) -> None:
        if self._selected_target is None:
            self.lbl_status.setText("Select a target first.")
            return

        self.activity_standard_type = self.edit_std_type.text().strip()
        self.activity_limit = int(self.spin_act_limit.value())

        self._set_busy(True, f"Fetching bioactivities for {self._selected_target.chembl_id} …")
        t0 = time.perf_counter()
        fut = self.executor.submit(
            self.svc_bio.fetch_for_target,
            self._selected_target.chembl_id,
            self.activity_standard_type,
            int(self.activity_limit),
        )
        fut.add_done_callback(lambda f: self._on_bio_ready(f, t0))

    def _on_bio_ready(self, fut, t0: float) -> None:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        self._dispatch_future_result(
            fut,
            "_update_bio",
            (object, int),
            "Bioactivity fetch failed",
            dt_ms,
        )

    @pyqtSlot(object, int)
    def _update_bio(self, recs, dt_ms: int) -> None:
        self._bio = list(recs or [])
        self._fill_bio(self._bio)
        self._fill_report(summarize_activity_types(self._bio))

        # Build outputs (Data + ChemMol) from bioactivities
        self._set_busy(True, "Building outputs (bioactivities) …")
        fut = self.executor.submit(self._build_outputs_from_bioactivities, self._bio, list(self.selected_prop_keys or []))
        fut.add_done_callback(self._on_outputs_ready)

        # Also refresh molecules table if it already exists (bio aggregation columns)
        if self._molecules:
            self._render_molecules_table()

        self._set_busy(False, f"Fetched {len(self._bio)} bioactivities in {dt_ms} ms.")
        self.tabs.setCurrentWidget(self.tab_bio)

    def _fill_bio(self, recs: List[ChemBLBioactivityRecord]) -> None:
        self.tbl_bio.setRowCount(0)
        for r in recs:
            row = self.tbl_bio.rowCount()
            self.tbl_bio.insertRow(row)
            self.tbl_bio.setItem(row, 0, QTableWidgetItem(str(getattr(r, "molecule_chembl_id", "") or "")))
            self.tbl_bio.setItem(row, 1, QTableWidgetItem(str(getattr(r, "smiles", "") or "")))
            self.tbl_bio.setItem(row, 2, QTableWidgetItem(str(getattr(r, "standard_type", "") or "")))
            self.tbl_bio.setItem(row, 3, QTableWidgetItem(format_number(getattr(r, "standard_value", None), 3)))
            self.tbl_bio.setItem(row, 4, QTableWidgetItem(str(getattr(r, "standard_units", "") or "")))
            self.tbl_bio.setItem(row, 5, QTableWidgetItem(format_number(getattr(r, "pchembl_value", None), 2)))
        self.tbl_bio.resizeColumnsToContents()

    def _fill_report(self, rows: List[SummaryRow]) -> None:
        self.tbl_report.setRowCount(0)
        for r in rows:
            i = self.tbl_report.rowCount()
            self.tbl_report.insertRow(i)
            self.tbl_report.setItem(i, 0, QTableWidgetItem(r.key))
            self.tbl_report.setItem(i, 1, QTableWidgetItem(str(r.count)))
            self.tbl_report.setItem(i, 2, QTableWidgetItem(format_number(r.pct, 1)))
        self.tbl_report.resizeColumnsToContents()

    # ---------------- molecules load ----------------

    def _on_load_molecules_from_target(self) -> None:
        if self._selected_target is None:
            self.lbl_status.setText("Select a target first.")
            return

        # reset coordination flags for this load
        self._auto_bio_inflight = False
        self._auto_bio_ready = False
        self._mol_props_ready = False
        self._molecule_outputs_built = False

        stype = (self.cmb_mol_standard_type.currentText() or "").strip()
        self.mol_from_target_standard_type = stype or "ANY"
        limit = int(self.spin_mol_limit.value())
        self.mol_from_target_limit = limit

        std = None if (not stype or stype.upper() == "ANY") else stype

        if bool(self.auto_fetch_bio_on_load):
            self._start_auto_bio_fetch()

        self._set_busy(True, f"Loading molecules for {self._selected_target.chembl_id} ({stype or 'ANY'}) …")
        t0 = time.perf_counter()
        fut = self.executor.submit(self.svc_mols.fetch_molecules_for_target, self._selected_target.chembl_id, std, limit)
        fut.add_done_callback(lambda f: self._on_molecules_ready(f, t0, std, limit))

    def _on_molecules_ready(self, fut, t0: float, std: Optional[str], limit: int) -> None:
        try:
            mols = fut.result()
            dt_ms = int((time.perf_counter() - t0) * 1000)

            # if type-filter returns nothing, retry ANY
            if (not mols) and std:
                methodinvoke(self, "_set_busy", (bool, str))(True, f"No molecules for '{std}'. Retrying ANY …")
                fut2 = self.executor.submit(self.svc_mols.fetch_molecules_for_target, self._selected_target.chembl_id, None, limit)
                fut2.add_done_callback(lambda f2: self._on_molecules_ready(f2, time.perf_counter(), None, limit))
                return

            methodinvoke(self, "_update_molecules", (object, int))(mols, dt_ms)
        except Exception as e:
            methodinvoke(self, "_on_error", (str,))(f"Molecule load failed: {e}")

    @pyqtSlot(object, int)
    def _update_molecules(self, mols, dt_ms: int) -> None:
        self._molecules = list(mols or [])
        self._mol_props_by_id = {}
        self._render_molecules_table()

        if not self._molecules:
            self._set_busy(False, f"No molecules found in {dt_ms} ms.")
            self.tabs.setCurrentWidget(self.tab_mols)
            self.Outputs.data.send(None)
            self.Outputs.molecules.send([])
            return

        ids = [m.chembl_id for m in self._molecules if m.chembl_id]
        keys = list(self.selected_prop_keys or [])

        # fetch properties in background
        self._set_busy(True, f"Fetching properties for {len(ids)} molecules …")
        fut = self.executor.submit(self._fetch_molecule_props_for_table, ids, keys)
        fut.add_done_callback(self._on_mol_props_ready)

        self._set_busy(False, f"Loaded {len(self._molecules)} molecules in {dt_ms} ms.")
        self.tabs.setCurrentWidget(self.tab_mols)

    def _fetch_molecule_props_for_table(self, ids: List[str], prop_keys: List[str]) -> Dict[str, ChemBLMoleculePropsRecord]:
        ids = (ids or [])[:1500]
        recs = self.svc_mols.fetch_molecules_with_properties(ids, prop_keys if prop_keys else None)
        return {r.chembl_id.strip().upper(): r for r in recs}

    def _on_mol_props_ready(self, fut) -> None:
        self._dispatch_future_result(
            fut,
            "_apply_mol_props",
            (object,),
            "Fetching molecule properties failed",
        )

    @pyqtSlot(object)
    def _apply_mol_props(self, mp) -> None:
        self._mol_props_by_id = dict(mp or {})
        # If nothing is selected yet, infer a sensible set of keys from fetched molecule_properties.
        self._ensure_selected_prop_keys(self._mol_props_by_id)
        self._render_molecules_table()

        self._mol_props_ready = True
        self._maybe_build_molecule_outputs()

    def _maybe_build_molecule_outputs(self) -> None:
        """Build Molecules outputs once required inputs are ready.

        If auto-bio is enabled, wait until the auto bioactivity fetch finishes so aggregated bio
        columns are present in both Table and ChemMol outputs.
        """
        if self._molecule_outputs_built:
            return
        if not self._molecules:
            return
        if not self._mol_props_ready:
            return
        if bool(self.auto_fetch_bio_on_load) and (self._auto_bio_inflight or not self._auto_bio_ready):
            return

        self._molecule_outputs_built = True
        self._set_busy(True, "Building outputs (molecules) …")
        fut = self.executor.submit(
            self._build_outputs_from_molecules,
            self._molecules,
            self._mol_props_by_id,
            list(self.selected_prop_keys or []),
        )
        fut.add_done_callback(self._on_molecule_outputs_ready)

    def _start_auto_bio_fetch(self) -> None:
        if self._selected_target is None:
            return
        if self._auto_bio_inflight:
            return

        std_type = (self.edit_std_type.text().strip() if hasattr(self, "edit_std_type") else "").strip() or self.activity_standard_type
        limit = int(self.spin_act_limit.value()) if hasattr(self, "spin_act_limit") else int(self.activity_limit)

        self._auto_bio_inflight = True
        self._auto_bio_ready = False

        fut = self.executor.submit(
            self.svc_bio.fetch_for_target,
            self._selected_target.chembl_id,
            std_type,
            limit,
        )
        fut.add_done_callback(self._on_auto_bio_ready)

    def _on_auto_bio_ready(self, fut) -> None:
        self._dispatch_future_result(
            fut,
            "_apply_auto_bio_records",
            (object,),
            "Auto bioactivity fetch failed",
            fallback_result=[],
            use_fallback=True,
        )

    @pyqtSlot(object)
    def _apply_auto_bio_records(self, recs) -> None:
        self._bio = list(recs or [])
        self._auto_bio_inflight = False
        self._auto_bio_ready = True
        self._render_molecules_table()
        self._maybe_build_molecule_outputs()

    def _on_molecule_outputs_ready(self, fut) -> None:
        try:
            table, mols = fut.result()
        except Exception as exc:
            self._invoke_slot("_on_error", (str,), f"Molecule output build failed: {exc}")
            return
        self._invoke_slot("_send_outputs", (object, object), table, mols)
        self._invoke_slot(
            "_set_busy",
            (bool, str),
            False,
            f"Ready: {0 if table is None else len(table)} rows, {len(mols)} molecules.",
        )

    def _render_molecules_table(self) -> None:
        prop_keys = list(self.selected_prop_keys or [])
        if (not prop_keys) and self._mol_props_by_id:
            prop_keys = self._derive_prop_keys_from_records(self._mol_props_by_id)

        # aggregated bio columns (only if bio exists)
        kind_map = {k: kind for k, kind in self.BIO_FIELD_SPECS}
        selected_bio = list(self.selected_bio_fields or [])
        bio_num_fields = [k for k in selected_bio if kind_map.get(k) == "num"]
        bio_meta_fields = [k for k in selected_bio if kind_map.get(k) in ("meta", "smiles") and k not in ("SMILES", "molecule_chembl_id", "pref_name")]

        headers = ["ChEMBL ID", "Name", "SMILES"] + prop_keys
        if self._bio:
            headers += ["bio_n"] + [k for k in bio_num_fields if k not in prop_keys] + bio_meta_fields

        self.tbl_mols.clear()
        self.tbl_mols.setColumnCount(len(headers))
        self.tbl_mols.setHorizontalHeaderLabels(headers)
        self.tbl_mols.setRowCount(0)

        bio_by_id = self._aggregate_bio_by_molecule(self._bio) if self._bio else {}

        for m in self._molecules:
            mid_raw = (m.chembl_id or "").strip()
            if not mid_raw:
                continue
            mid = mid_raw.upper()
            mp = self._mol_props_by_id.get(mid)
            name = (mp.pref_name if mp is not None else "") or (m.pref_name or "")
            smi = (mp.canonical_smiles if mp is not None else "") or (m.canonical_smiles or "")
            bio = bio_by_id.get(mid, {})

            r = self.tbl_mols.rowCount()
            self.tbl_mols.insertRow(r)
            self.tbl_mols.setItem(r, 0, QTableWidgetItem(mid_raw))
            self.tbl_mols.setItem(r, 1, QTableWidgetItem(name))
            self.tbl_mols.setItem(r, 2, QTableWidgetItem(smi))

            col = 3
            for k in prop_keys:
                v = (mp.props or {}).get(k) if mp is not None else None
                self.tbl_mols.setItem(r, col, QTableWidgetItem("" if v is None else str(v)))
                col += 1

            if self._bio:
                self.tbl_mols.setItem(r, col, QTableWidgetItem(str(bio.get("bio_n", "")) if bio else ""))
                col += 1
                for k in bio_num_fields:
                    if k in prop_keys:
                        continue
                    v = bio.get(k, None) if bio else None
                    self.tbl_mols.setItem(r, col, QTableWidgetItem("" if v is None else str(v)))
                    col += 1
                for k in bio_meta_fields:
                    v = bio.get(k, "") if bio else ""
                    self.tbl_mols.setItem(r, col, QTableWidgetItem(str(v or "")))
                    col += 1

        self.tbl_mols.resizeColumnsToContents()

    # ---------------- pickers ----------------

    def _on_flags_changed(self) -> None:
        self.include_props_in_table = bool(self.btn_props_in_table.isChecked())
        self.include_props_in_molecules = bool(self.btn_props_in_mols.isChecked())
        self.btn_props_in_table.setText("Props in Table: ON" if self.include_props_in_table else "Props in Table: OFF")
        self.btn_props_in_mols.setText("Props in Molecules: ON" if self.include_props_in_molecules else "Props in Molecules: OFF")

    def _render_bio_fields_list(self) -> None:
        selected = set(self.selected_bio_fields or [])
        self.list_bio_fields.blockSignals(True)
        try:
            self.list_bio_fields.clear()
            for key, _kind in self.BIO_FIELD_SPECS:
                it = QListWidgetItem(key)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if key in selected else Qt.Unchecked)
                self.list_bio_fields.addItem(it)
        finally:
            self.list_bio_fields.blockSignals(False)

    def _on_bio_fields_changed(self) -> None:
        keys: List[str] = []
        for i in range(self.list_bio_fields.count()):
            it = self.list_bio_fields.item(i)
            if it.checkState() == Qt.Checked:
                keys.append(it.text().strip())
        self.selected_bio_fields = keys
        # refresh molecules table header (bio columns)
        self._render_molecules_table()

    def _render_props_list(self) -> None:
        keys = list(self._available_prop_keys or [])
        selected = set(self.selected_prop_keys or [])
        self.list_props.blockSignals(True)
        try:
            self.list_props.clear()
            for k in keys:
                it = QListWidgetItem(k)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if k in selected else Qt.Unchecked)
                self.list_props.addItem(it)
        finally:
            self.list_props.blockSignals(False)

    def _on_prop_keys_changed(self) -> None:
        keys: List[str] = []
        for i in range(self.list_props.count()):
            it = self.list_props.item(i)
            if it.checkState() == Qt.Checked:
                keys.append(it.text().strip())
        self.selected_prop_keys = keys

        # refetch props and rebuild outputs if molecules already loaded
        self._render_molecules_table()
        if self._molecules:
            ids = [m.chembl_id for m in self._molecules if m.chembl_id]
            self._set_busy(True, f"Updating properties for {len(ids)} molecules …")
            fut = self.executor.submit(self._fetch_molecule_props_for_table, ids, list(self.selected_prop_keys or []))
            fut.add_done_callback(self._on_mol_props_ready)

    def _on_refresh_prop_keys_clicked(self) -> None:
        ids = self._current_molecule_ids_for_sampling()
        self._refresh_property_keys_background(sample_ids=ids)

    def _current_molecule_ids_for_sampling(self) -> Optional[List[str]]:
        ids: List[str] = []
        if self._bio:
            ids.extend([getattr(r, "molecule_chembl_id", "") for r in self._bio if getattr(r, "molecule_chembl_id", "")])
        if self._molecules:
            ids.extend([m.chembl_id for m in self._molecules if m.chembl_id])
        ids = [x.strip().upper() for x in ids if x and str(x).strip().upper().startswith("CHEMBL")]
        seen = set()
        out = []
        for x in ids:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out[:160] if out else None

    def _refresh_property_keys_background(self, sample_ids: Optional[Sequence[str]]) -> None:
        self._set_busy(True, "Refreshing molecule property keys …")
        fut = self.executor.submit(self.svc_mols.fetch_available_property_keys, sample_ids)
        fut.add_done_callback(self._on_prop_keys_ready)

    def _on_prop_keys_ready(self, fut) -> None:
        self._dispatch_future_result(
            fut,
            "_update_prop_keys",
            (object,),
            "Property key refresh failed",
        )

    @pyqtSlot(object)
    def _update_prop_keys(self, keys) -> None:
        self._available_prop_keys = list(keys or [])
        self._render_props_list()

        # pick sensible defaults once
        if not self.selected_prop_keys:
            defaults = ["full_mwt", "alogp", "psa", "hba", "hbd", "rtb", "qed_weighted"]
            self.selected_prop_keys = [k for k in defaults if k in self._available_prop_keys]
            self._render_props_list()

        self._set_busy(False, f"Loaded {len(self._available_prop_keys)} molecule property keys.")

    # ---------------- prop key helpers ----------------

    def _derive_prop_keys_from_records(
        self, props_by_id: Dict[str, ChemBLMoleculePropsRecord], max_keys: int = 25
    ) -> List[str]:
        """Derive a stable list of property keys from returned molecule_properties records."""
        return derive_prop_keys_from_records(props_by_id, max_keys=max_keys)

    def _ensure_selected_prop_keys(self, props_by_id: Dict[str, ChemBLMoleculePropsRecord]) -> None:
        """If user has not selected property keys yet, auto-select from fetched props."""
        if self.selected_prop_keys:
            return
        derived = self._derive_prop_keys_from_records(props_by_id)
        if not derived:
            return
        self.selected_prop_keys = derived
        # keep UI in sync
        self._available_prop_keys = sorted(set(self._available_prop_keys or []).union(set(derived)))
        self._render_props_list()
        self._render_molecules_table()

    def _aggregate_bio_by_molecule(self, recs: List[ChemBLBioactivityRecord]) -> Dict[str, Dict[str, Any]]:
        """Aggregate bioactivities per molecule_chembl_id so we can join into Molecules outputs."""
        return aggregate_bio_by_molecule(
            recs,
            list(self.selected_bio_fields or []),
            self.BIO_FIELD_SPECS,
        )

    # ---------------- output builders ----------------

    def _build_outputs_from_bioactivities(self, recs: List[ChemBLBioactivityRecord], prop_keys: List[str]):
        if not recs:
            return None, [], ""

        prop_keys = list(prop_keys or [])

        ids = [str(getattr(r, "molecule_chembl_id", "") or "").strip().upper() for r in recs]
        ids = [i for i in ids if i]
        props_by_id, warning = _safe_fetch_props_by_id(self.svc_mols, ids, prop_keys)
        table, mols = build_bioactivity_outputs(
            recs,
            prop_keys,
            props_by_id,
            list(self.selected_bio_fields or []),
            self.BIO_FIELD_SPECS,
            include_props_in_molecules=bool(self.include_props_in_molecules),
        )
        return table, mols, warning

    def _build_outputs_from_molecules(
        self,
        mols: List[ChemBLMoleculeRecord],
        props_by_id: Dict[str, ChemBLMoleculePropsRecord],
        prop_keys: List[str],
    ):
        """Build Table + ChemMol from molecules, enriched with selected properties AND aggregated bio."""
        table, out_mols = build_molecule_outputs(
            mols,
            props_by_id,
            list(prop_keys or []),
            list(self._bio or []),
            list(self.selected_bio_fields or []),
            self.BIO_FIELD_SPECS,
            include_props_in_molecules=bool(self.include_props_in_molecules),
            selected_target_id=(self._selected_target.chembl_id if self._selected_target is not None else None),
        )
        return table, out_mols, ""

    def _on_outputs_ready(self, fut) -> None:
        try:
            table, mols, warning = fut.result()
        except Exception as exc:
            self._invoke_slot("_on_error", (str,), f"Output build failed: {exc}")
            return
        self._invoke_slot("_send_outputs", (object, object, str), table, mols, warning)

    @pyqtSlot(object, object, str)
    def _send_outputs(self, table, mols, warning: str = "") -> None:
        self._last_table = table
        self._last_molecules = list(mols or [])
        self.Outputs.data.send(table)
        self.Outputs.molecules.send(self._last_molecules)
        status = format_output_summary(table, self._last_molecules)
        if warning:
            status = f"{status} | {warning}"
        self.lbl_status.setText(status)

    # ---------------- selected-only outputs ----------------

    def _on_send_selected_bio(self) -> None:
        if not self._bio:
            self.lbl_status.setText("No bioactivities loaded.")
            self.Outputs.selected_data.send(None)
            self.Outputs.selected_molecules.send([])
            return

        rows = _selected_row_indices(self.tbl_bio)
        if not rows:
            self.lbl_status.setText("Select rows in Bioactivities table first.")
            self.Outputs.selected_data.send(None)
            self.Outputs.selected_molecules.send([])
            return

        sel = [self._bio[row] for row in rows if 0 <= row < len(self._bio)]
        self._set_busy(True, f"Building selected outputs (bio: {len(sel)}) …")
        fut = self.executor.submit(self._build_outputs_from_bioactivities, sel, list(self.selected_prop_keys or []))
        fut.add_done_callback(self._on_selected_outputs_ready)

    def _on_send_selected_molecules(self) -> None:
        if not self._molecules:
            self.lbl_status.setText("No molecules loaded.")
            self.Outputs.selected_data.send(None)
            self.Outputs.selected_molecules.send([])
            return

        rows = _selected_row_indices(self.tbl_mols)
        if not rows:
            self.lbl_status.setText("Select rows in Molecules table first.")
            self.Outputs.selected_data.send(None)
            self.Outputs.selected_molecules.send([])
            return

        ids = []
        for row in rows:
            item = self.tbl_mols.item(row, 0)
            if item is None:
                continue
            ids.append(item.text().strip().upper())

        sel = [m for m in self._molecules if (m.chembl_id or "").strip().upper() in set(ids)]
        self._set_busy(True, f"Building selected outputs (mols: {len(sel)}) …")
        fut = self.executor.submit(self._build_outputs_from_molecules, sel, self._mol_props_by_id, list(self.selected_prop_keys or []))
        fut.add_done_callback(self._on_selected_outputs_ready)

    def _on_selected_outputs_ready(self, fut) -> None:
        try:
            table, mols, warning = fut.result()
        except Exception as exc:
            self._invoke_slot("_on_error", (str,), f"Selected output build failed: {exc}")
            return
        self._invoke_slot("_send_selected_outputs", (object, object, str), table, mols, warning)
        self._invoke_slot(
            "_set_busy",
            (bool, str),
            False,
            f"Selected: {0 if table is None else len(table)} rows, {len(mols)} molecules.",
        )

    @pyqtSlot(object, object, str)
    def _send_selected_outputs(self, table, mols, warning: str = "") -> None:
        self.Outputs.selected_data.send(table)
        self.Outputs.selected_molecules.send(list(mols or []))
        status = format_output_summary(table, list(mols or []), selected=True)
        if warning:
            status = f"{status} | {warning}"
        self.lbl_status.setText(status)

    # ---------------- export helpers ----------------

    def _on_export_csv(self) -> None:
        if self._last_table is None:
            self.lbl_status.setText(format_no_input_status("data to export"))
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "chembl_export.csv", "CSV (*.csv)")
        if not path:
            return

        try:
            self._last_table.save(path)
            self.lbl_status.setText(format_done_status(f"saved {os.path.basename(path)}"))
        except Exception as e:
            self.lbl_status.setText(format_error_status(f"Export failed: {e}"))

    def _on_export_sdf(self) -> None:
        if not self._last_molecules:
            self.lbl_status.setText(format_no_input_status("molecules to export"))
            return

        # selected if any, else all in Molecules tab
        sel_ids: List[str] = []
        rows = _selected_row_indices(self.tbl_mols)
        if rows:
            for row in rows:
                item = self.tbl_mols.item(row, 0)
                if item is not None:
                    sel_ids.append(item.text().strip().upper())

        out = self._last_molecules
        if sel_ids:
            sel = set(sel_ids)
            out = [m for m in self._last_molecules if (m.get_prop("molecule_chembl_id") or "").strip().upper() in sel]

        path, _ = QFileDialog.getSaveFileName(self, "Export SDF", "chembl_molecules.sdf", "SDF (*.sdf)")
        if not path:
            return

        try:
            from rdkit import Chem

            w = Chem.SDWriter(path)
            for cm in out:
                rdm = cm.to_rdkit() if hasattr(cm, "to_rdkit") else None
                if rdm is None:
                    continue
                for k, v in (cm.props or {}).items():
                    if v is None:
                        continue
                    try:
                        rdm.SetProp(str(k), str(v))
                    except Exception:
                        pass
                w.write(rdm)
            w.close()
            self.lbl_status.setText(format_done_status(f"saved {os.path.basename(path)}"))
        except Exception as e:
            self.lbl_status.setText(format_error_status(f"SDF export failed: {e}"))


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWChemBLBrowser).run()
