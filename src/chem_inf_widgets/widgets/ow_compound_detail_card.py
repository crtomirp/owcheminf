from __future__ import annotations
from typing import Optional

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QPixmap
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStyle,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from rdkit import Chem

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles
from chem_inf_widgets.chemcore.services.compound_detail_service import (
    CompoundDetail,
    CompoundReference,
    MotifHit,
    build_detail_outputs,
    compute_motif_hits,
    default_name_var_name,
    default_smiles_var_name,
    png_data_uri,
    render_fragment_detail_html,
    render_motif_detail_html,
    render_summary_html,
    references_from_molecules,
    references_from_table,
    row_text,
    scaffold_query_table,
    motif_hits_table,
    selected_motif_query_table,
    search_profile_table,
    build_detail,
)
from chem_inf_widgets.widgets.ui_helpers import (
    format_done_status,
    format_no_input_status,
    format_result_count_status,
)


def _pixmap_from_data_uri(uri: str) -> QPixmap:
    pix = QPixmap()
    prefix = "data:image/png;base64,"
    if uri.startswith(prefix):
        import base64

        pix.loadFromData(base64.b64decode(uri[len(prefix) :]))
    return pix


class OWCompoundDetailCard(OWWidget):
    name = "Compound Detail Card"
    description = "FAIRMol-style compound detail viewer with PharmaFP fragments and similar compounds."
    icon = "icons/editors_viewers/owcompounddetailcardwidget.svg"
    priority = 139

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)
        reference_data = Input("Reference Data", Table)

    class Outputs:
        selected_compound = Output("Selected Compound", Table)
        similar_compounds = Output("Similar Compounds", Table)
        matched_fragments = Output("Matched Fragments", Table)
        detected_motifs = Output("Detected Motifs", Table)
        motif_queries = Output("Motif Queries", Table)
        query_molecule = Output("Query Molecule", ChemMol, auto_summary=False)
        fragment_queries = Output("Fragment Queries", Table)
        scaffold_query = Output("Scaffold Query", Table)
        search_profile = Output("Search Profile", Table)

    smiles_var_name: str = Setting("")
    name_var_name: str = Setting("")
    top_k: int = Setting(5)
    selected_index: int = Setting(0)
    motif_logic: str = Setting("or")

    want_main_area = True
    resizing_enabled = True

    def __init__(self) -> None:
        super().__init__()
        self._apply_style()
        self.data: Optional[Table] = None
        self.molecules: list[ChemMol] = []
        self.reference_data: Optional[Table] = None
        self._current_detail: Optional[CompoundDetail] = None
        self._current_mol: Optional[Chem.Mol] = None
        self._current_refs: list[CompoundReference] = []
        self._current_motifs: tuple[MotifHit, ...] = ()

        form = QFormLayout()
        self.smiles_combo = QComboBox()
        self.smiles_combo.currentTextChanged.connect(self._on_controls_changed)
        form.addRow("SMILES:", self.smiles_combo)

        self.name_combo = QComboBox()
        self.name_combo.currentTextChanged.connect(self._on_controls_changed)
        form.addRow("Name:", self.name_combo)

        box = gui.widgetBox(self.controlArea, "Compound Detail", spacing=8)
        box.layout().addLayout(form)
        self.motif_logic_combo = QComboBox()
        self.motif_logic_combo.addItems(["or", "and"])
        self.motif_logic_combo.setCurrentText(self.motif_logic)
        self.motif_logic_combo.currentTextChanged.connect(self._on_motif_logic_changed)
        box.layout().addWidget(QLabel("Motif query logic"))
        box.layout().addWidget(self.motif_logic_combo)
        gui.spin(box, self, "top_k", 1, 20, 1, label="Similar hits:", callback=self._refresh_detail)
        gui.button(self.controlArea, self, "Refresh", callback=self._refresh_detail)

        self.status_label = QLabel("Waiting for input…")
        self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)

        root = QSplitter(Qt.Horizontal)
        self.mainArea.layout().addWidget(root)

        self.compound_list = QListWidget()
        self.compound_list.currentRowChanged.connect(self._on_selected_row_changed)
        self.compound_list.setMinimumWidth(180)
        self.compound_list.setMaximumWidth(260)
        root.addWidget(self.compound_list)

        detail_root = QWidget()
        detail_layout = QVBoxLayout(detail_root)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(8)

        self.structure_panel = QFrame()
        self.structure_panel.setObjectName("detailCardPanel")
        self.structure_panel.setFrameShape(QFrame.StyledPanel)
        structure_layout = QVBoxLayout(self.structure_panel)
        structure_layout.setContentsMargins(10, 10, 10, 10)
        structure_layout.setSpacing(8)
        self.structure_title = QLabel("Structure")
        self.structure_title.setObjectName("detailCardTitle")
        structure_layout.addWidget(self.structure_title)
        self.structure_label = QLabel("No structure")
        self.structure_label.setAlignment(Qt.AlignCenter)
        self.structure_label.setMinimumSize(420, 320)
        self.structure_label.setStyleSheet(
            "background:white; border:1px solid #d7dee7; border-radius:10px; padding:12px;"
        )
        structure_layout.addWidget(self.structure_label, 1)
        detail_layout.addWidget(self.structure_panel, 3)

        self.summary_panel = QFrame()
        self.summary_panel.setObjectName("detailCardPanel")
        self.summary_panel.setFrameShape(QFrame.StyledPanel)
        info_layout = QVBoxLayout(self.summary_panel)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(8)
        self.info_title = QLabel("Compound Summary")
        self.info_title.setObjectName("detailCardTitle")
        info_layout.addWidget(self.info_title)
        self.summary_browser = QTextBrowser()
        self.summary_browser.setOpenExternalLinks(False)
        self.summary_browser.setObjectName("detailSummaryBrowser")
        self.summary_browser.setMinimumHeight(170)
        self.summary_browser.setMaximumHeight(280)
        info_layout.addWidget(self.summary_browser, 1)
        detail_layout.addWidget(self.summary_panel, 1)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(2, 0, 2, 0)
        action_row.setSpacing(8)
        self.fragment_button = QPushButton("Fragments")
        self.fragment_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.fragment_button.setToolTip("Open matched PharmaFP fragments")
        self.fragment_button.clicked.connect(self._open_fragment_dialog)
        action_row.addWidget(self.fragment_button)
        self.motif_button = QPushButton("Motifs")
        self.motif_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogContentsView))
        self.motif_button.setToolTip("Open selected motif details")
        self.motif_button.clicked.connect(self._open_motif_dialog)
        action_row.addWidget(self.motif_button)
        self.similar_button = QPushButton("Similar")
        self.similar_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.similar_button.setToolTip("Open similar compounds")
        self.similar_button.clicked.connect(self._open_similar_dialog)
        action_row.addWidget(self.similar_button)
        self.copy_scaffold_button = QPushButton("Copy Scaffold")
        self.copy_scaffold_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogListView))
        self.copy_scaffold_button.setToolTip("Copy Murcko and generic scaffold for the current compound")
        self.copy_scaffold_button.clicked.connect(self._copy_current_scaffold)
        action_row.addWidget(self.copy_scaffold_button)
        action_row.addStretch(1)
        detail_layout.addLayout(action_row)

        motif_panel = QFrame()
        motif_panel.setObjectName("detailCardPanel")
        motif_panel.setFrameShape(QFrame.StyledPanel)
        motif_layout = QVBoxLayout(motif_panel)
        motif_layout.setContentsMargins(10, 10, 10, 10)
        motif_layout.setSpacing(8)
        motif_title = QLabel("Motifs")
        motif_title.setObjectName("detailCardTitle")
        motif_layout.addWidget(motif_title)
        self.motif_list = QListWidget()
        self.motif_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.motif_list.setAlternatingRowColors(True)
        self.motif_list.setObjectName("detailMotifList")
        self.motif_list.itemSelectionChanged.connect(self._on_motif_selection_changed)
        self.motif_list.itemDoubleClicked.connect(lambda _item: self._open_motif_dialog())
        motif_layout.addWidget(self.motif_list, 1)
        motif_actions = QHBoxLayout()
        self.copy_smarts_button = QPushButton("Copy SMARTS")
        self.copy_smarts_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogNewFolder))
        self.copy_smarts_button.setToolTip("Copy selected motif SMARTS patterns to clipboard")
        self.copy_smarts_button.clicked.connect(self._copy_selected_motif_smarts)
        motif_actions.addWidget(self.copy_smarts_button)
        self.copy_names_button = QPushButton("Copy Names")
        self.copy_names_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogInfoView))
        self.copy_names_button.setToolTip("Copy selected motif names to clipboard")
        self.copy_names_button.clicked.connect(self._copy_selected_motif_names)
        motif_actions.addWidget(self.copy_names_button)
        self.search_now_button = QPushButton("Search Now")
        self.search_now_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.search_now_button.setToolTip("Emit current motif selection to downstream search widgets")
        self.search_now_button.clicked.connect(self._search_selected_motifs_now)
        motif_actions.addWidget(self.search_now_button)
        motif_actions.addStretch(1)
        motif_layout.addLayout(motif_actions)
        detail_layout.addWidget(motif_panel, 2)

        root.addWidget(detail_root)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)

        self._build_aux_dialogs()

    def _build_aux_dialogs(self) -> None:
        self.fragment_dialog = QDialog(self)
        self.fragment_dialog.setWindowTitle("Matched Fragments")
        self.fragment_dialog.resize(680, 520)
        fragment_layout = QVBoxLayout(self.fragment_dialog)
        self.fragment_list = QListWidget()
        self.fragment_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.fragment_list.setAlternatingRowColors(True)
        self.fragment_list.itemSelectionChanged.connect(self._on_fragment_selection_changed)
        fragment_layout.addWidget(self.fragment_list, 1)
        self.fragment_detail = QTextBrowser()
        fragment_layout.addWidget(self.fragment_detail, 1)

        self.motif_dialog = QDialog(self)
        self.motif_dialog.setWindowTitle("Selected Motifs")
        self.motif_dialog.resize(560, 420)
        motif_dialog_layout = QVBoxLayout(self.motif_dialog)
        self.motif_detail = QTextBrowser()
        motif_dialog_layout.addWidget(self.motif_detail, 1)

        self.similar_dialog = QDialog(self)
        self.similar_dialog.setWindowTitle("Similar Compounds")
        self.similar_dialog.resize(520, 420)
        similar_layout = QVBoxLayout(self.similar_dialog)
        self.similar_list = QListWidget()
        self.similar_list.setAlternatingRowColors(True)
        similar_layout.addWidget(self.similar_list, 1)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QTabWidget::pane { border: none; }
            QTextBrowser { padding: 4px; }
            QFrame#detailCardPanel {
                background: #f8fafc;
                border: 1px solid #dbe4ee;
                border-radius: 12px;
            }
            QLabel#detailCardTitle {
                font-weight: 600;
                color: #334155;
                padding: 0 2px;
            }
            QTextBrowser#detailSummaryBrowser {
                background: white;
                border: 1px solid #d7dee7;
                border-radius: 10px;
                padding: 8px;
            }
            QListWidget#detailMotifList {
                background: white;
                border: 1px solid #d7dee7;
                border-radius: 10px;
            }
            """
        )

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_controls_changed(self) -> None:
        self.smiles_var_name = self.smiles_combo.currentText().strip()
        self.name_var_name = self.name_combo.currentText().strip()
        self._populate_compound_list()
        self._refresh_detail()

    def _on_motif_logic_changed(self, text: str) -> None:
        self.motif_logic = (text or "or").strip().lower() or "or"
        self._emit_search_outputs()

    def _on_selected_row_changed(self, row: int) -> None:
        self.selected_index = max(0, row)
        self._refresh_detail()

    def _selected_name(self, idx: int) -> str:
        if self.data is not None:
            row = self.data[idx]
            return row_text(row, self.name_var_name) or f"Compound {idx + 1}"
        if idx < len(self.molecules):
            return (self.molecules[idx].name or "").strip() or f"Compound {idx + 1}"
        return f"Compound {idx + 1}"

    def _table_string_vars(self) -> list[str]:
        if self.data is None:
            return []
        names = []
        for var in list(self.data.domain.metas) + list(self.data.domain.attributes) + list(self.data.domain.class_vars):
            if getattr(var, "is_string", False) or var.__class__.__name__ == "StringVariable":
                names.append(var.name)
        return names

    def _refresh_controls(self) -> None:
        vars_ = self._table_string_vars()
        self.smiles_combo.blockSignals(True)
        self.name_combo.blockSignals(True)
        self.smiles_combo.clear()
        self.name_combo.clear()
        self.smiles_combo.addItems(vars_)
        self.name_combo.addItem("")
        self.name_combo.addItems(vars_)
        if self.data is not None:
            smiles_default = self.smiles_var_name or default_smiles_var_name(self.data)
            name_default = self.name_var_name or default_name_var_name(self.data)
            if smiles_default:
                self.smiles_combo.setCurrentText(smiles_default)
                self.smiles_var_name = smiles_default
            if name_default or not self.name_combo.currentText():
                self.name_combo.setCurrentText(name_default)
                self.name_var_name = name_default
        self.smiles_combo.blockSignals(False)
        self.name_combo.blockSignals(False)

    def _populate_compound_list(self) -> None:
        self.compound_list.blockSignals(True)
        self.compound_list.clear()
        if self.data is not None:
            for idx, row in enumerate(self.data):
                item = QListWidgetItem(self._selected_name(idx))
                item.setData(Qt.UserRole, idx)
                self.compound_list.addItem(item)
        else:
            for idx, cm in enumerate(self.molecules):
                item = QListWidgetItem((cm.name or "").strip() or f"Compound {idx + 1}")
                item.setData(Qt.UserRole, idx)
                self.compound_list.addItem(item)
        if self.compound_list.count():
            self.compound_list.setCurrentRow(min(self.selected_index, self.compound_list.count() - 1))
        self.compound_list.blockSignals(False)

    def _reference_pool(self) -> list[CompoundReference]:
        if self.reference_data is not None:
            return references_from_table(self.reference_data)
        if self.data is not None:
            return references_from_table(self.data, smiles_var_name=self.smiles_var_name, name_var_name=self.name_var_name)
        return references_from_molecules(self.molecules)

    def _selected_mol_and_name(self) -> tuple[Optional[Chem.Mol], str, str]:
        if self.data is not None:
            if not len(self.data):
                return None, "", ""
            idx = min(max(0, self.selected_index), len(self.data) - 1)
            row = self.data[idx]
            smiles = row_text(row, self.smiles_var_name)
            mol = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol if smiles else None
            return mol, self._selected_name(idx), smiles
        if self.molecules:
            idx = min(max(0, self.selected_index), len(self.molecules) - 1)
            cm = self.molecules[idx]
            return cm.to_rdkit(), (cm.name or "").strip() or f"Compound {idx + 1}", cm.canonical_smiles()
        return None, "", ""

    def _selected_fragment_hits(self, *, fallback_current: bool = True) -> tuple:
        if self._current_detail is None:
            return ()
        rows = sorted(
            {
                item.data(Qt.UserRole)
                for item in self.fragment_list.selectedItems()
                if item.data(Qt.UserRole) is not None
            }
        )
        if rows:
            return tuple(
                self._current_detail.fragment_hits[row]
                for row in rows
                if 0 <= row < len(self._current_detail.fragment_hits)
            )
        if fallback_current:
            row = self.fragment_list.currentRow()
            if 0 <= row < len(self._current_detail.fragment_hits):
                return (self._current_detail.fragment_hits[row],)
        return ()

    def _render_fragment_detail(self) -> None:
        self.fragment_detail.setHtml(render_fragment_detail_html(self._selected_fragment_hits()))

    def _render_structure(self, highlight_atoms: Optional[list[int]] = None) -> None:
        if self._current_mol is None:
            self.structure_label.setText("No structure")
            self.structure_label.setPixmap(QPixmap())
            return
        pixmap = _pixmap_from_data_uri(png_data_uri(self._current_mol, highlight_atoms=highlight_atoms or []))
        self.structure_label.setText("")
        self.structure_label.setPixmap(pixmap.scaled(360, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _selected_motif_hits(self, *, fallback_all: bool = True) -> tuple[MotifHit, ...]:
        rows = sorted({item.data(Qt.UserRole) for item in self.motif_list.selectedItems() if item.data(Qt.UserRole) is not None})
        if rows:
            return tuple(self._current_motifs[row] for row in rows if 0 <= row < len(self._current_motifs))
        return self._current_motifs if fallback_all else ()

    def _active_highlight_atoms(self) -> list[int]:
        atoms = set()
        selected_motifs = self._selected_motif_hits(fallback_all=False)
        if selected_motifs:
            atoms.update(atom for hit in selected_motifs for atom in hit.matched_atoms)
        selected_fragments = self._selected_fragment_hits(fallback_current=False)
        if selected_fragments:
            atoms.update(atom for hit in selected_fragments for atom in hit.matched_atoms)
        elif self._current_detail is not None:
            row = self.fragment_list.currentRow()
            if 0 <= row < len(self._current_detail.fragment_hits):
                atoms.update(self._current_detail.fragment_hits[row].matched_atoms)
        return sorted(atoms)

    def _render_motif_detail(self) -> None:
        self.motif_detail.setHtml(
            render_motif_detail_html(
                self._selected_motif_hits(),
                motif_logic=self.motif_logic,
            )
        )

    def _emit_search_outputs(self) -> None:
        if self._current_detail is None:
            self.Outputs.detected_motifs.send(None)
            self.Outputs.motif_queries.send(None)
            self.Outputs.search_profile.send(None)
            return
        selected_motifs = self._selected_motif_hits()
        self.Outputs.detected_motifs.send(motif_hits_table(self._current_motifs))
        self.Outputs.motif_queries.send(selected_motif_query_table(selected_motifs))
        self.Outputs.search_profile.send(search_profile_table(self._current_detail, motif_queries=selected_motifs, motif_logic=self.motif_logic))

    def _refresh_detail(self) -> None:
        self.clear_messages()
        mol, name, smiles = self._selected_mol_and_name()
        if mol is None:
            self._current_detail = None
            self._current_mol = None
            self._set_status(format_no_input_status("valid molecule"))
            self.structure_label.setText("No structure")
            self.summary_browser.setHtml("<div style='color:#5f6b7a;'>Send a Table or Molecules input to inspect a compound.</div>")
            self.motif_list.clear()
            self.fragment_list.clear()
            self.similar_list.clear()
            self.fragment_detail.clear()
            self.motif_detail.clear()
            self._update_action_labels(None)
            self.Outputs.selected_compound.send(None)
            self.Outputs.similar_compounds.send(None)
            self.Outputs.matched_fragments.send(None)
            self.Outputs.detected_motifs.send(None)
            self.Outputs.motif_queries.send(None)
            self.Outputs.query_molecule.send(None)
            self.Outputs.fragment_queries.send(None)
            self.Outputs.scaffold_query.send(None)
            self.Outputs.search_profile.send(None)
            return

        self._current_mol = mol
        self._current_refs = self._reference_pool()
        detail = build_detail(
            mol,
            name=name,
            reference=self._current_refs,
            top_k=int(self.top_k),
            exclude_smiles=smiles,
        )
        self._current_detail = detail
        self._current_motifs = compute_motif_hits(mol)
        self._render_structure([])
        self.summary_browser.setHtml(render_summary_html(detail))
        self._update_action_labels(detail)

        self.fragment_list.blockSignals(True)
        self.fragment_list.clear()
        for idx, hit in enumerate(detail.fragment_hits):
            label = f"{hit.category}: {hit.name}"
            if hit.frequency:
                label += f" [{hit.frequency}]"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, idx)
            self.fragment_list.addItem(item)
        self.fragment_list.blockSignals(False)
        if self.fragment_list.count():
            self.fragment_list.setCurrentRow(0)
            first = self.fragment_list.item(0)
            if first is not None:
                first.setSelected(True)
            self._render_fragment_detail()
        else:
            self.fragment_detail.setHtml("<div style='color:#5f6b7a;'>No PharmaFP fragments matched this compound.</div>")

        self.motif_list.blockSignals(True)
        self.motif_list.clear()
        for idx, hit in enumerate(self._current_motifs):
            prefix = "Het" if hit.category == "heterocycle" else "FG"
            extra = f" · {hit.family}" if hit.family else ""
            item = QListWidgetItem(f"{prefix}: {hit.name}{extra}")
            item.setData(Qt.UserRole, idx)
            self.motif_list.addItem(item)
        self.motif_list.blockSignals(False)
        self._render_motif_detail()

        self.similar_list.clear()
        for hit in detail.similar_hits:
            item = QListWidgetItem(
                f"{hit.name or f'Reference {hit.source_index + 1}'}  |  {hit.similarity:.2f}  |  shared {hit.shared_fragments}"
            )
            self.similar_list.addItem(item)

        self._set_status(
            format_done_status(
                f"compound={detail.name or 'selected'}",
                f"PharmaFP matches={len(detail.fragment_hits)}",
                f"similar hits={len(detail.similar_hits)}",
            )
        )
        outputs = build_detail_outputs(
            detail,
            motif_hits=self._current_motifs,
            selected_motif_hits=self._selected_motif_hits(),
            motif_logic=self.motif_logic,
        )
        self.Outputs.selected_compound.send(outputs.selected_compound)
        self.Outputs.similar_compounds.send(outputs.similar_compounds)
        self.Outputs.matched_fragments.send(outputs.matched_fragments)
        self.Outputs.detected_motifs.send(outputs.detected_motifs)
        self.Outputs.query_molecule.send(outputs.query_molecule)
        self.Outputs.fragment_queries.send(outputs.fragment_queries)
        self.Outputs.motif_queries.send(outputs.motif_queries)
        self.Outputs.scaffold_query.send(outputs.scaffold_query)
        self.Outputs.search_profile.send(outputs.search_profile)

    def _on_fragment_selection_changed(self) -> None:
        self._render_structure(self._active_highlight_atoms())
        self._render_fragment_detail()

    def _on_motif_selection_changed(self) -> None:
        self._render_structure(self._active_highlight_atoms())
        self._emit_search_outputs()

    def _copy_selected_motif_smarts(self) -> None:
        selected = self._selected_motif_hits(fallback_all=False)
        if not selected:
            self._set_status("Select one or more motifs to copy SMARTS.")
            return
        text = "\n".join(hit.smarts for hit in selected if hit.smarts)
        QApplication.clipboard().setText(text)
        self._set_status(format_result_count_status(len(selected), item_label="motif SMARTS pattern(s) copied", prefix="Copied"))

    def _copy_selected_motif_names(self) -> None:
        selected = self._selected_motif_hits(fallback_all=False)
        if not selected:
            self._set_status("Select one or more motifs to copy names.")
            return
        text = "\n".join(hit.name for hit in selected if hit.name)
        QApplication.clipboard().setText(text)
        self._set_status(format_result_count_status(len(selected), item_label="motif name(s) copied", prefix="Copied"))

    def _copy_current_scaffold(self) -> None:
        if self._current_detail is None:
            self._set_status(format_no_input_status("compound loaded to copy scaffold from"))
            return
        table = scaffold_query_table(self._current_detail)
        row = table[0] if len(table) else None
        murcko = ""
        generic = ""
        if row is not None:
            try:
                murcko = str(row["Murcko Scaffold"]).strip()
            except Exception:
                murcko = ""
            try:
                generic = str(row["Generic Scaffold"]).strip()
            except Exception:
                generic = ""
        text = f"Murcko Scaffold\t{murcko or '—'}\nGeneric Scaffold\t{generic or '—'}"
        QApplication.clipboard().setText(text)
        self._set_status(format_done_status("copied current scaffold to clipboard", prefix="Done"))

    def _search_selected_motifs_now(self) -> None:
        selected = self._selected_motif_hits(fallback_all=False)
        if not selected:
            self._set_status("Select one or more motifs to send to search.")
            return
        self._emit_search_outputs()
        self._set_status(
            format_done_status(
                f"sent {len(selected)} selected motif querie(s)",
                "downstream search widgets updated",
            )
        )

    def _update_action_labels(self, detail: Optional[CompoundDetail]) -> None:
        fragment_count = len(detail.fragment_hits) if detail is not None else 0
        motif_count = len(self._current_motifs) if detail is not None else 0
        similar_count = len(detail.similar_hits) if detail is not None else 0
        self.fragment_button.setText(f"Fragments ({fragment_count})")
        self.motif_button.setText(f"Motifs ({motif_count})")
        self.similar_button.setText(f"Similar ({similar_count})")

    def _open_fragment_dialog(self) -> None:
        title = self._current_detail.name if self._current_detail is not None else "Matched Fragments"
        self.fragment_dialog.setWindowTitle(f"Fragments — {title}")
        self.fragment_dialog.show()
        self.fragment_dialog.raise_()
        self.fragment_dialog.activateWindow()

    def _open_motif_dialog(self) -> None:
        self._render_motif_detail()
        title = self._current_detail.name if self._current_detail is not None else "Selected Motifs"
        self.motif_dialog.setWindowTitle(f"Motifs — {title}")
        self.motif_dialog.show()
        self.motif_dialog.raise_()
        self.motif_dialog.activateWindow()

    def _open_similar_dialog(self) -> None:
        title = self._current_detail.name if self._current_detail is not None else "Similar Compounds"
        self.similar_dialog.setWindowTitle(f"Similar Compounds — {title}")
        self.similar_dialog.show()
        self.similar_dialog.raise_()
        self.similar_dialog.activateWindow()

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self.data = data
        if data is not None:
            self.molecules = []
        self._refresh_controls()
        self._populate_compound_list()
        self._refresh_detail()

    @Inputs.molecules
    def set_molecules(self, molecules: Optional[list[ChemMol]]) -> None:
        self.molecules = list(molecules or [])
        if molecules:
            self.data = None
        self._refresh_controls()
        self._populate_compound_list()
        self._refresh_detail()

    @Inputs.reference_data
    def set_reference_data(self, data: Optional[Table]) -> None:
        self.reference_data = data
        self._refresh_detail()
