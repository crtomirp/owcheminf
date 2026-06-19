from __future__ import annotations

from typing import Optional

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget
from Orange.data import StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.compound_detail_service import (
    default_name_var_name,
    default_smiles_var_name,
    fragments_from_query_table,
    pharmafp_search_hits_table,
    query_from_search_profile,
    references_from_table,
    run_pharmafp_search,
)
from chem_inf_widgets.widgets.ui_helpers import (
    format_done_status,
    format_loaded_status,
    format_no_input_status,
    format_required_inputs_status,
    format_waiting_status,
)


def _string_vars(data: Table) -> list[StringVariable]:
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    preferred = [
        variable
        for variable in variables
        if isinstance(variable, StringVariable) and variable.name.strip().lower() in {"smiles", "smiles_std", "canonical_smiles", "smile"}
    ]
    if preferred:
        return preferred + [variable for variable in variables if isinstance(variable, StringVariable) and variable not in preferred]
    return [variable for variable in variables if isinstance(variable, StringVariable)]


class OWPharmaFPSearch(OWWidget):
    name = "PharmaFP Search"
    description = "Database search driven by FAIRMol-style PharmaFP fragments, scaffold context, and query similarity."
    icon = "icons/standardization_filtering/owpharmafpsearchwidget.svg"
    priority = 141

    class Inputs:
        query_molecule = Input("Query Molecule", ChemMol, auto_summary=False)
        fragment_queries = Input("Fragment Queries", Table)
        motif_queries = Input("Motif Queries", Table)
        scaffold_query = Input("Scaffold Query", Table)
        search_profile = Input("Search Profile", Table)
        reference_data = Input("Reference Data", Table)

    class Outputs:
        ranked_hits = Output("Ranked Hits", Table)
        hit_compounds = Output("Hit Compounds", Table)

    reference_smiles_var_name: str = Setting("")
    reference_name_var_name: str = Setting("")
    search_mode_idx: int = Setting(3)
    motif_logic: str = Setting("or")
    top_k: int = Setting(25)
    min_similarity: float = Setting(0.15)
    auto_commit: bool = Setting(True)

    _SEARCH_MODES = [
        ("Fragment", "fragment"),
        ("Similarity", "similarity"),
        ("Scaffold", "scaffold"),
        ("Hybrid", "hybrid"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.query_molecule: Optional[ChemMol] = None
        self.fragment_queries: Optional[Table] = None
        self.motif_queries: Optional[Table] = None
        self.scaffold_query: Optional[Table] = None
        self.search_profile: Optional[Table] = None
        self.reference_data: Optional[Table] = None

        self.mainArea.hide()

        root = QWidget(self.controlArea)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        self.controlArea.layout().addWidget(root)

        self.status_label = QLabel(format_waiting_status("query inputs and reference data"))
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignLeft)

        self.reference_smiles_combo = QComboBox()
        self.reference_smiles_combo.currentTextChanged.connect(lambda text: setattr(self, "reference_smiles_var_name", text))
        form.addRow("Reference SMILES:", self.reference_smiles_combo)

        self.reference_name_combo = QComboBox()
        self.reference_name_combo.currentTextChanged.connect(lambda text: setattr(self, "reference_name_var_name", text))
        form.addRow("Reference name:", self.reference_name_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([label for label, _mode in self._SEARCH_MODES])
        self.mode_combo.setCurrentIndex(int(self.search_mode_idx))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Search mode:", self.mode_combo)

        self.logic_combo = QComboBox()
        self.logic_combo.addItems(["or", "and"])
        self.logic_combo.setCurrentText(self.motif_logic)
        self.logic_combo.currentTextChanged.connect(self._on_logic_changed)
        form.addRow("Motif logic:", self.logic_combo)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 500)
        self.top_k_spin.setValue(int(self.top_k))
        self.top_k_spin.valueChanged.connect(self._on_top_k_changed)
        form.addRow("Top hits:", self.top_k_spin)

        self.min_sim_spin = QDoubleSpinBox()
        self.min_sim_spin.setRange(0.0, 1.0)
        self.min_sim_spin.setSingleStep(0.05)
        self.min_sim_spin.setValue(float(self.min_similarity))
        self.min_sim_spin.valueChanged.connect(self._on_min_sim_changed)
        form.addRow("Min PharmaFP sim:", self.min_sim_spin)

        layout.addWidget(form_widget)

        self.query_summary = QLabel("No query loaded.")
        self.query_summary.setWordWrap(True)
        layout.addWidget(self.query_summary)

        self.auto_commit_check = QCheckBox("Auto-run")
        self.auto_commit_check.setChecked(bool(self.auto_commit))
        self.auto_commit_check.toggled.connect(self._on_auto_commit_toggled)
        layout.addWidget(self.auto_commit_check)

        self.run_button = QPushButton("Run PharmaFP search")
        self.run_button.clicked.connect(self.commit)
        layout.addWidget(self.run_button)
        layout.addStretch(1)

    def _update_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _update_query_summary(self) -> None:
        smiles = ""
        scaffold = ""
        generic = ""
        threshold = float(self.min_similarity)
        if self.search_profile is not None and len(self.search_profile):
            smiles, scaffold, generic, threshold, motif_logic = query_from_search_profile(self.search_profile)
            if motif_logic:
                self.motif_logic = motif_logic
                self.logic_combo.setCurrentText(self.motif_logic)
        elif self.query_molecule is not None:
            try:
                smiles = self.query_molecule.canonical_smiles()
            except Exception:
                smiles = ""

        fragment_count = len(fragments_from_query_table(self.fragment_queries))
        motif_count = len(fragments_from_query_table(self.motif_queries))
        scaffold_text = scaffold or generic or "—"
        smiles_text = smiles or "—"
        self.query_summary.setText(
            f"Query SMILES: {smiles_text}\nScaffold: {scaffold_text}\nFragment queries: {fragment_count}\nMotif queries: {motif_count} ({self.motif_logic.upper()})\nSuggested threshold: {threshold:.2f}"
        )

    def _maybe_commit(self) -> None:
        if self.auto_commit and self.reference_data is not None:
            self.commit()

    def _on_mode_changed(self, index: int) -> None:
        self.search_mode_idx = int(index)
        self._maybe_commit()

    def _on_logic_changed(self, text: str) -> None:
        self.motif_logic = (text or "or").strip().lower()
        self._update_query_summary()
        self._maybe_commit()

    def _on_top_k_changed(self, value: int) -> None:
        self.top_k = int(value)
        self._maybe_commit()

    def _on_min_sim_changed(self, value: float) -> None:
        self.min_similarity = float(value)
        self._update_query_summary()
        self._maybe_commit()

    def _on_auto_commit_toggled(self, checked: bool) -> None:
        self.auto_commit = bool(checked)
        if self.auto_commit:
            self._maybe_commit()

    def _populate_reference_combos(self) -> None:
        self.reference_smiles_combo.blockSignals(True)
        self.reference_name_combo.blockSignals(True)
        try:
            self.reference_smiles_combo.clear()
            self.reference_name_combo.clear()
            if self.reference_data is None:
                return
            vars_ = _string_vars(self.reference_data)
            names = [var.name for var in vars_]
            self.reference_smiles_combo.addItems(names)
            self.reference_name_combo.addItem("")
            self.reference_name_combo.addItems(names)

            smiles_name = self.reference_smiles_var_name or default_smiles_var_name(self.reference_data)
            name_name = self.reference_name_var_name or default_name_var_name(self.reference_data)
            if smiles_name in names:
                self.reference_smiles_combo.setCurrentText(smiles_name)
                self.reference_smiles_var_name = smiles_name
            elif names:
                self.reference_smiles_var_name = names[0]
                self.reference_smiles_combo.setCurrentText(names[0])
            if name_name:
                self.reference_name_combo.setCurrentText(name_name)
                self.reference_name_var_name = name_name
        finally:
            self.reference_smiles_combo.blockSignals(False)
            self.reference_name_combo.blockSignals(False)

    @Inputs.query_molecule
    def set_query_molecule(self, molecule: Optional[ChemMol]) -> None:
        self.query_molecule = molecule
        self._update_query_summary()
        self._maybe_commit()

    @Inputs.fragment_queries
    def set_fragment_queries(self, data: Optional[Table]) -> None:
        self.fragment_queries = data
        self._update_query_summary()
        self._maybe_commit()

    @Inputs.motif_queries
    def set_motif_queries(self, data: Optional[Table]) -> None:
        self.motif_queries = data
        self._update_query_summary()
        self._maybe_commit()

    @Inputs.scaffold_query
    def set_scaffold_query(self, data: Optional[Table]) -> None:
        self.scaffold_query = data
        self._update_query_summary()
        self._maybe_commit()

    @Inputs.search_profile
    def set_search_profile(self, data: Optional[Table]) -> None:
        self.search_profile = data
        if data is not None and len(data):
            _smiles, _scaffold, _generic, threshold, motif_logic = query_from_search_profile(data)
            self.min_similarity = float(threshold)
            self.min_sim_spin.setValue(float(threshold))
            self.motif_logic = motif_logic or "or"
            self.logic_combo.setCurrentText(self.motif_logic)
        self._update_query_summary()
        self._maybe_commit()

    @Inputs.reference_data
    def set_reference_data(self, data: Optional[Table]) -> None:
        self.reference_data = data
        self._populate_reference_combos()
        self._update_status(
            format_loaded_status(len(data), item_label="rows", prefix="Reference loaded")
            if data is not None
            else format_waiting_status("reference data")
        )
        self._maybe_commit()

    def _resolve_query_context(self) -> tuple[str, str, str]:
        smiles = ""
        scaffold = ""
        generic = ""
        if self.search_profile is not None and len(self.search_profile):
            smiles, scaffold, generic, _threshold, motif_logic = query_from_search_profile(self.search_profile)
            if motif_logic:
                self.motif_logic = motif_logic
        if not smiles and self.query_molecule is not None:
            try:
                smiles = self.query_molecule.canonical_smiles()
            except Exception:
                smiles = ""
        if self.scaffold_query is not None and len(self.scaffold_query):
            row = self.scaffold_query[0]
            try:
                scaffold = str(row["Murcko Scaffold"]).strip() or scaffold
            except Exception:
                pass
            try:
                generic = str(row["Generic Scaffold"]).strip() or generic
            except Exception:
                pass
        return smiles, scaffold, generic

    def commit(self) -> None:
        if self.reference_data is None:
            self._update_status(format_required_inputs_status("Reference data"))
            self.Outputs.ranked_hits.send(None)
            self.Outputs.hit_compounds.send(None)
            return

        query_smiles, query_scaffold, query_generic = self._resolve_query_context()
        if not query_smiles:
            self._update_status(format_no_input_status("query molecule"))
            self.Outputs.ranked_hits.send(None)
            self.Outputs.hit_compounds.send(None)
            return

        refs = references_from_table(
            self.reference_data,
            smiles_var_name=self.reference_smiles_var_name,
            name_var_name=self.reference_name_var_name,
        )
        fragment_queries = fragments_from_query_table(self.fragment_queries)
        motif_queries = fragments_from_query_table(self.motif_queries)
        mode = self._SEARCH_MODES[self.search_mode_idx][1]
        hits = run_pharmafp_search(
            query_smiles=query_smiles,
            reference=refs,
            fragment_queries=fragment_queries,
            motif_queries=motif_queries,
            motif_logic=self.motif_logic,
            query_scaffold=query_scaffold,
            query_generic_scaffold=query_generic,
            top_k=int(self.top_k),
            min_similarity=float(self.min_similarity),
            mode=mode,
        )
        hit_table = pharmafp_search_hits_table(hits)
        self.Outputs.ranked_hits.send(hit_table)
        ref_indices = [hit.source_index for hit in hits]
        self.Outputs.hit_compounds.send(self.reference_data[ref_indices] if ref_indices else self.reference_data[:0])
        self._update_status(
            format_done_status(
                f"ranked hits={len(hits)}",
                f"mode={mode}",
                f"motif logic={self.motif_logic.upper()}",
                prefix="Found",
            )
        )
