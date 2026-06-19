from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QPixmap
from AnyQt.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from rdkit import Chem

from chem_inf_widgets.chemcore.services import mol_depict
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles
from chem_inf_widgets.widgets.ui_helpers import (
    format_loaded_status,
    format_waiting_status,
)


_CLASSIC_PAIR_LEFT_CANDIDATES = [
    "smiles_a",
]
_CLASSIC_PAIR_RIGHT_CANDIDATES = [
    "smiles_b",
]
_TRANSFORMATION_INPUT_CANDIDATES = [
    "smiles_orig",
    "standardization_input_smiles",
    "std_input_smiles",
    "input_smiles",
    "raw_smiles",
    "canonical_smiles",
    "smiles",
]
_STANDARDIZATION_OUTPUT_CANDIDATES = [
    "smiles_std",
    "canonical_smiles_std",
    "standardized_smiles",
    "standardization_output_smiles",
    "std_output_smiles",
]
_TRANSFORMATION_OUTPUT_CANDIDATES = [
    *_STANDARDIZATION_OUTPUT_CANDIDATES,
    "canonical_smiles",
    "smiles",
]
_STANDARDIZATION_NAME_CANDIDATES = [
    "name",
    "compound_name",
    "compound_id",
    "molecule_id",
    "id",
    "identifier",
    "title",
    "row_id",
]


def _string_vars(data: Table) -> List[StringVariable]:
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    return [var for var in variables if isinstance(var, StringVariable)]


def _continuous_vars(data: Table) -> List[ContinuousVariable]:
    variables = list(data.domain.attributes) + list(data.domain.class_vars) + list(data.domain.metas)
    return [var for var in variables if isinstance(var, ContinuousVariable)]


def _normalize_var_name(name: str) -> str:
    text = str(name or "").strip().lower()
    text = re.sub(r"[\s\-/]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text


def _base_var_name(name: str) -> str:
    return re.sub(r"_\d+$", "", _normalize_var_name(name))


def _candidate_keys(candidates: List[str]) -> set[str]:
    return {_base_var_name(name) for name in candidates}


def _find_var_name(
    data: Table,
    candidates: List[str],
    *,
    string_only: bool = True,
    exclude: Optional[set[str]] = None,
) -> str:
    variables = _string_vars(data) if string_only else _continuous_vars(data)
    normalized = _candidate_keys(candidates)
    excluded = {_base_var_name(name) for name in (exclude or set())}
    match = next(
        (
            var.name
            for var in variables
            if _base_var_name(var.name) in normalized
            and _base_var_name(var.name) not in excluded
        ),
        "",
    )
    return match


def _is_smiles_like_var(var: StringVariable) -> bool:
    base_name = _base_var_name(var.name)
    if "smiles" in base_name or base_name == "smile":
        return True
    return str(var.attributes.get("format", "") or "").strip().upper() == "SMILES"


def _smiles_like_names(data: Optional[Table]) -> list[str]:
    if data is None:
        return []
    return [var.name for var in _string_vars(data) if _is_smiles_like_var(var)]


def _find_pair_by_candidates(
    data: Optional[Table],
    left_candidates: List[str],
    right_candidates: List[str],
) -> tuple[str, str]:
    if data is None:
        return "", ""
    left = _find_var_name(data, left_candidates)
    right = _find_var_name(data, right_candidates, exclude={left} if left else None)
    if left and right:
        return left, right
    return "", ""


def _detect_smiles_pair(data: Optional[Table]) -> tuple[str, str]:
    if data is None:
        return "", ""

    candidates = [
        (_CLASSIC_PAIR_LEFT_CANDIDATES, _CLASSIC_PAIR_RIGHT_CANDIDATES),
        (_TRANSFORMATION_INPUT_CANDIDATES[:-1], _STANDARDIZATION_OUTPUT_CANDIDATES),
        (_TRANSFORMATION_INPUT_CANDIDATES, _STANDARDIZATION_OUTPUT_CANDIDATES),
        (_TRANSFORMATION_INPUT_CANDIDATES[:-1], _TRANSFORMATION_OUTPUT_CANDIDATES),
        (_TRANSFORMATION_INPUT_CANDIDATES, _TRANSFORMATION_OUTPUT_CANDIDATES),
    ]
    for left_candidates, right_candidates in candidates:
        left, right = _find_pair_by_candidates(data, left_candidates, right_candidates)
        if left and right:
            return left, right

    smiles_like = _smiles_like_names(data)
    if len(smiles_like) >= 2:
        return smiles_like[0], smiles_like[1]
    return "", ""


def _comparison_mode(smiles_a_var: str, smiles_b_var: str) -> str:
    left = _base_var_name(smiles_a_var)
    right = _base_var_name(smiles_b_var)
    if left in _candidate_keys(_CLASSIC_PAIR_LEFT_CANDIDATES) and right in _candidate_keys(_CLASSIC_PAIR_RIGHT_CANDIDATES):
        return "classic"
    if right in _candidate_keys(_STANDARDIZATION_OUTPUT_CANDIDATES):
        return "standardization"
    if left in _candidate_keys(_TRANSFORMATION_INPUT_CANDIDATES) or right in _candidate_keys(_TRANSFORMATION_OUTPUT_CANDIDATES):
        return "transformation"
    if left and right and ("smiles" in left or "smiles" in right):
        return "transformation"
    return ""


def _single_name_var(data: Optional[Table]) -> str:
    if data is None:
        return ""
    name_var = _find_var_name(data, _STANDARDIZATION_NAME_CANDIDATES)
    if name_var:
        return name_var
    smiles_like = set(_smiles_like_names(data))
    for var in _string_vars(data):
        if var.name not in smiles_like:
            return var.name
    return ""


def _detect_name_pair(data: Optional[Table], smiles_a_var: str, smiles_b_var: str) -> tuple[str, str]:
    if data is None:
        return "", ""
    if _comparison_mode(smiles_a_var, smiles_b_var) != "classic":
        name_var = _single_name_var(data)
        return name_var, name_var
    return (
        _find_var_name(data, ["name_a", "name a"]),
        _find_var_name(data, ["name_b", "name b"]),
    )


def _row_text(row, var_name: str) -> str:
    if not var_name:
        return ""
    try:
        value = row[var_name]
    except Exception:
        return ""
    if value is None:
        return ""
    return str(value).strip()


def _row_float(row, var_name: str) -> float:
    if not var_name:
        return float("nan")
    try:
        return float(row[var_name])
    except Exception:
        return float("nan")


def _mol_pixmap(smiles: str, size: int = 320) -> Optional[QPixmap]:
    clean = (smiles or "").strip()
    if not clean:
        return None
    mol = safe_mol_from_smiles(clean, sanitize=True, remove_hs=True).mol
    if mol is None:
        return None
    png = mol_depict.render_mol_png(
        mol,
        size=size,
        suppress_isolated_metal_radicals=True,
    )
    pixmap = QPixmap()
    pixmap.loadFromData(png)
    return pixmap


def _pair_compounds_table(
    row,
    *,
    role_a: str,
    role_b: str,
    name_a_var: str,
    name_b_var: str,
    smiles_a_var: str,
    smiles_b_var: str,
    activity_a_var: str,
    activity_b_var: str,
    similarity_var: str,
    ratio_var: str,
    cliff_score_var: str,
    higher_active_var: str,
) -> Table:
    domain = Domain(
        [
            ContinuousVariable("Activity"),
            ContinuousVariable("Pair Similarity"),
            ContinuousVariable("Pair Activity Ratio"),
            ContinuousVariable("Pair Cliff Score"),
        ],
        metas=[
            StringVariable("Pair Role"),
            StringVariable("Name"),
            StringVariable("SMILES"),
            StringVariable("Higher Active"),
        ],
    )
    similarity = _row_float(row, similarity_var)
    ratio = _row_float(row, ratio_var)
    cliff_score = _row_float(row, cliff_score_var)
    higher_active = _row_text(row, higher_active_var)
    X = np.array(
        [
            [_row_float(row, activity_a_var), similarity, ratio, cliff_score],
            [_row_float(row, activity_b_var), similarity, ratio, cliff_score],
        ],
        dtype=float,
    )
    metas = np.array(
        [
            [role_a, _row_text(row, name_a_var), _row_text(row, smiles_a_var), higher_active],
            [role_b, _row_text(row, name_b_var), _row_text(row, smiles_b_var), higher_active],
        ],
        dtype=object,
    )
    return Table.from_numpy(domain, X=X, metas=metas)


class _MolPanel(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600; font-size: 14px;")
        layout.addWidget(self.title_label)

        self.image_label = QLabel("No structure")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(320)
        self.image_label.setStyleSheet("background: white; border: 1px solid #d0d7de;")
        layout.addWidget(self.image_label)

        self.name_label = QLabel("")
        self.name_label.setWordWrap(True)
        layout.addWidget(self.name_label)

        self.smiles_label = QLabel("")
        self.smiles_label.setWordWrap(True)
        self.smiles_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.smiles_label)

        self.activity_label = QLabel("")
        layout.addWidget(self.activity_label)

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)

    def set_content(self, *, name: str, smiles: str, activity: str, pixmap: Optional[QPixmap]) -> None:
        self.name_label.setText(f"Name: {name or '—'}")
        self.smiles_label.setText(f"SMILES: {smiles or '—'}")
        self.activity_label.setText(f"Activity: {activity}" if activity else "")
        self.activity_label.setVisible(bool(activity))
        if pixmap is None:
            self.image_label.setText("No structure")
            self.image_label.setPixmap(QPixmap())
        else:
            self.image_label.setText("")
            self.image_label.setPixmap(
                pixmap.scaled(
                    320,
                    320,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )


class OWPairViewer(OWWidget):
    name = "Pair Viewer"
    description = "Inspect paired compounds side by side from a table containing two SMILES columns."
    icon = "icons/editors_viewers/owpairviewerwidget.png"
    priority = 138

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        selected_pairs = Output("Selected Pairs", Table)
        pair_compounds = Output("Pair Compounds", Table)

    smiles_a_var_name: str = Setting("")
    smiles_b_var_name: str = Setting("")
    name_a_var_name: str = Setting("")
    name_b_var_name: str = Setting("")
    activity_a_var_name: str = Setting("")
    activity_b_var_name: str = Setting("")
    similarity_var_name: str = Setting("")
    ratio_var_name: str = Setting("")
    cliff_score_var_name: str = Setting("")
    higher_active_var_name: str = Setting("")

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[Table] = None
        self._selected_index: Optional[int] = None

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)

        self.smiles_a_combo = QComboBox()
        self.smiles_a_combo.currentTextChanged.connect(self._on_controls_changed)
        form.addRow("SMILES A:", self.smiles_a_combo)

        self.smiles_b_combo = QComboBox()
        self.smiles_b_combo.currentTextChanged.connect(self._on_controls_changed)
        form.addRow("SMILES B:", self.smiles_b_combo)

        self.name_a_combo = QComboBox()
        self.name_a_combo.currentTextChanged.connect(self._on_controls_changed)
        form.addRow("Name A:", self.name_a_combo)

        self.name_b_combo = QComboBox()
        self.name_b_combo.currentTextChanged.connect(self._on_controls_changed)
        form.addRow("Name B:", self.name_b_combo)

        controls = QWidget()
        controls.setLayout(form)
        self.controlArea.layout().addWidget(controls)

        self.status_label = QLabel(format_waiting_status("pair table"))
        self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)

        splitter = QSplitter(Qt.Horizontal)
        self.mainArea.layout().addWidget(splitter)

        self.pair_list = QListWidget()
        self.pair_list.currentRowChanged.connect(self._on_row_selected)
        splitter.addWidget(self.pair_list)

        pair_widget = QWidget()
        pair_layout = QHBoxLayout(pair_widget)
        pair_layout.setContentsMargins(0, 0, 0, 0)
        pair_layout.setSpacing(10)

        self.panel_a = _MolPanel("Compound A")
        self.panel_b = _MolPanel("Compound B")
        pair_layout.addWidget(self.panel_a)
        pair_layout.addWidget(self.panel_b)
        splitter.addWidget(pair_widget)
        splitter.setSizes([260, 840])

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self.data = data
        self._selected_index = None
        self._refresh_controls()
        self._refresh_rows()
        self._send_outputs()

    def _refresh_controls(self) -> None:
        string_names = [var.name for var in _string_vars(self.data)] if self.data is not None else []
        continuous_names = [var.name for var in _continuous_vars(self.data)] if self.data is not None else []
        for combo in (self.smiles_a_combo, self.smiles_b_combo, self.name_a_combo, self.name_b_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("")
            combo.addItems(string_names)
            combo.blockSignals(False)

        default_smiles_a, default_smiles_b = _detect_smiles_pair(self.data)
        default_name_a, default_name_b = _detect_name_pair(self.data, default_smiles_a, default_smiles_b)
        defaults = {
            "smiles_a_var_name": default_smiles_a,
            "smiles_b_var_name": default_smiles_b,
            "name_a_var_name": default_name_a,
            "name_b_var_name": default_name_b,
        }

        for attr_name, combo in (
            ("smiles_a_var_name", self.smiles_a_combo),
            ("smiles_b_var_name", self.smiles_b_combo),
            ("name_a_var_name", self.name_a_combo),
            ("name_b_var_name", self.name_b_combo),
        ):
            value = getattr(self, attr_name)
            if value not in string_names:
                value = defaults[attr_name]
                setattr(self, attr_name, value)
            combo.setCurrentText(value)

        if self.data is None:
            self.activity_a_var_name = ""
            self.activity_b_var_name = ""
            self.similarity_var_name = ""
            self.ratio_var_name = ""
            self.cliff_score_var_name = ""
            self.higher_active_var_name = ""
            self._update_panel_titles()
            return

        self.activity_a_var_name = self.activity_a_var_name if self.activity_a_var_name in continuous_names else _find_var_name(self.data, ["activity a", "activity_a"], string_only=False)
        self.activity_b_var_name = self.activity_b_var_name if self.activity_b_var_name in continuous_names else _find_var_name(self.data, ["activity b", "activity_b"], string_only=False)
        self.similarity_var_name = self.similarity_var_name if self.similarity_var_name in continuous_names else _find_var_name(self.data, ["similarity"], string_only=False)
        self.ratio_var_name = self.ratio_var_name if self.ratio_var_name in continuous_names else _find_var_name(self.data, ["activity ratio", "activity_ratio"], string_only=False)
        self.cliff_score_var_name = self.cliff_score_var_name if self.cliff_score_var_name in continuous_names else _find_var_name(self.data, ["cliff score", "cliff_score"], string_only=False)
        self.higher_active_var_name = self.higher_active_var_name if self.higher_active_var_name in string_names else _find_var_name(self.data, ["higher active", "higher_active"])
        self._update_panel_titles()

    def _refresh_rows(self) -> None:
        self.pair_list.clear()
        self.panel_a.set_content(name="", smiles="", activity="", pixmap=None)
        self.panel_b.set_content(name="", smiles="", activity="", pixmap=None)
        self._update_panel_titles()

        if self.data is None or len(self.data) == 0:
            self.status_label.setText(format_waiting_status("pair table"))
            return

        if not self.smiles_a_var_name or not self.smiles_b_var_name:
            self.status_label.setText("Select the two SMILES columns to inspect pairs.")
            return

        for index, row in enumerate(self.data):
            score = _row_float(row, self.cliff_score_var_name)
            similarity = _row_float(row, self.similarity_var_name)
            label_a = _row_text(row, self.name_a_var_name) or _row_text(row, self.smiles_a_var_name)[:28]
            label_b = _row_text(row, self.name_b_var_name) or _row_text(row, self.smiles_b_var_name)[:28]
            if _comparison_mode(self.smiles_a_var_name, self.smiles_b_var_name) != "classic":
                text = f"{index + 1}. {label_a or label_b or 'Structure comparison'}"
            else:
                text = f"{index + 1}. {label_a} vs {label_b}"
            if np.isfinite(score):
                text += f" | cliff={score:.3f}"
            if np.isfinite(similarity):
                text += f" | sim={similarity:.3f}"
            item = QListWidgetItem(text)
            self.pair_list.addItem(item)

        self.status_label.setText(
            format_loaded_status(len(self.data), item_label="pairs")
            + " Select one to inspect side by side."
        )
        if len(self.data):
            self.pair_list.setCurrentRow(0)

    def _on_controls_changed(self, *_args) -> None:
        self.smiles_a_var_name = self.smiles_a_combo.currentText()
        self.smiles_b_var_name = self.smiles_b_combo.currentText()
        self.name_a_var_name = self.name_a_combo.currentText()
        self.name_b_var_name = self.name_b_combo.currentText()
        self._update_panel_titles()
        self._refresh_rows()
        self._send_outputs()

    def _on_row_selected(self, row_index: int) -> None:
        self._selected_index = row_index if row_index >= 0 else None
        self._update_panels()
        self._send_outputs()

    def _update_panels(self) -> None:
        if self.data is None or self._selected_index is None or self._selected_index >= len(self.data):
            self.panel_a.set_content(name="", smiles="", activity="", pixmap=None)
            self.panel_b.set_content(name="", smiles="", activity="", pixmap=None)
            return

        row = self.data[self._selected_index]
        smiles_a = _row_text(row, self.smiles_a_var_name)
        smiles_b = _row_text(row, self.smiles_b_var_name)
        name_a = _row_text(row, self.name_a_var_name)
        name_b = _row_text(row, self.name_b_var_name)
        activity_a = _row_text(row, self.activity_a_var_name)
        activity_b = _row_text(row, self.activity_b_var_name)

        self.panel_a.set_content(name=name_a, smiles=smiles_a, activity=activity_a, pixmap=_mol_pixmap(smiles_a))
        self.panel_b.set_content(name=name_b, smiles=smiles_b, activity=activity_b, pixmap=_mol_pixmap(smiles_b))

    def _update_panel_titles(self) -> None:
        mode = _comparison_mode(self.smiles_a_var_name, self.smiles_b_var_name)
        if mode == "standardization":
            self.panel_a.set_title("Original Structure")
            self.panel_b.set_title("Standardized Structure")
        elif mode == "transformation":
            self.panel_a.set_title("Input Structure")
            self.panel_b.set_title("Output Structure")
        else:
            self.panel_a.set_title("Compound A")
            self.panel_b.set_title("Compound B")

    def _send_outputs(self) -> None:
        if self.data is None or self._selected_index is None or self._selected_index >= len(self.data):
            self.Outputs.selected_pairs.send(None)
            self.Outputs.pair_compounds.send(None)
            return

        selected = self.data[self._selected_index : self._selected_index + 1]
        self.Outputs.selected_pairs.send(selected)
        mode = _comparison_mode(self.smiles_a_var_name, self.smiles_b_var_name)
        if mode == "standardization":
            role_a, role_b = "Original", "Standardized"
        elif mode == "transformation":
            role_a, role_b = "Input", "Output"
        else:
            role_a, role_b = "A", "B"
        self.Outputs.pair_compounds.send(
            _pair_compounds_table(
                self.data[self._selected_index],
                role_a=role_a,
                role_b=role_b,
                name_a_var=self.name_a_var_name,
                name_b_var=self.name_b_var_name,
                smiles_a_var=self.smiles_a_var_name,
                smiles_b_var=self.smiles_b_var_name,
                activity_a_var=self.activity_a_var_name,
                activity_b_var=self.activity_b_var_name,
                similarity_var=self.similarity_var_name,
                ratio_var=self.ratio_var_name,
                cliff_score_var=self.cliff_score_var_name,
                higher_active_var=self.higher_active_var_name,
            )
        )
