from __future__ import annotations

import io
from typing import List, Optional

import numpy as np
from AnyQt.QtCore import Qt, pyqtSignal, pyqtSlot as Slot
from AnyQt.QtGui import QCursor, QImage, QPixmap
from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.scaffold_service import (
    NO_SCAFFOLD_LABEL,
    ScaffoldAnalysisResult,
    build_scaffold_summary,
    analyze_scaffolds,
)
from chem_inf_widgets.widgets.ui_helpers import (
    format_done_status,
    format_error_status,
    format_no_input_status,
    set_widget_error,
)

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    _RDKIT_DRAW_OK = True
except ImportError:
    _RDKIT_DRAW_OK = False

_CARD_W = 200
_CARD_IMG_H = 130
_COLS = 4


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_smiles_vars(data: Table) -> List[StringVariable]:
    wanted = {"smiles", "canonical_smiles", "smile"}
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    preferred = [v for v in variables if isinstance(v, StringVariable) and v.name.strip().lower() in wanted]
    if preferred:
        return preferred + [v for v in variables if isinstance(v, StringVariable) and v not in preferred]
    return [v for v in variables if isinstance(v, StringVariable)]


def _table_smiles(data: Table, var_name: str) -> List[str]:
    variables = _find_smiles_vars(data)
    selected_var = next((v for v in variables if v.name == var_name), None)
    if selected_var is None:
        raise ValueError("No SMILES column selected.")
    col = data.get_column(selected_var)
    return ["" if value is None else str(value).strip() for value in col]


def _molecule_smiles(molecules: List[ChemMol]) -> List[str]:
    smiles: List[str] = []
    for molecule in molecules:
        value = molecule.get_prop("SMILES") or molecule.get_prop("smiles")
        if isinstance(value, str) and value.strip():
            smiles.append(value.strip())
            continue
        try:
            smiles.append(molecule.canonical_smiles())
        except ValueError:
            smiles.append("")
    return smiles


def _smiles_to_pixmap(smiles: str, w: int, h: int) -> Optional[QPixmap]:
    if not _RDKIT_DRAW_OK or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        drawer.drawOptions().addStereoAnnotation = False
        drawer.drawOptions().explicitMethyl = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().encode()
        qimg = QImage.fromData(svg, "SVG")
        if qimg.isNull():
            # fallback: PIL image
            img = Draw.MolToImage(mol, size=(w, h))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            qimg = QImage.fromData(buf.getvalue())
        if qimg.isNull():
            return None
        return QPixmap.fromImage(qimg)
    except Exception:
        return None


# ── scaffold card widget ──────────────────────────────────────────────────────

class _ScaffoldCard(QFrame):
    clicked = pyqtSignal(str)   # emits scaffold SMILES

    _STYLE_NORMAL = "QFrame { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; }"
    _STYLE_SEL    = "QFrame { background:#EFF6FF; border:2px solid #2563EB; border-radius:8px; }"

    def __init__(self, rank: int, smiles: str, count: int, fraction: float, parent=None):
        super().__init__(parent)
        self._smiles = smiles
        self._selected = False
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(_CARD_W)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.setStyleSheet(self._STYLE_NORMAL)
        self.setCursor(QCursor(Qt.PointingHandCursor))

        vl = QVBoxLayout(self)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(4)

        # rank chip
        rank_lbl = QLabel(f"#{rank}")
        rank_lbl.setStyleSheet(
            "font-size:10px; font-weight:700; color:#64748B; "
            "background:#F1F5F9; border-radius:4px; padding:2px 6px;"
        )
        rank_lbl.setAlignment(Qt.AlignLeft)

        # molecule image
        self._img_lbl = QLabel()
        self._img_lbl.setFixedSize(_CARD_W - 16, _CARD_IMG_H)
        self._img_lbl.setAlignment(Qt.AlignCenter)
        self._img_lbl.setStyleSheet("background:#F8FAFC; border-radius:4px; border:none;")
        self._img_lbl.setText("…")

        # frequency bar
        bar_w = _CARD_W - 16
        bar_bg = QFrame()
        bar_bg.setFixedSize(bar_w, 6)
        bar_bg.setStyleSheet("background:#E2E8F0; border-radius:3px; border:none;")
        filled_w = max(4, int(bar_w * fraction))
        self._bar_fill = QFrame(bar_bg)
        self._bar_fill.setGeometry(0, 0, filled_w, 6)
        self._bar_fill.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #2563EB, stop:1 #38BDF8); border-radius:3px; border:none;"
        )

        # count / fraction label
        info_lbl = QLabel(f"{count} molecules  ·  {fraction*100:.1f}%")
        info_lbl.setStyleSheet("font-size:11px; color:#475569;")
        info_lbl.setAlignment(Qt.AlignLeft)

        # SMILES (truncated)
        smi_short = smiles if len(smiles) <= 28 else smiles[:25] + "…"
        smi_lbl = QLabel(smi_short)
        smi_lbl.setStyleSheet("font-size:9px; color:#94A3B8; font-family:monospace;")
        smi_lbl.setToolTip(smiles)
        smi_lbl.setWordWrap(False)

        vl.addWidget(rank_lbl)
        vl.addWidget(self._img_lbl)
        vl.addWidget(bar_bg)
        vl.addWidget(info_lbl)
        vl.addWidget(smi_lbl)

        # render molecule asynchronously-ish: set pixmap if available
        px = _smiles_to_pixmap(smiles, _CARD_W - 16, _CARD_IMG_H)
        if px is not None:
            self._img_lbl.setPixmap(px)
            self._img_lbl.setText("")
        else:
            self._img_lbl.setText("(no structure)")

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.setStyleSheet(self._STYLE_SEL if selected else self._STYLE_NORMAL)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._smiles)
        super().mousePressEvent(event)


# ── widget ────────────────────────────────────────────────────────────────────

class OWScaffoldAnalysis(OWWidget):
    name = "Scaffold Analysis"
    description = "Annotate molecules with Murcko scaffolds and build a scaffold frequency summary."
    icon = "icons/analysis/owscaffoldanalysiswidget.svg"
    priority = 135
    want_main_area = True
    resizing_enabled = True

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        annotated_data = Output("Annotated Data", Table)
        scaffold_summary = Output("Scaffold Summary", Table)
        annotated_molecules = Output("Annotated Molecules", list, auto_summary=False)
        selected = Output("Selected Scaffold", Table)

    smiles_var_name: str = Setting("")
    summary_kind_idx: int = Setting(0)
    include_acyclic: bool = Setting(True)
    top_n: int = Setting(20)
    selected_scaffold: str = Setting("")
    auto_run: bool = Setting(True)

    _SUMMARY_KINDS = [
        ("Exact Murcko", "murcko"),
        ("Generic Murcko", "generic"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[Table] = None
        self.molecules: List[ChemMol] = []
        self._executor = ThreadExecutor(self)
        self._cards: dict[str, _ScaffoldCard] = {}       # smiles → card
        self._last_annotated_data: Optional[Table] = None

        self._build_control_area()
        self._build_main_area()
        self._set_status("Waiting for input…", ok=True)
        self._update_smiles_controls()

    # ── control area ──────────────────────────────────────────────────────────

    def _build_control_area(self) -> None:
        ca = self.controlArea

        # header
        hdr = QWidget()
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(6, 6, 6, 6)
        left = QVBoxLayout()
        left.addWidget(QLabel("Scaffold Analysis", objectName="HdrTitle"))
        left.addWidget(QLabel("Murcko scaffold frequency with structure gallery", objectName="HdrSub"))
        hl.addLayout(left, 1)
        self._lbl_status = QLabel("Ready", objectName="StatusChip")
        self._lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hl.addWidget(self._lbl_status)
        ca.layout().addWidget(hdr)

        box = QGroupBox("Settings")
        vl = QVBoxLayout(box)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("SMILES column"))
        self.smiles_combo = QComboBox()
        self.smiles_combo.currentTextChanged.connect(self._on_smiles_changed)
        row1.addWidget(self.smiles_combo, 1)
        vl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Scaffold type"))
        self.summary_kind_combo = QComboBox()
        self.summary_kind_combo.addItems([label for label, _ in self._SUMMARY_KINDS])
        self.summary_kind_combo.setCurrentIndex(int(self.summary_kind_idx))
        self.summary_kind_combo.currentIndexChanged.connect(self._on_summary_kind_changed)
        row2.addWidget(self.summary_kind_combo, 1)
        vl.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Show top"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 200)
        self.top_n_spin.setValue(int(self.top_n))
        self.top_n_spin.valueChanged.connect(self._on_top_n_changed)
        row3.addWidget(self.top_n_spin, 1)
        vl.addLayout(row3)

        self.include_acyclic_check = QCheckBox("Include acyclic compounds")
        self.include_acyclic_check.setChecked(bool(self.include_acyclic))
        self.include_acyclic_check.toggled.connect(self._on_include_acyclic_changed)
        vl.addWidget(self.include_acyclic_check)

        self.auto_run_check = QCheckBox("Auto-run")
        self.auto_run_check.setChecked(bool(self.auto_run))
        self.auto_run_check.toggled.connect(self._on_auto_run_toggled)
        vl.addWidget(self.auto_run_check)

        self.run_button = QPushButton("Analyze scaffolds")
        self.run_button.clicked.connect(self.commit)
        vl.addWidget(self.run_button)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        vl.addWidget(self._progress)

        ca.layout().addWidget(box)
        ca.layout().addStretch(1)

    # ── main area ─────────────────────────────────────────────────────────────

    def _build_main_area(self) -> None:
        # header row with summary stats
        stats_row = QWidget()
        hl = QHBoxLayout(stats_row)
        hl.setContentsMargins(8, 4, 8, 4)
        hl.setSpacing(12)
        self._stat_total = self._make_stat_chip("Scaffolds", "—")
        self._stat_coverage = self._make_stat_chip("Coverage", "—")
        self._stat_top = self._make_stat_chip("Top scaffold", "—")
        for w in (self._stat_total, self._stat_coverage, self._stat_top):
            hl.addWidget(w)
        hl.addStretch(1)
        self.mainArea.layout().addWidget(stats_row)

        # divider
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setFixedHeight(1)
        div.setStyleSheet("background:#E2E8F0; border:none;")
        self.mainArea.layout().addWidget(div)

        # scroll area for scaffold gallery
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: #F8FAFC; }")

        self._gallery_widget = QWidget()
        self._gallery_widget.setStyleSheet("background: #F8FAFC;")
        self._gallery_layout = QVBoxLayout(self._gallery_widget)
        self._gallery_layout.setContentsMargins(12, 12, 12, 12)
        self._gallery_layout.setSpacing(12)
        self._scroll.setWidget(self._gallery_widget)
        self.mainArea.layout().addWidget(self._scroll, 1)

        # placeholder label
        self._placeholder = QLabel("Run Scaffold Analysis to see scaffold structures here.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color:#94A3B8; font-size:13px; padding:40px;")
        self._gallery_layout.addWidget(self._placeholder)
        self._gallery_layout.addStretch(1)

    @staticmethod
    def _make_stat_chip(title: str, value: str) -> QWidget:
        w = QWidget()
        w.setStyleSheet(
            "QWidget { background:#FFFFFF; border:1px solid #E2E8F0; "
            "border-radius:8px; padding:6px 12px; }"
        )
        vl = QVBoxLayout(w)
        vl.setContentsMargins(10, 6, 10, 6)
        vl.setSpacing(1)
        t = QLabel(title)
        t.setStyleSheet("font-size:10px; font-weight:700; color:#94A3B8; "
                        "letter-spacing:0.5px; text-transform:uppercase; border:none;")
        v = QLabel(value)
        v.setStyleSheet("font-size:15px; font-weight:700; color:#0F172A; border:none;")
        v.setObjectName("stat_value")
        vl.addWidget(t)
        vl.addWidget(v)
        return w

    @staticmethod
    def _set_stat(chip: QWidget, value: str) -> None:
        lbl = chip.findChild(QLabel, "stat_value")
        if lbl:
            lbl.setText(value)

    # ── status ────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, ok: bool = True) -> None:
        self._lbl_status.setText(msg)
        if ok:
            self._lbl_status.setStyleSheet(
                "padding:4px 8px;border:1px solid #e1e1e1;border-radius:10px;background:#fafafa;"
            )
        else:
            self._lbl_status.setStyleSheet(
                "padding:4px 8px;border:1px solid #f2c2c2;border-radius:10px;"
                "background:#fff5f5;color:#a40000;"
            )

    def _set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        self.smiles_combo.setEnabled(not busy and self.data is not None)
        self._progress.setVisible(busy)

    # ── inputs ────────────────────────────────────────────────────────────────

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self.data = data
        self._populate_smiles_combo()
        rows = 0 if data is None else len(data)
        self._set_status(f"Table: {rows} rows" if data else "No table input.", ok=data is not None)
        self._maybe_autorun()

    @Inputs.molecules
    def set_molecules(self, molecules: Optional[list]) -> None:
        self.molecules = [m for m in (molecules or []) if isinstance(m, ChemMol)]
        if not self.data:
            self._set_status(f"Molecules: {len(self.molecules)}", ok=bool(self.molecules))
        self._maybe_autorun()

    def _populate_smiles_combo(self) -> None:
        self.smiles_combo.blockSignals(True)
        try:
            self.smiles_combo.clear()
            if self.data is None:
                return
            smiles_vars = _find_smiles_vars(self.data)
            self.smiles_combo.addItems([v.name for v in smiles_vars])
            if smiles_vars:
                names = [v.name for v in smiles_vars]
                if self.smiles_var_name and self.smiles_var_name in names:
                    self.smiles_combo.setCurrentText(self.smiles_var_name)
                else:
                    self.smiles_var_name = names[0]
                    self.smiles_combo.setCurrentText(self.smiles_var_name)
        finally:
            self.smiles_combo.blockSignals(False)
        self._update_smiles_controls()

    def _update_smiles_controls(self) -> None:
        self.smiles_combo.setEnabled(self.data is not None)

    def _on_smiles_changed(self, text: str) -> None:
        self.smiles_var_name = text
        self._maybe_autorun()

    def _on_summary_kind_changed(self, index: int) -> None:
        self.summary_kind_idx = int(index)
        self._maybe_autorun()

    def _on_include_acyclic_changed(self, checked: bool) -> None:
        self.include_acyclic = bool(checked)
        self._maybe_autorun()

    def _on_top_n_changed(self, value: int) -> None:
        self.top_n = int(value)
        self._maybe_autorun()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        self.auto_run = bool(checked)
        self._maybe_autorun()

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and not self._progress.isVisible() and (self.data is not None or self.molecules):
            self.commit()

    # ── commit (async) ────────────────────────────────────────────────────────

    def commit(self) -> None:
        self.clear_messages()
        if self.data is None and not self.molecules:
            self._set_status(format_no_input_status(), ok=False)
            self._send_empty()
            return

        try:
            smiles = (
                _table_smiles(self.data, self.smiles_var_name)
                if self.data is not None
                else _molecule_smiles(self.molecules)
            )
        except ValueError as exc:
            set_widget_error(self, str(exc))
            self._set_status(format_error_status(str(exc)), ok=False)
            return

        self._set_busy(True)
        self._set_status("Analyzing…", ok=True)

        data_snapshot = self.data
        molecules_snapshot = list(self.molecules)
        kind = self._SUMMARY_KINDS[self.summary_kind_idx][1]
        include_acyclic = bool(self.include_acyclic)
        top_n = int(self.top_n)

        def _run():
            result = analyze_scaffolds(smiles)
            summary_rows = build_scaffold_summary(
                result, kind=kind, include_acyclic=include_acyclic, top_n=top_n
            )
            return result, summary_rows, data_snapshot, molecules_snapshot

        fut = self._executor.submit(_run)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut) -> None:
        try:
            payload = fut.result()
            methodinvoke(self, "_finish", (object,))(payload)
        except Exception as e:
            methodinvoke(self, "_fail", (str,))(str(e))

    @Slot(object)
    def _finish(self, payload: object) -> None:
        result, summary_rows, data_snapshot, molecules_snapshot = payload
        self._set_busy(False)

        annotated_data = self._annotate_table(data_snapshot, result) if data_snapshot is not None else None
        summary_table = self._rows_to_table(summary_rows)
        annotated_molecules = self._annotate_molecules(result, molecules_snapshot)

        self._last_annotated_data = annotated_data
        self.Outputs.annotated_data.send(annotated_data)
        self.Outputs.scaffold_summary.send(summary_table)
        self.Outputs.annotated_molecules.send(annotated_molecules)
        self._send_selected()

        kind_label = self._SUMMARY_KINDS[self.summary_kind_idx][0]
        top_smi = summary_rows[0].scaffold if summary_rows else NO_SCAFFOLD_LABEL
        self._set_status(
            format_done_status(
                f"valid={result.valid_count}",
                f"invalid={len(result.failed_indices)}",
                f"top={top_smi or NO_SCAFFOLD_LABEL}",
            ),
            ok=True,
        )

        n_unique = len({r.scaffold for r in summary_rows})
        cov = result.valid_count / max(1, result.valid_count + len(result.failed_indices))
        self._set_stat(self._stat_total, str(n_unique))
        self._set_stat(self._stat_coverage, f"{cov*100:.0f}%")
        self._set_stat(self._stat_top, f"{summary_rows[0].count} mol" if summary_rows else "—")

        self._populate_gallery(summary_rows)

    @Slot(str)
    def _fail(self, msg: str) -> None:
        self._set_busy(False)
        set_widget_error(self, msg)
        self._set_status(format_error_status(msg), ok=False)
        self._send_empty()

    def _send_empty(self) -> None:
        self._last_annotated_data = None
        self.Outputs.annotated_data.send(None)
        self.Outputs.scaffold_summary.send(None)
        self.Outputs.annotated_molecules.send([])
        self.Outputs.selected.send(None)

    # ── selection ─────────────────────────────────────────────────────────────

    def _on_card_clicked(self, smiles: str) -> None:
        # toggle: clicking the selected card deselects it
        if self.selected_scaffold == smiles:
            self.selected_scaffold = ""
        else:
            self.selected_scaffold = smiles
        self._update_card_selection()
        self._send_selected()

    def _update_card_selection(self) -> None:
        for smi, card in self._cards.items():
            card.set_selected(smi == self.selected_scaffold)

    def _send_selected(self) -> None:
        if not self.selected_scaffold or self._last_annotated_data is None:
            self.Outputs.selected.send(None)
            return
        data = self._last_annotated_data
        murcko_var = next(
            (v for v in data.domain.metas if v.name == "Murcko Scaffold"), None
        )
        if murcko_var is None:
            self.Outputs.selected.send(None)
            return
        col = data.get_column(murcko_var)
        mask = np.array([str(v).strip() == self.selected_scaffold for v in col])
        indices = np.where(mask)[0]
        self.Outputs.selected.send(data[indices] if len(indices) else None)

    # ── gallery ───────────────────────────────────────────────────────────────

    def _populate_gallery(self, summary_rows) -> None:
        # clear existing cards and layout
        self._cards.clear()
        while self._gallery_layout.count():
            item = self._gallery_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not summary_rows:
            lbl = QLabel("No scaffolds found.")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#94A3B8; font-size:13px; padding:40px;")
            self._gallery_layout.addWidget(lbl)
            self._gallery_layout.addStretch(1)
            return

        # render cards in rows of _COLS
        for row_start in range(0, len(summary_rows), _COLS):
            row_widget = QWidget()
            row_widget.setStyleSheet("background:transparent;")
            row_hl = QHBoxLayout(row_widget)
            row_hl.setContentsMargins(0, 0, 0, 0)
            row_hl.setSpacing(10)
            for rank_0, sr in enumerate(summary_rows[row_start: row_start + _COLS], start=row_start + 1):
                card = _ScaffoldCard(rank_0, sr.scaffold, sr.count, sr.fraction)
                card.clicked.connect(self._on_card_clicked)
                self._cards[sr.scaffold] = card
                row_hl.addWidget(card)
            row_hl.addStretch(1)
            self._gallery_layout.addWidget(row_widget)

        self._gallery_layout.addStretch(1)
        self._update_card_selection()  # restore previous selection if scaffold still present

    # ── table builders ────────────────────────────────────────────────────────

    def _rows_to_table(self, summary_rows) -> Table:
        domain = Domain(
            [ContinuousVariable("Count"), ContinuousVariable("Fraction")],
            metas=[StringVariable("Scaffold"), StringVariable("Scaffold Kind")],
        )
        if not summary_rows:
            return Table.from_numpy(domain, X=np.zeros((0, 2)), metas=np.zeros((0, 2), dtype=object))
        X = np.array([[float(r.count), float(r.fraction)] for r in summary_rows], dtype=float)
        metas = np.array([[r.scaffold, r.kind] for r in summary_rows], dtype=object)
        return Table.from_numpy(domain, X=X, metas=metas)

    def _annotate_table(self, data: Table, result: ScaffoldAnalysisResult) -> Table:
        dom = data.domain
        metas = list(dom.metas)
        metas_out = metas + [
            StringVariable("Murcko Scaffold"),
            StringVariable("Generic Scaffold"),
            StringVariable("Scaffold Status"),
        ]
        dom_out = Domain(dom.attributes, dom.class_vars, metas=metas_out)
        M = data.metas.copy() if data.metas is not None and data.metas.size else np.zeros((len(data), len(metas)), dtype=object)
        M_out = np.zeros((len(data), len(metas_out)), dtype=object)
        if len(metas) > 0:
            M_out[:, :len(metas)] = M
        for i, ann in enumerate(result.annotations):
            M_out[i, len(metas)] = ann.murcko or ""
            M_out[i, len(metas) + 1] = ann.generic or ""
            M_out[i, len(metas) + 2] = ann.status
        return Table.from_numpy(dom_out, X=data.X, Y=data.Y, metas=M_out)

    def _annotate_molecules(self, result: ScaffoldAnalysisResult, molecules: List[ChemMol]) -> List[ChemMol]:
        if len(molecules) != len(result.annotations):
            return []
        annotated = []
        for mol, ann in zip(molecules, result.annotations):
            copied = ChemMol(mol=mol.to_rdkit() or mol.mol, name=mol.name,
                             props=dict(mol.props), cache=dict(mol.cache))
            copied.set_prop("Murcko Scaffold", ann.murcko or "")
            copied.set_prop("Generic Scaffold", ann.generic or "")
            copied.set_prop("Scaffold Status", ann.status)
            annotated.append(copied)
        return annotated

    def onDeleteWidget(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWScaffoldAnalysis).run()
