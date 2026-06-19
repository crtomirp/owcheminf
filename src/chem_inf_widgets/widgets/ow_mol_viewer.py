from __future__ import annotations

from typing import Optional, List
from collections import OrderedDict

from AnyQt.QtCore import Qt, QObject, QRunnable, QThreadPool, pyqtSignal
from AnyQt.QtGui import QPixmap, QImage
from AnyQt.QtWidgets import (
    QWidget, QFrame, QLabel,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QScrollArea, QSplitter, QSpinBox,
    QListWidget, QListWidgetItem, QLineEdit,
    QComboBox
)

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output

from rdkit import Chem
from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services import mol_depict
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_canonical_smiles


# ============================================================
# Utilities
# ============================================================

def pixmap_from_png(png: bytes) -> QPixmap:
    img = QImage.fromData(png)
    return QPixmap.fromImage(img)


def clear_layout(layout: QGridLayout) -> None:
    while layout.count():
        it = layout.takeAt(0)
        w = it.widget()
        if w:
            w.setParent(None)
            w.deleteLater()


def format_property(key: str, value):
    if value is None:
        return ""

    k = key.lower()

    if isinstance(value, (int, float)):
        if k in ("mw", "molecular weight"):
            return f"{value:.1f} g/mol"
        if k in ("logp",):
            return f"{value:.2f}"
        if k in ("tpsa",):
            return f"{value:.1f} Å²"
        if abs(value) < 1:
            return f"{value:.3f}"
        return f"{value:.2f}"

    if isinstance(value, bool):
        return "✓" if value else "✗"

    return str(value)


def display_smiles(mol: Chem.Mol) -> str:
    """
    Canonical SMILES for display (no explicit H).
    """
    if mol is None:
        return ""
    return safe_canonical_smiles(mol, remove_hs=True, canonical=True, isomeric=True)


# ============================================================
# LRU cache (RAM)
# ============================================================

class PixmapLRUCache:
    def __init__(self, max_items: int = 512):
        self.max_items = max_items
        self.cache: OrderedDict[tuple, QPixmap] = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, pix):
        self.cache[key] = pix
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)


# ============================================================
# Property selector
# ============================================================

class PropertySelector(QWidget):
    changed = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)

        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter properties…")
        self.filter.textChanged.connect(self._filter)
        lay.addWidget(self.filter)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.MultiSelection)
        self.list.itemSelectionChanged.connect(self._emit)
        lay.addWidget(self.list)

    def set_properties(self, names, selected):
        self.list.clear()
        selected = set(selected)
        for n in sorted(names):
            it = QListWidgetItem(n)
            if n in selected:
                it.setSelected(True)
            self.list.addItem(it)

    def _emit(self):
        self.changed.emit([i.text() for i in self.list.selectedItems()])

    def _filter(self, text):
        t = text.lower()
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setHidden(t not in it.text().lower())


# ============================================================
# Async rendering
# ============================================================

class RenderSignals(QObject):
    finished = pyqtSignal(int, int, QPixmap)


class RenderTask(QRunnable):
    """
    Render one molecule to pixmap (async). Cache key includes highlight atoms.
    """
    def __init__(
        self,
        idx: int,
        mol: Chem.Mol,
        size: int,
        generation: int,
        cache: PixmapLRUCache,
        highlight_atoms: Optional[List[int]] = None,
    ):
        super().__init__()
        self.idx = idx
        self.mol = mol
        self.size = size
        self.generation = generation
        self.cache = cache
        self.highlight_atoms = highlight_atoms or []
        self.signals = RenderSignals()

    def run(self):
        key = (id(self.mol), self.size, tuple(self.highlight_atoms))
        pix = self.cache.get(key)
        if pix is None:
            png = mol_depict.render_mol_png(
                self.mol,
                size=self.size,
                highlight_atoms=self.highlight_atoms,
            )
            pix = pixmap_from_png(png)
            self.cache.put(key, pix)

        self.signals.finished.emit(self.idx, self.generation, pix)


# ============================================================
# Card
# ============================================================

class MolCard(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, idx, item, properties=None, pinned=None):
        super().__init__()
        self.idx = idx
        self.item = item
        self.properties = properties or []
        self.pinned_property = pinned

        self.setCursor(Qt.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setSpacing(4)

        self.img = QLabel("⏳")
        self.img.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.img)

        title = QLabel(item.title or f"#{idx+1}")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: 600;")
        lay.addWidget(title)

        smiles = display_smiles(item.mol)
        self.smiles_label = QLabel(smiles)
        self.smiles_label.setWordWrap(True)
        self.smiles_label.setAlignment(Qt.AlignCenter)
        self.smiles_label.setStyleSheet("font-size: 9px; color: #555;")
        lay.addWidget(self.smiles_label)

        self.prop_box = QVBoxLayout()
        lay.addLayout(self.prop_box)
        self._render_properties()

    def set_properties(self, props):
        self.properties = props
        self._render_properties()

    def set_pinned_property(self, key):
        self.pinned_property = key
        self._render_properties()

    def _render_properties(self):
        while self.prop_box.count():
            it = self.prop_box.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

        for key in self.properties:
            if key not in self.item.props:
                continue

            val = self.item.props.get(key)
            txt = format_property(key, val)

            lab = QLabel(f"{key}: {txt}")
            lab.setAlignment(Qt.AlignLeft)

            if key == self.pinned_property:
                lab.setStyleSheet("font-size: 13px; font-weight: 700; color: #1565c0;")
            else:
                lab.setStyleSheet("font-size: 10px; color: #666;")

            self.prop_box.addWidget(lab)

    def mousePressEvent(self, e):
        self.clicked.emit(self.idx)


# ============================================================
# Gallery (virtualized)
# ============================================================

class VirtualMolGallery(QWidget):
    selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.items = []
        self.columns = 4
        self.batch_size = 20
        self.card_size = 250

        self.rendered = 0
        self.generation = 0

        self.visible_properties = []
        self.pinned_property = None

        self.cache = PixmapLRUCache()
        self.pool = QThreadPool.globalInstance()

        layout = QVBoxLayout(self)

        self.progress = QLabel("Loaded 0 / 0")
        layout.addWidget(self.progress)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scroll)
        layout.addWidget(self.scroll, 1)

        self.canvas = QWidget()
        self.grid = QGridLayout(self.canvas)
        self.grid.setSpacing(12)
        self.scroll.setWidget(self.canvas)

    def set_items(self, items):
        clear_layout(self.grid)
        self.items = items or []
        self.rendered = 0
        self.generation += 1
        self.progress.setText(f"Loaded 0 / {len(self.items)}")
        self._render_more()

    def set_visible_properties(self, props):
        self.visible_properties = props
        for i in range(self.grid.count()):
            w = self.grid.itemAt(i).widget()
            if isinstance(w, MolCard):
                w.set_properties(props)

    def set_pinned_property(self, key):
        self.pinned_property = key
        for i in range(self.grid.count()):
            w = self.grid.itemAt(i).widget()
            if isinstance(w, MolCard):
                w.set_pinned_property(key)

    def _on_scroll(self, v):
        if v > self.scroll.verticalScrollBar().maximum() - 200:
            self._render_more()

    def _render_more(self):
        start = self.rendered
        end = min(start + self.batch_size, len(self.items))

        for idx in range(start, end):
            item = self.items[idx]
            card = MolCard(
                idx,
                item,
                properties=self.visible_properties,
                pinned=self.pinned_property,
            )
            card.clicked.connect(self.selected.emit)
            r, c = divmod(idx, self.columns)
            self.grid.addWidget(card, r, c)

            task = RenderTask(
                idx=idx,
                mol=item.mol,
                size=self.card_size,
                generation=self.generation,
                cache=self.cache,
                highlight_atoms=list(getattr(item, "highlight_atoms", []) or []),
            )
            task.signals.finished.connect(self._on_render)
            self.pool.start(task)

        self.rendered = end
        self.progress.setText(f"Loaded {self.rendered} / {len(self.items)}")

    def _on_render(self, idx, gen, pix):
        if gen != self.generation:
            return
        card = self.grid.itemAtPosition(
            idx // self.columns, idx % self.columns
        ).widget()
        if card:
            card.img.setPixmap(pix)


# ============================================================
# Main widget
# ============================================================

class OWMolViewer(OWWidget):
    name = "Molecular Viewer"
    description = "Modern, scalable molecular viewer (supports substructure highlight)"
    icon = "icons/editors_viewers/owmolviewerwidget.png"
    priority = 105

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        selected_index = Output("Selected index", int, auto_summary=False)

    batch_size = Setting(20)
    columns = Setting(4)
    card_size = Setting(250)

    def __init__(self):
        super().__init__()

        self.items = []
        self.selected_properties = []
        self.pinned_property = None

        splitter = QSplitter(Qt.Horizontal)
        self.mainArea.layout().addWidget(splitter)

        self.gallery = VirtualMolGallery()
        self.gallery.selected.connect(self._on_select)
        splitter.addWidget(self.gallery)

        # ---------- CONTROL PANEL ----------
        box = gui.widgetBox(self.controlArea, "Layout")

        gui.spin(
            box, self, "batch_size", 5, 200,
            label="Molecules per batch",
            callback=self._apply_layout
        )
        gui.spin(
            box, self, "columns", 1, 10,
            label="Columns",
            callback=self._apply_layout
        )
        gui.spin(
            box, self, "card_size", 150, 420,
            label="Card size (px)",
            callback=self._apply_layout
        )

        gui.separator(self.controlArea)
        gui.label(self.controlArea, self, "Displayed properties")

        self.prop_selector = PropertySelector()
        self.prop_selector.changed.connect(self._props_changed)
        self.controlArea.layout().addWidget(self.prop_selector)

        gui.label(self.controlArea, self, "Pinned property")
        self.pin_combo = QComboBox()
        self.pin_combo.currentTextChanged.connect(self._pin_changed)
        self.controlArea.layout().addWidget(self.pin_combo)

        self._apply_layout()

    # ---------- Inputs ----------

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.items = mol_depict.table_to_items(data) if data else []
        self._refresh_properties()
        self.gallery.set_items(self.items)

    @Inputs.molecules
    def set_molecules(self, mols: Optional[List[ChemMol]]):
        self.items = mol_depict.chemmols_to_items(mols) if mols else []
        self._refresh_properties()
        self.gallery.set_items(self.items)

    # ---------- helpers ----------

    def _refresh_properties(self):
        keys = set()
        for it in self.items:
            keys.update(k for k in it.props.keys() if (k or "").strip().lower() != "smiles")
        keys = sorted(keys)

        self.prop_selector.set_properties(keys, self.selected_properties)

        self.pin_combo.clear()
        self.pin_combo.addItem("")
        self.pin_combo.addItems(keys)

    def _props_changed(self, props):
        self.selected_properties = props
        self.gallery.set_visible_properties(props)

    def _pin_changed(self, key):
        self.pinned_property = key or None
        self.gallery.set_pinned_property(self.pinned_property)

    def _apply_layout(self):
        self.gallery.batch_size = int(self.batch_size)
        self.gallery.columns = int(self.columns)
        self.gallery.card_size = int(self.card_size)
        self.gallery.set_items(self.items)

    def _on_select(self, idx):
        self.Outputs.selected_index.send(int(idx))


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWMolViewer).run()
