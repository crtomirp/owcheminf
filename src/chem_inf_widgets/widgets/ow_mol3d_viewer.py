from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional, Tuple

from AnyQt.QtCore import Qt, QSize, pyqtSlot as Slot
from AnyQt.QtGui import QIcon, QPixmap
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke
from Orange.widgets.widget import Input, OWWidget

from rdkit import Chem
from rdkit.Chem import Draw

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services import mol_depict
from chem_inf_widgets.chemcore.services.mol3d_viewer_service import (
    Viewer3DConfig,
    build_3d_html_from_mol,
)

# QtWebEngine optional
try:
    from AnyQt.QtWebEngineWidgets import QWebEngineView  # type: ignore

    _HAS_WEBENGINE = True
except Exception:
    QWebEngineView = None  # type: ignore
    _HAS_WEBENGINE = False


class OWMol3DViewer(OWWidget):
    name = "3D Molecular Viewer"
    description = "3D viewer with RDKit 3D generation + gallery (py3Dmol)."
    icon = "icons/editors_viewers/owmol3dviewerwidget.png"
    priority = 106
    want_main_area = True
    resizing_enabled = True

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)
        selected_index = Input("Selected index", int, auto_summary=False)

    # UI settings
    style = Setting("stick")
    surface = Setting(False)
    surface_opacity = Setting(0.6)
    add_hs = Setting(True)
    optimize = Setting(True)
    max_opt_iters = Setting(200)

    thumb_size = Setting(92)
    auto_render_on_input = Setting(True)

    width = Setting(820)
    height = Setting(620)

    def __init__(self) -> None:
        super().__init__()

        self._data: Optional[Table] = None
        self._mols: Optional[List[ChemMol]] = None
        self._sel: int = 0

        self._executor = ThreadExecutor(self)
        self._cache = {}  # (src_id, idx, cfg_key) -> html
        self._current_job = 0

        self._build_ui()

        if not _HAS_WEBENGINE:
            self._set_status(
                "QtWebEngine missing. Install: conda install -c conda-forge pyqtwebengine",
                ok=False,
            )
        else:
            self._set_status("Ready.", ok=True)

    # ---------- UI ----------

    def _build_ui(self) -> None:
        box = gui.widgetBox(self.controlArea, "3D View Settings")

        self.cmb_style = QComboBox()
        self.cmb_style.addItems(["stick", "sphere", "line"])
        self.cmb_style.setCurrentText(self.style)
        self.cmb_style.currentTextChanged.connect(self._on_settings_changed)
        gui.widgetLabel(box, "Style")
        box.layout().addWidget(self.cmb_style)

        self.chk_surface = QCheckBox("Surface (VDW)")
        self.chk_surface.setChecked(bool(self.surface))
        self.chk_surface.toggled.connect(self._on_settings_changed)
        box.layout().addWidget(self.chk_surface)

        row = gui.hBox(box)
        row.layout().addWidget(QLabel("Opacity"))
        self.spin_op = QDoubleSpinBox()
        self.spin_op.setRange(0.05, 1.0)
        self.spin_op.setSingleStep(0.05)
        self.spin_op.setDecimals(2)
        self.spin_op.setValue(float(self.surface_opacity))
        self.spin_op.valueChanged.connect(self._on_settings_changed)
        row.layout().addWidget(self.spin_op)

        self.chk_addhs = QCheckBox("Add H for 3D")
        self.chk_addhs.setChecked(bool(self.add_hs))
        self.chk_addhs.toggled.connect(self._on_settings_changed)
        box.layout().addWidget(self.chk_addhs)

        self.chk_opt = QCheckBox("Optimize (UFF)")
        self.chk_opt.setChecked(bool(self.optimize))
        self.chk_opt.toggled.connect(self._on_settings_changed)
        box.layout().addWidget(self.chk_opt)

        gui.separator(self.controlArea)

        self.chk_auto = QCheckBox("Auto render on input/selection")
        self.chk_auto.setChecked(bool(self.auto_render_on_input))
        self.chk_auto.toggled.connect(self._on_settings_changed)
        self.controlArea.layout().addWidget(self.chk_auto)

        self.lbl_status = QLabel("Ready")
        self.controlArea.layout().addWidget(self.lbl_status)

        btn = QPushButton("Reload 3D")
        btn.clicked.connect(self._render_selected_async)
        self.controlArea.layout().addWidget(btn)

        self.controlArea.layout().addStretch(1)

        # --- main area: gallery + 3D view
        self.mainArea.layout().setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal, self.mainArea)

        # left: gallery list
        left = QWidget(splitter)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        self.gallery = QListWidget(left)
        self.gallery.setSelectionMode(QAbstractItemView.SingleSelection)
        self.gallery.setIconSize(QSize(int(self.thumb_size), int(self.thumb_size)))
        self.gallery.currentRowChanged.connect(self._on_gallery_row_changed)
        lv.addWidget(QLabel("Gallery"))
        lv.addWidget(self.gallery, 1)

        # right: 3D view
        right = QWidget(splitter)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        rv.addWidget(QLabel("3D View"))
        if _HAS_WEBENGINE:
            self.web = QWebEngineView(right)  # type: ignore[call-arg]
            rv.addWidget(self.web, 1)
        else:
            self.web = None  # type: ignore[assignment]
            lab = QLabel("QtWebEngine not available.\nInstall: conda install -c conda-forge pyqtwebengine")
            lab.setAlignment(Qt.AlignCenter)
            rv.addWidget(lab, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 900])

        self.mainArea.layout().addWidget(splitter)

    def _set_status(self, msg: str, ok: bool) -> None:
        self.lbl_status.setText(msg)
        if ok:
            self.lbl_status.setStyleSheet(
                "padding:4px 8px;border:1px solid #e1e1e1;border-radius:10px;background:#fafafa;"
            )
        else:
            self.lbl_status.setStyleSheet(
                "padding:4px 8px;border:1px solid #f2c2c2;border-radius:10px;background:#fff5f5;color:#a40000;"
            )

    # ---------- Inputs ----------

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._data = data
        self._mols = None
        self._sel = 0
        self._cache.clear()
        self._rebuild_gallery()

        if self.auto_render_on_input:
            self._render_selected_async()

    @Inputs.molecules
    def set_molecules(self, mols: Optional[List[ChemMol]]) -> None:
        self._mols = mols
        self._data = None
        self._sel = 0
        self._cache.clear()
        self._rebuild_gallery()

        if self.auto_render_on_input:
            self._render_selected_async()

    @Inputs.selected_index
    def set_selected_index(self, idx: Optional[int]) -> None:
        if idx is None:
            return
        self._sel = max(0, int(idx))
        self._sync_gallery_selection()
        if self.auto_render_on_input:
            self._render_selected_async()

    # ---------- Gallery ----------

    def _sync_gallery_selection(self) -> None:
        if self.gallery.count() == 0:
            return
        row = min(max(0, self._sel), self.gallery.count() - 1)
        self.gallery.blockSignals(True)
        try:
            self.gallery.setCurrentRow(row)
        finally:
            self.gallery.blockSignals(False)

    def _on_gallery_row_changed(self, row: int) -> None:
        if row < 0:
            return
        self._sel = row
        if self.auto_render_on_input:
            self._render_selected_async()

    def _rebuild_gallery(self) -> None:
        self.gallery.clear()

        mols = self._get_all_mols_for_gallery()
        if not mols:
            self._set_status("No molecules.", ok=False)
            return

        for i, (title, mol) in enumerate(mols):
            icon = self._make_2d_icon(mol)
            it = QListWidgetItem(f"{i+1}. {title}")
            if icon is not None:
                it.setIcon(icon)
            self.gallery.addItem(it)

        self._sync_gallery_selection()
        self._set_status(f"Loaded {len(mols)} molecules.", ok=True)

    def _get_all_mols_for_gallery(self) -> List[Tuple[str, Chem.Mol]]:
        out: List[Tuple[str, Chem.Mol]] = []

        if self._mols is not None:
            for i, cm in enumerate(self._mols):
                if cm is None or cm.mol is None:
                    continue
                out.append((cm.name or f"mol_{i+1}", cm.mol))
            return out

        if self._data is not None and len(self._data) > 0:
            items = mol_depict.table_to_items(self._data)
            for it in items:
                if it.mol is None:
                    continue
                out.append((it.title or "mol", it.mol))
        return out

    def _make_2d_icon(self, mol: Chem.Mol) -> Optional[QIcon]:
        try:
            size = int(self.thumb_size)
            img = Draw.MolToImage(mol, size=(size, size))
            import io

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            pix = QPixmap()
            pix.loadFromData(buf.getvalue(), "PNG")
            return QIcon(pix)
        except Exception:
            return None

    # ---------- Rendering ----------

    def _on_settings_changed(self, *_args) -> None:
        self.style = self.cmb_style.currentText()
        self.surface = bool(self.chk_surface.isChecked())
        self.surface_opacity = float(self.spin_op.value())
        self.add_hs = bool(self.chk_addhs.isChecked())
        self.optimize = bool(self.chk_opt.isChecked())
        self.auto_render_on_input = bool(self.chk_auto.isChecked())
        self._cache.clear()
        if self.auto_render_on_input:
            self._render_selected_async()

    def _cfg(self) -> Viewer3DConfig:
        return Viewer3DConfig(
            width=int(self.width),
            height=int(self.height),
            style=str(self.style),
            surface=bool(self.surface),
            surface_opacity=float(self.surface_opacity),
            add_hs=bool(self.add_hs),
            optimize=bool(self.optimize),
            max_opt_iters=int(self.max_opt_iters),
        )

    def _get_selected_mol(self) -> Optional[Chem.Mol]:
        if self._mols is not None:
            if 0 <= self._sel < len(self._mols):
                cm = self._mols[self._sel]
                return None if (cm is None) else cm.mol
            return None

        if self._data is not None and len(self._data) > 0:
            i = min(max(0, self._sel), len(self._data) - 1)
            items = mol_depict.table_to_items(self._data)
            if 0 <= i < len(items):
                return items[i].mol
        return None

    def _render_selected_async(self) -> None:
        if not _HAS_WEBENGINE or self.web is None:
            return

        mol = self._get_selected_mol()
        if mol is None:
            self._set_status("No molecule selected / no input.", ok=False)
            self.web.setHtml("<html><body><h3>No molecule.</h3></body></html>")
            return

        cfg = self._cfg()
        cfg_key = tuple(sorted(asdict(cfg).items()))
        src_id = "table" if self._data is not None else "mols"
        cache_key = (src_id, self._sel, cfg_key)

        if cache_key in self._cache:
            self.web.setHtml(self._cache[cache_key])
            self._set_status("Rendered (cache).", ok=True)
            return

        self._set_status("Rendering 3D…", ok=True)
        self._current_job += 1
        job_id = self._current_job

        def _worker() -> str:
            return build_3d_html_from_mol(mol, cfg)

        fut = self._executor.submit(_worker)
        fut.add_done_callback(lambda f: self._on_done(f, cache_key, job_id))

    def _on_done(self, fut, cache_key, job_id: int) -> None:
        try:
            html_s = fut.result()
            methodinvoke(self, "_apply_html", (str, object, int))(html_s, cache_key, job_id)
        except Exception as e:
            methodinvoke(self, "_fail", (str, int))(str(e), job_id)

    @Slot(str, object, int)
    def _apply_html(self, html_s: str, cache_key: object, job_id: int) -> None:
        if job_id != self._current_job:
            return
        if self.web is None:
            return
        self._cache[cache_key] = html_s
        self.web.setHtml(html_s)
        self._set_status("Rendered.", ok=True)

    @Slot(str, int)
    def _fail(self, msg: str, job_id: int) -> None:
        if job_id != self._current_job:
            return
        self._set_status(f"Error: {msg}", ok=False)
        if self.web is not None:
            self.web.setHtml(f"<html><body><h3>Error</h3><pre>{msg}</pre></body></html>")

    def onDeleteWidget(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWMol3DViewer).run()
