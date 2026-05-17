from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

from AnyQt.QtCore import QUrl, Qt, QTimer
from AnyQt.QtWidgets import (
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from Orange.data import Table
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting

from chem_inf_widgets.chemcore.services.mol_sketcher_core import MolSketcherCore, RDKIT_AVAILABLE
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_canonical_smiles, safe_mol_from_smiles
from chem_inf_widgets.chemcore.services.from_orange import (
    TableMolConversionReport,
    chemmols_to_table,
    table_to_chemmols_with_report,
)
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_conversion_report,
    format_skipped_rows_warning,
    set_widget_error,
    set_widget_warning,
)

try:
    from chem_inf_widgets.chemcore.mol import ChemMol
except Exception:  # pragma: no cover
    ChemMol = object  # type: ignore

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


QWebEnginePage = None
QWebEngineView = None
QWebEngineSettings = None
_WEBENGINE_OK = False
_WEBENGINE_ERR = ""
logger = logging.getLogger(__name__)


def _ensure_webengine():
    global QWebEnginePage, QWebEngineView, QWebEngineSettings, _WEBENGINE_OK, _WEBENGINE_ERR
    if QWebEngineView is not None and QWebEnginePage is not None:
        return True, ""
    try:
        from AnyQt.QtWebEngineWidgets import QWebEnginePage as _QWebEnginePage, QWebEngineView as _QWebEngineView
        try:
            from AnyQt.QtWebEngineWidgets import QWebEngineSettings as _QWebEngineSettings
        except Exception:
            _QWebEngineSettings = None
        QWebEnginePage = _QWebEnginePage
        QWebEngineView = _QWebEngineView
        QWebEngineSettings = _QWebEngineSettings
        _WEBENGINE_OK = True
        _WEBENGINE_ERR = ""
        return True, ""
    except Exception as e:
        _WEBENGINE_OK = False
        _WEBENGINE_ERR = str(e)
        return False, _WEBENGINE_ERR


def _apply_webengine_settings(view, *, enable_local_storage: bool = False) -> None:
    if view is None or QWebEngineSettings is None:
        return
    try:
        settings = view.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        if enable_local_storage:
            settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        if sys.platform == "darwin":
            settings.setFontFamily(QWebEngineSettings.StandardFont, "Helvetica")
            settings.setFontFamily(QWebEngineSettings.SansSerifFont, "Helvetica")
            settings.setFontFamily(QWebEngineSettings.SerifFont, "Helvetica")
    except Exception:
        logger.warning("Could not configure Qt WebEngine settings for the molecule editor.", exc_info=True)


def _connect_render_process_terminated(view, handler) -> None:
    if view is None:
        return
    try:
        signal = getattr(view, "renderProcessTerminated", None)
        if signal is None:
            return
        signal.connect(handler)
    except Exception:
        logger.warning("Could not connect renderProcessTerminated handler for the molecule editor.", exc_info=True)


class OWMolKetcher(widget.OWWidget, openclass=True):
    name = "Mol Ketcher"
    description = "Ketcher-based molecular editor for database creation"
    icon = "icons/editors_viewers/ketcher.png"
    priority = 112
    keywords = ["chemistry", "molecule", "sketcher", "ketcher"]
    prefer_jsme_fallback_on_macos = True

    class Inputs:
        smiles = widget.Input("SMILES", str, auto_summary=False)
        molecule = widget.Input("Molecule", ChemMol, auto_summary=False)
        molecules = widget.Input("Molecules", list, auto_summary=False)
        data = widget.Input("Data", Table, auto_summary=False)

    class Outputs:
        data = widget.Output("Compounds", Table)
        molecules = widget.Output("Molecules", list, auto_summary=False)

    want_main_area = True
    resizing_enabled = True

    json_path = Setting("")
    export_molecules: bool = Setting(True)

    # Editor mode state
    _editor_index: int = Setting(0)

    def __init__(self) -> None:
        super().__init__()
        self.core = MolSketcherCore()

        self._editor_molecules: List[ChemMol] = []
        self._editor_mode: bool = False
        
        self._original_smiles_col_name: Optional[str] = None
        self._meta_widgets: Dict[str, object] = {}
        self._metadata_values: Dict[str, str] = {}

        self._pending_smiles: str = ""
        self._pending_molblock: str = ""
        self._ketcher_try_count: int = 0
        self._ketcher_try_max: int = 50
        self._editor_backend: str = "ketcher"
        self._prefer_jsme_fallback: bool = (
            sys.platform == "darwin" and self.prefer_jsme_fallback_on_macos
        )
        
        # Stanje za določanje, kaj narediti s prejetim SMILES-om
        self._pending_action: Optional[str] = None  # "UPDATE" ali "ADD"
        self.web_view = None
        self.web_placeholder: Optional[QWidget] = None

        self._setup_ui()
        self._setup_web_placeholder()
        if self._prefer_jsme_fallback:
            self._editor_backend = "jsme"
            self._setup_jsme_fallback_view()
            self._set_info("Using JSME fallback editor on macOS.", ok=True)
        else:
            self._initialize_ketcher_on_demand()

        if self.json_path and os.path.exists(self.json_path):
            self._load_config(self.json_path)

        if not RDKIT_AVAILABLE:
            set_widget_warning(self, "RDKit not installed. Chemical properties unavailable.")

    # -------------------- Inputs --------------------

    @Inputs.smiles
    def set_smiles(self, smiles: Optional[str]) -> None:
        if not smiles: return
        self._editor_mode = False
        self._editor_molecules = []
        self._editor_index = 0
        self._original_smiles_col_name = None
        self._update_buttons_state()
        self._load_smiles_into_editor(str(smiles))
        self._show_properties(None)
        self._update_nav_ui()

    @Inputs.molecule
    def set_molecule(self, mol: Optional[ChemMol]) -> None:
        if mol is None: return
        self._original_smiles_col_name = None
        self._set_editor_molecules([mol])

    @Inputs.molecules
    def set_molecules(self, mols: Optional[Sequence[ChemMol]]) -> None:
        if not mols: return
        clean: List[ChemMol] = [m for m in mols if isinstance(m, ChemMol)]
        if not clean: return
        self._original_smiles_col_name = None
        self._set_editor_molecules(clean)

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        clear_widget_messages(self)
        if data is None: return
        if not RDKIT_AVAILABLE or Chem is None:
            set_widget_error(self, "RDKit knjižnica je potrebna za uvoz molekul.")
            return
        try:
            new_mols, report = table_to_chemmols_with_report(data)
        except ValueError as exc:
            set_widget_error(self, str(exc))
            return
        except Exception as exc:
            set_widget_error(self, f"Napaka pri uvozu molekul: {exc}")
            return

        self._original_smiles_col_name = report.smiles_column

        if new_mols:
            self._set_editor_molecules(new_mols)
            self._set_info(self._format_conversion_report(report), ok=True)
            if report.n_invalid > 0:
                set_widget_warning(self, self._format_conversion_warning(report))
        else:
            if report.smiles_column:
                set_widget_warning(
                    self,
                    f"Stolpec '{report.smiles_column}' ne vsebuje veljavnih molekul."
                )
            else:
                set_widget_warning(self, "Ne najdem stolpca s SMILES zapisi.")

    def _format_conversion_report(self, report: TableMolConversionReport) -> str:
        return format_conversion_report(
            report,
            prefix="Naloženo",
            item_label="molekul",
            column_label="SMILES stolpec",
        )

    def _format_conversion_warning(self, report: TableMolConversionReport) -> str:
        return format_skipped_rows_warning(
            report,
            prefix="Preskočenih vrstic",
            row_label="Vrstice",
        ) or ""

    # -------------------- UI setup --------------------

    def _find_ketcher_html(self) -> Optional[str]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "ketcher", "standalone", "indexqt.html"),
            os.path.join(base_dir, "ketcher", "standalone", "index.html"),
            os.path.join(base_dir, "ketcher", "index.html"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _setup_web_view(self) -> None:
        ok, err = _ensure_webengine()
        if not ok:
            set_widget_error(self, f"Qt WebEngine not available: {err}")
            return

        if self.web_view is not None:
            return

        if self.mainArea.layout() is None:
            self.mainArea.setLayout(QVBoxLayout())
            self.mainArea.layout().setContentsMargins(0,0,0,0)

        html_path = self._find_ketcher_html()
        if not html_path:
            set_widget_error(self, "Ketcher HTML datoteka ni najdena.")
            return

        if self.web_placeholder is not None:
            self.web_placeholder.hide()

        self.web_view = QWebEngineView(self.mainArea)
        
        class _KetcherWebPage(QWebEnginePage):
            def __init__(self, parent_widget):
                super().__init__(parent_widget)
                self.parent_widget = parent_widget

            def javaScriptConsoleMessage(self, level, message, line_number, source_id):
                if message.startswith("KETCHER_SMILES:"):
                    smiles = message[len("KETCHER_SMILES:"):]
                    self.parent_widget.receive_smiles_from_js(smiles)
                elif message.startswith("Ketcher instance found"):
                    self.parent_widget._set_info("Ketcher initialized.", ok=True)
                elif "Ketcher instance not found" in message:
                    self.parent_widget._set_info("Ketcher instance not found in page.", ok=False)

        self.web_view.setPage(_KetcherWebPage(self))

        _apply_webengine_settings(self.web_view, enable_local_storage=True)

        self.web_view.loadFinished.connect(self._on_load_finished)
        _connect_render_process_terminated(self.web_view, self._on_render_process_terminated)
        base_dir = os.path.dirname(os.path.abspath(html_path))
        base_url = QUrl.fromLocalFile(base_dir + os.sep)
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        self.web_view.setHtml(html, base_url)
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mainArea.layout().addWidget(self.web_view)

    def _setup_jsme_fallback_view(self) -> None:
        ok, err = _ensure_webengine()
        if not ok:
            set_widget_error(self, f"Qt WebEngine not available: {err}")
            return

        html_path = self.core.get_jsme_html_path()
        if not os.path.exists(html_path):
            set_widget_error(self, f"JSME HTML not found: {html_path}")
            return

        if self.web_placeholder is not None:
            self.web_placeholder.hide()

        self.web_view = QWebEngineView(self.mainArea)
        self.web_view.setPage(QWebEnginePage(self.web_view))

        _apply_webengine_settings(self.web_view, enable_local_storage=False)

        self.web_view.loadFinished.connect(self._on_jsme_load_finished)
        base_dir = os.path.dirname(os.path.abspath(html_path))
        base_url = QUrl.fromLocalFile(base_dir + os.sep)
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        self.web_view.setHtml(html, base_url)
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mainArea.layout().addWidget(self.web_view)

    def _replace_web_view(self) -> None:
        if self.web_view is not None:
            try:
                self.mainArea.layout().removeWidget(self.web_view)
            except Exception:
                logger.debug("Could not detach existing web view from the molecule editor layout.", exc_info=True)
            self.web_view.deleteLater()
            self.web_view = None

    def _activate_jsme_fallback(self, reason: str) -> None:
        if self._editor_backend == "jsme":
            return
        self._editor_backend = "jsme"
        self._replace_web_view()
        self._setup_jsme_fallback_view()
        self._set_info(f"{reason} Switched to embedded JSME fallback editor.", ok=False)
        if self._editor_mode and self._editor_molecules:
            self._load_current_editor_item()
        elif self._pending_smiles:
            self._load_smiles_into_editor(self._pending_smiles)

    def _setup_web_placeholder(self) -> None:
        if self.mainArea.layout() is None:
            self.mainArea.setLayout(QVBoxLayout())
            self.mainArea.layout().setContentsMargins(0, 0, 0, 0)

        self.web_placeholder = QWidget(self.mainArea)
        lay = QVBoxLayout(self.web_placeholder)
        lay.setContentsMargins(16, 16, 16, 16)
        msg = QLabel(
            "Ketcher editor could not be initialized automatically.\n"
            "Use the button below to retry loading the embedded editor."
        )
        msg.setWordWrap(True)
        lay.addWidget(msg)
        btn = QPushButton("Load Ketcher Editor")
        btn.clicked.connect(self._initialize_ketcher_on_demand)
        lay.addWidget(btn)
        lay.addStretch(1)
        self.mainArea.layout().addWidget(self.web_placeholder)

    def _initialize_ketcher_on_demand(self) -> None:
        if self.web_view is not None:
            return
        if self._prefer_jsme_fallback:
            self._editor_backend = "jsme"
            self._setup_jsme_fallback_view()
            self._set_info("Using JSME fallback editor on macOS.", ok=True)
            return
        try:
            self._setup_web_view()
            self._set_info("Initializing Ketcher…", ok=True)
        except Exception as e:
            set_widget_error(self, str(e))
            self._set_info(f"Ketcher initialization failed: {e}", ok=False)

    # -------------------- KETCHER JS helpers --------------------

    def _web_view_available(self) -> bool:
        return bool(self.web_view is not None)

    def _load_smiles_into_editor(self, smiles: str) -> None:
        if not self._web_view_available():
            return
        if self._editor_backend == "jsme":
            self._load_smiles_into_jsme(smiles)
            return
        smi = (smiles or "").strip()
        self._pending_smiles = smi
        self._try_flush_pending_smiles()

    def _on_load_finished(self, ok: bool) -> None:
        self._ketcher_try_count = 0
        if ok:
            self._set_info("Ketcher page loaded. Waiting for editor…", ok=True)
            self._try_flush_pending_smiles()
        else:
            self._activate_jsme_fallback("Ketcher page load failed.")

    def _on_render_process_terminated(self, termination_status, exit_code) -> None:
        self._activate_jsme_fallback(
            f"Ketcher render process terminated ({termination_status}, exit code {exit_code})."
        )

    def _try_flush_pending_smiles(self) -> None:
        if not self._web_view_available(): return
        
        js_check = "!!(window.ketcher && window.ketcher.setMolecule);"

        def _after_check(ready: bool) -> None:
            if bool(ready):
                smi = self._pending_smiles
                if not smi:
                     js_set = "ketcher.setMolecule('');"
                else:
                    smi_esc = json.dumps(smi)
                    js_set = f"ketcher.setMolecule({smi_esc});"

                self.web_view.page().runJavaScript(js_set)
                self._pending_smiles = "" 
                return

            self._ketcher_try_count += 1
            if self._ketcher_try_count <= self._ketcher_try_max:
                QTimer.singleShot(100, self._try_flush_pending_smiles)
            else:
                self._activate_jsme_fallback("Ketcher did not initialize in time.")

        self.web_view.page().runJavaScript(js_check, _after_check)

    def _load_smiles_into_jsme(self, smiles: str) -> None:
        if not self._web_view_available():
            return

        smi = (smiles or "").strip()
        self._pending_smiles = smi
        self._pending_molblock = ""

        if smi and Chem:
            try:
                m = safe_mol_from_smiles(smi, sanitize=True, remove_hs=True).mol
                if m:
                    m = Chem.AddHs(m)
                    AllChem.Compute2DCoords(m)
                    m = Chem.RemoveHs(m)
                    self._pending_molblock = Chem.MolToMolBlock(m)
            except Exception:
                self._pending_molblock = ""

        self._try_flush_pending_jsme()

    def _on_jsme_load_finished(self, ok: bool) -> None:
        if ok:
            self._set_info("JSME fallback editor ready.", ok=True)
            self._try_flush_pending_jsme()
        else:
            self._set_info("JSME fallback HTML load failed.", ok=False)

    def _try_flush_pending_jsme(self) -> None:
        if not self._web_view_available():
            return

        smi = (self._pending_smiles or "").strip()
        molblock = self._pending_molblock or ""
        js_check = """
        (function() {
          try {
            return !!(window && window.jsmeApplet);
          } catch(e) { return false; }
        })();
        """

        def _after_check(ready: bool) -> None:
            if not bool(ready):
                QTimer.singleShot(100, self._try_flush_pending_jsme)
                return

            smi_p = json.dumps(smi)
            mol_p = json.dumps(molblock)
            js_set = f"""
            (function(){{
              try {{
                var a = window.jsmeApplet;
                if (!a) return false;
                if (typeof a.reset === 'function') a.reset();
                if ({smi_p}) {{
                  if (typeof a.readSmiles === 'function') {{
                    a.readSmiles({smi_p});
                    return true;
                  }}
                  if (typeof a.readSMILES === 'function') {{
                    a.readSMILES({smi_p});
                    return true;
                  }}
                }}
                if ({mol_p} && typeof a.readMolFile === 'function') {{
                  a.readMolFile({mol_p});
                  return true;
                }}
                return true;
              }} catch (e) {{ return false; }}
            }})();
            """
            self.web_view.page().runJavaScript(js_set)
            self._pending_smiles = ""
            self._pending_molblock = ""

        self.web_view.page().runJavaScript(js_check, _after_check)

    # -------------------- Data Retrieval (NEW LOGIC) --------------------

    def _trigger_smiles_retrieval(self, action: str) -> None:
        """Sproži pridobivanje SMILES iz Ketcherja."""
        if not self._web_view_available():
            return
        
        self._pending_action = action # Zapomnimo si, kaj želimo narediti (UPDATE ali ADD)

        if self._editor_backend == "jsme":
            callback = self._apply_updated_smiles if action == "UPDATE" else self._handle_add_from_smiles
            self.web_view.page().runJavaScript("window.jsmeApplet ? window.jsmeApplet.smiles() : ''", callback)
            return
        
        # JS koda: pokliče getSmiles(), počaka na Promise in izpiše rezultat v konzolo.
        # Python (_KetcherWebPage) bo to prestregel.
        js = """
        (function() {
            if (window.ketcher && window.ketcher.getSmiles) {
                window.ketcher.getSmiles()
                .then(function(res) { console.log("KETCHER_SMILES:" + res); })
                .catch(function(err) { console.log("KETCHER_ERROR:" + err); });
            } else {
                console.log("KETCHER_SMILES:"); // Empty fallback
            }
        })();
        """
        self.web_view.page().runJavaScript(js)

    def receive_smiles_from_js(self, smiles: str) -> None:
        """To metodo pokliče _KetcherWebPage, ko prejme konzolno sporočilo."""
        if self._pending_action == "UPDATE":
            self._apply_updated_smiles(smiles)
        elif self._pending_action == "ADD":
            self._handle_add_from_smiles(smiles)
        
        self._pending_action = None

    # -------------------- UI setup (Buttons & Layout) --------------------

    def _setup_ui(self) -> None:
        box = gui.widgetBox(self.controlArea, "Configuration", orientation=Qt.Vertical)
        gui.button(box, self, "Load Config…", callback=self._load_config_dialog)
        self.config_label = QLabel("No configuration loaded")
        self.config_label.setStyleSheet("color:#666;margin-top:5px;")
        box.layout().addWidget(self.config_label)

        self.editor_box = gui.widgetBox(self.controlArea, "Editor", orientation=Qt.Vertical)
        row = QWidget(self.editor_box)
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(6)

        self.prev_btn = QPushButton("←")
        self.next_btn = QPushButton("→")
        self.idx_label = QLabel("-")
        self.update_btn = QPushButton("Posodobi")

        row_lay.addWidget(self.prev_btn)
        row_lay.addWidget(self.next_btn)
        row_lay.addWidget(self.idx_label)
        row_lay.addStretch(1)
        row_lay.addWidget(self.update_btn)
        self.editor_box.layout().addWidget(row)

        self.prev_btn.clicked.connect(lambda: self._step_editor(-1))
        self.next_btn.clicked.connect(lambda: self._step_editor(+1))
        self.update_btn.clicked.connect(self._update_current_structure_from_editor)

        self.props_box = gui.widgetBox(self.controlArea, "Properties", orientation=Qt.Vertical)
        self.props_scroll = QScrollArea(self.props_box)
        self.props_scroll.setWidgetResizable(True)
        self.props_scroll.setMinimumHeight(160)
        self.props_container = QWidget(self.props_scroll)
        self.props_layout = QVBoxLayout(self.props_container)
        self.props_scroll.setWidget(self.props_container)
        self.props_box.layout().addWidget(self.props_scroll)

        self.metadata_box = gui.widgetBox(self.controlArea, "Sample Metadata")
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color:#666;margin-top:8px;")
        self.controlArea.layout().addWidget(self.info_label)

        act = gui.widgetBox(self.controlArea, "Actions", orientation=Qt.Vertical)
        self.add_btn = gui.button(act, self, "Add Compound", callback=self._add_compound)
        self.delete_btn = gui.button(act, self, "Delete Compound", callback=self._delete_compound)
        self.clear_btn = gui.button(act, self, "Delete All", callback=self._delete_all)
        self.export_molecules_cb = gui.checkBox(act, self, "export_molecules", "Export Molecules output")
        
        self._update_buttons_state()

    def _update_buttons_state(self):
        has_items = bool(self._editor_molecules) if self._editor_mode else bool(getattr(self.core, 'rows', []))
        self.delete_btn.setEnabled(has_items)
        self.clear_btn.setEnabled(has_items)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.update_btn.setEnabled(False)
        if self._editor_mode and has_items:
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.update_btn.setEnabled(True)

    # -------------------- Config & Metadata --------------------

    def _load_config_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if path:
            self.json_path = path
            self._load_config(path)

    def _load_config(self, path: str) -> None:
        try:
            cfg = self.core.load_config(path)
            self.config_label.setText(f"Loaded: {os.path.basename(path)}")
            self._clear_metadata_inputs()
            self._metadata_values.clear()
            for meta in cfg.user_metadata:
                setattr(self, meta.name, "")
                w = gui.lineEdit(
                    self.metadata_box, self, meta.name,
                    label=f"{meta.label}:", tooltip=meta.description or "",
                    callback=self._on_metadata_changed,
                )
                self._meta_widgets[meta.name] = w
            self._set_info("Configuration loaded.", ok=True)
        except Exception as e:
            self._set_info(f"Invalid configuration: {e}", ok=False)

    def _clear_metadata_inputs(self) -> None:
        while self.metadata_box.layout().count():
            item = self.metadata_box.layout().takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._meta_widgets.clear()

    def _on_metadata_changed(self) -> None:
        if self.core.config is None: return
        for meta in self.core.config.user_metadata:
            self._metadata_values[meta.name] = str(getattr(self, meta.name, ""))

    # -------------------- Logic: Editor Mode --------------------

    def _set_editor_molecules(self, mols: Sequence[ChemMol]) -> None:
        self._editor_molecules = list(mols)
        self._editor_mode = True
        self._editor_index = max(0, min(int(self._editor_index), len(self._editor_molecules) - 1))
        self._update_buttons_state()
        self._load_current_editor_item()
        self._update_outputs()

    def _load_current_editor_item(self) -> None:
        if not self._editor_molecules:
            self._load_smiles_into_editor("")
            self._show_properties(None)
            self._update_nav_ui()
            return
        cm = self._editor_molecules[self._editor_index]
        smi = self._pick_smiles_from_chemmol(cm)
        self._load_smiles_into_editor(smi)
        self._show_properties(cm)
        self._update_nav_ui()

    @staticmethod
    def _smiles_from_props(props: object) -> str:
        if not isinstance(props, dict):
            return ""
        for key in ("SMILES", "smiles", "Canonical SMILES"):
            value = props.get(key)
            if value:
                return str(value)
        return ""

    @staticmethod
    def _canonical_smiles_from_mol(cm: ChemMol) -> str:
        mol = getattr(cm, "mol", None)
        if mol is None or Chem is None:
            return ""
        try:
            return safe_canonical_smiles(mol, remove_hs=False, canonical=True, isomeric=True)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            logger.debug("Failed to canonicalize molecule for Ketcher editor.", exc_info=True)
            return ""

    def _pick_smiles_from_chemmol(self, cm: ChemMol) -> str:
        if cm is None:
            return ""
        if self._original_smiles_col_name:
            props = getattr(cm, "props", None)
            if isinstance(props, dict):
                value = props.get(self._original_smiles_col_name)
                if value:
                    return str(value)

        smiles = self._canonical_smiles_from_mol(cm)
        if smiles:
            return smiles

        return self._smiles_from_props(getattr(cm, "props", {}))

    def _update_nav_ui(self) -> None:
        n = len(self._editor_molecules)
        if n <= 0:
            self.idx_label.setText("-")
            self._update_buttons_state()
            return
        self.idx_label.setText(f"{self._editor_index + 1} / {n}")
        self._update_buttons_state()
        self.prev_btn.setEnabled(n > 1 and self._editor_index > 0)
        self.next_btn.setEnabled(n > 1 and self._editor_index < n - 1)

    def _step_editor(self, delta: int) -> None:
        if not self._editor_molecules: return
        self._editor_index = max(0, min(self._editor_index + int(delta), len(self._editor_molecules) - 1))
        self._load_current_editor_item()

    def _show_properties(self, cm: Optional[ChemMol]) -> None:
        while self.props_layout.count():
            item = self.props_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        if cm is None:
            self.props_layout.addWidget(QLabel("(no molecule)"))
            self.props_layout.addStretch(1)
            return

        name = getattr(cm, "name", None) or "(unnamed)"
        lbl = QLabel(name)
        lbl.setStyleSheet("font-weight:600;")
        self.props_layout.addWidget(lbl)

        props = dict(getattr(cm, "props", None) or {})
        for k in sorted(props.keys()):
            self.props_layout.addWidget(QLabel(f"{k}: {props[k]}"))
        self.props_layout.addStretch(1)

    # -------------------- Update & Add Handlers --------------------

    def _update_current_structure_from_editor(self) -> None:
        """Kliče se ob pritisku na 'Posodobi'."""
        self._trigger_smiles_retrieval("UPDATE")

    def _add_compound(self) -> None:
        """Kliče se ob pritisku na 'Add Compound'."""
        self._trigger_smiles_retrieval("ADD")

    # -------------------- Core Logic --------------------

    def _apply_updated_smiles(self, smiles: str) -> None:
        smi = (smiles or "").strip()
        if not smi or not Chem: return
        mol = safe_mol_from_smiles(smi, sanitize=True, remove_hs=True).mol
        if not mol:
            self._set_info("Invalid SMILES from Ketcher.", ok=False)
            return
        
        canon = safe_canonical_smiles(mol, remove_hs=False, canonical=True, isomeric=True)
        cm = self._editor_molecules[self._editor_index]
        cm.mol = mol
        
        if not hasattr(cm, "props"): cm.props = {}
        if self._original_smiles_col_name:
            cm.props[self._original_smiles_col_name] = canon
            if self._original_smiles_col_name != "SMILES":
                 cm.props["SMILES"] = canon
        else:
            cm.props["SMILES"] = canon

        self._show_properties(cm)
        self._update_outputs()
        self._set_info(f"✓ Updated: {canon[:20]}...", ok=True)

    def _handle_add_from_smiles(self, smiles: str) -> None:
        smi = (smiles or "").strip()
        if not smi: return
        
        if self._editor_mode:
            if Chem:
                mol = safe_mol_from_smiles(smi, sanitize=True, remove_hs=True).mol
                if mol:
                    canon = safe_canonical_smiles(mol, remove_hs=False, canonical=True, isomeric=True)
                    cm = ChemMol.from_smiles(canon, name=f"New {len(self._editor_molecules)+1}")
                    cm.props["SMILES"] = canon
                    for k,v in self._metadata_values.items():
                        if v: cm.props[k] = v
                    self._editor_molecules.append(cm)
                    self._editor_index = len(self._editor_molecules) - 1
                    self._load_current_editor_item()
                    self._update_outputs()
        else:
            try:
                self.core.add_compound(smi, self._metadata_values)
                self._update_outputs()
                self._update_buttons_state()
            except Exception as e:
                self._set_info(str(e), ok=False)

    def _delete_compound(self) -> None:
        if self._editor_mode:
            if not self._editor_molecules: return
            self._editor_molecules.pop(self._editor_index)
            if not self._editor_molecules:
                self._editor_index = 0
                self._load_smiles_into_editor("")
                self._show_properties(None)
            else:
                self._editor_index = max(0, min(self._editor_index, len(self._editor_molecules)-1))
                self._load_current_editor_item()
            self._update_outputs()
        else:
            if getattr(self.core, "rows", None):
                self.core.rows.pop()
                self._update_outputs()
        self._update_buttons_state()

    def _delete_all(self) -> None:
        if self._editor_mode:
            self._editor_molecules = []
            self._editor_index = 0
            self._load_smiles_into_editor("")
            self._show_properties(None)
            self._update_nav_ui()
        else:
            self.core.clear()
        self._update_outputs()
        self._update_buttons_state()

    def _update_outputs(self) -> None:
        if self._editor_mode:
            table = chemmols_to_table(self._editor_molecules)
            self.Outputs.data.send(table)
            self.Outputs.molecules.send(list(self._editor_molecules) if bool(self.export_molecules) else [])
        else:
            table = self.core.build_table()
            self.Outputs.data.send(table)
            self.Outputs.molecules.send(self.core.build_molecules() if bool(self.export_molecules) else [])

    def _set_info(self, text: str, ok: bool) -> None:
        self.info_label.setText(text)
        self.info_label.setStyleSheet("color:#4CAF50;" if ok else "color:#f44336;")

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWMolKetcher).run()
