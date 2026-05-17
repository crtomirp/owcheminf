"""
Chem-Inf Widgets — centralised QSS theme.

DESIGN RULE: on macOS Orange uses the native Aqua style for most controls
(QPushButton, QCheckBox, QLineEdit, QComboBox, QSpinBox …).  Applying QSS
to those replaces crisp native rendering with flat Qt painting — which looks
worse.  We therefore style ONLY elements that Qt draws itself regardless of
platform: QGroupBox, QListWidget, QTreeWidget, QTableWidget, QHeaderView,
QTabBar, QScrollBar, QProgressBar, QMenu, QToolTip, QSplitter, QLabel.

Call apply_theme() once per process (done from __init__.py).  Idempotent.
"""

from __future__ import annotations

# ── Colour tokens ────────────────────────────────────────────────────────────

# Backgrounds
BG         = "#FFFFFF"
BG_SURFACE = "#F8FAFC"
BG_MUTED   = "#F1F5F9"
BG_DEEP    = "#E2E8F0"

# Text
TX         = "#0F172A"
TX_SEC     = "#475569"
TX_MUT     = "#94A3B8"

# Primary (blue)
PR         = "#2563EB"
PR_DARK    = "#1D4ED8"
PR_DEEP    = "#1E3A8A"
PR_LIGHT   = "#DBEAFE"
PR_MIST    = "#EFF6FF"

# Accent for gradients
AC         = "#38BDF8"

# Borders
BD         = "#CBD5E1"
BD_LIGHT   = "#E2E8F0"

# Semantic
OK = "#16A34A"; OK_BG = "#F0FDF4"; OK_BD = "#BBF7D0"
WA = "#D97706"; WA_BG = "#FFFBEB"; WA_BD = "#FDE68A"
ER = "#DC2626"; ER_BG = "#FEF2F2"; ER_BD = "#FECACA"

# Selection
SEL_BG  = PR_LIGHT
SEL_TX  = PR_DEEP
SEL_HL  = "#BFDBFE"


_QSS = f"""

/* ================================================================
   Chem-Inf Widgets — platform-safe QSS
   Targets Qt-drawn elements only; native controls are untouched.
   ================================================================ */


/* ── QGroupBox ──────────────────────────────────────────────────── */

QGroupBox {{
    background: {BG};
    border: 1px solid {BD_LIGHT};
    border-radius: 8px;
    margin-top: 20px;
    padding: 10px 6px 6px 6px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 0;
    padding: 1px 7px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    color: {TX_MUT};
    background: {BG};
    border-radius: 3px;
}}


/* ── QListWidget ────────────────────────────────────────────────── */

QListWidget {{
    background: {BG};
    border: 1px solid {BD_LIGHT};
    border-radius: 7px;
    outline: none;
    padding: 3px;
}}

QListWidget::item {{
    padding: 5px 9px;
    border-radius: 5px;
    color: {TX};
    min-height: 22px;
}}

QListWidget::item:hover:!selected {{
    background: {BG_SURFACE};
}}

QListWidget::item:selected:active {{
    background: {SEL_BG};
    color: {SEL_TX};
    border-left: 2px solid {PR};
    padding-left: 7px;
}}

QListWidget::item:selected:!active {{
    background: {BG_MUTED};
    color: {TX_SEC};
}}


/* ── QTreeWidget ────────────────────────────────────────────────── */

QTreeWidget {{
    background: {BG};
    border: 1px solid {BD_LIGHT};
    border-radius: 7px;
    outline: none;
    padding: 2px;
    alternate-background-color: {BG_SURFACE};
}}

QTreeWidget::item {{
    padding: 4px 6px;
    color: {TX};
    min-height: 22px;
}}

QTreeWidget::item:selected:active {{
    background: {SEL_BG};
    color: {SEL_TX};
}}

QTreeWidget::item:selected:!active {{
    background: {BG_MUTED};
    color: {TX_SEC};
}}

QTreeWidget::item:hover:!selected {{
    background: {BG_SURFACE};
}}


/* ── QTableWidget + QHeaderView ─────────────────────────────────── */

QTableWidget {{
    background: {BG};
    border: 1px solid {BD_LIGHT};
    border-radius: 7px;
    gridline-color: {BD_LIGHT};
    outline: none;
    alternate-background-color: {BG_SURFACE};
}}

QTableWidget::item {{
    padding: 4px 10px;
    color: {TX};
    border: none;
}}

QTableWidget::item:selected {{
    background: {SEL_BG};
    color: {SEL_TX};
}}

QTableWidget::item:hover:!selected {{
    background: {BG_SURFACE};
}}

QHeaderView {{
    background: {BG_MUTED};
    border: none;
    outline: none;
}}

QHeaderView::section {{
    background: {BG_MUTED};
    color: {TX_SEC};
    padding: 6px 12px;
    border: none;
    border-bottom: 1.5px solid {BD};
    border-right: 1px solid {BD_LIGHT};
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}}

QHeaderView::section:last {{
    border-right: none;
}}

QHeaderView::section:hover {{
    background: {BG_DEEP};
    color: {TX};
}}

QHeaderView::section:checked {{
    background: {SEL_BG};
    color: {SEL_TX};
}}


/* ── QTabWidget / QTabBar ───────────────────────────────────────── */

QTabWidget::pane {{
    border: 1px solid {BD_LIGHT};
    border-top: none;
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
    background: {BG};
    padding: 4px;
}}

QTabBar::tab {{
    background: transparent;
    color: {TX_SEC};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 18px;
    margin-right: 2px;
    font-size: 13px;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    color: {PR};
    border-bottom: 2.5px solid {PR};
    font-weight: 700;
}}

QTabBar::tab:hover:!selected {{
    color: {TX};
    background: {BG_SURFACE};
    border-radius: 6px 6px 0 0;
}}

QTabBar::tab:disabled {{
    color: {TX_MUT};
}}


/* ── QScrollBar ─────────────────────────────────────────────────── */

QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
    border: none;
}}

QScrollBar::handle:vertical {{
    background: {BD};
    border-radius: 4px;
    min-height: 28px;
    margin: 0 1px;
}}

QScrollBar::handle:vertical:hover  {{ background: {TX_MUT}; }}
QScrollBar::handle:vertical:pressed {{ background: {TX_SEC}; }}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,  QScrollBar::sub-page:vertical {{
    background: none; height: 0; width: 0;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 0;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background: {BD};
    border-radius: 4px;
    min-width: 28px;
    margin: 1px 0;
}}

QScrollBar::handle:horizontal:hover  {{ background: {TX_MUT}; }}
QScrollBar::handle:horizontal:pressed {{ background: {TX_SEC}; }}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none; height: 0; width: 0;
}}

QAbstractScrollArea::corner {{
    background: {BG_SURFACE};
    border: none;
}}


/* ── QProgressBar ───────────────────────────────────────────────── */

QProgressBar {{
    background: {BG_MUTED};
    border: none;
    border-radius: 4px;
    min-height: 5px;
    max-height: 5px;
    color: transparent;
    text-align: center;
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {PR}, stop:0.5 {AC}, stop:1 {PR});
    border-radius: 4px;
}}


/* ── QSplitter ──────────────────────────────────────────────────── */

QSplitter::handle:horizontal {{ background: {BD_LIGHT}; width:  1px; }}
QSplitter::handle:vertical   {{ background: {BD_LIGHT}; height: 1px; }}
QSplitter::handle:hover      {{ background: {PR}; }}


/* ── QFrame separators ──────────────────────────────────────────── */

QFrame[frameShape="4"],
QFrame[frameShape="5"] {{
    background: {BD_LIGHT};
    color:      {BD_LIGHT};
    border: none;
    max-height: 1px;
    max-width:  1px;
}}


/* ── QMenu ──────────────────────────────────────────────────────── */

QMenu {{
    background: {BG};
    border: 1px solid {BD_LIGHT};
    border-radius: 8px;
    padding: 5px;
}}

QMenu::item {{
    padding: 7px 30px 7px 14px;
    border-radius: 5px;
    color: {TX};
    font-size: 13px;
}}

QMenu::item:selected {{ background: {SEL_BG}; color: {SEL_TX}; }}
QMenu::item:disabled {{ color: {TX_MUT}; }}

QMenu::separator {{
    height: 1px;
    background: {BD_LIGHT};
    margin: 4px 10px;
}}


/* ── QToolTip ───────────────────────────────────────────────────── */

QToolTip {{
    background: #1E293B;
    color: #F1F5F9;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
}}


/* ── Named label chips (semantic status) ────────────────────────── */

QLabel#ChipOk   {{ color:{OK}; background:{OK_BG}; border:1px solid {OK_BD}; border-radius:10px; padding:3px 10px; font-size:12px; font-weight:600; }}
QLabel#ChipWarn {{ color:{WA}; background:{WA_BG}; border:1px solid {WA_BD}; border-radius:10px; padding:3px 10px; font-size:12px; font-weight:600; }}
QLabel#ChipErr  {{ color:{ER}; background:{ER_BG}; border:1px solid {ER_BD}; border-radius:10px; padding:3px 10px; font-size:12px; font-weight:600; }}

QLabel#StatusChip, QLabel#Chip {{
    background: {BG_SURFACE};
    border: 1px solid {BD_LIGHT};
    border-radius: 10px;
    padding: 3px 10px;
    font-size: 12px;
    color: {TX_SEC};
}}

QLabel#HdrTitle {{ font-size: 16px; font-weight: 700; color: {TX}; }}
QLabel#HdrSub   {{ font-size: 12px; color: {TX_SEC}; }}

"""


def apply_theme() -> None:
    """Append the Chem-Inf QSS to QApplication.  Idempotent."""
    try:
        from AnyQt.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return
        existing = app.styleSheet() or ""
        marker = "/* chem-inf-theme-v3 */"
        if marker in existing:
            return
        # Remove any previous version markers
        for old in ("/* chem-inf-theme-applied */", "/* chem-inf-theme-v2 */"):
            existing = existing.replace(old, "")
        app.setStyleSheet(existing + marker + _QSS)
    except Exception:
        pass
