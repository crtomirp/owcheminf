from __future__ import annotations

from chem_inf_widgets.widgets import ow_mol_ketcher_editor as _ketcher_editor


class OWMolSketcher(_ketcher_editor.OWMolKetcher, openclass=True):
    """Primary embedded sketcher widget.

    On macOS this widget now prefers the embedded Ketcher backend first and
    only falls back to JSME if Ketcher cannot initialize.
    """

    name = "Mol Editor"
    description = "Primary embedded molecular editor for database creation"
    icon = "icons/editors_viewers/owmoleditorwidget.png"
    priority = 111
    keywords = ["chemistry", "molecule", "sketcher", "editor", "ketcher", "jsme"]
    prefer_jsme_fallback_on_macos = False


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWMolSketcher).run()
