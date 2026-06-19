"""F1 widget help wiring.

These tests do not need Qt or a network: they exercise the generator
(``scripts/build_widget_help.py``) directly and check that every widget gets a
help page whose link text equals the widget's canonical ``name`` — which is what
Orange's ``html-index`` provider matches on.
"""
import html
import importlib.util
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

HAVE_MARKDOWN = find_spec("markdown") is not None


def _load_generator():
    path = PROJECT_ROOT / "scripts" / "build_widget_help.py"
    spec = importlib.util.spec_from_file_location("build_widget_help", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WidgetHelpTests(unittest.TestCase):
    def test_help_path_points_at_bundled_index(self):
        import chem_inf_widgets.widgets as widgets_pkg

        self.assertTrue(widgets_pkg.WIDGET_HELP_PATH)
        target, _xpath = widgets_pkg.WIDGET_HELP_PATH[0]
        self.assertTrue(target.replace("\\", "/").endswith("widgets/help/index.html"))

    @unittest.skipUnless(HAVE_MARKDOWN, "needs the 'markdown' package (pip install -e .[dev])")
    def test_every_widget_resolves_to_a_page(self):
        gen = _load_generator()
        out_dir = PROJECT_ROOT / "src" / "chem_inf_widgets" / "widgets" / "help"
        gen.OUT_DIR = out_dir  # build into the package (gitignored)
        gen.build()

        index = (out_dir / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="widgets"', index)

        for _module, name, _category, _doc in gen.discover_widgets():
            with self.subTest(widget=name):
                # canonical name appears as a link in the index (HTML-escaped in
                # the file; Orange's parser un-escapes it back when matching).
                self.assertIn(f">{html.escape(name)}</a>", index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
