import importlib
import pkgutil
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


try:
    import Orange  # noqa: F401
    import AnyQt  # noqa: F401
except Exception:  # pragma: no cover
    ORANGE_WIDGET_RUNTIME_AVAILABLE = False
else:
    ORANGE_WIDGET_RUNTIME_AVAILABLE = True


@unittest.skipUnless(ORANGE_WIDGET_RUNTIME_AVAILABLE, "Orange/AnyQt runtime is not available")
class WidgetImportSmokeTests(unittest.TestCase):
    def test_all_widget_modules_import(self):
        import chem_inf_widgets.widgets as widget_pkg

        failures = {}
        module_names = sorted(
            module.name
            for module in pkgutil.iter_modules(widget_pkg.__path__)
            if module.name.startswith("ow_")
        )

        for module_name in module_names:
            try:
                importlib.import_module(f"chem_inf_widgets.widgets.{module_name}")
            except Exception as exc:  # pragma: no cover - only hit when a widget breaks import
                failures[module_name] = f"{type(exc).__name__}: {exc}"

        self.assertFalse(
            failures,
            "Widget import failures detected:\n" + "\n".join(
                f"- {name}: {error}" for name, error in sorted(failures.items())
            ),
        )


if __name__ == "__main__":
    unittest.main()
