import sys
import unittest
import os
from importlib import resources
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class PackagingSmokeTests(unittest.TestCase):
    def test_core_namespace_imports(self):
        import chem_inf_widgets.chemcore  # noqa: F401
        import chem_inf_widgets.chemcore.admet  # noqa: F401
        import chem_inf_widgets.chemcore.io  # noqa: F401
        import chem_inf_widgets.chemcore.services  # noqa: F401

    def test_widget_category_metadata_exists(self):
        import chem_inf_widgets.widgets as widgets_pkg

        self.assertEqual(widgets_pkg.NAME, "Chemoinformatics")
        self.assertTrue(bool(widgets_pkg.DESCRIPTION))
        self.assertTrue(hasattr(widgets_pkg, "_CATEGORY_SPECS"))
        self.assertTrue(len(widgets_pkg._CATEGORY_SPECS) >= 1)
        self.assertTrue(all(bool(spec["icon"]) for spec in widgets_pkg._CATEGORY_SPECS))

    def test_pains_resource_is_packaged(self):
        resource = resources.files("chem_inf_widgets.chemcore.data").joinpath("smartspains.json")
        self.assertTrue(resource.is_file(), "chemcore/data/smartspains.json should be packaged")

    def test_pharmafp_resource_is_packaged(self):
        resource = resources.files("chem_inf_widgets.chemcore.data").joinpath("pharmafp250.json")
        self.assertTrue(resource.is_file(), "chemcore/data/pharmafp250.json should be packaged")

    def test_motif_resources_are_packaged(self):
        json_resource = resources.files("chem_inf_widgets.chemcore.data").joinpath("cyclic_registry.json")
        csv_resource = resources.files("chem_inf_widgets.chemcore.data").joinpath("patterns.csv")
        self.assertTrue(json_resource.is_file(), "chemcore/data/cyclic_registry.json should be packaged")
        self.assertTrue(csv_resource.is_file(), "chemcore/data/patterns.csv should be packaged")

    def test_padel_presets_exist(self):
        from chem_inf_widgets.chemcore.services.padel_descriptor_service import _PADEL_PRESET_DIR

        self.assertTrue((_PADEL_PRESET_DIR / "all_2d_descriptors.xml").is_file())
        self.assertTrue((_PADEL_PRESET_DIR / "fp_maccs.xml").is_file())

    def test_ketcher_qt_html_exists(self):
        widget_dir = PROJECT_ROOT / "src" / "chem_inf_widgets" / "widgets"
        indexqt = os.path.join(widget_dir, "ketcher", "standalone", "indexqt.html")
        self.assertTrue(os.path.isfile(indexqt), "Ketcher Qt HTML should be packaged with the widget")

    def test_jsme_panel_exists(self):
        resource = resources.files("chem_inf_widgets.chemcore").joinpath("resources/jsme/jsme_panel.html")
        self.assertTrue(resource.is_file(), "JSME panel HTML should be packaged with chemcore resources")


if __name__ == "__main__":
    unittest.main()
