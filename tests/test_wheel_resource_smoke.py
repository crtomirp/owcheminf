import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"


class WheelResourceSmokeTests(unittest.TestCase):
    def test_latest_wheel_contains_runtime_resources(self):
        wheels = sorted(DIST_DIR.glob("chem_inf_widgets-*.whl"))
        if not wheels:
            self.skipTest("No built wheel found in dist/")

        wheel_path = wheels[-1]
        with zipfile.ZipFile(wheel_path) as zf:
            names = set(zf.namelist())

        expected = {
            "chem_inf_widgets/chemcore/data/smartspains.json",
            "chem_inf_widgets/chemcore/data/pharmafp250.json",
            "chem_inf_widgets/chemcore/data/cyclic_registry.json",
            "chem_inf_widgets/chemcore/data/patterns.csv",
            "chem_inf_widgets/widgets/ketcher/standalone/indexqt.html",
            "chem_inf_widgets/widgets/ketcher/standalone/static/js/main.js",
            "chem_inf_widgets/widgets/icons/editors_viewers/owcompounddetailcardwidget.svg",
            "chem_inf_widgets/widgets/icons/standardization_filtering/owpharmafpsearchwidget.svg",
            "chem_inf_widgets/chemcore/resources/padel_presets/all_2d_descriptors.xml",
            "chem_inf_widgets/chemcore/resources/jsme/jsme_panel.html",
        }

        missing = sorted(path for path in expected if path not in names)
        self.assertFalse(missing, f"Wheel is missing expected packaged resources: {missing}")


if __name__ == "__main__":
    unittest.main()
