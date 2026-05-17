import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"


class WheelInstallSmokeTests(unittest.TestCase):
    def test_latest_wheel_installs_and_exposes_packaged_resources(self):
        wheels = sorted(DIST_DIR.glob("chem_inf_widgets-*.whl"))
        if not wheels:
            self.skipTest("No built wheel found in dist/")

        wheel_path = wheels[-1]
        with tempfile.TemporaryDirectory() as tmpdir:
            env_dir = Path(tmpdir) / "wheel-smoke-env"
            subprocess.run(
                [sys.executable, "-m", "venv", str(env_dir)],
                check=True,
                cwd=PROJECT_ROOT,
            )

            bin_dir = "Scripts" if os.name == "nt" else "bin"
            python_exe = env_dir / bin_dir / "python"
            subprocess.run(
                [str(python_exe), "-m", "pip", "install", "--no-deps", str(wheel_path)],
                check=True,
                cwd=PROJECT_ROOT,
            )

            check_script = textwrap.dedent(
                """
                import json
                from importlib import resources

                expected = {
                    "chem_inf_widgets.chemcore.data": [
                        "smartspains.json",
                        "pharmafp250.json",
                        "cyclic_registry.json",
                        "patterns.csv",
                    ],
                    "chem_inf_widgets.widgets": [
                        "icons/editors_viewers/owcompounddetailcardwidget.svg",
                        "icons/standardization_filtering/owpharmafpsearchwidget.svg",
                        "ketcher/standalone/indexqt.html",
                        "ketcher/standalone/static/js/main.js",
                    ],
                    "chem_inf_widgets.chemcore": [
                        "resources/padel_presets/all_2d_descriptors.xml",
                        "resources/jsme/jsme_panel.html",
                    ],
                }

                results = {}
                for package, relpaths in expected.items():
                    root = resources.files(package)
                    results[package] = {
                        relpath: root.joinpath(relpath).is_file()
                        for relpath in relpaths
                    }

                print(json.dumps(results))
                """
            )

            proc = subprocess.run(
                [str(python_exe), "-c", check_script],
                check=True,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            results = json.loads(proc.stdout.strip())

        missing = [
            f"{package}:{relpath}"
            for package, package_results in results.items()
            for relpath, exists in package_results.items()
            if not exists
        ]
        self.assertFalse(
            missing,
            f"Installed wheel is missing packaged runtime resources: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
