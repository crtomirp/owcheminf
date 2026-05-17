import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class RepositoryHygieneTests(unittest.TestCase):
    def test_no_source_side_egg_info_directory_remains(self):
        forbidden = []
        for egg_info_dir in SRC_ROOT.rglob("*.egg-info"):
            forbidden.append(egg_info_dir.relative_to(PROJECT_ROOT).as_posix())

        self.assertFalse(
            forbidden,
            "Source tree contains generated egg-info directories: " + ", ".join(sorted(forbidden)),
        )

    def test_no_duplicate_python_source_with_js_extension(self):
        stray = PROJECT_ROOT / "src" / "chem_inf_widgets" / "chemcore" / "descriptors" / "descriptors.js"
        self.assertFalse(
            stray.exists(),
            f"Unexpected duplicate Python source file remains in tree: {stray.relative_to(PROJECT_ROOT)}",
        )


if __name__ == "__main__":
    unittest.main()
