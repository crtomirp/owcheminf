import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.mol_sketcher_core import MolSketcherCore  # noqa: E402


class MolSketcherCoreTests(unittest.TestCase):
    def test_add_compound_rejects_invalid_smiles(self):
        core = MolSketcherCore()
        with self.assertRaises(ValueError):
            core.add_compound("not_a_smiles", {})

    def test_build_molecules_skips_invalid_rows(self):
        core = MolSketcherCore()
        core.rows.append({"smiles": "CCO", "Name": "ethanol"})
        core.rows.append({"smiles": "not_a_smiles", "Name": "bad"})

        molecules = core.build_molecules()

        self.assertEqual(len(molecules), 1)
        self.assertEqual(molecules[0].name, "ethanol")


if __name__ == "__main__":
    unittest.main()
