import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chem_inf_widgets.chemcore.mol import ChemMol  # noqa: E402
from chem_inf_widgets.chemcore.services.mol_standardizer import (  # noqa: E402
    STANDARDIZATION_PRESETS,
    MolStandardizer,
    get_standardization_config,
)


class MolStandardizerPhase1Tests(unittest.TestCase):
    def test_named_profiles_exist(self):
        for name in [
            "drug_discovery_default",
            "preserve_salts",
            "docking_pose_safe",
            "fingerprint_canonical",
        ]:
            self.assertIn(name, STANDARDIZATION_PRESETS)
            self.assertIs(get_standardization_config(name), STANDARDIZATION_PRESETS[name])

    def test_standardize_chemmols_does_not_mutate_input_props(self):
        service = MolStandardizer(profile="fingerprint_canonical")
        chem_mol = ChemMol.from_smiles("CCO", name="ethanol")
        chem_mol.set_prop("SMILES", "CCO")

        out_mols, results = service.standardize_chemmols([chem_mol])

        self.assertTrue(results[0].ok)
        self.assertIsNot(out_mols[0], chem_mol)
        self.assertFalse(chem_mol.has_prop("STD_PROFILE"))
        self.assertEqual(out_mols[0].get_prop("STD_PROFILE"), "fingerprint_canonical")
        self.assertEqual(out_mols[0].get_prop("STD_OK"), True)
        self.assertEqual(out_mols[0].get_prop("STD_INPUT_SMILES"), "CCO")


if __name__ == "__main__":
    unittest.main()
