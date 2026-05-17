import sys
import unittest
from pathlib import Path

import numpy as np
from Orange.data import Domain, StringVariable, Table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.mol import ChemMol  # noqa: E402
from chem_inf_widgets.chemcore.services.mol_depict import chemmols_to_items, table_to_items  # noqa: E402


class MolDepictTests(unittest.TestCase):
    def test_table_to_items_skips_invalid_smiles(self):
        smiles_var = StringVariable("SMILES")
        name_var = StringVariable("Name")
        domain = Domain([], metas=[smiles_var, name_var])
        metas = np.array(
            [
                ["CCO", "ethanol"],
                ["not_a_smiles", "bad"],
            ],
            dtype=object,
        )
        table = Table.from_numpy(domain, X=np.zeros((2, 0), dtype=float), metas=metas)

        items = table_to_items(table)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "ethanol")

    def test_chemmols_to_items_uses_smiles_fallback(self):
        chem_mol = ChemMol.from_smiles("CCO", name="ethanol")
        chem_mol.mol = None
        chem_mol.set_prop("SMILES", "CCO")

        items = chemmols_to_items([chem_mol])

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "ethanol")


if __name__ == "__main__":
    unittest.main()
