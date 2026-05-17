import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.reaction_enumerator_service import enumerate_reaction_products  # noqa: E402


class ReactionEnumeratorServiceTests(unittest.TestCase):
    def test_enumerate_reaction_products_creates_products(self):
        products = enumerate_reaction_products(
            [["CCO"], ["O"]],
            [("Oxidation", "CCO.O>>CC(=O)O")],
            max_products=10,
        )
        self.assertIsInstance(products, list)

    def test_enumerate_reaction_products_skips_invalid_reactants(self):
        products = enumerate_reaction_products(
            [["CCO", "not_a_smiles"], ["O"]],
            [("Oxidation", "CCO.O>>CC(=O)O")],
            max_products=10,
        )
        self.assertTrue(all("not_a_smiles" not in product.reactants for product in products))


if __name__ == "__main__":
    unittest.main()
