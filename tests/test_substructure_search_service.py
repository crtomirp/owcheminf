import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.substructure_search_service import (  # noqa: E402
    SearchConfig,
    canonical_smiles_no_h,
    parse_query_mol_auto,
    search_smiles,
)


class SubstructureSearchServiceTests(unittest.TestCase):
    def test_canonical_smiles_no_h_keeps_invalid_input_as_text(self):
        self.assertEqual(canonical_smiles_no_h("not-a-smiles"), "not-a-smiles")

    def test_parse_query_mol_auto_rejects_invalid_similarity_query(self):
        with self.assertRaises(ValueError):
            parse_query_mol_auto("not-a-smiles", "similarity")

    def test_search_smiles_skips_invalid_library_rows(self):
        cfg = SearchConfig(search_type="exact")
        hits = search_smiles(["CCO", "not-a-smiles", "O"], "CCO", cfg)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].idx, 0)


if __name__ == "__main__":
    unittest.main()
