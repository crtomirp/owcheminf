import sys
import unittest
import importlib.util
from pathlib import Path

from rdkit import Chem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ORANGE_AVAILABLE = importlib.util.find_spec("Orange") is not None

if ORANGE_AVAILABLE:
    from chem_inf_widgets.chemcore.services.compound_detail_service import (  # noqa: E402
        PHARMAFP_SIZE,
        build_detail,
        build_detail_outputs,
        compute_motif_hits,
        compute_pharmafp_hits,
        compiled_pharmafp_patterns,
        compute_properties,
        fragment_query_table,
        load_pharmafp_library,
        motif_hits_table,
        query_molecule_from_detail,
        render_fragment_detail_html,
        render_motif_detail_html,
        render_summary_html,
        run_pharmafp_search,
        scaffold_query_table,
        selected_motif_query_table,
        search_profile_table,
        tanimoto_bits,
    )


@unittest.skipUnless(ORANGE_AVAILABLE, "Orange is required for compound detail service tests")
class CompoundDetailServiceTests(unittest.TestCase):
    def test_pharmafp_library_is_available(self):
        library = load_pharmafp_library()
        self.assertEqual(len(library), PHARMAFP_SIZE)
        self.assertEqual(library[0].name, "Imidazole")

    def test_compiled_patterns_match_library_size(self):
        self.assertEqual(len(compiled_pharmafp_patterns()), PHARMAFP_SIZE)

    def test_imidazole_fragment_is_detected(self):
        mol = Chem.MolFromSmiles("c1c[nH]cn1")
        hits, bits = compute_pharmafp_hits(mol)
        names = {hit.name for hit in hits}
        self.assertIn("Imidazole", names)
        self.assertGreater(sum(bits), 0)

    def test_similarity_bits_returns_one_for_identical(self):
        mol = Chem.MolFromSmiles("c1c[nH]cn1")
        _hits_a, bits_a = compute_pharmafp_hits(mol)
        _hits_b, bits_b = compute_pharmafp_hits(mol)
        self.assertAlmostEqual(tanimoto_bits(bits_a, bits_b), 1.0)

    def test_build_detail_returns_sorted_similar_hits(self):
        query = Chem.MolFromSmiles("c1c[nH]cn1")
        refs = [
            type("Ref", (), {"source_index": 0, "name": "same", "smiles": "c1c[nH]cn1", "mol": Chem.MolFromSmiles("c1c[nH]cn1")}),
            type("Ref", (), {"source_index": 1, "name": "imidazole methyl", "smiles": "Cc1ncc[nH]1", "mol": Chem.MolFromSmiles("Cc1ncc[nH]1")}),
            type("Ref", (), {"source_index": 2, "name": "benzene", "smiles": "c1ccccc1", "mol": Chem.MolFromSmiles("c1ccccc1")}),
        ]
        detail = build_detail(query, name="query", reference=refs, top_k=2, exclude_smiles="c1c[nH]cn1")
        self.assertGreaterEqual(len(detail.fragment_hits), 1)
        self.assertGreaterEqual(len(detail.similar_hits), 1)
        self.assertGreaterEqual(detail.similar_hits[0].similarity, detail.similar_hits[-1].similarity)

    def test_properties_include_formula(self):
        props = compute_properties(Chem.MolFromSmiles("CCO"))
        self.assertTrue(props.formula)
        self.assertGreater(props.mol_weight, 0)

    def test_render_helpers_and_output_bundle(self):
        query = Chem.MolFromSmiles("c1c[nH]cn1")
        detail = build_detail(query, name="imidazole", reference=[], top_k=3)
        motifs = compute_motif_hits(query)

        summary_html = render_summary_html(detail)
        fragment_html = render_fragment_detail_html(detail.fragment_hits[:1])
        motif_html = render_motif_detail_html(motifs[:1], motif_logic="and")
        outputs = build_detail_outputs(
            detail,
            motif_hits=motifs,
            selected_motif_hits=motifs[:1],
            motif_logic="and",
        )

        self.assertIn("imidazole", summary_html.lower())
        self.assertIn("selected fragments", fragment_html)
        self.assertIn("motif queries", motif_html)
        self.assertIsNotNone(outputs.selected_compound)
        self.assertIsNotNone(outputs.query_molecule)
        self.assertIsNotNone(outputs.scaffold_query)

    def test_search_ready_outputs_and_pharma_search(self):
        query = Chem.MolFromSmiles("c1c[nH]cn1")
        refs = [
            type("Ref", (), {"source_index": 1, "name": "imidazole methyl", "smiles": "Cc1ncc[nH]1", "mol": Chem.MolFromSmiles("Cc1ncc[nH]1")}),
            type("Ref", (), {"source_index": 2, "name": "benzene", "smiles": "c1ccccc1", "mol": Chem.MolFromSmiles("c1ccccc1")}),
        ]
        detail = build_detail(query, name="query", reference=refs, top_k=2)
        self.assertIsNotNone(query_molecule_from_detail(detail))
        self.assertIsNotNone(fragment_query_table(detail.fragment_hits))
        self.assertIsNotNone(scaffold_query_table(detail))
        self.assertEqual(search_profile_table(detail)[0]["Preferred Search Mode"], "hybrid")

        hits = run_pharmafp_search(
            query_smiles=detail.properties.smiles,
            reference=refs,
            fragment_queries=[],
            motif_queries=[],
            motif_logic="or",
            query_scaffold="",
            query_generic_scaffold="",
            top_k=2,
            min_similarity=0.0,
            mode="hybrid",
        )
        self.assertEqual(len(hits), 2)
        self.assertGreaterEqual(hits[0].hybrid_score, hits[1].hybrid_score)

    def test_motif_detection_and_and_or_search(self):
        query = Chem.MolFromSmiles("Oc1ccc(cc1)C=NNC2=NC=CC=N2")
        refs = [
            type("Ref", (), {"source_index": 1, "name": "same motifs", "smiles": "Oc1ccccc1NNC2=NC=CC=N2", "mol": Chem.MolFromSmiles("Oc1ccccc1NNC2=NC=CC=N2")}),
            type("Ref", (), {"source_index": 2, "name": "only phenol", "smiles": "Oc1ccccc1", "mol": Chem.MolFromSmiles("Oc1ccccc1")}),
        ]
        motifs = compute_motif_hits(query)
        self.assertTrue(motifs)
        self.assertIsNotNone(motif_hits_table(motifs))
        self.assertIsNotNone(selected_motif_query_table(motifs[:2]))

        selected = tuple(hit for hit in motifs if hit.category in {"heterocycle", "functional_group"})[:2]
        self.assertTrue(selected)
        profile = search_profile_table(build_detail(query, name="q"), motif_queries=selected, motif_logic="and")
        self.assertEqual(profile[0]["Motif Logic"], "and")

        and_hits = run_pharmafp_search(
            query_smiles=Chem.MolToSmiles(query),
            reference=refs,
            fragment_queries=[],
            motif_queries=[hit.smarts for hit in selected],
            motif_logic="and",
            query_scaffold="",
            query_generic_scaffold="",
            top_k=5,
            min_similarity=0.0,
            mode="hybrid",
        )
        or_hits = run_pharmafp_search(
            query_smiles=Chem.MolToSmiles(query),
            reference=refs,
            fragment_queries=[],
            motif_queries=[hit.smarts for hit in selected],
            motif_logic="or",
            query_scaffold="",
            query_generic_scaffold="",
            top_k=5,
            min_similarity=0.0,
            mode="hybrid",
        )
        self.assertLessEqual(len(and_hits), len(or_hits))


if __name__ == "__main__":
    unittest.main()
