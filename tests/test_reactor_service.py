import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.reactor_service import (  # noqa: E402
    ReactionRule,
    ReactorEngine,
    build_preview_text,
    coerce_seed,
)


class ReactorServiceTests(unittest.TestCase):
    def test_coerce_seed_falls_back_to_default(self):
        self.assertEqual(coerce_seed("17"), 17)
        self.assertEqual(coerce_seed("bad-seed", default=9), 9)

    def test_build_preview_text_truncates_long_preview(self):
        preview = build_preview_text(["a", "b", "c"], 2)

        self.assertEqual(preview, "a\nb\n... (1 more lines)")

    def test_duplicate_rules_are_deduplicated_and_weights_are_summed(self):
        smirks = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
        rule_a = ReactionRule.from_row("r1", smirks, 1.5)
        rule_b = ReactionRule.from_row("r2", smirks, 0.5)

        engine = ReactorEngine(["CCO"], [rule_a, rule_b], seed=1)

        self.assertEqual(len(engine.rules), 1)
        self.assertEqual(engine.rules[0].weight, 2.0)

    def test_expand_pool_adds_unique_products_only(self):
        rule = ReactionRule.from_row(
            "oxidize",
            "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
        )
        engine = ReactorEngine(
            ["CCO"],
            [rule],
            seed=1,
            expand_pool=True,
            unique_products=True,
        )

        records = engine.step(draws_per_step=1, max_products_per_draw=1)

        self.assertEqual(len(records), 1)
        self.assertIn("CC=O", engine.smiles_pool)
        self.assertEqual(engine.smiles_pool.count("CC=O"), 1)

    def test_invalid_input_smiles_are_skipped_when_building_pool(self):
        rule = ReactionRule.from_row(
            "oxidize",
            "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
        )
        engine = ReactorEngine(["CCO", "not-a-smiles"], [rule], seed=1)

        self.assertEqual(engine.smiles_pool, ["CCO"])
        self.assertEqual(len(engine.mol_pool), 1)


if __name__ == "__main__":
    unittest.main()
