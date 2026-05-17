import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.reaction_viewer_service import (  # noqa: E402
    build_export_name,
    compose_reaction_string,
    parse_reaction_string,
    pick_preferred_column,
    safe_slug,
)


class ReactionViewerServiceTests(unittest.TestCase):
    def test_pick_preferred_column_matches_case_insensitively(self):
        chosen = pick_preferred_column(
            ["Name", "rxn_mapped", "SMILES"],
            ["reaction", "RXN_MAPPED"],
        )

        self.assertEqual(chosen, "rxn_mapped")

    def test_compose_reaction_string_normalizes_separators(self):
        rxn = compose_reaction_string("CCO + O", "CC=O.O")

        self.assertEqual(rxn, "CCO.O>>CC=O.O")

    def test_compose_reaction_string_skips_missing_values(self):
        self.assertIsNone(compose_reaction_string("?", "CCO"))
        self.assertIsNone(compose_reaction_string("CCO", ""))

    def test_parse_reaction_string_supports_smiles_mode(self):
        reaction = parse_reaction_string("[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]")

        self.assertIsNotNone(reaction)
        self.assertEqual(reaction.GetNumProductTemplates(), 1)

    def test_safe_slug_and_export_name(self):
        self.assertEqual(safe_slug("My reaction: 1/2"), "My_reaction_12")
        self.assertEqual(build_export_name("rxn_", 7, "My reaction"), "rxn_0007_My_reaction")


if __name__ == "__main__":
    unittest.main()
