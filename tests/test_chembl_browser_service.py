import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chem_inf_widgets.chemcore.mol import ChemMol  # noqa: E402
from chem_inf_widgets.chemcore.services.chembl_browser_service import (  # noqa: E402
    SummaryRow,
    compile_user_pattern,
    filter_targets,
    format_number,
    format_output_summary,
    query_needs_postfilter,
    summarize_activity_types,
)
from chem_inf_widgets.chemcore.services.chembl_models import (  # noqa: E402
    ChemBLBioactivityRecord,
    ChemBLTargetRecord,
)
from chem_inf_widgets.widgets.ow_chembl_browser import _safe_fetch_props_by_id  # noqa: E402

try:
    from Orange.data import Domain, StringVariable, Table
except Exception:  # pragma: no cover
    Domain = StringVariable = Table = None  # type: ignore


class ChemBLBrowserServiceTests(unittest.TestCase):
    def test_safe_fetch_props_by_id_surfaces_warning_on_failure(self):
        class _FailingService:
            def fetch_molecules_with_properties(self, ids, prop_keys):
                raise RuntimeError("network unavailable")

        props, warning = _safe_fetch_props_by_id(_FailingService(), ["CHEMBL1"], ["alogp"])

        self.assertEqual(props, {})
        self.assertIn("Property enrichment skipped", warning)
        self.assertIn("network unavailable", warning)

    def test_summarize_activity_types_uses_type_and_units(self):
        rows = summarize_activity_types(
            [
                ChemBLBioactivityRecord(
                    molecule_chembl_id="CHEMBL1",
                    target_chembl_id="CHEMBL_TARGET",
                    smiles="CCO",
                    standard_type="IC50",
                    standard_value=12.0,
                    standard_units="nM",
                    pchembl_value=8.1,
                    ic50_nM=12.0,
                ),
                ChemBLBioactivityRecord(
                    molecule_chembl_id="CHEMBL2",
                    target_chembl_id="CHEMBL_TARGET",
                    smiles="CCN",
                    standard_type="IC50",
                    standard_value=18.0,
                    standard_units="nM",
                    pchembl_value=7.8,
                    ic50_nM=18.0,
                ),
            ]
        )
        self.assertEqual(rows[0], SummaryRow(key="IC50 (nM)", count=2, pct=100.0))

    def test_compile_user_pattern_supports_wildcards(self):
        pattern = compile_user_pattern("CYP*")
        self.assertIsNotNone(pattern)
        self.assertTrue(pattern.search("CYP3A4"))
        self.assertFalse(pattern.search("kinase"))

    def test_filter_targets_uses_search_blob(self):
        targets = [
            ChemBLTargetRecord("CHEMBL1", "EGFR", "Human", "SINGLE PROTEIN"),
            ChemBLTargetRecord("CHEMBL2", "DRD2", "Human", "SINGLE PROTEIN"),
        ]
        filtered = filter_targets(targets, compile_user_pattern("EGFR"))
        self.assertEqual([target.chembl_id for target in filtered], ["CHEMBL1"])

    def test_query_needs_postfilter_detects_patterns(self):
        self.assertTrue(query_needs_postfilter("CYP*"))
        self.assertTrue(query_needs_postfilter("/kinase$/"))
        self.assertFalse(query_needs_postfilter("EGFR"))

    def test_format_number_is_stable(self):
        self.assertEqual(format_number(1.2345, 2), "1.23")
        self.assertEqual(format_number(None), "")

    @unittest.skipIf(Table is None, "Orange is required for output summary formatting")
    def test_format_output_summary_reports_rows_and_molecules(self):
        smiles_var = StringVariable("SMILES")
        smiles_var.attributes["format"] = "SMILES"
        domain = Domain([], metas=[smiles_var])
        table = Table.from_numpy(domain, X=[[]], metas=[["CCO"]])
        mols = [ChemMol.from_smiles("CCO", name="ethanol")]
        summary = format_output_summary(table, mols)
        self.assertIn("rows=1", summary)
        self.assertIn("molecules=1", summary)


if __name__ == "__main__":
    unittest.main()
