from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("Orange")

from chem_inf_widgets.chemcore.molecule_contract import QC_FLAGS, ROW_ID, TRANSFORM_LOG
from chem_inf_widgets.chemcore.services.from_orange import chemmols_to_table, table_to_chemmols_with_report
from chem_inf_widgets.chemcore.services.mol_standardizer import MolStandardizer
from chem_inf_widgets.chemcore.services.molecule_import_service import MoleculeImportConfig, import_molecule_file
from chem_inf_widgets.chemcore.services.molecule_qc_service import annotate_chemmols_with_qc, run_molecule_qc
from chem_inf_widgets.chemcore.services.qsar_dataset_builder_service import QSARDatasetBuilderConfig, build_qsar_dataset
from chem_inf_widgets.widgets.ow_mol_standardizer import OWMolStandardizer
from chem_inf_widgets.widgets.ow_molecule_qc_dashboard import OWMoleculeQCDashboard
from chem_inf_widgets.widgets.ow_qsar_dataset_builder import _table_from_records, _table_to_records


def _meta_names(table) -> list[str]:
    return [var.name for var in table.domain.metas]


def test_end_to_end_workflow_preserves_provenance_into_qsar_builder(tmp_path: Path):
    path = tmp_path / "workflow.csv"
    path.write_text(
        "compound_id,name,smiles,standard_type,standard_relation,standard_value,standard_units\n"
        "M001,ethanol_a,CCO,IC50,=,100,nM\n"
        "M002,ethanol_b,OCC,IC50,=,200,nM\n"
        "M003,ethanol_salt,CCO.Cl,IC50,=,150,nM\n"
        "M004,bad,C1CC,IC50,=,50,nM\n",
        encoding="utf-8",
    )

    imported = import_molecule_file(path, MoleculeImportConfig(smiles_column="smiles", name_column="name"))
    assert imported.summary.valid_records == 3
    assert imported.summary.failed_records == 1

    imported_table = chemmols_to_table(imported.mols)

    qc_input_mols, qc_conversion = table_to_chemmols_with_report(imported_table)
    qc_result = run_molecule_qc(qc_input_mols)
    qc_mols = annotate_chemmols_with_qc(qc_input_mols, qc_result.records)
    annotated_table = OWMoleculeQCDashboard._annotated_table(
        imported_table,
        qc_result.records,
        qc_mols,
        [max(0, int(i) - 1) for i in qc_conversion.skipped_rows],
        qc_conversion.errors,
    )

    qc_meta_names = _meta_names(annotated_table)
    assert "row_id" in qc_meta_names
    assert "transform_log" in qc_meta_names
    assert "qc_flags" in qc_meta_names
    assert "dropped_reason" in qc_meta_names

    std_input_mols, _ = table_to_chemmols_with_report(annotated_table)
    std_mols, std_results = MolStandardizer(profile="qsar_ready").standardize_chemmols(std_input_mols)
    std_mols = OWMolStandardizer._annotate_standardization_curation(std_mols, std_results, "qsar_ready")
    standardized_table = chemmols_to_table(std_mols)

    for cm in std_mols:
        assert cm.props.get(ROW_ID)
        assert cm.props.get(QC_FLAGS) is not None
        log = str(cm.props.get(TRANSFORM_LOG, "") or "")
        assert "import_table" in log
        assert "molecule_qc" in log
        assert "standardize_qsar_ready" in log
        assert "curation_standardization" in log

    standardized_meta_names = _meta_names(standardized_table)
    assert "row_id" in standardized_meta_names
    assert "transform_log" in standardized_meta_names
    assert "qc_flags" in standardized_meta_names
    assert "standardized_smiles" in standardized_meta_names

    builder_records = _table_to_records(standardized_table)
    builder_result = build_qsar_dataset(
        builder_records,
        QSARDatasetBuilderConfig(
            smiles_column="standardized_smiles",
            name_column="compound_id",
            activity_column="standard_value",
            unit_column="standard_units",
            relation_column="standard_relation",
            endpoint_column="standard_type",
            target_endpoint="IC50",
            duplicate_key="standard_inchikey",
        ),
    )

    assert builder_result.summary["prepared_compounds"] == 1
    assert builder_result.summary["rejected_records"] == 0

    row = builder_result.prepared_records[0]
    assert row["n_measurements"] == 3
    assert len([part for part in str(row["source_row_ids"]).split(";") if part]) == 3
    assert "standardize_qsar_ready" in str(row["source_transform_logs"])
    assert "duplicate_structure" in str(row["source_qc_flags_all"])
    assert "multi_fragment" in str(row["source_qc_flags_all"])
    assert str(row["source_dropped_reasons"] or "") == ""

    ready_table = _table_from_records(builder_result.prepared_records, class_column="pActivity", name="QSAR Ready Data")
    assert ready_table is not None
    assert ready_table.domain.class_var.name == "pActivity"
    ready_meta_names = _meta_names(ready_table)
    assert "source_row_ids" in ready_meta_names
    assert "source_transform_logs" in ready_meta_names
    assert "source_qc_flags_all" in ready_meta_names
