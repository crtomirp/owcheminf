from __future__ import annotations

import csv
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")

from chem_inf_widgets.chemcore.molecule_contract import DROPPED_REASON, QC_DUPLICATE_COUNT, QC_FLAGS, ROW_ID, TRANSFORM_LOG
from chem_inf_widgets.chemcore.services.molecule_import_service import (
    MoleculeImportConfig,
    detect_import_format,
    import_molecule_file,
    import_records_as_dicts,
    import_summary_as_rows,
)


def test_detect_import_format():
    assert detect_import_format("x.csv") == "table"
    assert detect_import_format("x.tsv") == "tsv"
    assert detect_import_format("x.smi") == "smi"
    assert detect_import_format("x.sdf") == "sdf"


def test_import_csv_demo(tmp_path: Path):
    p = tmp_path / "demo.csv"
    p.write_text(
        "name,smiles,group\n"
        "aspirin,CC(=O)Oc1ccccc1C(=O)O,drug\n"
        "pyridine,c1ccncc1,heterocycle\n"
        "bad,C1CC,bad\n",
        encoding="utf-8",
    )
    result = import_molecule_file(p, MoleculeImportConfig(smiles_column="smiles", name_column="name"))
    assert result.summary.total_records == 3
    assert result.summary.valid_records == 2
    assert result.summary.failed_records == 1
    assert len(result.mols) == 2
    assert result.failed_records[0].name == "bad"


def test_import_smi_demo(tmp_path: Path):
    p = tmp_path / "demo.smi"
    p.write_text("c1ccncc1 pyridine\nC1CC invalid\n", encoding="utf-8")
    result = import_molecule_file(p)
    assert result.summary.total_records == 2
    assert result.summary.valid_records == 1
    assert result.summary.failed_records == 1


def test_report_helpers(tmp_path: Path):
    p = tmp_path / "demo.csv"
    p.write_text("name,smiles\nbenzene,c1ccccc1\nbad,C1CC\n", encoding="utf-8")
    result = import_molecule_file(p)
    rows = import_records_as_dicts(result.records)
    summary_rows = import_summary_as_rows(result.summary)
    assert rows[0]["ok"] == 1
    assert rows[0]["qc_flags"] == ""
    assert rows[0]["dropped_reason"] == ""
    assert rows[1]["qc_flags"] == "invalid_structure"
    assert rows[1]["dropped_reason"] == "invalid_structure"
    assert any(row["metric"] == "valid_records" for row in summary_rows)


def test_imported_numeric_activity_becomes_orange_attribute(tmp_path: Path):
    Orange = pytest.importorskip("Orange")

    from chem_inf_widgets.chemcore.services.from_orange import chemmols_to_table

    p = tmp_path / "activity.csv"
    p.write_text(
        "compound_id,name,smiles,activity,source\n"
        "M001,Aspirin,CC(=O)Oc1ccccc1C(=O)O,6.43,teaching\n"
        "M002,Pyridine,c1ccncc1,4.12,teaching\n",
        encoding="utf-8",
    )
    result = import_molecule_file(p, MoleculeImportConfig(smiles_column="smiles", name_column="name"))
    table = chemmols_to_table(result.mols)

    assert "activity" in [var.name for var in table.domain.attributes]
    assert "activity" not in [var.name for var in table.domain.metas]
    assert "compound_id" in [var.name for var in table.domain.metas]
    assert "source" in [var.name for var in table.domain.metas]
    assert list(table.X[:, 0]) == [6.43, 4.12]


def test_import_hub_duplicate_detection_keeps_duplicates_by_default(tmp_path: Path):
    p = tmp_path / "dups.csv"
    p.write_text(
        "name,smiles\n"
        "ethanol_a,CCO\n"
        "ethanol_b,OCC\n"
        "benzene,c1ccccc1\n",
        encoding="utf-8",
    )

    result = import_molecule_file(p, MoleculeImportConfig(smiles_column="smiles", name_column="name"))

    assert result.summary.valid_records == 3
    assert result.summary.accepted_records == 3
    assert result.summary.rejected_records == 0
    assert result.summary.duplicate_groups == 1
    assert result.summary.duplicate_records == 2
    duplicate_rows = [r for r in result.records if r.duplicate_count > 1]
    assert [r.duplicate_group_index for r in duplicate_rows] == [1, 2]
    assert all(r.accepted for r in duplicate_rows)
    duplicate_mols = [cm for cm in result.mols if int(cm.props.get(QC_DUPLICATE_COUNT, 0)) > 1]
    assert len(duplicate_mols) == 2
    assert all("duplicate_structure" in str(cm.props.get(QC_FLAGS, "")) for cm in duplicate_mols)
    assert all(not str(cm.props.get(DROPPED_REASON, "")) for cm in duplicate_mols)
    assert all(cm.props.get(ROW_ID) for cm in result.mols)
    assert all(str(cm.props.get(TRANSFORM_LOG, "")).startswith("import_table") for cm in result.mols)


def test_import_hub_can_reject_duplicate_structures_after_first(tmp_path: Path):
    p = tmp_path / "dups.csv"
    p.write_text(
        "name,smiles\n"
        "ethanol_a,CCO\n"
        "ethanol_b,OCC\n"
        "bad,C1CC\n",
        encoding="utf-8",
    )

    result = import_molecule_file(
        p,
        MoleculeImportConfig(
            smiles_column="smiles",
            name_column="name",
            reject_duplicate_structures=True,
        ),
    )

    assert result.summary.valid_records == 2
    assert result.summary.failed_records == 1
    assert result.summary.accepted_records == 1
    assert result.summary.rejected_records == 2
    rejected_reasons = [r.rejection_reason for r in result.rejected_records]
    assert "duplicate_structure" in rejected_reasons
    assert any("SMILES" in reason or reason for reason in rejected_reasons)
    rejected_duplicate = next(cm for cm in result.mols if cm.props.get(DROPPED_REASON) == "duplicate_structure")
    assert rejected_duplicate.props["IMPORT_ACCEPTED"] == 0
    assert "duplicate_structure" in str(rejected_duplicate.props.get(QC_FLAGS, ""))
