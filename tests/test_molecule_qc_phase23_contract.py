from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.molecule_contract import (
    DROPPED_REASON,
    QC_DUPLICATE_COUNT,
    QC_DUPLICATE_KEY,
    QC_FLAGS,
    QC_ISSUE_CODES,
    QC_ISSUES,
    QC_N_ISSUES,
    QC_SEVERITY,
    QC_STATUS,
    QC_VERSION_FIELD,
    TRANSFORM_LOG,
)
from chem_inf_widgets.chemcore.services.molecule_qc_service import (
    annotate_chemmols_with_qc,
    qc_partition_indices,
    run_molecule_qc,
)


def test_qc_annotations_are_written_to_chemmol_contract_fields():
    mols = [ChemMol.from_smiles("CCO"), ChemMol.from_smiles("CCO"), ChemMol.from_smiles("[Na+].[O-]C(=O)C")]
    result = run_molecule_qc(mols)
    annotated = annotate_chemmols_with_qc(mols, result.records)

    assert len(annotated) == 3
    assert annotated[0].props[QC_STATUS] == "Needs review"
    assert annotated[0].props[QC_SEVERITY] == "WARNING"
    assert "DUPLICATE_STRUCTURE" in annotated[0].props[QC_ISSUE_CODES]
    assert annotated[0].props[QC_N_ISSUES] >= 1
    assert annotated[0].props[QC_DUPLICATE_COUNT] == 2
    assert annotated[0].props[QC_DUPLICATE_KEY]
    assert annotated[0].props[QC_VERSION_FIELD]
    assert "molecule_qc" in annotated[0].props[TRANSFORM_LOG]
    assert "duplicate_structure" in annotated[0].props[QC_FLAGS]
    assert annotated[2].props[QC_ISSUES]
    assert "multi_fragment" in annotated[2].props[QC_FLAGS]


def test_qc_annotations_mark_rejected_rows_with_dropped_reason():
    mols = [ChemMol.from_smiles("CCO"), ChemMol.from_smiles("CCO")]
    result = run_molecule_qc(["", "CCO"])
    annotated = annotate_chemmols_with_qc(mols, result.records, copy_molecules=True)

    assert annotated[0].props[DROPPED_REASON] == "invalid_structure"
    assert "invalid_structure" in annotated[0].props[QC_FLAGS]


def test_qc_partitions_clean_problem_rejected_are_stable():
    result = run_molecule_qc(["CCO", "C[N+](C)(C)C", ""])
    partitions = qc_partition_indices(result)

    assert partitions["clean"] == [0]
    assert partitions["problem"] == [1]
    assert partitions["rejected"] == [2]
