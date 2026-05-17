from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("Orange")

import numpy as np
from Orange.data import ContinuousVariable, Domain, StringVariable, Table

from chem_inf_widgets.chemcore.services.from_orange import table_to_chemmols_with_report
from chem_inf_widgets.chemcore.services.molecule_qc_service import annotate_chemmols_with_qc, qc_partition_indices, run_molecule_qc
from chem_inf_widgets.widgets.ow_molecule_qc_dashboard import OWMoleculeQCDashboard


def _names(vars_):
    return [var.name for var in vars_]


def test_qc_modeling_table_is_clean_and_slim():
    domain = Domain(
        [ContinuousVariable("activity"), ContinuousVariable("descriptor_1")],
        metas=[
            StringVariable("SMILES"),
            StringVariable("Name"),
            StringVariable("qc_status"),
            StringVariable("inchikey"),
        ],
    )
    table = Table.from_numpy(
        domain,
        X=np.asarray([[5.0, 1.1], [6.0, 2.2], [7.0, 3.3]], dtype=float),
        metas=np.asarray(
            [
                ["CCO", "ethanol", "old-qc", "old-key"],
                ["C[N+](C)(C)C", "charged", "old-qc", "old-key"],
                ["CCN", "ethylamine", "old-qc", "old-key"],
            ],
            dtype=object,
        ),
    )

    mols, _ = table_to_chemmols_with_report(table)
    result = run_molecule_qc(mols)
    partitions = qc_partition_indices(result)
    modeling = OWMoleculeQCDashboard._modeling_table(table, result.records, mols, partitions["clean"])

    all_names = _names(modeling.domain.attributes) + _names(modeling.domain.class_vars) + _names(modeling.domain.metas)
    assert "SMILES" in _names(modeling.domain.metas)
    assert "qc_status" not in all_names
    assert "inchikey" in _names(modeling.domain.metas)
    assert "Name" in _names(modeling.domain.metas)
    assert "activity" in _names(modeling.domain.attributes)
    assert len(modeling) == 2
    assert list(modeling[:, "SMILES"].metas.ravel()) == ["CCO", "CCN"]


def test_qc_annotated_table_contains_shared_audit_columns():
    domain = Domain(
        [ContinuousVariable("activity")],
        metas=[StringVariable("SMILES"), StringVariable("Name")],
    )
    table = Table.from_numpy(
        domain,
        X=np.asarray([[5.0], [6.0], [7.0]], dtype=float),
        metas=np.asarray(
            [
                ["CCO", "ethanol_a"],
                ["OCC", "ethanol_b"],
                ["C1CC", "broken"],
            ],
            dtype=object,
        ),
    )

    mols, conversion_report = table_to_chemmols_with_report(table)
    result = run_molecule_qc(mols)
    annotated_mols = annotate_chemmols_with_qc(mols, result.records)
    annotated = OWMoleculeQCDashboard._annotated_table(
        table,
        result.records,
        annotated_mols,
        [max(0, int(i) - 1) for i in conversion_report.skipped_rows],
        conversion_report.errors,
    )

    meta_names = _names(annotated.domain.metas)
    assert "row_id" in meta_names
    assert "transform_log" in meta_names
    assert "qc_flags" in meta_names
    assert "dropped_reason" in meta_names
    assert str(annotated[0, "qc_flags"]) == "duplicate_structure"
    assert str(annotated[2, "dropped_reason"]) == "invalid_structure"


def test_qc_annotated_table_handles_read_only_input_arrays():
    domain = Domain(
        [ContinuousVariable("activity")],
        metas=[StringVariable("SMILES"), StringVariable("Name")],
    )
    X = np.asarray([[5.0], [6.0], [7.0]], dtype=float)
    metas = np.asarray(
        [
            ["CCO", "ethanol_a"],
            ["OCC", "ethanol_b"],
            ["C1CC", "broken"],
        ],
        dtype=object,
    )
    X.setflags(write=False)
    metas.setflags(write=False)
    table = Table.from_numpy(domain, X=X, metas=metas)

    mols, conversion_report = table_to_chemmols_with_report(table)
    result = run_molecule_qc(mols)
    annotated_mols = annotate_chemmols_with_qc(mols, result.records)
    annotated = OWMoleculeQCDashboard._annotated_table(
        table,
        result.records,
        annotated_mols,
        [max(0, int(i) - 1) for i in conversion_report.skipped_rows],
        conversion_report.errors,
    )

    assert len(annotated) == 3
    assert str(annotated[0, "qc_status"]) == "Needs review"
    assert str(annotated[2, "dropped_reason"]) == "invalid_structure"
