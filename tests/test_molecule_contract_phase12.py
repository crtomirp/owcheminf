from __future__ import annotations

import numpy as np

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.molecule_contract import (
    CANONICAL_SMILES,
    DROPPED_REASON,
    INCHIKEY,
    INPUT_SMILES,
    MOL_ID,
    QC_FLAGS,
    ROW_ID,
    STANDARDIZATION_STATUS,
    STANDARDIZED_SMILES,
    TRANSFORM_LOG,
    append_qc_flag,
    append_transform_step,
    ensure_contract_props,
    set_dropped_reason,
)
from chem_inf_widgets.chemcore.services.mol_standardizer import MolStandardizer
from chem_inf_widgets.chemcore.services.safe_feature_selection import safe_f_regression


def test_molecule_contract_props_are_filled():
    cm = ChemMol.from_smiles("CCO", name="ethanol")
    ensure_contract_props(cm, row_index=1, source_format="unit-test", input_smiles="CCO")

    assert cm.props[MOL_ID] == "ethanol"
    assert cm.props[INPUT_SMILES] == "CCO"
    assert cm.props[CANONICAL_SMILES] == "CCO"
    assert cm.props[INCHIKEY]
    assert cm.props[ROW_ID]


def test_transform_log_appends_without_duplicate_last_step():
    cm = ChemMol.from_smiles("CCO", name="ethanol")
    ensure_contract_props(cm, row_index=1, source_format="unit-test", input_smiles="CCO")

    append_transform_step(cm, "import_csv")
    append_transform_step(cm, "import_csv")
    append_transform_step(cm, "table_to_chemmols")

    assert cm.props[TRANSFORM_LOG] == "import_csv | table_to_chemmols"


def test_qc_flags_and_dropped_reason_share_contract_helpers():
    cm = ChemMol.from_smiles("CCO", name="ethanol")
    ensure_contract_props(cm, row_index=1, source_format="unit-test", input_smiles="CCO")

    append_qc_flag(cm, "duplicate_structure")
    append_qc_flag(cm, "duplicate_structure")
    append_qc_flag(cm, "salt_detected")
    set_dropped_reason(cm, "duplicate_structure")

    assert cm.props[QC_FLAGS] == "duplicate_structure | salt_detected"
    assert cm.props[DROPPED_REASON] == "duplicate_structure"


def test_standardizer_writes_audit_contract_fields():
    cm = ChemMol.from_smiles("CCO.Cl", name="salt")
    ensure_contract_props(cm, row_index=1, input_smiles="CCO.Cl")

    mols, results = MolStandardizer().standardize_chemmols([cm])

    assert results[0].ok
    assert mols[0].props[STANDARDIZATION_STATUS] == "ok"
    assert mols[0].props[STANDARDIZED_SMILES] == "CCO"
    assert mols[0].props["STD_LOG"]


def test_safe_f_regression_handles_constant_columns_without_nan_scores():
    X = np.asarray([[1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.0]])
    y = np.asarray([1.0, 2.0, 3.0])

    scores, pvalues = safe_f_regression(X, y)

    assert np.isfinite(scores).all()
    assert np.isfinite(pvalues).all()
    assert scores[0] == 0.0
    assert scores[2] == 0.0
