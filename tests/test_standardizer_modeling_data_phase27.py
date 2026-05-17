from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("Orange")

import numpy as np
from Orange.data import ContinuousVariable, Domain, StringVariable, Table

from chem_inf_widgets.widgets.ow_mol_standardizer import OWMolStandardizer


def _names(vars_):
    return [v.name for v in vars_]


def test_standardizer_modeling_table_uses_smiles_and_removes_audit_columns():
    domain = Domain(
        [ContinuousVariable("activity"), ContinuousVariable("descriptor_1")],
        metas=[
            StringVariable("SMILES"),
            StringVariable("Name"),
            StringVariable("qc_status"),
            StringVariable("standardization_status"),
            StringVariable("inchikey"),
        ],
    )
    table = Table.from_numpy(
        domain,
        X=np.asarray([[5.0, 1.1], [6.0, 2.2]], dtype=float),
        metas=np.asarray(
            [
                ["CCO.Cl", "ethanol_salt", "Clean", "old", "old-key"],
                ["CCN", "ethylamine", "Clean", "old", "old-key"],
            ],
            dtype=object,
        ),
    )

    modeling = OWMolStandardizer._modeling_table(table, [0, 1], ["CCO", "CCN"])
    all_names = _names(modeling.domain.attributes) + _names(modeling.domain.class_vars) + _names(modeling.domain.metas)

    assert "activity" in _names(modeling.domain.attributes)
    assert "Name" in _names(modeling.domain.metas)
    assert "SMILES" in _names(modeling.domain.metas)
    assert "inchikey" in _names(modeling.domain.metas)
    assert "qc_status" not in all_names
    assert "standardization_status" not in all_names
    assert "SMILES_STD" not in all_names
    assert list(modeling[:, "SMILES"].metas.ravel()) == ["CCO", "CCN"]
