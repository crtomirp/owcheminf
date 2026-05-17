from __future__ import annotations

from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
Orange = pytest.importorskip("Orange")

from chem_inf_widgets.chemcore.services.from_orange import chemmols_to_table, dataset_to_table
from chem_inf_widgets.chemcore.molecule_contract import ROW_ID, TRANSFORM_LOG, append_transform_step, ensure_contract_props
from chem_inf_widgets.chemcore.services.molecule_import_service import MoleculeImportConfig, import_molecule_file
from chem_inf_widgets.chemcore.services.qsar_dataset_builder_service import QSARDatasetBuilderConfig, build_qsar_dataset
from chem_inf_widgets.widgets.ow_qsar_dataset_builder import _table_from_records, _table_to_records
from chem_inf_widgets.chemcore.models.chembl_dataset import ChemBLDataset
from chem_inf_widgets.chemcore.mol import ChemMol


def _names(seq):
    return [v.name for v in seq]


def test_dataset_to_table_keeps_numeric_activity_as_attribute():
    ds = ChemBLDataset(
        mols=[ChemMol.from_smiles("CCO", name="ethanol")],
        props=[{"chembl_id": "CHEMBL1", "canonical_smiles": "CCO", "activity": "6.25", "source": "demo"}],
    )
    table = dataset_to_table(ds)
    assert "activity" in _names(table.domain.attributes)
    assert "activity" not in _names(table.domain.metas)
    assert "chembl_id" in _names(table.domain.metas)
    assert "canonical_smiles" in _names(table.domain.metas)
    assert float(table[0, "activity"]) == pytest.approx(6.25)


def test_import_to_qsar_builder_pactivity_becomes_class_var(tmp_path: Path):
    p = tmp_path / "pactivity.csv"
    p.write_text(
        "compound_id,name,smiles,pActivity,source\n"
        "M001,Aspirin,CC(=O)Oc1ccccc1C(=O)O,6.43,teaching\n"
        "M002,Pyridine,c1ccncc1,4.12,teaching\n",
        encoding="utf-8",
    )
    imported = import_molecule_file(p, MoleculeImportConfig(smiles_column="smiles", name_column="name"))
    table = chemmols_to_table(imported.mols)

    assert "pActivity" in _names(table.domain.attributes)
    assert "pActivity" not in _names(table.domain.metas)

    records = _table_to_records(table)
    result = build_qsar_dataset(
        records,
        QSARDatasetBuilderConfig(
            smiles_column="SMILES",
            name_column="compound_id",
            activity_column="pActivity",
            duplicate_key="canonical_smiles",
        ),
    )
    ready = _table_from_records(result.prepared_records, class_column="pActivity", name="QSAR Ready Data")

    assert ready is not None
    assert _names(ready.domain.class_vars) == ["pActivity"]
    assert "pActivity" not in _names(ready.domain.attributes)
    assert "pActivity" not in _names(ready.domain.metas)
    assert float(ready.Y[0]) == pytest.approx(6.43)


def test_contract_provenance_fields_roundtrip_as_metas():
    cm = ChemMol.from_smiles("CCO", name="ethanol")
    ensure_contract_props(cm, row_index=3, source_format="teaching", input_smiles="C(C)O")
    append_transform_step(cm, "import_csv")
    append_transform_step(cm, "standardize_qsar")

    table = chemmols_to_table([cm])

    assert ROW_ID in _names(table.domain.metas)
    assert TRANSFORM_LOG in _names(table.domain.metas)
    assert str(table[0, ROW_ID])
    assert str(table[0, TRANSFORM_LOG]) == "import_csv | standardize_qsar"
