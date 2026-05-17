from chem_inf_widgets.chemcore.services.qsar_dataset_builder_service import (
    QSARDatasetBuilderConfig,
    build_qsar_dataset,
)


def demo_records():
    return [
        {"compound_id": "A", "canonical_smiles": "c1ccncc1", "standard_type": "IC50", "standard_relation": "=", "standard_value": 100, "standard_units": "nM"},
        {"compound_id": "A2", "canonical_smiles": "C1=CC=NC=C1", "standard_type": "IC50", "standard_relation": "=", "standard_value": 200, "standard_units": "nM"},
        {"compound_id": "B", "canonical_smiles": "c1ccccc1", "standard_type": "IC50", "standard_relation": ">", "standard_value": 10000, "standard_units": "nM"},
        {"compound_id": "C", "canonical_smiles": "bad", "standard_type": "IC50", "standard_relation": "=", "standard_value": 50, "standard_units": "nM"},
        {"compound_id": "D", "canonical_smiles": "c1ccoc1", "standard_type": "Ki", "standard_relation": "=", "standard_value": 0.5, "standard_units": "uM"},
    ]


def test_qsar_dataset_builder_filters_and_aggregates():
    result = build_qsar_dataset(
        demo_records(),
        QSARDatasetBuilderConfig(target_endpoint="IC50", duplicate_key="canonical_smiles"),
    )
    assert result.summary["input_records"] == 5
    assert result.summary["prepared_compounds"] == 1
    assert result.summary["duplicate_groups"] == 1
    assert result.summary["rejected_records"] == 3
    row = result.prepared_records[0]
    assert row["n_measurements"] == 2
    assert 6.6 < row["pActivity"] < 7.1


def test_qsar_dataset_builder_allows_inequalities_when_requested():
    result = build_qsar_dataset(
        demo_records(),
        QSARDatasetBuilderConfig(target_endpoint="IC50", relation_policy="allow_inequalities", duplicate_key="canonical_smiles"),
    )
    assert result.summary["prepared_compounds"] == 2


def test_qsar_dataset_builder_unit_conversion_um():
    result = build_qsar_dataset(
        demo_records(),
        QSARDatasetBuilderConfig(target_endpoint="Ki", duplicate_key="canonical_smiles"),
    )
    assert result.summary["prepared_compounds"] == 1
    assert abs(result.prepared_records[0]["pActivity"] - 6.3010) < 0.02
