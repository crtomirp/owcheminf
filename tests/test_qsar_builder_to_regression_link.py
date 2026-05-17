from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("Orange")

from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table
from chem_inf_widgets.chemcore.services.qsar_dataset_builder_service import QSARDatasetBuilderConfig, build_qsar_dataset
from chem_inf_widgets.chemcore.services.qsar_regression_service import (
    available_algorithms,
    build_run_config,
    prepare_qsar_model_matrix,
    run_qsar_regression,
)


def _records():
    return [
        {"compound_id": "M001", "smiles": "CCO", "pActivity": 4.2},
        {"compound_id": "M002", "smiles": "CCN", "pActivity": 4.6},
        {"compound_id": "M003", "smiles": "CCC", "pActivity": 4.9},
        {"compound_id": "M004", "smiles": "CCCl", "pActivity": 5.1},
        {"compound_id": "M005", "smiles": "c1ccccc1", "pActivity": 5.8},
        {"compound_id": "M006", "smiles": "c1ccncc1", "pActivity": 6.0},
    ]


def _ready_table():
    result = build_qsar_dataset(
        _records(),
        QSARDatasetBuilderConfig(
            smiles_column="smiles",
            name_column="compound_id",
            activity_column="pActivity",
            duplicate_key="canonical_smiles",
        ),
    )
    return records_to_orange_table(result.prepared_records, class_column="pActivity", name="QSAR Ready Data")


def test_qsar_dataset_builder_output_prepares_auto_descriptors_for_regression():
    table = _ready_table()
    prepared = prepare_qsar_model_matrix(table)

    assert prepared["target_var"].name == "pActivity"
    assert prepared["generated_descriptors"] is True
    assert prepared["X"].shape[0] == len(table)
    assert prepared["X"].shape[1] >= 8
    assert "pActivity_raw" not in prepared["feature_names"]
    assert "activity_value" not in prepared["feature_names"]


def test_qsar_regression_runs_on_qsar_dataset_builder_output():
    table = _ready_table()
    config = build_run_config(
        selected_algorithm=0,
        normalization_method=0,
        imputation_method=1,
        cv_folds=2,
        test_size=0.34,
        tuning_method=0,
        n_iter=5,
        hyperparameters="",
        enable_feature_selection=False,
        num_features=10,
        algorithms=available_algorithms(),
    )

    result = run_qsar_regression(table, None, config)

    assert result["target_column"] == "pActivity"
    assert result["generated_descriptors"] is True
    assert result["train_table"].domain.class_var.name == "pActivity"
    assert "Predicted" in [var.name for var in result["test_table"].domain.attributes]
