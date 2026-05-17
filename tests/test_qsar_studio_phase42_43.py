from __future__ import annotations

import pandas as pd

from chem_inf_widgets.chemcore.services.qsar_model_hub_service import (
    QSARModelHubConfig,
    available_model_keys,
    train_qsar_model_hub,
)
from chem_inf_widgets.chemcore.services.qsar_validation_dashboard_service import (
    QSARValidationConfig,
    validate_qsar_predictions,
)


def _demo_df():
    return pd.DataFrame(
        {
            "compound_id": [f"C{i:03d}" for i in range(12)],
            "pActivity": [5.1, 5.4, 5.6, 5.9, 6.0, 6.2, 6.4, 6.7, 6.9, 7.0, 7.2, 7.3],
            "MW": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
            "LogP": [1.0, 1.1, 1.3, 1.4, 1.8, 2.0, 2.1, 2.3, 2.5, 2.8, 3.0, 3.2],
            "TPSA": [10, 12, 15, 16, 19, 20, 22, 23, 25, 26, 29, 31],
        }
    )


def test_available_model_keys_contains_random_forest():
    assert "random_forest" in available_model_keys()


def test_qsar_model_hub_trains_and_outputs_predictions():
    result = train_qsar_model_hub(
        _demo_df(),
        QSARModelHubConfig(target_column="pActivity", id_column="compound_id", model_key="ridge", cv_folds=3),
    )
    assert result.n_rows_used == 12
    assert result.n_features_used >= 3
    assert {"observed", "predicted", "residual", "split"}.issubset(result.predictions.columns)
    assert not result.metrics_table.empty


def test_qsar_validation_dashboard_flags_large_residuals():
    df = pd.DataFrame(
        {
            "compound_id": ["A", "B", "C"],
            "split": ["test", "test", "test"],
            "observed": [5.0, 6.0, 7.0],
            "predicted": [5.1, 6.1, 4.0],
        }
    )
    result = validate_qsar_predictions(df, QSARValidationConfig(residual_threshold=1.0))
    assert len(result.metrics) >= 1
    assert len(result.outliers) == 1
    assert result.summary["n_outliers"] == 1
