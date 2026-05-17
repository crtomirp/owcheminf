from __future__ import annotations

import pandas as pd

from chem_inf_widgets.chemcore.services.qsar_model_hub_service import QSARModelHubConfig, train_qsar_model_hub
from chem_inf_widgets.chemcore.services.qsar_prediction_packager_service import QSARPredictionPackagerConfig, predict_with_qsar_model
from chem_inf_widgets.chemcore.services.qsar_report_generator_service import QSARReportConfig, generate_qsar_report


def _demo_df():
    return pd.DataFrame(
        {
            "compound_id": [f"C{i}" for i in range(10)],
            "pActivity": [5.0, 5.2, 5.1, 5.7, 5.8, 6.0, 6.1, 6.3, 6.2, 6.5],
            "MW": [100, 110, 105, 130, 135, 150, 155, 170, 168, 180],
            "LogP": [1.0, 1.2, 1.1, 1.8, 1.9, 2.0, 2.1, 2.4, 2.3, 2.5],
            "fp_001": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        }
    )


def test_qsar_report_generator_creates_markdown_sections():
    df = _demo_df()
    metrics = pd.DataFrame({"metric": ["r2", "rmse"], "value": [0.7, 0.3]})
    report = generate_qsar_report(dataset=df, metrics=metrics, predictions=None, config=QSARReportConfig(title="Demo"))
    assert "# Demo" in report.markdown
    assert report.summary["dataset_rows"] == 10
    assert "Executive summary" in set(report.sections["section"])


def test_qsar_prediction_packager_predicts_query_rows():
    df = _demo_df()
    model = train_qsar_model_hub(df, QSARModelHubConfig(target_column="pActivity", id_column="compound_id", model_key="ridge")).pipeline
    query = df.drop(columns=["pActivity"]).head(3).copy()
    result = predict_with_qsar_model(model, query, QSARPredictionPackagerConfig(id_column="compound_id"))
    assert len(result.predictions) == 3
    assert "predicted_pActivity" in result.predictions.columns
    assert result.package_manifest["features_used"] >= 1
