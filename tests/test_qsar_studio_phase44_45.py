from __future__ import annotations

import numpy as np
import pandas as pd

from chem_inf_widgets.chemcore.services.ad_workbench_service import (
    ADWorkbenchConfig,
    evaluate_applicability_domain_workbench,
)
from chem_inf_widgets.chemcore.services.model_explanation_service import (
    ModelExplanationConfig,
    explain_qsar_model,
)


def _demo_df(n=24):
    rng = np.random.default_rng(11)
    x1 = np.linspace(0, 1, n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 5 + 1.2 * x1 - 0.4 * x2 + rng.normal(0, 0.05, size=n)
    return pd.DataFrame({
        "compound_id": [f"C{i:03d}" for i in range(n)],
        "pActivity": y,
        "desc_1": x1,
        "desc_2": x2,
        "desc_3": x3,
    })


def test_ad_workbench_scores_query_and_flags_outlier():
    ref = _demo_df(20)
    query = _demo_df(5)
    query.loc[4, "desc_1"] = 10.0
    result = evaluate_applicability_domain_workbench(ref, query, ADWorkbenchConfig(use_knn=True, use_williams=True))
    assert len(result.scored_query) == 5
    assert "AD_in_domain" in result.scored_query.columns
    assert len(result.feature_names) == 3
    assert len(result.out_of_domain) >= 1


def test_model_explanation_returns_feature_importance():
    df = _demo_df(24)
    result = explain_qsar_model(df, model=None, config=ModelExplanationConfig(max_features=10))
    assert len(result.feature_importance) > 0
    assert "feature" in result.feature_importance.columns
    assert result.summary_dict["n_features_used"] == 3
    assert len(result.local_contributions) == len(df)
