import numpy as np
import pandas as pd

from chem_inf_widgets.chemcore.services.descriptor_filter_service import (
    DescriptorFilterConfig,
    run_descriptor_filter,
)


def test_descriptor_filter_uses_pre_correlation_cap():
    rng = np.random.default_rng(42)
    n_rows = 40
    n_features = 120
    X = rng.normal(size=(n_rows, n_features))
    X[:, 0] = np.nan  # empty descriptor
    X[:, 1] = 1.0     # low variance descriptor
    X[:, 3] = X[:, 2] + rng.normal(scale=1e-4, size=n_rows)  # correlated pair

    df = pd.DataFrame(X, columns=[f"d{i}" for i in range(n_features)])
    df["SMILES"] = ["CCO"] * n_rows
    df["inchikey"] = [f"mol-{i}" for i in range(n_rows)]
    df["pActivity"] = rng.normal(size=n_rows)

    filtered, result = run_descriptor_filter(
        df,
        DescriptorFilterConfig(
            target_column="pActivity",
            max_missing_fraction=0.2,
            min_variance=0.001,
            max_correlation=0.90,
            max_features_before_correlation=25,
            max_correlation_features=10,
        ),
    )

    assert result.n_input == n_features
    assert "d0" in result.removed_empty
    assert "d1" in result.removed_low_variance
    assert len(result.removed_pre_correlation_cap) > 0
    assert result.n_after_pre_correlation_cap <= 25
    assert result.n_output <= 25
    assert "SMILES" in filtered.columns
    assert "inchikey" in filtered.columns
    assert "pActivity" in filtered.columns
    assert "pre_corr_cap" in set(result.report_df["status"])
