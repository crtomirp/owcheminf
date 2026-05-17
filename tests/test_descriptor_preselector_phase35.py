import numpy as np
import pandas as pd

from chem_inf_widgets.chemcore.services.descriptor_filter_service import (
    DescriptorFilterConfig,
    run_descriptor_filter,
)


def test_descriptor_preselector_final_cap_keeps_identifier_and_target_columns():
    rng = np.random.default_rng(123)
    n_rows = 80
    n_features = 60
    df = pd.DataFrame(
        rng.normal(size=(n_rows, n_features)),
        columns=[f"d{i}" for i in range(n_features)],
    )
    df["empty_descriptor"] = np.nan
    df["constant_descriptor"] = 1.0
    df["copy_of_d0"] = df["d0"]
    df["SMILES"] = ["CCO"] * n_rows
    df["inchikey"] = [f"KEY{i:03d}" for i in range(n_rows)]
    df["pActivity"] = rng.normal(size=n_rows)

    filtered, result = run_descriptor_filter(
        df,
        DescriptorFilterConfig(
            target_column="pActivity",
            max_missing_fraction=0.2,
            min_variance=0.0,
            max_correlation=0.90,
            max_correlation_features=20,
            max_features_before_correlation=30,
            max_output_features=10,
        ),
    )

    descriptor_cols = [c for c in filtered.columns if c.startswith("d") or c == "copy_of_d0"]
    assert len(descriptor_cols) == 10
    assert result.n_output == 10
    assert result.removed_final_cap
    assert "SMILES" in filtered.columns
    assert "inchikey" in filtered.columns
    assert "pActivity" in filtered.columns
    assert "final_cap" in set(result.report_df["status"])
