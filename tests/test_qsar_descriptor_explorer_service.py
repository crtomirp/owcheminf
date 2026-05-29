import numpy as np
import pandas as pd

from chem_inf_widgets.chemcore.services.qsar_descriptor_explorer_service import (
    QSARDescriptorExplorerConfig,
    explore_qsar_descriptors,
    infer_descriptor_category,
)


def test_descriptor_explorer_flags_missing_low_variance_and_correlation():
    df = pd.DataFrame(
        {
            "compound_id": ["a", "b", "c", "d", "e"],
            "smiles": ["C", "CC", "CCC", "CCCC", "CCCCC"],
            "pActivity": [5, 6, 7, 8, 9],
            "MolWt": [16, 30, 44, 58, 72],
            "ExactMolWt": [16.1, 30.1, 44.1, 58.1, 72.1],
            "const_desc": [1, 1, 1, 1, 1],
            "bad_missing": [1.0, np.nan, np.nan, np.nan, 5.0],
            "MolLogP": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    result = explore_qsar_descriptors(
        df,
        QSARDescriptorExplorerConfig(
            target_column="pActivity",
            missing_threshold=0.4,
            high_correlation_threshold=0.99,
        ),
    )

    summary = result.descriptor_summary.set_index("descriptor")
    assert "low_variance" in summary.loc["const_desc", "status"]
    assert "high_missing" in summary.loc["bad_missing", "status"]
    assert not result.correlation_pairs.empty
    assert "ExactMolWt" not in result.filtered_data.columns or "MolWt" not in result.filtered_data.columns
    assert "pActivity" in result.filtered_data.columns
    assert "Descriptor Explorer Report" in result.markdown_report


def test_descriptor_category_inference():
    assert infer_descriptor_category("MolLogP") == "physicochemical"
    assert infer_descriptor_category("fr_Ar_N") == "fragment/count"
    assert infer_descriptor_category("Morgan_bit_001") == "fingerprint"
