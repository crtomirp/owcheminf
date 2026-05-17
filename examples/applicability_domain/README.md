# Applicability Domain Example Data

This folder contains small teaching datasets for the **Applicability Domain** widget.

Files:

- `ad_reference_training_set.csv` — a small reference/training-like set with SMILES, an illustrative `pIC50`, and numeric descriptor columns.
- `ad_query_prediction_set.csv` — a query/external-like set containing normal compounds and intentional outliers.

The descriptor columns are simplified teaching values intended for classroom workflows, not curated experimental descriptors.
For real QSAR projects, calculate descriptors with the package descriptor widgets before evaluating the applicability domain.

Recommended Orange workflow:

```text
File(reference) ─────────────┐
                             ↓ Reference Data
File(query) → Applicability Domain → Data Results → Data Table
                             ↓
                          AD Summary → Data Table
```

Use continuous descriptor columns such as `MW`, `LogP`, `TPSA`, `HBD`, `HBA`, `RotB`, and `AromaticRings` as the descriptor space.
