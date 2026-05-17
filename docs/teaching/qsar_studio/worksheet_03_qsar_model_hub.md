# Worksheet 03 — Training a QSAR Model with QSAR Model Hub

## Learning outcomes

Students will train a regression model from molecular descriptors/fingerprints, inspect train/test/CV metrics, and export predictions for validation.

## Orange workflow

```text
File → QSAR Model Hub → Data Table
                   ↓
          QSAR Validation Dashboard
```

Use `examples/qsar_studio/qsar_model_hub_demo.csv`.

## Tasks

1. Load the example table.
2. Set the target column to `pActivity`.
3. Set the ID column to `compound_id`.
4. Train at least two models, for example `random_forest` and `ridge`.
5. Compare test RMSE and cross-validation RMSE.
6. Send the predictions output to `QSAR Validation Dashboard`.

## Questions

- Which model performs better on the test split?
- Is the difference large enough to trust?
- Why is cross-validation useful?
- Which feature source would you use next: descriptors, Morgan fingerprints, or cyclic registry fingerprints?

## Evidence for assessment

- Screenshot or export of the metrics table.
- Short paragraph interpreting train/test/CV metrics.
- Prediction table sent to validation.
