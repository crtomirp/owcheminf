# Worksheet 04 — Validating QSAR Predictions

## Learning outcomes

Students will evaluate residuals, identify problematic predictions, and explain why model validation is separate from model training.

## Orange workflow

```text
File → QSAR Validation Dashboard → Validation Metrics
                              ↓
                         Outlier Records
```

Use `examples/qsar_studio/qsar_predictions_demo.csv` or predictions produced by `QSAR Model Hub`.

## Tasks

1. Load the prediction table.
2. Confirm the observed column is `observed` and the predicted column is `predicted`.
3. Run the dashboard.
4. Inspect R², RMSE, MAE and bias.
5. Inspect the outlier/review table.
6. Decide which compounds require chemical review.

## Questions

- Which split has worse residuals?
- Are the outliers chemically meaningful or just statistical?
- How would an applicability domain widget complement this dashboard?
- What should be reported in a publication?

## Evidence for assessment

- Validation metrics table.
- List of review compounds.
- Short interpretation of residuals and limitations.
