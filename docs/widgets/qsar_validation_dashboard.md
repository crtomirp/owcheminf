# QSAR Validation Dashboard

`QSAR Validation Dashboard` evaluates prediction tables from QSAR models.

It expects at least:

- observed values,
- predicted values,
- optionally a split column and compound identifier.

Outputs:

- **Validation Metrics**: R², RMSE, MAE, bias and residual statistics by split.
- **Residual Diagnostics**: row-level residuals and review flags.
- **Outlier Records**: records that should be inspected.
- **Validation Summary**: compact provenance and thresholds.

CLI:

```bash
owcheminf-qsar-validation-dashboard examples/qsar_studio/qsar_predictions_demo.csv \
  --out-prefix outputs/qsar_validation_demo
```
