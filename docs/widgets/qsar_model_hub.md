# QSAR Model Hub

`QSAR Model Hub` trains a compact regression model from a QSAR-ready descriptor or fingerprint table.

Recommended workflow:

```text
QSAR Dataset Builder → Mol Descriptors / Fingerprint Generator → QSAR Model Hub → QSAR Validation Dashboard
```

Outputs:

- **Model**: fitted scikit-learn pipeline.
- **Predictions**: train/test predictions with observed, predicted, residual and split columns.
- **Metrics**: train, test and cross-validation metrics.
- **Model Summary**: compact provenance summary.

CLI:

```bash
owcheminf-qsar-model-hub examples/qsar_studio/qsar_model_hub_demo.csv \
  --target-column pActivity \
  --id-column compound_id \
  --model random_forest \
  --out-prefix outputs/qsar_model_demo
```
