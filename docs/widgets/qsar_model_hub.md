# QSAR Model Hub

`QSAR Model Hub` trains a compact regression model from a QSAR-ready descriptor or fingerprint table.
The `Model` output is a prediction-ready bundle, and the widget can also export a FAIR-style model package
with the pickle, manifest, full feature contract and selected-feature list for later reuse.

Recommended workflow:

```text
QSAR Dataset Builder → Mol Descriptors / Fingerprint Generator → QSAR Model Hub → QSAR Validation Dashboard
```

Outputs:

- **Model**: prediction-ready `QSARPredictionModelBundle` for direct use in `QSAR Prediction Packager`.
- **Predictions**: train/test predictions with observed, predicted, residual and split columns.
- **Metrics**: train, test and cross-validation metrics.
- **Model Summary**: compact provenance summary.

Export:

- Use `Export FAIR model bundle` after training to write:
- `*.model.pkl`
- `*.manifest.json`
- `*.features.txt`
- `*.selected_features.txt`

CLI:

```bash
owcheminf-qsar-model-hub examples/qsar_studio/qsar_model_hub_demo.csv \
  --target-column pActivity \
  --id-column compound_id \
  --model random_forest \
  --out-prefix outputs/qsar_model_demo
```
