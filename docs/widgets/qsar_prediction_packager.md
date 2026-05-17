# QSAR Prediction Packager

**Category:** Cheminf - Modeling  
**Widget:** QSAR Prediction Packager

The QSAR Prediction Packager applies a trained QSAR model to external/query compounds and creates a prediction package with predictions, feature report, manifest and failed records.

## Inputs

- Model: trained model object from QSAR Model Hub.
- Query Data: descriptor/fingerprint table for external compounds.

## Outputs

- Predictions
- Feature Report
- Package Manifest
- Failed Records

## Typical workflow

```text
Training data → QSAR Model Hub → Model ─────┐
External descriptor table ──────────────────┤→ QSAR Prediction Packager → Predictions
```

## CLI

Train a model from a training table and predict a query table:

```bash
owcheminf-qsar-prediction-packager \
  examples/qsar_studio/qsar_prediction_query_demo.csv \
  --training-data examples/qsar_studio/qsar_model_hub_demo.csv \
  --target-column pActivity \
  --id-column compound_id \
  --model ridge \
  --out-prefix outputs/qsar_prediction_demo
```

Use `--save-model` to include a pickled model in the package.
