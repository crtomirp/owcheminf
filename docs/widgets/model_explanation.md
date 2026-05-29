# Model Explanation

**Category:** Cheminf - QSAR  
**Widget:** Model Explanation

The Model Explanation widget ranks descriptor and fingerprint features that influence QSAR models. It can use a model supplied by QSAR Model Hub or fit an internal fallback model for teaching and rapid interpretation.

## Inputs

- **Data**: descriptor/fingerprint table with a target column.
- **Model**: optional trained model, for example from QSAR Model Hub.

## Outputs

- **Feature Importance**: ranked feature table.
- **Local Contributions**: approximate local feature contributions per compound.
- **Feature Summary**: compact feature/method summary.
- **Explanation Summary**: reproducibility metadata.

## Methods

- model importances when available,
- absolute coefficients for linear models,
- permutation importance,
- univariate correlation fallback.

If no model is connected, the widget fits an internal fallback regressor so the
explanation workflow can still be used for teaching, quick audits and descriptor
triage.

## Recommended workflow

```text
QSAR Dataset Builder → Descriptors/Fingerprints → QSAR/QSPR Model Hub → Model Explanation
```

For cyclic registry fingerprints, connect **Cyclic Registry Fingerprint → Matched Registry Entries** alongside Model Explanation to interpret important bits chemically.

## CLI

```bash
owcheminf-model-explanation \
  examples/qsar_studio/qsar_ad_explanation_demo.csv \
  --target-column pActivity \
  --id-column compound_id \
  --out-prefix outputs/model_explanation_demo
```
