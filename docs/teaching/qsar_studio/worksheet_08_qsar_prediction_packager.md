# Worksheet 08 — QSAR Prediction Packager

## Learning goals

Students learn how to apply a trained QSAR model to an external prediction set and document feature compatibility.

## Orange workflow

```text
Training table → QSAR Model Hub → Model ─────┐
External table → QSAR Prediction Packager ───┘
```

## Tasks

1. Train a model on `qsar_model_hub_demo.csv`.
2. Load `qsar_prediction_query_demo.csv` as an external query table.
3. Apply the model with QSAR Prediction Packager.
4. Inspect the feature report and package manifest.
5. Send predictions to Applicability Domain Workbench when possible.

## Guiding questions

- Why must the external set contain the same descriptor/fingerprint columns?
- What can go wrong if feature matching is silent?
- Should out-of-domain external predictions be trusted?

## Evidence of learning

Students submit the prediction table, feature report and a short reliability assessment.
