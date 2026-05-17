# Worksheet 07: External Prediction Set

**Estimated time:** 60–90 min  
**Level:** intermediate  
**Main widgets:** QSAR Regression, Data Table

## Context

A useful QSAR workflow should be able to make predictions for new compounds, while clearly communicating uncertainty and domain limits.

## Intended learning outcomes

Students will be able to:

1. prepare an external prediction set,
2. apply the same descriptor/fingerprint workflow to new molecules,
3. interpret external predictions cautiously,
4. identify molecules outside the likely training domain.

## Input data

Training file:

```text
examples/qsar_widgets/qsar_training_set.csv
```

External file:

```text
examples/qsar_widgets/qsar_external_prediction_set.csv
```

## Orange workflow

```text
Training File → Mol Descriptors 2 → QSAR Regression
External File → Mol Descriptors 2 → QSAR Regression External Data input
QSAR Regression → External Results → Data Table
```

If using fingerprints, ensure both training and external data use the same fingerprint settings.

## Student tasks

1. Train a model on the training dataset.
2. Load the external prediction dataset.
3. Apply the same descriptor or fingerprint calculation.
4. Generate predictions.
5. Mark which external molecules are chemically similar to the training set.
6. Write a cautionary prediction statement.

## Guiding questions

- Why must training and external data use the same descriptor settings?
- Which external compounds are most similar to the training set?
- Which predictions are least trustworthy?
- How would you define the model domain of applicability?

## Expected output

A table of external predictions with compound names and predicted values.
