# Worksheet 05 — External Prediction Reliability

## Context

External prediction is one of the most important use cases for applicability domain. A model may predict new compounds, but AD analysis helps decide whether those predictions should be accepted, flagged, or rejected.

## Intended learning outcomes

Students will be able to:

1. Use a reference training set and separate query set.
2. Flag outside-domain predictions.
3. Write a reliability statement for QSAR predictions.

## Orange workflow

```text
File(reference training set) ─────────────┐
                                          ↓ Reference Data
File(external prediction set) → Applicability Domain → Data Results → Data Table
```

Optional modeling workflow:

```text
File(reference) → Descriptors → QSAR Regression
File(query)     → Descriptors → Predictions → Applicability Domain
```

## Student tasks

1. Evaluate the query set against the reference set.
2. Add a new column in your notes: `Prediction decision`.
3. Classify each query compound as:
   - Accept prediction,
   - Accept with caution,
   - Do not trust prediction.
4. Justify each decision using AD output columns.

## Guiding questions

1. Which query compounds are clearly supported by the training data?
2. Which compounds are extrapolations?
3. Would you include outside-domain predictions in a publication table?
4. How would you mark them?

## Expected reporting phrase

Example:

```text
Predictions were interpreted only for compounds within the applicability domain. Compounds outside the domain were flagged as extrapolations and excluded from model-based ranking.
```
