# Worksheet 01 — What Is an Applicability Domain?

## Context

A QSAR model can produce a numerical prediction for almost any molecule, but that does not mean the prediction is reliable. Applicability domain analysis asks whether a new molecule is sufficiently similar to the reference/training molecules.

## Intended learning outcomes

Students will be able to:

1. Define applicability domain in their own words.
2. Explain the difference between prediction and reliable prediction.
3. Identify why extrapolation is risky in QSAR.

## Orange workflow

```text
File(reference) ─────────────┐
                             ↓ Reference Data
File(query) → Applicability Domain → Data Results → Data Table
                             ↓
                          AD Summary → Data Table
```

Use:

```text
examples/applicability_domain/ad_reference_training_set.csv
examples/applicability_domain/ad_query_prediction_set.csv
```

## Suggested settings

- Williams leverage: ON
- kNN distance: ON
- Mahalanobis distance: OFF
- Combine: `and`

## Student tasks

1. Load the reference dataset.
2. Load the query dataset.
3. Run the Applicability Domain widget.
4. Open `Data Results` in Data Table.
5. Sort by `AD_in_domain`.
6. Identify compounds outside the domain.

## Guiding questions

1. Which query compounds are inside the applicability domain?
2. Which are outside?
3. What descriptor values make the outside compounds unusual?
4. Would you report predictions for outside-domain compounds without warning?

## Expected observation

Large lipophilic or highly polar query molecules should be more likely to fall outside the reference descriptor space.

## Short reflection

Write 3–5 sentences explaining why applicability domain is part of responsible QSAR modeling.
