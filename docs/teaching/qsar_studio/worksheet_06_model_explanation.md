# Worksheet 06 — Model Explanation

## Context

QSAR models should not only predict; they should also be interpretable enough for chemical reasoning. This worksheet uses the Model Explanation widget to identify important descriptor and fingerprint features.

## Intended learning outcomes

After this activity, students can:

1. explain the difference between model performance and model interpretation,
2. rank descriptors/fingerprint bits by importance,
3. distinguish global and approximate local explanations,
4. connect important features to chemical hypotheses,
5. discuss limitations of feature importance.

## Data

Use:

```text
examples/qsar_studio/qsar_ad_explanation_demo.csv
```

## Orange workflow

```text
File → QSAR Model Hub → Model Explanation → Feature Importance → Data Table
                    └──────── Model ────────┘
```

Alternative quick workflow:

```text
File → Model Explanation → Feature Importance → Data Table
```

In the alternative workflow, the widget fits an internal fallback model for teaching.

## Tasks

1. Load the example data.
2. Train a QSAR model in QSAR Model Hub.
3. Connect the model and data to Model Explanation.
4. Inspect the top 10 features.
5. Compare the feature list with your chemical intuition.
6. Review local contributions for individual compounds.

## Questions

- Which descriptor or fingerprint feature is most important?
- Is the most important feature chemically plausible?
- Are registry/fingerprint bits easier or harder to interpret than physicochemical descriptors?
- Why can feature importance be misleading in correlated descriptor sets?

## Assessment evidence

Students submit a short interpretation of the top five features and one local explanation example.
