# Worksheet 10 — Capstone Project: QSAR with Applicability Domain

## Project goal

Build a small QSAR/QSPR workflow and include an applicability domain analysis for external predictions.

## Intended learning outcomes

Students will be able to:

1. Prepare descriptor data for QSAR.
2. Train or compare QSAR models.
3. Evaluate external compounds with Applicability Domain.
4. Produce a reproducible mini-report.

## Suggested Orange workflow

```text
File(training) → Mol Descriptors 2 → QSAR Regression
File(query)    → Mol Descriptors 2 → Applicability Domain
                                     ↑
File(training descriptors) ──────────┘ Reference Data
```

Optional interpretation:

```text
File → Cyclic Registry Fingerprint → Matched Registry Entries
```

## Required report sections

1. Dataset description
2. Descriptor preparation
3. QSAR model and validation
4. Applicability domain method
5. External prediction reliability
6. Limitations and next steps

## Minimum outputs

- Model performance table
- AD Summary table
- Table of external compounds with AD status
- Short discussion of outside-domain compounds

## Rubric

| Criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Data preparation | Missing | Partly correct | Mostly correct | Clean and reproducible |
| QSAR workflow | Missing | Basic | Correct | Well justified |
| AD analysis | Missing | Superficial | Correct | Carefully interpreted |
| Outlier discussion | Missing | Minimal | Reasonable | Chemically insightful |
| Reporting | Unclear | Basic | Clear | Publication-style |

Total: 15 points.

## Final reflection question

How would your conclusions change if the best-predicted compounds were outside the applicability domain?
