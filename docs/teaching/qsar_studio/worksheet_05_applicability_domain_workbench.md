# Worksheet 05 — Applicability Domain Workbench

## Context

A QSAR prediction is only meaningful when the query compound is reasonably similar to the chemical space used to train the model. This worksheet introduces descriptor-space applicability domain analysis.

## Intended learning outcomes

After this activity, students can:

1. define applicability domain in QSAR,
2. explain why external predictions require reliability checks,
3. compare Williams leverage, kNN distance, and Mahalanobis distance,
4. identify out-of-domain compounds,
5. report AD limitations in a QSAR study.

## Data

Use:

```text
examples/qsar_studio/qsar_ad_explanation_demo.csv
examples/qsar_studio/qsar_ad_query_demo.csv
```

## Orange workflow

```text
File(reference) ────────────────┐
                                ↓ Reference Data
File(query) → Applicability Domain Workbench → AD Results → Data Table
                                ↓
                         Out-of-Domain Records → Data Table
```

## Tasks

1. Load the reference data and query data.
2. Enable Williams leverage and kNN distance.
3. Run the workbench.
4. Open `Out-of-Domain Records`.
5. Identify which descriptors caused suspicious behavior.
6. Repeat with Mahalanobis distance enabled.

## Questions

- Which compounds are outside the domain?
- Do all AD methods flag the same compounds?
- Why might a high molecular weight or high logP compound be outside the domain?
- Why should AD be reported together with QSAR predictions?

## Assessment evidence

Students submit a short table containing compound ID, AD status, failed method, and interpretation.
