# QSAR Descriptor Explorer

**Category:** Cheminf - QSAR  
**Widget:** `QSAR Descriptor Explorer`

The QSAR Descriptor Explorer inspects a descriptor matrix before model training.
It helps you detect missing-value problems, constant or near-constant features,
redundant highly correlated descriptors, and broad descriptor family balance.

## Main tasks

- Detect numeric descriptor columns while excluding common ID, structure and target columns.
- Summarize descriptor missingness, variance, range and category.
- Flag descriptors with high missing-value rate or near-zero variance.
- Detect highly correlated descriptor pairs and suggest a reduced descriptor set.
- Produce HTML and Markdown reports for downstream QSAR documentation.

## Outputs

- **Filtered Data**: input table with recommended descriptor removals applied.
- **Descriptor Summary**: one row per descriptor with missingness, variance, summary statistics and flags.
- **Category Summary**: grouped overview by descriptor family.
- **Correlation Pairs**: descriptor pairs above the configured correlation threshold.
- **Quality Report**: compact metrics table for rows, candidates, filtered descriptors and redundancy counts.
- **Report HTML** and **Report Markdown**: reusable text reports for documentation or report assembly.

## Recommended workflow

```text
QSAR Dataset Builder
  → Mol Descriptors / Fingerprint Generator
  → QSAR Descriptor Explorer
  → Descriptor Pre-selector
  → QSAR/QSPR Model Hub / QSAR Regression / MLR Model Selection
```

## Notes

This widget is intended for descriptor-matrix triage, not final supervised feature
selection. A common pattern is to use it first for transparent quality control,
then pass the filtered output to `Descriptor Pre-selector` or directly to a model
trainer when the descriptor set is already compact.
