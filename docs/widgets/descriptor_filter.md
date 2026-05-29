# Descriptor Pre-selector

**Category:** Cheminf - QSAR  
**Widget:** `Descriptor Pre-selector`

The Descriptor Pre-selector removes uninformative descriptors before model
training. It focuses on three practical QSAR filters: missing-value rate,
low variance and high inter-feature correlation.

## Main tasks

- Drop descriptors above a configurable missing-value threshold.
- Remove constant or near-constant descriptors.
- Collapse highly correlated descriptor sets with Pearson or Spearman correlation.
- Emit a modeling-clean table suitable for direct use in QSAR model widgets.
- Show a richer dashboard-style report with filter cascade, quality flags and recommended next steps.

## Outputs

- **Filtered Data**: filtered table with descriptor and metadata columns preserved.
- **Modeling Data**: modeling-oriented table intended for direct use in downstream QSAR widgets.
- **Filter Report**: tabular summary of thresholds, counts and removed descriptors.

## Recommended workflow

```text
QSAR Dataset Builder
  → Mol Descriptors / Fingerprint Generator
  → QSAR Descriptor Explorer
  → Descriptor Pre-selector
  → QSAR/QSPR Model Hub / QSAR Regression / MLR Model Selection
```

## Notes

`QSAR Descriptor Explorer` and `Descriptor Pre-selector` are complementary. The
Explorer is better for transparent matrix diagnostics and category summaries,
while the Pre-selector is the stronger handoff point for building a compact
modeling table.
