# Applicability Domain

## Status

Current widget in the package.

Source:
- [ow_applicability_domain.py](../../src/chem_inf_widgets/widgets/ow_applicability_domain.py)
- [applicability_domain_service.py](../../src/chem_inf_widgets/chemcore/services/applicability_domain_service.py)

## Purpose

`Applicability Domain` estimates whether compounds lie inside the descriptor-space region that is reasonably covered by a reference set.

This is especially useful after QSAR modeling, when you want to know whether a prediction should be trusted.

## Input

- `Data`
- optional `Reference Data`

If `Reference Data` is not provided, the widget uses `Data` as both reference and query set.

## Output

- `Data Results`
- `Reference Results`
- `AD Summary`

The result tables append AD-related columns to the input data, such as leverage, kNN distance, Mahalanobis distance and final in-domain flags.

## Supported methods

- Williams leverage
- kNN distance
- Mahalanobis distance
- logical combination through `and` or `or`

## Typical workflow

1. `Mol Standardizer`
2. descriptor widget such as `Fingerprint Generator`, `Mol Descriptor` or `PaDEL Descriptors`
3. `QSAR Regression`
4. `Applicability Domain`

## Notes

- The widget only works on shared continuous descriptor columns present in both reference and query tables.
- For meaningful results, keep descriptor-generation settings identical across both branches.
