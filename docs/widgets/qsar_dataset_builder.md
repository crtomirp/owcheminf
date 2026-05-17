# QSAR Dataset Builder

**Category:** Cheminf - Modeling  
**Widget:** `QSAR Dataset Builder`

The QSAR Dataset Builder converts raw or semi-curated bioactivity records into a
QSAR-ready table. It is intended to sit between ChEMBL/import/curation widgets
and descriptor/fingerprint/modeling widgets.

## Main tasks

- Detect or select SMILES, compound ID, activity, unit, relation and endpoint columns.
- Convert concentration values such as nM, µM and mM to `pActivity`.
- Reject non-exact activity relations by default, for example `>` or `<`.
- Filter by endpoint type, for example `IC50`, `Ki`, `Kd` or `EC50`.
- Aggregate duplicate compounds using median/mean/min/max/first.
- Produce a transparent curation report and dataset summary.

## Outputs

- **QSAR Ready Data**: curated compounds with `pActivity` as class variable.
- **Rejected Records**: records excluded during curation.
- **Curation Report**: row-level accepted/rejected status and reasons.
- **Dataset Summary**: one-row summary of curation settings and counts.

## Recommended workflow

```text
Molecule Import Hub
  → Molecule QC Dashboard
  → Mol Standardizer
  → QSAR Dataset Builder
  → Mol Descriptors 2 / Fingerprint Generator / Cyclic Registry Fingerprint
  → QSAR Regression
  → Applicability Domain
```

## Notes

The widget does not replace expert chemical curation. It makes common curation
decisions explicit, reproducible and easy to inspect.
