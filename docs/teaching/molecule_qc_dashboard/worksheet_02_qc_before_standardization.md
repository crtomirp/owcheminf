# Worksheet 2: QC Before Standardization

## Context

Standardization should not be treated as a magic repair button. First inspect the records and decide what should be standardized, rejected, or manually reviewed.

## Workflow

```text
File → Molecule QC Dashboard → Clean Data → Mol Standardizer → Data Table
                         ↓
                    Problem Data → Data Table
```

## Tasks

1. Run Molecule QC Dashboard.
2. Send Clean Data to Mol Standardizer.
3. Inspect Problem Data separately.
4. Decide whether charged and salt records should be kept or curated.

## Questions

1. Why should invalid structures not be sent blindly to standardization?
2. Which warnings are chemically meaningful but not necessarily fatal?
3. Which records should be manually corrected?
4. How would your decision change for docking versus QSAR?

## Expected product

A small table with three decisions: keep, standardize, manually review/remove.
