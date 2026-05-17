# Worksheet 3: Duplicates and Molecular Identity

## Context

Duplicate records can bias QSAR models and distort library statistics.

## Workflow

```text
File → Molecule QC Dashboard → QC Report → Data Table
```

## Tasks

1. Run QC using `canonical_smiles` as duplicate key.
2. Run QC using `inchikey` as duplicate key.
3. Compare duplicate counts.
4. Discuss when each key is appropriate.

## Questions

1. Why can duplicate records be problematic in modelling?
2. What is the difference between duplicate records and replicate measurements?
3. Should duplicate bioactivity measurements be removed or aggregated?
4. Which duplicate policy is most appropriate for ChEMBL data?

## Expected product

A short duplicate handling policy for a QSAR dataset.
