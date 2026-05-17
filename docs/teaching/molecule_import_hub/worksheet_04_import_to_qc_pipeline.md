# Worksheet 04 — From Import to Quality Control

## Context

Import is not the same as quality control. A molecule may import correctly but still require review because it is a salt, mixture, charged compound or duplicate.

## Workflow

```text
Molecule Import Hub → Molecule QC Dashboard → QC Report → Data Table
```

## Tasks

1. Import `molecule_import_demo.csv`.
2. Connect **Molecules** output to **Molecule QC Dashboard**.
3. Run QC.
4. Inspect clean and problem outputs.

## Questions

1. Which molecules import successfully but still require QC review?
2. Why is sodium acetate not an import failure?
3. What is the difference between parse validity and dataset suitability?

## Expected evidence

A short comparison between import failures and QC warnings.
