# Worksheet 03 — Using Import Reports and Failed Records

## Context

Silent failure is dangerous. If invalid molecules disappear without a report, QSAR and library analyses become biased.

## Workflow

```text
Molecule Import Hub → Import Report → Data Table
Molecule Import Hub → Failed Records → Data Table
```

## Tasks

1. Import the demonstration CSV.
2. Sort the import report by `ok` or `status`.
3. Open the failed records output.
4. Write a short explanation for each failed record.

## Questions

1. Should failed molecules be removed, corrected or kept for reporting?
2. How would failed import records affect QSAR model reproducibility?
3. What additional metadata would you include in a publication?

## Expected output

A short import quality statement, for example:

> The dataset contained 6 records. Five were imported successfully. One record failed due to an invalid ring closure and was excluded from downstream molecular analysis but retained in the import report.
