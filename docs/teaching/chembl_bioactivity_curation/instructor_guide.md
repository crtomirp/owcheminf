# Instructor Guide: ChEMBL Bioactivity Curation

## Target audience

This package is suitable for undergraduate or MSc-level chemistry, pharmacy, cheminformatics, medicinal chemistry, and computational drug design courses.

## Suggested duration

- Short version: 2–3 hours, using only Worksheets 01–05.
- Full practical: 6–8 hours, using Worksheets 01–09.
- Project version: 1–2 weeks, ending with Worksheet 10.

## Learning outcomes

After completing the package, students should be able to:

1. Explain the difference between a target, assay, compound, and activity record.
2. Retrieve bioactivity data using ChEMBL widgets.
3. Filter records by endpoint type, units, relations, target confidence, and assay metadata.
4. Standardize molecular structures before descriptor or fingerprint calculation.
5. Handle duplicates and repeated measurements transparently.
6. Construct a QSAR-ready dataset with a documented curation protocol.
7. Report data provenance and limitations.

## Recommended workflow for class

```text
ChEMBL Browser → ChEMBL Bioactivity Retriever → Data Table
```

Then extend to:

```text
Bioactivity table → Mol Standardizer → Mol Descriptors 2 → QSAR Regression
```

For validation:

```text
Curated QSAR table → Scaffold Splitter → QSAR Regression → Applicability Domain
```

## Assessment suggestions

Ask students to submit:

- the final curated table,
- a curation log,
- a short explanation of included and excluded records,
- a QSAR-ready CSV,
- a short reproducibility report.

## Common mistakes to watch for

- Mixing IC50, Ki, Kd, and EC50 without justification.
- Ignoring `standard_relation` values such as `>`, `<`, or `~`.
- Treating all units as equivalent.
- Keeping duplicate compounds without aggregation rules.
- Ignoring target confidence and assay type.
- Building a QSAR model from a heterogeneous or poorly documented dataset.
