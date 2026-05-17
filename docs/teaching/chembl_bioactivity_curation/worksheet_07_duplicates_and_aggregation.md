# Worksheet 07: Duplicate Handling and Value Aggregation

## Context

Students identify repeated measurements and decide how to aggregate activity values.

This worksheet belongs to the **ChEMBL Bioactivity Curation** teaching package. It focuses on data quality before QSAR modelling.

## Intended learning outcomes

After this worksheet, students should be able to:

1. Explain the main curation concept addressed in this activity.
2. Use the relevant Orange widget workflow to inspect or process data.
3. Record curation decisions in a reproducible way.
4. Identify at least one limitation of the resulting dataset.

## Recommended Orange workflow

```text
Data Table → Select Rows / manual summary
```

## Recommended input data

Use one of the files in:

```text
examples/chembl_bioactivity_curation/
```

For live work, use the ChEMBL widgets and save the raw retrieved table before filtering.

## Student tasks

Group records by molecule identifier or canonical SMILES and compare repeated pChEMBL values.

Recommended steps:

1. Load or retrieve the data.
2. Inspect all relevant metadata columns.
3. Apply one curation decision at a time.
4. Save the intermediate table if a major filter is applied.
5. Write a short justification for each exclusion rule.

## Guiding questions

When should duplicates be averaged? When should they be kept separate? Why is median often robust?

Additional questions:

- Which records would you exclude and why?
- Which records are ambiguous?
- What metadata must be preserved for reproducibility?
- How would your decision affect a later QSAR model?

## Expected output

Students should produce:

- a filtered or annotated table,
- a short curation log,
- a list of inclusion/exclusion rules,
- a brief explanation of limitations.

## Assessment rubric

| Criterion | 0 points | 1 point | 2 points | 3 points |
|---|---|---|---|---|
| Data inspection | Not attempted | Minimal | Relevant columns inspected | Thorough metadata inspection |
| Curation logic | No rules | Vague rules | Clear rules | Clear and justified rules |
| Reproducibility | Not documented | Partial notes | Mostly reproducible | Fully reproducible curation log |
| Chemical/biological reasoning | Missing | Superficial | Reasonable | Strong and critical reasoning |

## Teacher notes

Encourage students to avoid treating downloaded bioactivity records as automatically model-ready. The main lesson is that curation decisions are scientific decisions, not only technical steps.
