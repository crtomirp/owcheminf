# Integrated Cheminformatics Quality-Control Pipeline

## Main widget(s)

```text
Multiple widgets
```

## Context

This worksheet covers a practical cheminformatics task that complements the main QSAR, standardization, ChEMBL, applicability domain, and scaffold/activity-cliff teaching packages.

## Intended learning outcomes

1. Build a complete preprocessing and inspection workflow.
2. Combine standardization, filtering, descriptors, fingerprints, and visualization.
3. Prepare a dataset for downstream QSAR or screening.

## Recommended Orange workflow

```text
File → Mol Standardizer → Drug Filter → Fingerprint Generator / Cyclic Registry Fingerprint → Scaffold Analysis → Data Table
```


## General setup

Recommended starting workflow:

```text
File / SDF Reader → Mol Standardizer → [target widget] → Data Table
```

For structure-based examples, inspect representative molecules with:

```text
Mol Viewer / Mol 3D Viewer / Compound Detail Card
```

Where appropriate, compare results against:

```text
Fingerprint Generator
Cyclic Registry Fingerprint
Mol Descriptors 2
QSAR Regression
Applicability Domain
```

## Deliverables

Students should submit:

1. the Orange workflow screenshot,
2. the exported result table,
3. a short interpretation of the chemical meaning,
4. a note about limitations and possible sources of error.


## Student tasks

1. Load `integrated_workflow_demo.csv`.
2. Standardize the structures.
3. Apply basic filters.
4. Calculate fingerprints and cyclic registry matches.
5. Summarize scaffold distribution.
6. Export the cleaned and annotated dataset.

## Guiding questions

1. Which compounds were removed or flagged?
2. Which scaffolds dominate?
3. Which registry motifs are common?
4. Is the dataset ready for QSAR?
5. What further curation would you perform?

## Expected output

Students should obtain at least one processed data table and a short written interpretation. Where the widget provides multiple outputs, students should inspect both molecule-level and summary-level tables.

## Assessment

Use `assessment_rubric.md`. Award extra credit for comparing at least two alternative workflows or parameter settings.
