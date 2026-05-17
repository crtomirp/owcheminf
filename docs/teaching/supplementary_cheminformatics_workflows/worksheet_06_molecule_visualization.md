# Molecular Visualization and Quality Control

## Main widget(s)

```text
Mol Viewer / Mol 3D Viewer / Compound Detail Card
```

## Context

This worksheet covers a practical cheminformatics task that complements the main QSAR, standardization, ChEMBL, applicability domain, and scaffold/activity-cliff teaching packages.

## Intended learning outcomes

1. Use visual inspection to detect suspicious structures.
2. Connect tabular data to molecular structure.
3. Recognize when 2D and 3D views answer different questions.

## Recommended Orange workflow

```text
File → Mol Standardizer → Mol Viewer / Mol 3D Viewer / Compound Detail Card
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

1. Load `visualization_qc_demo.csv`.
2. Inspect aromatic systems, salts, charged compounds, and fused rings.
3. Compare raw and standardized forms.
4. Record at least three visual quality-control observations.

## Guiding questions

1. Which errors are easier to see visually than in a table?
2. When is a 3D view useful?
3. Can visualization prove that a structure is chemically correct?
4. How should visual QC be documented?

## Expected output

Students should obtain at least one processed data table and a short written interpretation. Where the widget provides multiple outputs, students should inspect both molecule-level and summary-level tables.

## Assessment

Use `assessment_rubric.md`. Award extra credit for comparing at least two alternative workflows or parameter settings.
