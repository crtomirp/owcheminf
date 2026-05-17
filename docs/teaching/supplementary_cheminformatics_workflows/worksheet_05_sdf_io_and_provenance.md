# SDF Reading, Writing, and Data Provenance

## Main widget(s)

```text
SDF Reader / SDF Writer
```

## Context

This worksheet covers a practical cheminformatics task that complements the main QSAR, standardization, ChEMBL, applicability domain, and scaffold/activity-cliff teaching packages.

## Intended learning outcomes

1. Import molecules from SDF files.
2. Preserve molecule properties during processing.
3. Export processed molecules with provenance fields.

## Recommended Orange workflow

```text
SDF Reader → Mol Standardizer → Cyclic Registry Fingerprint → SDF Writer
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

1. Load an SDF or use CSV and convert where supported.
2. Inspect available molecule properties.
3. Standardize molecules.
4. Add fingerprint or registry annotations.
5. Export the processed dataset.

## Guiding questions

1. Which properties are preserved?
2. Which new properties are added?
3. How can exported SDF files support reproducibility?
4. What information should never be lost during curation?

## Expected output

Students should obtain at least one processed data table and a short written interpretation. Where the widget provides multiple outputs, students should inspect both molecule-level and summary-level tables.

## Assessment

Use `assessment_rubric.md`. Award extra credit for comparing at least two alternative workflows or parameter settings.
