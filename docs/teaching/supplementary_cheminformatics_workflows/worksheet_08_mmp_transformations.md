# Matched Molecular Pairs and Transformation Effects

## Main widget(s)

```text
Matched Molecular Pairs
```

## Context

This worksheet covers a practical cheminformatics task that complements the main QSAR, standardization, ChEMBL, applicability domain, and scaffold/activity-cliff teaching packages.

## Intended learning outcomes

1. Define a matched molecular pair.
2. Identify transformations between similar compounds.
3. Estimate the effect of substituent changes on activity or properties.

## Recommended Orange workflow

```text
File → Mol Standardizer → Matched Molecular Pairs → Data Table
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

1. Load `mmp_demo.csv`.
2. Run matched molecular pair analysis.
3. Find transformations that change pIC50, logP, or solubility.
4. Select three transformations and interpret them chemically.

## Guiding questions

1. Which transformations improve activity?
2. Which transformations worsen physicochemical properties?
3. Can an MMP result prove causality?
4. How could MMP analysis guide the next synthesis?

## Expected output

Students should obtain at least one processed data table and a short written interpretation. Where the widget provides multiple outputs, students should inspect both molecule-level and summary-level tables.

## Assessment

Use `assessment_rubric.md`. Award extra credit for comparing at least two alternative workflows or parameter settings.
