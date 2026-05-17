# Pharmacophore Fingerprint Search

## Main widget(s)

```text
PharmaFP Search
```

## Context

This worksheet covers a practical cheminformatics task that complements the main QSAR, standardization, ChEMBL, applicability domain, and scaffold/activity-cliff teaching packages.

## Intended learning outcomes

1. Describe pharmacophore features such as donors, acceptors, aromatic rings, and hydrophobes.
2. Use pharmacophore-like fingerprints to find functionally similar compounds.
3. Compare pharmacophore similarity with structural similarity.

## Recommended Orange workflow

```text
File → Mol Standardizer → PharmaFP Search → Data Table
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

1. Load `pharmacophore_demo.csv`.
2. Choose a query molecule.
3. Search for compounds with similar pharmacophore patterns.
4. Compare hits to Morgan similarity hits.
5. Inspect examples visually.

## Guiding questions

1. Can pharmacophore similarity retrieve structurally different compounds?
2. Which features drive the matches?
3. When is pharmacophore similarity useful in scaffold hopping?
4. What information is missing without 3D alignment?

## Expected output

Students should obtain at least one processed data table and a short written interpretation. Where the widget provides multiple outputs, students should inspect both molecule-level and summary-level tables.

## Assessment

Use `assessment_rubric.md`. Award extra credit for comparing at least two alternative workflows or parameter settings.
