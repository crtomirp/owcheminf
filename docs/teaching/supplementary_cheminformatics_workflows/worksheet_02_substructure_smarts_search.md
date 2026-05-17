# Substructure and SMARTS Search

## Main widget(s)

```text
Substructure Search
```

## Context

This worksheet covers a practical cheminformatics task that complements the main QSAR, standardization, ChEMBL, applicability domain, and scaffold/activity-cliff teaching packages.

## Intended learning outcomes

1. Describe the difference between exact molecule similarity and substructure search.
2. Use SMARTS queries to identify chemical motifs.
3. Interpret false positives and missed matches.

## Recommended Orange workflow

```text
File → Mol Standardizer → Substructure Search → Data Table
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

1. Load `substructure_search_demo.csv`.
2. Search for benzene, pyridine, tertiary amine, amide, nitro, and sulfoxide motifs.
3. Compare aromatic and kekulized query behavior.
4. Export the matched molecules.

## Guiding questions

1. Which queries are broad and which are specific?
2. Why can SMARTS matching be sensitive to aromaticity?
3. How would you design a query for a heterocycle family rather than a single ring?
4. When is substructure search better than fingerprint similarity?

## Expected output

Students should obtain at least one processed data table and a short written interpretation. Where the widget provides multiple outputs, students should inspect both molecule-level and summary-level tables.

## Assessment

Use `assessment_rubric.md`. Award extra credit for comparing at least two alternative workflows or parameter settings.
