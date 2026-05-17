# Molecular Similarity Search

## Main widget(s)

```text
Similarity Search
```

## Context

This worksheet covers a practical cheminformatics task that complements the main QSAR, standardization, ChEMBL, applicability domain, and scaffold/activity-cliff teaching packages.

## Intended learning outcomes

1. Explain Tanimoto similarity in the context of molecular fingerprints.
2. Use a query molecule to find similar compounds.
3. Discuss why similarity depends on representation.

## Recommended Orange workflow

```text
File → Mol Standardizer → Fingerprint Generator → Similarity Search → Data Table
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

1. Load `similarity_search_demo.csv`.
2. Choose caffeine, nicotine, or indole-like compounds as queries.
3. Run similarity search using Morgan fingerprints.
4. Inspect the top 10 nearest neighbors.
5. Repeat after changing fingerprint settings if available.

## Guiding questions

1. Are the nearest neighbors chemically intuitive?
2. What structural motifs dominate the top hits?
3. Would the results change with MACCS or cyclic registry fingerprints?
4. How would you choose a similarity threshold for library filtering?

## Expected output

Students should obtain at least one processed data table and a short written interpretation. Where the widget provides multiple outputs, students should inspect both molecule-level and summary-level tables.

## Assessment

Use `assessment_rubric.md`. Award extra credit for comparing at least two alternative workflows or parameter settings.
