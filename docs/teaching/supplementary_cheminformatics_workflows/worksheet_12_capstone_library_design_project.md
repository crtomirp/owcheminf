# Capstone: Design a Small Focused Screening Library

## Main widget(s)

```text
Multiple widgets
```

## Context

This worksheet covers a practical cheminformatics task that complements the main QSAR, standardization, ChEMBL, applicability domain, and scaffold/activity-cliff teaching packages.

## Intended learning outcomes

1. Use multiple cheminformatics widgets to design a small screening subset.
2. Balance similarity, diversity, drug-likeness, and interpretability.
3. Justify compound selection with chemical evidence.

## Recommended Orange workflow

```text
File → Mol Standardizer → Similarity Search / Diversity Picker / Drug Filter / Scaffold Analysis / Cyclic Registry Fingerprint
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

1. Start from `library_diversity_demo.csv`.
2. Remove problematic structures.
3. Select a diverse subset.
4. Ensure at least three scaffold families are represented.
5. Use cyclic registry matches to document heterocycle coverage.
6. Prepare a final ranked list of 10 compounds.

## Guiding questions

1. Why did you select these compounds?
2. How diverse is the final set?
3. Which heterocycles and functional groups are represented?
4. What would you test experimentally first?
5. What are the limitations of the computational selection?

## Expected output

Students should obtain at least one processed data table and a short written interpretation. Where the widget provides multiple outputs, students should inspect both molecule-level and summary-level tables.

## Assessment

Use `assessment_rubric.md`. Award extra credit for comparing at least two alternative workflows or parameter settings.
