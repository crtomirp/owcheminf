# Diversity Picking for Library Selection

## Main widget(s)

```text
Diversity Picker
```

## Context

This worksheet covers a practical cheminformatics task that complements the main QSAR, standardization, ChEMBL, applicability domain, and scaffold/activity-cliff teaching packages.

## Intended learning outcomes

1. Explain why diverse subset selection is useful.
2. Select a representative subset from a chemical library.
3. Compare diversity picking with random selection.

## Recommended Orange workflow

```text
File → Mol Standardizer → Fingerprint Generator → Diversity Picker → Data Table
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

1. Load `library_diversity_demo.csv`.
2. Generate fingerprints.
3. Pick a diverse subset of 5 to 10 molecules.
4. Compare selected molecules with a random subset.
5. Inspect scaffolds and cyclic registry motifs in the selected subset.

## Guiding questions

1. Does the selected subset cover multiple scaffolds?
2. Are rare heterocycles preserved?
3. What is the trade-off between diversity and activity enrichment?
4. How could this help design an experimental screening plate?

## Expected output

Students should obtain at least one processed data table and a short written interpretation. Where the widget provides multiple outputs, students should inspect both molecule-level and summary-level tables.

## Assessment

Use `assessment_rubric.md`. Award extra credit for comparing at least two alternative workflows or parameter settings.
