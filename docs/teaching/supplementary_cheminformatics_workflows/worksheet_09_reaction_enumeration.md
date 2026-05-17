# Reaction Enumeration and Virtual Library Design

## Main widget(s)

```text
Reaction Enumerator / Reactor
```

## Context

This worksheet covers a practical cheminformatics task that complements the main QSAR, standardization, ChEMBL, applicability domain, and scaffold/activity-cliff teaching packages.

## Intended learning outcomes

1. Explain virtual reaction enumeration.
2. Generate hypothetical products from reactants.
3. Filter products using descriptors or cyclic registry motifs.

## Recommended Orange workflow

```text
File(reactants) → Reaction Enumerator / Reactor → Mol Standardizer → Drug Filter / Cyclic Registry Fingerprint
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

1. Use `reaction_enumeration_demo.csv` as conceptual reactant input.
2. Select or define a simple reaction pattern if available.
3. Generate virtual products.
4. Filter products using drug-likeness or cyclic registry motifs.
5. Select a small prioritized product set.

## Guiding questions

1. Are all enumerated products chemically reasonable?
2. Which filters remove unrealistic products?
3. How could enumeration support library design?
4. What additional checks are needed before synthesis?

## Expected output

Students should obtain at least one processed data table and a short written interpretation. Where the widget provides multiple outputs, students should inspect both molecule-level and summary-level tables.

## Assessment

Use `assessment_rubric.md`. Award extra credit for comparing at least two alternative workflows or parameter settings.
