# Drug-Likeness and Chemical Filters

## Main widget(s)

```text
Drug Filter
```

## Context

This worksheet covers a practical cheminformatics task that complements the main QSAR, standardization, ChEMBL, applicability domain, and scaffold/activity-cliff teaching packages.

## Intended learning outcomes

1. Apply simple drug-likeness and property filters.
2. Recognize limitations of rule-based filtering.
3. Connect filters to practical compound triage.

## Recommended Orange workflow

```text
File → Mol Standardizer → Mol Descriptors 2 → Drug Filter → Data Table
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

1. Load `drug_filter_demo.csv`.
2. Calculate descriptors.
3. Apply drug-likeness filters.
4. Identify compounds removed by the filter.
5. Discuss whether any removed compounds might still be useful.

## Guiding questions

1. Which rules are most restrictive?
2. Can natural products violate drug-likeness rules and still be bioactive?
3. Why should filters not replace expert chemical judgement?
4. How would filtering affect chemical diversity?

## Expected output

Students should obtain at least one processed data table and a short written interpretation. Where the widget provides multiple outputs, students should inspect both molecule-level and summary-level tables.

## Assessment

Use `assessment_rubric.md`. Award extra credit for comparing at least two alternative workflows or parameter settings.
