# R-Group Decomposition for Chemical Series

## Main widget(s)

```text
R-Group Decomposition
```

## Context

This worksheet covers a practical cheminformatics task that complements the main QSAR, standardization, ChEMBL, applicability domain, and scaffold/activity-cliff teaching packages.

## Intended learning outcomes

1. Explain the idea of a common core and variable substituents.
2. Use R-group decomposition to summarize a congeneric series.
3. Relate substituent changes to activity trends.

## Recommended Orange workflow

```text
File → Mol Standardizer → Scaffold Analysis → R-Group Decomposition → Data Table
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

1. Load `rgroup_series_demo.csv`.
2. Identify the dominant scaffold.
3. Run R-group decomposition if the widget is available.
4. Compare R groups against activity values.
5. Create a short SAR table.

## Guiding questions

1. Is the selected core appropriate?
2. Which substituents correlate with higher activity?
3. What happens if molecules do not share a clean common core?
4. How is R-group decomposition related to medicinal chemistry optimization?

## Expected output

Students should obtain at least one processed data table and a short written interpretation. Where the widget provides multiple outputs, students should inspect both molecule-level and summary-level tables.

## Assessment

Use `assessment_rubric.md`. Award extra credit for comparing at least two alternative workflows or parameter settings.
