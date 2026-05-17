# Activity Cliff Concepts

## Context

An activity cliff occurs when two structurally similar molecules have a large activity difference. This challenges the assumption that similar compounds have similar properties.

## Intended learning outcomes

Students can:

- define an activity cliff,
- explain similarity and potency thresholds,
- distinguish cliffs from ordinary SAR trends.

## Recommended data

Use:

```text
examples/scaffold_activity_cliff_analysis/activity_cliff_demo_set.csv
```

## Orange workflow

```text
File → Activity Cliff Finder → Cliff Pairs → Data Table
```

## Suggested settings

```text
Similarity threshold: 0.65
Activity fold threshold: 10
Activity scale: Log potency if using pIC50
```

## Tasks

1. Load the activity cliff demo set.
2. Select the SMILES column.
3. Select the `pIC50` activity column.
4. Run **Activity Cliff Finder**.
5. Inspect `Cliff Pairs`.

## Guiding questions

1. Which compound pairs form cliffs?
2. Which compound is more active in each pair?
3. Are cliff pairs usually from the same scaffold family?
4. What small structural changes may explain the cliff?

## Expected output

Students should identify at least one similar pair with a large activity difference and explain why it is important.

## Assessment rubric

| Criterion | Basic | Good | Excellent |
|---|---|---|---|
| Workflow execution | Runs the main widgets | Correct settings and outputs | Clear, reproducible workflow |
| Chemical interpretation | Minimal description | Correct scaffold/cliff explanation | Insightful SAR reasoning |
| Validation discussion | Mentions train/test split | Explains random vs scaffold split | Connects validation, AD, and cliffs |
| Reporting | Fragmentary notes | Clear short answers | Publication-style transparency |
