# What Molecular Scaffolds Represent

## Context

A molecular scaffold is a simplified representation of the core framework of a molecule. In medicinal chemistry, molecules sharing a scaffold often form an analogue series.

## Intended learning outcomes

After this worksheet, students can:

- explain the difference between a whole molecule and its scaffold,
- identify scaffold families by visual inspection,
- explain why scaffold information matters in QSAR.

## Recommended data

Use:

```text
examples/scaffold_activity_cliff_analysis/scaffold_activity_training_set.csv
```

## Orange workflow

```text
File → Scaffold Analysis → Data Table
```

## Tasks

1. Load the dataset.
2. Before using the scaffold widget, inspect the names and SMILES.
3. Predict which compounds may share a scaffold.
4. Run **Scaffold Analysis**.
5. Compare your manual grouping with the `Murcko Scaffold` and `Generic Scaffold` columns.

## Guiding questions

1. Which compounds share a benzodiazepine-like or diaryl-like core?
2. Which compounds are acyclic or have no meaningful ring scaffold?
3. Why might two molecules with different substituents share the same scaffold?
4. Why might scaffold grouping be useful before QSAR modelling?

## Expected output

Students should be able to describe scaffold families and distinguish core frameworks from substituent changes.

## Assessment rubric

| Criterion | Basic | Good | Excellent |
|---|---|---|---|
| Workflow execution | Runs the main widgets | Correct settings and outputs | Clear, reproducible workflow |
| Chemical interpretation | Minimal description | Correct scaffold/cliff explanation | Insightful SAR reasoning |
| Validation discussion | Mentions train/test split | Explains random vs scaffold split | Connects validation, AD, and cliffs |
| Reporting | Fragmentary notes | Clear short answers | Publication-style transparency |
