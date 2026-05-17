# Random Split Versus Scaffold Split

## Context

Random splits often produce optimistic results when close analogues appear in both training and test sets. Scaffold splits are more demanding.

## Intended learning outcomes

Students can:

- compare random and scaffold validation concepts,
- predict when random split may overestimate model performance,
- explain validation leakage using scaffold families.

## Orange workflow

Random split concept:

```text
File → Mol Descriptors 2 → QSAR Regression
```

Scaffold split concept:

```text
File → Scaffold Splitter → QSAR Regression
```

## Tasks

1. Run or discuss a random validation workflow.
2. Run a scaffold split workflow.
3. Compare which molecules/scaffolds are present in train and test.
4. Discuss expected model performance differences.

## Guiding questions

1. Why is random split often easier for the model?
2. Which split better estimates performance on new chemical series?
3. Does lower scaffold-split performance mean the model is worse?
4. Which split would you report for a publication?

## Expected output

Students should understand validation leakage and conservative validation.

## Assessment rubric

| Criterion | Basic | Good | Excellent |
|---|---|---|---|
| Workflow execution | Runs the main widgets | Correct settings and outputs | Clear, reproducible workflow |
| Chemical interpretation | Minimal description | Correct scaffold/cliff explanation | Insightful SAR reasoning |
| Validation discussion | Mentions train/test split | Explains random vs scaffold split | Connects validation, AD, and cliffs |
| Reporting | Fragmentary notes | Clear short answers | Publication-style transparency |
