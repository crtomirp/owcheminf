# Scaffold Frequency and Chemical Series

## Context

A scaffold frequency table shows whether a dataset is dominated by a few chemical series or is structurally diverse.

## Intended learning outcomes

Students can:

- read a scaffold frequency table,
- identify dominant scaffold families,
- discuss consequences for model validation.

## Orange workflow

```text
File → Scaffold Analysis → Scaffold Summary → Data Table
```

## Tasks

1. Run scaffold analysis with `Top scaffolds = 20`.
2. Sort the `Scaffold Summary` by `Count`.
3. Identify the largest scaffold family.
4. Compare the largest family with rare scaffolds.

## Guiding questions

1. Is the dataset balanced across scaffolds?
2. Could a random split place the same scaffold family in both training and test sets?
3. Why might this inflate test performance?
4. Which scaffolds would need more data for reliable SAR interpretation?

## Expected output

Students should link scaffold frequency to dataset bias and validation design.

## Assessment rubric

| Criterion | Basic | Good | Excellent |
|---|---|---|---|
| Workflow execution | Runs the main widgets | Correct settings and outputs | Clear, reproducible workflow |
| Chemical interpretation | Minimal description | Correct scaffold/cliff explanation | Insightful SAR reasoning |
| Validation discussion | Mentions train/test split | Explains random vs scaffold split | Connects validation, AD, and cliffs |
| Reporting | Fragmentary notes | Clear short answers | Publication-style transparency |
