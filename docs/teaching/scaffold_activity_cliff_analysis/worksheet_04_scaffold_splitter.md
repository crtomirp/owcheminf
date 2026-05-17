# Scaffold-Based Train/Validation/Test Splitting

## Context

Scaffold splitting assigns entire scaffold families to train, validation, or test sets. This tests model generalisation to new chemical series.

## Intended learning outcomes

Students can:

- create a scaffold split,
- interpret the split summary,
- explain why scaffold split is stricter than random split.

## Orange workflow

```text
File → Scaffold Splitter → Train Data / Validation Data / Test Data
                       ↓
                 Split Summary → Data Table
```

## Suggested settings

```text
Train fraction: 0.70
Validation fraction: 0.15
Test fraction: 0.15
Scaffold kind: Generic Murcko
Random seed: 1
```

## Tasks

1. Create a scaffold split.
2. Inspect the `Split Summary` output.
3. Confirm that compounds from the same scaffold are assigned to one split.
4. Compare split sizes with the requested fractions.

## Guiding questions

1. Why may the final split sizes differ from exactly 70/15/15?
2. What happens when a scaffold family is large?
3. Why is scaffold split important for prospective QSAR?
4. When might scaffold split be too strict?

## Expected output

Students should be able to justify scaffold splitting for external-like validation.

## Assessment rubric

| Criterion | Basic | Good | Excellent |
|---|---|---|---|
| Workflow execution | Runs the main widgets | Correct settings and outputs | Clear, reproducible workflow |
| Chemical interpretation | Minimal description | Correct scaffold/cliff explanation | Insightful SAR reasoning |
| Validation discussion | Mentions train/test split | Explains random vs scaffold split | Connects validation, AD, and cliffs |
| Reporting | Fragmentary notes | Clear short answers | Publication-style transparency |
