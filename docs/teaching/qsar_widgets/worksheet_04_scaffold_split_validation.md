# Worksheet 04: Scaffold-Aware Validation

**Estimated time:** 90 min  
**Level:** intermediate  
**Main widgets:** Scaffold Splitter, QSAR Regression, Data Table

## Context

Random splitting can place very similar molecules in both train and test sets. Scaffold splitting is stricter and asks whether the model generalizes to new core structures.

## Intended learning outcomes

Students will be able to:

1. explain scaffold splitting conceptually,
2. create train/validation/test subsets,
3. compare random-like and scaffold-aware performance,
4. discuss domain of applicability.

## Orange workflow

```text
File → Mol Descriptors 2 → Scaffold Splitter
Scaffold Splitter → Train Data → QSAR Regression
Scaffold Splitter → Test Data → QSAR Regression External/Test input
```

## Student tasks

1. Split the dataset with `Scaffold Splitter`.
2. Inspect `Split Summary`.
3. Count compounds in train/validation/test sets.
4. Train a QSAR model on the training set.
5. Evaluate on the test set if supported by the workflow.
6. Compare the result with a non-scaffold split.

## Guiding questions

- Why is scaffold splitting harder than random splitting?
- What happens if a scaffold appears only in the test set?
- Why is lower scaffold-split performance not necessarily a bad sign?
- How does scaffold splitting relate to real medicinal chemistry prediction?

## Expected output

Separate train, validation, test tables and a split summary table.
