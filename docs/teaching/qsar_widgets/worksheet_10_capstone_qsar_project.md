# Worksheet 10: Mini QSAR Capstone Project

**Estimated time:** 2–4 hours  
**Level:** intermediate to advanced  
**Main widgets:** all QSAR-related widgets

## Context

This capstone integrates molecular representation, validation, modelling, prediction, and interpretation.

## Intended learning outcomes

Students will be able to:

1. design a complete QSAR workflow,
2. justify descriptor/fingerprint choices,
3. compare at least two modelling approaches,
4. interpret model outputs chemically,
5. prepare a reproducible mini-report.

## Input data

Use:

```text
examples/qsar_widgets/qsar_training_set.csv
examples/qsar_widgets/qsar_external_prediction_set.csv
```

Students may also bring their own small SMILES dataset.

## Suggested Orange workflow

```text
File
 → Mol Descriptors 2
 → Scaffold Splitter
 → MLR Model Selection
 → QSAR Regression
 → Data Table
```

Alternative representation:

```text
File
 → Cyclic Registry Fingerprint
 → QSAR Regression
 → Matched Registry Entries → Data Table
```

## Student tasks

1. Define the modelling question.
2. Choose a molecular representation.
3. Build at least two QSAR models or two descriptor/fingerprint workflows.
4. Use a validation strategy.
5. Generate external predictions.
6. Identify at least one limitation.
7. Present results in a short report.

## Required report sections

- Dataset and target.
- Molecular representation.
- Model and validation strategy.
- Results.
- Chemical interpretation.
- Domain-of-applicability discussion.
- Conclusion.

## Guiding questions

- Which workflow is most interpretable?
- Which workflow gives better predictive performance?
- Are the external predictions within the training domain?
- What would you do next with a larger dataset?

## Expected output

A mini-project report and an Orange workflow screenshot or `.ows` file if students save the workflow.
