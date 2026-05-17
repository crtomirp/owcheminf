# Worksheet 01: Molecular Descriptor Preparation

**Estimated time:** 60–90 min  
**Level:** introductory  
**Main widgets:** File, Mol Descriptors 2, PaDEL Descriptors, ISIDA Descriptors, Data Table

## Context

Before building a QSAR model, molecules must be converted into numerical variables. These variables may be physicochemical descriptors, fragment counts, or fingerprints.

## Intended learning outcomes

By the end of this worksheet, students will be able to:

1. load a SMILES table in Orange,
2. calculate molecular descriptors,
3. distinguish descriptor tables from raw molecular data,
4. identify missing or invalid descriptor values,
5. explain why descriptor calculation is a necessary QSAR preprocessing step.

## Input data

```text
examples/qsar_widgets/qsar_training_set.csv
```

## Orange workflow

```text
File → Mol Descriptors 2 → Data Table
```

Optional comparisons:

```text
File → PaDEL Descriptors → Data Table
File → ISIDA Descriptors → Data Table
```

## Student tasks

1. Load the training CSV file.
2. Select the `smiles` column as the molecular structure column if prompted.
3. Calculate descriptors with `Mol Descriptors 2`.
4. Open the output in `Data Table`.
5. Identify at least five descriptor columns.
6. Check whether any descriptors contain missing values.

## Guiding questions

- What is the difference between a molecule name and a molecular descriptor?
- Why can descriptor calculation fail?
- Which descriptors are easy to interpret chemically?
- Why should constant or near-constant descriptors be removed before modelling?

## Expected output

A descriptor-augmented table containing the original compound metadata and calculated numerical columns.

## Assessment rubric

| Criterion | Excellent | Satisfactory | Needs improvement |
|---|---|---|---|
| Data loading | Correct file and SMILES column | File loaded with minor issues | Wrong or missing input |
| Descriptor calculation | Correct and inspected | Calculated but not interpreted | Not calculated |
| Interpretation | Explains descriptor meaning | Basic description | No chemical explanation |
