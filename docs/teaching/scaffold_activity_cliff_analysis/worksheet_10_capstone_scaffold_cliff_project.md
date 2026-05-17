# Capstone Project: Scaffold and Activity Cliff Analysis

## Context

This capstone combines data cleaning, scaffold analysis, activity cliff detection, QSAR validation, and reproducible reporting.

## Intended learning outcomes

Students can:

- design a scaffold/cliff analysis workflow,
- justify parameter choices,
- prepare a short reproducible report,
- connect results to medicinal chemistry and QSAR reliability.

## Project task

Choose a dataset of at least 30 molecules with SMILES and activity values. You may use ChEMBL-derived data, a provided example dataset, or a curated teaching dataset.

## Required workflow

```text
File
 → Mol Standardizer
 → Scaffold Analysis
 → Scaffold Splitter
 → Activity Cliff Finder
 → QSAR Regression
 → Applicability Domain
```

## Required report sections

1. Dataset description
2. Standardization protocol
3. Scaffold analysis summary
4. Scaffold split design
5. Activity cliff settings and results
6. QSAR validation strategy
7. Applicability domain discussion
8. Limitations and recommendations

## Guiding questions

1. Which scaffold families dominate the dataset?
2. Are there activity cliffs within major scaffolds?
3. How different are random and scaffold validation expectations?
4. Which compounds would you prioritise for manual review?
5. What would you improve in the dataset before publication?

## Expected output

A short report of 3–5 pages or a presentation with scaffold tables, cliff pair examples, and validation recommendations.

## Assessment rubric

| Criterion | Basic | Good | Excellent |
|---|---|---|---|
| Workflow execution | Runs the main widgets | Correct settings and outputs | Clear, reproducible workflow |
| Chemical interpretation | Minimal description | Correct scaffold/cliff explanation | Insightful SAR reasoning |
| Validation discussion | Mentions train/test split | Explains random vs scaffold split | Connects validation, AD, and cliffs |
| Reporting | Fragmentary notes | Clear short answers | Publication-style transparency |
