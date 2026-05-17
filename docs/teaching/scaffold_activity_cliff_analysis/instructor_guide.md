# Instructor Guide: Scaffold and Activity Cliff Analysis

## Target audience

This module is suitable for undergraduate or MSc-level courses in cheminformatics, medicinal chemistry, QSAR/QSPR, computational drug design, or data-driven chemistry.

## Recommended duration

- Short version: 2 teaching hours
- Full practical: 4 to 6 teaching hours
- Capstone project: 1 to 2 weeks

## Required preparation

Install the package and verify the widgets:

```bash
conda activate owcheminf
cd cinf
pip install -e .
orange-canvas
```

The following widgets should be visible in Orange:

```text
Cheminf - Processing → Mol Standardizer
Cheminf - Processing → Scaffold Analysis
Cheminf - Processing → Scaffold Splitter
Cheminf - Processing → Activity Cliff Finder
Cheminf - Processing → Fingerprint Generator
Cheminf - Processing → Cyclic Registry Fingerprint
Cheminf - Processing → Applicability Domain
```

## Recommended teaching strategy

1. Start with visual/conceptual scaffold examples.
2. Let students group molecules by eye before running the scaffold widget.
3. Use the widget output to compare manual and algorithmic grouping.
4. Show why random splits can leak scaffold information.
5. Introduce activity cliffs only after students understand molecular similarity.
6. Connect cliff pairs to medicinal chemistry reasoning.
7. Finish by asking how scaffold splits and cliffs affect QSAR reliability.

## Common misconceptions

| Misconception | Teaching correction |
|---|---|
| A scaffold is the whole molecule. | A scaffold is a core framework, not every substituent. |
| Similar molecules always have similar activity. | Activity cliffs are counterexamples. |
| Random split is always fair. | Random split may place close analogues in both train and test sets. |
| Activity cliffs are bad data. | They may be chemically meaningful SAR signals. |
| Scaffold split always gives better models. | It often gives more realistic but lower performance estimates. |

## Assessment ideas

Students can be assessed on:

- correct use of the widgets,
- interpretation of scaffold summary tables,
- explanation of random-split leakage,
- identification and discussion of activity cliffs,
- quality of QSAR validation discussion,
- clarity of reproducibility notes.
