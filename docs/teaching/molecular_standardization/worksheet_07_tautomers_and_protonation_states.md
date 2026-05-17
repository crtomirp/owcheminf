# Worksheet 07: Tautomers and Protonation States

## Context

Students learn that standardization is not the same as tautomer enumeration or pH-specific protonation.

## Widget

```text
Cheminf - Processing → Mol Standardizer
```

## Recommended input data

```text
examples/molecular_standardization/standardization_training_set.csv
examples/molecular_standardization/standardization_edge_cases.csv
```

## Intended learning outcomes

After this activity, students should be able to:

1. describe the relevant molecular standardization problem;
2. use the Mol Standardizer widget on a small chemical dataset;
3. interpret standardized SMILES and standardization logs;
4. decide whether the transformation is chemically appropriate for a given downstream task;
5. document the decision in a reproducible way.

## Orange workflow

```text
File → Mol Standardizer → Data Table
```

Optional extension:

```text
Mol Standardizer → Fingerprint Generator / Cyclic Registry Fingerprint / Mol Descriptors 2
```

## Student tasks

1. Identify molecules where protonation or tautomerism could matter.
2. Run default standardization and inspect what changes and what does not change.
3. Discuss whether a pH-specific preprocessing tool would be needed.
4. Propose an additional workflow for pH-dependent standardization.
5. Write a short warning paragraph for a QSAR report.

## Guiding questions

- Does standardization automatically choose the biologically active tautomer?
- Why does pH matter?
- Why should tautomer/protonation choices be documented separately from basic cleanup?

## Expected outputs

Students should produce:

- a table containing original and standardized SMILES;
- a short list of changed, unchanged, and failed molecules;
- interpretation of at least two standardization logs;
- a brief explanation of whether the standardized output is suitable for QSAR, fingerprinting, docking, or registry matching.

## Assessment rubric

| Criterion | 0 points | 1 point | 2 points | 3 points |
|---|---|---|---|---|
| Data loading | Not completed | Loaded with errors | Loaded correctly | Loaded and inspected carefully |
| Widget use | Not used | Used with unclear settings | Used correctly | Settings compared and justified |
| Chemical interpretation | Missing | Superficial | Mostly correct | Chemically well justified |
| Reproducibility | Missing | Minimal notes | Settings documented | Settings, limitations, and failed cases documented |

## Teacher notes

Emphasize that molecular standardization is not a universal truth-generating step. It is a protocol. Different scientific questions may require different protocols. Students should learn to preserve the original structure, inspect logs, and avoid silent changes.
