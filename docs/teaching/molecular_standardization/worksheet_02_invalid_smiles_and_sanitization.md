# Worksheet 02: Invalid SMILES and Sanitization Problems

## Context

Students learn that some input structures cannot be parsed or sanitized and must be tracked rather than silently discarded.

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

1. Load `standardization_edge_cases.csv`.
2. Run `Mol Standardizer` with sanitize before/after enabled.
3. Find rows with empty standardized SMILES or warning logs.
4. Classify each failed molecule as missing, syntactically invalid, or chemically problematic.
5. Create a short data-curation action for each failed row: fix, remove, or investigate.

## Guiding questions

- What is the difference between a syntactically invalid SMILES and a chemically suspicious structure?
- Why is silent failure dangerous in a modelling dataset?
- Should invalid molecules be removed automatically?

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
