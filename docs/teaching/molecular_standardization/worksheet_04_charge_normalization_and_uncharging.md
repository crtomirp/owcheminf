# Worksheet 04: Charge Normalization and Uncharging

## Context

Students study how reionization and uncharging affect carboxylates, zwitterions, and other charged species.

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

1. Use molecules labelled `salt`, `zwitterion`, and `quaternary_ammonium`.
2. Run standardization with Reionizer and Uncharger on.
3. Repeat with Uncharger off.
4. Compare outputs for glycine zwitterion, sodium carboxylate, and tetramethylammonium.
5. Explain why permanent charges should not always disappear.

## Guiding questions

- Why are charge states important for solubility and binding?
- What is the difference between a removable charge and a permanent charge?
- Would you use uncharged molecules for docking? Why or why not?

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
