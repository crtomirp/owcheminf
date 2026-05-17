# Worksheet 02 — From Curated Dataset to First QSAR Model

## Context

A QSAR model should start from a documented and reproducible dataset. This
activity connects dataset curation to descriptor generation and model training.

## Orange workflow

```text
Molecule Import Hub
  → QSAR Dataset Builder
  → Mol Descriptors 2
  → QSAR Regression
```

## Tasks

1. Curate the demo dataset with `QSAR Dataset Builder`.
2. Generate a small descriptor set with `Mol Descriptors 2`.
3. Train a simple QSAR model with `QSAR Regression`.
4. Record which settings were used in the curation step.
5. Discuss why the model is only a teaching example and not a validated research model.

## Guiding questions

- Why should duplicate measurements be aggregated before modeling?
- What would happen if `IC50` and `Ki` were mixed without annotation?
- Why is a curation report useful for reproducibility?
