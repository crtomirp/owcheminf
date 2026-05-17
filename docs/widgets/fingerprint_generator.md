# Fingerprint Generator

## Status

Current widget in the package.

Source:
- [ow_fingerprint_generator.py](../../src/chem_inf_widgets/widgets/ow_fingerprint_generator.py)
- [fingerprints.py](../../src/chem_inf_widgets/chemcore/descriptors/fingerprints.py)

## Purpose

`Fingerprint Generator` converts molecules into machine-readable structural fingerprints for similarity search, clustering, modeling and chemical space analysis.

## Input

One of:

- `Data` as Orange `Table`
- `Molecules` as `ChemMol` list

For table input, the widget expects a text column containing SMILES. It tries to auto-detect a sensible default and also lets the user choose the column explicitly.

## Output

- `Fingerprints` as Orange `Table`
- `Molecules` as `ChemMol` list with fingerprint information attached

Depending on settings, the output can also:

- keep selected input meta columns
- include original input columns as metas
- append numeric descriptors for QSAR-style downstream modeling

## Supported fingerprint types

- Morgan
- RDKit
- MACCS

## Main controls

- fingerprint type
- bit size
- Morgan radius
- whether to sanitize structures
- whether to also output `ChemMol` objects
- whether to append numeric descriptors

## Typical workflows

### Similarity / diversity

1. `Mol Standardizer`
2. `Fingerprint Generator`
3. `Similarity Search` or `Diversity Picker`

### Chemical space

1. `Mol Standardizer`
2. `Fingerprint Generator`
3. `PCA`
4. `Scatter Plot`

### QSAR preparation

1. `Mol Standardizer`
2. `Fingerprint Generator`
3. `QSAR Regression`

## Notes

- This widget is a core bridge between chemistry-specific preprocessing and generic Orange machine learning widgets.
- For richer descriptor sets, use `Mol Descriptor` or `PaDEL Descriptors`.
