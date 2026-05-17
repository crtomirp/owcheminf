# Mol Standardizer

## Status

Current widget in the package.

Source:
- [ow_mol_standardizer.py](../../src/chem_inf_widgets/widgets/ow_mol_standardizer.py)

## Purpose

`Mol Standardizer` makes molecular representations more consistent before downstream analysis. It is typically the first chemistry-specific cleanup step in a workflow.

## Input

- Orange `Table`
- expected molecule column: a SMILES text column, usually named `SMILES`

## Output

A standardized Orange `Table` that preserves the original structure information and adds normalized structure output suitable for later widgets.

Depending on the current widget configuration, the output may include:

- original SMILES
- standardized SMILES
- change or processing notes

## Typical operations

The exact set depends on the widget implementation, but standardization workflows usually include combinations of:

- cleanup
- normalization
- metal disconnection
- largest-fragment selection
- reionization
- uncharging
- tautomer normalization

## Typical workflows

### Preprocessing before descriptors

1. `SDF Reader` or `File`
2. `Mol Standardizer`
3. `Fingerprint Generator` or `Mol Descriptor`

### Library curation

1. `SDF Reader`
2. `Mol Standardizer`
3. `Drug Filter`
4. `Scaffold Analysis`

### QSAR preparation

1. `Mol Standardizer`
2. `PaDEL Descriptors` or `Mol Descriptor`
3. `QSAR Regression`

## Notes

- Standardization improves reproducibility, but it can also change the exact representation of salts, charges and tautomeric forms.
- For teaching workflows, this widget is one of the most useful early examples of why chemical data curation matters.
