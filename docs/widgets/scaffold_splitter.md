# Scaffold Splitter

## Status

Current widget in the package.

Source:
- [ow_scaffold_splitter.py](../../src/chem_inf_widgets/widgets/ow_scaffold_splitter.py)
- [scaffold_splitter_service.py](../../src/chem_inf_widgets/chemcore/services/scaffold_splitter_service.py)

## Purpose

`Scaffold Splitter` separates a dataset into train, validation and test partitions using scaffold-aware logic instead of a purely random split.

This is especially useful for QSAR workflows, where random splitting often gives over-optimistic results.

## Input

- Orange `Table`
- molecule structures as SMILES-containing rows

## Output

- split subsets such as train / validation / test
- scaffold-oriented summary output

## Typical workflow

1. `Mol Standardizer`
2. `Scaffold Splitter`
3. descriptor widget
4. `QSAR Regression`
5. `Applicability Domain`

## Notes

- If you want more realistic model validation, this widget is one of the best upgrades over a default random split.
- It also works well as a teaching example for why chemically aware evaluation matters.
