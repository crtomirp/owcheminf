# Pair Viewer

## Status

Current widget in the package.

Source:
- [ow_pair_viewer.py](../../src/chem_inf_widgets/widgets/ow_pair_viewer.py)

## Purpose

`Pair Viewer` displays two compounds side by side from a table that contains two SMILES columns. It is the natural downstream viewer for `Activity Cliff Finder` and `Matched Molecular Pairs`.

## Input

- `Data` — Orange `Table` with two SMILES columns (e.g. `smiles_a`, `smiles_b`) and optional activity, similarity, cliff-score columns

## Output

- `Selected Pairs` — currently selected rows
- `Pair Compounds` — both compounds of the selected pair as a two-row table

## Auto-detected columns

The widget tries to auto-detect:
- `smiles_a` / `smiles_b`
- `activity_a` / `activity_b`
- `similarity`, `activity_ratio`, `cliff_score`, `higher_active`

## Typical workflow

1. `Activity Cliff Finder` or `Matched Molecular Pairs`
2. `Pair Viewer`
3. `Compound Detail Card` or `Molecular Viewer`

## Notes

- Most useful in teaching when you want to show a non-linear SAR pair visually.
