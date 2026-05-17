# R-Group Decomposition

## Status

Current widget in the package.

Source:
- [ow_rgroup_decomposition.py](../../src/chem_inf_widgets/widgets/ow_rgroup_decomposition.py)
- [rgroup_decomposition_service.py](../../src/chem_inf_widgets/chemcore/services/rgroup_decomposition_service.py)

## Purpose

`R-Group Decomposition` turns a set of related molecules into a more SAR-friendly table by separating a shared core from substituent positions such as `R1`, `R2`, `R3`.

## Input

- `Data`
- molecule structures as SMILES-containing rows
- optional core query, depending on the current widget mode

## Output

- decomposition table with core and substituent assignments
- matched and unmatched subsets, depending on the current widget path

## Typical workflow

1. `Scaffold Analysis` or `Scaffold Splitter`
2. `R-Group Decomposition`
3. `Data Table`
4. `QSAR Regression` or manual SAR review

## Notes

- This widget is most useful when compounds already belong to a coherent chemical series.
- It pairs naturally with `Scaffold Analysis`, `Activity Cliff Finder` and `Compound Detail Card`.
