# Molecular Viewer

## Status

Current widget in the package.

Source:
- [ow_mol_viewer.py](../../src/chem_inf_widgets/widgets/ow_mol_viewer.py)

## Purpose

`Molecular Viewer` displays a scalable grid of 2D structure images from an Orange `Table` or a `ChemMol` list. It supports substructure highlighting.

## Input

- `Data` — Orange `Table` with a SMILES column
- `Molecules` — `ChemMol` list

## Output

- `Selected index` — integer index of the row selected in the grid

## Typical workflow

1. any filtering or search widget
2. `Molecular Viewer`
3. optional `Compound Detail Card` (connected via `Selected index`)

## Notes

- Substructure highlighting is applied when a query molecule is available upstream.
- Use `3D Molecular Viewer` for conformer-based 3D inspection.
