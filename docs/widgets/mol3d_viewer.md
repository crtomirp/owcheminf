# 3D Molecular Viewer

## Status

Current widget in the package. Requires the `viewer3d` extra (`pip install -e .[viewer3d]`).

Source:
- [ow_mol3d_viewer.py](../../src/chem_inf_widgets/widgets/ow_mol3d_viewer.py)

## Purpose

`3D Molecular Viewer` generates 3D conformers with RDKit and renders them in an interactive `py3Dmol` gallery inside Orange.

## Input

- `Data` — Orange `Table` with a SMILES column
- `Molecules` — `ChemMol` list
- `Selected index` — integer index to highlight a specific compound

## Output

None — viewer only.

## Typical workflow

1. `Mol Standardizer`
2. `3D Molecular Viewer`

Or as a downstream viewer:

1. `Molecular Viewer` → `Selected index`
2. `3D Molecular Viewer`

## Notes

- Requires `py3Dmol` (`pip install py3Dmol` or use the `viewer3d` extra).
- Conformer generation can be slow for large or complex molecules.
