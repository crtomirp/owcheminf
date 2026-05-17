# SDF Reader

## Status

Current widget in the package.

Source:
- [ow_sdf_reader.py](../../src/chem_inf_widgets/widgets/ow_sdf_reader.py)

## Purpose

`SDF Reader` loads molecules from an SDF file and passes them downstream as an Orange `Table` and/or a `ChemMol` list.

## Input

None — reads directly from a file path selected in the widget UI.

## Output

- `Data` — Orange `Table` with properties from SDF fields
- `Molecules` — `ChemMol` list

## Typical workflow

1. `SDF Reader`
2. `Mol Standardizer`
3. descriptor or analysis widget

## Notes

- This is the standard entry point for SDF-based compound libraries.
- Use `SDF Writer` to export results back to SDF format.
