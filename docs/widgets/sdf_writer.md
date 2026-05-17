# SDF Writer

## Status

Current widget in the package.

Source:
- [ow_sdf_writer.py](../../src/chem_inf_widgets/widgets/ow_sdf_writer.py)

## Purpose

`SDF Writer` exports molecules from an Orange `Table` or a `ChemMol` list to an SDF file on disk.

## Input

- `Data` — Orange `Table` with a SMILES column
- `Molecules` — `ChemMol` list

## Output

None — writes directly to a user-selected file path.

## Typical workflow

1. any chemistry widget producing a `Table` or `Molecules`
2. `SDF Writer`

## Notes

- Use `SDF Reader` to reload written files in subsequent workflows.
