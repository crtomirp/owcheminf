# Diversity Picker

## Status

Current widget in the package.

Source:
- [ow_diversity_picker.py](../../src/chem_inf_widgets/widgets/ow_diversity_picker.py)

## Purpose

`Diversity Picker` selects a maximally diverse subset of compounds using fingerprint-based algorithms.

## Input

- `Data` — Orange `Table` with a SMILES column
- `Molecules` — `ChemMol` list

## Output

- `Selected Data` — diverse subset as Orange `Table`
- `Remainder Data` — compounds not selected
- `Selected Molecules` — diverse subset as `ChemMol` list
- `Remainder Molecules` — remainder as `ChemMol` list

## Supported algorithms

- MaxMin (default, deterministic)
- Sphere exclusion
- Butina clustering

## Typical workflow

1. `Mol Standardizer`
2. `Fingerprint Generator`
3. `Diversity Picker`
4. `Molecular Viewer`

## Notes

- MaxMin is the recommended default for teaching because it is fast and reproducible.
- The subset size is set directly in the widget controls.
