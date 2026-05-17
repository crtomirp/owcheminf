# Scaffold Analysis

## Status

Current widget in the package.

Source:
- [ow_scaffold_analysis.py](../../src/chem_inf_widgets/widgets/ow_scaffold_analysis.py)

## Purpose

`Scaffold Analysis` annotates molecules with their Murcko scaffolds and builds a scaffold frequency summary table.

## Input

- `Data` — Orange `Table` with a SMILES column
- `Molecules` — `ChemMol` list

## Output

- `Annotated Data` — input table with scaffold SMILES and scaffold ID columns appended
- `Scaffold Summary` — one row per unique scaffold with frequency counts
- `Annotated Molecules` — `ChemMol` list with scaffold annotations

## Typical workflow

1. `Mol Standardizer`
2. `Scaffold Analysis`
3. `Data Table` or `Molecular Viewer`

Or for SAR follow-up:

1. `Scaffold Analysis`
2. `R-Group Decomposition`

## Notes

- Murcko scaffolds are computed with RDKit.
- Use `Scaffold Splitter` when you need train/test splits rather than scaffold annotations.
