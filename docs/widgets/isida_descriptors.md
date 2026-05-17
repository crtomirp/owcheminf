# ISIDA Descriptors

## Status

Current widget in the package.

Source:
- [ow_isida_descriptors.py](../../src/chem_inf_widgets/widgets/ow_isida_descriptors.py)

## Purpose

`ISIDA Descriptors` generates fragment-count descriptor vectors using ISIDA-style fragmentation: sequences, shells, and triplets with configurable topology, fragment length, and minimum support.

## Input

- `Data` — Orange `Table` with a SMILES column
- `Molecules` — `ChemMol` list

## Output

- `Data` — Orange `Table` with ISIDA fragment count columns

## Main controls

- topology (linear sequences, circular shells, triplets)
- minimum and maximum fragment length
- minimum fragment support (frequency cutoff)
- include atom labels / bond labels

## Typical workflow

1. `Mol Standardizer`
2. `ISIDA Descriptors`
3. `QSAR Regression` or `Applicability Domain`

## Notes

- ISIDA descriptors encode local chemical environments as fragment frequencies and are particularly useful for QSAR on congeneric series.
- Fragment vocabulary is built from the input set, so descriptors are dataset-specific.
