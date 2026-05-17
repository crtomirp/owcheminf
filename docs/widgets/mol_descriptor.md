# Mol Descriptors (Mordred)

## Status

Current widget in the package. Requires the `descriptors` extra (`pip install -e .[descriptors]`).

Source:
- [ow_mol_descriptor.py](../../src/chem_inf_widgets/widgets/ow_mol_descriptor.py)

## Purpose

`Mol Descriptors 2` computes Mordred descriptors from a `Table` (SMILES column) or `ChemMol` list. It offers curated presets and a manual selector for choosing descriptor families.

## Input

- `Data` — Orange `Table` with a SMILES column
- `Molecules` — `ChemMol` list

## Output

- `Data` — Orange `Table` with descriptor columns appended
- `Molecules` — `ChemMol` list

## Presets

- `Manual` — choose categories and module prefixes manually
- `Balanced` — size, lipophilicity, H-bond, and rotatable-bond families
- `Constitutional` — counts, atom/bond composition, ring-centric families
- `Topological` — topological indices, path and walk descriptors
- `QSAR` — lipophilicity, H-bond, EState, BCUT, TPSA and related families
- `3D` — geometry and conformational families (requires 3D conformers)
- `Mordred Extended` — MoRSE, ETA, framework, aromaticity, Barysz families
- `Full Mordred` — entire Mordred catalog

## Typical workflow

1. `Mol Standardizer`
2. `Mol Descriptors`
3. `QSAR Regression` or `Applicability Domain`

## Notes

- Requires `mordred` (`pip install mordred` or use the `descriptors` extra).
- For simpler fingerprint-based workflows use `Fingerprint Generator` instead.
- For Java-based PaDEL descriptors use `PaDEL Descriptors`.
