# PaDEL Descriptors

## Status

Current widget in the package. Requires `padelpy` and Java.

Source:
- [ow_padel_descriptors.py](../../src/chem_inf_widgets/widgets/ow_padel_descriptors.py)

## Purpose

`PaDEL Descriptors` computes PaDEL descriptors and fingerprints via the `padelpy` wrapper. PaDEL covers a large range of 2D/3D descriptors and fingerprint types not available in pure-Python toolkits.

## Input

- `Data` — Orange `Table` with a SMILES column
- `Molecules` — `ChemMol` list

## Output

- `Data` — Orange `Table` with descriptor columns appended
- `Molecules` — `ChemMol` list

## Typical workflow

1. `Mol Standardizer`
2. `PaDEL Descriptors`
3. `QSAR Regression` or `Applicability Domain`

## Notes

- Requires Java (`java -version` to check) and `padelpy` (`pip install padelpy`).
- PaDEL computation can be slower than RDKit or Mordred for large datasets.
- If PaDEL is not available, use `Mol Descriptors` (Mordred) or `Fingerprint Generator` instead.
- See [troubleshooting.md](../troubleshooting.md) if PaDEL fails to run.
