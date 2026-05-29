# Molecule Export Hub

**Category:** Cheminf - Core  
**Widget:** Molecule Export Hub

The Molecule Export Hub provides a single entry point for exporting molecular datasets from Orange workflows to text or structure files on disk.

## Supported inputs

- Orange `Data` tables with a SMILES column
- `Molecules` (`ChemMol` list)

## Supported outputs on disk

- CSV / TSV / TXT
- `.smi` / `.smiles`
- SDF / SD

## Widget outputs

- **Export Report** — one row per input record with conversion/export status.
- **Failed Records** — rows that could not be converted into exportable molecules.
- **Export Summary** — compact summary of written format, columns and counts.

## Recommended workflow

```text
Molecule QC Dashboard → Molecule Export Hub
```

## Why this matters

Cheminformatics workflows often end with ad hoc export steps that lose provenance, silently skip invalid rows, or write inconsistent structure columns. This widget makes export explicit, inspectable and reproducible.
