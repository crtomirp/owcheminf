# Molecule Import Hub

**Category:** Cheminf - Data  
**Widget:** Molecule Import Hub

The Molecule Import Hub provides a single entry point for importing molecular datasets into OWChemInf workflows.

## Supported inputs

- CSV / TSV / TXT tables with a SMILES column
- `.smi` / `.smiles` files
- SDF / SD files

## Outputs

- **Data** — imported molecules as an Orange table.
- **Molecules** — list of `ChemMol` objects for downstream chemistry widgets.
- **Import Report** — one row per input record, including parse status and errors.
- **Failed Records** — records that could not be imported.
- **Import Summary** — compact summary of detected format, columns and counts.

## Recommended first workflow

```text
Molecule Import Hub → Molecule QC Dashboard → Data Table
```

## Why this matters

Cheminformatics workflows often fail because the input structure column was misdetected, invalid structures were silently skipped, or SDF properties were lost. This widget makes the import step explicit, inspectable and reproducible.
