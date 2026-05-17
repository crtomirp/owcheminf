# Substructure & Similarity Search

## Status

Current widget in the package.

Source:
- [ow_substructure_search.py](../../src/chem_inf_widgets/widgets/ow_substructure_search.py)

## Purpose

`Substructure & Similarity Search` filters a compound library by exact substructure match (SMARTS) or by Tanimoto fingerprint similarity threshold. It also appends highlight indices for matched atoms.

## Input

- `Query` — SMARTS or SMILES string
- `Query Data` — Orange `Table` containing a query molecule
- `Query Molecule` — `ChemMol` object
- `Compounds` — Orange `Table` to search

## Output

- `Filtered Compounds` — matching rows with highlight atom indices appended

## Typical workflow

### Substructure search

1. `Mol Editor` or `Mol Ketcher` → `Query`
2. `Substructure & Similarity Search`
3. `Molecular Viewer`

### Fragment-guided search from inspection

1. `Compound Detail Card` → `Query Molecule`
2. `Substructure & Similarity Search`
3. `Molecular Viewer`

## Notes

- Use `PharmaFP Search` for ranked, multi-motif retrieval; use this widget for exact SMARTS containment filtering.
- Highlight indices are consumed automatically by `Molecular Viewer`.
