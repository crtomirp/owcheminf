# ChEMBL Browser

## Status

Current widget in the package.

Source:
- [ow_chembl_browser.py](../../src/chem_inf_widgets/widgets/ow_chembl_browser.py)

## Purpose

`ChEMBL Browser` provides an interactive search and browsing interface for the ChEMBL database. It retrieves compound records with exportable properties and bioactivities suitable for QSAR and clustering workflows.

## Input

None — queries ChEMBL directly via the ChEMBL web API.

## Output

- `Data` — Orange `Table` with compound properties
- `Molecules` — `ChemMol` list
- `Selected Data` — currently selected rows
- `Selected Molecules` — currently selected rows as `ChemMol` list

## Typical workflow

1. `ChEMBL Browser`
2. `Mol Standardizer`
3. `Fingerprint Generator` or `Mol Descriptors`
4. `QSAR Regression`

## Notes

- Requires an active internet connection.
- For programmatic bioactivity retrieval use `ChEMBL Bioactivity Retriever`.
