# ChEMBL Bioactivity Retriever

## Status

Current widget in the package.

Source:
- [ow_chembl_dataretriever.py](../../src/chem_inf_widgets/widgets/ow_chembl_dataretriever.py)

## Purpose

`ChEMBL Bioactivity Retriever` fetches bioactivity data with drug design properties for a given target or compound set from the ChEMBL web API.

## Input

None — queries ChEMBL directly via target ID or compound search criteria entered in the widget UI.

## Output

- `Bioactivity Data` — Orange `Table` with activity values, compound identifiers, and drug design properties

## Typical workflow

1. `ChEMBL Bioactivity Retriever`
2. `Mol Standardizer`
3. `Drug Filter`
4. `QSAR Regression` or `Activity Cliff Finder`

## Notes

- Requires an active internet connection.
- For interactive compound browsing use `ChEMBL Browser` instead.
- The output table is suitable as direct input for descriptor and modeling widgets.
