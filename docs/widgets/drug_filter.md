# Drug Filter

## Status

Current widget in the package.

Source:
- [ow_drug_filter.py](../../src/chem_inf_widgets/widgets/ow_drug_filter.py)
- [drug_filter_service.py](../../src/chem_inf_widgets/chemcore/admet/drug_filter_service.py)

## Purpose

`Drug Filter` evaluates molecules against practical drug-likeness rules and forwards either all molecules, only those that pass, or only those that fail.

It combines:

- Lipinski-style checks
- Veber-style checks
- QED-related scoring
- PAINS pattern detection

## Input

- Orange `Table`
- expected molecular representation: a SMILES-like text column, ideally named `SMILES`

## Output

- `Filtered Compounds`

The output table includes computed numeric descriptors and meta information such as:

- original SMILES
- pass/fail label
- optional PAINS identifiers
- optional highlighted atom indices when PAINS highlighting is enabled

## Main controls

- `Rule`
  - `Lipinski`
  - `Veber`
  - `Lipinski + Veber`
  - `None`
- `Selection`
  - `Forward All Molecules`
  - `Within Criteria`
  - `Out of Criteria`
- `Highlight PAINS substructures`

## Typical workflow

1. Load molecules with `SDF Reader` or `File`
2. Standardize them with `Mol Standardizer`
3. Run `Drug Filter`
4. Inspect the result in `Data Table` or `Molecular Viewer`

## Notes

- The canonical packaged PAINS resource is [smartspains.json](../../src/chem_inf_widgets/chemcore/data/smartspains.json).
- The widget is intended as a screening aid, not as a medicinal chemistry decision engine on its own.
