# Matched Molecular Pairs

## Status

Current widget in the package.

Source:
- [ow_matched_molecular_pairs.py](../../src/chem_inf_widgets/widgets/ow_matched_molecular_pairs.py)

## Purpose

`Matched Molecular Pairs` finds pairs of molecules that share a common core but differ by a single local structural transformation. This is the standard method for extracting SAR rules from compound series.

## Input

- `Data` — Orange `Table` with a SMILES column

## Output

- `Pair Table` — one row per matched pair with transformation SMARTS and activity difference
- `Pair Compounds` — individual compounds involved in at least one matched pair

## Typical workflow

1. `Mol Standardizer`
2. `Matched Molecular Pairs`
3. `Pair Viewer`
4. `Compound Detail Card`

## Notes

- Pairs with large activity differences are the most informative for medicinal chemistry.
- Combine with `Pair Viewer` to inspect pairs visually side by side.
- Complements `Activity Cliff Finder`: cliff finder focuses on similarity + activity gap, MMP focuses on minimal structural change.
