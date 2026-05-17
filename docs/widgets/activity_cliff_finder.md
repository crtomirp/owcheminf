# Activity Cliff Finder

## Status

Current widget in the package.

Source:
- [ow_activity_cliff_finder.py](../../src/chem_inf_widgets/widgets/ow_activity_cliff_finder.py)
- [activity_cliff_service.py](../../src/chem_inf_widgets/chemcore/services/activity_cliff_service.py)

## Purpose

`Activity Cliff Finder` highlights pairs of highly similar molecules that show a large difference in activity. These pairs are often among the most informative SAR examples in a dataset.

## Input

- Orange `Table`
- expected contents:
  - a SMILES column
  - a numeric activity column

## Output

- `Cliff Pairs`
- `Cliff Compounds`
- `Scaffold Summary`

## Main controls

- activity column
- similarity threshold
- activity threshold
- activity scale handling such as linear vs log interpretation
- top-k or ranking-related limits depending on current widget settings

## Typical workflow

1. `Mol Standardizer`
2. `Activity Cliff Finder`
3. `Pair Viewer`
4. `Molecular Viewer` or `Compound Detail Card`

## Notes

- This widget is especially useful in medicinal chemistry teaching because it makes non-linear SAR effects immediately visible.
- Pair inspection is much more useful when combined with `Pair Viewer`.
