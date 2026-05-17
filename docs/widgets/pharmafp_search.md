# PharmaFP Search

## Status

Current widget in the package.

Source:
- [ow_pharmafp_search.py](../../src/chem_inf_widgets/widgets/ow_pharmafp_search.py)
- [compound_detail_service.py](../../src/chem_inf_widgets/chemcore/services/compound_detail_service.py)

## Purpose

`PharmaFP Search` ranks a reference library using a combination of:

- PharmaFP-style fragment overlap
- whole-compound similarity
- scaffold context
- selected motif logic

It is designed as the downstream partner to `Compound Detail Card`.

## Input

- `Query Molecule`
- `Fragment Queries`
- `Motif Queries`
- `Scaffold Query`
- `Search Profile`
- `Reference Data`

## Output

- `Ranked Hits`
- `Hit Compounds`

## Search modes

- `Fragment`
- `Similarity`
- `Scaffold`
- `Hybrid`

## Main controls

- reference SMILES column
- reference name column
- search mode
- motif logic `or` / `and`
- top-k size
- minimum PharmaFP similarity
- auto-run

## Typical workflow

1. `Compound Detail Card`
2. select one or more motifs or fragments
3. `PharmaFP Search`
4. inspect results in `Data Table`, `Molecular Viewer` or `Pair Viewer`

## Notes

- `Auto run` is useful for a one-click exploration flow.
- `Hybrid` mode is usually the best default when you want a balance between substructure semantics and whole-molecule similarity.
