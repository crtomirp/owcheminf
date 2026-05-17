# Similarity Search

## Status

Current widget in the package.

Source:
- [ow_similarity_search.py](../../src/chem_inf_widgets/widgets/ow_similarity_search.py)
- [similarity_search_service.py](../../src/chem_inf_widgets/chemcore/services/similarity_search_service.py)

## Purpose

`Similarity Search` finds near neighbors of query compounds inside a reference library and ranks them by structural similarity.

It is one of the most generally useful glue widgets in the package.

## Input

- query compounds
- reference compounds

The exact widget labels depend on the current implementation, but both branches should provide SMILES-bearing molecular tables.

## Output

- ranked similarity hits
- hit compounds or pairwise result tables, depending on the selected output path

## Typical workflow

1. `Mol Standardizer`
2. `Similarity Search`
3. `Molecular Viewer`, `Compound Detail Card` or `Pair Viewer`

## Common uses

- analog expansion around a seed compound
- nearest-neighbor lookup for a QSAR outlier
- finding similar compounds before `Activity Cliff Finder`
- searching a curated internal library from `Compound Detail Card`

## Notes

- This widget complements `PharmaFP Search`: use `Similarity Search` for whole-molecule structural proximity and `PharmaFP Search` for fragment- and motif-guided retrieval.
