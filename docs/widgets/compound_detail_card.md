# Compound Detail Card

## Status

Current widget in the package.

Source:
- [ow_compound_detail_card.py](../../src/chem_inf_widgets/widgets/ow_compound_detail_card.py)
- [compound_detail_service.py](../../src/chem_inf_widgets/chemcore/services/compound_detail_service.py)

## Purpose

`Compound Detail Card` is a FAIRMol-style inspection widget for a single selected compound.

It is designed to act both as:

- a compact structure-and-summary viewer
- a search launcher for downstream database exploration

## Input

- `Data`
- `Molecules`
- optional `Reference Data`

## Output

- `Selected Compound`
- `Similar Compounds`
- `Matched Fragments`
- `Detected Motifs`
- `Motif Queries`
- `Query Molecule`
- `Fragment Queries`
- `Scaffold Query`
- `Search Profile`

## What it shows

- structure image
- compact compound summary
- detected heterocycles and functional groups
- matched PharmaFP-style fragments
- similar compounds from a reference library

## Search-oriented features

The widget can emit search-ready queries derived from the currently selected compound:

- motif SMARTS queries
- scaffold queries
- fragment-driven search profiles

Users can select one or more motifs and choose `AND` or `OR` logic for downstream search.

## Typical workflow

1. load a compound library
2. inspect one row in `Compound Detail Card`
3. select motifs or fragments
4. send outputs into `PharmaFP Search` or `Substructure Search`

## Notes

- The current UI is intentionally minimalist: the main card keeps focus on structure, summary and motifs, while fragments and similar hits open in lighter secondary dialogs.
- This widget is one of the best entry points for “SmartChemist” style guided exploration in the current package.
