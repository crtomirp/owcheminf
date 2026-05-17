# RDKit Reactor

## Status

Current widget in the package.

Source:
- [ow_reactor.py](../../src/chem_inf_widgets/widgets/ow_reactor.py)

## Purpose

`RDKit Reactor` applies SMIRKS reaction transformations to a pool of SMILES reactants. Up to three independent reaction inputs can be combined in a single run.

## Input

- `Molecules` — reactant `Table` (default)
- `Reactions (SMIRKS) 1` — `Table` with SMIRKS strings
- `Reactions (SMIRKS) 2` — optional second reaction table
- `Reactions (SMIRKS) 3` — optional third reaction table

## Output

- `Products` — `Table` of reaction products
- `Log` — `Table` with per-reaction success/failure records

## Typical workflow

1. load reactants via `SDF Reader` or `File`
2. load or define SMIRKS reaction rules
3. `RDKit Reactor`
4. `Drug Filter`
5. `Molecular Viewer`

## Notes

- For systematic library enumeration from reagent combinations use `Reaction Enumerator`.
- Use `Reaction Viewer` to inspect reaction records visually.
