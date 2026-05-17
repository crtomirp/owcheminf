# Reaction Enumerator

## Status

Current widget in the package.

Source:
- [ow_reaction_enumerator.py](../../src/chem_inf_widgets/widgets/ow_reaction_enumerator.py)
- [reaction_enumerator_service.py](../../src/chem_inf_widgets/chemcore/services/reaction_enumerator_service.py)

## Purpose

`Reaction Enumerator` builds virtual product sets from reaction rules and reagent combinations.

It is the library-generation companion to the simpler single-run reaction widgets.

## Input

Depending on the current workflow, the widget consumes:

- reactant tables
- reaction-rule definitions
- optional seed or template structures

## Output

- enumerated products
- summary information about attempted and successful combinations
- optional failed combinations or discarded rows, depending on current implementation

## Typical workflow

1. prepare reagent libraries
2. `Reaction Enumerator`
3. `Drug Filter`
4. `Similarity Search` or `Compound Detail Card`

## Notes

- This widget is especially useful for virtual library generation and teaching combinatorial design ideas inside Orange.
- It pairs naturally with `RDKit Reactor` and downstream filtering widgets.
