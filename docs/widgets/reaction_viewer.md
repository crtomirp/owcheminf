# Reaction Viewer

## Status

Current widget in the package.

Source:
- [ow_reactionviewer.py](../../src/chem_inf_widgets/widgets/ow_reactionviewer.py)
- [reaction_viewer_service.py](../../src/chem_inf_widgets/chemcore/services/reaction_viewer_service.py)

## Purpose

`Reaction Viewer` inspects reaction records and presents reaction strings, reactants and products in a cleaner chemistry-oriented view.

## Input

- Orange `Table`
- expected reaction-related columns such as reaction strings, reactant strings or product strings

## Output

The widget is primarily viewer-oriented, but it also supports export and downstream inspection paths depending on the current workflow.

## Typical workflow

1. import or construct a reaction table
2. inspect rows in `Reaction Viewer`
3. export or route selected reactions downstream for reporting

## Notes

- This widget is most useful when reaction strings are already available in a table.
- It complements `RDKit Reactor` and `Reaction Enumerator` by making the results easier to inspect.
