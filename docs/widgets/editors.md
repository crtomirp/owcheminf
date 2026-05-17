# Editors

## Status

Current editor-related widgets:

- [ow_mol_editor.py](../../src/chem_inf_widgets/widgets/ow_mol_editor.py)
- [ow_mol_ketcher_editor.py](../../src/chem_inf_widgets/widgets/ow_mol_ketcher_editor.py)

## Purpose

The editor widgets let users draw, edit or prototype molecular structures directly inside Orange workflows.

## Available options

### Mol Editor

Primary built-in molecular editor for general use.

### Mol Ketcher

WebEngine-based editor with a richer sketching experience.

## Recommended usage

Use an editor widget when you want to:

- draw a query for `Substructure Search`
- sketch a seed compound for `Compound Detail Card`
- prototype reactants before `Reactor` or `Reaction Enumerator`

## Stability note

`Ketcher` uses lazy WebEngine initialization in this package to reduce startup crashes. If WebEngine remains unstable on a given machine, see [troubleshooting.md](../troubleshooting.md).
