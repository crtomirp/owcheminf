# Worksheet 02 — Importing `.smi` and `.smiles` Files

## Context

Many screening libraries are distributed as simple SMILES files, where the first token is the SMILES string and the remaining tokens are interpreted as a molecule name.

## Input file

```text
examples/molecule_import_hub/mini_import.smi
```

## Orange workflow

```text
Molecule Import Hub → Import Report → Data Table
```

## Steps

1. Select `mini_import.smi` in **Molecule Import Hub**.
2. Leave SMILES and name columns empty because `.smi` parsing is token-based.
3. Import molecules.
4. Inspect `Import Report` and `Failed Records`.

## Questions

1. How is the name detected in a `.smi` file?
2. Which line fails?
3. What are the limitations of simple whitespace-delimited `.smi` files?

## Extension

Convert the same molecules into a CSV file with explicit `name` and `smiles` columns. Compare the import report.
