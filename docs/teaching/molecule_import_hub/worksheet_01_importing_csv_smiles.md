# Worksheet 01 — Importing CSV Files with SMILES

## Context

Most cheminformatics datasets begin as CSV files. A professional workflow should not assume that the first text column is the structure column.

## Intended learning outcomes

Students will be able to:

- identify the correct SMILES column;
- preserve compound identifiers and metadata;
- inspect valid and failed records after import.

## Input file

```text
examples/molecule_import_hub/molecule_import_demo.csv
```

## Orange workflow

```text
Molecule Import Hub → Data Table
Molecule Import Hub → Import Report → Data Table
Molecule Import Hub → Failed Records → Data Table
```

## Steps

1. Open Orange.
2. Add **Molecule Import Hub** from `Cheminf - Data`.
3. Browse to `molecule_import_demo.csv`.
4. Set `SMILES column = smiles`.
5. Set `Name column = name`.
6. Click **Import molecules**.
7. Inspect all outputs in separate **Data Table** widgets.

## Questions

1. How many records were imported successfully?
2. Which record failed and why?
3. Which original metadata columns were preserved?
4. Why is an import report useful before standardization?

## Expected evidence

A short table containing the total number of records, valid records, failed records and the error message for the invalid molecule.
